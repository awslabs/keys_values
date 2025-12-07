# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Optional, Dict, Any, Tuple
from itertools import accumulate
from functools import partial
from dataclasses import replace

import torch

from litgpt.config import Config

from keys_values.attention import do_softcapping
from keys_values.head_model import HeadModel
from keys_values.kvcache.base import KVCacheReplayLog
from keys_values.kvcache.basics import KVCacheWithBuffers
from keys_values.kvcache.buffers import (
    KVCacheBuffersParams,
    DefaultKVCacheBuffers,
)
from keys_values.kvcache.factory import (
    SUPPORTED_QUANTIZERS,
    create_quantized_kv_buffers_for_checkpoints,
    deallocate_kv_cache_buffers,
)
from keys_values.kvcache.gradient.autograd_hooks import (
    AutogradHooks,
    CellComputationAutogradHooks,
    AnnotationUsageLog,
    do_ignore_annotation,
)
from keys_values.kvcache.gradient.cell import (
    CellComputation,
    cell_computation,
    GetInputSlice,
    WriteOutputsSlice,
)
from keys_values.kvcache.gradient.checkpoints import (
    KVCacheBufferCheckpoints,
    KVCacheBufferQuantizedCheckpoints,
    KVCacheBufferDefaultCheckpoints,
)
from keys_values.kvcache.gradient.gpu_memory import RecordGPUMemory
from keys_values.kvcache.gradient.inference_replay import inference_replay_cache_factory
from keys_values.kvcache.stack_layers import CellBlocks
from keys_values.kvcache.utils import (
    VerbosityLevels,
    bytes_for_torch_dtype,
)
from keys_values.long_context import (
    get_chunks_for_cells,
    get_chunk_of_targets,
    compute_loss_for_chunk,
    HEAD_OR_INITIAL_TENSORS_MAX_BYTES,
)
from keys_values.model import GPT, device_for_layer


def checkpoint_hook(
    buffers: DefaultKVCacheBuffers,
    chunk_idx: int,
    checkpoints: KVCacheBufferCheckpoints,
):
    checkpoints.set_checkpoint(chunk_idx=chunk_idx, buffers=buffers)


def copy_requires_grad(x: torch.Tensor) -> torch.Tensor:
    return x.clone().detach().requires_grad_(True)


class GradientAccumulator:
    """
    Implements gradient accumulation for a row of cells, using activation
    checkpointing.

    The initial forward pass in inference mode must have been run already,
    giving rise to KV cache replay logs for every layer, as well as layer input
    checkpoints. The main method :meth:`run` selects one row in the lattice,
    it is supposed to be called in loop from top to bottom. It does the following:

    * Forward inference pass to compute KV cache buffer checkpoints of type
      :class:`KVCacheBufferCheckpoints`. This is done using inference replay
      caches created by :func:`inference_replay_cache_factory`.
    * Forward-backward computations over columns, from right to left, using
      :class:`CellComputation`. These use head gradients on top, produce head
      gradients on the right, and apart from accumulating the model gradients
      also compute head gradients for the row below. If `autograd_hooks` is
      given and of type :class:`CellComputationAutogradHooks`, the largest
      tensors are dealt with in a special way, saving most of the memory.

    If `len(chunks_per_cell) == 1`, this is a special case, where no caching is
    done along the row. This is mostly useful for testing, or if context widths
    are small.

    Cleaning up autograd graph tensors and `autograd_hooks`:

    If `autograd_hooks` is given, it provides autograd saved tensors hooks,
    which are used to track arrays stored in the computation graph, via
    `autograd_hooks.arrays_cleanup`. This information can be used in order
    to clean things up if an out of memory error is caught. Note that if
    `autograd_hooks` is of type :class:`CellComputationAutogradHooks`, it
    plays two different toles (see above).

    """
    def __init__(
        self,
        config: Config,
        autograd_hooks: Optional[AutogradHooks],
        qname: Optional[str] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        verbose: VerbosityLevels = VerbosityLevels.NONE,
        train_cache_kwargs: Optional[Dict[str, Any]] = None,
        debug_tensors: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if qname is None:
            qname = "torch-quantized8"
        elif qname not in SUPPORTED_QUANTIZERS:
            raise ValueError(f"qname = {qname} is not supported, must be in {SUPPORTED_QUANTIZERS}")

        self.config = config
        self.autograd_hooks = autograd_hooks
        self.verbose = verbose
        self._verbose_more = verbose is VerbosityLevels.MORE or verbose is VerbosityLevels.ALL
        self.qname = qname
        if cache_kwargs is None:
            cache_kwargs = dict()
        self.cache_kwargs = cache_kwargs
        self._kv_cache_checkpoints = None  # Created when needed
        self._batch_size = None
        self._clear_internal()
        # Annotation usage logs
        self._annotation_usage_logs: Dict[int, AnnotationUsageLog] = dict()
        if train_cache_kwargs is None:
            train_cache_kwargs = dict()
        self._train_cache_kwargs = train_cache_kwargs
        self._debug_tensors = debug_tensors  # DEBUG

    def annotation_usage_logs(self) -> Dict[int, AnnotationUsageLog]:
        return self._annotation_usage_logs

    def _clear_internal(self):
        self.replay_logs = None
        self.chunks_per_cell = None
        self.seq_length = None
        self.do_checkpointing = None
        self.inputs_ranges = None
        self.top_bottom_ranges = None
        self._batch_size = None
        if self._kv_cache_checkpoints is not None:
            del self._kv_cache_checkpoints
            self._kv_cache_checkpoints = None

    def _initialize_internal(
        self,
        replay_logs: List[KVCacheReplayLog],
        chunks_per_cell: List[int],
        weights_dtype: torch.dtype,
        head_model_needs_logits: bool = True,
    ):
        if self._batch_size is None:
            raise IndexError("batch_size must be set")
        CellComputation.check_args(
            self.config,
            self._hooks_for_cell_computation(),
            replay_logs,
            self._batch_size,
        )
        if len(replay_logs) != self.config.n_layer:
            raise ValueError(f"len(replay_logs) = {len(replay_logs)} != {self.config.n_layer} = config.n_layer")
        if len(chunks_per_cell) == 0:
            raise ValueError(f"chunks_per_cell must not be empty")
        if any(x <= 0 for x in chunks_per_cell):
            raise ValueError(f"chunks_per_cell = {chunks_per_cell}: All entries must be positive")
        num_chunks = len(replay_logs[0].token_chunks)
        if sum(chunks_per_cell) != num_chunks:
            raise ValueError(f"chunks_per_cell = {chunks_per_cell}: Entries must sum to {num_chunks}")
        self.replay_logs = replay_logs.copy()
        self.chunks_per_cell = chunks_per_cell.copy()
        self.seq_length = len(self.replay_logs[0])
        self.do_checkpointing = len(chunks_per_cell) > 1
        # Ranges to split up `inputs`
        chunk_lens = list(idx.shape[-1] for idx in self.replay_logs[0].token_chunks)
        self.inputs_ranges = [
            elem.input_range
            for elem in get_chunks_for_cells(chunks_per_cell, chunk_lens)
        ]
        if not self.do_checkpointing:
            # Sanity check
            assert self.inputs_ranges == [(0, self.seq_length)]
        # Ranges to be used in `run_head_model` and `run_input_embeddings`
        self._initialize_top_bottom_ranges(weights_dtype, head_model_needs_logits)
        # Reset
        self._annotation_usage_logs: Dict[int, AnnotationUsageLog] = dict()

    def _initialize_top_bottom_ranges(
        self,
        weights_dtype: torch.dtype,
        head_model_needs_logits: bool,
    ):
        """
        These ranges are used for chunking in :meth:`run_head_model` and
        :meth:`run_input_embeddings`, instead of `inputs_ranges`. Computations
        materialize tensors of shape
        `(batch_size, chunk_size, config.padded_vocab_size)`, who are to be
        kept below :const:`HEAD_OR_INITIAL_TENSORS_MAX_BYTES` bytes.

        """
        dim = self.config.padded_vocab_size if head_model_needs_logits else self.config.n_embd
        bytes_per_token = self._batch_size * dim * bytes_for_torch_dtype(weights_dtype)
        chunk_size = min(
            max(HEAD_OR_INITIAL_TENSORS_MAX_BYTES // bytes_per_token, 1),
            self.seq_length,
        )
        points = list(range(0, self.seq_length, chunk_size)) + [self.seq_length]
        self.top_bottom_ranges = list(zip(points[:-1], points[1:]))

    def _create_checkpoints_and_buffers(
        self, model_part: CellBlocks,
    ) -> Tuple[List[DefaultKVCacheBuffers], List[Optional[KVCacheBufferCheckpoints]]]:
        """
        Creates buffers for inference replay caches and checkpointing objects, the
        latter only if checkpointing is used. Note that respective buffers are stored
        on the same device as the block of a layer is on. This makes sure things work
        in a model-parallel context as well.

        """
        assert self.do_checkpointing
        # Buffers for inference replay caches
        # See also :meth:`KVCacheFactory.create`.
        # Note: The inference replay caches use buffers of type
        # :class:`DefaultKVCacheBuffers`, which do not quantize. Quantization is
        # only done when storing and retrieving the checkpoints.
        cache_buffers = []
        buffer_params = None
        for _, block in model_part.blocks():
            kv_cache = block.attn.kv_cache
            if buffer_params is None:
                buffer_params = KVCacheBuffersParams.from_params(kv_cache.get_params())
            cache_buffers.append(
                DefaultKVCacheBuffers(
                    params=replace(buffer_params, device=block.attn.device),
                    cache_length=kv_cache.cache_length,
                )
            )
        # Checkpointing objects, which may include quantizers and dequantization
        # buffers
        chunk_numbers = list(accumulate(self.chunks_per_cell))[:-1]
        checkpoints = []
        if self.qname == "default":
            for _, block in model_part.blocks():
                cache_length = block.attn.kv_cache.cache_length
                if self.qname == "default":
                    # Checkpoints are not quantized
                    checkpoints.append(
                        KVCacheBufferDefaultCheckpoints(
                            chunk_numbers=chunk_numbers,
                            params=replace(
                                buffer_params,
                                device=block.attn.device,
                            ),
                            cache_length=cache_length,
                            batch_size=self._batch_size,
                        )
                    )
        else:
            # Checkpoints are quantized
            dequant_kwargs = dict(
                max_num_ranges=self.cache_kwargs.get("max_num_ranges"),
            )
            quant_buffers = create_quantized_kv_buffers_for_checkpoints(
                model_part=model_part,
                qname=self.qname,
                batch_size=self._batch_size,
                cache_kwargs=self.cache_kwargs,
                dequant_kwargs=dequant_kwargs,
            )
            for _, block in model_part.blocks():
                device = block.attn.device
                cache_length = block.attn.kv_cache.cache_length
                checkpoints.append(
                    KVCacheBufferQuantizedCheckpoints(
                        chunk_numbers=chunk_numbers,
                        quant_buffers=quant_buffers[device],
                        cache_length=cache_length,
                    )
                )

        return cache_buffers, checkpoints

    def _create_inference_replay_caches(
        self, model_part: CellBlocks,
    ) -> List[KVCacheWithBuffers]:
        assert self.do_checkpointing
        cache_buffers, checkpoints = self._create_checkpoints_and_buffers(model_part)
        # For easy reference outside of inference replay caches
        self._kv_cache_checkpoints = checkpoints
        infer_replay_caches = []
        for (block_idx, block), buffers, checkpoint in zip(
            model_part.blocks(),
            cache_buffers,
            checkpoints,
        ):
            ir_cache = inference_replay_cache_factory(
                kv_cache=block.attn.kv_cache,
                config=self.config,
                buffers=buffers,
                block_idx=block_idx,
                replay_log=self.replay_logs[block_idx],
                **self.cache_kwargs,
            )
            # Set hook to write checkpoints
            ir_cache.set_checkpoint_hook(
                checkpoint_hook=partial(checkpoint_hook, checkpoints=checkpoint),
            )
            infer_replay_caches.append(ir_cache)
        return infer_replay_caches

    def _deallocate_buffers(self, ir_caches: List[KVCacheWithBuffers]):
        if self.do_checkpointing:
            # Deallocate GPU buffers for checkpoints
            for checkpoint in self._kv_cache_checkpoints:
                if isinstance(checkpoint, KVCacheBufferQuantizedCheckpoints):
                    checkpoint.quant_buffers.deallocate()
            # Deallocate GPU buffers for inference replay caches
            deallocate_kv_cache_buffers(ir_caches)
            for cache in ir_caches:
                cache.set_checkpoint_hook(None)

    def _hooks_for_cell_computation(self) -> Optional[CellComputationAutogradHooks]:
        if self.autograd_hooks is not None and isinstance(self.autograd_hooks, CellComputationAutogradHooks):
            return self.autograd_hooks
        else:
            return None

    def run(
        self,
        model_part: CellBlocks,
        get_inputs_slice: GetInputSlice,
        get_head_gradients_slice: GetInputSlice,
        write_head_gradients_slice: WriteOutputsSlice,
        record_gpu_memory_snapshots: Optional[RecordGPUMemory] = None,
    ):
        """
        Runs gradient accumulation for row of cells represented by the
        model part `model_part`. The gradients for blocks in `model_part`
        are accumulated.

        If the blocks in `model_part` have KV caches assigned, these are
        temporarily replaced by specific replau caches, the setup is restored
        in the end.

        Note that `get_inputs_slice` and `write_head_gradients_slice` can refer
        to the same checkpoint object. We guarantee that any slice is read
        before it is written to.

        Args:
            model_part: Represents layers of model for the cell
            get_inputs_slice: Function `f(start, end)` which returns a slice
                `range(start, end)` of the input to layer `first_layer_idx`.
            get_head_gradients_slice: Function `f(start, end)` which returns a
                slice `range(start, end)` of the head gradients for output of
                layer `first_layer_idx + num_layers - 1`.
            write_head_gradients_slice: Function `f(start, value)` which writes
                a slice `range(start, end)` of the head gradients for output
                of layer `first_layer_idx - 1`.

        """
        assert self.replay_logs is not None, "Call 'run_head_model' for a new batch"
        assert self._batch_size is not None
        first_layer_idx = model_part.first_layer_idx
        num_layers = model_part.num_layers
        self._check_run_args(first_layer_idx, num_layers)
        if self._hooks_for_cell_computation() is not None:
            self._annotation_usage_logs = dict()  # Reset
        if self._verbose_more:
            if num_layers > 1:
                print(f"\nProcessing row of cells: Layers {first_layer_idx} ... {first_layer_idx + num_layers - 1}")
            else:
                print(f"\nProcessing row of cells: Layer {first_layer_idx}")
        if record_gpu_memory_snapshots is not None:
            record_gpu_memory_snapshots.start_recording()

        # Run inference forward and store KV cache checkpoints
        if self.do_checkpointing:
            if self._verbose_more:
                print("Forward pass to store KV cache checkpoints")
            infer_replay_caches = self._create_inference_replay_caches(model_part)
            self._compute_checkpoints(
                model_part,
                infer_replay_caches,
                get_inputs_slice,
            )
            # We could delete `infer_replay_caches` here. But we still use their
            # buffers to de-quantize checkpoints below
        else:
            infer_replay_caches = None

        try:
            # Loop over cells from right to left
            # Important to switch MHA to memory efficient version for use in training
            # mode
            cell = CellComputation(
                model_part=model_part,
                autograd_hooks=self._hooks_for_cell_computation(),
                replay_logs=self.replay_logs[first_layer_idx:(first_layer_idx + num_layers)],
                batch_size=self._batch_size,
                debug_tensors=self._debug_tensors,
                **self._train_cache_kwargs,
            )
            first_layer_device = None
            last_layer_device = None
            for block_idx, block in model_part.blocks():
                if block_idx == first_layer_idx:
                    first_layer_device = block.attn.device
                elif block_idx == first_layer_idx + num_layers - 1:
                    last_layer_device = block.attn.device
            head_gradients_k = None
            head_gradients_v = None
            chunk_idxs = [0]
            if self.do_checkpointing:
                chunk_idxs += self._kv_cache_checkpoints[0].chunk_numbers
            if self._verbose_more:
                print(f"Process row of {len(chunk_idxs)} cells in reverse order")

            for col_idx, (first_chunk_idx, num_chunks, (start, end)) in reversed(
                list(
                    enumerate(
                        zip(chunk_idxs, self.chunks_per_cell, self.inputs_ranges)
                    )
                )
            ):
                # Gather inputs and head gradients:
                # - Inputs bottom:   cell_inputs
                # - Inputs left:     k_buffers, v_buffers
                # - Gradients top:   head_gradients_top
                # - Gradients right: head_gradients_k, head_gradients_v
                cell_inputs = copy_requires_grad(
                    get_inputs_slice(start, end).to(device=first_layer_device)
                )
                head_gradients_top = get_head_gradients_slice(start, end).to(
                    device=last_layer_device
                )
                if col_idx == 0:
                    k_buffers = None
                    v_buffers = None
                else:
                    # Note: It is `chunk_idxs[col_idx]`, not
                    # `chunk_idxs[col_idx - 1]`, because `chunk_idxs[0] == 0` is a
                    # dummy entry
                    k_buffers, v_buffers = self._get_checkpoints(
                        infer_replay_caches,
                        chunk_idx=chunk_idxs[col_idx],
                        model_part=model_part,
                    )
                    k_buffers = [copy_requires_grad(x) for x in k_buffers]
                    v_buffers = [copy_requires_grad(x) for x in v_buffers]
                # Forward-backward, using the autograd hooks (if given)
                scalar_output = None
                try:
                    if self.autograd_hooks is not None:
                        with torch.autograd.graph.saved_tensors_hooks(
                            lambda x: self.autograd_hooks.pack_hook(x),
                            lambda x: self.autograd_hooks.unpack_hook(x),
                        ):
                            scalar_output = self.forward_computation(
                                cell=cell,
                                cell_inputs=cell_inputs,
                                k_buffers=k_buffers,
                                v_buffers=v_buffers,
                                head_gradients_top=head_gradients_top,
                                head_gradients_k=head_gradients_k,
                                head_gradients_v=head_gradients_v,
                                first_chunk_idx=first_chunk_idx,
                                num_chunks=num_chunks,
                                first_layer_idx=first_layer_idx,
                                autograd_hooks=self._hooks_for_cell_computation(),
                            )
                    else:
                        scalar_output = self.forward_computation(
                            cell=cell,
                            cell_inputs=cell_inputs,
                            k_buffers=k_buffers,
                            v_buffers=v_buffers,
                            head_gradients_top=head_gradients_top,
                            head_gradients_k=head_gradients_k,
                            head_gradients_v=head_gradients_v,
                            first_chunk_idx=first_chunk_idx,
                            num_chunks=num_chunks,
                            first_layer_idx=first_layer_idx,
                        )

                    scalar_output.backward()
                    write_head_gradients_slice(start, cell_inputs.grad)
                    if col_idx > 0:
                        head_gradients_k = [x.grad for x in k_buffers]
                        head_gradients_v = [x.grad for x in v_buffers]
                finally:
                    del scalar_output
                    del cell_inputs
                    if k_buffers is not None:
                        del k_buffers
                    if v_buffers is not None:
                        del v_buffers
                # Store annotation usage logs
                if self.autograd_hooks is not None:
                    cell_hooks = self._hooks_for_cell_computation()
                    if cell_hooks is not None:
                        self._annotation_usage_logs[
                            first_chunk_idx
                        ] = cell_hooks.annotation_usage_log()
                    if self._verbose_more:
                        arrays_cleanup = self.autograd_hooks.arrays_cleanup
                        if arrays_cleanup is not None:
                            stats = arrays_cleanup.stats()
                            if stats.num > 0:
                                print(f"Remaining arrays not cleaned up: {stats.num} ({stats.total_mem} GB) of {stats.max_num}")
                    if cell_hooks is not None:
                        cell_hooks.clear()  # Clear memory
                        if self.verbose is VerbosityLevels.ALL:
                            if num_layers > 1:
                                part = f"layers {first_layer_idx} to {first_layer_idx + num_layers - 1}"
                            else:
                                part = f"layer {first_layer_idx}"
                            print(f"\nAnnotation usage log [{part}; chunks {first_chunk_idx} to {first_chunk_idx + num_chunks - 1}]")
                            print(self._annotation_usage_logs[first_chunk_idx].report())

        finally:
            self._deallocate_buffers(infer_replay_caches)
            while infer_replay_caches:
                cache = infer_replay_caches.pop()
                del cache
            if record_gpu_memory_snapshots is not None:
                record_gpu_memory_snapshots.store_current_snapshot()
                record_gpu_memory_snapshots.stop_recording()

    @staticmethod
    def forward_computation(
        cell: CellComputation,
        cell_inputs: torch.Tensor,
        k_buffers: Optional[List[torch.Tensor]],
        v_buffers: Optional[List[torch.Tensor]],
        head_gradients_top: torch.Tensor,
        head_gradients_k: Optional[List[torch.Tensor]],
        head_gradients_v: Optional[List[torch.Tensor]],
        first_chunk_idx: int,
        num_chunks: int,
        first_layer_idx: int,
        autograd_hooks: Optional[CellComputationAutogradHooks] = None,
    ):
        # Why not use `get_inputs_slice` passed to :meth:`run`? This won't work,
        # since we need to create the input to the cell as tensor with
        # `requires_grad=True`.
        input_pos = cell.get_input_pos(first_chunk_idx)

        def get_inputs_slice_local(start, end):
            assert input_pos <= start < end
            return cell_inputs[:, (start - input_pos):(end - input_pos), :]

        cell_outputs, output_k_buffers, output_v_buffers = cell(
            get_inputs_slice=get_inputs_slice_local,
            k_buffers=k_buffers,
            v_buffers=v_buffers,
            first_chunk_idx=first_chunk_idx,
            num_chunks=num_chunks,
            debug_print_annotations=False if autograd_hooks is None else autograd_hooks.debug_print_annotations,
        )
        scalar_output = (cell_outputs * head_gradients_top).sum()

        # If autograd hooks are used, we need to create "ignore-headgrad"
        # annotations for the head gradients, because they are stored in the
        # autograd graph
        if head_gradients_k is not None:
            for idx, (o_k, g_k) in enumerate(zip(output_k_buffers, head_gradients_k)):
                if autograd_hooks is not None:
                    GradientAccumulator._ignore_head_gradients(
                        autograd_hooks=autograd_hooks,
                        head_gradients=g_k,
                        layer_idx=first_layer_idx + idx,
                        first_chunk_idx=first_chunk_idx,
                    )
                scalar_output += (o_k * g_k).sum()
        if head_gradients_v is not None:
            for idx, (o_v, g_v) in enumerate(zip(output_v_buffers, head_gradients_v)):
                if autograd_hooks is not None:
                    GradientAccumulator._ignore_head_gradients(
                        autograd_hooks=autograd_hooks,
                        head_gradients=g_v,
                        layer_idx=first_layer_idx + idx,
                        first_chunk_idx=first_chunk_idx,
                    )
                scalar_output += (o_v * g_v).sum()
        return scalar_output

    @staticmethod
    def _ignore_head_gradients(
        autograd_hooks: CellComputationAutogradHooks,
        head_gradients: torch.Tensor,
        layer_idx: int,
        first_chunk_idx: int,
    ):
        do_ignore_annotation(
            x=head_gradients,
            node_annotations=autograd_hooks.node_annotations,
            kind="ignore-headgrad",
            layer_idx=layer_idx,
            chunk_idx=first_chunk_idx,
        )

    def _check_run_args(
        self,
        first_layer_idx: int,
        num_layers: int,
    ):
        if first_layer_idx < 0 or num_layers < 1 or first_layer_idx + num_layers > self.config.n_layer:
            raise ValueError(f"first_layer_idx = {first_layer_idx}, num_layers = {num_layers}, config.n_layer = {self.config.n_layer}")

    def _compute_checkpoints(
        self,
        model_part: CellBlocks,
        infer_replay_caches: List[KVCacheWithBuffers],
        get_inputs_slice: GetInputSlice,
    ):
        # Setup KV caches in `gpt_model`. These record the required checkpoints
        kv_caches_copy = []
        num_layers = len(infer_replay_caches)
        for (_, block), replay_cache in zip(
            model_part.blocks(),
            infer_replay_caches,
        ):
            attn = block.attn
            kv_caches_copy.append(attn.kv_cache)
            attn.kv_cache = replay_cache

        # DEBUG
        if self._debug_tensors is not None:
            for layer_idx, checkpoints in zip(
                range(model_part.first_layer_idx, model_part.first_layer_idx + model_part.num_layers),
                self._kv_cache_checkpoints[:num_layers],
            ):
                checkpoints.set_debug_layer_idx(layer_idx)
        # END DEBUG

        # Run forward in order to compute checkpoints
        with torch.no_grad():
            cell_computation(
                token_idxs=self.replay_logs[0].token_chunks,
                model_part=model_part,
                get_inputs_slice=get_inputs_slice,
                input_pos=0,
            )

        # Restore
        for (_, block), old_cache in zip(
            model_part.blocks(),
            kv_caches_copy,
        ):
            block.attn.kv_cache = old_cache

    def _get_checkpoints(
        self,
        infer_replay_caches: List[KVCacheWithBuffers],
        chunk_idx: int,
        model_part: CellBlocks,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        `gpt_model` and `first_layer_idx` are used for sanity checks that
        buffers are on the right device.

        """
        k_buffers = []
        v_buffers = []
        for (block_idx, block), kv_cache, checkpoints in zip(
            model_part.blocks(),
            infer_replay_caches,
            self._kv_cache_checkpoints,
        ):
            buffers = kv_cache.kv_buffers
            # Copy from checkpoint (CPU) into buffer (on device)
            checkpoints.get_checkpoint(chunk_idx=chunk_idx, out=buffers)
            # Dequantize
            k_and_v = buffers.get_keys_values()
            device = block.attn.device
            keys = k_and_v.keys().clone()
            if keys.device != device:
                raise IndexError(f"Internal error _get_checkpoint, l_ix={block_idx}: attn.device={device}, keys.device={keys.device}, must be the same")
            k_buffers.append(keys)
            values = k_and_v.values().clone()
            if values.device != device:
                raise IndexError(f"Internal error _get_checkpoint, l_ix={block_idx}: attn.device={device}, values.device={values.device}, must be the same")
            v_buffers.append(values)
        return k_buffers, v_buffers

    # TODO: Support having the model part only
    def run_input_embeddings(
        self,
        gpt_model: GPT,
        input_ids: torch.Tensor,
        get_head_gradients_slice: GetInputSlice,
    ):
        """
        Runs gradient accumulation for the initial embeddings of the model.

        Args:
            gpt_model: GPT model
            input_ids: Tensor of input token IDs
            get_head_gradients_slice: Function `f(start, end)` which returns a
                slice `range(start, end)` of the head gradients for inputs to
                the first layer.

        """
        if input_ids.ndim != 2 or input_ids.shape[1] != self.seq_length:
            raise ValueError(f"input_ids.shape = {input_ids.shape}, must be 2D with latter size {self.seq_length}")
        assert self.replay_logs is not None, "Call 'run_head_model' and 'run' for a new batch"
        assert self._batch_size is not None
        # We do gradient accumulation in chunks, which saves memory
        wte = gpt_model.transformer.wte
        if wte.weight.requires_grad:
            # Only run this if embedding weights are to be updated
            alpha = self.config.n_embd ** 0.5
            wte_device = wte.weight.device
            if self._verbose_more:
                print("\nGradient accumulation for input embeddings")
            for start, end in self.top_bottom_ranges:
                head_grads_part = get_head_gradients_slice(start, end).to(
                    device=wte_device
                )
                embed_part = wte(input_ids[:, start:end].to(device=wte_device))
                scalar_output = (head_grads_part * embed_part).sum()
                if self.config.scale_embeddings:
                    scalar_output = scalar_output * alpha
                scalar_output.backward()
                del scalar_output
        # End of gradient accumulation on a batch: Clear internals
        self._clear_internal()

    # TODO: Support having the model part only
    def run_head_model(
        self,
        gpt_model: GPT,
        head_model: HeadModel,
        replay_logs: List[KVCacheReplayLog],
        chunks_per_cell: List[int],
        get_inputs_slice: GetInputSlice,
        write_head_gradients_slice: WriteOutputsSlice,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        The loss function is represented in `head_model`, input tokens are
        given by `input_ids`, targets by `targets`. These two are aligned on
        the right.

        Note that `get_inputs_slice` and `write_outputs_slice` can refer to the
        same underlying buffer or checkpoint object. We guarantee that any slice
        is read before it is written to.

        Args:
            gpt_model: GPT model
            head_model: Head model and loss function
            replay_logs: KV cache replay logs recorded during the initial
                forward pass. Needed in calls of :meth:`run`, stored as
                internal here.
            chunks_per_cell: List of number of chunks for each cell. Needed in
                calls of :meth:`run`, stored as internal here.
            get_inputs_slice: Function `f(start, end)` which returns a slice
                `range(start, end)` of the final layer output
            write_head_gradients_slice: Function `f(start, value)` which writes
                a slice `range(start, end)` of the head gradients for the final
                transformer layer. This can be used as argument of :meth:`run`
                for the topmost row of cells.
            targets: Tensor of targets, aligned with `input_ids` on the right

        Returns:
            Loss function value. We use mean reduction over the sequence.

        """
        assert self.replay_logs is None, "Call 'run_input_embeddings' to end processing a batch"
        assert targets.ndim == 2
        num_output_tokens = targets.shape[1]
        # Initialize members which are needed for processing this batch
        batch_size = targets.shape[0]
        max_batch_size = gpt_model.get_kv_cache_params(0).max_batch_size
        if not (1 <= batch_size <= max_batch_size):
            raise ValueError(f"targets.shape[0] = {batch_size}, must be in [1, {max_batch_size}]")
        self._batch_size = batch_size
        weights_dtype = gpt_model.transformer.wte.weight.dtype
        self._initialize_internal(
            replay_logs,
            chunks_per_cell,
            weights_dtype=weights_dtype,
            head_model_needs_logits=head_model.needs_logits(),
        )
        # Ensure that model supports the sequence length
        if gpt_model.max_seq_length < self.seq_length:
            gpt_model.max_seq_length = self.seq_length
        if not (1 <= num_output_tokens <= self.seq_length):
            raise ValueError(f"targets.shape[1] = {num_output_tokens} must in [1, seq_length = {self.seq_length}]")
        if head_model.needs_logits():
            clamp_head = partial(do_softcapping, thresh=self.config.final_logit_softcapping)
        else:
            clamp_head = None
        # Head model must be on the same device as the final outputs
        device_final = device_for_layer(gpt_model, -1)
        head_model = head_model.to(device_final)
        targets = targets.to(device_final)
        if self._verbose_more:
            print("\nGradient accumulation for head model")
        # First loop to obtain normalization constants
        num_target_entries = [
            head_model.num_target_entries(
                get_chunk_of_targets(
                    targets=targets,
                    input_pos=start,
                    chunk_size=end - start,
                    num_input_tokens=self.seq_length,
                )
            )
            for start, end in self.top_bottom_ranges
        ]
        if num_target_entries[0] is None:
            weight_per_chunk = [1] * len(num_target_entries)
        else:
            total_sum = sum(num_target_entries)
            weight_per_chunk = [x / total_sum for x in num_target_entries]
        # Second loop to compute loss value and gradients
        loss_full = 0
        for (start, end), weight in zip(self.top_bottom_ranges, weight_per_chunk):
            x = copy_requires_grad(get_inputs_slice(start, end))
            model_outputs = gpt_model.transformer.ln_f(x)
            if head_model.needs_logits():
                model_outputs = clamp_head(gpt_model.lm_head(model_outputs))
            loss_part = compute_loss_for_chunk(
                head_model=head_model,
                model_outputs_for_chunk=model_outputs,
                targets=targets,
                num_input_tokens=self.seq_length,
                input_pos=start,
                scale_factor=weight,
            )
            loss_part = loss_part.mean()
            if loss_part.grad_fn is not None:
                loss_part.backward()
                head_grad_part = x.grad
            else:
                head_grad_part = torch.zeros_like(x.detach())
            loss_part = loss_part.detach()
            del x
            write_head_gradients_slice(start, head_grad_part)
            loss_full = loss_part + loss_full
        return loss_full
