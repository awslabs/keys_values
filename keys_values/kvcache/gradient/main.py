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
from functools import partial
import gc
from pathlib import Path
import time
from typing import Optional, Dict, Any, Tuple, Union, List

import torch

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.attention import MultiHeadSelfAttention
from keys_values.head_model import HeadModel
from keys_values.kvcache.factory import (
    SUPPORTED_QUANTIZERS,
    deallocate_kv_cache_buffers_of_model,
)
from keys_values.kvcache.gradient.accumulate import GradientAccumulator
from keys_values.kvcache.gradient.autograd_hooks import (
    CellComputationAutogradHooks,
    CleanupArraysAutogradHooks,
    AnnotationUsageLog,
)
from keys_values.kvcache.gradient.checkpoints import (
    LayerInputQuantizedCheckpoints,
    LayerInputDefaultCheckpoints,
)
from keys_values.kvcache.gradient.cleanup import (
    ArraysForCleanup,
    protect_named_params_buffers_of_model,
)
from keys_values.kvcache.gradient.gpu_memory import RecordGPUMemory
from keys_values.kvcache.stack_layers import DefaultCellBlocks
from keys_values.kvcache.utils import (
    wrap_tqdm_if_verbose,
    VerbosityLevels,
    message_with_device_memory,
)
from keys_values.long_context import (
    LongContextInferenceModel, GPTAndHeadModel, oom_exception_action,
)
from keys_values.model import GPT, block_iterator, device_for_layer
from keys_values.optimize.clone_model import (
    clone_model_shard_via_flat_vectors,
    copy_flat_vectors_to,
)
from keys_values.optimize.model_factory import GPTShardCellBlock
from keys_values.optimize.module_wrapper import AccessWeightsGradients


class LossValue(torch.Tensor):
    """
    Specific subclass of :class:`torch.Tensor`, overwrites
    :meth:`backward` with backward call to :class:`LongContextGradientModel`.
    See :func:`long_context_loss_value`.

    """
    # See https://discuss.pytorch.org/t/subclassing-torch-tensor/23754
    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        model: "LongContextGradientModel",
        *args,
        **kwargs,
    ):
        return super().__new__(cls, data, *args, **kwargs)

    def __init__(
        self, data: torch.Tensor, model: "LongContextGradientModel",
    ):
        super().__init__()
        self._model = model

    def detach(self, *args, **kwargs):
        return super().detach(*args, **kwargs)

    def clone(self, *args, **kwargs):
        if hasattr(self, "_model"):
            return LossValue(super().clone(*args, **kwargs), self._model)
        else:
            return super().clone(*args, **kwargs)

    def to(self, *args, **kwargs):
        new_obj = super().to(*args, **kwargs)
        if new_obj is self:
            return self
        if hasattr(self, "_model"):
            return LossValue(new_obj, self._model)
        else:
            return new_obj

    def backward(self, *args, **kwargs):
        if args or kwargs:
            raise ValueError(
                "LossValue.backward() takes no arguments, but got:\n"
                f"args = {args}\n"
                f"kwargs = {kwargs}"
            )
        if not self._model.ready_for_backward():
            raise IndexError("Model not ready to run 'backward'")
        self._model.backward()


def check_model_is_on_device(
    model: torch.nn.Module,
    device: torch.device,
    model_name: str,
):
    for name, param in model.named_parameters():
        if param.device != device:
            raise ValueError(f"Model {model_name} must be on {device}, but device['{name}'] = {param.device}")


def accumulate_gradients(
    module_pairs: List[Tuple[torch.nn.Module, torch.nn.Module]],
    debug_modules: Optional[List[torch.nn.Module]] = None,
):
    """
    `module_pairs` contains tuples `(mod_from, mod_to)`. For each tuple,
    read gradients of parameters in `mod_from` and add to gradients in
    `mod_to`.

    Args:
        module_pairs: List of `(mod_from, mod_to)` tuples
        debug_modules: Use for debugging only

    """
    if debug_modules is None:
        debug_modules = [None] * len(module_pairs)
    else:
        assert len(debug_modules) == len(module_pairs)
    for (mod_from, mod_to), mod_debug in zip(module_pairs, debug_modules):
        access = AccessWeightsGradients(mod_from)
        flat_vectors = copy_flat_vectors_to(
            access.get_gradients(), device=torch.device("cpu"),
        )
        mod_from.zero_grad(set_to_none=True)
        AccessWeightsGradients(mod_to).accumulate_gradients(flat_vectors)
        if mod_debug is not None:
            for name, param in mod_debug.named_parameters():
                param_comp = mod_from.get_parameter(name)
                print(f"Compare {name}")
                torch.testing.assert_close(param.data, param_comp.data)
                if param.requires_grad:
                    src_arg = mod_from.get_parameter(name).grad.data
                    if param.grad is None:
                        param.grad = torch.nn.Parameter(src_arg)
                    else:
                        param.grad.data.copy_(src_arg)


class LongContextGradientModel(LongContextInferenceModel):
    """
    Wraps a `GPT` model, provides both inference and gradient computation
    for long contexts. Gradient computation:

    * This is done by nested activation checkpointing (outer over layers,
      inner over chunks of the sequence). Forward-backward computations are done
      on cell (some layers, some chunks), see :class:`GradientAccumulator`,
      :class:`CellComputation`.
    * The computation per cell is made GPU memory efficient by using specific
      autograd saved tensors hooks, see :class:`CellComputationAutogradHooks`
      and :class:`TrainingAttnWeightsReplayCache`.
    * In :class:`TrainingAttnWeightsReplayCache`, we also use special operators
      for MHA and KV cache updates, which require much less device memory than
      the default SDPA variants. This is only done during gradient computation.

    The GPT model `model` must have KV caches assigned to every layer. The
    caches can be of different type, but must have a fixed `cache_length`.

    All memory required here is allocated anew for every :meth:`forward` call,
    depending on the sequence length, and is deallocated at the end of
    :meth:`backward`. This means that available GPU and CPU memory is shared
    between the forward pass (in particular, the KV caches) and the gradient
    computations here.

    Chunks and cells for gradient computation:

    Think of a lattice of blocks, with layers as rows and chunks as columns.
    Activation checkpointing operates on cells, which are rectangular groups
    of blocks. A cell has `layers_per_cell` layers. Cell widths are
    determined automatically so that the cell length (sum of chunk sizes) is
    `<= cache_length`, but as close as possible.

    The choice of `layers_per_cell` determines GPU memory requirements: they
    scale linearly with this number. Overall runtime is shorter and CPU memory
    requirements for checkpointing is smaller for larger `layers_per_cell`.

    Autograd hooks and annotation usage logs:

    The most advanced (and potentially brittle) part of the workflow is using
    autograd saved tensors hooks in order to save memory during
    forward-backward cell computations. For details, see
    :class:`CellComputationAutogradHooks` and
    :class:`TrainingAttnWeightsReplayCache`. The idea is that the largest
    tensors stored in each block can be reconstructed from much smaller
    tensors. This requires matching an input to the pack hook with an annotation
    created during the forward pass.

    If more GPU memory than expected is used, you can look at annotation
    usage logs returned by :meth:`annotation_usage_logs`. There is one log
    per cell, identified by the key `(first_layer_idx, first_chunk_idx)`.

    Sharing device memory between forward and backward computations:

    In :meth:`backward`, we deallocate buffers of all KV caches, then
    allocate members required for the backward computations. The latter are
    deallocated at the end of :meth:`backward`. This means that device memory
    is shared between forward and backward computations. The KV cache buffers
    are automatically reallocated when required next.

    CPU offloading:

    Only on training mode. If `offload_device` is given, we run a form of CPU
    offloading. Namely, `gpt_model` is on the CPU, while computations are done
    using suitable copies on device `offload_device`.

    * `gpt_model` is on the CPU, it is used to accumulate gradients. Its
      parameters are temporarily copied.
    * If `head_model` has parameters, they are on `offload_device`.
    * For evaluation, use :meth:`copy_model_for_evaluation` to obtain a
      :class:`LongContextInferenceModel` copy on device `offload_device`.

    """
    def __init__(
        self,
        gpt_model: GPT,
        head_model: HeadModel,
        layers_per_cell: int,
        chunk_size: int = 16,
        randomize_chunk_sizes: bool = False,
        chunks_per_cell_multiplier: float = 1.0,
        single_tokens_for_targets: bool = False,
        verbose: VerbosityLevels = VerbosityLevels.SOME,
        tmp_array_limit_gb: Optional[TemporaryArrayLimit] = None,
        debug_single_cell_per_row: bool = False,
        qname: Optional[str] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        train_cache_kwargs: Optional[Dict[str, Any]] = None,
        backward_tmp_array_limit_gb: Optional[TemporaryArrayLimit] = None,
        autograd_hooks_kwargs: Optional[Dict[str, Any]] = None,
        debug_dont_use_autograd_hooks: bool = False,
        use_arrays_cleanup: bool = True,
        profile_steps: bool = False,
        offload_device: Optional[torch.device] = None,
        layer_checkpoint_chunk_size: Optional[int] = None,
        debug_gpt_model: Optional[GPT] = None,
        debug_store_intermediates: bool = False,
    ):
        """
        Args:
            gpt_model: GPT model to train on sequence data. All layers must have
                KV caches assigned, and these must not be dense. For now, all
                caches must have the same `cache_length`.
            head_model: Head model and loss function
            layers_per_cell: Number of layers per cell. GPU memory requirements
                scale linearly with this number.
            chunk_size: Data batches are processed in chunks of this size
                (except the first one). See above.
            randomize_chunk_sizes: If `True`, chunk sizes are randomized (with
                mean `chunk_size`). This may have advantages for model
                training. Defaults to `False`.
            chunks_per_cell_multiplier: Each cell contains a number of chunks.
                The length of a cell is the sum of lengths of its cells. We
                assign chunks to cells so that cell lengths are close to
                `int(cache_length * chunks_per_cell_multiplier)`, but not
                larger. The larger this multiplier, the fewer cells per row,
                which speeds up computation, but also memory requirements of
                gradient computation per cell scales linearly in this value.
            single_tokens_for_targets: If `True`, the targets part of a
                sequence is processed token per token (i.e., with chunk size
                1). This is slower, but more realistic, mirroring how inference
                looks like.
            verbose: Verbosity level, defaults to ``VerbosityLevels.SOME``.
                For ``VerbosityLevels.ALL``, we print deep diagnostic
                information
            tmp_array_limit_gb: Size limit for temporary buffers in device
                memory, for forward computations
            debug_single_cell_per_row: Internal option, used for unit testing.
            qname: Determines how checkpoints are stored. See
                :const:`SUPPORTED_QUANTIZERS`.
            cache_kwargs: Additional kwargs for creating the cache buffers for
                checkpointing, and inference replay caches
            train_cache_kwargs: Arguments for training replay caches in
                :class:`CellComputation`.
            backward_tmp_array_limit_gb: Same role as `tmp_array_limit_gb`, but
                for backward computations. Overrides "tmp_array_limit_gb"
                entries in `cache_kwargs`, `train_cache_kwargs`.
            debug_dont_use_autograd_hooks: Internal option, used for unit
                testing. If this is set, autograd saved tensors hooks are not
                used, and we also do not use memory efficient attention.
            use_arrays_cleanup: We try and track arrays allocated during the
                backward computation and free them in :meth:`_clear_backward`.
                Supports recovery from OOM errors mechanism.
            profile_steps: We measure times of different parts of a gradient
                computation.
            offload_device: See above.
            layer_checkpoint_chunk_size: If `qname != "default"`, layer input
                checkpoints are quantized. Quantization is done in chunks of
                this length. Determines GPU memory requirements. A value close
                to the cache length is recommended.

        """
        super().__init__(
            gpt_model,
            head_model,
            chunk_size,
            randomize_chunk_sizes,
            chunks_per_cell_multiplier,
            single_tokens_for_targets,
            verbose,
            tmp_array_limit_gb,
            debug_single_cell_per_row,
            debug_store_intermediates,
        )
        if qname is None:
            qname = "torch-quantized8"
        elif qname not in SUPPORTED_QUANTIZERS:
            raise ValueError(f"qname = {qname} is not supported, must be in {SUPPORTED_QUANTIZERS}")
        if not (1 <= layers_per_cell <= gpt_model.config.n_layer):
            raise ValueError(f"layers_per_cell = {layers_per_cell}, must be in [1, {gpt_model.config.n_layer}]")
        self.layers_per_cell = layers_per_cell
        self.qname = qname
        if cache_kwargs is None:
            cache_kwargs = dict()
        elif "tmp_array_limit_gb" in cache_kwargs:
            del cache_kwargs["tmp_array_limit_gb"]
            print("Use `backward_tmp_array_limit_gb` instead of `cache_kwargs['tmp_array_limit_gb']`")
        self.cache_kwargs = cache_kwargs
        if train_cache_kwargs is None:
            train_cache_kwargs = dict()
        elif "tmp_array_limit_gb" in train_cache_kwargs:
            del train_cache_kwargs["tmp_array_limit_gb"]
            print("Use `backward_tmp_array_limit_gb` instead of `train_cache_kwargs['tmp_array_limit_gb']`")
        self._train_cache_kwargs = train_cache_kwargs
        if autograd_hooks_kwargs is None:
            autograd_hooks_kwargs = dict()
        self._autograd_hooks_kwargs = autograd_hooks_kwargs
        # Device memory limit for backward computations:
        self._backward_tmp_array_limit_gb = backward_tmp_array_limit_gb
        self._debug_dont_use_autograd_hooks = debug_dont_use_autograd_hooks
        self._use_arrays_cleanup = use_arrays_cleanup
        # Attention logit softcapping is not supported by the special operators
        # used during gradient computations
        if self.config.attention_logit_softcapping is not None:
            raise ValueError("Long context gradient computation requires gpt_model.config.attention_logit_softcapping = None")
        # Annotation usage logs
        self._annotation_usage_logs: Dict[Tuple[int, int], AnnotationUsageLog] = dict()
        # Status is "init" or "forward_done"
        self._status = "init"
        self.layer_checkpoints = None
        self.autograd_hooks = None
        self.accumulator = None
        self._input_ids = None
        self._targets = None
        self._replay_logs = None
        self._record_gpu_memory_snapshots = None
        self._record_gpu_memory_kind = None
        self._profile_records = [] if profile_steps else None
        self._timer_start = None
        self.offload_device = offload_device
        self._init_cpu_offloading()
        if qname != "default" and layer_checkpoint_chunk_size is None:
            raise ValueError("layer_checkpoint_chunk_size must be given if qname != 'default'")
        self._layer_checkpoint_chunk_size = layer_checkpoint_chunk_size
        self._debug_gpt_model = debug_gpt_model
        if debug_store_intermediates:
            self._train_cache_kwargs = dict(
                self._train_cache_kwargs,
                debug_intermediates=self.debug_intermediates,
            )

    @property
    def status(self) -> str:
        return self._status

    def _init_cpu_offloading(self):
        if self.offload_device is not None:
            check_model_is_on_device(
                self.gpt_model, torch.device("cpu"),"gpt_model",
            )
            check_model_is_on_device(
                self.head_model, self.offload_device, "head_model",
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        scale_factor: float = 1.0,
        **kwargs,
    ) -> Union[LossValue, torch.Tensor]:
        """
        Different to `GPT.forward`, this is processing a batch of full
        sequences. It also evaluates the head model and computes the loss
        function.

        Args:
            input_ids: Batch of full input token sequences
            targets: Targets, these are right-aligned with `input_ids`
            scale_factor: Loss is multiplied by this factor. Defaults to 1.

        Returns:
            Loss value(s). In training mode, this is of type :class:`LossValue`,
            for which :meth:`backward` is overwritten, and of shape `(1,)`.
            In evaluation mode, we return loss values for batch dimension,
            shape `(batch_size,)`.

        """
        self._check_status("init")
        self._init_members_from_tokens(input_ids, targets)
        if not isinstance(self.gpt_model.mha, MultiHeadSelfAttention):
            raise ValueError(f"type(self.gpt_model.mha) = {type(self.gpt_model.mha)}, must be MultiHeadSelfAttention")
        if self._record_gpu_memory_snapshots is None:
            self._record_gpu_memory_snapshots = kwargs.get("record_gpu_memory_snapshots")
            if self._record_gpu_memory_snapshots is not None:
                self._record_gpu_memory_kind = kwargs.get("record_gpu_memory_kind")
        if self.training:
            self._timer_start = time.perf_counter()  # Start timer
            loss_value = self._inference_forward_pass(
                input_ids, targets, scale_factor,
            )
        else:
            loss_value =  self._forward_only(
                input_ids, targets, scale_factor,
            )
        return loss_value

    def _check_status(self, required_status: str):
        if self._status != required_status:
            raise IndexError(f"status = '{self._status}', must be '{required_status}'")

    def ready_for_backward(self) -> bool:
        return self._status == "forward_done"

    def backward(self):
        """
        Runs gradient computation for a batch, passed as `input_ids` to
        :meth:`forward`, and `targets` to :meth:`complete_forward`. Here,
        `targets` can be shorter than `input_ids`, in which case they are
        right-aligned: `input_ids[:, -k]` goes with `targets[:, -k]`.

        Note that all input and target sequences in the batch must have the
        same length. We recommend to cluster data so that real input and output
        lengths are similar in a batch. Then, pad inputs on the left and
        outputs on the right.

        """
        self._check_status("forward_done")
        if not self.training:
            raise IndexError("Must be in training mode for gradient computations")
        self._backward_accumulate_gradients()
        self.clear()  # Reset, also status to "init"

    def clear(self):
        """
        Resets members created in `_init_members_from_tokens` to `None`.

        """
        super().clear()
        self._status = "init"
        self.layer_checkpoints = None
        self._input_ids = None
        self._targets = None
        self._replay_logs = None
        self._record_gpu_memory_snapshots = None
        self._record_gpu_memory_kind = None
        self._clear_backward()

    def _clear_backward(self):
        del self.accumulator
        self.accumulator = None
        if self.autograd_hooks is not None:
            # Sometimes, arrays of the autograd graph do not get deallocated.
            # We do that here.
            if self._use_arrays_cleanup:
                self.autograd_hooks.arrays_cleanup.cleanup()
            self.autograd_hooks.clear()
            del self.autograd_hooks
            self.autograd_hooks = None
        self._annotation_usage_logs = dict()
        gc.collect()
        torch.cuda.empty_cache()

    def profile_records(self) -> Optional[List[Dict[str, float]]]:
        return self._profile_records

    def annotation_usage_logs(self) -> Dict[Tuple[int, int], AnnotationUsageLog]:
        """
        See header comments.

        Returns:
            Annotation usage logs, as dictionary with keys
            `(first_layer_idx, first_chunk_idx)`.

        """
        return self._annotation_usage_logs

    def copy_model_for_evaluation(self) -> LongContextInferenceModel:
        """
        Only if `offload_device` is given.

        Returns:
            :class:`LongContextInferenceModel` copy of this model on device
            `offload_device`.

        """
        if self.offload_device is None:
            raise IndexError("Only to be used if `offload_device` is set")
        do_timing = self.verbose is not VerbosityLevels.NONE
        timer_start = None
        if self.verbose is not VerbosityLevels.NONE:
            print(f"Copy parameters to {self.offload_device}")
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
            timer_start = time.perf_counter()
        gpt_model_copy = clone_model_shard_via_flat_vectors(
            model=self.gpt_model,
            device=self.offload_device,
            shard_type=None,
            lm_head=True,
        )
        model_copy = LongContextInferenceModel(
            gpt_model=gpt_model_copy,
            head_model=self.head_model,
            chunk_size=self.chunk_size,
            randomize_chunk_sizes=self.randomize_chunk_sizes,
            chunks_per_cell_multiplier=self.chunks_per_cell_multiplier,
            single_tokens_for_targets=self.single_tokens_for_targets,
            verbose=self.verbose,
            tmp_array_limit_gb=self._tmp_array_limit_gb,
            debug_single_cell_per_row=self._debug_single_cell_per_row,
            debug_store_intermediates=self.debug_intermediates is not None,
        )
        if do_timing:
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
            time_in_secs = time.perf_counter() - timer_start
            print(f"Done in {time_in_secs:.2f} seconds")
        return model_copy

    def _init_members_from_tokens(
        self, input_ids: torch.Tensor, targets: torch.Tensor,
    ):
        """
        Initialize members required for processing the current batch.

        """
        super()._init_members_from_tokens(input_ids, targets)
        if self.training:
            # Create checkpointing members
            self._create_layer_checkpointers()
        # These are needed in :meth:`backward`
        self._input_ids = input_ids
        self._targets = targets

    def _create_layer_checkpointers(self):
        # Layer input checkpoints
        layer_numbers = self._create_layer_numbers()
        dtype = self.gpt_model.get_kv_cache_params(0).dtype
        if self.qname == "default":
            # Checkpoints are not quantized
            self.layer_checkpoints = LayerInputDefaultCheckpoints(
                layer_numbers=layer_numbers,
                batch_size=self.batch_size,
                max_seq_length=self.gpt_model.max_seq_length,
                n_embd=self.config.n_embd,
                dtype=dtype,
            )
        else:
            # Checkpoints are quantized
            self.layer_checkpoints = LayerInputQuantizedCheckpoints(
                model=self.gpt_model,
                layer_numbers=layer_numbers,
                batch_size=self.batch_size,
                chunk_size=self._layer_checkpoint_chunk_size,
                qname=self.qname,
                cache_kwargs=dict(
                    self.cache_kwargs,
                    tmp_array_limit_gb=self._tmp_array_limit_gb,
                ),
                allocate_buffers=True,
                device=self.offload_device,
            )

    def _create_layer_numbers(self) -> List[int]:
        """
        These are layer numbers so that cells run over layers
        `range(layer_numbers[i], layer_numbers[i + 1])`, and
        `layer_numbers[-2] == self.config.n_layer`. They are chosen based
        on `self.layers_per_cell`, but with the constraint that cells do not
        range over several devices.

        The slot corresponding to `layer_numbers[-1] ==
        self.config.n_layer + 1` is used to store head gradients during the
        backward computation.

        """
        n_layer = self.config.n_layer
        layer_numbers = list(range(0, n_layer, self.layers_per_cell))
        # Don't want slim final row of cells
        if self.layers_per_cell > 1 and layer_numbers[-1] == n_layer - 1:
            layer_numbers = layer_numbers[:-1]
        device_points = []
        start = 0
        prev_device = device_for_layer(self.gpt_model, 0)
        for pos in range(1, n_layer):
            device = device_for_layer(self.gpt_model, pos)
            if device != prev_device:
                device_points.append(start)
                start = pos
                prev_device = device
        device_points.append(start)
        if len(device_points) > 1:
            if len(device_points) >= len(layer_numbers):
                # Just use the partitioning dictated by the devices
                layer_numbers = device_points
            else:
                # Consolidate
                consolidated = [0]
                assert device_points.pop(0) == 0
                device_point = device_points.pop(0)
                device_points.append(self.config.n_layer + 1)
                prev_is_device = True  # Is `prev_point` a device point?
                for point in layer_numbers[1:]:
                    prev_point = consolidated[-1]
                    if prev_point == device_point:
                        prev_is_device = True
                    # We have: `prev_point <= device_point`
                    do_pop = True
                    if point <= device_point:
                        # Accept original segment
                        consolidated.append(point)
                        do_pop = point == device_point
                        prev_is_device = do_pop
                    else:
                        # Original segment needs to be split:
                        # `prev_point <= device_point < point`
                        # Fuse smaller piece, create new segment for larger
                        left = device_point - prev_point
                        right = point - device_point
                        if left <= right:
                            # Fuse left part with previous
                            if not prev_is_device or left == 0:
                                consolidated[-1] = device_point
                            else:
                                # Cannot fuse, so create segment
                                consolidated.append(device_point)
                            # Right part becomes new segment
                            consolidated.append(point)
                            prev_is_device = False
                        else:
                            # New segment for left part
                            consolidated.append(device_point)
                            # Try to fuse
                            prev_is_device = True
                    if do_pop:
                        device_point = device_points.pop(0)
                if consolidated[-1] < device_point < n_layer:
                    consolidated.append(device_point)
                layer_numbers = consolidated

        return layer_numbers + [n_layer, n_layer + 1]

    def _deallocate_buffers(self):
        if self.qname != "default":
            assert isinstance(self.layer_checkpoints, LayerInputQuantizedCheckpoints)
            self.layer_checkpoints.clear()

    def _create_members_for_backward(self):
        if self._use_arrays_cleanup:
            arrays_cleanup = ArraysForCleanup(
                protected_ids=protect_named_params_buffers_of_model(self.gpt_model, map_names=True)
            )
        else:
            arrays_cleanup = None
        if not self._debug_dont_use_autograd_hooks:
            # Autograd hooks for cell computations
            self.autograd_hooks = CellComputationAutogradHooks(
                config=self.config,
                batch_size=self.batch_size,
                arrays_cleanup=arrays_cleanup,
                **self._autograd_hooks_kwargs,
            )
        elif self._use_arrays_cleanup:
            self.autograd_hooks = CleanupArraysAutogradHooks(arrays_cleanup)
        else:
            self.autograd_hooks = None
        # Accumulator object
        # Key and value buffers should not be annotated for the first chunk if
        # there is only a single chunk in the first cell
        self.accumulator = GradientAccumulator(
            config=self.config,
            autograd_hooks=self.autograd_hooks,
            qname=self.qname,
            cache_kwargs=dict(
                self.cache_kwargs,
                tmp_array_limit_gb=self._backward_tmp_array_limit_gb,
            ),
            verbose=self.verbose,
            train_cache_kwargs=dict(
                self._train_cache_kwargs,
                tmp_array_limit_gb=self._backward_tmp_array_limit_gb,
            ),
        )

    def _checkpoint_layer_input(
        self,
        x: torch.Tensor,
        layer_idx: int,
        input_pos: int,
    ):
        if self.training:
            self.layer_checkpoints.set_checkpoint(
                layer_idx=layer_idx,
                buffers=x,
                input_pos=input_pos,
            )

    def _inference_forward_pass(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        scale_factor: float,
    ) -> LossValue:
        if self.verbose is not VerbosityLevels.NONE:
            lines = [
                f"\nbatch_size      = {self.batch_size}",
                f"seq_length      = {self.gpt_model.max_seq_length}"
            ]
            # Caches can have different lengths
            cache_lengths = [
                (l_ix, block.attn.kv_cache.cache_length)
                for l_ix, block in enumerate(block_iterator(self.gpt_model))
            ]
            cache_length = cache_lengths[0][1]
            if all (x[1] == cache_length for x in cache_lengths):
                lines.append(f"cache_length    = {cache_length}")
            else:
                cl_str = ", ".join(f"{i}:{j}" for i, j in cache_lengths)
                lines.append("cache_length    = " + cl_str)
            lines.extend(
                [
                    f"chunk_sizes     = {self.chunk_sizes}",
                    f"layers_per_cell = {self.layers_per_cell}",
                    f"chunks_per_cell = {self.chunks_per_cell}\n",
                    f"Forward pass over {len(self.chunk_sizes)} chunks, grouped into {len(self.chunks_per_cell)} cells (training mode)"
                ]
            )
            print("\n".join(lines))

        if self.offload_device is not None:
            # Clone `gpt_model` to `offload_device`
            gpt_model = clone_model_shard_via_flat_vectors(
                model=self.gpt_model,
                device=self.offload_device,
                shard_type=None,
                lm_head=self.head_model.needs_logits(),
            )
            if self.verbose is not VerbosityLevels.NONE:
                print(
                    f"\nCopied complete model to device {self.offload_device}:\n"
                    + message_with_device_memory(self.offload_device)
                )
        else:
            gpt_model = self.gpt_model

        gpt_model_old = self.gpt_model
        try:
            self.gpt_model = gpt_model
            # Ensure that all KV caches record replay logs
            for block in block_iterator(self.gpt_model):
                block.attn.kv_cache.switch_replay_logging(True)

            # Run inference forward pass. Layer inputs are checkpointed. Mean
            # reduction over batch dimension.
            loss_full = self._forward_internal(
                input_ids, targets, scale_factor,
            ).mean()
        finally:
            # Restore
            self.gpt_model = gpt_model_old

        # Replay logs from KV caches are required in
        # :meth:`_backward_accumulate_gradients`.
        self._replay_logs = []
        for block in block_iterator(gpt_model):
            cache = block.attn.kv_cache
            self._replay_logs.append(cache.get_replay_log())
            cache.switch_replay_logging(False)

        deallocate_kv_cache_buffers_of_model(gpt_model)
        if self.offload_device is not None:
            del gpt_model
        gc.collect()
        torch.cuda.empty_cache()
        if self.offload_device is not None and self.verbose is not VerbosityLevels.NONE:
            print(
                f"\nDeallocated weights of model on device {self.offload_device}:\n"
                + message_with_device_memory(self.offload_device)
            )
        self._status = "forward_done"
        return LossValue(loss_full, model=self)

    def _backward_accumulate_gradients(self):
        """
        Wrapper around :meth:`_backward_accumulate_gradients_nocheck`.
        First, KV cache buffers are deallocated and replay logs are gathered.

        If `tmp_array_limit_backward` is set, we catch out of memory errors,
        reduce the limit value and try again. Done only a limited
        number of times, see :class:`TemporaryArrayLimit`.

        """
        if self._profile_records is not None:
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
            prof_record = {
                "forward_time": time.perf_counter() - self._timer_start
            }
            self._timer_start = time.perf_counter()
        else:
            prof_record = None
        if self._replay_logs is None:
            raise IndexError("No KV cache replay logs: Must call `forward` before `backward`")
        if self.verbose is not VerbosityLevels.NONE:
            print("\nAllocate storage for backward computation")

        # Call :meth:`_backward_accumulate_gradients_nocheck`. May be done
        # several times with reduced memory limits
        if self._backward_tmp_array_limit_gb is None:
            self._backward_accumulate_gradients_nocheck(0)
        else:
            is_done = False
            count = 0
            while not is_done:
                try:
                    self._backward_accumulate_gradients_nocheck(count)
                    is_done = True
                except RuntimeError as ex:
                    oom_exception_action(ex, self._backward_tmp_array_limit_gb)
                    self.gpt_model.zero_grad(set_to_none=True)
                    self._clear_backward()
                    self._status = "forward_done"
                    count += 1

        if self._profile_records is not None:
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
            prof_record["backward_time"] = time.perf_counter() - self._timer_start
            self._profile_records.append(prof_record)
        self._status = "init"  # Reset
        # Summary of annotation usage logs
        if not self._debug_dont_use_autograd_hooks and self.verbose is not VerbosityLevels.NONE:
            num_unmatched_args = [
                (
                    idx,
                    len(log.unmatched_pack_args),
                    log.num_matched_annotations,
                    log.num_comparisons,
                    log.num_4d_indexes,
                    log.num_unmatched_scatter_cat,
                )
                for idx, log in self._annotation_usage_logs.items()
            ]
            total_num_unmatched = sum(x[1] for x in num_unmatched_args)
            if total_num_unmatched == 0:
                print("\nSuccess: All pack arguments were matched in all cells.\n")
            else:
                lines = [
                            "\nThere were unmatched pack arguments in some cells. Use --kv_cache.verbose all for full information."
                        ] + [
                            f"{num} unmatched in ({fli},{fci}): {n_ma} matches, {n_cmp} comparisons, {n_unm} scatter/cat, {n_4d} 4D indexes"
                            for (fli, fci), num, n_ma, n_cmp, n_4d, n_unm
                            in num_unmatched_args if num > 0
                        ] + [""]
                print("\n".join(lines))

    def _backward_accumulate_gradients_nocheck(self, count: int):
        """
        Main workhorse. Runs nested activation checkpointing in order to
        accumulate gradients in the model.

        Head gradients are written to `layer_checkpoints`, using
        `layer_idx = config.n_layer + 1`. This way, they do not overwrite the
        layer input checkpoints, so this method can be called again after an
        OOM error.

        """
        def device_for_slice(layer_idx):
            if self.offload_device is not None:
                return self.offload_device
            else:
                return device_for_layer(
                    self.gpt_model,
                    min(layer_idx, self.gpt_model.config.n_layer - 1),
                )

        def get_inputs_slice(
            start: int, end: int, layer_idx: int,
        ) -> torch.Tensor:
            return self.layer_checkpoints.get_checkpoint(
                layer_idx=layer_idx,
                input_pos=start,
                num=end - start,
                device=device_for_slice(layer_idx),
            )

        def get_head_gradients_slice(start: int, end: int) -> torch.Tensor:
            n_layer = self.gpt_model.config.n_layer
            return self.layer_checkpoints.get_checkpoint(
                layer_idx=n_layer + 1,
                input_pos=start,
                num=end - start,
                device=device_for_slice(n_layer - 1),
            )

        def write_head_gradients_slice(
            input_pos: int, value: torch.Tensor,
        ) -> Optional[int]:
            n_layer = self.gpt_model.config.n_layer
            return self.layer_checkpoints.set_checkpoint(
                layer_idx=n_layer + 1,
                buffers=value,
                input_pos=input_pos,
            )

        # Sanity check:
        assert self.layer_checkpoints.layer_numbers[-1] == self.gpt_model.config.n_layer + 1
        if self._record_gpu_memory_kind in (0, 2):
            self._record_gpu_memory_snapshots.store_current_snapshot()
            if self._record_gpu_memory_kind == 2:
                self._record_gpu_memory_snapshots.stop_recording()
                self._record_gpu_memory_snapshots.set_path(
                    self._record_gpu_memory_snapshots.path.parent / f"snapshot_backward{count}.pickle"
                )
                self._record_gpu_memory_snapshots.start_recording()

        # Allocate members needed for backward computations
        self._create_members_for_backward()
        # Reset annotation usage logs
        self._annotation_usage_logs = dict()

        # Start with gradient w.r.t. head model, which also provides the
        # head gradients for the final layer.
        if self.verbose is VerbosityLevels.SOME:
            num_rows = len(self.layer_checkpoints.layer_numbers) - 2
            print(f"\nRunning backward pass over {num_rows} rows of cells, {self.config.n_layer} layers, using activation checkpointing")
        if self.offload_device is not None:
            shard_on_device = clone_model_shard_via_flat_vectors(
                model=self.gpt_model,
                device=self.offload_device,
                shard_type="lm_head",
                lm_head=self.head_model.needs_logits(),
            )
            target_device = self.offload_device
        else:
            shard_on_device = self.gpt_model
            target_device = device_for_layer(self.gpt_model, -1)
        self._targets = self._targets.to(device=target_device)
        self.accumulator.run_head_model(
            gpt_model=shard_on_device,
            head_model=self.head_model,
            replay_logs=self._replay_logs,
            chunks_per_cell=self.chunks_per_cell,
            get_inputs_slice=partial(get_inputs_slice, layer_idx=self.config.n_layer),
            write_head_gradients_slice=write_head_gradients_slice,
            targets=self._targets,
        )
        if self.offload_device is not None:
            module_pairs = [
                (
                    shard_on_device.transformer.ln_f,
                    self.gpt_model.transformer.ln_f,
                )
            ]
            if self.head_model.needs_logits():
                module_pairs.append(
                    (shard_on_device.lm_head, self.gpt_model.lm_head)
                )
            if self._debug_gpt_model is not None:
                debug_modules = [self._debug_gpt_model.transformer.ln_f]
                if self.head_model.needs_logits():
                    debug_modules.append(self._debug_gpt_model.lm_head)
            else:
                debug_modules = None
            accumulate_gradients(module_pairs, debug_modules)
            del shard_on_device

        if self._record_gpu_memory_kind == 1:
            # End of recording for initial snapshot (everything before the
            # backward loop over layers)
            self._record_gpu_memory_snapshots.store_current_snapshot()
            self._record_gpu_memory_snapshots.stop_recording()

        # Loop over rows of cells, from the top down.
        # Note that `layer_checkpoints.layer_numbers[-1] == n_layer + 1` is used
        # for storing head gradients
        layer_numbers = self.layer_checkpoints.layer_numbers[:-1]
        for first_layer_idx, end_layer_idx in wrap_tqdm_if_verbose(
            reversed(
                list(zip(layer_numbers[:-1], layer_numbers[1:]))
            ),
            verbose=self.verbose is VerbosityLevels.SOME,
        ):
            if self._use_arrays_cleanup and self.autograd_hooks is not None:
                self.autograd_hooks.arrays_cleanup.reset()
            num_layers = end_layer_idx - first_layer_idx
            if self.offload_device is None:
                shard_on_device = None
                model_part = DefaultCellBlocks(
                    model=self.gpt_model,
                    first_layer_idx=first_layer_idx,
                    num_layers=num_layers,
                )
            else:
                shard_on_device = clone_model_shard_via_flat_vectors(
                    model=self.gpt_model,
                    device=self.offload_device,
                    shard_type=f"h{first_layer_idx}:{end_layer_idx}",
                    lm_head=self.head_model.needs_logits(),
                )
                model_part = GPTShardCellBlock(shard_on_device)
            # Does gradient accumulation for all weights in layers covered
            # by `model_part`. Also,
            # `head_gradients` is overwritten by the "bottom gradients", which
            # are head gradients for the row of cells below.
            record_path = None if self._record_gpu_memory_snapshots is None else self._record_gpu_memory_snapshots.path
            if record_path is not None and self._record_gpu_memory_kind == 1:
                # Change path for storage
                record_path = str(Path(record_path).parent / f"snapshot_layer{first_layer_idx}.pickle")
                snapshots = RecordGPUMemory(
                    path=record_path,
                    max_entries=self._record_gpu_memory_snapshots.max_entries,
                )
            elif self._record_gpu_memory_kind is None:
                snapshots = self._record_gpu_memory_snapshots
            else:
                snapshots = None
            self.accumulator.run(
                model_part=model_part,
                get_inputs_slice=partial(get_inputs_slice, layer_idx=first_layer_idx),
                get_head_gradients_slice=get_head_gradients_slice,
                write_head_gradients_slice=write_head_gradients_slice,
                record_gpu_memory_snapshots=snapshots,
            )
            if self.offload_device is not None:
                module_pairs = [
                    (
                        shard_on_device.transformer.h[i - first_layer_idx],
                        self.gpt_model.transformer.h[i],
                    )
                    for i in range(first_layer_idx, end_layer_idx)
                ]
                if self._debug_gpt_model is not None:
                    debug_modules = [
                        self._debug_gpt_model.transformer.h[i]
                        for i in range(first_layer_idx, end_layer_idx)
                    ]
                else:
                    debug_modules = None
                accumulate_gradients(module_pairs, debug_modules)
                del model_part
                del shard_on_device

            for first_chunk_idx, annot_log in self.accumulator.annotation_usage_logs().items():
                self._annotation_usage_logs[
                    (first_layer_idx, first_chunk_idx)
                ] = annot_log
            if self._record_gpu_memory_kind in (0, 2):
                # Store results up to now
                self._record_gpu_memory_snapshots.store_current_snapshot()
            # DEBUG:
            print("Done backward for one row of cells: STOP HERE")
            exit(0)
            # END DEBUG

        # Accumulate gradients for input embeddings
        if self._record_gpu_memory_kind == 1:
            # Start recording for final snapshot
            record_path = Path(self._record_gpu_memory_snapshots.path)
            self._record_gpu_memory_snapshots.path = str(record_path.parent / "snapshot_final.pickle")
            self._record_gpu_memory_snapshots.start_recording()
        if self.offload_device is not None:
            shard_on_device = clone_model_shard_via_flat_vectors(
                model=self.gpt_model,
                device=self.offload_device,
                shard_type="wte",
                lm_head=self.head_model.needs_logits(),
            )
        else:
            shard_on_device = self.gpt_model
        self.accumulator.run_input_embeddings(
            gpt_model=shard_on_device,
            input_ids=self._input_ids,
            get_head_gradients_slice=get_head_gradients_slice,
        )
        if self.offload_device is not None:
            module_pairs = [
                (
                    shard_on_device.transformer.wte,
                    self.gpt_model.transformer.wte,
                )
            ]
            if self._debug_gpt_model is not None:
                debug_modules = [self._debug_gpt_model.transformer.wte]
            else:
                debug_modules = None
            accumulate_gradients(module_pairs, debug_modules)
            del shard_on_device

        self._deallocate_buffers()
        if self._record_gpu_memory_kind in (0, 2):
            self._record_gpu_memory_snapshots.store_current_snapshot()
            if self._record_gpu_memory_kind == 2:
                self._record_gpu_memory_snapshots.stop_recording()


class NaiveGPTAndHeadModel(GPTAndHeadModel):
    def __init__(
        self,
        gpt_model: GPT,
        head_model: HeadModel,
    ):
        super().__init__(gpt_model, head_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        scale_factor: float = 1.0,
        **kwargs,
    ) -> Union[LossValue, torch.Tensor]:
        model_outputs = self.gpt_model(input_ids)
        loss_value = self.head_model(model_outputs, targets, input_pos=0) * scale_factor
        return loss_value
