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
from collections.abc import Callable
from typing import List, Tuple, Optional, Dict
from itertools import accumulate

import torch
import torch.nn as nn

from litgpt.config import Config

from keys_values.attention import DefaultKeysAndValues
from keys_values.kvcache.base import KVCacheReplayLog
from keys_values.kvcache.gradient.autograd_hooks import CellComputationAutogradHooks
from keys_values.kvcache.gradient.train_attn_weights_replay import TrainingAttnWeightsReplayCache
from keys_values.kvcache.stack_layers import CellBlocks


GetInputSlice = Callable[[int, int], torch.Tensor]

WriteOutputsSlice = Callable[[int, torch.Tensor], None]


def cell_computation(
    token_idxs: List[torch.Tensor],
    model_part: CellBlocks,
    get_inputs_slice: GetInputSlice,
    input_pos: int,
) -> List[torch.Tensor]:
    """
    Implements forward pass for one cell, outer loop over chunks in `token_idxs`,
    inner loop over layers `first_layer_idx` until
    `first_layer_idx` + `num_layers - 1`.

    Args:
        token_idxs: List to token indexes for chunks
        model_part: Layers of GPT model to be used. KV caches must have been
            assigned.
        get_inputs_slice: Inputs to layer `first_layer_idx`
        input_pos: Input position for first chunk

    """
    output_parts = []
    for token_idx in token_idxs:
        chunk_len = token_idx.shape[-1]
        x = get_inputs_slice(input_pos, input_pos + chunk_len)
        x = model_part.forward(
            x=x,
            idx=token_idx,
            input_pos=input_pos,
        )
        output_parts.append(x)
        input_pos += chunk_len
    return output_parts


class CellComputation(nn.Module):
    """
    A long context forward pass for a :class:`GPT` model can be seen as 2D
    lattice, with layers as rows and token chunks as columns. Here, token
    chunks are the units for which :meth:`GPT.forward` is called for (the
    `idx` and `input_pos` arguments). The first token chunk (`input_pos=0`)
    is usually much larger than subsequent ones. This lattice consists of
    blocks, indexed by `(l, s)`, `l` the layer index, `s` the chunk index.

    For activation checkpointing, we form cells as rectangular groups of
    blocks. Forward-backward computations to accumulate gradients are done
    on cells. This class is representing these computations for one cell.
    This works by each layer (or row) in the cell being represented by a
    replay cache of type :class:`AttnWeightsReplayCache`. These caches are
    linked to a :class:`CellComputationAutogradHooks` object which creates
    `autograd` saved tensors packing and unpacking hooks. The first
    forward pass, apart from storing layer input checkpoints, runs KV
    caching in all layers and stores cache replay logs of type
    :class:`AttnWeightsReplayLog`, with which the replay caches are
    parameterized.

    This module is used to run these computations for all cells of a
    row, spanning `num_layers` layers starting from `first_layer_idx`, and with
    batch size `batch_size`. :meth:`forward` can process the different cells
    of a row.

    """
    def __init__(
        self,
        model_part: CellBlocks,
        autograd_hooks: Optional[CellComputationAutogradHooks],
        replay_logs: List[KVCacheReplayLog],
        batch_size: int,
        debug_tensors: Optional[Dict[str, torch.Tensor]] = None,
        **cache_kwargs,
    ):
        """
        Args:
            model_part: Layers of GPT model whose gradients are to be
                accumulated. In :meth:`forward`, we temporarily replace the
                KV caches with training replay caches.
            autograd_hooks: Maintains the autograd saved tensors hooks. This is
                optional, cell computations can be done without the hooks
            replay_logs: KV cache replay logs for layers treated here
            batch_size: Batch size

        """
        super().__init__()
        self.check_args(
            model_part.config, autograd_hooks, replay_logs, batch_size,
        )
        if model_part.num_layers != len(replay_logs):
            raise ValueError(f"model_part.num_layers = {model_part.num_layers} != {len(replay_logs)} = len(replay_logs)")
        self.model_part = model_part
        self.config = model_part.config
        self.autograd_hooks = autograd_hooks
        self.replay_logs = replay_logs
        self.batch_size = batch_size
        # Arguments for `TrainingAttnWeightsReplayCache` cache objects.
        kwargs = dict(
            cache_kwargs,
            config=self.config,
            batch_size=batch_size,
        )
        if autograd_hooks is not None:
            kwargs["node_annotations"] = autograd_hooks.node_annotations
        self._train_cache_kwargs = [
            dict(
                kwargs,
                device=block.attn.device,
                cache_length=block.attn.kv_cache.cache_length,
            )
            for _, block in model_part.blocks()
        ]
        self._token_pos_per_chunk = [0] + list(
            accumulate(
                idx.shape[-1] for idx in replay_logs[0].token_chunks
            )
        )
        self._debug_tensors = debug_tensors  # DEBUG

    @staticmethod
    def check_args(
        config: Config,
        autograd_hooks: Optional[CellComputationAutogradHooks],
        replay_logs: List[KVCacheReplayLog],
        batch_size: int,
    ):
        if autograd_hooks is not None:
            if autograd_hooks.batch_size < batch_size:
                raise ValueError(f"autograd_hooks.batch_size = {autograd_hooks.batch_size}, must be >= batch_size = {batch_size}")
            if config.n_query_groups != autograd_hooks.n_query_groups or config.head_size != autograd_hooks.head_size:
                raise ValueError(f"config and autograd_hooks not consistent")

        def chunks_equal(a: List[torch.Tensor], b: List[torch.Tensor]) -> bool:
            return len(a) == len(b) and all(
                xa.equal(xb) for xa, xb in zip(a, b)
            )

        dtype = replay_logs[0].dtype
        for i, log in enumerate(replay_logs):
            if not chunks_equal(log.token_chunks, replay_logs[0].token_chunks):
                raise ValueError(
                    "All replay_logs must have the same chunk sequence:\n"
                    f"0: {replay_logs[0].token_chunks}\n"
                    f"{i}: {log.token_chunks}"
                )
            if log.dtype != dtype:
                raise ValueError(
                    "All replay_logs must have the dtype:\n"
                    f"0: {dtype}\n"
                    f"{i}: {log.dtype}"
                )

    def _create_train_replay_caches(
        self,
        first_chunk_idx: int,
        num_chunks: int,
        debug_print_annotations: bool,
    ) -> List[TrainingAttnWeightsReplayCache]:
        start_token_pos = self._token_pos_per_chunk[first_chunk_idx]
        first_layer_idx = self.model_part.first_layer_idx
        return [
            TrainingAttnWeightsReplayCache(
                **kwargs,
                replay_log=replay_log,
                start_token_pos=start_token_pos,
                layer_idx=first_layer_idx + rel_block_idx,
                num_chunks=num_chunks,
                debug_tensors=self._debug_tensors,
                debug_print_annotations=debug_print_annotations,
            )
            for rel_block_idx, (replay_log, kwargs) in enumerate(
                zip(self.replay_logs, self._train_cache_kwargs)
            )
        ]

    def get_input_pos(self, first_chunk_idx: int) -> int:
        return sum(
            idx.shape[-1] for idx in self.replay_logs[0].token_chunks[:first_chunk_idx]
        )

    def forward(
        self,
        get_inputs_slice: GetInputSlice,
        k_buffers: Optional[List[torch.Tensor]],
        v_buffers: Optional[List[torch.Tensor]],
        first_chunk_idx: int,
        num_chunks: int,
        debug_print_annotations: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        The cell receives input from the bottom (`inputs`) and from the
        left (`k_buffers`, `v_buffers`), the latter only if
        `first_chunk_idx > 0`. It has outputs on the top (`outputs`) and on the
        right (`k_buffers`, `v_buffers`).

        Args:
            get_inputs_slice: Access to inputs of layer `first_layer_idx`
            k_buffers: Keys buffers for all layers, before chunk
                `first_chunk_idx` is processed (or `None` if
                `first_chunk_idx == 0`). Shape of entries is
                `(batch_size, n_query_groups, cache_length, head_dim)`
            v_buffers: Values buffers for all layers, before chunk
                `first_chunk_idx` is processed (or `None` if
                `first_chunk_idx == 0`). Shape of entries is
                `(batch_size, n_query_groups, cache_length, head_dim)`
            first_chunk_idx: Index of first chunk in cell
            num_chunks: Number of chunks in cell

        Returns:
            (outputs, k_buffers, v_buffers)

        """
        token_idxs = self.replay_logs[0].token_chunks[
            first_chunk_idx:(first_chunk_idx + num_chunks)
        ]
        num_layers = self.model_part.num_layers
        if first_chunk_idx > 0:
            if k_buffers is None or v_buffers is None:
                raise ValueError("Both k_buffers and v_buffers must be given if first_chunk_idx > 0")
            if len(k_buffers) != num_layers or len(v_buffers) != num_layers:
                raise ValueError(f"k_buffers and v_buffers must have length {num_layers}")
            shape = [
                self.batch_size,
                self.config.n_query_groups,
                0,
                self.config.head_size,
            ]
            for (block_idx, block), k_buffer, v_buffer in zip(
                self.model_part.blocks(),
                k_buffers,
                v_buffers,
            ):
                cache_length = block.attn.kv_cache.cache_length
                k_size = k_buffer.shape[2]
                if k_size > cache_length:
                    raise ValueError(f"layer_idx={block_idx}: k_buffers.shape = {k_buffer.shape}, cache_length = {cache_length}. Invalid!")
                shape[2] = k_size
                if k_buffer.shape != tuple(shape):
                    raise ValueError(f"layer_idx={block_idx}: k_buffers.shape = {k_buffer.shape}, must be {shape}")
                if v_buffer.shape != tuple(shape):
                    raise ValueError(f"layer_idx={block_idx}: v_buffers.shape = {v_buffer.shape}, must be {shape}")
        else:
            k_buffers = [None] * num_layers
            v_buffers = [None] * num_layers
        # Hook training replay caches into self attention blocks, and assign
        # buffers.
        train_replay_caches = self._create_train_replay_caches(
            first_chunk_idx, num_chunks, debug_print_annotations,
        )
        kv_caches_copy = []
        cache_lengths = []
        for (block_idx, block), replay_cache, keys, values in zip(
            self.model_part.blocks(),
            train_replay_caches,
            k_buffers,
            v_buffers,
        ):
            attn = block.attn
            device = attn.device
            kv_caches_copy.append(attn.kv_cache)
            if first_chunk_idx > 0:
                # Sanity check
                if keys.device != device or values.device != device:
                    raise IndexError(f"Layer {block_idx}: keys.device={keys.device}, values.device={values.device}, attn.device={device}: Must all be the same")
                buffers = DefaultKeysAndValues(keys=keys, values=values)
                replay_cache.initialize_buffers(buffers)
            cache_lengths.append(attn.kv_cache.cache_length)
            attn.kv_cache = replay_cache
        # Initialize autograd hooks
        if self.autograd_hooks is not None:
            self.autograd_hooks.initialize_cell(
                eff_num_layers=num_layers,
                num_chunks=num_chunks,
                first_layer_idx=self.model_part.first_layer_idx,
                first_chunk_idx=first_chunk_idx,
                cache_lengths=list(cache_lengths),
                replay_logs=self.replay_logs,
            )

        # Outer loop over chunks
        output_parts = cell_computation(
            token_idxs=token_idxs,
            model_part=self.model_part,
            get_inputs_slice=get_inputs_slice,
            input_pos=self.get_input_pos(first_chunk_idx),
        )
        # Reset `model` and compose results
        for (_, block), old_cache in zip(
            self.model_part.blocks(), kv_caches_copy,
        ):
            block.attn.kv_cache = old_cache
        outputs = torch.cat(output_parts, dim=1)
        k_buffers = []
        v_buffers = []
        for kv_cache in train_replay_caches:
            buffers = kv_cache.kv_buffers
            k_buffers.append(buffers.keys())
            v_buffers.append(buffers.values())
            kv_cache.deallocate_buffers()
        return outputs, k_buffers, v_buffers
