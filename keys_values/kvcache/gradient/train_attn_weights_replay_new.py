# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file exc ept in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import replace
from typing import Optional, Tuple, Dict, Any

import torch

from litgpt.config import Config

from keys_values.attention import (
    KeysAndValues,
    DefaultKeysAndValues,
    SDPA_IMPL_QPADDED_PYTORCH,
    SDPA_IMPL_EAGER_BLOCKS,
)
from keys_values.kvcache.attn_weights import (
    update_token_positions,
    UpdateTokenPositionsGracePeriod,
)
from keys_values.kvcache.base import DefaultKVCache, KVCacheReplayLog
from keys_values.kvcache.gradient.autograd_hooks import (
    NodeAnnotation,
    Annotations,
    MAX_DELTA_TRANS_LENGTH,
    create_random_index,
)
from keys_values.kvcache.gradient.sdpa_op import (
    scatter_on_buffers,
    cat_on_buffers,
)
from keys_values.kvcache.utils import shape_to_tuple, for_debug
from keys_values.utils import (
    expand_index,
    need_repeat_interleave,
    repeat_interleave,
)


class TrainingAttnWeightsReplayCacheNew(DefaultKVCache):
    """
    Variant of :class:`TrainingAttnWeightsReplayCache`. Here, we do not use
    the special operators of `sdpa_op`, but employ
    :class:`MultiHeadSelfAttention` as part of the `autograd` graph, with
    `sdpa_mode = SDPA_IMPL_QPADDED_PYTORCH`.

    As in :class:`TrainingAttnWeightsReplayCache`, the main difficulty here is
    to support the autograd saved tensors hook mechanism by annotating the
    tensors properly.

    """
    def __init__(
        self,
        config: Config,
        batch_size: int,
        cache_length: int,
        replay_log: KVCacheReplayLog,
        start_input_pos: int,
        layer_idx: int,
        num_chunks: int,
        node_annotations: Optional[Annotations] = None,
        debug_tensors: Optional[Dict[str, torch.Tensor]] = None,
        debug_print_annotations: bool = False,
        debug_full_args: bool = False,
        debug_intermediates: Optional[Tuple[Dict[str, torch.Tensor], str]] = None,
        **base_kwargs,
    ):
        if not (0 <= start_input_pos < len(replay_log)):
            raise ValueError(f"start_input_pos {start_input_pos}, must be in [0, {len(replay_log)})")
        if layer_idx < 0:
            raise ValueError(f"layer_idx {layer_idx}, must be nonnegative")
        if num_chunks <= 0:
            raise ValueError(f"num_chunks {num_chunks}, must be positive")
        if config.attention_logit_softcapping is not None:
            raise ValueError(
                "This replay cache does not support "
                "config.attention_logit_softcapping being used, since this "
                "forbids the use of fast SDPA kernels. Please choose a "
                "different model."
            )
        super().__init__(
            config=config,
            max_batch_size=batch_size,
            cache_length=cache_length,
            block_idx=layer_idx,
            dtype=None,
            **base_kwargs,
        )
        if len(replay_log) == 0:
            raise ValueError("replay_log must not be empty")
        self._device = replay_log.device
        if self.mha.use_eager_sdpa_always:
            raise ValueError("This replay cache does not support mha.use_eager_sdpa_always = True")
        self.replay_log = replay_log
        self._batch_size = batch_size
        self.kv_buffers = None
        self.layer_idx = layer_idx
        self.num_chunks = num_chunks
        self.grace_period = replay_log.grace_period
        self._next_grace_pos = None
        self._node_annotations = node_annotations
        self._input_pos = start_input_pos  # Will be validated
        self._token_chunk_pos = None
        self._start_token_chunk_pos = None
        self._end_token_chunk_pos = None
        shape = (batch_size, config.n_query_groups, cache_length)
        self._token_positions = torch.zeros(
            shape, dtype=torch.int, device=self._device,
        )
        self.current_length = 0
        self._initialize_replay()
        self._debug_tensors = debug_tensors
        # If this is set, we log all annotations being created
        self.debug_print_annotations = debug_print_annotations
        self._debug_full_args = debug_full_args
        sliding_window_size = self.mha._get_sliding_window_size(layer_idx)
        if sliding_window_size is not None:
            print(
                "WARNING: config.sliding_window_size is used. This means that "
                "a naive SDPA kernel has to be used, for which computations "
                "can be slow. Consider switching to a model which does not use "
                "config.sliding_window_size."
            )
            sdpa_mode = SDPA_IMPL_EAGER_BLOCKS
        else:
            sdpa_mode = SDPA_IMPL_QPADDED_PYTORCH
        self._sdpa_kwargs = dict(
            sdpa_mode=sdpa_mode,
            sliding_window_size=sliding_window_size,
        )
        if debug_intermediates is not None:
            self._debug_intermediates = debug_intermediates[0]
            self._debug_prefix = debug_intermediates[1]
        else:
            self._debug_intermediates = None
            self._debug_prefix = None

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    def _initialize_replay(self):
        # Initialize `_token_chunk_pos`, `_start_token_chunk_pos`,
        # `_end_token_chunk_pos`, `_input_pos`, and
        # `_next_grace_pos`.
        input_pos = self._input_pos  # target value to reach
        self._input_pos = 0
        self._token_chunk_pos = 0
        self.prefill_length = None
        do_grace_period = self.grace_period > 0
        done = False
        for chunk in self.replay_log.token_chunks:
            if self._input_pos >= input_pos:
                done = self._input_pos == input_pos
                break
            # This advances `_token_chunk_pos`
            num = chunk.shape[-1]
            self._process_token_chunk(chunk)
            self._input_pos += num
            if do_grace_period:
                prefix = self.cache_length - self.grace_period
                if self._next_grace_pos is None:
                    # First chunk: prefill
                    self._next_grace_pos = prefix
                elif self.current_length == self.cache_length and num <= self.grace_period:
                    # Forward, cache is full, and chunk not larger than grace period
                    self._next_grace_pos = (
                        self._next_grace_pos - prefix + num
                    ) % self.grace_period + prefix
            if self.current_length < self.cache_length:
                self.current_length += num
                if self.current_length > self.cache_length:
                    raise ValueError(f"current_length from {self.current_length - num} to {self.current_length}, jumps over cache_length = {self.cache_length}")
            if self.prefill_length is None:
                self.prefill_length = num
        if not done:
            raise ValueError(f"start_input_pos = {input_pos} does not map to start of a token chunk")
        self._start_token_chunk_pos = self._token_chunk_pos
        self._end_token_chunk_pos = self._token_chunk_pos + self.num_chunks
        if self._end_token_chunk_pos > len(self.replay_log.token_chunks):
            raise ValueError(f"token_chunk_pos = {self._token_chunk_pos}, num_chunks = {self.num_chunks}, sum must be <= {len(self.replay_log.token_chunks)}")

    def initialize_buffers(
        self, kv_buffers: KeysAndValues
    ):
        shape = (self.batch_size, self.n_query_groups, self.current_length, self.head_size)
        keys = kv_buffers.keys()
        if keys.shape != shape:
            raise ValueError(f"kv_buffers.keys().shape = {kv_buffers.keys().shape}, must be {shape}")
        if keys.device != self._device:
            raise ValueError(f"kv_buffers.device = {keys.device}, must be {self._device}")
        self.kv_buffers = kv_buffers

    def deallocate_buffers(self):
        if self.kv_buffers is not None:
            self.kv_buffers.clear()
            self.kv_buffers = None

    def token_positions(self) -> torch.Tensor:
        """
        Returns:
            Token positions in slots of the cache, shape
            `(batch_size, n_query_groups, T)`.where `T <= cache_length`
            is the current cache length.
        """
        # Note: The `clone` is necessary to avoid this error during backward:
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.IntTensor [5, 4, 512]] is at version 8; expected version 7 instead
        return self._token_positions[:, :, :self.current_length].clone()

    @property
    def max_forward_length(self) -> int:
        diff = self.cache_length - self.current_length
        result = self.cache_length - self.grace_period
        if diff > 0:
            result = min(result, diff)
        return result

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> torch.Tensor:
        self._forward_check_args(query, key, value, token_idx)
        if query.shape[0] != self.batch_size:
            raise ValueError(f"query.shape[0] = {query.shape[0]}, batch_size = {self.batch_size}, must be equal")
        # For prefill, we use the default implementation based on :meth:`_prefill`.
        input_pos = self._input_pos  # Value before update
        if input_pos == 0:
            attn_outputs = super().forward(query, key, value, token_idx)
        else:
            attn_outputs = self._forward_if_not_prefill(
                query, key, value, token_idx,
            )
            self._input_pos += key.shape[2]
        # DEBUG
        if self._debug_intermediates is not None:
            prefix = self._debug_prefix + f"_inputpos{input_pos}_"
            for name, x in (
                ("query", query),
                ("key", key),
                ("value", value),
                ("token_idx", token_idx),
                ("attn_outputs", attn_outputs),
            ):
                self._debug_intermediates[prefix + name] = for_debug(x)
        # END DEBUG
        return attn_outputs

    def _forward_if_not_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> torch.Tensor:
        assert self.input_pos > 0  # Not to be used for prefill
        num = key.shape[2]
        # Process token chunk. This also updates `_token_positions` and the
        # counters (including `_input_pos`)
        index, update_result = self._process_token_chunk(token_idx)
        # Cache full: Single scatter. `update_result` is given iff there is
        # a grace period.
        do_grace_period = update_result is not None
        if not do_grace_period:
            positions = None
        else:
            # All slices are the same
            positions = update_result.positions[0, 0, :]
        # DEBUG
        if self._debug_intermediates is not None:
            prefix = self._debug_prefix + f"_sdpa_inputpos{self.input_pos}"
            debug_intermediates = (self._debug_intermediates, prefix)
        else:
            debug_intermediates = None
        # END DEBUG
        do_scatter = self.current_length >= self.cache_length
        if do_scatter:
            index_e = expand_index(index, head_size=self.head_size)
            # Annotations of inputs, before new nodes are created
            self._create_node_before_creator(
                kind="scatter-key",
                index=index_e,
                positions=positions,
                debug_msg="scatter-before",
            )
            self._create_node_before_creator(
                kind="scatter-value",
                index=index_e,
                positions=positions,
                debug_msg="scatter-before",
            )
            # "scatter" update of KV cache buffers
            key_buffer_new, value_buffer_new = scatter_on_buffers(
                key,
                value,
                self.kv_buffers.keys(),
                self.kv_buffers.values(),
                index,
                positions,
            )
            # Post-processing w.r.t. annotations
            self._create_node_after_creator(
                x=key_buffer_new,
                kind="scatter-key",
                index=index_e,
                debug_msg="scatter-after",
            )
            self._create_node_after_creator(
                x=value_buffer_new,
                kind="scatter-value",
                index=index_e,
                debug_msg="scatter-after",
            )
            self.kv_buffers = DefaultKeysAndValues(
                key_buffer_new, value_buffer_new,
            )
            if update_result is not None and update_result.num1 == 0:
                # Increment in round-robin fashion (only if num <= grace_period)
                prefix = update_result.prefix
                self._next_grace_pos = (
                    self._next_grace_pos - prefix + num
                ) % self.grace_period + prefix
        else:
            assert not do_grace_period  # Sanity check
            # Annotations of inputs, before new nodes are created
            debug_msg = f"cat-before (clen={self.current_length})"
            self._create_node_before_creator(
                kind="cat-key",
                index=None,
                debug_msg=debug_msg,
            )
            self._create_node_before_creator(
                kind="cat-value",
                index=None,
                debug_msg=debug_msg,
            )
            # "cat" update of KV cache buffers
            key_buffer_new, value_buffer_new, _, _ = cat_on_buffers(
                key,
                value,
                self.kv_buffers.keys(),
                self.kv_buffers.values(),
            )
            # Post-processing w.r.t. annotations
            debug_msg = f"cat-after (clen={self.current_length})"
            self._create_node_after_creator(
                x=key_buffer_new,
                kind="cat-key",
                index=None,
                debug_msg=debug_msg,
            )
            self._create_node_after_creator(
                x=value_buffer_new,
                kind="cat-value",
                index=None,
                debug_msg=debug_msg,
            )
            self.kv_buffers = DefaultKeysAndValues(
                key_buffer_new, value_buffer_new,
            )
            self.current_length += num

        # SDPA computation
        if self._node_annotations is not None:
            self._padded_query_annotation(query)
        attn_outputs, _ = self.mha.scaled_dot_product_attention(
            query=query,
            k_and_v=self.kv_buffers,
            input_pos=self.input_pos,
            token_positions=self.token_positions(),
            return_attn_weights=False,
            **self._sdpa_kwargs,
        )
        return attn_outputs.reshape(self.batch_size, num, -1)

    def _padded_query_annotation(self, query: torch.Tensor):
        if self._sdpa_kwargs["sdpa_mode"] == SDPA_IMPL_QPADDED_PYTORCH:
            # Create annotation for padded query
            _, _, q_len, head_size = query.shape
            kv_len = self.kv_buffers.keys().shape[2]
            shape = tuple(query.shape[:2]) + (kv_len, head_size)
            # Pick a delta which includes zeros and non-zeros
            delta_len = min(MAX_DELTA_TRANS_LENGTH, kv_len - q_len)
            num_nonzeros = min(delta_len - 2, q_len)
            num_zeros = delta_len - num_nonzeros
            delta = torch.cat(
                (
                    torch.zeros(
                        (1, 1, 1, 1), dtype=query.dtype, device=query.device,
                    ).expand(*shape[:2], num_zeros, head_size),
                    query[:, :, :num_nonzeros, :],
                ),
                dim=2,
            )
            start = kv_len - q_len - num_zeros
            end = start + delta_len
            index = torch.arange(
                start, end, dtype=torch.int32, device=query.device,
            ).view(1, 1, -1, 1).expand(*shape[:2], -1, head_size)
            self._append_annotation(
                NodeAnnotation(
                    kind="padded-query",
                    layer_idx=self.layer_idx,
                    chunk_idx=self._token_chunk_pos - 1,  # `token_chunk_pos` has already been advanced
                    shape=shape,
                    index=index,
                    delta=delta,
                    extra_info={"q_len": q_len},
                )
            )

    def _append_annotation(self, annotation: NodeAnnotation):
        self._node_annotations.append_safe(annotation)
        if self.debug_print_annotations:
            print(f"Create {str(annotation)}")

    def _random_index(self, num: int) -> torch.Tensor:
        shape = (self.batch_size, self.n_query_groups, num, self.head_size)
        return create_random_index(
            shape=shape,
            length=self.current_length,
            device=self._device,
            dtype=torch.int32,
        )

    def _append_random_index(self, index: torch.Tensor) -> torch.Tensor:
        num = MAX_DELTA_TRANS_LENGTH - index.shape[2]
        if num > 0:
            index2 = self._random_index(num)
            index = torch.cat((index, index2), dim=2)
        return index

    def _index_for_before(
        self,
        kind: str,
        index: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        if NodeAnnotation.kind_is_scatter(kind):
            assert index is not None
            assert index.ndim == 4  # Sanity check
            index = index.detach()
            # If index is too small, we extend it by random entries. This
            # lowers the probability of false matches
            extra_info = {"index_len": index.shape[2]}
            index = self._append_random_index(index)
        else:
            index = self._random_index(MAX_DELTA_TRANS_LENGTH)
            extra_info = None
        return index, extra_info

    def _create_node_before_creator(
        self,
        kind: str,
        index: Optional[torch.Tensor],
        positions: Optional[torch.Tensor] = None,
        debug_msg: Optional[str] = None,
    ):
        chunk_idx = self._token_chunk_pos - 1  # Counter has already been advanced
        # If this is the first cell and `chunk_idx == 1`, we do not have to create
        # an annotation, because the corresponding "ext" annotation does not need
        # to be supported.
        thres_idx = max(self._start_token_chunk_pos, 1)
        if self._node_annotations is not None and chunk_idx > thres_idx:
            # This annotation is for `x` before the node is created, so the
            # chunk index is `chunk_idx - 1`.
            is_keys = NodeAnnotation.kind_is_keys(kind)
            x = self.kv_buffers.keys() if is_keys else self.kv_buffers.values()
            x = x.detach()[:, :, :self.current_length, :]
            index, extra_info = self._index_for_before(kind, index)
            delta = x.gather(2, index)
            if positions is not None:
                positions = positions.detach().to(self._device)
            annotation = NodeAnnotation(
                kind=kind,
                layer_idx=self.layer_idx,
                chunk_idx=chunk_idx - 1,
                shape=shape_to_tuple(x),
                index=index,
                delta=delta,
                positions=positions,
                extra_info=extra_info,
                debug_msg=debug_msg,
            )
            if self._debug_full_args:
                annotation = replace(
                    annotation,
                    debug_full_arg=x.clone(),
                )
            self._append_annotation(annotation)

    @staticmethod
    def _transform_index(
        index: torch.Tensor, sort_index: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_query_groups, num, head_size = index.shape
        si_len = sort_index.shape[-1]
        assert sort_index.shape == (batch_size, n_query_groups, si_len)
        sort_index = sort_index.to(dtype=index.dtype)
        index = index[:, :, :, 0]
        result = torch.empty_like(sort_index).scatter_(
            2,
            sort_index,
            torch.arange(
                si_len, dtype=index.dtype, device=index.device,
            ).view(1, 1, -1).expand(batch_size, n_query_groups, -1)
        ).gather(2, index)
        return expand_index(result, head_size)

    def _create_node_after_creator(
        self,
        x: torch.Tensor,
        kind: str,
        index: Optional[torch.Tensor],
        debug_msg: Optional[str] = None,
    ):
        chunk_idx = self._token_chunk_pos - 1  # Counter has already been advanced
        x = x.detach()
        if self._node_annotations is not None:
            if chunk_idx == self._end_token_chunk_pos - 1:
                # We need to store the final node in order to start the
                # reconstruction of all earlier ones.
                self._node_annotations.set_final(
                    x=x,
                    layer_idx=self.layer_idx,
                    chunk_idx=chunk_idx,
                    kind=kind,
                )
            if NodeAnnotation.kind_is_scatter(kind):
                assert index is not None
                # Create "ext-*" annotation for node `x` after it was created, so
                # the chunk index is `chunk_idx`
                # Create index
                need_ri = need_repeat_interleave(self.n_head, self.n_query_groups)
                is_keys = NodeAnnotation.kind_is_keys(kind)
                ext_kind = "ext-key" if is_keys else "ext-value"
                # Used for reordering in padded-query SDPA
                sort_index = torch.argsort(
                    self.token_positions().detach(), dim=-1,
                ).to(dtype=torch.int32)
                extra_info = {"sort_index": sort_index}
                # `delta_index` is equal to initial slices of `index`.
                # `ext_index` must be such that if
                # `delta = gather(x, delta_index)` and
                # `x1 = gather(x, sort_index)`, then
                # `delta == gather(x1, ext_index)`. This means that
                # `ext_index` can be used to extract `delta` from the pack
                # argument, which is transformed by `sort_index`.
                ext_index = self._append_random_index(index)[:, :, :MAX_DELTA_TRANS_LENGTH, :]
                delta = x.gather(2, ext_index)
                ext_index = repeat_interleave(
                    self._transform_index(
                        index=ext_index, sort_index=sort_index,
                    ),
                    self.n_head,
                )
                delta = repeat_interleave(delta, self.n_head)
                shape = shape_to_tuple(x)
                if need_ri:
                    shape = (shape[0], self.n_head) + shape[2:]
                annotation = NodeAnnotation(
                    kind=ext_kind,
                    layer_idx=self.layer_idx,
                    chunk_idx=chunk_idx,
                    shape=shape,
                    index=ext_index,
                    delta=delta,
                    extra_info=extra_info,
                    debug_msg=debug_msg,
                )
                if self._debug_full_args:
                    sort_index = extra_info["sort_index"]
                    x = x.gather(2, expand_index(sort_index, self.head_size))
                    if need_ri:
                        x = repeat_interleave(x, self.n_head)
                    annotation = replace(
                        annotation,
                        debug_full_arg=x,
                    )
                self._append_annotation(annotation)

        # DEBUG:
        if self._debug_tensors is not None:
            name = f"c{self._token_chunk_pos - 1}-l{self.layer_idx}-{kind}"
            if name in self._debug_tensors:
                raise IndexError(f"Entry {name} already in debug_tensor!")
            self._debug_tensors[name] = x.clone()

    @property
    def max_prefill_length(self) -> int:
        return self.replay_log.max_prefill_length

    def _prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        if self.input_pos != 0:
            raise IndexError(f"input_pos = {self.input_pos}, can only be called if this is 0")
        if key.shape[0] != self.batch_size:
            raise ValueError(f"key.shape = {key.shape}, first dimension must be batch_size = {self.batch_size}")
        # Sanity checks
        assert self.current_length == 0
        assert self._token_chunk_pos == 0
        assert key.shape == value.shape
        init_length = token_idx.shape[-1]
        assert key.shape[2] == init_length
        self._process_token_chunk(token_idx)
        self.kv_buffers = DefaultKeysAndValues(keys=key, values=value)
        self.current_length = init_length
        if self.grace_period > 0:
            # First slot to move out once the cache is full
            self._next_grace_pos = self.cache_length - self.grace_period
        # DEBUG:
        if self._debug_tensors is not None:
            prefix = f"c0-l{self.layer_idx}-prefill-"
            for name, node in [
                (prefix + "key", key),
                (prefix + "value", value),
            ]:
                if name in self._debug_tensors:
                    raise IndexError(f"Entry {name} already in debug_tensor!")
                self._debug_tensors[name] = node.detach().clone()

    def _process_token_chunk(
        self, token_idx: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[UpdateTokenPositionsGracePeriod]]:
        """
         Processes token chunk, comparing `token_idx` against the current chunk
        in the replay log, and advancing `token_chunk_pos`. We also update
        `_token_positions`.

        `_input_pos` is not modified here.

        Args:
            token_idx: Token chunk to process

        Returns:
            (index, update_result), to be used in `_forward`. Either can be
            `None`.

        """
        other = self.replay_log.token_chunks[self._token_chunk_pos]
        if not token_idx.equal(other):
            raise ValueError(f"token_idx:\n{token_idx}\nreplay_log.token_chunks[{self._token_chunk_pos}]:\n{other}\nShould be the same!")
        self._token_chunk_pos += 1
        num = token_idx.shape[-1]
        index = None
        update_result = None
        if self.current_length >= self.cache_length:
            # scatter case
            index_kwargs = {"dtype": torch.int32, "device": self._device}
            index = self.replay_log.extract_index(
                self.input_pos, num, **index_kwargs
            ).detach()
            update_result = update_token_positions(
                token_positions=self._token_positions,
                input_pos=self.input_pos,
                num=num,
                batch_size=self.batch_size,
                index=index,
                grace_period=self.grace_period,
                next_grace_pos=self._next_grace_pos,
            )
        return index, update_result

    def _reset(self):
        raise NotImplementedError("This type of cache is not meant to be reset")
