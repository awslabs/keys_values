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
from dataclasses import replace
from functools import partial
from typing import Optional, Tuple, Dict, Any

import torch

from keys_values.config import Config

from keys_values.attention.base import (
    MultiHeadSelfAttention,
    KeysAndValues,
    DefaultKeysAndValues,
)
from keys_values.attention.sdpa_op import (
    scatter_on_buffers,
    cat_on_buffers,
)
from keys_values.attention.sdpa_wrapper import ReorderAnnotationCallback
from keys_values.kvcache.attn_weights import (
    update_token_positions,
    UpdateTokenPositionsGracePeriod,
)
from keys_values.kvcache.gradient.autograd_hooks import Annotations
from keys_values.kvcache.gradient.annotation import (
    NodeAnnotation,
    create_random_index,
    MAX_DELTA_TRANS_LENGTH,
    create_ext_annotations,
)
from keys_values.kvcache.new.basics import PositionsNewEntriesType
from keys_values.kvcache.new.driver import DefaultKVCacheDriver
from keys_values.kvcache.test_shadow import (
    KVCacheReplayLog_oldornew,
    use_new_replay_log,
    get_chunks_from_log,
)
from keys_values.tools.debug_utils import for_debug
from keys_values.utils import (
    expand_index,
    shape_to_tuple,
    is_index_1d,
)


class TrainingReplayKVCacheDriver(DefaultKVCacheDriver):
    """
    Specialized driver for training replay. Instead of using buffers which
    are overwritten, the subsequent buffer states are nodes in a computation
    graph. The main task is to create all necessary annotations for
    :class:`CellComputationAutogradHooks` to work.
    """

    def __init__(
        self,
        config: Config,
        batch_size: int,
        cache_length: int,
        block_idx: int,
        replay_log: KVCacheReplayLog_oldornew,
        start_input_pos: int,
        layer_idx: int,
        num_chunks: int,
        node_annotations: Optional[Annotations] = None,
        dtype: Optional[torch.dtype] = None,
        head_size: Optional[int] = None,
        mha: Optional[MultiHeadSelfAttention] = None,
        debug_tensors: Optional[Dict[str, torch.Tensor]] = None,
        debug_print_annotations: bool = False,
        debug_full_args: bool = False,
        debug_intermediates: Optional[Tuple[Dict[str, torch.Tensor], str]] = None,
        **mha_kwargs,
    ):
        if not (0 <= start_input_pos < len(replay_log)):
            raise ValueError(
                f"start_input_pos {start_input_pos}, must be in [0, {len(replay_log)})"
            )
        if layer_idx < 0:
            raise ValueError(f"layer_idx {layer_idx}, must be nonnegative")
        if num_chunks <= 0:
            raise ValueError(f"num_chunks {num_chunks}, must be positive")
        super().__init__(
            config,
            batch_size,
            cache_length,
            block_idx,
            dtype,
            head_size,
            mha,
            **mha_kwargs,
        )

        self._use_new_code = use_new_replay_log(replay_log)
        if self.mha.use_eager_sdpa_always:
            raise ValueError(
                "This replay cache does not support mha.use_eager_sdpa_always = True"
            )
        if (
            config.attention_logit_softcapping is not None
            and not self.mha.has_flex_attention
        ):
            raise ValueError(
                "This replay cache does not support "
                "config.attention_logit_softcapping being used, unless together "
                "with FlexAttention. Choose a model without attention logit "
                "softcapping, or use FlexAttention by passing flexatt_args "
                "when creating the MultiHeadSelfAttention object."
            )
        self.token_chunks = get_chunks_from_log(replay_log)
        self.kv_buffers = None
        self.layer_idx = layer_idx
        self.num_chunks = num_chunks
        self._node_annotations = node_annotations
        self.input_pos = start_input_pos  # Will be validated
        self._token_chunk_pos = None
        self._start_token_chunk_pos = None
        self._end_token_chunk_pos = None
        self._current_length = 0
        self._annotation_callback = None
        self._initialize_replay()
        # For debugging
        # TODO: Is this still needed?
        self._debug_tensors = debug_tensors
        self.debug_print_annotations = debug_print_annotations
        self._debug_full_args = debug_full_args
        if debug_intermediates is not None:
            self._debug_intermediates = debug_intermediates[0]
            self._debug_prefix = debug_intermediates[1]
        else:
            self._debug_intermediates = None
            self._debug_prefix = None

    @property
    def current_length(self) -> int:
        return self._current_length

    @property
    def batch_size(self) -> Optional[int]:
        return self.max_batch_size

    # TODO: Which members here are really needed?
    # token_positions is maintained outside!
    def _initialize_replay(self):
        # Initialize `_token_chunk_pos`, `_start_token_chunk_pos`,
        # `_end_token_chunk_pos`, `input_pos`, and
        input_pos = self.input_pos  # target value to reach
        self.input_pos = 0
        self._token_chunk_pos = 0
        self.prefill_length = None
        do_grace_period = self.grace_period > 0
        done = False
        for chunk in self.token_chunks:
            if self.input_pos >= input_pos:
                done = self.input_pos == input_pos
                break
            # This advances `_token_chunk_pos`
            num = chunk.shape[-1]
            self._process_token_chunk(chunk)
            self.input_pos += num
            if self.current_length < self.cache_length:
                self._current_length += num
                if self.current_length > self.cache_length:
                    raise ValueError(
                        f"current_length from {self.current_length - num} to {self.current_length}, jumps over cache_length = {self.cache_length}"
                    )
            if self.prefill_length is None:
                self.prefill_length = num
        if not done:
            raise ValueError(
                f"start_input_pos = {input_pos} does not map to start of a token chunk"
            )
        self._start_token_chunk_pos = self._token_chunk_pos
        self._end_token_chunk_pos = self._token_chunk_pos + self.num_chunks
        if self._end_token_chunk_pos > len(self.token_chunks):
            raise ValueError(
                f"token_chunk_pos = {self._token_chunk_pos}, num_chunks = {self.num_chunks}, sum must be <= {len(self.token_chunks)}"
            )

    def initialize_buffers(self, kv_buffers: KeysAndValues):
        shape = (
            self.batch_size,
            self.n_query_groups,
            self.current_length,
            self.head_size,
        )
        keys = kv_buffers.keys()
        if keys.shape != shape:
            raise ValueError(
                f"kv_buffers.keys().shape = {kv_buffers.keys().shape}, must be {shape}"
            )
        if keys.device != self._device:
            raise ValueError(
                f"kv_buffers.device = {keys.device}, must be {self._device}"
            )
        self.kv_buffers = kv_buffers

    def deallocate_buffers(self):
        if self.kv_buffers is not None:
            self.kv_buffers.clear()
            self.kv_buffers = None

    def extra_mha_kwargs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        input_pos: int,
        positions: Optional[PositionsNewEntriesType],
        token_positions: Optional[torch.Tensor],
        return_attn_weights,
    ) -> Dict[str, Any]:
        """
        This is called just before `self.mha(...)` in the superclass
        :meth:`forward` method. Two things are done:

        - If `self._annotation_callback`, this is returned here, so passed as
          extra argument to `self.mha(...)`
        - Calling `self._padded_query_annotation(query)`

        """
        if self._node_annotations is not None:
            self._padded_query_annotation(query)
        if self._annotation_callback is not None:
            result = {"annotation_callback" : self._annotation_callback}
            # Reset: To be used only once
            self._annotation_callback = None
        else:
            result = dict()
        return result

    def _append_debug_intermediates(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        attn_outputs: torch.Tensor,
    ):
        if self._debug_intermediates is not None:
            prefix = self._debug_prefix + f"_inputpos{self.input_pos}_"
            for name, x in (
                    ("query", query),
                    ("key", key),
                    ("value", value),
                    ("token_idx", token_idx),
                    ("attn_outputs", attn_outputs),
            ):
                self._debug_intermediates[prefix + name] = for_debug(x)

    def prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> torch.Tensor:
        if query.shape[0] != self.batch_size:
            raise ValueError(
                f"query.shape[0] = {query.shape[0]}, batch_size = {self.batch_size}, must be equal"
            )
        self._process_token_chunk(token_idx)
        attn_outputs = super().prefill(query, key, value, token_idx)
        self._append_debug_intermediates(query, key, value, token_idx, attn_outputs)
        return attn_outputs

    def _prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        if self.input_pos != 0:
            raise IndexError(
                f"input_pos = {self.input_pos}, can only be called if this is 0"
            )
        # Sanity checks
        assert self.current_length == 0
        assert self._token_chunk_pos == 0
        assert key.shape == value.shape
        init_length = token_idx.shape[-1]
        assert key.shape[2] == init_length
        self.kv_buffers = DefaultKeysAndValues(keys=key, values=value)
        self._current_length = init_length
        if self._debug_tensors is not None:
            prefix = f"c0-l{self.layer_idx}-prefill-"
            for name, node in [
                (prefix + "key", key),
                (prefix + "value", value),
            ]:
                if name in self._debug_tensors:
                    raise IndexError(f"Entry {name} already in debug_tensor!")
                self._debug_tensors[name] = node.detach().clone()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        positions: PositionsNewEntriesType,
        token_positions: torch.Tensor,
        input_pos: int,
        return_attn_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if query.shape[0] != self.batch_size:
            raise ValueError(
                f"query.shape[0] = {query.shape[0]}, batch_size = {self.batch_size}, must be equal"
            )
        if input_pos != self.input_pos:
            raise ValueError(f"Internal error: input_pos = {self.input_pos} internal, but {input_pos} as argument")
        self._process_token_chunk(token_idx)
        attn_outputs, attn_weights = super().forward(
            query, key, value, token_idx, positions, token_positions, input_pos, return_attn_weights
        )
        self._append_debug_intermediates(query, key, value, token_idx, attn_outputs)
        return attn_outputs, attn_weights

    # Needs to be rewritten, to be based on `positions`. Also cater for
    # special cases!
    def _forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: PositionsNewEntriesType,
    ) -> KeysAndValues:
        assert self.input_pos > 0  # Not to be used for prefill
        num = key.shape[2]
        # Cache full: Single scatter. `update_result` is given iff there is
        # a grace period.
        do_grace_period = update_result is not None
        if not do_grace_period:
            positions = None
        else:
            # All slices are the same
            positions = update_result.positions[0, 0, :]
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
            self._annotation_callback = self._create_callback_after_creator(
                key=key_buffer_new,
                value=value_buffer_new,
                kind="scatter",
            )
            self.kv_buffers = DefaultKeysAndValues(
                key_buffer_new,
                value_buffer_new,
            )
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
            self._annotation_callback = self._create_callback_after_creator(
                key=key_buffer_new,
                value=value_buffer_new,
                kind="cat",
            )
            self.kv_buffers = DefaultKeysAndValues(
                key_buffer_new,
                value_buffer_new,
            )
            self._current_length += num

        return self.kv_buffers

    def _padded_query_annotation(self, query: torch.Tensor):
        sliding_window_size = self.mha._get_sliding_window_size(self.layer_idx)
        if not self.mha.has_flex_attention and sliding_window_size is None:
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
                        (1, 1, 1, 1),
                        dtype=query.dtype,
                        device=query.device,
                    ).expand(*shape[:2], num_zeros, head_size),
                    query[:, :, :num_nonzeros, :],
                ),
                dim=2,
            )
            start = kv_len - q_len - num_zeros
            end = start + delta_len
            index = (
                torch.arange(
                    start,
                    end,
                    dtype=torch.int32,
                    device=query.device,
                )
                .view(1, 1, -1, 1)
                .expand(*shape[:2], -1, head_size)
            )
            self._append_annotation(
                NodeAnnotation(
                    kind="padded-query",
                    layer_idx=self.layer_idx,
                    chunk_idx=self._token_chunk_pos - 1,
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
        shape = (
            self.batch_size,
            self.n_query_groups,
            min(num, self.current_length),
            self.head_size,
        )
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
            x = x.detach()[:, :, : self.current_length, :]
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

    def _create_callback_after_creator(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kind: str,
    ) -> Optional[ReorderAnnotationCallback]:
        chunk_idx = self._token_chunk_pos - 1  # Counter has already been advanced
        annotation_callback = None
        if self._node_annotations is not None:
            if chunk_idx == self._end_token_chunk_pos - 1:
                # We need to store the final node in order to start the
                # reconstruction of all earlier ones.
                self._node_annotations.set_final(
                    x=key.detach(),
                    layer_idx=self.layer_idx,
                    chunk_idx=chunk_idx,
                    kind=kind + "-key",
                )
                self._node_annotations.set_final(
                    x=value.detach(),
                    layer_idx=self.layer_idx,
                    chunk_idx=chunk_idx,
                    kind=kind + "-value",
                )
            annot_kwargs = dict(
                layer_idx=self.layer_idx,
                chunk_idx=chunk_idx,
            )
            annotation_callback = partial(
                create_ext_annotations,
                node_annotations=self._node_annotations.nodes,
                annot_kwargs=annot_kwargs,
                verbose=self.debug_print_annotations,
            )
        if self._debug_tensors is not None:
            name = f"c{self._token_chunk_pos - 1}-l{self.layer_idx}-{kind}"
            if name in self._debug_tensors:
                raise IndexError(f"Entry {name} already in debug_tensor!")
            self._debug_tensors[name + "-key"] = key.clone()
            self._debug_tensors[name + "-value"] = value.clone()
        return annotation_callback

    # TODO: Is this needed?
    def _process_token_chunk(
        self,
        token_idx: torch.Tensor,
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
        other = self.token_chunks[self._token_chunk_pos]
        if not token_idx.equal(other):
            raise ValueError(
                f"token_idx:\n{token_idx}\nreplay_log.token_chunks[{self._token_chunk_pos}]:\n{other}\nShould be the same!"
            )
        self._token_chunk_pos += 1
        num = token_idx.shape[-1]
        index = None
        update_result = None
        if self.current_length >= self.cache_length:
            # scatter case
            index = self.replay_log.extract_index(
                self.input_pos,
                num,
                dtype=torch.int32,
                device=self._device,
            ).detach()
            if is_index_1d(self._token_positions) and not is_index_1d(index):
                self._token_positions = self._token_positions.contiguous()
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
