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
    KVCacheCatUpdateAndSDPAFunction,
    KVCacheScatterUpdateAndSDPAFunction,
)
from keys_values.kvcache.utils import shape_to_tuple
from keys_values.utils import expand_index


class TrainingAttnWeightsReplayCache(DefaultKVCache):
    """
    Replay cache corresponding to :class:`AttnWeightsKVCache`, to be used in
    training mode.

    A replay cache is used in order to compute gradients by way of activation
    checkpointing. We first process the batch of input sequences in inference
    mode, store checkpoints along the way and retrieve the replay log in the end.

    We then run forward-backward between checkpoints. For the forward pass in
    training mode, we run the same processing, but using replay caches
    (configured by replay logs) in place of the KV caches. We start the process
    from a particular token position and KV cache checkpoint.

    Importantly, this KV cache is using the :class:`KVCacheScatterUpdateAndSDPAFunction`
    or :class:`KVCacheCatUpdateAndSDPAFunction` operators to implement
    :meth:`forward`. This keeps GPU memory usage to the minimum and allows
    control when supporting packing.

    Support of autograd saved tensors hooks:

    A number of tensor nodes in the autograd computation graph are large, but
    can be obtained from others by way of `scatter` or `cat`, so the argument
    we really have to store is much smaller. We support this by packing and
    unpacking these saved tensors, see:

    https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html

    The tensor nodes in question receive :class:`NodeAnnotation` annotations.
    These are communicated to the pack hook by appending the annotation to the
    `node_annotations` list. The pack hook pops the annotation from the list.
    Its return value contains all information necessary for the unpack hook to
    compute the full-sized value. If the node-creating operation is
    `x_new = f(x, index, delta)`, the annotation is for reconstructing `x` (not
    `x_new`).

    Note: Why do we need :class:`TrainingAttnWeightsReplayCache` and
    :class:`InferenceAttnWeightsReplayCache`, why not just one of them? The
    former is used to run forward in inference mode along a row of cells to
    compute quantized KV cache checkpoints. This second forward inference pass
    is needed because we cannot store all KV cache checkpoints. It is a simple
    variant of :class:`AttnWeightsReplayCache` without any scoring. The training
    replay cache here is different. No quantization is done, and no memory is
    reused, so that forward-backward works robustly. We also do not maintain any
    buffers here, the whole purpose is to build a computation graph.

    In principle, we could use :class:`TrainingAttnWeightsReplayCache` for
    computing KV cache checkpoints, but this would need quite a bit more memory,
    since no quantization or dequantization are done. Since
    :class:`InferenceAttnWeightsReplayCache` is easily obtained from
    :class:`AttnWeightsReplayCache`, this is the simpler option.

    """
    def __init__(
        self,
        config: Config,
        batch_size: int,
        cache_length: int,
        replay_log: KVCacheReplayLog,
        start_token_pos: int,
        layer_idx: int,
        num_chunks: int,
        device: torch.device,
        node_annotations: Optional[Annotations] = None,
        debug_tensors: Optional[Dict[str, torch.Tensor]] = None,
        debug_print_annotations: bool = False,
        debug_full_args: bool = False,
        **base_kwargs,
    ):
        if not (0 <= start_token_pos < len(replay_log)):
            raise ValueError(f"start_token_pos {start_token_pos}, must be in [0, {len(replay_log)})")
        if layer_idx < 0:
            raise ValueError(f"layer_idx {layer_idx}, must be nonnegative")
        if num_chunks <= 0:
            raise ValueError(f"num_chunks {num_chunks}, must be positive")
        super().__init__(
            config=config,
            max_batch_size=batch_size,
            cache_length=cache_length,
            block_idx=layer_idx,
            dtype=None,
            **base_kwargs,
        )
        self.replay_log = replay_log
        self._device = device
        self._batch_size = batch_size
        self.kv_buffers = None
        self.layer_idx = layer_idx
        self.num_chunks = num_chunks
        self.grace_period = replay_log.grace_period
        self._next_grace_pos = None
        self._node_annotations = node_annotations
        self._next_token_pos = start_token_pos  # Will be validated
        self._token_chunk_pos = None
        self._start_token_chunk_pos = None
        self._end_token_chunk_pos = None
        shape = (batch_size, config.n_query_groups, cache_length)
        self._token_positions = torch.zeros(
            shape, dtype=torch.int, device=device,
        )
        self.current_length = 0
        self._initialize_replay()
        self._debug_tensors = debug_tensors
        # If this is set, we log all annotations being created
        self.debug_print_annotations = debug_print_annotations
        self._debug_full_args = debug_full_args

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def _initialize_replay(self):
        # Initialize `_token_chunk_pos`, `_next_token_pos`, and
        # `_next_grace_pos`.
        next_token_pos = self._next_token_pos  # target value to reach
        self._next_token_pos = 0
        self._token_chunk_pos = 0
        self.prefill_length = None
        do_grace_period = self.grace_period > 0
        done = False
        for chunk in self.replay_log.token_chunks:
            if self._next_token_pos >= next_token_pos:
                done = self._next_token_pos == next_token_pos
                break
            self._process_token_chunk(chunk)
            num = chunk.shape[-1]
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
            raise ValueError(f"start_token_pos = {next_token_pos} does not map to start of a token chunk")
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
        # Note: The `clone` is necessary to avoid this error during backward,
        # using the :class:`KVCacheUpdateAndSDPAFunction` or
        # :class:`SDPAFunction` operator:
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.IntTensor [5, 4, 512]] is at version 8; expected version 7 instead
        return self._token_positions[:, :, :self.current_length].clone()

    @property
    def next_token_pos(self) -> Optional[int]:
        return self._next_token_pos

    @property
    def max_tokens_forward(self) -> int:
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
        input_pos: int,
    ) -> torch.Tensor:
        self._forward_check_args(query, key, value, token_idx, input_pos)
        if query.shape[0] != self.batch_size:
            raise ValueError(f"query.shape[0] = {query.shape[0]}, batch_size = {self.batch_size}, must be equal")
        # For prefill, we use the default implementation based on :meth:`_prefill`.
        # If `input_pos > 0`, we use our own special-purpose operators.
        if input_pos == 0:
            return super().forward(query, key, value, token_idx, input_pos)
        else:
            return self._forward_if_not_prefill(
                query, key, value, token_idx, input_pos,
            )

    def _forward_if_not_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        input_pos: int,
    ) -> torch.Tensor:
        """
        If the cache is full, we use the fused operator
        :class:`KVCacheScatterUpdateAndSDPAFunction`, otherwise we use the operator
        :class:`KVCacheCatUpdateAndSDPAFunction`. These minimize GPU memory
        requirements during training.

        For prefill, we use the superclass method. This is because our fused
        operators have a backward implementation which reserves an amount of GPU
        memory which is large for prefill. In this case, it is better to call
        the SDPA operators by PyTorch.

        """
        assert input_pos > 0  # Not to be used for prefill
        num = key.shape[2]
        # Process token chunk. This also updates `_token_positions` and the
        # counters
        index, update_result = self._process_token_chunk(token_idx)
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
            # "scatter" update of KV cache buffers and MHA computation, via
            # special operator
            attn_output, key_buffer_new, value_buffer_new = KVCacheScatterUpdateAndSDPAFunction.apply(
                query,
                key,
                value,
                self.kv_buffers.keys(),
                self.kv_buffers.values(),
                index,
                self.token_positions(),
                input_pos,
                self.mha._get_scale_factor(),
                positions,
                self.mha._get_sliding_window_size(self.block_idx),
                self.mha.sdpa_kernels,
                self.mha.tmp_array_limit_gb_value(),
            )
            # Post-processing w.r.t. annotations
            self._create_node_after_creator(
                x=key_buffer_new,
                kind="scatter-key",
            )
            self._create_node_after_creator(
                x=value_buffer_new,
                kind="scatter-value",
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
            # "cat" update of KV cache buffers and MHA computation, via
            # special operator
            attn_output, key_buffer_new, value_buffer_new = KVCacheCatUpdateAndSDPAFunction.apply(
                query,
                key,
                value,
                self.kv_buffers.keys(),
                self.kv_buffers.values(),
                self.mha._get_scale_factor(),
                self.mha._get_sliding_window_size(self.block_idx),
                self.mha.sdpa_kernels,
                self.mha.tmp_array_limit_gb_value(),
            )
            # Post-processing w.r.t. annotations
            debug_msg = f"cat-after (clen={self.current_length})"
            self._create_node_after_creator(
                x=key_buffer_new,
                kind="cat-key",
            )
            self._create_node_after_creator(
                x=value_buffer_new,
                kind="cat-value",
            )
            # Update buffers
            self.kv_buffers = DefaultKeysAndValues(
                key_buffer_new, value_buffer_new,
            )
            self.current_length += num

        return attn_output.transpose(1, 2).reshape(self.batch_size, num, -1)

    def _append_annotation(self, annotation: NodeAnnotation):
        self._node_annotations.append_safe(annotation)
        if self.debug_print_annotations:
            print(f"Create {str(annotation)}")

    def _random_index(
        self, num: int, device: torch.device,
    ) -> torch.Tensor:
        shape = (self.batch_size, self.n_query_groups, num, self.head_size)
        return create_random_index(
            shape=shape,
            length=self.current_length,
            device=device,
            dtype=torch.int32,
        )

    def _append_random_index(self, index: torch.Tensor) -> torch.Tensor:
        index_len = index.shape[2]
        if index_len < MAX_DELTA_TRANS_LENGTH:
            index2 = self._random_index(
                num=MAX_DELTA_TRANS_LENGTH - index_len,
                device=index.device,
            )
            index = torch.cat((index, index2), dim=2)
        return index

    def _index_for_before(
        self,
        kind: str,
        index: Optional[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        if NodeAnnotation.kind_is_scatter(kind):
            assert index is not None
            assert index.ndim == 4  # Sanity check
            index = index.detach().to(device)
            # If index is too small, we extend it by random entries. This
            # lowers the probability of false matches
            extra_info = {"index_len": index.shape[2]}
            index = self._append_random_index(index)
        else:
            index = self._random_index(
                num=MAX_DELTA_TRANS_LENGTH,
                device=device,
            )
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
        if self._node_annotations is not None and chunk_idx > self._start_token_chunk_pos:
            is_keys = NodeAnnotation.kind_is_keys(kind)
            x = self.kv_buffers.keys() if is_keys else self.kv_buffers.values()
            x = x.detach()[:, :, :self.current_length, :]
            index, extra_info = self._index_for_before(
                kind=kind,
                index=index,
                device=x.device,
            )
            delta = x.gather(2, index)
            if positions is not None:
                positions = positions.detach().to(x.device)
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

    def _create_node_after_creator(
        self,
        x: torch.Tensor,
        kind: str,
    ):
        chunk_idx = self._token_chunk_pos - 1  # Counter has already been advanced
        x = x.detach()
        if self._node_annotations is not None and chunk_idx == self._end_token_chunk_pos - 1:
            # We need to store the final node in order to start the
            # reconstruction of all earlier ones
            self._node_annotations.set_final(
                x=x,
                layer_idx=self.layer_idx,
                chunk_idx=chunk_idx,
                kind=kind,
            )

        # DEBUG:
        if self._debug_tensors is not None:
            name = f"c{self._token_chunk_pos - 1}-l{self.layer_idx}-{kind}"
            if name in self._debug_tensors:
                raise IndexError(f"Entry {name} already in debug_tensor!")
            self._debug_tensors[name] = x.clone()

    def _update(self, *args, **kwargs):
        pass

    def update_requires_attn_weights(self) -> bool:
        return False

    @property
    def max_prefill_length(self) -> Optional[int]:
        return self.replay_log.max_prefill_length

    def _prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        if self._next_token_pos != 0:
            raise IndexError(f"token_pos = {self._next_token_pos}, can only be called if this is 0")
        if key.shape[0] != self.batch_size:
            raise ValueError(f"key.shape = {key.shape}, first dimension must be batch_size = {self.batch_size}")
        # Sanity checks
        assert self.current_length == 0
        assert self._token_chunk_pos == 0
        assert key.shape == value.shape
        init_length = token_idx.shape[-1]
        assert key.shape[2] == init_length
        assert self.max_prefill_length is None or init_length <= self.max_prefill_length  # Sanity check
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
        in the replay log, and advancing `token_chunk_pos`. We also increase
        `_next_token_pos` and `_token_positions`.

        Args:
            token_idx: Token chunk to process

        Returns:
            (index, update_result), to be used in `_forward`. Either can be
            `None`.

        """
        other = self.replay_log.token_chunks[self._token_chunk_pos]
        if not token_idx.to(device=other.device).equal(other):
            raise ValueError(f"token_idx:\n{token_idx}\nreplay_log.token_chunks[{self._token_chunk_pos}]:\n{other}\nShould be the same!")
        self._token_chunk_pos += 1
        num = token_idx.shape[-1]
        index = None
        update_result = None
        if self.current_length >= self.cache_length:
            # scatter case
            index_kwargs = {"dtype": torch.int32, "device": self._device}
            index = self.replay_log.extract_index(
                self._next_token_pos, num, **index_kwargs
            ).detach()
            update_result = update_token_positions(
                token_positions=self._token_positions,
                next_token_pos=self._next_token_pos,
                num=num,
                batch_size=self.batch_size,
                index=index,
                grace_period=self.grace_period,
                next_grace_pos=self._next_grace_pos,
            )
        self._next_token_pos += num
        return index, update_result
