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
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass

import torch

from litgpt.config import Config

from keys_values.attention import KeysAndValues
from keys_values.kvcache.base import (
    KVCacheParams,
    DefaultKVCacheReplayLog,
    KVCacheReplayLog,
)
from keys_values.kvcache.basics import KVCacheWithBuffers
from keys_values.kvcache.buffers import (
    DefaultKVCacheBuffers,
    KVCacheBuffers,
    KVCacheBuffersParams,
    positions_wrap_around,
    PositionsType,
)
from keys_values.kvcache.utils import bitsize_of, bits_for_torch_dtype
from keys_values.utils import index_to_3d


class AttnWeightsReplayLog(DefaultKVCacheReplayLog):
    """
    Stores token indexes and slot positions used for inserting new KV content.

    `token_chunks` is the list of `token_idx` chunks passed to
    :meth:`AttnWeightsKVCache.forward` and :meth:`AttnWeightsKVCache.prefill`.
    `slot_positions` contains blocks of shape
    `(batch_size, n_query_groups, blocksize)`, containing insert positions
    for token positions >= `cache_length`.

    """
    def __init__(
        self,
        token_chunks: List[torch.Tensor],
        slot_positions: List[torch.Tensor],
        cache_length: int,
        max_prefill_length: int,
        blocksize: int,
        grace_period: int = 0,
    ):
        super().__init__(
            token_chunks,
            cache_length=cache_length,
            max_prefill_length=max_prefill_length,
            grace_period=grace_period,
        )
        if slot_positions:
            if not self.token_chunks:
                raise ValueError("Cannot have empty token_chunks, non-empty slot_positions")
            self._slot_positions = None
            device = self.device
            if any(b.device != device for b in slot_positions):
                raise ValueError(f"All token_chunks and slot_position entries must be on same device {device}")
            shape = slot_positions[0].shape[:-1]
            if not all(b.shape[:-1] == shape for b in slot_positions):
                raise ValueError("Blocks in slot_positions have incompatible shapes")
            batch_size = shape[0]
            if any(c.shape[0] != batch_size for c in self.token_chunks):
                raise ValueError(f"slot_positions has batch size {batch_size}, but some token_chunks entries differ")
        self._slot_positions = slot_positions
        self.blocksize = blocksize

    @property
    def slot_positions(self) -> List[torch.Tensor]:
        return self._slot_positions

    @property
    def device(self) -> Optional[torch.device]:
        if self.slot_positions:
            return self.slot_positions[0].device
        else:
            return super().device

    def finalize(
        self,
        input_pos: int,
    ) -> "AttnWeightsReplayLog":
        if input_pos <= self.cache_length:
            slot_positions = []
        else:
            end = (input_pos - self.cache_length) % self.blocksize
            if end == 0:
                slot_positions = self._slot_positions
            else:
                last_block = self._slot_positions[-1][:, :, :end]
                slot_positions = self._slot_positions[:-1] + [last_block]
        return AttnWeightsReplayLog(
            token_chunks=self.token_chunks,
            slot_positions=slot_positions,
            cache_length=self.cache_length,
            max_prefill_length=self.max_prefill_length,
            blocksize=self.blocksize,
            grace_period=self.grace_period,
        )

    def append_position_index(
        self, index: torch.Tensor, input_pos: int,
    ):
        if self.device is not None and index.device != self.device:
            raise ValueError(f"index.device = {index.device}, must be{self.device}")
        kwargs = dict(dtype=torch.uint32)
        shape = index.shape[:2] + (self.blocksize,)
        bstart = (input_pos - self.cache_length) % self.blocksize
        if bstart == 0:
            self._append_new_pos_log_block(shape, index.device)
        block = self._slot_positions[-1]
        num = index.shape[2]
        bend = min(bstart + num, self.blocksize)
        istart = 0
        iend = bend - bstart
        block[:, :, bstart:bend] = index[:, :, istart:iend].to(**kwargs)
        while iend < num:
            istart = iend
            block = self._append_new_pos_log_block(shape, index.device)
            bend = min(num - istart, self.blocksize)
            iend = istart + bend
            block[:, :, :bend] = index[:, :, istart:iend].to(**kwargs)

    def _append_new_pos_log_block(
        self, shape: Tuple[int, ...], device: torch.device,
    ):
        new_block = torch.zeros(shape, device=device, dtype=torch.uint32)
        self._slot_positions.append(new_block)
        return new_block

    def _position_and_offset(self, input_pos: int) -> Tuple[int, int]:
        if input_pos < self.cache_length:
            raise ValueError(f"token_pos = {input_pos} must be >= {self.cache_length} = cache_length")
        assert input_pos >= self.cache_length
        offset = None
        pos = self.cache_length
        relpos = None
        for relpos, block in enumerate(self._slot_positions):
            block_len = block.shape[-1]
            if input_pos < pos + block_len:
                offset = input_pos - pos
                break
            pos += block_len
        if offset is None:
            raise ValueError(f"token_pos = {input_pos} is too large")
        return relpos, offset

    def extract_index(
        self,
        input_pos: int,
        num: int,
        **kwargs,
    ) -> torch.Tensor:
        relpos, bstart = self._position_and_offset(input_pos)
        parts = []
        block = self._slot_positions[relpos]
        blocksize = block.shape[-1]
        bend = min(bstart + num, blocksize)
        parts.append(block[:, :, bstart:bend].to(**kwargs))
        done = bend - bstart
        while done < num:
            relpos += 1
            block = self._slot_positions[relpos]
            blocksize = block.shape[-1]
            bend = min(num - done, blocksize)
            parts.append(block[:, :, :bend].to(**kwargs))
            done += bend
        return torch.cat(parts, dim=-1)


@dataclass(frozen=True)
class UpdateTokenPositionsGracePeriod:
    positions: torch.Tensor
    num1: int
    prefix: int


def update_token_positions(
    token_positions: torch.Tensor,
    input_pos: int,
    num: int,
    batch_size: int,
    index: Optional[torch.Tensor],
    grace_period: int = 0,
    next_grace_pos: Optional[int] = None,
) -> Optional[UpdateTokenPositionsGracePeriod]:
    """
    `token_positions` of shape `(batch_size, n_query_groups, cache_length)`
    contains token positions for each slot. If `batch_size < batch_size`,
    only the leading rows are used. It is updated here with new token positions
    `input_pos` to `input_pos + num - 1`.

    If `input_pos < cache_length`, must have `input_pos + num <=
    cache_length`. In this case, slot positions are just token positions.
    Otherwise, `index` must be given, with semantics as in
    :class:`AttnWeightsKVCache`. Afterwards, `input_pos` should be
    increased by `num`.

    """
    assert token_positions.ndim == 3
    bs, n_query_groups, cache_length = token_positions.shape
    assert 1 <= batch_size <= bs
    assert num >= 1 and input_pos >= 0
    assert cache_length >= 1
    cache_full = input_pos >= cache_length
    assert cache_full or input_pos + num <= cache_length
    assert grace_period >= 0
    device = token_positions.device
    arange_kwargs = dict(dtype=token_positions.dtype, device=device)
    result = None
    if not cache_full:
        start = input_pos
        end = input_pos + num
        token_positions[:batch_size, :, start:end] = torch.arange(
            start, end, **arange_kwargs,
        ).view(1, 1, -1)
    else:
        assert index is not None
        shape = (batch_size, n_query_groups, num)
        assert index.shape == shape, (index.shape, shape)
        if grace_period > 0:
            assert next_grace_pos is not None
            # Grace period, and the cache buffer is already full:
            # We need to copy
            num2 = min(num, grace_period)
            num1 = num - num2  # num1 == 0 <-> num <= grace_period
            prefix = cache_length - grace_period
            # `positions` of length `num2`, possibly wrap-around
            positions = positions_wrap_around(
                num=num2,
                current=next_grace_pos,
                start=prefix,
                end=cache_length,
                batch_size=batch_size,
                n_query_groups=n_query_groups,
                device=device,
                return_tensor=True,
            )
            # Update token_pos
            new_pos = torch.arange(
                input_pos + num1, input_pos + num, **arange_kwargs,
            ).view(1, 1, -1)
            pos_flat = positions[0, 0, :]
            token_positions[:batch_size, :, pos_flat] = new_pos
            start_token_pos = input_pos - grace_period
            result = UpdateTokenPositionsGracePeriod(
                positions=positions,
                num1=num1,
                prefix=prefix,
            )
        else:
            start_token_pos = input_pos

        new_token_pos = index_to_3d(
            torch.arange(start_token_pos, start_token_pos + num, **arange_kwargs),
            batch_size,
            n_query_groups,
        )
        token_positions[:batch_size, ...].scatter_(
            -1, index, new_token_pos,
        )
        return result


DEFAULT_REPLAY_LOG_BLOCKSIZE = 1024

DEFAULT_KEEP_INITIAL_FRACTION = 0.05


class AttnWeightsKVCache(KVCacheWithBuffers):
    """
    Base class for key-value caches which need attention weights to be passed
    (via :meth:`update`) in every round. In general, these weights are used to
    compute scores, based on which eviction decisions are taken. All of this
    happens in :meth:`_compute_scores`, which subclasses need to implement.

    Grace period:

    If `grace_period > 0` (must be `< cache_length`), tokens are kept in the
    cache for at least this many rounds before being considered for eviction.
    This prevents the most recent tokens to be evicted based on noisy score
    values.

    Technically, the final `grace_period` slots are reserved for these grace
    tokens. This part is organized as a ring buffer, the next slot to be
    written to is `next_grace_pos`. In :meth:`forward`, the slot at
    `next_grace_pos` is copied to `next_positions`, and the new token content
    is written to `next_grace_pos`. In subclasses, scores are computed and
    accumulated for all slots, but only the values left of the grace slots
    are used to determine `next_positions`.

    Score buffers:

    Child classes may maintain one or more score buffers, each of shape
    `(max_batch_size, n_query_groups, cache_length)` and type `torch.float32`,
    independent of `self.dtype`.
    They need to be registered in :meth:`_score_buffers` and
    :meth:`_score_buffer_names`, so the generic code can manipulate them, e.g.
    initializing them or copying values if there is a grace period. For score
    buffers not to be initialized with 0, :meth:`_initial_scores_in_forward`
    needs to be implemented.

    Replay log:

    Replay logging is required for gradient computation (fine-tuning), but not
    for inference. It needs to be activated by
    :code:`switch_replay_logging(True)`, and can be deactivated by passing
    `False`. If active, We maintain the slot positions for all tokens at
    positions `>= cache_length`, as well as all `token_idx` values passed to
    :meth:`prefill` and :meth:`forward`. These positions depend on batch and head
    index. They are stored on the same device as the cache buffers, as list of
    `uint32` tensors of shape `(batch_size, n_query_groups, replay_log_blocksize)`.
    These slot positions allow for re-playing all insert decisions later on,
    which is required for gradient computations.

    First :meth:`forward` call when cache is full:

    In general, the initial (prefill) call fills up the cache, while not
    returning attention weights. This means that for the second call, we cannot
    use a score-based decision which slots to overwrite. Instead, for the first
    :meth:`forward` after the cache is filled up, we use a heuristic rule.
    Namely, we evict slots starting at position
    `int(cache_length * keep_initial_fraction)`. This rule is used once only,
    afterwards we use H2O scores. If `keep_initial_fraction > 0`, the KV
    information for the initial tokens is not evicted in this step. For
    `keep_fraction == 0`, this one-time rule corresponds to what a
    :class:`LastRecentlyInsertedKVCache` would do.

    Debugging/testing with `debug_next_positions`:

    If this is set, it must be a list. Then, `index` used in
    :meth:`_forward_internal` are appended there. In this case, we sort the
    index in :meth:`_update` if `max_chunk_size` is given.

    """
    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        grace_period: int = 0,
        replay_log_blocksize: Optional[int] = None,
        detach_attn_weights: bool = False,
        keep_initial_fraction: Optional[float] = None,
        max_chunk_size: Optional[int] = None,
        **base_kwargs,
    ):
        """
        Use :meth:`from_config` in order to use default KV cache buffers, which
        are allocated here.

        Args:
            config: Model config
            buffers: KV cache buffers to be used
            block_idx: Index of block (or layer)
            grace_period: Grace period, see header comment. Defaults to 0
                (no grace period)
            replay_log_blocksize: See header comment.
            detach_attn_weights: If `True`, `attn_weights` are not detached before
                passing it into score computations. This can create a very
                complex computation graph. Defaults to `False`.
            keep_initial_fraction: See above. Defaults to
                :const:`DEFAULT_KEEP_INITIAL_FRACTION`.
            max_chunk_size: If given, any :meth:`forward` call with
                `input_pos > 0`, argument lengths must be `<= max_chunk_size`.
                This is used to speed up :meth:`_update`.

        """
        super().__init__(config, buffers, block_idx=block_idx, **base_kwargs)
        if not (0 <= grace_period < buffers.cache_length):
            raise ValueError(f"Must have 0 <= grace_period < {buffers.cache_length}, but grace_period = {grace_period}")
        if replay_log_blocksize is None:
            replay_log_blocksize = DEFAULT_REPLAY_LOG_BLOCKSIZE
        elif replay_log_blocksize <= 0:
            raise ValueError(f"replay_log_blocksize must be positive, but got {replay_log_blocksize}")
        if keep_initial_fraction is None:
            keep_initial_fraction = DEFAULT_KEEP_INITIAL_FRACTION
        elif not (0 <= keep_initial_fraction < 1):
            raise ValueError(f"keep_initial_fraction = {keep_initial_fraction}, must be in [0, 1)")
        self._keep_initial_fraction = keep_initial_fraction
        self.grace_period = grace_period
        self.replay_log_blocksize = replay_log_blocksize
        self._detach_attn_weights = detach_attn_weights
        cache_length = buffers.cache_length
        if max_chunk_size is not None:
            if not (0 < max_chunk_size <= cache_length):
                raise ValueError(f"max_chunk_size = {max_chunk_size}, must be in (0, {cache_length}]")
            if max_chunk_size > cache_length // 2:
                print(f"max_chunk_size = {max_chunk_size} too large to provide savings. Switching it off.")
                max_chunk_size = None
        self._max_chunk_size = max_chunk_size
        shape = (buffers.max_batch_size, self.n_query_groups, cache_length)
        device = self._default_device_for_new_params()
        self.register_buffer(
            "token_pos",
            torch.zeros(shape, device=device, dtype=torch.int),
            persistent=False,
        )
        # Slot positions where :meth:`forward` writes new key, value tensors.
        # Integer array of shape `(batch_size, n_query_groups, num)`, where
        # `num <= cache_length`. Initialized by :meth:`prefill`.
        self._next_positions = None
        self.next_grace_pos = None
        self.prefill_length = None
        # Signals :meth:`forward` to use the initial rule instead of
        # `_next_positions`. This happens for the first call after the cache
        # has been filled.
        self._use_initial_rule = None
        # For replay log
        self._replay_log = None
        # For debugging/testing
        self.debug_next_positions = None

    @classmethod
    def _parameter_names(cls) -> List[str]:
        return super()._parameter_names() + ["token_pos"]

    def next_positions(self, num: int) -> Optional[torch.Tensor]:
        """
        Returns:
            Batched positions where key and value tensors in next :meth:`forward`
            call are written to, shape `(batch_size, n_query_groups, num)`,
            where `num <= max_tokens_forward`, the remaining ones are not used.
        """
        return None if self._next_positions is None else self._next_positions[:, :, :num]

    @property
    def max_prefill_length(self) -> int:
        """
        Note that :meth:`prefill` must not fill the cache, as we need to
        compute scores in order to make a good eviction decision, and this can
        be done only after the first :meth:`forward` call.

        Returns:
            Maximum length for arguments to :meth:prefill`.
        """
        return self.cache_length

    @property
    def max_tokens_forward(self) -> int:
        diff = self.cache_length - self.current_length
        result = self.cache_length - self.grace_period
        if diff > 0:
            result = min(result, diff)
        return result

    def _forward_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> KeysAndValues:
        num = key.shape[2]
        if self._use_initial_rule is None or (
            not self._use_initial_rule and self.next_positions(num=1) is None
        ):
            raise IndexError("Cache needs to be initialized with 'prefill' before being used")
        if self._max_chunk_size is not None and num > self._max_chunk_size:
            raise ValueError(f"key.shape[2] = {num}, must be <= max_chunk_size = {self._max_chunk_size}")
        diff = self.cache_length - self.current_length
        if not 1 <= num <= self.max_tokens_forward:
            if 0 < diff < num:
                # Cache is almost full, with `diff < num` slots free. There is no
                # good solution for this. We don't know which `num - diff` slots to
                # evict, because we did not compute scores so far.
                raise ValueError(f"key.shape[2] = {num}, must be <= {diff} as long as the cache is not full")
            else:
                raise ValueError(f"key.shape[2] = {num}, must be in [1, {self.max_tokens_forward}]")
        if self._use_initial_rule:
            # Set `_next_positions` according to the initial rule
            num_keep = min(
                int(self._keep_initial_fraction * self.cache_length),
                self.cache_length - num,
            )
            self._next_positions = index_to_3d(
                torch.arange(
                    num_keep,
                    num_keep + num,
                    dtype=torch.int64,
                    device=self.device
                ),
                self.batch_size,
                self.n_query_groups,
            )
            # Only use once:
            self._use_initial_rule = False
        # We need to know how score buffers are initialized for new content. By
        # default, they are filled with 0. But other initial values for scores
        # can be passed via `_initial_scores_in_forward`.
        init_score_values = self._initial_scores_in_forward(key, value)
        # Positions (from last recent `update`) where new content is to be
        # written to:
        index = self.next_positions(num)
        if self.debug_next_positions is not None:
            self.debug_next_positions.append(index.detach().clone().cpu())
        # Update `token_pos`
        update_result = update_token_positions(
            token_positions=self.token_pos,
            input_pos=self.input_pos,
            num=num,
            batch_size=self.batch_size,
            index=index,
            grace_period=self.grace_period,
            next_grace_pos=self.next_grace_pos,
        )
        if update_result is not None:
            # Grace period, and the cache buffer is already full:
            # - Copy buffer part in grace area to `index`
            # - Copy new content into grace area `positions`
            # This is done in a single operation, by fusing `index` and
            # `positions`.
            # New KV content:
            # [ num1 | num2 ], where num = num1 + num2, num2 <= grace_period
            # Copied to initial `num` of `next_positions`:
            # [ old(grace_period) | new[:num1] ]
            num1 = update_result.num1
            prefix = update_result.prefix
            positions = update_result.positions
            assert isinstance(positions, torch.Tensor)  # Sanity check
            key_cp, value_cp = self.kv_buffers.get_slots(positions)
            key_cp = torch.cat((key_cp, key), dim=-2)
            value_cp = torch.cat((value_cp, value), dim=-2)
            fused_index = torch.cat((index, positions), dim=-1)
            # Combined copy and write new content:
            k_and_v = self.kv_buffers.forward(
                positions=fused_index,
                key=key_cp,
                value=value_cp,
            )
            # Update score buffers
            self._update_score_buffers_in_forward(
                grace_period_case=True,
                positions=positions,
                index=index,
                init_score_values=init_score_values,
                num1=num1,
            )
            if num1 == 0:
                # Increment in round-robin fashion (only if num <= grace_period)
                self.next_grace_pos = (self.next_grace_pos - prefix + num) % self.grace_period + prefix
        else:
            # No grace period, or buffer net yet full
            k_and_v = self.kv_buffers.forward(
                positions=index,
                key=key,
                value=value,
            )
            # Update score buffers
            self._update_score_buffers_in_forward(
                grace_period_case=False,
                positions=None,
                index=index,
                init_score_values=init_score_values,
                num1=None,
            )

        if self._replay_log is not None:
            if not isinstance(self._replay_log, AttnWeightsReplayLog):
                raise IndexError("Cannot switch on replay logging in the middle of inference run. Call 'prefill'.")
            self._append_token_idx(token_idx)
            if self.input_pos >= self.cache_length:
                self._replay_log.append_position_index(
                    index=index, input_pos=self.input_pos,
                )
        self._next_positions = None  # Set by next :meth:`update` call
        return k_and_v

    def _update_score_buffers_in_forward(
        self,
        grace_period_case: bool,
        positions: Optional[PositionsType],
        index: torch.Tensor,
        init_score_values: Dict[str, torch.Tensor],
        num1: Optional[int],
    ):
        if grace_period_case:
            pos_do_wrap_around = isinstance(positions, torch.Tensor)
            if pos_do_wrap_around:
                # `positions` is broadcast from shape (1, 1, num)
                pos_flat = positions[0, 0, :]
            else:
                start, end = positions  # end - start == num2
            for scores, name in self._score_buffers():
                # Copy score values from grace region
                if pos_do_wrap_around:
                    scores_src = scores[:self.batch_size, :, pos_flat]
                else:
                    scores_src = scores[:self.batch_size, :, start:end]
                init_vals = init_score_values.get(name)
                if num1 > 0:
                    if init_vals is not None:
                        part2 = init_vals[:, :, :num1]
                    else:
                        part2 = torch.zeros(
                            (1, 1, 1), dtype=scores.dtype, device=scores.device
                        ).expand(self.batch_size, self.n_query_groups, num1)
                    scores_src = torch.cat((scores_src, part2), dim=-1)
                scores[:self.batch_size, ...].scatter_(-1, index, scores_src)
                if pos_do_wrap_around:
                    if init_vals is None:
                        scores[:self.batch_size, :, pos_flat].fill_(0.0)
                    else:
                        scores[:self.batch_size, :, pos_flat] = init_vals[:, :, num1:]
                else:
                    if init_vals is None:
                        scores[:self.batch_size, :, start:end].fill_(0.0)
                    else:
                        scores[:self.batch_size, :, start:end] = init_vals[:, :, num1:]
        else:
            for scores, name in self._score_buffers():
                init_vals = init_score_values.get(name, 0.0)
                scores[:self.batch_size, ...].scatter_(-1, index, init_vals)

    def _append_token_idx(self, token_idx: torch.Tensor):
        if self._replay_log is None:
            self._replay_log = AttnWeightsReplayLog(
                token_chunks=[],
                slot_positions=[],
                cache_length=self.cache_length,
                max_prefill_length=self.max_prefill_length,
                blocksize=self.replay_log_blocksize,
                grace_period=self.grace_period,
            )
        self._replay_log.append_token_chunk(token_idx)

    def _initial_scores_in_forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Called by :meth:`forward`. For some score buffers, initial values are
        computed here based on :meth:`forward` arguments. All other score
        buffers are initialized with 0. See also
        :meth:`_initial_scores_in_prefill`.

        Args:
            key: New keys, `(batch_size, n_query_groups, num, head_size)`,
                where `1 <= num <= max_tokens_forward`
            value: New values, `(batch_size, n_query_groups, num, head_size)`

        Returns:
            Dictionary with keys in :meth:`_score_buffer_names`, values are
            initial score values, shape `(batch_size, n_query_groups, num)`

        """
        return dict()

    def update_requires_attn_weights(self) -> bool:
        return True

    def _update(self, *args, **kwargs):
        """
        Needs arguments `attn_weights` and `query_length` to be passed. This
        method needs to set `self.next_positions` to the slot position where
        :meth:`forward` is to write the new key, value information. If
        `grace_period > 0` and the cache is full, entries in
        `self.next_positions` must be smaller than `self.cache_length
        - self.grace_period`.

        Args:
            attn_weights: Attention weights for the multi-head attention
                computation done just after the last recent :meth:`forward`
                call, summed over the query axis. Shape must be
                `(batch_size, n_query_groups, T)`, where `T <= cache_length` is
                the current cache length. Also, `dtype=float32`.
            query_length: Size of query axis

        """
        if len(args) >= 1:
            attn_weights = args[0]
        else:
            attn_weights = kwargs.get("attn_weights")
            if attn_weights is None:
                raise ValueError("Need to pass 'attn_weights' argument")
        if len(args) >= 2:
            query_length = args[1]
        else:
            query_length = kwargs.get("query_length")
            if query_length is None:
                raise ValueError("Need to pass 'query_length' argument")
        if not isinstance(attn_weights, torch.Tensor):
            raise TypeError("attn_weights argument needs to be torch.Tensor")
        if not isinstance(query_length, int) or query_length <= 0:
            raise TypeError("query_length argument needs to be positive int")
        if attn_weights.device != self.device:
            raise ValueError(f"attn_weights.device = {attn_weights.device}, self.device = {self.device}. Must be the same")
        shape = (self.batch_size, self.n_query_groups, self.current_length)
        if attn_weights.shape != shape:
            raise ValueError(f"Shape of attn_weights must be {shape}, but attn_weights.shape = {attn_weights.shape}")
        # Block gradients
        if self._detach_attn_weights:
            attn_weights = attn_weights.detach()
        attn_weights = attn_weights.to(dtype=torch.float32)

        # Set `next_positions`
        scores = self._compute_scores(attn_weights, query_length)
        if self.current_length < self.cache_length:
            self._set_next_positions_to_free_slots()
        elif self._max_chunk_size is None:
            self._next_positions = scores.argsort(dim=-1)
        else:
            # If `debug_next_positions`, we use `sorted=True` to be able
            # to compare against not using `max_chunk_size`
            self._next_positions = scores.topk(
                k=self._max_chunk_size,
                dim=-1,
                largest=False,
                sorted=self.debug_next_positions is not None,
            )[1]

    def _compute_scores(
        self,
        attn_weights: torch.Tensor,
        query_length: int,
    ) -> Optional[torch.Tensor]:
        """
        Called by :meth:`update`. Updates scores, based on `attn_weights`.
        If the cache is full, the scores are returned, based on which
        `_next_positions` is determined, namely as score minimizers. This
        excludes the grace region at the end of the buffer, since these tokens
        must not be evicted.

        Args:
            attn_weights: Attention weights for the multi-head attention
                computation done just after the last recent :meth:`forward`
                call, summed over the query axis. Shape must be
                `(batch_size, n_query_groups, T)`, where `T <= cache_length` is
                the current cache length. Also, `dtype=float32`.
            query_length: Size of query axis

        Returns:
            scores, shape `(batch_size, n_query_groups, cache_length - grace_period)`,
            only if cache is full.
        """
        raise NotImplementedError()

    def _prefill_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        """
        Starts a generation loop by passing key and value tensors coming from
        a prefill with embeddings coming from the prompts. The length `T` must
        be smaller than `cache_length`. The batch size must be
        `batch_size <= max_batch_size`.

        Note: The largest `T` this method can be called for is `cache_length - 1`.
        This is because we cannot fill the cache and call :meth:`forward`
        before calling :meth:`update`, since otherwise `next_positions` is not
        properly set. The underlying issue is that the prefill computation uses
        the efficient "training" computation of multi-head attention, for which
        we do not obtain the attention weights. But these are needed (for the
        final token filling the cache) in :meth:`update`.

        Note: All score buffers are initialized with 0 here. For scores which
        are initialized differently (see :meth:`_initial_scores_in_forward`),
        this has to be overwritten in :meth:`_initial_scores_in_prefill`.

        Args:
            key: Prefill keys, `(batch_size, n_query_groups, T, head_size)`
            value: Prefill values, `(batch_size, n_query_groups, T, head_size)`
            token_idx: Token indices of input sequence, `(batch_size, T)`

        """
        init_length = key.shape[2]
        self.kv_buffers.prefill(key, value)
        # Update `token_pos`
        update_token_positions(
            token_positions=self.token_pos,
            input_pos=0,
            num=init_length,
            batch_size=self.batch_size,
            index=None,
            grace_period=self.grace_period,
        )
        self._use_initial_rule = self.current_length == self.cache_length
        if not self._use_initial_rule:
            self._set_next_positions_to_free_slots()
        if self.grace_period > 0:
            # First slot to move out once the cache is full
            self.next_grace_pos = self.cache_length - self.grace_period
        self.prefill_length = self.current_length
        if self._replay_log is not None:
            # Reset and log `token_idx`
            self._replay_log = None
            self._append_token_idx(token_idx)
        self._initial_scores_in_prefill(key, value)

    def _initial_scores_in_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """
        Called at the end of :meth:`prefill`. The default implementation here
        initializes all score buffers with zeros. In subclasses, for score
        buffers which are not initialized with 0, their initialization up to
        `current_length` has to be done here.

        This method is linked with :meth:`_initial_scores_in_forward`, the same
        score buffers need to be covered. Why two methods? This one writes
        directly into the score buffers, while :meth:`_initial_scores_in_forward`
        returns values, as it is called with arguments of smaller length.

        Args:
            key: Prefill keys, `(batch_size, n_query_groups, T, head_size)`
            value: Prefill values, `(batch_size, n_query_groups, T, head_size)`

        """
        for scores, _ in self._score_buffers():
            scores.fill_(0.0)

    def token_positions(self) -> torch.Tensor:
        if self.current_length is None:
            raise IndexError("Cache is not initialized, call 'prefill' first")
        return self.token_pos[:self.batch_size, :, :self.current_length]

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        """
        Note that we do not count `_next_positions`. It is usually smallest and
        does not have a fixed size.

        """
        sz_buffs, dct_sz = self.kv_buffers.size_estimate()
        sz_tp = bitsize_of(self.token_pos)
        dct_sz["token_pos"] = sz_tp
        return sz_buffs + sz_tp, dct_sz

    @classmethod
    def size_estimate_apriori(cls, params: KVCacheParams, **kwargs) -> Tuple[int, Dict[str, int]]:
        """
        `cache_length` is required in `kwargs`. If `buffer_type` is given in
        `kwargs`, the size for this type is used, otherwise for the default
        type `DefaultKVCacheBuffers`.

        """
        buff_params = KVCacheBuffersParams.from_params(params)
        buffer_type = kwargs.get("buffer_type", DefaultKVCacheBuffers)
        sz_buffs, dct_sz = buffer_type.size_estimate_apriori(
            buff_params, cache_length=params.cache_length, **kwargs,
        )
        dtype = params.dtype
        if dtype is None:
            raise ValueError("params.dtype must be provided")
        sz_tp = params.max_batch_size * params.n_query_groups * params.cache_length * bits_for_torch_dtype(torch.int)
        return sz_buffs + sz_tp, dict(dct_sz, token_pos=sz_tp)

    def switch_replay_logging(self, status: bool):
        if status:
            if self._replay_log is None:
                self._replay_log = []  # Dummy, created later
        else:
            self._replay_log = None

    @property
    def do_replay_logging(self) -> bool:
        return self._replay_log is not None

    def get_replay_log(self) -> Optional[KVCacheReplayLog]:
        """
        Returns list of slot position tensors.
        If we applied `torch.cat(..., dim=-1)` to this, we would get a tensor `res`
        of shape `(batch_size, n_query_groups, T)`, where
        `T = input_pos - cache_length`, so the content for any token
        position `t + cache_length` was written to slots `res[:, :, t]`.

        Returns:
            Replay log object for this cache, which contains all token chunks
            and the slot positions

        """
        if self._replay_log is None:
            return None
        else:
            return self._replay_log.finalize(self.input_pos)

    def _score_buffers(self) -> List[Tuple[torch.Tensor, str]]:
        """
        The class uses a number of score buffers, each of shape
        `(max_batch_size, n_query_groups, cache_length)`. Subclasses need to
        register their score buffers here.

        Returns:
            List of `(array, name)` entries, one for each score buffer
        """
        raise NotImplementedError()

    @classmethod
    def _score_buffer_names(cls) -> List[str]:
        """
        Subclasses need to register names of their score buffers here.

        Returns:
            List of names of score buffers
        """
        raise NotImplementedError()

    def _set_next_positions_to_free_slots(self):
        """
        If `current_length < cache_length`, this method sets `_next_positions`
        to cover the remaining free slots.
        """
        if self.current_length < self.cache_length:
            self._next_positions = index_to_3d(
                torch.arange(
                    self.current_length,
                    self.cache_length,
                    dtype=torch.int64,
                    device=self.device
                ),
                self.batch_size,
                self.n_query_groups,
            )

    def _base_kwargs_for_clone(self) -> Dict[str, Any]:
        base_kwargs = super()._base_kwargs_for_clone()
        base_kwargs.update(
            dict(
                grace_period=self.grace_period,
                replay_log_blocksize=self.replay_log_blocksize,
                detach_attn_weights=self._detach_attn_weights,
                keep_initial_fraction=self._keep_initial_fraction,
                max_chunk_size=self._max_chunk_size,
            )
        )
        return base_kwargs
