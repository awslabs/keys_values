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
"""
Tentative KV cache updates, as needed for speculative decoding: tokens are
inserted into the KV caches, and only some of them are kept afterwards.

This is an add-on on top of buffer switching
(:meth:`KVCacheWithBuffers.switch_buffers`). The KV cache API itself is not
changed.

"""

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from keys_values.attention import KeysAndValues
from keys_values.kvcache.base import KVCache
from keys_values.kvcache.basics import (
    DenseKVCache,
    KVCacheWithBuffers,
    KVCacheWithBuffersState,
    LastRecentlyInsertedKVCache,
    LastRecentlyInsertedKVCacheState,
)
from keys_values.kvcache.buffers import (
    DefaultKVCacheBuffers,
    KVCacheBuffers,
    KVCacheBuffersParams,
    PositionsType,
)
from keys_values.kvcache.quant_buffers import QuantizedKVCacheBuffers
from keys_values.kvcache.replay_buffers import ReplayKVCacheBuffers


@dataclass(frozen=True)
class _UndoRecord:
    """
    Undo information for one :meth:`UndoKVCacheBuffers.forward` call.
    `saved_key`, `saved_value` are the previous contents of `saved_positions`,
    which are the first `num_saved` positions written by the call. They are
    `None` if no occupied slot was overwritten.

    """

    num: int
    saved_positions: Optional[PositionsType]
    saved_key: Optional[torch.Tensor]
    saved_value: Optional[torch.Tensor]


class UndoKVCacheBuffers(KVCacheBuffers):
    """
    Wraps :class:`DefaultKVCacheBuffers`, so that writes for the most recent
    tokens can be undone, see :meth:`commit_prefix`.

    This plays the role of :class:`ReplayKVCacheBuffers` for buffers without
    quantization. There is no de-quantized copy here which could serve as
    scratch space, and copying the buffers would be expensive. Instead, writes
    go to the base buffers, and the previous content of the slots which are
    overwritten is kept. For dense caches, nothing is overwritten. For lastrec
    caches, these are as many slots as tentative tokens once the cache is
    full.

    """

    def __init__(self, base: DefaultKVCacheBuffers):
        if not isinstance(base, DefaultKVCacheBuffers):
            raise TypeError(f"type(base) = {type(base)}, must be DefaultKVCacheBuffers")
        if not base.buffers_are_allocated or base.current_length is None:
            raise ValueError("base must have allocated buffers. Call 'prefill' first")
        super().__init__(base.get_params(), base.cache_length)
        self.base = base
        self.batch_size = base.batch_size
        self.current_length = base.current_length
        self._start_length = base.current_length
        self._records: List[_UndoRecord] = []

    @property
    def device(self) -> Optional[torch.device]:
        return self.base.device

    @property
    def buffers_are_allocated(self) -> bool:
        return self.base.buffers_are_allocated

    @property
    def num_updated_tokens(self) -> int:
        """
        Returns:
            Number of tokens written since these buffers were created

        """
        return sum(record.num for record in self._records)

    def _save_overwritten(
        self,
        positions: PositionsType,
    ) -> Tuple[Optional[PositionsType], Optional[torch.Tensor], Optional[torch.Tensor]]:
        current_length = self.base.current_length
        if isinstance(positions, tuple):
            start, end = positions
            end = min(end, current_length)
            if end <= start:
                return None, None, None
            key, value = self.base.get_slots((start, end))
            return (start, end), key.clone(), value.clone()
        # Slots at or beyond `current_length` are not occupied (and may not
        # exist yet if buffers grow). Clamping keeps the gather in range. What
        # is saved for these slots is restored into slots beyond the current
        # length, which are not read.
        key, value = self.base.get_slots(positions.clamp(max=current_length - 1))
        return positions, key, value

    def _forward(
        self,
        positions: PositionsType,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> KeysAndValues:
        # Note: `self.current_length` has been increased already, while
        # `self.base.current_length` has not
        saved_positions, saved_key, saved_value = self._save_overwritten(positions)
        k_and_v = self.base.forward(positions, key, value)
        self._records.append(
            _UndoRecord(
                num=key.shape[2],
                saved_positions=saved_positions,
                saved_key=saved_key,
                saved_value=saved_value,
            )
        )
        return k_and_v

    @staticmethod
    def _undo(record: _UndoRecord, num_keep: int):
        """
        Returns the part of `record.saved_*` for the positions of tokens
        `range(num_keep, record.num)`, or `None` if nothing is to be restored.

        """
        positions = record.saved_positions
        if positions is None:
            return None
        if isinstance(positions, tuple):
            start, end = positions
            first = start + num_keep
            if first >= end:
                return None
            return (
                (first, end),
                record.saved_key[:, :, num_keep:, :],
                record.saved_value[:, :, num_keep:, :],
            )
        return (
            positions[:, :, num_keep:],
            record.saved_key[:, :, num_keep:, :],
            record.saved_value[:, :, num_keep:, :],
        )

    def commit_prefix(self, num: int):
        """
        Keeps the writes for the first `num` tokens and undoes the writes for
        all later ones. Afterwards, the base buffers are in the state of having
        received only these `num` tokens. These buffers here should not be
        used anymore.

        Args:
            num: Number of tokens to keep, in `[0, num_updated_tokens]`

        """
        total = self.num_updated_tokens
        if not (0 <= num <= total):
            raise ValueError(f"num = {num}, must be in [0, {total}]")
        num_undo = total - num
        for record in reversed(self._records):
            if num_undo == 0:
                break
            num_record = min(record.num, num_undo)
            undo = self._undo(record, record.num - num_record)
            if undo is not None:
                self.base.set_slots(*undo)
            num_undo -= num_record
        self.base.current_length = min(self.base.cache_length, self._start_length + num)
        self.current_length = self.base.current_length
        self._records = []

    def get_slots(
        self,
        positions: PositionsType,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base.get_slots(positions)

    def get_keys_values(self) -> Optional[KeysAndValues]:
        return self.base.get_keys_values()

    def write_back(self):
        self.base.write_back()

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        return self.base.size_estimate()

    @staticmethod
    def _raise_error(name: str):
        raise NotImplementedError(
            f"Cannot call '{name}' while KV cache updates are tentative. Call "
            "TentativeKVCacheUpdates.keep or discard first"
        )

    def _allocate_buffers(self, device: Optional[torch.device] = None):
        self._raise_error("_allocate_buffers")

    def set_slots(
        self,
        positions: PositionsType,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        self._raise_error("set_slots")

    def _prefill(self, key: torch.Tensor, value: torch.Tensor):
        self._raise_error("_prefill")

    def reset(self):
        self._raise_error("reset")

    def _deallocate(self):
        self._raise_error("_deallocate")

    @staticmethod
    def size_estimate_apriori(
        params: KVCacheBuffersParams,
        **kwargs,
    ) -> Tuple[int, Dict[str, int]]:
        UndoKVCacheBuffers._raise_error("size_estimate_apriori")


SUPPORTED_CACHE_TYPES = (DenseKVCache, LastRecentlyInsertedKVCache)


def _advance_state(
    state: KVCacheWithBuffersState,
    num: int,
) -> KVCacheWithBuffersState:
    """
    Args:
        state: State of a :class:`DenseKVCache` or
            :class:`LastRecentlyInsertedKVCache`
        num: Number of tokens inserted after `state`

    Returns:
        State after inserting `num` more tokens

    """
    new_state = copy.copy(state)
    new_state.input_pos = state.input_pos + num
    if type(state) is LastRecentlyInsertedKVCacheState:
        # Same updates as in `LastRecentlyInsertedKVCache._forward_internal`
        start = state.init_grace_tokens
        end = state.cache_length
        next_position = state.next_position
        num_left = num
        while num_left > 0:
            num_part = min(num_left, end - next_position)
            next_position += num_part
            num_left -= num_part
            if next_position == end:
                next_position = start
        new_state.next_position = next_position
    elif type(state) is not KVCacheWithBuffersState:
        raise NotImplementedError(f"State of type {type(state)} is not supported")
    return new_state


class TentativeKVCacheUpdates:
    """
    Makes updates of a list of KV caches (typically all caches of a model)
    tentative for a while. After :meth:`start`, tokens inserted into the
    caches are tentative. :meth:`keep` keeps the first `num` of them and
    discards the others, so that each cache is in the state of having received
    only the kept tokens. :meth:`discard` is the same as `keep(0)`.

    This is what speculative decoding needs: draft tokens are inserted into
    the caches, and only some of them are accepted. The number of tentative
    tokens is small (at most `speculative_k + 1`).

    :meth:`start` saves the state of each cache and switches its buffers to
    :class:`ReplayKVCacheBuffers` (quantized buffers) or
    :class:`UndoKVCacheBuffers` (default buffers). :meth:`keep` writes the
    kept tokens to the original buffers and switches back, with the cache
    state advanced by the kept tokens.

    Only :class:`DenseKVCache` and :class:`LastRecentlyInsertedKVCache` are
    supported, since their state after the kept tokens can be computed. This
    is not the case for caches whose scores depend on attention weights,
    since these are summed over all tokens of a call.

    Note: With quantized buffers, entries inserted while tentative are not
    quantized before :meth:`keep` (see :class:`ReplayKVCacheBuffers`). If
    several calls are made in between, later calls see these entries without
    quantization error, unlike with the original buffers.

    """

    def __init__(self, kv_caches: Sequence[KVCache]):
        for layer_idx, cache in enumerate(kv_caches):
            if type(cache) not in SUPPORTED_CACHE_TYPES:
                names = ", ".join(c.__name__ for c in SUPPORTED_CACHE_TYPES)
                raise NotImplementedError(
                    f"Layer {layer_idx}: Tentative updates not supported for "
                    f"KV caches of type {type(cache).__name__}. Supported: {names}"
                )
        self.kv_caches: List[KVCacheWithBuffers] = list(kv_caches)
        self._saved: Optional[List[Tuple[KVCacheWithBuffersState, KVCacheBuffers]]] = (
            None
        )

    @property
    def is_active(self) -> bool:
        """
        Returns:
            Has :meth:`start` been called, but not :meth:`keep` or
            :meth:`discard` yet?

        """
        return self._saved is not None

    @property
    def num_tentative(self) -> int:
        """
        Returns:
            Number of tokens inserted since :meth:`start`

        """
        if not self.is_active:
            return 0
        nums = {
            cache.input_pos - saved[0].input_pos
            for cache, saved in zip(self.kv_caches, self._saved)
            if saved is not None
        }
        if len(nums) != 1:
            raise IndexError(
                f"KV caches received different numbers of tokens since start: {nums}"
            )
        return next(iter(nums))

    @staticmethod
    def _check_cache(layer_idx: int, cache: KVCacheWithBuffers):
        prefix = f"Layer {layer_idx}: "
        if not cache.buffers_are_allocated or cache.input_pos == 0:
            raise IndexError(prefix + "KV cache must be prefilled before start")
        if cache.do_replay_logging:
            raise IndexError(
                prefix + "Tentative updates not supported while replay logging is on"
            )
        if cache._checkpoint_hook is not None:
            raise IndexError(
                prefix + "Tentative updates not supported with a checkpoint hook"
            )
        if not isinstance(
            cache.kv_buffers, (QuantizedKVCacheBuffers, DefaultKVCacheBuffers)
        ):
            raise NotImplementedError(
                prefix + f"Buffers of type {type(cache.kv_buffers)} not supported"
            )

    def start(self):
        """
        From now on, tokens inserted into the caches are tentative.

        """
        if self.is_active:
            raise IndexError("Tentative updates have already been started")
        for layer_idx, cache in enumerate(self.kv_caches):
            self._check_cache(layer_idx, cache)
        saved = []
        try:
            for cache in self.kv_caches:
                base = cache.kv_buffers
                state = cache.get_state()
                if isinstance(base, QuantizedKVCacheBuffers):
                    cache.switch_buffers(ReplayKVCacheBuffers(base))
                else:
                    cache.switch_buffers(UndoKVCacheBuffers(base))
                saved.append((state, base))
        except Exception:
            # Leave caches as they were
            for cache, (state, base) in zip(self.kv_caches, saved):
                cache.switch_buffers(base, cache_state=state)
            raise
        self._saved = saved

    def keep(self, num: int):
        """
        Keeps the first `num` tokens inserted since :meth:`start` and discards
        the others.

        Caches are processed one after the other. If this fails for one of
        them, the caches would contain different numbers of tokens. In this
        case, all caches are switched back to their original buffers and
        reset, before the exception is raised again. They have to be prefilled
        before they can be used again.

        Args:
            num: Number of tokens to keep, in `[0, num_tentative]`

        """
        if not self.is_active:
            raise IndexError("Tentative updates have not been started")
        if num != 0:
            num_tentative = self.num_tentative
            if not (0 <= num <= num_tentative):
                raise ValueError(f"num = {num}, must be in [0, {num_tentative}]")
        try:
            for pos, cache in enumerate(self.kv_caches):
                state, base = self._saved[pos]
                cache.kv_buffers.commit_prefix(num)
                cache.switch_buffers(base, cache_state=_advance_state(state, num))
                self._saved[pos] = None
        except BaseException:
            self._reset_after_failure()
            raise
        self._saved = None

    def _reset_after_failure(self):
        for pos, cache in enumerate(self.kv_caches):
            saved = self._saved[pos]
            if saved is not None:
                state, base = saved
                cache.switch_buffers(base, cache_state=state)
            cache.reset()
        self._saved = None

    def discard(self):
        """
        Discards all tokens inserted since :meth:`start`. This also works if
        the caches received different numbers of tokens, for example if a
        forward pass failed: each cache returns to its state at :meth:`start`.

        """
        self.keep(0)
