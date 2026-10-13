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
from typing import Tuple, Dict, Any, List, Optional

import torch

from keys_values.config import Config
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.new.basics import (
    KVCachePolicy,
    KVCacheReplayLog,
    KVCacheReplayLogStatus,
    PositionsNewEntriesType,
)
from keys_values.kvcache.new.policy import DefaultKVCachePolicy
from keys_values.kvcache.attn_weights import (
    update_token_positions,
    DEFAULT_KEEP_INITIAL_FRACTION,
)
from keys_values.kvcache.utils import PositionsType
from keys_values.utils import (
    index_to_3d,
    bits_for_torch_dtype,
    bitsize_of,
)


class AttnWeightsKVCachePolicy(DefaultKVCachePolicy):
    """
    Base class for KV cache policies which need attention weights (summed over
    the query axis) to be passed (via :meth:`pass_attn_weights`) in every
    round. In general, these weights are used to compute scores, based on which
    eviction decisions are taken, once the cache is full. Score computations
    happen in :meth:`_compute_scores`, which subclasses need to implement.

    Grace period:

    If `grace_period > 0` (must be `< cache_length`), tokens are kept in the
    cache for at least this many rounds before being considered for eviction,
    independent of their score values. This prevents the most recent tokens
    to be evicted based on noisy score values (scores are often cumulative).

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
    for inference. It is (de)activated by :meth:`switch_replay_logging`. If
    active, we maintain the slot positions for all tokens at positions
    `>= cache_length`, as well as all `token_idx` values passed to
    :meth:`forward`. These positions depend on batch and head index. They are
    stored on the same device as the cache buffers, as list of `uint32`
    tensors of shape `(batch_size, n_query_groups, replay_log_blocksize)`.
    These slot positions allow for re-playing all insert decisions later on,
    which is required for gradient computations.

    First :meth:`forward` call when cache is full:

    In general, the initial (prefill) call fills up the cache, while not
    returning attention weights. This means that for the second call, we cannot
    use a score-based decision for which slots to overwrite. Instead, for the
    first :meth:`forward` after the cache is filled up, we use a heuristic
    rule. Namely, we evict slots starting at position
    `int(cache_length * keep_initial_fraction)`. This rule is used once only,
    afterwards we use H2O scores. If `keep_initial_fraction > 0`, the KV
    information for the initial tokens is not evicted in this step. For
    `keep_fraction == 0`, this one-time rule corresponds to what a
    :class:`LastRecentlyInsertedKVCache` would do.

    Limiting the chunk size:

    The length of tensors passed to :meth:`forward` (along the sequence axis)
    is called chunk size. The first call (after :meth:`reset`) is prefilling
    with chunks up to `max_prefill_length`. If `max_chunk_size` is used, chunks
    other than prefill must not be longer than this. The limitation is used in
    order to process score values by `torch.topk` instead of `torch.argsort` in
    :meth:`pass_attn_weights`, which can be significantly faster.

    Debugging/testing with `debug_next_positions`:

    If this is set, it must be a list. Then, `index` used in
    :meth:`_forward_internal` are appended there. In this case, we sort the
    index in :meth:`pass_attn_weights` if `max_chunk_size` is given.
    """

    def __init__(
        self,
        config: Config,
        cache_length: int,
        max_batch_size: int,
        use_replay_logging: bool = True,
        replay_log: Optional[KVCacheReplayLog] = None,
        grace_period: int = 0,
        detach_attn_weights: bool = False,
        keep_initial_fraction: Optional[float] = None,
        max_chunk_size: Optional[int] = None,
    ):
        super().__init__(config, cache_length, use_replay_logging, replay_log)
        if not (0 <= grace_period < cache_length):
            raise ValueError(
                f"Must have 0 <= grace_period < {cache_length}, but grace_period = {grace_period}"
            )
        if keep_initial_fraction is None:
            keep_initial_fraction = DEFAULT_KEEP_INITIAL_FRACTION
        elif not (0 <= keep_initial_fraction < 1):
            raise ValueError(
                f"keep_initial_fraction = {keep_initial_fraction}, must be in [0, 1)"
            )
        self._keep_initial_fraction = keep_initial_fraction
        self.grace_period = grace_period
        self._detach_attn_weights = detach_attn_weights
        if max_chunk_size is not None:
            if not (0 < max_chunk_size <= cache_length):
                raise ValueError(
                    f"max_chunk_size = {max_chunk_size}, must be in (0, {cache_length}]"
                )
            if max_chunk_size > cache_length // 2:
                print(
                    f"max_chunk_size = {max_chunk_size} too large to provide savings. Switching it off."
                )
                max_chunk_size = None
        self.max_chunk_size = max_chunk_size
        shape = (max_batch_size, self.n_query_groups, cache_length)
        device = self._device_internal()
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
        # For debugging/testing
        self.debug_next_positions = None
        # Tracked here as well:
        self._current_length = 0

    @classmethod
    def _parameter_names(cls) -> List[str]:
        return super()._parameter_names() + ["token_pos"]

    def next_positions(self, num: int) -> Optional[torch.Tensor]:
        """
        Returns:
            Batched positions where key and value tensors in next :meth:`forward`
            call are written to, shape `(batch_size, n_query_groups, num)`,
            where `num <= max_forward_length()`, the remaining ones are not used.
        """
        return (
            None if self._next_positions is None else self._next_positions[:, :, :num]
        )

    def token_positions(self) -> torch.Tensor:
        if self.batch_size is None:
            raise IndexError("Cache is not initialized, call 'prefill' first")
        return self.token_pos[: self.batch_size, :, : self._current_length]

    # We initialize score buffers here even if `replay_log_status == "play"`,
    # so that scores are not updated down the line.
    def _forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        positions: Optional[PositionsNewEntriesType] = None,
    ) -> PositionsNewEntriesType:
        num = key.shape[2]
        device = self._device_internal()
        if self._use_initial_rule is None or (
            not self._use_initial_rule and self.next_positions(num=1) is None
        ):
            raise IndexError(
                "Cache needs to be initialized with 'prefill' before being used"
            )
        diff = self.cache_length - self._current_length
        if not 1 <= num <= self.max_forward_length():
            if 0 < diff < num:
                # Cache is almost full, with `diff < num` slots free. There is no
                # good solution for this. We don't know which `num - diff` slots to
                # evict, because we did not compute scores so far.
                raise ValueError(
                    f"key.shape[2] = {num}, must be <= {diff} as long as the cache is not full"
                )
            else:
                raise ValueError(
                    f"key.shape[2] = {num}, must be in [1, {self.max_forward_length()}]"
                )
        if self._use_initial_rule:
            # Set `_next_positions` according to the initial rule
            num_keep = min(
                int(self._keep_initial_fraction * self.cache_length),
                self.cache_length - num,
            )
            self._next_positions = index_to_3d(
                torch.arange(
                    num_keep, num_keep + num, dtype=torch.int64, device=device,
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
        if positions is None:
            # Positions (from last recent `update`) where new content is to be
            # written to:
            index = self.next_positions(num)
        else:
            # `positions` is given. Update `token_pos`, etc.
            if isinstance(positions, list) or positions.ndim != 3:
                raise ValueError(f"positions = {positions} invalid (must not 3D index, not a list)")
            if positions.shape[-1] < num:
                raise ValueError(f"positions.shape = {positions.shape}, num = {num}: too short")
            index = positions[:, :, :num]
        if self.debug_next_positions is not None:
            self.debug_next_positions.append(index.detach().clone().cpu())

        # Update `token_pos`
        update_result = update_token_positions(
            token_positions=self.token_pos,
            input_pos=self._input_pos,
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
            extra_pos = update_result.positions
            assert isinstance(extra_pos, torch.Tensor) and extra_pos.ndim == 3  # Sanity check
            if positions is None:
                # Special case: `positions.shape[-1] > num`. This means
                # that buffer entries at `extra_pos` are to be copied to
                # different slots as part of the update.
                positions = torch.cat((index, extra_pos), dim=-1)
            else:
                # Just validate
                should_be = positions[:, :, num:].to(device=extra_pos.device)
                if not extra_pos.equal(should_be):
                    raise ValueError(
                        f"extra_pos:\n{extra_pos} -- {extra_pos.shape}\nshould_be:\n{should_be} -- {should_be.shape}\nShould be the same!"
                    )
            # Update score buffers
            self._update_score_buffers_in_forward(
                grace_period_case=True,
                positions=extra_pos,
                index=index,
                init_score_values=init_score_values,
                num1=num1,
            )
            if num1 == 0:
                # Increment in round-robin fashion (only if num <= grace_period)
                self.next_grace_pos = (
                    self.next_grace_pos - prefix + num
                ) % self.grace_period + prefix
        else:
            # No grace period, or buffer net yet full
            if positions is None:
                positions = index
            elif positions.shape[-1] != num:
                raise ValueError(f"positions.shape = {positions.shape}, num = {num}: must be same length")
            # Update score buffers
            self._update_score_buffers_in_forward(
                grace_period_case=False,
                positions=None,
                index=index,
                init_score_values=init_score_values,
                num1=None,
            )

        self._next_positions = None  # Set by next :meth:`update` call
        self._current_length = min(self._input_pos + num, self.cache_length)
        return positions

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
                    scores_src = scores[: self.batch_size, :, pos_flat]
                else:
                    scores_src = scores[: self.batch_size, :, start:end]
                init_vals = init_score_values.get(name)
                if num1 > 0:
                    if init_vals is not None:
                        part2 = init_vals[:, :, :num1]
                    else:
                        part2 = torch.zeros(
                            (1, 1, 1), dtype=scores.dtype, device=scores.device
                        ).expand(self.batch_size, self.n_query_groups, num1)
                    scores_src = torch.cat((scores_src, part2), dim=-1)
                scores[: self.batch_size, ...].scatter_(-1, index, scores_src)
                if pos_do_wrap_around:
                    if init_vals is None:
                        scores[: self.batch_size, :, pos_flat].fill_(0.0)
                    else:
                        scores[: self.batch_size, :, pos_flat] = init_vals[:, :, num1:]
                else:
                    if init_vals is None:
                        scores[: self.batch_size, :, start:end].fill_(0.0)
                    else:
                        scores[: self.batch_size, :, start:end] = init_vals[:, :, num1:]
        else:
            for scores, name in self._score_buffers():
                init_vals = init_score_values.get(name, 0.0)
                scores[: self.batch_size, ...].scatter_(-1, index, init_vals)

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
                where `1 <= num <= max_forward_length()`
            value: New values, `(batch_size, n_query_groups, num, head_size)`

        Returns:
            Dictionary with keys in :meth:`_score_buffer_names`, values are
            initial score values, shape `(batch_size, n_query_groups, num)`

        """
        return dict()

    def requires_attn_weights(self) -> bool:
        return self.replay_log_status() != "play"

    def pass_attn_weights(
        self,
        attn_weights: torch.Tensor,
        query_length: int,
    ):
        if not isinstance(query_length, int) or query_length <= 0:
            raise TypeError("query_length argument needs to be positive int")
        if attn_weights.device != self.device:
            raise ValueError(
                f"attn_weights.device = {attn_weights.device}, self.device = {self.device}. Must be the same"
            )
        shape = (self.batch_size, self.n_query_groups, self._current_length)
        if attn_weights.shape != shape:
            raise ValueError(
                f"Shape of attn_weights must be {shape}, but attn_weights.shape = {attn_weights.shape}"
            )
        # Block gradients
        if self._detach_attn_weights:
            attn_weights = attn_weights.detach()
        attn_weights = attn_weights.to(dtype=torch.float32)

        # Set `next_positions`
        scores = self._compute_scores(attn_weights, query_length)
        if self._current_length < self.cache_length:
            self._set_next_positions_to_free_slots()
        elif self.max_chunk_size is None:
            self._next_positions = scores.argsort(dim=-1)
        else:
            # If `debug_next_positions`, we use `sorted=True` to be able
            # to compare against not using `max_chunk_size`
            self._next_positions = scores.topk(
                k=self.max_chunk_size,
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
        Called by :meth:`pass_attn_weights`. Updates scores, based on
        `attn_weights`. If the cache is full, the scores are returned, based on
        which `_next_positions` is determined, namely as score minimizers. This
        excludes the grace region at the end of the buffer, since these tokens
        must not be evicted.

        Note that score values are updated even if the cache is not full.
        They are just not returned in this case.

        Args:
            attn_weights: Attention weights for the multi-head attention
                computation done just after the last recent :meth:`forward`
                call, summed over the query axis. Shape must be
                `(batch_size, n_query_groups, current_length)`, where
                `current_length <= cache_length` is the current cache length.
                Also, `dtype=float32`.
            query_length: Size of query axis

        Returns:
            scores, shape `(batch_size, n_query_groups, cache_length - grace_period)`,
            only if cache is full. Note that the grace slots at the end of the
            cache are excluded.

        """
        raise NotImplementedError

    def _prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        """
        Note: All score buffers are initialized with 0 here. For scores which
        are initialized differently (see :meth:`_initial_scores_in_forward`),
        this has to be overwritten in :meth:`_initial_scores_in_prefill`.

        """
        init_length = token_idx.shape[-1]
        _batch_size = token_idx.shape[0]
        if _batch_size > self.token_pos.shape[0]:
            raise ValueError(f"token_idx.shape = {token_idx.shape}: Batch size {_batch_size} > {self.token_pos.shape[0]} = max_batch_size")
        self._batch_size = _batch_size
        # Update `token_pos`
        update_token_positions(
            token_positions=self.token_pos,
            input_pos=0,
            num=init_length,
            batch_size=self.batch_size,
            index=None,
            grace_period=self.grace_period,
        )
        self._current_length = init_length
        self._use_initial_rule = init_length == self.cache_length
        if not self._use_initial_rule:
            self._set_next_positions_to_free_slots()
        if self.grace_period > 0:
            # First slot to move out once the cache is full
            self.next_grace_pos = self.cache_length - self.grace_period
        self.prefill_length = init_length
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

    def max_forward_length(self) -> int:
        diff = self.cache_length - self._current_length
        result = self.cache_length - self.grace_period
        if diff > 0:
            result = min(result, diff)
        if self.max_chunk_size is not None:
            result = min(result, self.max_chunk_size)
        return result

    def _base_kwargs_for_clone(self) -> Dict[str, Any]:
        return dict(
            super()._base_kwargs_for_clone(),
            max_batch_size=self.token_pos.shape[0],
            grace_period=self.grace_period,
            detach_attn_weights=self._detach_attn_weights,
            keep_initial_fraction=self._keep_initial_fraction,
            max_chunk_size=self.max_chunk_size,
        )

    def clone(self) -> KVCachePolicy:
        return AttnWeightsKVCachePolicy(**self._base_kwargs_for_clone())

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        tk_p = bitsize_of(self.token_pos)
        return tk_p, {"token_pos": tk_p}

    @classmethod
    def size_estimate_apriori(
        cls, params: KVCacheParams, **kwargs
    ) -> Tuple[int, Dict[str, int]]:
        sz_tp = (
            params.max_batch_size
            * params.n_query_groups
            * params.cache_length
            * bits_for_torch_dtype(torch.int)
        )
        return sz_tp, {"token_pos": sz_tp}

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

    # TODO: Is this still needed? Lightning Fabric was removed!
    def fix_dtype_of_score_buffers(self):
        """
        Iterates over registered score buffers. If any does not have
        `dtype == float32`, it is replaced by one with that `dtype`.

        This special method is needed because `Lightning Fabric` has the
        annoying property of changing `dtype` of registered buffers if
        a model is wrapped with `fabric.setup`.

        """
        raise NotImplementedError

    def _set_next_positions_to_free_slots(self):
        """
        If `current_length < cache_length`, this method sets `_next_positions`
        to cover the remaining free slots.
        """
        if self._current_length < self.cache_length:
            self._next_positions = index_to_3d(
                torch.arange(
                    self._current_length,
                    self.cache_length,
                    dtype=torch.int64,
                    device=self._device_internal(),
                ),
                self.batch_size,
                self.n_query_groups,
            )

    # Note: We refrain from doing more than setting `current_length = 0`.
    # The next valid call must be :meth:`prefill`, which does a proper
    # reset (this may need its args).
    def reset(self, replay_log_status: KVCacheReplayLogStatus):
        super().reset(replay_log_status)
        self._current_length = 0
