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
import math
from typing import Optional, Tuple, Dict, List, Any, Union

import torch

from keys_values.config import Config

from keys_values.kvcache.attn_weights import (
    AttnWeightsKVCache,
    AttnWeightsKVCacheState,
)
from keys_values.kvcache.base import KVCacheParams, KVCache
from keys_values.kvcache.basics import (
    KVCacheWithBuffers,
    KVCacheWithBuffersState,
)
from keys_values.kvcache.buffers import KVCacheBuffers, PositionsType


class PagedQueryAwareKVCacheState(AttnWeightsKVCacheState):
    def __init__(
        self,
        n_head: int,
        n_query_groups: int,
        head_size: int,
        max_batch_size: int,
        cache_length: int,
        block_idx: int,
        page_size: int,
        dtype: Optional[torch.dtype],
        device: Optional[torch.device],
        input_pos: int,
        replay_log_blocksize: int,
        keep_initial_fraction: float,
        prefill_length: Optional[int],
        token_pos: torch.Tensor,
        next_positions: Optional[torch.Tensor],
        avg_keys: torch.Tensor,
    ):
        super().__init__(
            n_head,
            n_query_groups,
            head_size,
            max_batch_size,
            cache_length,
            block_idx,
            dtype,
            device,
            input_pos,
            0,
            replay_log_blocksize,
            keep_initial_fraction,
            None,
            prefill_length,
            token_pos,
            next_positions,
        )
        self.page_size = page_size
        self.avg_keys = avg_keys.clone()

    def is_compatible(
        self,
        state_or_buffers: Union["KVCacheWithBuffersState", KVCacheBuffers],
    ) -> Optional[str]:
        return self._is_compatible(
            state_or_buffers,
            (
                "page_size",
                "replay_log_blocksize",
                "keep_initial_fraction",
            ),
        )


REMOVE_ARG_NAMES = ("grace_period", "detach_attn_weights")


class PagedQueryAwareKVCache(AttnWeightsKVCache):
    """
    This policy is implemented in order to run comparisons with OOMB. Given we
    do not enforce a paged memory model, using this makes little sense
    otherwise.

    Their granularity of sparsification is a page, which extends over all
    batch positions and heads, and also covers a contiguous slice of
    `page_size` tokens. Think of slicing the KV cache buffers into
    `cache_length // page_size` slices of shape
    `(batch_size, n_query_groups, page_size, head_size)` each.

    Important differences to what OOMB implements:

    * They keep the complete KV cache around in the form of pages. Then, for
      each chunk processed, they load pages into GPU memory, as selected by
      their score. Our policy overwrites the lowest scoring pages (or slices)
      when processing each chunk. Once a page (corresponding to a token range)
      is overwritten, it cannot be used in the future.
    * When processing a new chunk of size `q_len == page_size * num_q_tiles)`,
      they allow for a different set of KV pages to be used for each of the
      `num_q_tiles` tiles of the query tensor. Our policy overwrites the
      `q_len` lowest scoring slots (or `num_q_tiles` lowest scoring pages)
      in the cache before computing SDPA. The two are equivalent if
      `num_q_tiles == 1`.

    Implementation:

    Bit of a hack, since this policy (1) does not require attention weights,
    and (2) is not based on scores accumulated over chunks. Instead,
    the "avg_keys" buffer needs to be maintained.

    The index `self._next_positions` is set in :meth:`_update` (which is
    overwritten here). We also overwrite :meth:`_update_score_buffers_in_forward`
    here. This is because we need information on which buffer positions have
    been overwritten in :meth:`_update`, and also the current `query` tensor.
    """

    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        page_size: int,
        replay_log_blocksize: Optional[int] = None,
        keep_initial_fraction: Optional[float] = None,
        max_chunk_size: Optional[int] = None,
        **base_kwargs,
    ):
        """
        Additional args:
            page_size: Size of pages. Cache length must be a multiple of
                this.

        """
        super().__init__(
            config=config,
            buffers=buffers,
            block_idx=block_idx,
            grace_period=0,
            replay_log_blocksize=replay_log_blocksize,
            detach_attn_weights=False,
            keep_initial_fraction=keep_initial_fraction,
            max_chunk_size=max_chunk_size,
            **base_kwargs,
        )
        if page_size <= 0 or buffers.cache_length % page_size != 0:
            raise ValueError(
                f"page_size = {page_size}, must be positive, and cache_length = "
                f"{buffers.cache_length} must be a multiple of page_size"
            )
        self.page_size = page_size
        shape = (
            buffers.max_batch_size,
            self.n_query_groups,
            config.head_size,
            buffers.cache_length // page_size,
        )
        device = self._default_device_for_new_params()
        self.register_buffer(
            "avg_keys",
            torch.zeros(shape, device=device, dtype=torch.float32),
            persistent=False,
        )
        # Set in :meth:`_update_score_buffers_in_forward` and then used in
        # :meth:`_update`. Note that this is 1D.
        self._index_just_written_to = None
        self._query_current_forward = None

    @staticmethod
    def from_config(
        config: Config,
        max_batch_size: int,
        cache_length: int,
        block_idx: int,
        page_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        replay_log_blocksize: Optional[int] = None,
        keep_initial_fraction: Optional[float] = None,
        max_chunk_size: Optional[int] = None,
        **base_kwargs,
    ) -> "PagedQueryAwareKVCache":
        buffers_kwargs = KVCacheWithBuffers.extract_default_buffers_kwargs(base_kwargs)
        buffers = KVCacheWithBuffers.create_default_buffers(
            config=config,
            max_batch_size=max_batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
            **buffers_kwargs,
        )
        return PagedQueryAwareKVCache(
            config,
            buffers,
            block_idx,
            page_size,
            replay_log_blocksize,
            keep_initial_fraction,
            max_chunk_size,
            **base_kwargs,
        )

    @classmethod
    def _parameter_names(cls) -> List[str]:
        return super()._parameter_names() + cls._score_buffer_names()

    def _score_buffers(self) -> List[Tuple[torch.Tensor, str]]:
        return [(self.scores, "avg_keys")]

    @classmethod
    def _score_buffer_names(cls) -> List[str]:
        return ["avg_keys"]

    def update_requires_attn_weights(self) -> bool:
        return False

    def _update_score_buffers_in_forward(
        self,
        query: torch.Tensor,
        grace_period_case: bool,
        positions: Optional[PositionsType],
        index: torch.Tensor,
        init_score_values: Dict[str, torch.Tensor],
        num1: Optional[int],
    ):
        assert not grace_period_case and positions is None  # Sanity check
        assert index.ndim == 3  # Sanity check
        assert query is not None and query.ndim == 4  # Sanity check
        assert query.shape[-2] == index.shape[-1]
        self._index_just_written_to = index[:, 0, :].clone()
        self._query_current_forward = query

    def _check_index_slice_is_valid(self, rel_start: int) -> Tuple[torch.Tensor, int]:
        index = self._index_just_written_to
        query_length = index.shape[-1]
        start = index[:, rel_start]
        sz = min(self.page_size, query_length - rel_start)
        if not (start % self.page_size == 0).all().item():
            raise ValueError(
                f"Internal error: rel_start = {rel_start}. start = {start}"
                f"not multiple of page_size = {self.page_size}. _index_just_written_to:\n"
                f"{index}"
            )
        compare_to = start.unsqueeze(-1) + torch.arange(
            sz, dtype=index.dtype, device=index.device
        ).unsqueeze(0)
        if not (index[:, rel_start : (rel_start + sz)] == compare_to).all().item():
            raise ValueError(
                f"Internal error: Slice {rel_start}:{rel_start + sz} not equal to:\n"
                f"{compare_to}\n_index_just_written_to:\n"
                f"{index}"
            )
        return start, sz

    def _update(self, *args, **kwargs):
        """
        Determine `self._next_positions` for the next call of
        :meth:`forward`. In `_index_just_written_to`, we obtain the
        buffer positions to which new entries have just been written to.
        """

        index = self._index_just_written_to
        query = self._query_current_forward
        if index is None:
            raise IndexError("Internal error: _index_just_written_to not set")
        if query is None:
            raise IndexError("Internal error: _query_current_forward not set")
        assert index.ndim == 2
        assert query.ndim == 4
        assert index.shape == (query.shape[0], query.shape[2])
        batch_size, query_length = index.shape
        # Update `self.avg_keys`
        for rel_start in range(0, query_length, self.page_size):
            start, sz = self._check_index_slice_is_valid(rel_start)
            # If `query_length % page_size != 0`, we can have
            # `sz < page_size` for the final slice. But `self.avg_keys` is
            # always updated based on the full slice
            positions = (
                (
                    start.unsqueeze(-1)
                    + torch.arange(
                        self.page_size, dtype=index.dtype, device=index.device
                    ).unsqueeze(0)
                )
                .unsqueeze(1)
                .expand(-1, self.n_query_groups, -1)
            )
            keys_slice, _ = self.kv_buffers.get_slots(positions)
            new_avg_key = keys_slice.mean(dim=2)
            page_pos = start // self.page_size
            self.avg_keys[:, :, :, page_pos] = new_avg_key

        # Compute local scores, based on `query` and `avg_keys`
        # query: (batch_size, n_head, num, head_size)
        # avg_keys: (batch_size, n_query_groups, head_size, num_slices)
        query = query.contiguous().unflatten(1, (self.n_query_heads, -1)).sum(2)
        # query: (batch_size, n_query_groups, num, head_size)
        scores = (query @ self.avg_keys).softmax(dim=-1).sum([1, 2])
        # scores: (batch_size, num_slices)
        # Determine `_next_positions`.
        max_num_slices = math.ceil(self.max_chunk_size / self.page_size)
        next_slices_to_overwrite = scores.topk(
            k=max_num_slices,
            dim=-1,
            largest=False,
        )[
            1
        ]  # (batch_size, max_num_slices)
        next_positions_2d = (
            next_slices_to_overwrite.unsqueeze(-1) * self.page_size
            + torch.arange(
                self.page_size,
                dtype=next_slices_to_overwrite.dtype,
                device=next_slices_to_overwrite.device,
            ).view(1, 1, -1)
        ).view(batch_size, -1)
        self._next_positions = next_positions_2d.unsqueeze(1).expand(
            -1, self.n_query_groups, -1
        )
        # Reset
        self._index_just_written_to = None
        self._query_current_forward = None

    def _initial_scores_in_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        # (batch_size, n_query_groups, head_size, num_slices)
        num_init_slices = self.current_length // self.page_size
        # key: (batch_size, n_query_groups, current_length, head_size)
        self.avg_keys[:, :, :, :num_init_slices] = (
            key.unflatten(2, (-1, self.page_size)).sum(3).transpose(2, 3)
        )
        extra = self.current_length % self.page_size
        if extra > 0:
            start = num_init_slices * self.page_size
            self.avg_keys[:, :, :, num_init_slices] = key[
                :, :, start : (start + extra), :
            ].sum(2)

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        raise NotImplementedError()

    @classmethod
    def size_estimate_apriori(
        cls, params: KVCacheParams, **kwargs
    ) -> Tuple[int, Dict[str, int]]:
        raise NotImplementedError()

    def clone(self) -> KVCache:
        return PagedQueryAwareKVCache(**self._base_kwargs_for_clone())

    def _base_kwargs_for_clone(self) -> Dict[str, Any]:
        base_kwargs = {
            k: v
            for k, v in super()._base_kwargs_for_clone().items()
            if k not in REMOVE_ARG_NAMES
        }
        base_kwargs["page_size"] = self.page_size
        return base_kwargs

    def get_state(self) -> KVCacheWithBuffersState:
        return PagedQueryAwareKVCacheState(
            n_head=self.n_head,
            n_query_groups=self.n_query_groups,
            head_size=self.head_size,
            max_batch_size=self.max_batch_size,
            cache_length=self.cache_length,
            block_idx=self.block_idx,
            page_size=self.page_size,
            dtype=self.dtype,
            device=self.device,
            input_pos=self.input_pos,
            replay_log_blocksize=self.replay_log_blocksize,
            keep_initial_fraction=self._keep_initial_fraction,
            prefill_length=self.prefill_length,
            token_pos=self.token_pos,
            next_positions=self._next_positions,
            avg_keys=self.avg_keys,
        )

    def switch_buffers(
        self,
        new_buffers: KVCacheBuffers,
        cache_state: Optional[KVCacheWithBuffersState] = None,
    ):
        super().switch_buffers(new_buffers, cache_state)
        if cache_state is not None:
            if not isinstance(cache_state, PagedQueryAwareKVCacheState):
                raise TypeError(
                    f"type(cache_state) = {type(cache_state)}, must be PagedQueryAwareKVCacheState"
                )
            self.avg_keys.copy_(cache_state.avg_keys)
