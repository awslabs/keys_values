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
from typing import Union, List, Tuple, Optional

import torch

from keys_values.config import Config
from keys_values.kvcache.base import KVCache, KVCacheParams


PositionsNewEntriesType = Union[torch.Tensor, List[Tuple[torch.Tensor, int, int]]]


# TODO: size_estimate, size_estimate_apriori: Which arguments?
class KVCachePolicy:
    """
    Base class for KV cache policies.

    A policy decides on where new `key`, `value` information is written to
    in a cache. These overwrite existing slots, i.e. the current information
    there is evicted.

    It also maintains `token_positions` for the cache. Some policies receive
    summed attention weights from scaled dot product attention (SDPA).

    """
    @property
    def cache_length(self) -> int:
        """
        Returns:
            Maximum cache length.

        """
        raise NotImplementedError

    @property
    def batch_size(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def n_query_groups(self) -> Optional[int]:
        raise NotImplementedError

    def token_positions(self) -> torch.Tensor:
        """
        See :meth:`keys_values.kvcache.base.KVCache.token_positions`.

        """
        raise NotImplementedError

    def positions_new_entries(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> PositionsNewEntriesType:
        """
        Given new information `key`, `value`, `query`, `token_idx`, returns
        slot positions where new `key`, `value` is to be stored. This can be
        an index of shape `key.shape[:-1]`, with values in
        `range(cache_length)`, or a list with entries `(positions, off, num)`,
        where `positions.shape = (*key.shape[:-2], num)`, and the sum of `num`
        is equal to `key.shape[-2]`. In the latter case, the ranges
        `key[:, :, off:(off + num), :]`, `value[:, :, off:(off + num), :]`
        are to be written to `positions` sequentially.

        The arguments are the same as in
        :meth:`keys_values.kvcache.base.KVCache.forward`.

        """
        raise NotImplementedError

    def prefill(self, token_idx: torch.Tensor):
        """
        Called during prefill. Can be used to initiate replay log and token
        positions.

        """
        raise NotImplementedError

    @staticmethod
    def requires_attn_weights(self) -> bool:
        """
        Returns:
            Do we require attention weights? If `True`, `update` needs to
            be called passing the attention weights.

        """
        return False

    def pass_attn_weights(
        self,
        attn_weights: torch.Tensor,
        query_length: int,
    ):
        """
        Has to be called as part of each :meth:`forward` call if
        `requires_attn_weights` returns `True`.

        Args:
            attn_weights: Attention weights for the multi-head attention
                computation done just after the last recent :meth:`forward`
                call, summed over the query axis. Shape must be
                `(batch_size, n_query_groups, current_length)`, where
                `current_length <= cache_length` is the current cache length.
                Also, `dtype=float32`.
            query_length: Size of query axis

        """
        raise NotImplementedError

    def reset(self, batch_size: Optional[int] = None) -> None:
        """
        Called before the cache is used for a new sequence batch. All
        score buffers (if any) are reset. If the new batch has a different
        size than before, pass its `batch_size`.

        Args:
            batch_size: Size of the new batch. If not given, the previous
                batch size is used.

        """
        raise NotImplementedError()


class KVCacheDriver:
    """
    Base class for KV cache drivers.

    A driver maintains the KV buffers and performs the scaled dot product
    attention (SDPA) computation. K and V buffers have shape
    `(batch_size, n_query_groups, cache_length, head_size)`. For each
    `(b, h)`, up to `cache_length` slots can be stored. Once more than
    `cache_length` tokens are processed, slots are overwritten. Overwrite
    (or eviction) decisions are made by the KV cache logic.

    Note that KV buffers and SDPA are not separated here, but are dealt with
    together.

    """
    @property
    def config(self) -> Config:
        raise NotImplementedError

    @property
    def max_batch_size(self) -> int:
        raise NotImplementedError

    @property
    def device(self) -> Optional[torch.device]:
        """
        A KV cache is created without association to a device. Its device
        is set based on the arguments of the first :meth:`forward` call. It
        cannot be changed afterwards (use :meth:`clone` to transport a
        cache to a different device).

        Returns:
            Device the KV cache buffers are kept on, and which arguments
            to :meth:`forward` need to be on. Returns `None` as long as the
            device is not fixed (upon first :meth:`forward` call).

        """
        raise NotImplementedError

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """
        Returns:
            Data type of the KV cache buffers. This can be set at construction
            or is otherwise set with the first :meth:`forward` call. Cannot
            be changed afterwards.

        """
        raise NotImplementedError

    @property
    def cache_length(self) -> int:
        """
        Returns:
            Maximum cache length.

        """
        raise NotImplementedError

    @property
    def current_length(self) -> int:
        """
        Returns:
            Current cache length `<= cache_length`. This is equal to
            `cache_length` once `input_pos >= cache_length`.

        """
        raise NotImplementedError

    @property
    def batch_size(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def n_query_groups(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def head_size(self) -> Optional[int]:
        raise NotImplementedError

    def prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Does the job of :meth:`forward` for `input_pos == 0` (i.e., the
        first call after :meth:`reset`). We must have `current_length == 0`.
        The length of inputs must be `<= cache_length`.

        Args:
            query: Initial queries
            key: Initial keys
            value: Initial values
            token_idx: Initial token indices

        Returns:
            Attention outputs

        """
        raise NotImplementedError

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        positions: PositionsNewEntriesType,
        token_positions: torch.Tensor,
        return_attn_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward computation if `input_pos > 0`.
        First, the K and V buffers are updated for the new entries `key`,
        `value`, using overwrite positions `positions`, as determined by
        :meth:`KVCachePolicy.positions_new_entries` called with the same
        arguments. Second, run SDPA with `query` and the extended K, V
        buffers, and `token_positions`. Here, `token_positions` has already
        been updated in the last recent
        :meth:`KVCachePolicy.positions_new_entries` call. Returns the
        attention outputs.

        If `return_attn_weights == True`, the attention weights summed
        over the query axis are returned as well.

        Args:
            query: Queries for the new entries.
            key: Keys for the new entries.
            value: Values for the new entries.
            token_idx: Token indices for the new entries.
            positions: Positions for the new entries, coming from a call
                of :meth:`KVCachePolicy.positions_new_entries`.
            token_positions: Token positions for KV buffer after update,
                obtained from :meth:`KVCachePolicy.token_positions`.
            return_attn_weights: If `True`, the attention weights summed over
                the query axis are returned as well.

        Returns:
            `attn_output, attn_weights`, where `attn_weights` is `None` if
            attention weights are not returned. If given, we have
            `attn_weights.shape == (batch_size, n_query_groups, kv_len)`
            and `attn_weights.dtype == float32`, independent of `query.dtype`.
            Here, `kv_len == current_length` is the current cache length.

        """
        raise NotImplementedError

    def reset(self, batch_size: Optional[int] = None) -> None:
        """
        Resets the cache buffers so that `input_pos == 0` and
        `current_length == 0` afterwards. Needs to be called before the
        cache can be used for a new sequence. Has to be followed by a
        :meth:`prefill` call, before any :meth:`forward` call.

        Args:
            batch_size: If given, this is the new batch size. Must be
                `<= max_batch_size`. If not given, the current batch size
                will be used.

        """
        raise NotImplementedError()


class GenericKVCache(KVCache):
    """
    Generic implementation for most of the :class:`KVCache` variants supported.
    The implementation is configured by a :class:`KVCachePolicy` object and a
    :class:`KVCacheDriver` object.

    """
    def __init__(
        self,
        policy: KVCachePolicy,
        driver: KVCacheDriver,
        block_idx: int,
    ):
        self._check_compatible(policy, driver)
        super().__init__(
            driver.config,
            driver.max_batch_size,
            driver.cache_length,
            block_idx,
            driver.dtype,
            driver.head_size,
        )
        self.policy = policy
        self.driver = driver
