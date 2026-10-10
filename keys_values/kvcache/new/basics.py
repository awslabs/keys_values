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
from typing import Union, List, Tuple

import torch

from keys_values.kvcache.base import KVCache, KVCacheParams


PositionsNewEntriesType = Union[torch.Tensor, List[Tuple[torch.Tensor, int, int]]]


class KVCachePolicy:
    """
    Base class for KV cache policies.

    A policy decides on where new `key`, `value` information is written to
    in a cache. These overwrite existing slots, i.e. the current information
    there is evicted.

    It also maintains `token_positions` for the cache. Some policies receive
    summed attention weights from scaled dot product attention (SDPA).

    """
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
