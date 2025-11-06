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
from typing import Optional

import torch

from litgpt.config import Config

from keys_values.attention import DefaultKeysAndValues
from keys_values.kvcache.basics import LastRecentlyInsertedKVCache
from keys_values.kvcache.buffers import KVCacheBuffers


class LastRecentlyInsertedAltKVCache(LastRecentlyInsertedKVCache):
    """
    Implements :class:`LastRecentlyInsertedKVCache` in a different way. Namely,
    we call "training" `is_causal=True` MHA for every chunk. By exchanging the
    new KV information with that currently at the right end, the causal
    attention mask works out. Note that internally, we use the same buffer
    organization as in other caches.

    """
    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        **base_kwargs,
    ):
        """
        Args:
            config: Model config
            buffers: KV cache buffers to be used
        """
        super().__init__(config, buffers, block_idx, **base_kwargs)

    @staticmethod
    def from_config(
        config: Config,
        max_batch_size: int,
        cache_length: int,
        block_idx: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **base_kwargs,
    ) -> "LastRecentlyInsertedAltKVCache":
        """
        Creates KV cache with default buffers.

        Args:
            config: Model config
            max_batch_size: Maximum batch size supported
            cache_length: Number of slots (i.e., tokens) in cache
            device: Device for buffers
            dtype: Data type for buffers

        """
        tmp_cache = LastRecentlyInsertedKVCache.from_config(
            config,
            max_batch_size,
            cache_length,
            block_idx,
            device,
            dtype,
            **base_kwargs,
        )
        return LastRecentlyInsertedAltKVCache(
            config, tmp_cache.kv_buffers, block_idx, **base_kwargs,
        )

    # We overwrite :math:`forward`, but still make use of :meth:`_prefill`
    # and :meth:`forward` internally.
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        input_pos: int,
    ) -> torch.Tensor:
        self._forward_check_args(query, key, value, token_idx, input_pos)
        for_prefill = input_pos == 0
        num = query.shape[2]
        self.mha.set_seq_length(input_pos + num, device=query.device)

        current_next_position = self.next_position
        need_to_modify = False
        if for_prefill:
            self._prefill(key, value, token_idx)
            # In this case, `k_and_v` can vend both keys and values at the same
            # time.
            k_and_v = DefaultKeysAndValues(key, value)
        else:
            # We first extend the cache normally, overwriting information
            # starting from `current_next_position`
            need_to_modify = (
                self.current_length + num > self.cache_length > num and
                current_next_position + num != self.cache_length
            )
            k_and_v = self._forward(key, value, token_idx)

        if need_to_modify:
            # Rearrange KV info in `k_and_v` and extend `query`. We can then
            # call MHA in default causal mode
            old_keys = k_and_v.keys()
            old_values = k_and_v.values()
            # R1 = [start1, end1), where `key`, `value` was written to
            # R2 = [start2, end2), final `num` slots. This is where `key`,
            #      `value` need to land
            start1 = current_next_position
            end1 = start1 + num
            end2 = self.cache_length
            start2 = end2 - num
            if end1 <= start2:
                # R1 left of R2, no overlap
                new_keys = torch.cat(
                    (
                        old_keys[:, :, :start1, :],
                        old_keys[:, :, start2:, :],  # R2
                        old_keys[:, :, end1:start2, :],
                        old_keys[:, :, start1:end1, :],  # R1
                    ),
                    dim=2,
                )
                new_values = torch.cat(
                    (
                        old_values[:, :, :start1, :],
                        old_values[:, :, start2:, :],
                        old_values[:, :, end1:start2, :],
                        old_values[:, :, start1:end1, :],
                    ),
                    dim=2,
                )
            elif start1 < start2:
                # R1 left of R2, R1, R2 overlap, but R1 not split into two.
                # R1 = [R1a, R1b], R2 = [R2a, R2b], but R2a has been overwritten
                # by R1b
                new_keys = torch.cat(
                    (
                        old_keys[:, :, :start1, :],
                        old_keys[:, :, end1:, :],  # R2b
                        old_keys[:, :, start1:end1, :],  # R1
                    ),
                    dim=2,
                )
                new_values = torch.cat(
                    (
                        old_values[:, :, :start1, :],
                        old_values[:, :, end1:, :],
                        old_values[:, :, start1:end1, :],
                    ),
                    dim=2,
                )
            else:
                # R1 right of R2, splits into two
                # R1 = [R1a, R2b], R2 = [R2a, R2b], R2b overwritten by R1a
                end1b = end1 - self.cache_length
                end2a = start2 + end1b
                if end1b <= start2:
                    new_keys = torch.cat(
                        (
                            old_keys[:, :, start2:end2a, :],  # R2a
                            old_keys[:, :, end1b:start2, :],
                            old_keys[:, :, start1:, :],  # R1a
                            old_keys[:, :, :end1b, :],  # R1b
                        ),
                        dim=2,
                    )
                    new_values = torch.cat(
                        (
                            old_values[:, :, start2:end2a, :],
                            old_values[:, :, end1b:start2, :],
                            old_values[:, :, start1:, :],
                            old_values[:, :, :end1b, :],
                        ),
                        dim=2,
                    )
                else:
                    # Can happen if `num` is large. R1a overwrites R2b, but
                    # a part of R1b also overwrites a part of R2a. Say that
                    # R2* is the part of R2 not overwritten by R1, namely
                    # R2* = [end1b, start1). Then, the whole range is
                    # covered as [R1b, R2*, R1a]
                    new_keys = torch.cat(
                        (
                            old_keys[:, :, end1b:start1, :],  # R2*
                            old_keys[:, :, start1:, :],  # R1a
                            old_keys[:, :, :end1b, :],  # R1b
                        ),
                        dim=2,
                    )
                    new_values = torch.cat(
                        (
                            old_values[:, :, end1b:start1, :],
                            old_values[:, :, start1:, :],
                            old_values[:, :, :end1b, :],
                        ),
                        dim=2,
                    )
            k_and_v = DefaultKeysAndValues(new_keys, new_values)
            fill_left = torch.zeros(
                (1, 1, 1, 1), dtype=query.dtype, device=query.device,
            ).expand(*query.shape[:2], self.cache_length - num, query.shape[-1])
            query = torch.cat((fill_left, query), dim=2)
            for_prefill = True  # Use MHA as in prefill
            input_pos = 0

        # Multi-head self-attention main computation
        y, _ = self.mha(
            query=query,
            k_and_v=k_and_v,
            block_idx=self.block_idx,
            input_pos=input_pos,
            return_attn_weights=False,
            token_positions=None if for_prefill else self.token_positions(),
        )
        if need_to_modify:
            y = y[:, (-num):, :]

        return y
