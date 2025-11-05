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
from typing import Optional, Tuple, Dict, Callable, List, Any

import torch

from litgpt.config import Config

from keys_values.attention import KeysAndValues
from keys_values.kvcache.base import (
    DefaultKVCacheReplayLog,
    KVCacheParams,
    KVCacheReplayLog,
)
from keys_values.kvcache.basics import KVCacheWithBuffers
from keys_values.kvcache.buffers import (
    DefaultKVCacheBuffers,
    KVCacheBuffers,
    KVCacheBuffersParams,
    positions_wrap_around,
)
from keys_values.kvcache.utils import bitsize_of, bits_for_torch_dtype


class LastRecentlyInsertedAltKVCache(KVCacheWithBuffers):
    """
    Implements :class:`LastRecentlyInsertedKVCache` in a different way. Namely,
    buffer slots have the same ordering as tokens, and we use square
    `is_causal=True` MHA for every chunk.

    For now, this is just to check whether this is competitive for not too
    large chunk sizes. This class does not yet work with gradient computation!

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
        # May not be really needed:
        self.register_buffer(
            "token_pos",
            torch.zeros(buffers.cache_length, device=buffers.device, dtype=torch.int),
            persistent=False,
        )
        self._next_token_pos = None

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
        buffers_kwargs = KVCacheWithBuffers.extract_default_buffers_kwargs(base_kwargs)
        buffers = KVCacheWithBuffers.create_default_buffers(
            config=config,
            max_batch_size=max_batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
            **buffers_kwargs,
        )
        return LastRecentlyInsertedAltKVCache(
            config, buffers, block_idx, **base_kwargs,
        )

    @property
    def next_token_pos(self) -> Optional[int]:
        return self._next_token_pos

    @property
    def max_prefill_length(self) -> Optional[int]:
        return None  # This KV cache can be prefilled with any length

    @property
    def max_tokens_forward(self) -> int:
        return self.cache_length

    def _forward_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> KeysAndValues:
        if self._next_token_pos is None:
            raise IndexError("Cache needs to be initialized with 'prefill' before being used")
        num = key.shape[2]
        positions = positions_wrap_around(
            num=num,
            current=self.next_position,
            start=0,
            end=self.cache_length,
            batch_size=self.batch_size,
            n_query_groups=self.n_query_groups,
            device=self.device,
        )
        k_and_v = self.kv_buffers.forward(
            positions=positions,
            key=key,
            value=value,
        )
        np = self.next_position
        num1 = min(num, self.cache_length - np)
        diff = num - num1
        ntp = self._next_token_pos
        self.token_pos[np:(np + num1)] = torch.arange(
            ntp, ntp + num1, device=self.device, dtype=torch.int
        )
        if diff > 0:
            self.token_pos[:diff] = torch.arange(
                ntp + num1, ntp + num, device=self.device, dtype=torch.int
            )
        self.next_position = (np + num) % self.cache_length
        if self._replay_log is not None:
            if not isinstance(self._replay_log, DefaultKVCacheReplayLog):
                raise IndexError("Cannot switch on replay logging in the middle of inference run. Call 'prefill'.")
            self._replay_log.append_token_chunk(token_idx)
        self._next_token_pos += num
        return k_and_v

    def _update(self, *args, **kwargs):
        pass

    def _prefill_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        init_length = key.shape[2]
        eff_init_length = min(init_length, self.cache_length)
        if eff_init_length < init_length:
            key = key[:, :, -eff_init_length:, :]
            value = value[:, :, -eff_init_length:, :]
        self.kv_buffers.prefill(key, value)
        self._next_token_pos = init_length
        self.token_pos[:eff_init_length] = torch.arange(
            init_length - eff_init_length,
            init_length,
            dtype=self.token_pos.dtype,
            device=self.device,
        )

    def token_positions(self) -> torch.Tensor:
        return self.token_pos[:self.current_length].reshape(1, 1, -1).expand(
            self.batch_size, self.n_query_groups, -1
        )

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        tk_p = bitsize_of(self.token_pos)
        sz_total, dct_sz = self.kv_buffers.size_estimate()
        return sz_total + tk_p, {**dct_sz, "token_pos": tk_p}

    @classmethod
    def size_estimate_apriori(cls, params: KVCacheParams, **kwargs) -> Tuple[int, Dict[str, int]]:
        """
        `cache_length` is required in `kwargs`. If `buffer_type` is given in
        `kwargs`, the size for this type is used, otherwise for the default
        type `DefaultKVCacheBuffers`.

        """
        buff_params = KVCacheBuffersParams.from_params(params)
        buffer_type = kwargs.get("buffer_type", DefaultKVCacheBuffers)
        sz_total, dct_sz = buffer_type.size_estimate_apriori(
            buff_params, cache_length=params.cache_length, **kwargs,
        )
        tk_p = params.cache_length * bits_for_torch_dtype(torch.int)
        return sz_total + tk_p, {**dct_sz, "token_pos": tk_p}

    def switch_replay_logging(self, status: bool):
        if status:
            raise NotImplementedError("Replay logging not yet supported")

    @property
    def do_replay_logging(self) -> bool:
        return False

    def get_replay_log(self) -> Optional[KVCacheReplayLog]:
        return None
