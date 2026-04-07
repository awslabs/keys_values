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
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch

from keys_values.attention import KeysAndValues
from keys_values.kvcache.buffers import (
    KVCacheBuffers,
    PositionsType,
    KVCacheBuffersParams,
)
from keys_values.kvcache.quant_buffers import (
    QuantizedKVCacheBuffers,
    DequantizedKVCacheBuffers,
)
from keys_values.utils import expand_index, is_index_1d


@dataclass(frozen=True)
class ScatterInformation:
    index: PositionsType
    key: torch.Tensor
    value: torch.Tensor

    def __post_init__(self):
        setups = (("key", 4), ("value", 4))
        is_tuple = isinstance(self.index, tuple)
        if is_tuple:
            start, end = self.index
            if not(0 <= start < end):
                raise ValueError(f"{self.index}: Must be start < end")
        else:
            setups += (("index", 3),)
        for name, ndim in setups:
            if getattr(self, name).ndim != ndim:
                raise ValueError(f"{name} must have {ndim} dimensions")
        if not is_tuple and self.key.shape[:-1] != self.index.shape:
            raise ValueError(f"index.shape = {self.index.shape}, key.shape = {self.key.shape}: Not compatible")
        if self.value.shape != self.key.shape:
            raise ValueError(f"key.shape = {self.key.shape}, value.shape = {self.value.shape}: Must be the same")

    @property
    def head_size(self) -> int:
        return self.key.shape[-1]

    def _get_index(self) -> torch.Tensor:
        if is_index_1d(self.index):
            index = self.index[0, 0, :]
        else:
            index = expand_index(self.index, self.head_size)
        return index

    def apply(self, keys: torch.Tensor, values: torch.Tensor):
        if keys.ndim != 4 or keys.shape[:2] != self.key.shape[:2] or keys.shape[-1] != self.head_size:
            raise ValueError(f"keys.shape = {keys.shape}, key.shape = {self.key.shape}: Not compatible")
        if values.shape != keys.shape:
            raise ValueError(f"values.shape = {values.shape}, key.shape = {self.key.shape}: Not compatible")
        if isinstance(self.index, tuple):
            start, end = self.index
            keys[:, :, start:end, :] = self.key
            values[:, :, start:end, :] = self.value
        else:
            index = self._get_index()
            if is_index_1d(self.index):
                keys[:, :, index, :] = self.key
                values[:, :, index, :] = self.value
            else:
                keys.scatter_(2, index, self.key)
                values.scatter_(2, index, self.value)


class ReplayKVCacheBuffers(KVCacheBuffers):
    """
    Wrapper around :class:`QuantizedKVCacheBuffers`, drives efficient token
    generation.

    A prompt has been processed into a list of quantized cache buffers,
    which use a common :class:`DequantizedKVCacheBuffers`. As tokens are
    generated, we store the `q_len=1` updates here in terms of `scatter`
    operations. Each time the buffer contents are required, we compute them
    by de-quantization from the base buffers, then replaying all the stored
    updates in sequence.

    This works for generating a limited number of tokens. We need to loop
    through all layers for each token, but cannot store all de-quantized
    buffers in GPU memory. Small scatter operations (`q_len=1`) are faster
    than full de-quantization and quantization.

    After a certain number of tokens have been generated, it makes sense
    to update the :class:`QuantizedKVCacheBuffers` buffers.
    """

    def __init__(self, quant_buffers: QuantizedKVCacheBuffers):
        """
        Args:
            quant_buffers: Base buffers, on top of which we track additional
                small updates.
        """

        cache_length = quant_buffers.quantizer_k.shape[2]
        if not quant_buffers.buffers_are_allocated:
            raise ValueError("quant_buffers must have allocated buffers")
        super().__init__(
            quant_buffers.dequant_buffers.get_params(), cache_length,
        )
        self.quant_buffers = quant_buffers
        self._updates: List[ScatterInformation] = []
        self._last_recent_replay_len = -1

    @property
    def dequant_buffers(self) -> DequantizedKVCacheBuffers:
        return self.quant_buffers.dequant_buffers

    @property
    def device(self) -> Optional[torch.device]:
        return self.quant_buffers.device

    @property
    def buffers_are_allocated(self) -> bool:
        if not self.quant_buffers.buffers_are_allocated:
            raise ValueError("quant_buffers must have allocated buffers")
        return True

    def _replay_updates(self):
        if not self.dequant_buffers.buffers_are_allocated:
            raise ValueError("dequant_buffers must have allocated buffers")
        keys = self.dequant_buffers.k_buff
        values = self.dequant_buffers.v_buff
        for scat_info in self._updates:
            scat_info.apply(keys, values)
        self._last_recent_replay_len = len(self._updates)

    def _need_to_replay(self) -> bool:
        return (not self.dequant_buffers._quantized_cache is self) or (len(self._updates) != self._last_recent_replay_len)

    def get_slots(
        self,
        positions: PositionsType,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.quant_buffers._assert_buffers_allocated()
        if self._need_to_replay():
            self.dequant_buffers.set_quantized_cache(self)
            self._replay_updates()
        return self.dequant_buffers.get_slots(positions)

    def _forward(
        self,
        positions: PositionsType,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> KeysAndValues:
        self._assert_buffers_allocated()
        if self._need_to_replay():
            self.dequant_buffers.set_quantized_cache(self)
            self._replay_updates()
        scat_info = ScatterInformation(
            index=positions,
            key=key,
            value=value,
        )
        scat_info.apply(
            self.dequant_buffers.k_buff, self.dequant_buffers.v_buff,
        )
        self._updates.append(scat_info)
        return self.dequant_buffers.get_keys_values()

    def drop_association(self):
        self.dequant_buffers.set_quantized_cache(None)

    def update_base_buffers(self):
        """
        Updates base buffers `quant_buffers` by scatter updates in
        `_updates`. The latter list is emptied.

        """
        for scat_info in self._updates:
            self.quant_buffers.set_slots(
                scat_info.index,
                scat_info.key,
                scat_info.value,
            )
        self.quant_buffers.write_back()
        self._updates = []

    @staticmethod
    def _raise_error(name: str):
        raise NotImplementedError(f"Cannot call '{name}' for replay buffers")

    def reset(self):
        self._raise_error("reset")

    def _allocate_buffers(
        self,
        device: Optional[torch.device] = None,
    ):
        self._raise_error("_allocate_buffers")

    def set_slots(
        self,
        positions: PositionsType,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        self._raise_error("set_slots")

    def write_back(self):
        self._raise_error("write_back")

    def get_keys_values(self) -> Optional[KeysAndValues]:
        self._raise_error("get_keys_values")

    def _prefill(self, key: torch.Tensor, value: torch.Tensor):
        self._raise_error("_prefill")

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        self._raise_error("size_estimate")

    @staticmethod
    def size_estimate_apriori(
        params: KVCacheBuffersParams,
        **kwargs,
    ) -> Tuple[int, Dict[str, int]]:
        ReplayKVCacheBuffers._raise_error("size_estimate_apriori")


def create_replay_kv_buffers(
    quant_buffers_list: List[QuantizedKVCacheBuffers]
) -> List[ReplayKVCacheBuffers]:
    return [
        ReplayKVCacheBuffers(quant_buffers)
        for quant_buffers in quant_buffers_list
    ]
