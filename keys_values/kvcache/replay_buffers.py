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
from keys_values.utils import expand_index, is_index_1d, index_to_3d


@dataclass(frozen=True)
class ScatterInformation:
    index: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor

    def __post_init__(self):
        for name, ndim in (("index", 3), ("key", 4), ("value", 4)):
            if getattr(self, name).ndim != ndim:
                raise ValueError(f"{name} must have {ndim} dimensions")
        if self.key.shape[:-1] != self.index.shape:
            raise ValueError(f"index.shape = {self.index.shape}, key.shape = {self.key.shape}: Not compatible")
        if self.value.shape != self.key.shape:
            raise ValueError(f"index.shape = {self.index.shape}, value.shape = {self.value.shape}: Not compatible")

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
        index = self._get_index()
        if is_index_1d(self.index):
            keys[:, :, index, :] = self.key
            values[:, :, index, :] = self.value
        else:
            keys.scatter_(2, index, self.key)
            values.scatter_(2, index, self.value)


# TODO:
# - What about dequant_buffers._needs_write_back? [OK]
#   We don't call set_slots or _forward: Remains False.
# - What about dequant_buffers.current_length?
# - Implement updating the quant buffers (and remove replay entries)
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

    def reset(self):
        raise NotImplementedError("Cannot call 'reset' for replay buffers")

    def _allocate_buffers(
        self,
        device: Optional[torch.device] = None,
    ):
        raise NotImplementedError("Cannot call '_allocate_buffers' for replay buffers")

    def _replay_updates(self):
        if not self.dequant_buffers.buffers_are_allocated:
            raise ValueError("dequant_buffers must have allocated buffers")
        keys = self.dequant_buffers.k_buff
        values = self.dequant_buffers.v_buff
        for scat_info in self._updates:
            scat_info.apply(keys, values)
        self._last_recent_replay_len = len(self._updates)

    def _need_to_replay(self) -> bool:
        if not self.dequant_buffers._quantized_cache is self:
            return True
        return len(self._updates) != self._last_recent_replay_len

    def get_slots(
        self,
        positions: PositionsType,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.quant_buffers._assert_buffers_allocated()
        if self._need_to_replay():
            self.dequant_buffers.set_quantized_cache(self)
            self._replay_updates()
        return self.dequant_buffers.get_slots(positions)

    def set_slots(
        self,
        positions: PositionsType,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        raise NotImplementedError("Cannot call 'set_slots' for replay buffers")

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
        if isinstance(positions, tuple):
            start, end = positions
            positions = index_to_3d(
                torch.arange(start, end, dtype=torch.int32, device=key.device),
                key.shape[0],
                key.shape[1],
            )
        scat_info = ScatterInformation(
            index=positions,
            key=key,
            value=value,
        )
        scat_info.apply(
            self.dequant_buffers.k_buff, self.dequant_buffers.v_buff,
        )
        self._updates.append(scat_info)
        # We don't call `dequant_buffers.forward`, so have to adjust
        # `dequant_buffers.current_length` here:
        num = key.shape[2]
        new_current_length = min(
            self.dequant_buffers.eff_cache_length,
            self.dequant_buffers.current_length + num,
        )
        self.dequant_buffers.current_length = new_current_length
        return self.dequant_buffers.get_keys_values()

    def write_back(self):
        raise NotImplementedError("Cannot call 'set_slots' for replay buffers")

    def drop_association(self):
        self.dequant_buffers.set_quantized_cache(None)

    def get_keys_values(self) -> Optional[KeysAndValues]:
        raise NotImplementedError("Cannot call 'get_keys_values' for replay buffers")

    def _prefill(self, key: torch.Tensor, value: torch.Tensor):
        raise NotImplementedError("Cannot call '_prefill' for replay buffers")

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        raise NotImplementedError("Cannot call 'size_estimate' for replay buffers")

    @staticmethod
    def size_estimate_apriori(
        params: KVCacheBuffersParams,
        **kwargs,
    ) -> Tuple[int, Dict[str, int]]:
        raise NotImplementedError("Cannot call 'size_estimate_apriori' for replay buffers")
