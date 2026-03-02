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
from functools import partial
from typing import List, Optional, Dict, Any

import torch

from keys_values.config import Config
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.quant_buffers import create_quantized_kv_buffers
from keys_values.kvcache.quantize.quantization import (
    Quantizer,
    QuantizerState,
)


class KVCacheOffloader:
    """
    Implements CPU offloading of quantized KV cache buffers during the
    forward pass.

    For a model with `N == config.n_layer` layers, we have:
    * `N` :class:`QuantizedKVCacheBuffers` cache buffers, sharing a single
      :class:`DequantizedKVCacheBuffers` object.
    * All these share a single pair `(quantizer_k, quantizer_v)` of type
      :class:`Quantizer`.
    * We maintain `2 * N` :class:`QuantizerState` objects on the CPU here.
      These contain the quantizer states for each of the layers, whereas
      the GPU states of `(quantizer_k, quantizer_v)` are for the current
      cache being used.

    Switching between quantizer states is done by a callback passed to
    the quantizer.

    Current restrictions:
    * All caches (for all layers) must have the same length. This is so
      that the quantizers can be shared as they are. This could be changed
      with more work.
    * We do not support offloading with non-quantized KV cache buffers.

    """

    def __init__(
        self,
        config: Config,
        cache_length: int,
        max_batch_size: int,
        qname: str,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        dequant_kwargs: Optional[Dict[str, Any]] = None,
        allocate_buffers: bool = False,
    ):
        """
        Creates `N == config.n_layer` :class:`QuantizedKVCacheBuffers`
        buffers, a shared pair `(quantizer_k, quantizer_v)` of quantizers
        depending on `qname`, a shared :class:`DequantizedKVCacheBuffers`
        object (all on `device`), and `N` quantization states on CPU.

        Args:
            config: Configuration of model
            cache_length: Cache length
            max_batch_size: Maximum batch size
            qname: Quantization name
            dtype: Source data type
            device: Device to use for buffers (except quantization states, which
                are stored on CPU). If not given, this is determined with
                first usage.

        """
        cache_params = KVCacheParams.from_config(
            config,
            max_batch_size,
            cache_length,
            dtype=dtype,
        )
        # Create cache buffers and quantizers
        self.cache_buffers = create_quantized_kv_buffers(
            qname=qname,
            cache_lengths=[cache_length] * config.n_layer,
            cache_params=cache_params,
            cache_kwargs=cache_kwargs,
            dequant_kwargs=dequant_kwargs,
            allocate_buffers=allocate_buffers,
            device=device,
            shared_quantizers=True,
        )
        # Create quantizer states on CPU
        cpu_device = torch.device("cpu")
        quantizer_k = self.cache_buffers[0].quantizer_k
        quantizer_v = self.cache_buffers[0].quantizer_v
        self.quantizer_states_k = [
            quantizer_k.create_quantizer_state(cpu_device)
            for _ in range(config.n_layer)
        ]
        self.quantizer_states_v = [
            quantizer_v.create_quantizer_state(cpu_device)
            for _ in range(config.n_layer)
        ]
        # Assign callbacks
        quantizer_k.assign_callback(
            partial(
                quantizer_callback,
                quantizer_states=self.quantizer_states_k,
            )
        )
        quantizer_v.assign_callback(
            partial(
                quantizer_callback,
                quantizer_states=self.quantizer_states_v,
            )
        )

    @property
    def quantizer_k(self) -> Quantizer:
        return self.cache_buffers[0].quantizer_k

    @property
    def quantizer_v(self) -> Quantizer:
        return self.cache_buffers[0].quantizer_v

    def flush(self):
        """
        Ensures that quantizers write back their current states to CPU.
        This should be called at the end of an inference run.

        """
        for quantizer, states in (
            (self.quantizer_k, self.quantizer_states_k),
            (self.quantizer_v, self.quantizer_states_v),
        ):
            states[quantizer.current_block_idx].copy_()


def quantizer_callback(
    new_block_idx: int,
    current_block_idx: int,
    quantizer_states: List[QuantizerState],
):
    if new_block_idx != current_block_idx:
        # Write back quantizer -> state
        quantizer_states[current_block_idx].copy_()
        # Restore state -> quantizer
        quantizer_states[new_block_idx].restore()
