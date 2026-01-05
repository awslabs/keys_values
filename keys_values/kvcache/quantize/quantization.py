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
from typing import Optional, Tuple, Dict

import torch

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.attention_utils import DEFAULT_TMP_ARRAY_LIMIT_GB
from keys_values.kvcache.buffers import KVCacheBuffersParams


class Quantizer(torch.nn.Module):
    """
    Provides quantization and de-quantization for a buffer of shape
    `shape = (batch_size, n_query_groups, cache_length, head_size)`.
    Quantized buffer and quantization states are maintained here.

    If `blocks_over_heads == False`, we use quantization blocks of
    size `head_size`, and quantization states have shape
    `(batch_size, n_query_groups, cache_length)`. If
    `blocks_over_heads == True`, we use quantization blocks of
    size `n_query_groups * head_size`, and quantization states have shape
    `(batch_size, cache_length)`.

    Deallocating and allocating buffers:

    If `allocate_buffers == True`, buffers are allocated when the object is
    created. If this is `False`, allocation is delayed. Buffers can be
    deallocated by calling :meth:`deallocate`, to save GPU memory when they
    are not needed.

    Note: Different to :class:`KVCacheBuffers` and subclasses, late buffer
    allocation or reallocation needs to be done by calling
    :meth:`_allocate_buffers` explicitly. It is not automatically done.

    As with :class:`DefaultKVCacheBuffers`, the device for buffers is undefined
    as long as they are not allocated, and can change with the next allocation.

    """
    def __init__(
        self,
        shape: Tuple[int, int, int, int],
        source_dtype: torch.dtype,
        blocks_over_heads: bool = False,
        tmp_array_limit_gb: Optional[TemporaryArrayLimit] = None,
    ):
        super().__init__()
        if len(shape) != 4 or any(x < 1 for x in shape):
            raise ValueError(f"shape = {shape}, must be 4D and all positive")
        self.shape = shape
        self.source_dtype = source_dtype
        self.blocks_over_heads = blocks_over_heads
        self._tmp_array_limit_gb = tmp_array_limit_gb

    @property
    def device(self) -> Optional[torch.device]:
        raise NotImplementedError

    @property
    def tmp_array_limit_gb(self) -> Optional[TemporaryArrayLimit]:
        return self._tmp_array_limit_gb

    def tmp_array_limit_gb_value(self) -> float:
        if self._tmp_array_limit_gb is not None:
            limit_gb = self._tmp_array_limit_gb()
        else:
            limit_gb = DEFAULT_TMP_ARRAY_LIMIT_GB
        return limit_gb

    def quantize(
        self,
        start: int,
        end: int,
        values: torch.Tensor,
    ):
        """
        Quantizes slots `range(start, end)`, overwriting corresponding parts of
        the quantized buffer and quantization states.

        Args:
            start: Determines slots `range(start, end)`
            end: Determines slots `range(start, end)`
            values: New content to be quantized,
                `(batch_size, n_query_groups, end - start, head_size)`

        """
        self._check_slots(start, end)
        self._check_shape_dtype(values, end - start, "values")
        self._quantize(start, end, values)

    def _quantize(
        self,
        start: int,
        end: int,
        values: torch.Tensor,
    ):
        raise NotImplementedError

    def dequantize(
        self,
        start: int,
        end: int,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        De-quantizes slots `range(start, end)`.

        Args:
            start: Determines slots `range(start, end)`
            end: Determines slots `range(start, end)`
            out: If given, must have shape
                `(batch_size, n_query_groups, end - start, head_size)` and
                source dtype. Result written there.

        Returns:
            De-quantized tensor of shape
            `(batch_size, n_query_groups, end - start, head_size)`. Just
            `out` if this is given.

        """
        self._check_slots(start, end)
        if out is not None:
            self._check_shape_dtype(out, end - start, "out")
        return self._dequantize(start, end, out)

    def _dequantize(
        self,
        start: int,
        end: int,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
       raise NotImplementedError

    def deallocate(self):
        """
        Deallocates the buffers. They are automatically reallocated with the
        next :meth:`prefill` call. Use this method only if device memory is
        scarce and is needed by other operations in between inference calls.

        """
        raise NotImplementedError

    @property
    def buffers_are_allocated(self) -> bool:
        """
        Returns:
            Are buffers currently allocated?

        """
        raise NotImplementedError

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        """
        Estimate of storage taken by buffers and states.

        Returns:
            num_bits_total, bits_by_part (unit is bit)

        """
        raise NotImplementedError

    @staticmethod
    def size_estimate_apriori(
        params: KVCacheBuffersParams, **kwargs,
    ) ->Tuple[int, Dict[str, int]]:
        """
        Same semantics as :meth:`size_estimate`, but can be called without a
        cache being created. Results may not be exactly the same, but should
        be very close.

        Args:
            params: KV cache buffers parameters
            **kwargs: Extra arguments (optional)

        Returns:
            num_bits_total, bits_by_part (unit is bit)

        """
        raise NotImplementedError

    def quantization_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes quantization error for `x`, without changing any content here.

        Args:
            x: Array to be quantized, shape `(batch_size, n_query_groups, num, head_size)`

        Returns:
            L2 quantization error `vector_norm(x - dequant(quant(x)), dim=-1)`.
            Here, `dtype=float32` independent of `x.dtype`.

        """
        raise NotImplementedError

    def _check_slots(self, start: int, end: int):
        cache_length = self.shape[2]
        if not (0 <= start < end <= cache_length):
            raise ValueError(f"start = {start}, end = {end}, range must be in [0, {cache_length})")

    def _check_shape_dtype(self, values: torch.Tensor, num: int, name: str):
        batch_size = values.shape[0]
        if batch_size > self.shape[0]:
            raise ValueError(f"{name}.shape[0] = {batch_size}, must be <= {self.shape[0]}")
        desired_shape = (batch_size, self.shape[1], num, self.shape[3])
        if values.shape != desired_shape:
            raise ValueError(f"{name}.shape = {values.shape}, must be {desired_shape}")
        if values.dtype != self.source_dtype:
            raise ValueError(f"{name}.dtype = {values.dtype}, must be {self.source_dtype}")

    def create_quantizer_state(
        self,
        device: torch.device,
        cache_length: Optional[int] = None,
    ) -> "QuantizerState":
        raise NotImplementedError

    @staticmethod
    def minimum_blocksize() -> int:
        raise NotImplementedError

    def allocate_buffers(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ):
        raise NotImplementedError

    @property
    def batch_size(self) -> Optional[int]:
        raise NotImplementedError


class QuantizerState:
    """
    Allows to copy the content of a :class:`Quantizer` to a different
    device.

    """
    def __init__(
        self,
        quantizer: Quantizer,
        device: Optional[torch.device] = None,
        cache_length: Optional[int] = None,
    ):
        """
        Note that the content of `quantizer` is not copied here, this needs
        an initial :meth:`copy_` call.

        Args:
            quantizer: Associated quantizer to copy from and to restore
            device: Device for buffers here. Defaults to CPU

        """
        if device is None:
            device = torch.device("cpu")
        self.quantizer = quantizer
        self.device = device
        if cache_length is None:
            cache_length = self.quantizer.shape[2]
        self.cache_length = cache_length

    def copy_(
        self,
        start: int = 0,
        end: Optional[int] = None,
    ):
        """
        Copy content from `quantizer` to the buffers here. If `start, `end`
        are given, only this slice is copied.

        """
        raise NotImplementedError

    def restore(
        self,
        start: int = 0,
        end: Optional[int] = None,
):
        """
        Restores content of `quantizer` from the buffers here. If `start, `end`
        are given, only this slice is restored.

        """
        raise NotImplementedError

    def _check_range(
        self, start: int = 0, end: Optional[int] = None,
    ) -> Tuple[int, int]:
        if end is None:
            end = self.cache_length
        if not (0 <= start < end <= self.cache_length):
            raise ValueError(f"start = {start}, end = {end}, must have 0 <= start < end <= {self.cache_length}")
        return start, end
