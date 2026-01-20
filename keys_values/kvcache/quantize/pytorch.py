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
from typing import Tuple, Optional, Dict

import torch
from torch.linalg import vector_norm
from torch.ao.quantization.observer import (
    PerChannelMinMaxObserver,
    ObserverBase,
)

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.kvcache.buffers import KVCacheBuffersParams
from keys_values.kvcache.quantize.quantization import (
    Quantizer,
    QuantizerState,
)
from keys_values.kvcache.utils import bitsize_of, bits_for_torch_dtype


ALLOWED_SOURCE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

MIN_BLOCKSIZE = 16


# Adapted from PyTorch: torch/ao/quantization/observer.py
class MyPerChannelMinMaxObserver(PerChannelMinMaxObserver):
    """
    Computing quantization statistics with the original
    :class:`PerChannelMinMaxObserver` can be slow. The version here does not
    allow calling :meth:`_forward` more than once, which is all we need. We
    also overwrite the constructor not to create any registered buffers.

    For our application, both quantization and in particular de-quantization
    must be very fast, which justifies pulling the basics out here.

    """

    def __init__(
        self,
        dtype: torch.dtype,
        quant_min: int,
        quant_max: int,
    ) -> None:
        """
        We do not call the superclass constructor. They create registered
        buffers on CPU, which we want to avoid here. Instead, we initialize
        members from `base_observer`.

        Args:
            dtype: Quantization dtype
            quant_min: Minimum value internal data
            quant_max: Maximum value internal data

        """
        ObserverBase.__init__(
            self,
            dtype=dtype,
            is_dynamic=False,
        )
        self.qscheme = torch.per_channel_affine
        self.reduce_range = False
        self.has_customized_qrange = False
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.eps = torch.finfo(torch.float32).eps
        self.ch_axis = 0
        self._target_dtype = torch.float32
        self.min_val = None
        self.max_val = None

    def _forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        if self.min_val is not None:
            raise IndexError("Cannot call more than once")
        y = torch.flatten(x_orig.detach(), start_dim=1)
        min_val, max_val = torch.aminmax(y, dim=1)
        # dtype casting only here, not on `y`
        self.min_val = min_val.to(dtype=self._target_dtype)
        self.max_val = max_val.to(dtype=self._target_dtype)
        return x_orig

    # Modified from PyTorch: torch/ao/quantization/observer.py:
    # - Strip out checks (which somehow make this slow)
    # - Only default case
    @torch.jit.export
    def _calculate_qparams(
        self, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quant_min, quant_max = self.quant_min, self.quant_max
        assert len(min_val.shape) > 0
        print(f"min_val = {min_val[0]}, max_val = {max_val[0]}")
        print(f"quant_min = {quant_min}, quant_max = {quant_max}")
        # Original code has this and
        #   scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        # Makes no sense to me!
        # https://github.com/pytorch/pytorch/issues/173075
        #min_val_neg = torch.clamp(min_val, max=0)
        #max_val_pos = torch.clamp(max_val, min=0)
        scale = torch.clamp(
            (max_val - min_val) / float(quant_max - quant_min),
            min=self.eps,
        )
        zero_point = quant_min - torch.round(min_val / scale).to(torch.int32)
        return scale, zero_point


# Adapted from PyTorch: torch/ao/quantization/fx/_decomposed.py
def quantize_per_channel(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    assert input.dtype == torch.float32
    assert input.ndim == 2
    num_channels = input.shape[0]
    assert scales.numel() == num_channels and zero_points.numel() == num_channels
    scales = scales.view(-1, 1)
    zero_points = zero_points.view(-1, 1)
    res = torch.clamp(
        torch.round(input * (1.0 / scales)) + zero_points, quant_min, quant_max,
    )
    return res.to(dtype)


# Adapted from PyTorch: torch/ao/quantization/fx/_decomposed.py
def dequantize_per_channel(
    input: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
) -> torch.Tensor:
    assert input.ndim == 2
    num_channels = input.shape[0]
    assert scales.numel() == num_channels and zero_points.numel() == num_channels
    scales = scales.view(-1, 1)
    zero_points = zero_points.view(-1, 1)
    res = (input - zero_points) * scales
    return res.to(dtype=torch.float32)


class TorchBasicQuantizer(Quantizer):
    """
    The quantized content is represented internally as `quant_buffer`, along
    with quantization statistics `quant_scales`, `quant_zero_points`. Here,
    `quant_buffer` uses a 8-bit or 4-bit storage type.

    Internally, 4-bit storage is implemented as 8-bit storage with half the
    blocksize, which is why the blocksize must be even in this case.

    `tmp_array_limit_gb` provides access to the maximum size of temporary
    buffers which can be used here.

    """
    def __init__(
        self,
        shape: Tuple[int, int, int, int],
        source_dtype: torch.dtype,
        num_bits: int,
        blocks_over_heads: bool = False,
        allocate_buffers: bool = False,
        device: Optional[torch.device] = None,
        tmp_array_limit_gb: Optional[TemporaryArrayLimit] = None,
    ):
        super().__init__(
            shape,
            source_dtype,
            blocks_over_heads,
            tmp_array_limit_gb,
        )
        if source_dtype not in self.supported_source_dtypes():
            raise ValueError(
                f"source_dtype = {source_dtype} is not supported, must be in {self.supported_source_dtypes()}"
            )
        if num_bits not in (4, 8):
            raise ValueError(f"num_bits = {num_bits}, must be 4 or 8")
        self._is_4bit = num_bits == 4
        self._quant_buffer_dtype = torch.uint8
        self._quant_min = 0
        self._quant_max = 2 ** num_bits - 1
        batch_size, n_query_groups, cache_length, head_size = shape
        self.max_batch_size = batch_size
        self.n_query_groups = n_query_groups
        # This is not for storage, but temporary memory for quant / de-quant:
        bits_per_entry = num_bits + bits_for_torch_dtype(torch.float32)
        if self._is_4bit:
            # Need another intermediate array
            bits_per_entry += bits_for_torch_dtype(torch.uint8)
        self._bytes_per_entry = (
            batch_size * n_query_groups * head_size / 8
        ) * bits_per_entry
        self._init_blocksize_quant_shape()
        # Allocate buffers
        self.quant_buffer = None
        self.quant_scales = None
        self.quant_zero_points = None
        self._batch_size = None
        if allocate_buffers:
            self.allocate_buffers(batch_size, device)

    @property
    def device(self) -> Optional[torch.device]:
        return self.quant_buffer.device if self.quant_buffer is not None else None

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    def _validate_blocksize(self):
        if self.blocksize < MIN_BLOCKSIZE:
            raise ValueError(
                f"blocksize = {self.blocksize}, must be at least {MIN_BLOCKSIZE}"
            )
        if self._is_4bit and self.blocksize % 2 == 1:
            raise ValueError(f"blocksize = {self.blocksize}, must be even for 4-bit quantization")

    def _init_blocksize_quant_shape(self):
        batch_size, n_query_groups, cache_length, head_size = self.shape
        self.blocksize = n_query_groups * head_size if self.blocks_over_heads else head_size
        self._validate_blocksize()
        first_dim = batch_size if self.blocks_over_heads else batch_size * n_query_groups
        final_dim = self.blocksize if not self._is_4bit else self.blocksize // 2
        self._quant_shape = (first_dim, cache_length, final_dim)

    def _buffer_dim0(self) -> Optional[int]:
        if self.blocks_over_heads:
            return self._batch_size
        else:
            return (
                None
                if self._batch_size is None
                else self.batch_size * self.n_query_groups
            )

    def allocate_buffers(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ):
        if not (0 < batch_size <= self.max_batch_size):
            raise ValueError(
                f"batch_size = {batch_size} must be in (0, {self.max_batch_size}]"
            )
        if device is None:
            if self.buffers_are_allocated:
                device = self.device
            else:
                device = torch.get_default_device()
        # Note: If buffers are allocated with batch size >= `batch_size`, they
        # are not re-allocated
        if (
            (not self.buffers_are_allocated)
            or batch_size > self.shape[0]
            or device != self.device
        ):
            self.shape = (batch_size,) + self.shape[1:]
            self._init_blocksize_quant_shape()
            shape = self._quant_shape
            self.quant_buffer = torch.empty(
                shape,
                dtype=self._quant_buffer_dtype,
                device=device,
            )
            self.quant_scales = torch.zeros(
                shape[:-1],
                dtype=torch.float32,
                device=device,
            )
            self.quant_zero_points = torch.zeros(
                shape[:-1],
                dtype=torch.int32,
                device=device,
            )
        self._batch_size = batch_size  # Effective batch size

    def deallocate(self):
        if self.buffers_are_allocated:
            del self.quant_buffer
            self.quant_buffer = None
            del self.quant_scales
            self.quant_scales = None
            del self.quant_zero_points
            self.quant_zero_points = None
            self._batch_size = None

    @property
    def buffers_are_allocated(self) -> bool:
        return self.quant_buffer is not None

    def _quantize(
        self,
        start: int,
        end: int,
        values: torch.Tensor,
    ):
        if not self.buffers_are_allocated:
            raise IndexError("Quantizer buffers are not allocated")
        if self.batch_size != values.shape[0]:
            raise ValueError(
                f"batch_size = {self.batch_size}, values.shape[0] = {values.shape[0]}. Must be equal. Use `allocate_buffers` to adjust batch_size"
            )
        num_slots = end - start
        chunk_size = self._chunk_size(num_slots)
        # `q_x` and `_values` are temporary. Their combined size needs to be
        # below `tmp_array_limit_gb`. The sizes of `scales` and
        # `zero_points` are ignored.
        curr_start = start
        for lstart in range(0, num_slots, chunk_size):
            lend = lstart + min(chunk_size, num_slots - lstart)
            csize = lend - lstart
            _values = values[:, :, lstart:lend, :].to(torch.float32)
            if self.blocks_over_heads:
                _values = _values.transpose(1, 2)
            _values = _values.reshape(-1, self.blocksize)
            scales, zero_points = self._quantization_states(_values)
            print(f"scale = {scales[0]}, zero_point = {zero_points[0]}")
            q_x = self._quantize_internal(
                input_float=_values,
                scales=scales,
                zero_points=zero_points,
            ).view(-1, csize, self._quant_shape[-1])
            scales = scales.view(-1, csize)
            zero_points = zero_points.view(-1, csize)
            dim0 = self._buffer_dim0()
            assert q_x.shape[0] == dim0  # Sanity check
            assert scales.shape[0] == dim0
            assert zero_points.shape[0] == dim0
            # Look at [curr_start, end)
            curr_end = curr_start + csize
            self.quant_buffer[:dim0, curr_start:curr_end, :] = q_x
            del q_x
            self.quant_scales[:dim0, curr_start:curr_end] = scales
            self.quant_zero_points[:dim0, curr_start:curr_end] = zero_points
            curr_start = curr_end

    def _chunk_size(self, num_slots: int) -> int:
        max_tmp_sizes_bytes = self.tmp_array_limit_gb_value() * (2**30)
        return max(
            min(num_slots, int(max_tmp_sizes_bytes / self._bytes_per_entry)),
            1,
        )

    def _quantization_states(
        self,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            values.ndim == 2 and values.shape[1] == self.blocksize
        ), f"values.shape = {values.shape}, blocksize = {self.blocksize}"
        obs = MyPerChannelMinMaxObserver(
            dtype=self._quant_buffer_dtype,
            quant_min=self._quant_min,
            quant_max=self._quant_max,
        )
        obs(values)
        scales, zero_points = obs.calculate_qparams()
        device = values.device
        return (
            scales.to(device=device),
            zero_points.to(device=device, dtype=torch.int32),
        )

    @staticmethod
    def _compress_internal(x: torch.Tensor, is_4bit: bool) -> torch.Tensor:
        if is_4bit:
            blocksize_half = x.shape[-1] // 2
            return torch.bitwise_or(
                torch.bitwise_left_shift(x[:, :blocksize_half], 4),
                x[:, blocksize_half:],
            )
        else:
            return x

    def _quantize_internal(
        self,
        input_float: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
    ) -> torch.Tensor:
        int_data = quantize_per_channel(
            input=input_float,
            scales=scales,
            zero_points=zero_points,
            quant_min=self._quant_min,
            quant_max=self._quant_max,
            dtype=self._quant_buffer_dtype,
        )
        return self._compress_internal(int_data, is_4bit=self._is_4bit)

    def _dequantize(
        self,
        start: int,
        end: int,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.buffers_are_allocated:
            raise IndexError("Quantizer buffers are not allocated")
        dim0 = self._buffer_dim0()
        q_x = self.quant_buffer[:dim0, start:end, :]
        q_blocksize = self._quant_shape[-1]
        assert q_x.shape[-1] == q_blocksize  # Sanity check
        scales = self.quant_scales[:dim0, start:end]
        zero_points = self.quant_zero_points[:dim0, start:end]
        num_slots = end - start
        chunk_size = self._chunk_size(num_slots)
        out_parts = []  # Used only if `out` not given
        for lstart in range(0, num_slots, chunk_size):
            lend = lstart + min(chunk_size, num_slots - lstart)
            csize = lend - lstart
            _out = self._dequantize_internal(
                int_data=q_x[:, lstart:lend, :].reshape(-1, q_blocksize),
                scales=scales[:, lstart:lend].reshape(-1),
                zero_points=zero_points[:, lstart:lend].reshape(-1),
            )
            if self.blocks_over_heads:
                _out = _out.reshape(
                    self.batch_size,
                    csize,
                    self.shape[1],
                    self.shape[3],
                ).transpose(1, 2)
            else:
                _out = _out.reshape(
                    self.batch_size, self.shape[1], csize, self.shape[3]
                )
            if out is not None:
                out[:, :, lstart:lend, :] = _out
                del _out
            else:
                out_parts.append(_out)
        if out is not None:
            return out
        else:
            return torch.cat(out_parts, dim=-2)

    @staticmethod
    def _uncompress_internal(x: torch.Tensor, is_4bit: bool) -> torch.Tensor:
        if is_4bit:
            return torch.cat(
                (torch.bitwise_right_shift(x, 4), torch.bitwise_and(x, 15)),
                dim=-1,
            )
        else:
            return x

    def _dequantize_internal(
        self,
        int_data: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
    ) -> torch.Tensor:
        int_data = self._uncompress_internal(int_data, is_4bit=self._is_4bit)
        return dequantize_per_channel(
            input=int_data,
            scales=scales,
            zero_points=zero_points,
        ).to(dtype=self.source_dtype)

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        if not self.buffers_are_allocated:
            raise IndexError("Buffers are not allocated. Call 'quantize' first")
        sz_buffer = bitsize_of(self.quant_buffer)
        sz_states = bitsize_of(self.quant_scales) + bitsize_of(self.quant_zero_points)
        return sz_buffer + sz_states, dict(buffer=sz_buffer, q_states=sz_states)

    @staticmethod
    def _quant_buffer_blocksize_num_channels_apriori(
        params: KVCacheBuffersParams,
        **kwargs,
    ) -> Tuple[int, int]:
        cache_length = kwargs.get("cache_length")
        if cache_length is None:
            raise IndexError("Argument 'cache_length' is missing")
        else:
            cache_length = int(cache_length)
        blocks_over_heads = kwargs.get("blocks_over_heads")
        if blocks_over_heads is None:
            raise IndexError("Argument 'blocks_over_heads' is missing")
        else:
            blocks_over_heads = bool(blocks_over_heads)
        if blocks_over_heads:
            blocksize = params.n_query_groups * params.head_size
            num_channels = params.max_batch_size * cache_length
        else:
            blocksize = params.head_size
            num_channels = params.max_batch_size * params.n_query_groups * cache_length
        return blocksize, num_channels

    @staticmethod
    def size_estimate_apriori(
        params: KVCacheBuffersParams,
        **kwargs,
    ) -> Tuple[int, Dict[str, int]]:
        blocksize, num_channels = (
            TorchBasicQuantizer._quant_buffer_blocksize_num_channels_apriori(
                params,
                **kwargs,
            )
        )
        num_bits = kwargs.get("num_bits")
        if num_bits not in (4, 8):
            raise ValueError(f"num_bits = {num_bits}, must be in (4, 8)")
        sz_buffer = blocksize * num_channels * num_bits
        sz_states = num_channels * (
            bits_for_torch_dtype(torch.float32) + bits_for_torch_dtype(torch.int32)
        )
        return sz_buffer + sz_states, dict(buffer=sz_buffer, q_states=sz_states)

    def quantization_error(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize
        _x = x.transpose(1, 2) if self.blocks_over_heads else x
        fin_shape = _x.shape
        _x = _x.reshape(-1, self.blocksize).to(torch.float32)
        scales, zero_points = self._quantization_states(_x)
        q_x = self._quantize_internal(
            _x,
            scales,
            zero_points,
        )
        dq_x = self._dequantize_internal(
            q_x,
            scales,
            zero_points,
        ).reshape(fin_shape)
        if self.blocks_over_heads:
            dq_x = dq_x.transpose(1, 2)
        return vector_norm(x - dq_x, dim=-1, dtype=torch.float32)

    def create_quantizer_state(
        self,
        device: torch.device,
        cache_length: Optional[int] = None,
    ) -> "QuantizerState":
        return TorchBasicQuantizerState(self, device, cache_length)

    @staticmethod
    def supported_source_dtypes() -> Tuple[torch.dtype, ...]:
        return ALLOWED_SOURCE_DTYPES

    @staticmethod
    def minimum_blocksize() -> int:
        return MIN_BLOCKSIZE


class TorchBasicQuantizerState(QuantizerState):
    def __init__(
        self,
        quantizer: TorchBasicQuantizer,
        device: Optional[torch.device] = None,
        cache_length: Optional[int] = None,
    ):
        if not isinstance(quantizer, TorchBasicQuantizer):
            raise ValueError(
                f"type(quantizer) = {type(quantizer)}, must be TorchBasicQuantizer"
            )
        super().__init__(quantizer, device, cache_length)
        # Create buffers
        shape = (
            quantizer._quant_shape[0],
            self.cache_length,
            quantizer._quant_shape[2],
        )
        self.quant_buffer = torch.zeros(
            shape,
            dtype=quantizer._quant_buffer_dtype,
            device=self.device,
        )
        self.quant_scales = torch.zeros(
            shape[:-1],
            dtype=torch.float32,
            device=self.device,
        )
        self.quant_zero_points = torch.zeros(
            shape[:-1],
            dtype=quantizer._quant_buffer_dtype,
            device=self.device,
        )

    def copy_(
        self,
        start: int = 0,
        end: Optional[int] = None,
    ):
        if not self.quantizer.buffers_are_allocated:
            raise IndexError("Buffers of self.quantizer are not allocated")
        start, end = self._check_range(start, end)
        # Due to changing `batch_size`, the 0 dimension may be smaller
        dim0 = self.quantizer.quant_buffer.shape[0]
        self.quant_buffer[:dim0, start:end, :].copy_(
            self.quantizer.quant_buffer[:, start:end, :],
            non_blocking=True,
        )
        self.quant_scales[:dim0, start:end].copy_(
            self.quantizer.quant_scales[:, start:end],
            non_blocking=True,
        )
        self.quant_zero_points[:dim0, start:end].copy_(
            self.quantizer.quant_zero_points[:, start:end],
            non_blocking=True,
        )

    def restore(
        self,
        start: int = 0,
        end: Optional[int] = None,
    ):
        if not self.quantizer.buffers_are_allocated:
            raise IndexError("Buffers of self.quantizer are not allocated")
        start, end = self._check_range(start, end)
        # Due to changing `batch_size`, the 0 dimension may be smaller
        dim0 = self.quantizer.quant_buffer.shape[0]
        self.quantizer.quant_buffer[:, start:end, :].copy_(
            self.quant_buffer[:dim0, start:end, :],
            non_blocking=True,
        )
        self.quantizer.quant_scales[:, start:end].copy_(
            self.quant_scales[:dim0, start:end],
            non_blocking=True,
        )
        self.quantizer.quant_zero_points[:, start:end].copy_(
            self.quant_zero_points[:dim0, start:end],
            non_blocking=True,
        )
