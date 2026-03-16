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
from functools import partial

import torch
from torch.linalg import vector_norm

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.kvcache.buffers import KVCacheBuffersParams
from keys_values.kvcache.quantize.quantization import (
    Quantizer,
    QuantizerState,
)
from keys_values.kvcache.utils import bitsize_of, bits_for_torch_dtype


ALLOWED_BLOCK_SIZE = (64, 128, 256, 512, 1024, 2048, 4096)

ALLOWED_SOURCE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def determine_blocksize(shape: Tuple[int, ...]) -> Optional[Tuple[int, int]]:
    batch_size, n_query_groups, _, head_size = shape
    a = batch_size * n_query_groups * head_size
    blocksize = None
    for _blocksize in reversed(ALLOWED_BLOCK_SIZE):
        if a % _blocksize == 0 and a >= _blocksize:
            blocksize = _blocksize
            break
    if blocksize is None:
        return None
    else:
        return blocksize, a // blocksize


class BitsAndBytesQuantizer(Quantizer):
    def __init__(
        self,
        shape: Tuple[int, int, int, int],
        source_dtype: torch.dtype,
        num_bits: int,
        device: Optional[torch.device] = None,
        blocks_over_heads: bool = False,
        allocate_buffers: bool = False,
        tmp_array_limit_gb: Optional[TemporaryArrayLimit] = None,
    ):
        """
        For this quantizer, the blocksize must lie in :const:`ALLOWED_BLOCK_SIZE`,
        which constrains us a bit more.

        If `blocks_over_heads == False`, we try `blocksize = head_size` first.
        If this does not work, we use `_determine_blocksize` to choose the
        blocksize. If `blocks_over_heads == True`, we do this immediately.

        For this quantizer, if `self.batch_size < self.shape[0]`, we still
        quantize and dequantize the full buffers, but then only use the slices
        according to `self.batch_size`.

        `tmp_array_limit_gb` provides access to the maximum size of temporary
        buffers which can be used here.

        """
        super().__init__(
            shape, source_dtype, blocks_over_heads, tmp_array_limit_gb,
        )
        if source_dtype not in self.supported_source_dtypes():
            raise ValueError(f"source_dtype = {source_dtype} is not supported, must be in {self.supported_source_dtypes()}")
        if num_bits not in (4, 8):
            raise ValueError(f"num_bits = {num_bits}, must be 4 or 8")
        self._four_bits = num_bits == 4
        self.target_dtype = torch.uint8
        batch_size, n_query_groups, cache_length, head_size = shape
        self.max_batch_size = batch_size
        if head_size % 2 == 1:
            raise ValueError(f"head_size {head_size}, must be even")
        bits_per_entry = num_bits + 2 * bits_for_torch_dtype(torch.float32)
        self._bytes_per_entry = (batch_size * n_query_groups * head_size / 8) * bits_per_entry
        self._device = device
        self._init_blocksize_quant_shape()
        # Allocate buffers (optional)
        self.quant_buffer = None
        self.quant_absmax = None
        self._batch_size = None
        if allocate_buffers:
            self.allocate_buffers(batch_size)
        self._quant_code = None
        self._initialize()

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    def _init_blocksize_quant_shape(self):
        batch_size, n_query_groups, cache_length, head_size = self.shape
        fin_denom = 2 if self._four_bits else 1
        done = False
        blocks_over_heads = self.blocks_over_heads
        while not done:
            if blocks_over_heads:
                # If `blocks_over_head == True`, we need transposes anyway, so
                # we keep the slots in the left-most dimension, where they are
                # easiest to handle.
                self.blocksize, reminder = self._determine_blocksize()
                self._quant_shape = (cache_length, reminder, self.blocksize // fin_denom)
                self.blocks_over_heads = True
            else:
                self.blocksize = head_size
                self._quant_shape = (batch_size * n_query_groups, cache_length, self.blocksize // fin_denom)
            if self.blocksize in ALLOWED_BLOCK_SIZE:
                done = True
            elif not blocks_over_heads:
                print(f"blocksize = {self.blocksize} not supported. Trying with blocks_over_heads=True.")
                blocks_over_heads = True

    @property
    def device(self) -> torch.device:
        if self.quant_buffer is not None:
            self._device = self.quant_buffer.device
        return self._device

    def _determine_blocksize(self) -> Tuple[int, int]:
        result = determine_blocksize(self.shape)
        if result is None:
            a = self.shape[0] * self.shape[1] * self.shape[3]
            raise ValueError(
                f"Cannot find blocksize for shape = {self.shape}: "
                f"a = {a} must be divisible by one of:\n"
                f"{ALLOWED_BLOCK_SIZE}"
            )
        return result

    def allocate_buffers(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ):
        if not (0 < batch_size <= self.max_batch_size):
            raise ValueError(f"batch_size = {batch_size} must be in (0, {self.max_batch_size}]")
        if device is None:
            device = self.device
        if not self.buffers_are_allocated or batch_size > self.shape[0] or device != self.device:
            self.shape = (batch_size,) + self.shape[1:]
            self._init_blocksize_quant_shape()
            shape = self._quant_shape
            self.quant_buffer = torch.zeros(
                shape, dtype=self.target_dtype, device=device,
            )
            self.quant_absmax = torch.zeros(
                shape[:-1], dtype=torch.float32, device=device,
            )
        self._batch_size = batch_size  # Effective batch size

    def _initialize(self):
        quant_func = self._quantize_func()
        x = torch.arange(self.blocksize, dtype=self.source_dtype, device=self.device)
        _, quant_state = quant_func(x)
        self._quant_code = quant_state.code

    def deallocate(self):
        if self.buffers_are_allocated:
            del self.quant_buffer
            self.quant_buffer = None
            del self.quant_absmax
            self.quant_absmax = None
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
            raise ValueError(f"batch_size = {self.batch_size}, values.shape[0] = {values.shape[0]}. Must be equal. Use `allocate_buffers` to adjust batch_size")
        num_slots = end - start
        chunk_size = self._chunk_size(num_slots)
        quant_func = self._quantize_func()
        final_dim = self.quant_buffer.shape[-1]
        # `q_x` and `_values` are temporary. The complexity here is to keep them
        # below :const:`MAX_TEMP_SIZE_IN_BYTES` bytes. The sizes of `scales`
        # and `zero_points` are ignored.
        curr_start = start
        for lstart in range(0, num_slots, chunk_size):
            lend = lstart + min(chunk_size, num_slots - lstart)
            csize = lend - lstart
            if self.batch_size == self.shape[0]:
                _values = values[:, :, lstart:lend, :]
            else:
                add_me = self.shape[0] - self.batch_size
                assert add_me > 0
                add_shape = (add_me, values.shape[1], csize, values.shape[3])
                _values = torch.cat(
                    (
                        values[:, :, lstart:lend, :],
                        torch.zeros(add_shape, dtype=values.dtype, device=values.device),
                    ),
                    dim=0,
                )
            if self.blocks_over_heads:
                _values = _values.transpose(0, 2)
            _values = _values.reshape(
                -1, self.blocksize,
            ).contiguous()
            q_x, quant_state = quant_func(_values)
            curr_end = curr_start + csize
            if self.blocks_over_heads:
                q_x = q_x.view(csize, -1, final_dim)
                assert q_x.shape[1] == self.quant_buffer.shape[1], (q_x.shape, self.quant_buffer.shape)
                absmax = quant_state.absmax.view(csize, -1)
                assert absmax.shape[-1] == self.quant_absmax.shape[-1]
                # Look at [curr_start, curr_end)
                self.quant_buffer[curr_start:curr_end, :, :] = q_x
                self.quant_absmax[curr_start:curr_end, :] = absmax
            else:
                q_x = q_x.view(-1, csize, final_dim)
                assert q_x.shape[0] == self.quant_buffer.shape[0], (q_x.shape, self.quant_buffer.shape)
                absmax = quant_state.absmax.view(-1, csize)
                assert absmax.shape[0] == self.quant_absmax.shape[0]
                # Look at [curr_start, curr_end)
                self.quant_buffer[:, curr_start:curr_end, :] = q_x
                self.quant_absmax[:, curr_start:curr_end] = absmax
            del q_x
            curr_start = curr_end

    def _dequantize(
        self,
        start: int,
        end: int,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.buffers_are_allocated:
            raise IndexError("Quantizer buffers are not allocated")
        if self.blocks_over_heads:
            q_x = self.quant_buffer[start:end, :, :]
            absmax = self.quant_absmax[start:end, :]
        else:
            q_x = self.quant_buffer[:, start:end, :]
            absmax = self.quant_absmax[:, start:end]
        num_slots = end - start
        chunk_size = self._chunk_size(num_slots)
        final_dim = self.quant_buffer.shape[-1]
        dequant_func = self._dequantize_func()
        out_parts = []  # Used only if `out` not given
        for lstart in range(0, num_slots, chunk_size):
            lend = lstart + min(chunk_size, num_slots - lstart)
            csize = lend - lstart
            if self.blocks_over_heads:
                quant_state = self._get_quantstate(
                    absmax=absmax[lstart:lend, :],
                    shape=(self.quant_buffer.shape[1] * csize, self.blocksize),
                )
                qq_x = q_x[lstart:lend, :, :].reshape(-1, final_dim).contiguous()
            else:
                quant_state = self._get_quantstate(
                    absmax=absmax[:, lstart:lend],
                    shape=(self.quant_buffer.shape[0] * csize, self.blocksize),
                )
                qq_x = q_x[:, lstart:lend, :].reshape(-1, final_dim).contiguous()
            _out = dequant_func(qq_x, quant_state=quant_state)
            del qq_x
            if self.blocks_over_heads:
                _out = _out.reshape(
                    csize, self.shape[1], self.shape[0], self.shape[3],
                ).transpose(0, 2)
            else:
                _out = _out.reshape(*self.shape[:2], csize, self.shape[3])
            if out is not None:
                out[:, :, lstart:lend, :] = _out[:self.batch_size, ...]
                del _out
            else:
                out_parts.append(_out[:self.batch_size, ...])
        if out is not None:
            return out
        else:
            return torch.cat(out_parts, dim=-2)

    def _chunk_size(self, num_slots: int) -> int:
        max_tmp_sizes_bytes = self.tmp_array_limit_gb_value() * (2 ** 30)
        return max(
            min(num_slots, int(max_tmp_sizes_bytes / self._bytes_per_entry)),
            1,
        )

    def _quantize_func(self) -> callable:
        if self._four_bits:
            from bitsandbytes.functional import quantize_4bit

            quant_func = partial(quantize_4bit, blocksize=self.blocksize)
        else:
            from bitsandbytes.functional import quantize_blockwise
            quant_func = partial(quantize_blockwise, blocksize=self.blocksize)

        return quant_func

    def _dequantize_func(self) -> callable:
        if self._four_bits:
            from bitsandbytes.functional import dequantize_4bit
            dequant_func = partial(dequantize_4bit, blocksize=self.blocksize)
        else:
            from bitsandbytes.functional import dequantize_blockwise
            dequant_func = partial(dequantize_blockwise, blocksize=self.blocksize)

        return dequant_func

    def _get_quantstate(
        self, absmax: torch.Tensor, shape: Tuple[int, ...],
    ):
        from bitsandbytes.functional import QuantState

        return QuantState(
            absmax.flatten(),
            shape=shape if self._four_bits else None,
            code=self._quant_code,
            blocksize=self.blocksize,
            quant_type="fp4" if self._four_bits else None,
            dtype=self.source_dtype,
        )

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        if not self.buffers_are_allocated:
            raise IndexError("Buffers are not allocated. Call 'quantize' first")
        sz_buffer = bitsize_of(self.quant_buffer)
        sz_states = bitsize_of(self.quant_absmax)
        return sz_buffer + sz_states, dict(buffer=sz_buffer, q_states=sz_states)

    @staticmethod
    def size_estimate_apriori(
        params: KVCacheBuffersParams, **kwargs,
    ) -> Tuple[int, Dict[str, int]]:
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
        source_dtype = params.dtype
        if source_dtype is None:
            raise IndexError("Argument 'params.dtype' must be given")
        else:
            assert isinstance(source_dtype, torch.dtype)
        num_bits = kwargs.get("num_bits")
        if num_bits is None:
            raise IndexError("Argument 'num_bits' is missing")
        else:
            num_bits = int(num_bits)
            if num_bits not in (4, 8):
                raise ValueError("Argument 'num_bits' must be either 4 or 8")
        if blocks_over_heads:
            blocksize = params.n_query_groups * params.head_size
            num_channels = params.max_batch_size * cache_length
        else:
            blocksize = params.head_size
            num_channels = params.max_batch_size * params.n_query_groups * cache_length
        sz_buffer = blocksize * num_channels * num_bits
        sz_states = num_channels * bits_for_torch_dtype(torch.float32)
        return sz_buffer + sz_states, dict(buffer=sz_buffer, q_states=sz_states)

    def quantization_error(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize
        quant_func = self._quantize_func()
        dequant_func = self._dequantize_func()
        if self.blocks_over_heads:
            _x = x.transpose(0, 2)
            q_x, state = quant_func(_x.reshape(-1, self.blocksize).contiguous())
            dq_x = dequant_func(q_x, quant_state=state).view_as(_x).transpose(0, 2)
        else:
            q_x, state = quant_func(x.reshape(-1, self.blocksize).contiguous())
            dq_x = dequant_func(q_x, quant_state=state).view_as(x)
        return vector_norm(x - dq_x, dim=-1, dtype=torch.float)

    def create_quantizer_state(
        self,
        device: torch.device,
        cache_length: Optional[int] = None,
    ) -> "QuantizerState":
        return BitsAndBytesQuantizerState(self, device, cache_length)

    @staticmethod
    def supported_source_dtypes() -> Tuple[torch.dtype, ...]:
        return ALLOWED_SOURCE_DTYPES

    @staticmethod
    def minimum_blocksize() -> int:
        return min(ALLOWED_BLOCK_SIZE)

    @staticmethod
    def supported_blocksizes() -> Tuple[int, ...]:
        return ALLOWED_BLOCK_SIZE


class BitsAndBytesQuantizerState(QuantizerState):
    def __init__(
        self,
        quantizer: BitsAndBytesQuantizer,
        device: Optional[torch.device] = None,
        cache_length: Optional[int] = None,
    ):
        if not isinstance(quantizer, BitsAndBytesQuantizer):
            raise ValueError(f"type(quantizer) = {type(quantizer)}, must be BitsAndBytesQuantizer")
        super().__init__(quantizer, device, cache_length)
        # Create buffers
        shape = list(quantizer._quant_shape)
        pos = 0 if self.quantizer.blocks_over_heads else 1
        shape[pos] = self.cache_length
        self.quant_buffer = torch.zeros(
            shape, dtype=quantizer.target_dtype, device=device,
        )
        self.quant_absmax = torch.zeros(
            shape[:-1], dtype=torch.float32, device=device,
        )

    def copy_(
        self,
        start: int = 0,
        end: Optional[int] = None,
    ):
        if not self.quantizer.buffers_are_allocated:
            raise IndexError("Buffers of self.quantizer are not allocated")
        start, end = self._check_range(start, end)
        # Due to changing `batch_size`, the dimension may be smaller
        if self.quantizer.blocks_over_heads:
            dim1 = self.quantizer.quant_buffer.shape[1]
            self.quant_buffer[start:end, :dim1, :] = self.quantizer.quant_buffer[start:end, :, :]
            self.quant_absmax[start:end, :dim1] = self.quantizer.quant_absmax[start:end, :]
        else:
            dim0 = self.quantizer.quant_buffer.shape[0]
            self.quant_buffer[:dim0, start:end, :] = self.quantizer.quant_buffer[:, start:end, :]
            self.quant_absmax[:dim0, start:end] = self.quantizer.quant_absmax[:, start:end]

    def restore(
        self,
        start: int = 0,
        end: Optional[int] = None,
    ):
        if not self.quantizer.buffers_are_allocated:
            raise IndexError("Buffers of self.quantizer are not allocated")
        start, end = self._check_range(start, end)
        # Due to changing `batch_size`, the 0 dimension may be smaller
        if self.quantizer.blocks_over_heads:
            dim1 = self.quantizer.quant_buffer.shape[1]
            self.quantizer.quant_buffer[start:end, :, :] = self.quant_buffer[start:end, :dim1, :]
            self.quantizer.quant_absmax[start:end, :] = self.quant_absmax[start:end, :dim1]
        else:
            dim0 = self.quantizer.quant_buffer.shape[0]
            self.quantizer.quant_buffer[:, start:end, :] = self.quant_buffer[:dim0, start:end, :]
            self.quantizer.quant_absmax[:, start:end] = self.quant_absmax[:dim0, start:end]
