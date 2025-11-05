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
from typing import Tuple, Optional

import torch

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.kvcache.quantize.pytorch import TorchBasicQuantizer
from keys_values.kvcache.utils import bits_for_torch_dtype


ALLOWED_TARGET_DTYPES = (torch.uint8, torch.uint4)


class TorchAOQuantizer(TorchBasicQuantizer):
    """
    Internal representation works much the same as for :class:`TorchBasicQuantizer`,
    but 4-bit quantization is supported here.

    Note: We follow `torchao` in storing the internal representation in `uint8`,
    using half the block size.

    """
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
        if source_dtype not in self.supported_source_dtypes():
            raise ValueError(f"source_dtype = {source_dtype} is not supported, must be in {self.supported_source_dtypes()}")
        if num_bits not in (4, 8):
            raise ValueError(f"num_bits = {num_bits}, must be 4 or 8")
        self._is_4bit = num_bits == 4
        target_dtype = torch.uint4 if self._is_4bit else torch.uint8
        super().__init__(
            shape,
            source_dtype,
            target_dtype,
            device,
            blocks_over_heads,
            allocate_buffers,
            tmp_array_limit_gb,
        )
        batch_size, n_query_groups, _, head_size = shape
        bits_per_entry = num_bits + bits_for_torch_dtype(torch.float32)
        if self._is_4bit:
            # Need another intermediate array
            bits_per_entry += bits_for_torch_dtype(torch.uint8)
        self._bytes_per_entry = (batch_size * n_query_groups * head_size / 8) * bits_per_entry

    @staticmethod
    def _validate_blocksize(blocksize):
        TorchBasicQuantizer._validate_blocksize(blocksize)
        if blocksize % 2 == 1:
            raise ValueError(f"blocksize = {blocksize} must be even")

    @property
    def _quant_buffer_blocksize(self) -> int:
        return self.blocksize if not self._is_4bit else self.blocksize // 2

    @property
    def _quant_buffer_dtype(self) -> torch.dtype:
        return torch.uint8

    def _quantization_states(
        self, values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from torchao.quantization.observer import AffineQuantizedMinMaxObserver
        from torchao.quantization.granularity import PerAxis
        from torchao.quantization.quant_primitives import MappingType

        assert (
            values.ndim == 2 and values.shape[1] == self.blocksize
        ), f"values.shape = {values.shape}, blocksize = {self.blocksize}"
        obs = AffineQuantizedMinMaxObserver(
            mapping_type=MappingType.ASYMMETRIC,
            target_dtype=self.target_dtype,
            granularity=PerAxis(axis=0),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float32,
            zero_point_dtype=torch.float32,
        )
        obs(values)
        scales, zero_points = obs.calculate_qparams()
        device = values.device
        return scales.to(device=device), zero_points.to(device=device)

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
        from torchao.dtypes import to_affine_quantized_intx_static

        quant_tensor = to_affine_quantized_intx_static(
            input_float=input_float,
            scale=scales,
            zero_point=zero_points,
            block_size=(1, self.blocksize),
            target_dtype=self.target_dtype,
        )
        int_data = quant_tensor.tensor_impl.get_plain()[0]
        if int_data.dtype != self._quant_buffer_dtype or int_data.shape != input_float.shape:
            raise NotImplementedError(
                f"int_data.dtype = {int_data.dtype} [should be {self._quant_buffer_dtype}], "
                f"int_data.shape = {int_data.shape} [should be {input_float.shape}]. "
                "Check 'torchao' sources. Maybe something changed?"
            )
        return self._compress_internal(int_data, is_4bit=self._is_4bit)

    @staticmethod
    def _uncompress_internal(x: torch.Tensor, is_4bit: bool) -> torch.Tensor:
        if is_4bit:
            return torch.cat(
                (torch.bitwise_right_shift(x, 4), torch.bitwise_or(x, 16)),
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
        from torchao.dtypes.affine_quantized_tensor import (
            get_tensor_impl_constructor,
            AffineQuantizedTensor,
        )
        from torchao.dtypes.utils import PlainLayout

        int_data = self._uncompress_internal(int_data, is_4bit=self._is_4bit)
        tensor_impl_ctr = get_tensor_impl_constructor(PlainLayout)
        tensor_impl = tensor_impl_ctr(
            int_data, scales, zero_points, PlainLayout(),
        )
        return AffineQuantizedTensor(
            tensor_impl,
            block_size=(1, self.blocksize),
            shape=int_data.shape,
            dtype=self.source_dtype,
        ).dequantize(output_dtype=self.source_dtype)

    @staticmethod
    def supported_target_dtypes() -> Tuple[torch.dtype, ...]:
        return ALLOWED_TARGET_DTYPES
