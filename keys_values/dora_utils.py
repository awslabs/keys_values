# Original Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# Modification Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from typing import Tuple, Union, Optional

import torch
from torch.linalg import vector_norm
import torch.nn as nn
import torch.nn.functional as F

from keys_values.lora_utils import LoRALinear, LoRAQKVLinear


DORA_SCALES_NAME = "dora_scales"


def row_lengths(x: torch.Tensor, eps: float) -> torch.Tensor:
    return vector_norm(x, dim=-1, dtype=torch.float32) + eps


def dora_forward(
    x: torch.Tensor,
    linear: nn.Linear,
    lora_ab: Optional[torch.Tensor],
    scales: Optional[nn.Parameter],
    eps: float,
) -> torch.Tensor:
    if lora_ab is None:
        return linear(x)
    # Shape (out_features, in_features)
    merged_weight = lora_ab + linear.weight.data
    # As suggested in Section 4.3 of the DoRA paper, we detach
    # `weight_norm` from the graph in order to save GPU memory
    weight_norm = row_lengths(merged_weight.detach(), eps)
    multipliers = (scales / weight_norm).unsqueeze(-1).to(dtype=merged_weight.dtype)
    return F.linear(
        x, merged_weight * multipliers, linear.bias,
    )


def dora_merge(
    linear: nn.Linear,
    lora_ab: torch.Tensor,
    scales: nn.Parameter,
    eps: float,
):
    dtype = linear.weight.data.dtype
    assert dtype != torch.uint8
    linear.weight.data += lora_ab
    weight_norm = row_lengths(linear.weight.data, eps)
    multipliers = (scales.data / weight_norm).unsqueeze(-1).to(dtype=dtype)
    linear.weight.data *= multipliers


class DoRALinear(LoRALinear):
    """
    Implements DoRA block:

    Shih-Yang Liu, etal.
    DoRA: Weight-Decomposed Low-Rank Adaptation
    https://arxiv.org/abs/2402.09353

    This is done by computing the merged linear map on the fly. As suggested
    in Section 4.3 of the paper, we detach the normalizing term from the
    compute graph.

    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        eps: float = 1e-9,
        **kwargs,
    ):
        super().__init__(
            in_features,
            out_features,
            r,
            lora_alpha,
            lora_dropout=0.0,
            **kwargs,
        )
        if self.linear.weight.data.dtype == torch.uint8:
            raise NotImplementedError("DoRA does not support quantized weights")
        self._eps = eps
        if r > 0:
            self.dora_scales = nn.Parameter(self._init_scales())
        else:
            self.dora_scales = None

    def _init_scales(self) -> torch.Tensor:
        return row_lengths(self.linear.weight.data, self._eps)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.dora_scales.data[:] = self._init_scales()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        do_lora = not(self.r == 0 or self.merged)
        return dora_forward(
            x=x,
            linear=self.linear,
            lora_ab=self.get_lora_AB() if do_lora else None,
            scales=self.dora_scales,
            eps=self._eps,
        )

    def merge(self) -> None:
        """Merges the DoRA weights into the full-rank weights (W = W + delta_W)."""
        if self.r > 0 and not self.merged:
            dora_merge(
                linear=self.linear,
                lora_ab=self.get_lora_AB(),
                scales=self.dora_scales,
                eps=self._eps,
            )
            self.merged = True


class DoRAQKVLinear(LoRAQKVLinear):
    def __init__(
        self,
        in_features: int,
        head_size: int,
        n_head: int,
        n_query_groups: int,
        r: int = 0,
        lora_alpha: int = 1,
        enable_lora: Union[bool, Tuple[bool, bool, bool]] = False,
        eps: float = 1e-9,
        **kwargs,
    ):
        super().__init__(
            in_features,
            head_size,
            n_head,
            n_query_groups,
            r,
            lora_alpha,
            0.0,
            enable_lora,
            False,
            **kwargs,
        )
        if self.linear.weight.data.dtype == torch.uint8:
            raise NotImplementedError("DoRA does not support quantized weights")
        self._eps = eps
        if r > 0 and any(self.enable_lora):
            self.dora_scales = nn.Parameter(self._init_scales())
        else:
            self.dora_scales = None

    def _init_scales(self) -> torch.Tensor:
        return row_lengths(self.linear.weight.data, self._eps)

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.dora_scales.data[:] = self._init_scales()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        do_lora = self.dora_scales is not None and not self.merged
        return dora_forward(
            x=x,
            linear=self.linear,
            lora_ab=self.get_lora_AB() if do_lora else None,
            scales=self.dora_scales,
            eps=self._eps,
        )

    def merge(self) -> None:
        """Merges the DoRA weights into the full-rank weights (W = W + delta_W)."""
        if self.dora_scales is not None and not self.merged:
            dora_merge(
                linear=self.linear,
                lora_ab=self.get_lora_AB(),
                scales=self.dora_scales,
                eps=self._eps,
            )
            self.merged = True
