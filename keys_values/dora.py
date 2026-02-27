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

import torch
from torch.linalg import vector_norm
import torch.nn.functional as F

from keys_values.lora import LoRALinear, LoRAQKVLinear


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
        lora_dropout: float = 0.0,
        eps: float = 1e-9,
        **kwargs,
    ):
        super().__init__(
            in_features,
            out_features,
            r,
            lora_alpha,
            lora_dropout,
            **kwargs,
        )
        self._eps = eps
        if r > 0:
            self.scales = torch.nn.Parameter(self._init_scales())
        else:
            self.scales = None

    def _init_scales(self) -> torch.Tensor:
        return vector_norm(self.linear.weight.data, dim=-1, dtype=torch.float32) + self._eps

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.scales.data[:] = self._init_scales()

    def merge(self) -> None:
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        raise NotImplementedError("merge not implemented yet")
        if self.r > 0 and not self.merged:
            pretrained_dtype = self.linear.weight.data.dtype
            lora_data = self.get_lora_AB()
            # if only the pretrained are in quantized form - dequantize, sum with LoRA and quantize the result
            if pretrained_dtype == torch.uint8:
                import bitsandbytes as bnb

                weight = self.linear.weight
                # dequantize the pretrained weights
                weight_data = bnb.functional.dequantize_4bit(
                    weight.data, weight.quant_state
                ).to(lora_data.dtype)
                # add pretrained and LoRA weights
                weight_data += lora_data
                # assign updated weights and quantize by moving to CUDA device
                self.linear.weight = bnb.nn.Params4bit(
                    weight_data, requires_grad=False, **weight.__dict__
                )
                self.linear.weight.cuda(weight.device)
            else:
                # self.linear might be on CPU and lora_data on CUDA
                # the inplace add will preserve the dtype of linear.weight
                self.linear.weight.data += lora_data.to(
                    device=self.linear.weight.data.device
                )
            self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r == 0:
            return self.linear(x)
        # Shape (out_features, in_features)
        merged_weight = self.get_lora_AB() + self.linear.weight.data
        weight_norm = vector_norm(
            merged_weight.detach(), dim=-1, dtype=torch.float32,
        ) + self._eps
        multipliers = (self.scales / weight_norm).unsqueeze(-1).to(dtype=merged_weight.dtype)
        return F.linear(
            x, merged_weight * multipliers, self.linear.bias,
        )
