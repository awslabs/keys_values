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

# Derived from https://github.com/microsoft/LoRA
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

r"""
    Low Ranking Adaptation for LLMs scheme.

             ┌───────────────────┐
             ┆         h         ┆
             └───────────────────┘
                       ▲
                       |
                       +
                    /     \
    ┌─────────────────┐    ╭───────────────╮     Matrix initialization:
    ┆                 ┆     \      B      /      B = 0
    ┆   pretrained    ┆      \    r*d    /       A = N(0, sigma^2)
    ┆    weights      ┆       ╰─────────╯
    ┆                 ┆       |    r    |        r - rank
    ┆   W e R^(d*d)   ┆       | ◀─────▶ |
    ┆                 ┆       ╭─────────╮
    └─────────────────┘      /     A     \
              ▲             /     d*r     \
               \           ╰───────────────╯
                \                ▲
                 \              /
                  \            /
             ┌───────────────────┐
             ┆         x         ┆
             └───────────────────┘

With LoRA (Low Ranking Adaptation: https://arxiv.org/abs/2106.09685) instead of learning weights of size d*d,
we can freeze the pretrained weights and instead learn two matrices of size d*r and r*d (they will store weight updates
for the pretrained weights): the number of parameters in this case will be reduced drastically (depending on the rank of
course) yet after multiplication of matrices d*r and r*d we will get a matrix d*d which we can sum with frozen
pretrained weights and thus fine-tune the model.

The goal of this approach is to move weight updates into a separate matrix which is decomposed with
two matrices of a lower rank.
"""
from typing import Any, Optional, Union, Tuple

import torch
from torch import nn as nn
from torch.nn import functional as F

from litgpt.lora import (
    LoRALinear as BaseLoRALinear,
    LoRAQKVLinear as BaseLoRAQKVLinear,
)

from keys_values.model import RMSNorm
from keys_values.utils import check_for_nan


class LoRALinear(BaseLoRALinear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        use_rms_norm: bool = False,
        **kwargs: Any,
    ):
        """LoRA wrapper around linear class.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            out_features: number of output features of the pretrained weights
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
            use_rms_norm: See :class:`Config` above

        """
        super().__init__(
            in_features,
            out_features,
            r,
            lora_alpha,
            lora_dropout,
            **kwargs,
        )
        if r > 0 and use_rms_norm:
            self.rms_norm = RMSNorm(size=self.linear.out_features)
        else:
            self.rms_norm = None

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, "rms_norm") and self.rms_norm is not None:
            self.rms_norm.reset_parameters()

    def get_lora_AB(self) -> torch.Tensor:
        """Return merged lora_A and lora_B matrices with the same shape as the pretrained weights."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def merge(self) -> None:
        """Merges the LoRA weights into the full-rank weights (W = W + delta_W)."""
        if self.rms_norm is not None:
            raise NotImplementedError("merge not implemented yet for use_rms_norm=True")
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
        # if weights are merged or rank is less or equal to zero (LoRA is disabled) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = self.linear(x)
        if self.r == 0 or self.merged:
            return pretrained
        lora = (
            self.lora_dropout(x)
            @ self.lora_A.transpose(0, 1)
            @ self.lora_B.transpose(0, 1)
        ) * self.scaling
        if self.rms_norm is not None:
            lora = self.rms_norm(lora)
        return pretrained + lora


class LoRAQKVLinear(BaseLoRAQKVLinear):
    def __init__(
        self,
        in_features: int,
        head_size: int,
        n_head: int,
        n_query_groups: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: Union[bool, Tuple[bool, bool, bool]] = False,
        use_rms_norm: bool = False,
        **kwargs: Any,
    ):
        """LoRA wrapper around linear class that is used for calculation of q, k and v matrices.

        This class has three weight matrices:
            1. Pretrained weights are stored as `self.linear.weight`
            2. LoRA A matrix as `self.lora_A`
            3. LoRA B matrix as `self.lora_B`
        Only LoRA's A and B matrices are updated, pretrained weights stay frozen.

        Args:
            in_features: number of input features of the pretrained weights
            head_size: size of a single attention head
            n_head: number of attention heads
            n_query_groups: number of query groups (see diagram in `litgpt/config.py`)
            r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
                the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
            lora_alpha: alpha is needed for scaling updates as alpha/r
                "This scaling helps to reduce the need to retune hyperparameters when we vary r"
                https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
            lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
            enable_lora: MergeLinear class is for attention mechanism where qkv are calculated with a single weight matrix. If we
                don't want to apply LoRA we can set it as False. For example if we want to apply LoRA only to `query`
                and `value` but keep `key` without weight updates we should pass `[True, False, True]`
            use_rms_norm: See :class:`Config` above

        """
        super(BaseLoRALinear, self).__init__(
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        out_features = head_size * (n_head + 2 * n_query_groups)
        self.linear = torch.nn.Linear(in_features, out_features, **kwargs)
        self.head_size = head_size
        self.n_head = n_head
        self.n_query_groups = n_query_groups
        if isinstance(enable_lora, bool):
            enable_lora = [enable_lora] * 3
        assert len(enable_lora) == 3
        self.enable_lora = enable_lora

        self._all_qkv_shapes = (
            # if `head_size` is explicitly specified in the config, `n_embd` (or `in_features`)
            # might not be equal to `head_size * n_head`, thus we use it directly here
            head_size * n_head,
            head_size * n_query_groups,
            head_size * n_query_groups,
        )
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(torch.empty((r * sum(enable_lora), in_features)))
            # qkv_shapes will be used to split a tensor with weights correctly
            self.qkv_shapes = [
                s for s, e in zip(self._all_qkv_shapes, enable_lora) if e
            ]
            self.lora_B = nn.Parameter(torch.empty(sum(self.qkv_shapes), r))
            if use_rms_norm:
                self.rms_norm = RMSNorm(size=self.linear.out_features)
            else:
                self.rms_norm = None

            # Scaling:
            # This balances the pretrained model`s knowledge and the new task-specific adaptation
            # https://lightning.ai/pages/community/tutorial/lora-llm/
            # So, set alpha to 1.0 to fully add LoRA. If the LoRA seems to have too much effect (i.e., overfitted), set
            # alpha to lower value. If the LoRA seems to have too little effect, set alpha to higher than 1.0. You can
            # tune these values to your needs. This value can be even slightly greater than 1.0!
            # https://github.com/cloneofsimo/lora
            self.scaling = self.lora_alpha / self.r

            self.reset_parameters()

    # Taken from `litgpt.lora`. We add `debug_intermediates` annotation.
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        debug_intermediates = kwargs.get("debug_intermediates")
        do_debug = debug_intermediates is not None
        pretrained = self.linear(x)
        if do_debug:
            debug_intermediates(
                value=pretrained,
                postfix="_attn_qkv_pretrained",
            )
        if self.r == 0 or not any(self.enable_lora) or self.merged:
            return pretrained
        after_A = F.linear(self.lora_dropout(x), self.lora_A)
        after_B = self.conv1d(
            after_A.transpose(-2, -1),
            self.lora_B.unsqueeze(-1),
        ).transpose(-2, -1)
        lora_mod = self.zero_pad(after_B) * self.scaling
        if do_debug:
            debug_intermediates(
                value=after_A,
                postfix="_attn_qkv_after_a",
            )
            debug_intermediates(
                value=after_B,
                postfix="_attn_qkv_after_b",
            )
            debug_intermediates(
                value=lora_mod,
                postfix="_attn_qkv_lora",
            )
        if self.rms_norm is not None:
            lora_mod = self.rms_norm(lora_mod)
            if do_debug:
                debug_intermediates(
                    value=lora_mod,
                    postfix="_attn_qkv_dora",
                )
        return pretrained + lora_mod

    def check_for_nan(
        self,
        extra_msg: Optional[str] = None,
        do_grads: bool = False,
    ):
        if hasattr(self, "lora_A"):
            check_for_nan(self.lora_A, "LoRAQKVLinear", "lora_A", extra_msg)
            check_for_nan(self.lora_B, "LoRAQKVLinear", "lora_B", extra_msg)
            if do_grads:
                if self.lora_A.grad is not None:
                    check_for_nan(
                        self.lora_A.grad, "LoRAQKVLinear", "lora_A.grad", extra_msg
                    )
                if self.lora_B.grad is not None:
                    check_for_nan(
                        self.lora_B.grad, "LoRAQKVLinear", "lora_B.grad", extra_msg
                    )

    @property
    def lora_ind(self) -> torch.Tensor:
        """Lazy creation of a buffer with LoRA indices to overcome the limitation when FSDP with meta device is used."""
        # Indices are needed to properly pad weight updates with zeros.
        if not hasattr(self, "_lora_ind"):
            off = 0
            lora_ind = []
            for enable, size in zip(self.enable_lora, self._all_qkv_shapes):
                if enable:
                    lora_ind.extend(range(off, off + size))
                off += size
            assert len(lora_ind) == sum(self.qkv_shapes)  # Sanity check
            self.register_buffer(
                "_lora_ind",
                torch.tensor(lora_ind, device=self.linear.weight.device),
                persistent=False,
            )

        return self._lora_ind

    def reset_parameters(self):
        super().reset_parameters()
        if self.rms_norm is not None:
            self.rms_norm.reset_parameters()
        self.check_for_nan()
