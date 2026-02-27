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
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, Type
from typing_extensions import Self

import torch
import torch.nn as nn
import torch.nn.functional as F

import litgpt
from litgpt.lora import (
    LoRALinear,
    LoRAQKVLinear as BaseLoRAQKVLinear,
    create_lora_linear,
)
from litgpt.scripts.convert_hf_checkpoint import qkv_reassemble
from litgpt.utils import map_old_state_dict_weights

from keys_values.attention import MultiHeadSelfAttention
from keys_values.config import Config as BaseConfig
from keys_values.kvcache.base import KVCache
from keys_values.model import GPT as BaseModel
from keys_values.model import Block as BaseBlock
from keys_values.model import CausalSelfAttention as BaseCausalSelfAttention
from keys_values.use_eager_kernel import transform_mha_kwargs
from keys_values.utils import check_for_nan


@dataclass
class Config(BaseConfig):
    """
    Args:
        lora_r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
            the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
        lora_alpha: alpha is needed for scaling updates as alpha/r
            "This scaling helps to reduce the need to retune hyperparameters when we vary r"
            https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
        lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        lora_*: whether to apply LoRA to the specified weights or not
    """

    lora_r: int = 0
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    lora_query: bool = False
    lora_key: bool = False
    lora_value: bool = False
    lora_projection: bool = False
    lora_mlp: bool = False
    lora_head: bool = False

    @property
    def mlp_class(self) -> Type:
        return getattr(litgpt.lora, self.mlp_class_name)


class LoRAQKVLinear(BaseLoRAQKVLinear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        # ↓ this part is for pretrained weights
        in_features: int,
        # ↓ the remaining part is for LoRA
        head_size: int,
        n_head: int,
        n_query_groups: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: Union[bool, Tuple[bool, bool, bool]] = False,
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
        """
        super(LoRALinear, self).__init__(
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

        # Actual trainable parameters
        # To better understand initialization let's imagine that we have such parameters:
        # ⚬ in_features: 128 (embeddings_size)
        # ⚬ out_features: 384 (3 * embedding_size)
        # ⚬ r: 2
        # ⚬ enable_lora: [True, False, True]
        self._all_qkv_shapes = (
            # if `head_size` is explicitly specified in the config, `n_embd` (or `in_features`)
            # might not be equal to `head_size * n_head`, thus we use it directly here
            head_size * n_head,
            head_size * n_query_groups,
            head_size * n_query_groups,
        )
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                torch.empty((r * sum(enable_lora), in_features))
            )  # (4, 128)
            # qkv_shapes will be used to split a tensor with weights correctly
            self.qkv_shapes = [
                s for s, e in zip(self._all_qkv_shapes, enable_lora) if e
            ]
            self.lora_B = nn.Parameter(
                torch.empty(sum(self.qkv_shapes), r)
            )  # (256, 2))
            # Notes about shapes above
            # - self.lora_A has shape (4, 128): 4 because rank is 2 and LoRA is applied only to two matrices;
            # 128 is the input size of the x (embedding size). (4, 128) and not (128, 4) because later on in
            # F.linear function weights are automatically transposed. In addition conv1d requires channels to
            # be before seq length
            # - self.lora_B has shape (256, 2): 256 because LoRA is applied only to two matrices, so the output is
            # 128*2; 2 tells to have two channels per group for group convolution

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

        # Let's assume that:
        # ⚬ x: (64, 64, 128) or (batch_size, context_length, embedding_size)
        # ⚬ self.linear.weight: (384, 128) or (3 * embedding_size, embedding_size)
        # ⚬ self.lora_A.data: (4, 128)
        # ⚬ self.lora_B.data: (256, 2)

        # if weights are merged or LoRA is disabled (r <= 0 or all `enable_lora` are False) - it's only a regular nn.Linear forward pass;
        # otherwise in addition do the forward pass with LoRA weights and add it's output to the output from pretrained weights
        pretrained = self.linear(x)
        if do_debug:
            debug_intermediates(
                value=pretrained, postfix="_attn_qkv_pretrained",
            )
        if self.r == 0 or not any(self.enable_lora) or self.merged:
            return pretrained
        after_A = F.linear(self.lora_dropout(x), self.lora_A)  # (64, 64, 128) @ (4, 128) -> (64, 64, 4)
        # For F.conv1d:
        # ⚬ input: input tensor of shape (mini-batch, in_channels, iW)
        # ⚬ weight: filters of shape (out_channels, in_channels/groups, kW)
        after_B = self.conv1d(
            after_A.transpose(-2, -1),  # (64, 64, 4) -> (64, 4, 64)
            self.lora_B.unsqueeze(-1),  # (256, 2) -> (256, 2, 1)
        ).transpose(-2, -1)  # (64, 4, 64) @ (256, 2, 1) -> (64, 256, 64) -> (64, 64, 256)
        lora = self.zero_pad(after_B) * self.scaling  # (64, 64, 256) after zero_pad (64, 64, 384)
        if do_debug:
            debug_intermediates(
                value=after_A, postfix="_attn_qkv_after_a",
            )
            debug_intermediates(
                value=after_B, postfix="_attn_qkv_after_b",
            )
            debug_intermediates(
                value=lora, postfix="_attn_qkv_lora",
            )
        return pretrained + lora

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
        self.check_for_nan()


class GPT(BaseModel):
    # Copy & paste from :class:`model.GPT`. Note that :class:`Block` is new here.
    def __init__(self, config: Config, **mha_kwargs) -> None:
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = create_lora_linear(
            config,
            config.n_embd,
            config.padded_vocab_size,
            bias=config.lm_head_bias,
            use_r=config.lora_head,
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(
                    Block(config, block_idx) for block_idx in range(config.n_layer)
                ),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.mha = MultiHeadSelfAttention(
            config,
            **transform_mha_kwargs(mha_kwargs, config),
        )
        self.max_seq_length = self.config.block_size
        self._start_of_layer_hook = None
        # Have dense KV caches been created by `set_kv_caches`?
        self._default_kv_cache = False

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`. Unused method left for completeness."""
        super()._init_weights(module)
        if isinstance(module, LoRALinear):
            module.reset_parameters()

    def _load_from_state_dict(
        self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        """For compatibility with base checkpoints."""
        mapping = {
            "lm_head.weight": "lm_head.linear.weight",
            "lm_head.bias": "lm_head.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _empty_clone(self, device: Optional[torch.device] = None) -> "GPT":
        if device is None:
            model_copy = GPT(self.config)
        else:
            with torch.device(device):
                model_copy = GPT(self.config)
        model_copy.mha = self.mha
        return model_copy

    def check_for_nan(self, do_grads: bool = False) -> None:
        for block in self.transformer.h:
            block.attn.check_for_nan(do_grads)


class Block(BaseBlock):
    def __init__(
        self,
        config: Config,
        block_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> None:
        super().__init__(config, block_idx, kv_cache)
        self.attn = CausalSelfAttention(config, block_idx, kv_cache=kv_cache)
        self.mlp = config.mlp_class(config)


class CausalSelfAttention(BaseCausalSelfAttention):
    def __init__(
        self,
        config: Config,
        block_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> None:
        super().__init__(config, block_idx, kv_cache)
        # key, query, value projections for all heads, but in a batch
        self.qkv = LoRAQKVLinear(
            in_features=config.n_embd,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            enable_lora=(config.lora_query, config.lora_key, config.lora_value),
            bias=config.bias or config.attn_bias,
            # for MQA/GQA support
            head_size=config.head_size,
            n_head=config.n_head,
            n_query_groups=config.n_query_groups,
        )

        def qkv_apply(x: torch.Tensor, **kwargs) -> torch.Tensor:
            return self.qkv(x, **kwargs)

        self._qkv_apply = qkv_apply
        # output projection
        self.proj = create_lora_linear(
            config,
            config.head_size * config.n_head,
            config.n_embd,
            use_r=config.lora_projection,
        )

    def _load_from_state_dict(
        self, state_dict: Dict, prefix: str, *args: Any, **kwargs: Any
    ) -> None:
        """For compatibility with base and/or legacy checkpoints."""
        mapping = {
            "qkv.weight": "qkv.linear.weight",
            "qkv.bias": "qkv.linear.bias",
            "proj.weight": "proj.linear.weight",
            "proj.bias": "proj.linear.bias",
        }
        state_dict = map_old_state_dict_weights(state_dict, mapping, prefix)

        for attr in ("weight", "bias"):
            legacy_key = f"{prefix}attn.linear.{attr}"
            current_key = f"{prefix}qkv.linear.{attr}"
            if legacy_key in state_dict:
                state_dict[current_key] = qkv_reassemble(
                    state_dict.pop(legacy_key), self.config
                )

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def check_for_nan(self, do_grads: bool = False) -> None:
        self.qkv.check_for_nan(
            extra_msg=f"CausalSelfAttention: block_idx={self.block_idx}",
            do_grads=do_grads,
        )
