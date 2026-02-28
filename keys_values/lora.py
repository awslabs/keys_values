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
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Type, Literal, Callable, Union
from typing_extensions import Self

import torch
import torch.nn as nn

import litgpt
from litgpt.lora import mark_only_lora_as_trainable as litgpt_mark_only_lora_as_trainable
from litgpt.scripts.convert_hf_checkpoint import qkv_reassemble
from litgpt.utils import map_old_state_dict_weights

from keys_values.attention import MultiHeadSelfAttention
from keys_values.config import Config as BaseConfig
from keys_values.dora_utils import DoRALinear, DoRAQKVLinear, DORA_SCALES_NAME
from keys_values.kvcache.base import KVCache
from keys_values.lora_utils import LoRALinear, LoRAQKVLinear
from keys_values.model import (
    GPT as BaseModel,
    Block as BaseBlock,
    CausalSelfAttention as BaseCausalSelfAttention,
)
from keys_values.use_eager_kernel import transform_mha_kwargs


@dataclass
class Config(BaseConfig):
    """
    `lora_kind` selects different variants:

    * "default": Standard LoRA as implemented in `LitGPT`.
    * "rms_norm": Modification suggested by Sebastian Raschka:
        https://github.com/rasbt/dora-from-scratch/blob/main/Using-LinearDoRA.ipynb
        He calls this DoRA, but the modification is simpler, runs faster, but
        may work less well.
    * "dora": DoRA, see :class:`keys_values.dora.DoRALinear`. Note that
        `lora_dropout` is ignored for this variant

    Args:
        lora_r: rank of the weight update matrices. To make sense of using LoRA the rank should be smaller than the rank of
            the weights of the model. The rank can be as low as 1: https://arxiv.org/pdf/2106.09685.pdf (section 7.2)
        lora_alpha: alpha is needed for scaling updates as alpha/r
            "This scaling helps to reduce the need to retune hyperparameters when we vary r"
            https://arxiv.org/pdf/2106.09685.pdf (section 4.1)
        lora_dropout: dropout that is applied on the input in the LoRA branch (before multiplying by matrix A)
        lora_*: whether to apply LoRA to the specified weights or not
        lora_kind: See above

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
    lora_kind: Literal["default", "rms_norm", "dora"] = "default"

    @property
    def mlp_class(self) -> Type:
        return getattr(litgpt.lora, self.mlp_class_name)


def create_lora_linear(
    config: Config,
    in_size: int,
    out_size: int,
    bias: Optional[Union[float, bool]] = None,
    use_r: Optional[bool] = None,
    use_rms_norm: bool = False,
) -> LoRALinear:
    if bias is None:
        bias = config.bias
    if use_r is None:
        use_r = config.lora_mlp
    return LoRALinear(
        in_size,
        out_size,
        bias=bias,
        r=(config.lora_r if use_r else 0),
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        use_rms_norm=use_rms_norm,
    )


def create_dora_linear(
    config: Config,
    in_size: int,
    out_size: int,
    bias: Optional[Union[float, bool]] = None,
    use_r: Optional[bool] = None,
    eps: float = 1e-9,
) -> DoRALinear:
    if bias is None:
        bias = config.bias
    if use_r is None:
        use_r = config.lora_mlp
    return DoRALinear(
        in_size,
        out_size,
        bias=bias,
        r=(config.lora_r if use_r else 0),
        lora_alpha=config.lora_alpha,
        eps=eps,
    )


def create_lm_head(config: Config) -> nn.Module:
    if config.lora_kind != "dora":
        return create_lora_linear(
            config,
            config.n_embd,
            config.padded_vocab_size,
            bias=config.lm_head_bias,
            use_r=config.lora_head,
            use_rms_norm=config.lora_kind == "rms_norm",
        )
    else:
        return create_dora_linear(
            config,
            config.n_embd,
            config.padded_vocab_size,
            bias=config.lm_head_bias,
            use_r=config.lora_head,
        )


class GPT(BaseModel):
    # Copy & paste from :class:`model.GPT`. Note that :class:`Block` is new here.
    def __init__(self, config: Config, **mha_kwargs) -> None:
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config
        self.lm_head = create_lm_head(config)
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


def create_qkv_and_proj(
    config: Config,
) -> Tuple[nn.Module, Callable, nn.Module]:
    if config.lora_kind != "dora":
        use_rms_norm = config.lora_kind == "rms_norm"
        qkv = LoRAQKVLinear(
            in_features=config.n_embd,
            head_size=config.head_size,
            n_head=config.n_head,
            n_query_groups=config.n_query_groups,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            enable_lora=(config.lora_query, config.lora_key, config.lora_value),
            bias=config.bias or config.attn_bias,
            use_rms_norm=use_rms_norm,
        )

        def qkv_apply(x: torch.Tensor, **kwargs) -> torch.Tensor:
            return qkv(x, **kwargs)

        proj = create_lora_linear(
            config,
            config.head_size * config.n_head,
            config.n_embd,
            use_r=config.lora_projection,
            use_rms_norm=use_rms_norm,
        )
    else:
        qkv = DoRAQKVLinear(
            in_features=config.n_embd,
            head_size=config.head_size,
            n_head=config.n_head,
            n_query_groups=config.n_query_groups,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            enable_lora=(config.lora_query, config.lora_key, config.lora_value),
            bias=config.bias or config.attn_bias,
        )

        def qkv_apply(x: torch.Tensor, **kwargs) -> torch.Tensor:
            return qkv(x)

        proj = create_dora_linear(
            config,
            config.head_size * config.n_head,
            config.n_embd,
            use_r=config.lora_projection,
        )

    return qkv, qkv_apply, proj


class CausalSelfAttention(BaseCausalSelfAttention):
    def __init__(
        self,
        config: Config,
        block_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> None:
        super().__init__(config, block_idx, kv_cache)
        self.qkv, self._qkv_apply, self.proj = create_qkv_and_proj(config)

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


def mark_only_lora_as_trainable(
    model: nn.Module,
    lora_kind: str = "default",
    bias: str = "none",
):
    litgpt_mark_only_lora_as_trainable(model, bias)
    if lora_kind == "dora":
        for name, param in model.named_parameters():
            if DORA_SCALES_NAME in name:
                param.requires_grad = True
