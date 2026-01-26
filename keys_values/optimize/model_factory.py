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
import re
from typing import Dict, List, Optional, Union, Tuple, Iterable, Any

import torch
import torch.nn as nn

from litgpt.config import Config as ConfigFull
from litgpt.lora import (
    LoRALinear,
    create_lora_linear,
    Config as ConfigLoRA,
)

from keys_values.attention import MultiHeadSelfAttention
from keys_values.kvcache.stack_layers import CellBlocks
from keys_values.lora import GPT as GPTLoRA, Block as BlockLoRA
from keys_values.model import GPT as GPTFull, Block as BlockFull
from keys_values.optimize.module_wrapper import (
    AccessWeightsGradients,
    FlatVectors,
)
from keys_values.use_eager_kernel import transform_mha_kwargs


class BlockComponentName:
    """
    Represents parameter name prefixes. Must be kept in sync with
    :class:`keys_values.model.GPT`.

    """

    @staticmethod
    def wte() -> str:
        return "transformer.wte"

    @staticmethod
    def ln_f() -> str:
        return "transformer.ln_f"

    @staticmethod
    def h(layer_idx: int) -> str:
        return "transformer.h." + str(layer_idx)

    @staticmethod
    def lm_head() -> str:
        return "lm_head"

    @staticmethod
    def components(
        model: GPTFull,
        lm_head: bool = True,
    ) -> List[Tuple[str, torch.nn.Module]]:
        result = [
            (BlockComponentName.wte(), model.transformer.wte),
            (BlockComponentName.ln_f(), model.transformer.ln_f),
        ] + [
            (BlockComponentName.h(layer_idx), block)
            for layer_idx, block in enumerate(model.transformer.h)
        ]
        if lm_head:
            result.append((BlockComponentName.lm_head(), model.lm_head))
        return result

    REGEX_H_NAME = re.compile(r"transformer\.h\.(\d+)$")

    @staticmethod
    def is_h(name: str) -> Optional[int]:
        m = BlockComponentName.REGEX_H_NAME.match(name)
        if m:
            return int(m.group(1))
        else:
            return None


def parent_of_parameter(
    module: nn.Module,
    param_name: str,
) -> Tuple[nn.Module, str]:
    parts = param_name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def debug_print_params(module: nn.Module):
    rows = ["", "Names and types of all params", ""]
    for name, param in module.named_parameters():
        parent, pname = parent_of_parameter(module, name)
        pobj = getattr(parent, pname)
        rows.append(f"{name}: {type(pobj)}")
    print("\n".join(rows))


def get_weights_as_flat_vectors(
    model: Union[GPTFull, GPTLoRA],
    lm_head: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, FlatVectors]:
    return {
        name: AccessWeightsGradients(block).get_weights(device=device)
        for name, block in BlockComponentName.components(model, lm_head)
    }


def device_of_flat_vectors(
    flat_vectors: Union[FlatVectors, Dict[str, FlatVectors]],
) -> torch.device:
    test_val = next(iter(flat_vectors.values()))
    if isinstance(test_val, torch.Tensor):
        devices = {vec.device for vec in flat_vectors.values()}
    else:
        devices = {
            vec.device for vecs in flat_vectors.values() for vec in vecs.values()
        }
    if len(devices) > 1:
        raise ValueError(
            f"flat_vectors contains vectors on more than one device: {devices}"
        )
    return next(iter(devices))


REGEX_SHARD_TYPE = re.compile(r"h(\d+):(\d+)$")


def names_and_modules_for_shard(
    gpt_model: Union[GPTFull, GPTLoRA],
    shard_type: Optional[str],
    use_lm_head: bool = True,
) -> Tuple[List[Tuple[str, torch.nn.Module]], Optional[Tuple[int, int]]]:
    """
    Returns list of `(name, module)` tuples for modules of a shard determined
    by `shard_type`. This can be "wte", "lm_head" (which includes "ln_f"), or
    "h{start}:{end}". If `shard_type is None`, the union of all shards is used.

    Args:
        gpt_model: Model to extract modules of
        shard_type: See above
        use_lm_head: If `False`, the `lm_head` module is not included

    Returns:
        List of `(name, module)` tuples, see above. If `shard_type ==
        "h{start}:{end}"`, we also return `(start, end)`.

    """
    names_and_modules = []
    start = None
    end = None
    include_all = shard_type is None
    if include_all or shard_type == "wte":
        names_and_modules.append((BlockComponentName.wte(), gpt_model.transformer.wte))
    if include_all or ((not names_and_modules) and shard_type == "lm_head"):
        names_and_modules.append(
            (BlockComponentName.ln_f(), gpt_model.transformer.ln_f)
        )
        if use_lm_head:
            names_and_modules.append((BlockComponentName.lm_head(), gpt_model.lm_head))
    if include_all or not names_and_modules:
        if include_all:
            start = 0
            end = gpt_model.config.n_layer
        else:
            m = REGEX_SHARD_TYPE.match(shard_type)
            if m:
                start = int(m.group(1))
                end = int(m.group(2))
                if not (0 <= start < end <= gpt_model.config.n_layer):
                    start = end = None
        if start is not None:
            names_and_modules.extend(
                [
                    (
                        BlockComponentName.h(layer_idx),
                        gpt_model.transformer.h[layer_idx],
                    )
                    for layer_idx in range(start, end)
                ]
            )
    ret2 = None
    if not include_all:
        if not names_and_modules:
            raise ValueError(
                f"shard_type = {shard_type} unknown, must be 'wte', 'lm_head', or 'h<start>:<end>'"
            )
        if start is not None:
            ret2 = (start, end)
    return names_and_modules, ret2


class GPTFullWrapper(GPTFull):
    """
    Represents :class:`keys_values.model.GPT`, but based on components
    passed at construction.

    """

    def __init__(
        self,
        config: ConfigFull,
        components: Dict[str, nn.Module],
        mha: Optional[MultiHeadSelfAttention] = None,
        **mha_kwargs,
    ):
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        name = BlockComponentName.lm_head()
        if name in components:
            self.lm_head = components[name]
        else:
            self.lm_head = nn.Linear(
                config.n_embd,
                config.padded_vocab_size,
                bias=config.lm_head_bias,
            )
        self.transformer = nn.ModuleDict(
            dict(
                wte=components[BlockComponentName.wte()],
                h=nn.ModuleList(
                    components[BlockComponentName.h(block_idx)]
                    for block_idx in range(config.n_layer)
                ),
                ln_f=components[BlockComponentName.ln_f()],
            )
        )
        if mha is None:
            self.mha = MultiHeadSelfAttention(
                config,
                **transform_mha_kwargs(mha_kwargs, config),
            )
        else:
            self.mha = mha
        self.max_seq_length = config.block_size
        self._start_of_layer_hook = None
        self._default_kv_cache = False

    def get_weights_as_flat(
        self,
        lm_head: bool = True,
        device: Optional[torch.device] = None,
    ) -> Dict[str, FlatVectors]:
        return get_weights_as_flat_vectors(self, lm_head, device)


class GPTLoRAWrapper(GPTLoRA):
    """
    Represents :class:`keys_values.lora.GPT`, but based on components
    passed at construction.

    """

    def __init__(
        self,
        config: ConfigLoRA,
        components: Dict[str, nn.Module],
        mha: Optional[MultiHeadSelfAttention] = None,
        **mha_kwargs,
    ):
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        name = BlockComponentName.lm_head()
        if name in components:
            self.lm_head = components[name]
        else:
            self.lm_head = create_lora_linear(
                config,
                config.n_embd,
                config.padded_vocab_size,
                bias=config.lm_head_bias,
                use_r=config.lora_head,
            )
        self.transformer = nn.ModuleDict(
            dict(
                wte=components[BlockComponentName.wte()],
                h=nn.ModuleList(
                    components[BlockComponentName.h(block_idx)]
                    for block_idx in range(config.n_layer)
                ),
                ln_f=components[BlockComponentName.ln_f()],
            )
        )
        if mha is None:
            self.mha = MultiHeadSelfAttention(
                config,
                **transform_mha_kwargs(mha_kwargs, config),
            )
        else:
            self.mha = mha
        self.max_seq_length = config.block_size
        self._start_of_layer_hook = None
        self._default_kv_cache = False

    def get_weights_as_flat(
        self,
        lm_head: bool = True,
        device: Optional[torch.device] = None,
    ) -> Dict[str, FlatVectors]:
        return get_weights_as_flat_vectors(self, lm_head, device)


class GPTShardOfBlocks(GPTFull):
    """
        Represents a shard of a model of type :class:`keys_values.model.GPT` or
        :class:`keys_values.lora.GPT`. Different to :class:`GPTFullWrapper` or
        :class:`GPTLoRAWrapper`, not all model components are present. Objects of
        this class cannot be used in the normal way, but only with methods of
        :class:`keys_values.kvcache.gradient.accumulate.GradientAccumulator`.
    `
    """

    def __init__(
        self,
        config: ConfigFull,
        components: Dict[str, nn.Module],
        mha: Optional[MultiHeadSelfAttention] = None,
        **mha_kwargs,
    ):
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = components.get(BlockComponentName.lm_head())
        modules = dict()
        for name, kname in (
            (BlockComponentName.wte(), "wte"),
            (BlockComponentName.ln_f(), "ln_f"),
        ):
            block = components.get(name)
            if block is not None:
                modules[kname] = block
        h_modules = sorted(
            [
                (BlockComponentName.is_h(name), mod)
                for name, mod in components.items()
                if BlockComponentName.is_h(name) is not None
            ],
            key=lambda x: x[0],
        )
        if h_modules:
            idxs, mods = zip(*h_modules)
            first_layer_idx = idxs[0]
            if idxs != tuple(range(first_layer_idx, first_layer_idx + len(idxs))):
                raise ValueError(
                    f"Layer components have idxs {idxs}, must be {list(range(first_layer_idx, first_layer_idx + len(idxs)))}"
                )
            modules["h"] = nn.ModuleList(mods)
        else:
            first_layer_idx = None
        self.first_layer_idx = first_layer_idx
        if modules:
            self.transformer = nn.ModuleDict(modules)
        else:
            self.transformer = None
        if mha is None:
            self.mha = MultiHeadSelfAttention(
                config,
                **transform_mha_kwargs(mha_kwargs, config),
            )
        else:
            self.mha = mha
        if self._has_layers():
            self.max_seq_length = config.block_size

    def _has_layers(self) -> bool:
        return self.transformer is not None and hasattr(self.transformer, "h")

    @property
    def max_seq_length(self) -> int:
        if self._has_layers():
            return self._max_seq_length
        else:
            raise NotImplementedError("Cannot be used for this model shard")

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        if self._has_layers():
            # See https://gist.github.com/Susensio/979259559e2bebcd0273f1a95d7c1e79
            # Would be best to get rid of the setter!
            super(GPTShardOfBlocks, type(self)).max_seq_length.fset(self, value)
        else:
            raise NotImplementedError("Cannot be used for this model shard")

    def get_gradients_as_flat(
        self,
        device: Optional[torch.device] = None,
    ) -> Dict[str, FlatVectors]:
        if self.first_layer_idx is not None:
            result = {
                BlockComponentName.h(self.first_layer_idx + i): AccessWeightsGradients(
                    block
                ).get_gradients(device=device)
                for i, block in enumerate(self.transformer.h)
            }
        else:
            result = dict()
        if self.lm_head is not None:
            result[BlockComponentName.lm_head()] = AccessWeightsGradients(
                self.lm_head
            ).get_gradients(device=device)
        if self.transformer is not None:
            for name, kname in (
                (BlockComponentName.wte(), "wte"),
                (BlockComponentName.ln_f(), "ln_f"),
            ):
                if hasattr(self.transformer, kname):
                    result[name] = AccessWeightsGradients(
                        getattr(self.transformer, kname)
                    ).get_gradients(device=device)
        return result

    def forward(
        self,
        idx: torch.Tensor,
        skip_lm_head: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        raise NotImplementedError("Must not call `forward` for this object")

    def set_kv_caches(
        self,
        batch_size: int,
        dtype: Optional[torch.dtype] = None,
        max_seq_length: Optional[int] = None,
    ):
        raise NotImplementedError("Must not call `set_kv_caches` for this object")

    def clone(self, device: Optional[torch.device] = None) -> GPTFull:
        raise NotImplementedError("Must not call `clone` for this object")


class GPTShardCellBlock(CellBlocks):
    """
    Wraps shard of type :class:`GPTShardOfBlocks` into
    :class:`keys_values.kvcache.stack_layers.CellBlocks`, to be used with
    :class:`keys_values.kvcache.gradient.accumulate.GradientAccumulator`.

    """

    def __init__(self, shard: GPTShardOfBlocks):
        super().__init__(shard.config)
        if shard.first_layer_idx is None:
            raise ValueError("shard must be stack of layers")
        self._shard = shard

    @property
    def max_seq_length(self) -> int:
        return self._shard.max_seq_length

    def blocks_with_kwargs(self) -> Iterable[Tuple[int, BlockFull, Dict[str, Any]]]:
        block_kwargs = dict(mha=self._shard.mha)
        fli = self.first_layer_idx
        return zip(
            range(fli, fli + self.num_layers),
            self._shard.transformer.h,
            [block_kwargs] * self.num_layers,
        )

    @property
    def first_layer_idx(self) -> int:
        return self._shard.first_layer_idx

    @property
    def num_layers(self) -> int:
        return len(self._shard.transformer.h)


FLOAT_DTYPES_FROM_STR = {
    str(dtype): dtype for dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float64,
    )
}


class ModelFromFlatVectorsFactory:
    """
    Collects static methods for creating specific modules, setting their
    weights from flat vectors.

    """

    @staticmethod
    def create_full_block(
        config: ConfigFull,
        block_idx: int,
        weights_vecs: FlatVectors,
    ) -> BlockFull:
        """
        Create block of full model and set weights from `weights_vecs`. Make
        sure to dellocate `weights_vecs` afterwards.

        Args:
            config: See :class:`litgpt.model.GPT`
            block_idx: Index of block in complete model
            weights_vecs: Flat vectors the weights are initialized from

        Returns:
            :class:`litgpt.model.Block` object with initialized weights

        """
        device = device_of_flat_vectors(weights_vecs)
        with torch.device(device):
            block = BlockFull(config, block_idx)
        AccessWeightsGradients(block).set_weights(weights_vecs)
        return block

    @staticmethod
    def create_lora_block(
        config: ConfigLoRA,
        block_idx: int,
        weights_vecs: FlatVectors,
    ) -> BlockLoRA:
        """
        Create block of LoRA model and set weights from `weights_vecs`. Make
        sure to deallocate `weights_vecs` afterwards.

        Args:
            config: See :class:`litgpt.lora.GPT`
            block_idx: Index of block in complete model
            weights_vecs: Flat vectors the weights are initialized from

        Returns:
            :class:`litgpt.lora.Block` object with initialized weights

        """
        device = device_of_flat_vectors(weights_vecs)
        with torch.device(device):
            block = BlockLoRA(config, block_idx)
        AccessWeightsGradients(block).set_weights(weights_vecs)
        return block

    @staticmethod
    def _set_default_dtype(weights_vecs: FlatVectors):
        dtypes = [
            FLOAT_DTYPES_FROM_STR[name]
            for name in weights_vecs.keys()
            if name in FLOAT_DTYPES_FROM_STR
        ]
        # If `dtypes` is empty, `weights_vecs` has no float entries. This can
        # happen if the module has no parameters.
        if len(dtypes) == 1:
            torch.set_default_dtype(dtypes[0])
        elif len(dtypes) > 1:
            raise ValueError(f"weights_vecs.keys() = {list(weights_vecs.keys())}, must have exactly one float dtype")

    @staticmethod
    def create_wte(
        config: ConfigFull,
        weights_vecs: FlatVectors,
    ) -> nn.Embedding:
        device = device_of_flat_vectors(weights_vecs)
        ModelFromFlatVectorsFactory._set_default_dtype(weights_vecs)
        with torch.device(device):
            wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        AccessWeightsGradients(wte).set_weights(weights_vecs)
        return wte

    @staticmethod
    def create_ln_f(
        config: ConfigFull,
        weights_vecs: FlatVectors,
    ) -> nn.Module:
        device = device_of_flat_vectors(weights_vecs)
        ModelFromFlatVectorsFactory._set_default_dtype(weights_vecs)
        with torch.device(device):
            ln_f = config.norm_class(config.n_embd, eps=config.norm_eps)
        AccessWeightsGradients(ln_f).set_weights(weights_vecs)
        return ln_f

    @staticmethod
    def create_full_lm_head(
        config: ConfigFull,
        weights_vecs: FlatVectors,
    ) -> nn.Linear:
        device = device_of_flat_vectors(weights_vecs)
        ModelFromFlatVectorsFactory._set_default_dtype(weights_vecs)
        with torch.device(device):
            lm_head = nn.Linear(
                config.n_embd,
                config.padded_vocab_size,
                bias=config.lm_head_bias,
            )
        AccessWeightsGradients(lm_head).set_weights(weights_vecs)
        return lm_head

    @staticmethod
    def create_lora_lm_head(
        config: ConfigLoRA,
        weights_vecs: FlatVectors,
    ) -> LoRALinear:
        device = device_of_flat_vectors(weights_vecs)
        ModelFromFlatVectorsFactory._set_default_dtype(weights_vecs)
        with torch.device(device):
            lm_head = create_lora_linear(
                config,
                config.n_embd,
                config.padded_vocab_size,
                bias=config.lm_head_bias,
                use_r=config.lora_head,
            )
        AccessWeightsGradients(lm_head).set_weights(weights_vecs)
        return lm_head

    @staticmethod
    def _get_component_keys(all_keys: List[str], n_layer: int) -> List[str]:
        component_keys = [BlockComponentName.h(i) for i in range(n_layer)] + [
            BlockComponentName.wte(),
            BlockComponentName.ln_f(),
        ]
        for key in component_keys:
            if key not in all_keys:
                raise ValueError(f"weights_vecs is missing key {key}")
        name = BlockComponentName.lm_head()
        if name in all_keys:
            component_keys.append(name)
        return component_keys

    CHOICES_FULL = {
        BlockComponentName.wte(): create_wte,
        BlockComponentName.ln_f(): create_ln_f,
        BlockComponentName.lm_head(): create_full_lm_head,
    }

    CHOICES_LORA = {
        BlockComponentName.wte(): create_wte,
        BlockComponentName.ln_f(): create_ln_f,
        BlockComponentName.lm_head(): create_lora_lm_head,
    }

    @staticmethod
    def _create_components_full_model(
        config: ConfigFull,
        weights_vecs: Dict[str, FlatVectors],
        component_keys: List[str],
    ) -> Dict[str, nn.Module]:
        components = dict()
        for comp_name in component_keys:
            _weights_vecs = weights_vecs[comp_name]
            creator = ModelFromFlatVectorsFactory.CHOICES_FULL.get(comp_name)
            if creator is not None:
                components[comp_name] = creator(
                    config=config,
                    weights_vecs=_weights_vecs,
                )
            else:
                block_idx = BlockComponentName.is_h(comp_name)
                if block_idx is None:
                    raise ValueError(
                        f"Entry '{comp_name}' in component_keys is not valid"
                    )
                components[comp_name] = ModelFromFlatVectorsFactory.create_full_block(
                    config=config,
                    block_idx=block_idx,
                    weights_vecs=_weights_vecs,
                )
            # Deallocate
            del weights_vecs[comp_name]
            del _weights_vecs
        return components

    @staticmethod
    def create_full_model(
        config: ConfigFull,
        weights_vecs: Dict[str, FlatVectors],
        **mha_kwargs,
    ) -> GPTFullWrapper:
        """
        Creates complete :class:`litgpt.model.GPT` model, initializing all
        weights from `weights_vecs`.

        `weights_vecs` has entries for keys "wte", "layer{block_idx}". The
        entry for "lm_head" is optional. The entries of this dictionary
        are deleted once not needed anymore.

        Args:
            config: See :class:`litgpt.model.GPT`
            weights_vecs: Flat vectors the weights are initialized from.
                Entries are deleted as model is built. Empty on return.

        Returns:
            :class:`litgpt.model.GPT` object with initialized weights

        """
        # Create components
        component_keys = ModelFromFlatVectorsFactory._get_component_keys(
            list(weights_vecs.keys()),
            config.n_layer,
        )
        components = ModelFromFlatVectorsFactory._create_components_full_model(
            config=config,
            weights_vecs=weights_vecs,
            component_keys=component_keys,
        )
        # Complete model from components
        return GPTFullWrapper(
            config=config,
            components=components,
            **mha_kwargs,
        )

    @staticmethod
    def _create_components_lora_model(
        config: ConfigLoRA,
        weights_vecs: Dict[str, FlatVectors],
        component_keys: List[str],
    ) -> Dict[str, nn.Module]:
        components = dict()
        for comp_name in component_keys:
            _weights_vecs = weights_vecs[comp_name]
            creator = ModelFromFlatVectorsFactory.CHOICES_LORA.get(comp_name)
            if creator is not None:
                components[comp_name] = creator(
                    config=config,
                    weights_vecs=_weights_vecs,
                )
            else:
                block_idx = BlockComponentName.is_h(comp_name)
                if block_idx is None:
                    raise ValueError(
                        f"Entry '{comp_name}' in component_keys is not valid"
                    )
                components[comp_name] = ModelFromFlatVectorsFactory.create_lora_block(
                    config=config,
                    block_idx=block_idx,
                    weights_vecs=_weights_vecs,
                )
            # Deallocate
            del weights_vecs[comp_name]
            del _weights_vecs
        return components

    @staticmethod
    def create_lora_model(
        config: ConfigLoRA,
        weights_vecs: Dict[str, FlatVectors],
        **mha_kwargs,
    ) -> GPTLoRAWrapper:
        """
        Creates complete :class:`litgpt.lora.GPT` model, initializing all
        weights from `weights_vecs`.

        `weights_vecs`has entries for keys "wte", "layer{block_idx}". The
        entry for "lm_head" is optional. The entries of this dictionary
        are deleted once not needed anymore.

        Args:
            config: See :class:`litgpt.model.GPT`
            weights_vecs: Flat vectors the weights are initialized from.
                Entries are deleted as model is built. Empty on return.

        Returns:
            :class:`litgpt.lora.GPT` object with initialized weights

        """
        # Create components
        component_keys = ModelFromFlatVectorsFactory._get_component_keys(
            list(weights_vecs.keys()),
            config.n_layer,
        )
        components = ModelFromFlatVectorsFactory._create_components_lora_model(
            config=config,
            weights_vecs=weights_vecs,
            component_keys=component_keys,
        )
        # Complete model from components
        return GPTLoRAWrapper(
            config=config,
            components=components,
            **mha_kwargs,
        )
