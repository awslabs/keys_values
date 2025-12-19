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
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn

from litgpt.config import Config as ConfigFull
from litgpt.lora import (
    LoRALinear,
    create_lora_linear,
    Config as ConfigLoRA,
)

from keys_values.attention import MultiHeadSelfAttention
from keys_values.lora import GPT as GPTLoRA, Block as BlockLoRA
from keys_values.model import GPT as GPTFull, Block as BlockFull
from keys_values.optimize.module_wrapper import (
    AccessWeightsGradients,
    FlatVectors,
    ParameterStructure,
)
from keys_values.use_eager_kernel import transform_mha_kwargs


def parent_of_parameter(
    module: nn.Module, param_name: str,
) -> Tuple[nn.Module, str]:
    parts = param_name.split('.')
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
    parts = [("wte", model.transformer.wte)] + [
        (f"layer{block_idx}", block)
        for block_idx, block in enumerate(model.transformer.h)
    ]
    if lm_head:
        parts.append(("lm_head", model.lm_head))
    return {
        name: AccessWeightsGradients(block).get_weights(device=device)
        for name, block in parts
    }


class GPTFullWrapper(GPTFull):
    def __init__(
        self,
        config: ConfigFull,
        components: Dict[str, nn.Module],
        **mha_kwargs,
    ):
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        if "lm_head" in components:
            self.lm_head = components["lm_head"]
        else:
            self.lm_head = nn.Linear(
                config.n_embd,
                config.padded_vocab_size,
                bias=config.lm_head_bias,
            )
        self.transformer = nn.ModuleDict(
            dict(
                wte=components["wte"],
                h=nn.ModuleList(
                    components[f"layer{block_idx}"]
                    for block_idx in range(config.n_layer)
                ),
                ln_f=components["ln_f"],
            )
        )
        self.mha = MultiHeadSelfAttention(
            config, **transform_mha_kwargs(mha_kwargs, config),
        )
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
    def __init__(
        self,
        config: ConfigLoRA,
        components: Dict[str, nn.Module],
        **mha_kwargs,
    ):
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        if "lm_head" in components:
            self.lm_head = components["lm_head"]
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
                wte=components["wte"],
                h=nn.ModuleList(
                    components[f"layer{block_idx}"]
                    for block_idx in range(config.n_layer)
                ),
                ln_f=components["ln_f"],
            )
        )
        self.mha = MultiHeadSelfAttention(
            config, **transform_mha_kwargs(mha_kwargs, config),
        )
        self.max_seq_length = config.block_size
        self._start_of_layer_hook = None
        self._default_kv_cache = False

    def get_weights_as_flat(
        self,
        lm_head: bool = True,
        device: Optional[torch.device] = None,
    ) -> Dict[str, FlatVectors]:
        return get_weights_as_flat_vectors(self, lm_head, device)


class GPTStackBlocks(GPTFull):
    def __init__(
        self,
        config: ConfigFull,
        components: Union[List[BlockFull], List[BlockLoRA]],
        **mha_kwargs,
    ):
        nn.Module.__init__(self)
        assert config.padded_vocab_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(h=nn.ModuleList(components)))
        self.lm_head = None
        self.mha = MultiHeadSelfAttention(
            config, **transform_mha_kwargs(mha_kwargs, config),
        )
        self.max_seq_length = config.block_size
        self._default_kv_cache = False

    def forward(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
        input_pos: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if idx.shape != (batch_size, seq_len):
            raise ValueError(f"idx.shape = {idx.shape}, must be {(batch_size, seq_len)}")
        if self.max_seq_length < seq_len:
            raise ValueError(f"Cannot forward sequence of length {seq_len}, max seq length is only {self.max_seq_length}.")
        for_prefill = False
        if input_pos is not None:
            # Few tokens generation. This needs a KV cache. If none is assigned,
            # the call fails
            if not self.are_kv_caches_assigned():
                raise ValueError(
                    "KV caches are not assigned. Assign KV caches with 'assign_kv_caches' or create default caches with 'set_kv_caches'"
                )
            for_prefill = input_pos == 0
            if not for_prefill:
                for block_idx, block in enumerate(self.transformer.h):
                    kv_cache = block.attn.kv_cache
                    if kv_cache.next_token_pos is None:
                        raise ValueError("Inference calls need to start with pre-fill, i.e. 'input_pos=0'")
                    if kv_cache.next_token_pos != input_pos:
                        raise ValueError(
                            f"KV cache for layer {block_idx}: input_pos = {input_pos} != {kv_cache.next_token_pos} = kv_cache.next_token_pos"
                        )
                    if kv_cache.max_tokens_forward < seq_len:
                        raise ValueError(
                            f"KV cache for layer {block_idx}: seq_len = {seq_len}, must be <= max_tokens_forward = {kv_cache.max_tokens_forward}"
                        )

        for block_idx, block in enumerate(self.transformer.h):
            if for_prefill:
                # Complain if batch size of cache is too small
                attn = block.attn
                if attn.kv_cache.max_batch_size < batch_size:
                    raise ValueError(
                        f"Batch size {batch_size} is too large for KV cache layer {block_idx} (batch size {attn.kv_cache.max_batch_size}). Use 'assign_kv_caches' or `set_kv_caches'"
                    )
            x = block(x, idx, self.mha, input_pos)

        return x

    def get_gradients_as_flat(
            self,
            first_block_idx: int,
            device: Optional[torch.device] = None,
    ) -> Dict[str, FlatVectors]:
        return {
            f"layer{first_block_idx + i}": AccessWeightsGradients(block).get_gradients(device=device)
            for i, block in enumerate(self.transformer.h)
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
        sure to dellocate `weights_vecs` afterwards.

        Args:
            config: See :class:`litgpt.lora.GPT`
            block_idx: Index of block in complete model
            weights_vecs: Flat vectors the weights are initialized from

        Returns:
            :class:`litgpt.lora.Block` object with initialized weights

        """
        block = BlockLoRA(config, block_idx)
        AccessWeightsGradients(block).set_weights(weights_vecs)
        return block

    @staticmethod
    def create_wte(
        config: ConfigFull,
        weights_vecs: FlatVectors,
    ) -> nn.Embedding:
        wte = nn.Embedding(config.padded_vocab_size, config.n_embd)
        AccessWeightsGradients(wte).set_weights(weights_vecs)
        return wte

    @staticmethod
    def create_ln_f(
        config: ConfigFull,
        weights_vecs: FlatVectors,
    ) -> nn.Module:
        ln_f = config.norm_class(config.n_embd, eps=config.norm_eps)
        AccessWeightsGradients(ln_f).set_weights(weights_vecs)
        return ln_f

    @staticmethod
    def create_full_lm_head(
        config: ConfigFull,
        weights_vecs: FlatVectors,
    ) -> nn.Linear:
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
        component_keys = [f"layer{i}" for i in range(n_layer)] + ["wte", "ln_f"]
        for key in component_keys:
            if key not in all_keys:
                raise ValueError(f"weights_vecs is missing key {key}")
        if "lm_head" in all_keys:
            component_keys.append("lm_head")
        return component_keys

    @staticmethod
    def _create_components_full_model(
        config: ConfigFull,
        weights_vecs: Dict[str, FlatVectors],
        component_keys: List[str],
    ) -> Dict[str, nn.Module]:
        components = dict()
        for comp_name in component_keys:
            _weights_vecs = weights_vecs[comp_name]
            if comp_name.startswith("layer"):
                block_idx = int(comp_name[len("layer") :])
                components[comp_name] = ModelFromFlatVectorsFactory.create_full_block(
                    config=config,
                    block_idx=block_idx,
                    weights_vecs=_weights_vecs,
                )
            elif comp_name == "wte":
                components[comp_name] = ModelFromFlatVectorsFactory.create_wte(
                    config=config,
                    weights_vecs=_weights_vecs,
                )
            elif comp_name == "ln_f":
                components[comp_name] = ModelFromFlatVectorsFactory.create_ln_f(
                    config=config,
                    weights_vecs=_weights_vecs,
                )
            else:
                assert comp_name == "lm_head"
                components[comp_name] = ModelFromFlatVectorsFactory.create_full_lm_head(
                    config=config,
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
    ) -> GPTFull:
        """
        Creates complete :class:`litgpt.model.GPT` model, initializing all
        weights from `weights_vecs`.

        `weights_vecs`has entries for keys "wte", "layer{block_idx}". The
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
            list(weights_vecs.keys()), config.n_layer,
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
            if comp_name.startswith("layer"):
                block_idx = int(comp_name[len("layer") :])
                components[comp_name] = ModelFromFlatVectorsFactory.create_lora_block(
                    config=config,
                    block_idx=block_idx,
                    weights_vecs=_weights_vecs,
                )
            elif comp_name == "wte":
                components[comp_name] = ModelFromFlatVectorsFactory.create_wte(
                    config=config,
                    weights_vecs=_weights_vecs,
                )
            elif comp_name == "ln_f":
                components[comp_name] = ModelFromFlatVectorsFactory.create_ln_f(
                    config=config,
                    weights_vecs=_weights_vecs,
                )
            else:
                assert comp_name == "lm_head"
                components[comp_name] = ModelFromFlatVectorsFactory.create_lora_lm_head(
                    config=config,
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
    ) -> GPTLoRA:
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
            list(weights_vecs.keys()), config.n_layer,
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

    @staticmethod
    def remove_params_of_model(
        model: Union[GPTFull, GPTLoRA],
        lm_head: bool = True,
    ) -> Dict[str, Dict[str, ParameterStructure]]:
        """
        Removes all named parameters from `model`. Relies on PyTorch
        default naming convention.

        Note: Do not call this method on a model whose parameters are
        referred to from elsewhere, e.g. from an optimizer.

        Args:
            model: Module to remove named parameters from
            lm_head: Remove `lm_head` params as well?

        Returns:
            Nested dictionary, containing parameter structures for
            all blocks whose params have been removed

        """
        components = [
            (f"layer{i}", block) for i, block in enumerate(model.transformer.h)
        ] + [
            ("wte", model.transformer.wte),
            ("ln_f", model.transformer.ln_f),
        ]
        if lm_head:
            components.append(("lm_head", model.transformer.lm_head))
        result = dict()
        for comp_name, block in components:
            param_structure = AccessWeightsGradients(block).param_structure()
            result[comp_name] = param_structure
            for name in [
                pspec.name
                for struct in param_structure.values()
                for pspec in struct.entries
            ]:
                parent, pname = parent_of_parameter(model, name)
                delattr(parent, pname)
        return result

    @staticmethod
    def restore_params_of_model(
        model: Union[GPTFull, GPTLoRA],
        param_structures: Dict[str, Dict[str, ParameterStructure]],
        weights_vecs: Dict[str, FlatVectors],
    ):
        """
        Given a model `model` whose parameters have been removed by
        :meth:`remove_params_of_model`, restore these parameters from
        the flat vectors in `weights_vecs`.

        Note: Do not call this method on a model whose parameters are
        referred to from elsewhere, e.g. from an optimizer.

        Args:
            model: Module to restore named parameters
            param_structures: Dictionary of parameter structures, returned
                by :meth:`remove_params_of_model`
            weights_vecs: Flat vectors the parameters are restored from.
                Entries are deleted as model parameters are restored.

        """
        # Loop over model compenents, and data types
        for comp_name, param_structure in param_structures.items():
            for dtype, structure in param_structure.items():
                weight_vec = weights_vecs[comp_name][dtype]
                if weight_vec.numel() != structure.size:
                    raise ValueError(f"comp_name={comp_name}, dtype={dtype}: weights_vecs[comp_name][dtype].numel()={weight_vec.numel()}, param_structures[comp_name][dtype].size={structure.size}. Must be the same")
                for pspec in structure.entries:
                    start, end = pspec.range
                    src_arg = weight_vec[start:end].view(*pspec.shape)
                    parent, pname = parent_of_parameter(model, pspec.name)
                    parent.register_parameter(pname, nn.Parameter(src_arg))
                del weights_vecs[comp_name][dtype]
            del weights_vecs[comp_name]
