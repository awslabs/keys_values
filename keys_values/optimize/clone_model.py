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
from typing import Union, Callable, Optional

import torch

from keys_values.adapter import GPT as GPTAdapter
from keys_values.kvcache.basics import KVCacheWithBuffers
from keys_values.lora import GPT as GPTLoRA
from keys_values.model import GPT as GPTFull
from keys_values.optimize.model_factory import (
    BlockComponentName,
    GPTFullWrapper,
    GPTLoRAWrapper,
    GPTShardOfBlocks,
    ModelFromFlatVectorsFactory,
    names_and_modules_for_shard,
)
from keys_values.optimize.module_wrapper import (
    FlatVectors,
    AccessWeightsGradients,
)


def copy_flat_vectors_to(
    flat_vectors: FlatVectors,
    device: torch.device,
) -> FlatVectors:
    return {
        dtype: vec.to(device=device, non_blocking=True)
        for dtype, vec in flat_vectors.items()
    }


CopyFlatVectorFunction = Callable[[FlatVectors, torch.device], FlatVectors]


def clone_model_shard_via_flat_vectors(
    model: GPTFull,
    device: torch.device,
    shard_type: Optional[str],
    copy_function: Optional[CopyFlatVectorFunction] = None,
    lm_head: bool = True,
) -> Union[GPTFullWrapper, GPTLoRAWrapper, GPTShardOfBlocks]:
    """
    Creates copy of shard `shard_type` from `model` on device `device`. If
    `shard_type is None`, the whole model is copied.
    Different to `model.clone(device)`, this is done by creating flat vectors, copying them
    to the device, and creating the model there from the flat vectors.

    This function loops over submodules, copying flat vectors for each. This
    requires less memory on device for the flat vectors.

    """
    if copy_function is None:
        copy_function = copy_flat_vectors_to
    is_lora = isinstance(model, GPTLoRA)
    if isinstance(model, GPTAdapter):
        raise NotImplementedError("model must not be GPTAdapter: Not implemented")
    names_and_modules, ret2 = names_and_modules_for_shard(
        gpt_model=model,
        shard_type=shard_type,
        use_lm_head=lm_head,
    )
    if ret2 is None:
        if shard_type is None:
            ret2 = (0, len(model.transformer.h))
        else:
            ret2 = (None, None)
    start, end = ret2
    kv_caches = []
    try:
        # Remove KV caches before copy is created
        for l_ix, block in enumerate(model.transformer.h):
            kv_cache = block.attn.kv_cache
            if kv_cache is not None and isinstance(kv_cache, KVCacheWithBuffers) and kv_cache.buffers_are_allocated:
                raise ValueError(f"KV cache of layer {l_ix} has buffers allocated. Deallocate buffers with `deallocate_kv_cache_buffers_of_model`")
            kv_caches.append(kv_cache)
            block.attn.kv_cache = None

        # Loop to create components on the target device
        components = dict()
        choices = ModelFromFlatVectorsFactory.CHOICES_LORA if is_lora else ModelFromFlatVectorsFactory.CHOICES_FULL
        for comp_name, src_module in names_and_modules:
            flat_vecs_src = AccessWeightsGradients(src_module).get_weights()
            flat_vecs_trg = copy_function(flat_vecs_src, device)
            creator = choices.get(comp_name)
            if creator is not None:
                components[comp_name] = creator(
                    config=model.config,
                    weights_vecs=flat_vecs_trg,
                )
            else:
                block_idx = BlockComponentName.is_h(comp_name)
                if block_idx is None:
                    raise ValueError(f"Entry '{comp_name}' in model is not valid")
                if is_lora:
                    _creator = ModelFromFlatVectorsFactory.create_lora_block
                else:
                    _creator = ModelFromFlatVectorsFactory.create_full_block
                components[comp_name] = _creator(
                    config=model.config,
                    block_idx=block_idx,
                    weights_vecs=flat_vecs_trg,
                )
    finally:
        for kv_cache, block in zip(kv_caches, model.transformer.h):
            block.attn.kv_cache = kv_cache

    if shard_type is None:
        if not is_lora:
            _target_class = GPTFullWrapper
        else:
            _target_class = GPTLoRAWrapper
    else:
        _target_class = GPTShardOfBlocks
    model_copy = _target_class(
        config=model.config,
        components=components,
        mha=model.mha,
    )
    _transfer_requires_grad(model, model_copy)
    # Deal with KV caches. Default device for buffers should be `device`.
    if start is not None:
        # Only if shard contains any layers:
        model_copy.max_seq_length = model.max_seq_length
        # Sanity check
        assert len(model_copy.transformer.h) == end - start
        for kv_cache, block in zip(
            kv_caches[start:end],
            model_copy.transformer.h
        ):
            if kv_cache is not None:
                block.attn.kv_cache = kv_cache.clone(device=device)

    return model_copy


def _transfer_requires_grad(
    model_from: torch.nn.Module,
    model_to: torch.nn.Module
):
    for name, param_to in model_to.named_parameters():
        param_from = model_from.get_parameter(name)
        param_to.requires_grad_(param_from.requires_grad)
