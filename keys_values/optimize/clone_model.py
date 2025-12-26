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
from typing import Dict, Union, Callable, Optional

import torch

from keys_values.adapter import GPT as GPTAdapter
from keys_values.kvcache.basics import KVCacheWithBuffers
from keys_values.lora import GPT as GPTLoRA
from keys_values.model import GPT as GPTFull
from keys_values.optimize.model_factory import (
    get_weights_as_flat_vectors,
    GPTFullWrapper,
    GPTLoRAWrapper,
    ModelFromFlatVectorsFactory,
)
from keys_values.optimize.module_wrapper import FlatVectors


def copy_flat_vectors_to(
    flat_vectors: Union[FlatVectors, Dict[str, FlatVectors]],
    device: torch.device,
) -> Union[FlatVectors, Dict[str, FlatVectors]]:
    if not flat_vectors:
        return dict()
    test_val = next(iter(flat_vectors.values()))
    if isinstance(test_val, torch.Tensor):
        return {
            dtype: vec.to(device=device, non_blocking=True)
            for dtype, vec in flat_vectors.items()
        }
    else:
        return {
            name: {
                dtype: vec.to(device=device, non_blocking=True)
                for dtype, vec in flat_vector.items()
            }
            for name, flat_vector in flat_vectors.items()
        }


CopyFlatVectorFunction = Callable[[Dict[str, FlatVectors], torch.device], Dict[str, FlatVectors]]


def clone_model_via_flat_vectors(
    model: GPTFull,
    device: torch.device,
    copy_function: Optional[CopyFlatVectorFunction] = None,
) -> Union[GPTFullWrapper, GPTLoRAWrapper]:
    """
    Creates copy of `model` on device `device`. Different to
    `model.clone(device)`, this is done by creating flat vectors, copying them
    to the device, and creating the model there from the flat vectors.

    """
    if copy_function is None:
        copy_function = copy_flat_vectors_to
    is_lora = isinstance(model, GPTLoRA)
    if isinstance(model, GPTAdapter):
        raise NotImplementedError("model must not be GPTAdapter: Not implemented")
    kv_caches = []
    try:
        # Remove KV caches before copy is created
        for l_ix, block in enumerate(model.transformer.h):
            kv_cache = block.attn.kv_cache
            if kv_cache is not None and isinstance(kv_cache, KVCacheWithBuffers) and kv_cache.buffers_are_allocated:
                raise ValueError(f"KV cache of layer {l_ix} has buffers allocated. Deallocate buffers with `deallocate_kv_cache_buffers_of_model`")
            kv_caches.append(kv_cache)
            block.attn.kv_cache = None
        flat_vectors = get_weights_as_flat_vectors(model)
    finally:
        for kv_cache, block in zip(kv_caches, model.transformer.h):
            block.attn.kv_cache = kv_cache

    flat_vectors = copy_function(flat_vectors, device)
    if not is_lora:
        model_copy = ModelFromFlatVectorsFactory.create_full_model(
            config=model.config,
            weights_vecs=flat_vectors,
            mha=model.mha,
        )
    else:
        model_copy = ModelFromFlatVectorsFactory.create_lora_model(
            config=model.config,
            weights_vecs=flat_vectors,
            mha=model.mha,
        )
    _transfer_requires_grad(model, model_copy)
    # Deal with KV caches. Default device for buffers should be `device`.
    for kv_cache, block in zip(kv_caches, model_copy.transformer.h):
        if kv_cache is not None:
            block.attn.kv_cache = kv_cache.clone(device=device)
    return model_copy


def _transfer_requires_grad(model_from: GPTFull, model_to: GPTFull):
    for name, param_to in model_to.named_parameters():
        param_from = model_from.get_parameter(name)
        param_to.requires_grad_(param_from.requires_grad)
