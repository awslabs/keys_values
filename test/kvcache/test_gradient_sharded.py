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
from itertools import product
from typing import Dict, Tuple, List

import torch
import pytest

from litgpt.config import name_to_config
from litgpt.lora import Config, mark_only_lora_as_trainable
from litgpt.utils import _RunIf

from keys_values.head_model import HeadModel
from keys_values.head_model_factory import HeadModelFactory
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.gradient.accumulate import copy_requires_grad
from keys_values.kvcache.gradient.main import LongContextGradientModel
from keys_values.kvcache.test_utils import create_kv_cache, copy_gradients
from keys_values.lora import GPT
from keys_values.optimize.clone_model import (
    clone_model_shard_via_flat_vectors,
    copy_flat_vectors_to,
)
from keys_values.optimize.model_factory import (
    GPTShardCellBlock,
    AccessWeightsGradients,
)
from keys_values.utils import copy_parameters


def args_gradient_sharded():
    name_kwargs = [
        ("lastrec-default", dict()),
        ("h2o-default", {"replay_log_blocksize": 64}),
        ("h2o-torch-quantized8", {"grace_period": 10, "replay_log_blocksize": 64}),
    ]
    setups = [
        (128, 32, 128),
        (128, 32, 128 + 2 * 32 + 15),
    ]
    return [
        a + b + (c,)
        for a, b, c in product(
            name_kwargs,
            setups,
            [1, 2],
        )
    ]


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "cache_name, cache_kwargs, cache_length, chunk_size, seq_length, layers_per_cell",
    args_gradient_sharded(),
)
def test_gradient_sharded(
    cache_name,
    cache_kwargs,
    cache_length,
    chunk_size,
    seq_length,
    layers_per_cell,
):
    seed = 31415927
    torch.random.manual_seed(seed)

    device = torch.device("cpu")
    dtype = torch.float32
    cpu_offload_device = torch.device("cuda", 0)
    batch_size = 2
    model_name = "Qwen2.5-0.5B"
    head_model_name = "next_token_prediction"
    num_output_tokens = max(1, seq_length // 4)
    torch.set_default_dtype(dtype)  # Set default dtype
    debug_store_intermediates = False
    use_debug_gpt_model = False

    # Create model and KV caches
    config_dict = name_to_config[model_name].copy()
    config_dict["n_layer"] = 3 * layers_per_cell
    config_dict["block_size"] = seq_length + 2
    # Note: `lora_dropout = 0` essential here!
    config_dict.update(
        dict(
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            lora_query=True,
            lora_key=False,
            lora_value=True,
            lora_projection=False,
            lora_mlp=False,
            lora_head=False,
        )
    )
    config = Config(**config_dict)
    # We need two versions of the model, on the different devices. The first
    # is for computations without CPU offloading, the second for computations
    # with CPU offloading
    gpt_models = []
    head_models = []
    lcg_models = []
    debug_gpt_model = None
    for _device in (cpu_offload_device, device):
        with torch.device(cpu_offload_device):
            gpt_model = GPT(config)
            head_model = HeadModelFactory.create(name=head_model_name, config=config)
        mark_only_lora_as_trainable(gpt_model)
        gpt_model = gpt_model.to(device=_device)
        cache_params = KVCacheParams.from_config(
            config=config,
            max_batch_size=batch_size,
            cache_length=cache_length,
            dtype=dtype,
        )
        gpt_model.assign_kv_caches(
            [
                create_kv_cache(
                    name=cache_name,
                    params=cache_params,
                    block_idx=block_idx,
                    **cache_kwargs,
                )
                for block_idx in range(config.n_layer)
            ]
        )
        if gpt_models:
            gpt_model.apply(gpt_model._init_weights)  # Initialize
            # Copy from CPU to GPU
            copy_parameters(gpt_model, gpt_models[0])
            copy_parameters(head_model, head_models[0])
            lcg_kwargs = dict(offload_device=cpu_offload_device)
            if use_debug_gpt_model:
                with torch.device(cpu_offload_device):
                    debug_gpt_model = GPT(config)
                mark_only_lora_as_trainable(debug_gpt_model)
                copy_parameters(gpt_model, debug_gpt_model)
                debug_gpt_model.zero_grad()
        else:
            lcg_kwargs = dict()
        gpt_models.append(gpt_model)
        head_models.append(head_model)
        lcg_models.append(
            LongContextGradientModel(
                gpt_model=gpt_model,
                head_model=head_model,
                layers_per_cell=layers_per_cell,
                chunk_size=chunk_size,
                qname="default",
                debug_gpt_model=debug_gpt_model,
                debug_store_intermediates=debug_store_intermediates,
                **lcg_kwargs,
            )
        )
    # Create data batch
    token_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_length + 1),
        device=cpu_offload_device,
    )
    input_ids = token_ids[:, :-1]
    targets = token_ids[:, (-num_output_tokens):]
    # Compute gradients in two different ways: without and with CPU
    # offloading
    gradients = []
    loss_values = []
    debug_intermediates = []
    for model in lcg_models:
        is_offload = len(gradients) > 0
        model.zero_grad()
        model.train()
        if is_offload:
            print("\n*** Computing gradients with CPU offloading ***")
            if use_debug_gpt_model:
                debug_gpt_model.zero_grad()
        else:
            print("\n*** Computing gradients normally (no CPU offloading) ***")
        loss = model(input_ids, targets)
        loss.backward()
        loss_values.append(loss.detach())
        gradients.append(copy_gradients(model.gpt_model, device=torch.device("cpu")))
        if debug_store_intermediates:
            debug_intermediates.append(model.debug_intermediates)
    # Compare the two
    print("\nComparing loss values:")
    torch.testing.assert_close(loss_values[0], loss_values[1])
    if debug_store_intermediates:
        print("\nComparing intermediates during forward:")
        for name, value in debug_intermediates[0].items():
            value_comp = debug_intermediates[1].get(name)
            if value_comp is None:
                raise IndexError(
                    f"name = {name} is in debug_intermediates[0], but not in debug_intermediates[1]"
                )
            print(name)
            torch.testing.assert_close(value, value_comp)
    print("\nComparing gradients:")
    for name, value in gradients[0].items():
        value_comp = gradients[1].get(name)
        if value_comp is None:
            raise IndexError(
                f"name = {name} is in gradients[0], but not in gradients[1]"
            )
        print(name)
        if use_debug_gpt_model:
            value_debug = debug_gpt_model.get_parameter(name).grad.data.to(
                device=device
            )
            torch.testing.assert_close(value, value_debug)
        torch.testing.assert_close(value, value_comp)


def accumulate_gradients(
    module_pairs: List[Tuple[torch.nn.Module, torch.nn.Module]],
):
    """
    `module_pairs` contains tuples `(mod_from, mod_to)`. For each tuple,
    read gradients of parameters in `mod_from` and add to gradients in
    `mod_to`.

    Args:
        module_pairs: List of `(mod_from, mod_to)` tuples
        debug_modules: Use for debugging only

    """
    for mod_from, mod_to in module_pairs:
        access = AccessWeightsGradients(mod_from)
        flat_vectors = copy_flat_vectors_to(
            access.get_gradients(),
            device=torch.device("cpu"),
        )
        mod_from.zero_grad(set_to_none=True)
        AccessWeightsGradients(mod_to).accumulate_gradients(flat_vectors)


def compute_gradients_on_device(
    gpt_model: GPT,
    head_model: HeadModel,
    cpu_offload_device: torch.device,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    gpt_model.zero_grad()
    # Forward pass:
    # - Store inputs to each layer
    # - Compute head gradient for last layer
    cmsod_kwargs = dict(
        model=gpt_model,
        device=cpu_offload_device,
        lm_head=head_model.needs_logits(),
    )
    gpt_model_on_device = clone_model_shard_via_flat_vectors(
        shard_type=None,
        **cmsod_kwargs,
    )
    gpt_model_on_device.train()
    input_ids = input_ids.to(device=cpu_offload_device)
    targets = targets.to(device=cpu_offload_device)
    with torch.no_grad():
        x = gpt_model_on_device.transformer.wte(input_ids)
        layer_inputs = []
        for block in gpt_model_on_device.transformer.h:
            layer_inputs.append(x)
            x = block(x, input_ids, gpt_model_on_device.mha)
    head_input = copy_requires_grad(x)
    x = gpt_model_on_device.transformer.ln_f(head_input)
    if head_model.needs_logits():
        x = gpt_model_on_device.lm_head(x)
    loss = head_model(x, targets, input_pos=0).mean()
    loss.backward()
    loss_value = loss.detach()
    del loss
    head_gradient = head_input.grad
    # Compute gradients layer per layer. Each layer is one shard
    for layer_idx, layer_input in reversed(list(enumerate(layer_inputs))):
        x = copy_requires_grad(layer_input)
        shard_type = f"h{layer_idx}:{layer_idx + 1}"
        shard_on_device = clone_model_shard_via_flat_vectors(
            shard_type=shard_type,
            **cmsod_kwargs,
        )
        model_part = GPTShardCellBlock(shard_on_device)
        output = model_part.forward(x, input_ids)
        _loss = (output * head_gradient).sum()
        _loss.backward()
        module_pairs = [
            (
                shard_on_device.transformer.h[0],
                gpt_model.transformer.h[layer_idx],
            ),
        ]
        accumulate_gradients(module_pairs)
        head_gradient = x.grad

    return loss_value, copy_gradients(gpt_model, device=torch.device("cpu"))


@_RunIf(min_cuda_gpus=1)
def test_gradient_sharded_simple():
    seed = 31415927
    torch.random.manual_seed(seed)

    seq_length = 128
    dtype = torch.float32
    device = torch.device("cpu")
    cpu_offload_device = torch.device("cuda", 0)
    batch_size = 2
    model_name = "Qwen2.5-0.5B"
    head_model_name = "next_token_prediction"
    num_output_tokens = max(1, seq_length // 4)
    torch.set_default_dtype(dtype)  # Set default dtype

    # Create model and KV caches
    config_dict = name_to_config[model_name].copy()
    config_dict["n_layer"] = 3
    config_dict["block_size"] = seq_length + 2
    # Note: `lora_dropout = 0` essential here!
    config_dict.update(
        dict(
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            lora_query=True,
            lora_key=False,
            lora_value=True,
            lora_projection=False,
            lora_mlp=False,
            lora_head=False,
        )
    )
    config = Config(**config_dict)
    # We need two versions of the model, on the different devices. The first
    # is for computations without CPU offloading, the second for computations
    # with CPU offloading
    gpt_models = []
    head_models = []
    offload_model = None
    for _device in (cpu_offload_device, device):
        with torch.device(cpu_offload_device):
            gpt_model = GPT(config)
            head_model = HeadModelFactory.create(name=head_model_name, config=config)
        mark_only_lora_as_trainable(gpt_model)
        gpt_model = gpt_model.to(device=_device)
        if gpt_models:
            gpt_model.apply(gpt_model._init_weights)  # Initialize
            # Copy from CPU to GPU
            copy_parameters(gpt_model, gpt_models[0])
            copy_parameters(head_model, head_models[0])
        gpt_models.append(gpt_model)
        head_models.append(head_model)
    # Create data batch
    token_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_length + 1),
        device=cpu_offload_device,
    )
    input_ids = token_ids[:, :-1]
    targets = token_ids[:, (-num_output_tokens):]
    # Compute gradients in two different ways: without and with CPU
    # offloading
    gradients = []
    loss_values = []
    for gpt_model, head_model in zip(gpt_models, head_models):
        is_offload = len(gradients) > 0
        gpt_model.zero_grad()
        gpt_model.train()
        if is_offload:
            print("\n*** Computing gradients with CPU offloading ***")
            loss_value, grads = compute_gradients_on_device(
                gpt_model=gpt_model,
                head_model=head_model,
                cpu_offload_device=cpu_offload_device,
                input_ids=input_ids,
                targets=targets,
            )
            loss_values.append(loss_value)
            gradients.append(grads)
        else:
            print("\n*** Computing gradients normally (no CPU offloading) ***")
            outputs = gpt_model(input_ids)
            loss = head_model(outputs, targets, input_pos=0).mean()
            loss.backward()
            loss_values.append(loss.detach())
            gradients.append(copy_gradients(gpt_model, device=torch.device("cpu")))
    # Compare the two
    print("\nComparing loss values:")
    torch.testing.assert_close(loss_values[0], loss_values[1])
    print("\nComparing gradients:")
    for name, value in gradients[0].items():
        value_comp = gradients[1].get(name)
        if value_comp is None:
            raise IndexError(
                f"name = {name} is in gradients[0], but not in gradients[1]"
            )
        print(f"Comparing gradient for {name}")
        torch.testing.assert_close(value, value_comp)


if __name__ == "__main__":
    test_gradient_sharded(
        cache_name="h2o-torch-quantized8",
        cache_kwargs={"grace_period": 10, "replay_log_blocksize": 64},
        cache_length=128,
        chunk_size=32,
        seq_length=128 + 2 * 32 + 15,
        layers_per_cell=1,
    )
