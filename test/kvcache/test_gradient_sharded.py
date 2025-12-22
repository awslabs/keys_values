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
import random
from itertools import product

import torch
import pytest

from litgpt.config import name_to_config
from litgpt.lora import Config, mark_only_lora_as_trainable
from litgpt.utils import _RunIf

from keys_values.head_model_factory import HeadModelFactory
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.gradient.main import LongContextGradientModel
from keys_values.kvcache.test_utils import (
    create_kv_cache,
    copy_gradients,
)
from keys_values.lora import GPT
from keys_values.utils import copy_parameters


def args_gradient_sharded():
    name_kwargs = [
        ("lastrec-default", dict()),
        ("h2o-default", {"replay_log_blocksize": 64}),
        ("h2o-default", {"grace_period": 10, "replay_log_blocksize": 64}),
        ("h2o-torch-quantized8", {"grace_period": 10, "replay_log_blocksize": 64}),
    ]
    setups = [
        (128, 32, 128),
        (128, 32, 128 + 2 * 32),
        (256, 48, 256 + 2 * 48 + 19),
    ]
    return [
        a + (b,) + c + d
        for a, b, c, d in product(
            name_kwargs,
            #[torch.bfloat16, torch.float32],
            [torch.float32],
            setups,
            [(1, False), (1, True), (2, False)],
        )
    ]


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "cache_name, cache_kwargs, dtype, cache_length, chunk_size, seq_length, layers_per_cell, clone_model_via_flat_vectors",
    args_gradient_sharded()[:1],
)
def test_gradient_sharded(
    cache_name,
    cache_kwargs,
    dtype,
    cache_length,
    chunk_size,
    seq_length,
    layers_per_cell,
    clone_model_via_flat_vectors,
):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)

    device = torch.device("cpu")
    cpu_offload_device = torch.device("cuda", 0)
    batch_size = 2
    model_name = "Qwen2.5-0.5B"
    head_model_name = "next_token_prediction"
    num_output_tokens = max(1, seq_length // 4)
    torch.set_default_dtype(dtype)  # Set default dtype

    # Create model and KV caches
    config_dict = name_to_config[model_name].copy()
    config_dict["n_layer"] = 3 * layers_per_cell
    config_dict["block_size"] = seq_length + 2
    config_dict.update(
        dict(
            lora_r = 8,
            lora_alpha = 16,
            lora_dropout = 0.05,
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
        with torch.device(_device):
            gpt_model = GPT(config)
            mark_only_lora_as_trainable(gpt_model)
            head_models.append(
                HeadModelFactory.create(name=head_model_name, config=config)
            )
            cache_params = KVCacheParams.from_config(
                config=config,
                max_batch_size=batch_size,
                cache_length=cache_length,
                device=_device,
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
                lcg_kwargs = dict(
                    cpu_offload_device=cpu_offload_device,
                    clone_model_via_flat_vectors=clone_model_via_flat_vectors,
                )
                # DEBUG
                with torch.device(cpu_offload_device):
                    debug_gpt_model = GPT(config)
                mark_only_lora_as_trainable(debug_gpt_model)
                copy_parameters(gpt_model, debug_gpt_model)
                debug_gpt_model.zero_grad()
                # END DEBUG
            else:
                lcg_kwargs = dict()
            gpt_models.append(gpt_model)
            lcg_models.append(
                LongContextGradientModel(
                    gpt_model=gpt_model,
                    head_model=head_models[-1],
                    layers_per_cell=layers_per_cell,
                    chunk_size=chunk_size,
                    qname="default",
                    debug_gpt_model=debug_gpt_model,  # DEBUG
                    debug_store_intermediates=True,  # DEBUG
                    **lcg_kwargs,
                )
            )
    # Create data batch
    token_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_length + 1),
        device=device,
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
            _input_ids = input_ids
            _targets = targets
            debug_gpt_model.zero_grad()
        else:
            print("\n*** Computing gradients normally (no CPU offloading) ***")
            _input_ids = input_ids.to(device=cpu_offload_device)
            _targets = targets.to(device=cpu_offload_device)
        loss = model(_input_ids, _targets)
        loss.backward()
        loss_values.append(loss.detach())
        gradients.append(
            copy_gradients(model.gpt_model, device=torch.device("cpu"))
        )
        debug_intermediates.append(model.debug_intermediates)
    # Compare the two
    print("\nComparing loss values:")
    torch.testing.assert_close(loss_values[0], loss_values[1])
    print("\nComparing intermediates during forward:")
    for name, value in debug_intermediates[0].items():
        value_comp = debug_intermediates[1].get(name)
        if value_comp is None:
            raise IndexError(f"name = {name} is in debug_intermediates[0], but not in debug_intermediates[1]")
        print(name)
        torch.testing.assert_close(value, value_comp)
    print("\nComparing gradients:")
    for name, value in gradients[0].items():
        value_comp = gradients[1].get(name)
        if value_comp is None:
            raise IndexError(f"name = {name} is in gradients[0], but not in gradients[1]")
        # DEBUG
        value_debug = debug_gpt_model.get_parameter(name).grad.data.to(
            device=device
        )
        print(name)
        torch.testing.assert_close(value, value_debug)
        #print(f"Comparing gradient for {name}")
        #torch.testing.assert_close(value, value_comp)


def test_gradient_sharded_simple():
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)

    #cache_name = "lastrec-default"
    #cache_length = 128
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
    config_dict["n_layer"] = 2
    config_dict["block_size"] = seq_length + 2
    config_dict.update(
        dict(
            lora_r = 8,
            lora_alpha = 16,
            lora_dropout = 0.05,
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
    for _device in (cpu_offload_device, device):
        with torch.device(_device):
            gpt_model = GPT(config)
            mark_only_lora_as_trainable(gpt_model)
            #cache_params = KVCacheParams.from_config(
            #    config=config,
            #    max_batch_size=batch_size,
            #    cache_length=cache_length,
            #    device=_device,
            #    dtype=dtype,
            #)
            #gpt_model.assign_kv_caches(
            #    [
            #        create_kv_cache(
            #            name=cache_name,
            #            params=cache_params,
            #            block_idx=block_idx,
            #        )
            #        for block_idx in range(config.n_layer)
            #    ]
            #)
            if gpt_models:
                gpt_model.apply(gpt_model._init_weights)  # Initialize
                # Copy from CPU to GPU
                copy_parameters(gpt_model, gpt_models[0])
            gpt_models.append(gpt_model)
            head_models.append(
                HeadModelFactory.create(name=head_model_name, config=config)
            )
    # Create data batch
    token_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_length + 1),
        device=device,
    )
    input_ids = token_ids[:, :-1]
    targets = token_ids[:, (-num_output_tokens):]
    # Compute gradients in two different ways: without and with CPU
    # offloading
    gradients = []
    loss_values = []
    for gpt_model in gpt_models:
        is_offload = len(gradients) > 0
        gpt_model.zero_grad()
        gpt_model.train()
        if is_offload:
            print("\n*** Computing gradients with CPU offloading ***")
            _input_ids = input_ids
            _targets = targets
        else:
            print("\n*** Computing gradients normally (no CPU offloading) ***")
            _input_ids = input_ids.to(device=cpu_offload_device)
            _targets = targets.to(device=cpu_offload_device)
        outputs = gpt_model(_input_ids)
        loss = model(_input_ids, _targets)
        loss.backward()
        loss_values.append(loss.detach())
        gradients.append(
            copy_gradients(model.gpt_model, device=torch.device("cpu"))
        )
        debug_intermediates.append(model.debug_intermediates)
    # Compare the two
    print("\nComparing loss values:")
    torch.testing.assert_close(loss_values[0], loss_values[1])
    print("\nComparing intermediates during forward:")
    for name, value in debug_intermediates[0].items():
        value_comp = debug_intermediates[1].get(name)
        if value_comp is None:
            raise IndexError(f"name = {name} is in debug_intermediates[0], but not in debug_intermediates[1]")
        print(name)
        torch.testing.assert_close(value, value_comp)
    print("\nComparing gradients:")
    for name, value in gradients[0].items():
        value_comp = gradients[1].get(name)
        if value_comp is None:
            raise IndexError(f"name = {name} is in gradients[0], but not in gradients[1]")
        # DEBUG
        value_debug = debug_gpt_model.get_parameter(name).grad.data.to(
            device=device
        )
        print(name)
        torch.testing.assert_close(value, value_debug)
        #print(f"Comparing gradient for {name}")
        #torch.testing.assert_close(value, value_comp)
