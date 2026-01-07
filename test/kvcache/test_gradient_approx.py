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
from typing import Dict, Tuple, Any, List

import torch
import pytest

from litgpt.config import name_to_config
from litgpt.lora import Config
from litgpt.utils import _RunIf

from keys_values.head_model import HeadModel
from keys_values.head_model_factory import HeadModelFactory
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.gradient.main import LongContextGradientModel
from keys_values.kvcache.test_utils import create_kv_cache, copy_gradients
from keys_values.long_context import (
    compute_loss_for_chunk,
    chunk_weights_for_loss,
)
from keys_values.lora import GPT


def compute_gradients_exact_backprop(
    gpt_model: GPT,
    head_model: HeadModel,
    cache_params: KVCacheParams,
    cache_name: str,
    cache_kwargs: Dict[str, Any],
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    chunk_sizes: List[int],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Assign KV caches. Buffers are default (no quantization)
    gpt_model.assign_kv_caches(
        [
            create_kv_cache(
                name=cache_name + "-default",
                params=cache_params,
                block_idx=block_idx,
                **cache_kwargs,
            )
            for block_idx in range(gpt_model.config.n_layer)
        ]
    )
    # Run normal forward-backward. Note that gradients are blocked inside the
    # KV caches
    gpt_model.zero_grad()
    gpt_model.train()
    loss_full = 0
    num_input_tokens = input_ids.shape[-1]
    weight_per_chunk = chunk_weights_for_loss(
        head_model=head_model,
        targets=targets,
        chunk_sizes=chunk_sizes,
        num_input_tokens=num_input_tokens,
    )
    input_pos = 0
    for num, weight in zip(chunk_sizes, weight_per_chunk):
        output_chunk = gpt_model(
            input_ids[:, input_pos:(input_pos + num)],
            input_pos=input_pos,
            skip_lm_head=not head_model.needs_logits(),
        )
        loss_part = compute_loss_for_chunk(
            head_model=head_model,
            model_outputs_for_chunk=output_chunk,
            targets=targets,
            num_input_tokens=num_input_tokens,
            input_pos=input_pos,
            scale_factor=weight,
        )
        loss_full = loss_part + loss_full
        input_pos += num

    loss_full.backward()
    gradients = copy_gradients(gpt_model, device=torch.device("cpu"))
    loss_full = loss_full.detach()
    return loss_full, gradients


def args_compare_gradient_to_approximations():
    return [
        ("lastrec", dict()),
        ("h2o", {"replay_log_blocksize": 64}),
        ("h2o", {"grace_period": 10, "replay_log_blocksize": 64}),
    ]


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "cache_name, cache_kwargs", args_compare_gradient_to_approximations(),
)
def test_compare_gradient_to_approximations(cache_name, cache_kwargs):
    """
    In this test, we compute gradients by exact backpropagation for a
    model of realistic size, comparing them to gradients approximated in
    various ways using activation checkpointing, w/o quantization.
    Gradients are compared with a number of metrics.

    We use the model with randomly initialized weights and sample the
    sequences at random. This test could be made more realistic by
    loading weights of a pre-trained model and use sequences from a
    real dataset.

    """
    pytest.skip("Test is still incomplete")
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)

    batch_size = 2
    sequence_length = 2 ** 12
    cache_length = 2 ** 10
    layers_per_cell = 1
    chunk_size = 256
    model_name = "Qwen2.5-0.5B"
    head_model_name = "next_token_prediction"
    device = torch.device("cuda")
    dtype = torch.float32
    torch.set_default_dtype(dtype)  # Set default dtype

    # Model: Qwen2.5-0.5B with 3 layers and LoRA parameierization
    config_dict = name_to_config[model_name].copy()
    config_dict["n_layer"] = 3
    config_dict["block_size"] = 2 ** 14
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
    config = Config.from_dict(config_dict)
    gpt_model = GPT(config)
    head_model = HeadModelFactory.create(name=head_model_name, config=config)
    # Parameters for KV caches
    cache_params = KVCacheParams.from_config(
        config=config,
        max_batch_size=batch_size,
        cache_length=cache_length,
        dtype=dtype,
    )

    # Random data
    token_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, sequence_length),
        device=device,
    )
    num_output_tokens = random.randint(4, int(sequence_length * 0.75))
    input_ids = token_ids[:, :-1]
    targets = token_ids[:, (-num_output_tokens):]

    gpt_model.assign_kv_caches(
        [
            create_kv_cache(
                name=cache_name + "-" + qname,
                params=params,
                block_idx=block_idx,
                **cache_kwargs,
            )
            for block_idx in range(config.n_layer)
        ]
    )

    # Create data batches
    head_model_name = "next_token_prediction"
    all_input_ids = []
    all_targets = []
    for batch_size in batch_sizes:
        seq_length = random.randint(min_sequence_length, max_sequence_length)
        token_ids = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(batch_size, seq_length),
            device=device,
        )
        num_output_tokens = random.randint(4, int(seq_length * 0.75))
        all_input_ids.append(token_ids[:, :-1])
        all_targets.append(token_ids[:, (-num_output_tokens):])
    head_model = HeadModelFactory.create(name=head_model_name, config=config)

    # Main loop: First is default gradient computation, which uses several
    # cells per row. Second is using a single cell per row.
    gradients = []
    train_loss_values = []
    eval_loss_values = []
    for debug_flag in [False, True]:
        if not debug_flag:
            print("\n*** Default computation of gradients ***")
        else:
            print("\n*** Gradient computation with single cell per row and no autograd hooks ***")
        model = LongContextGradientModel(
            gpt_model=gpt_model,
            head_model=head_model,
            layers_per_cell=layers_per_cell,
            chunk_size=chunk_size,
            qname=qname,
            debug_single_cell_per_row=debug_flag,
            debug_dont_use_autograd_hooks=debug_flag,
        )
        model.zero_grad()
        # Evaluate only
        model.eval()
        total_loss = 0
        for input_ids, targets in zip(all_input_ids, all_targets):
            loss = model(input_ids, targets)
            total_loss = loss.detach() + total_loss
        eval_loss_values.append(total_loss)
        model.train()
        total_loss = 0
        for input_ids, targets in zip(all_input_ids, all_targets):
            loss = model(input_ids, targets)
            loss.backward()
            total_loss = loss.detach() + total_loss
        gradients.append(copy_gradients(gpt_model, device=torch.device("cpu")))
        train_loss_values.append(total_loss)

    # Compare the two
    print("\nComparing total loss values")
    torch.testing.assert_close(train_loss_values[0], train_loss_values[1])
    torch.testing.assert_close(eval_loss_values[0], eval_loss_values[1])
    print("Comparing training and evaluation losses")
    torch.testing.assert_close(train_loss_values[0], eval_loss_values[0])
    for name, value in gradients[0].items():
        value_comp = gradients[1].get(name)
        if value_comp is None:
            raise IndexError(f"name = {name} is in gradients[0], but not in gradients[1]")
        print(f"Comparing gradient for {name}")
        torch.testing.assert_close(value, value_comp)
