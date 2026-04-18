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
from dataclasses import replace
from typing import List, Union
from unittest import mock

import torch
import pytest

from keys_values.config import Config

from keys_values.flex_attention import FlexAttentionArgs
from keys_values.generate.base import batched_generate_fn
from keys_values.head_model import CrossEntropyOnLogits
from keys_values.head_model_factory import HeadModelFactory
from keys_values.kvcache.base import KVCacheParams
from keys_values.long_context import LongContextInferenceModel
from keys_values.kvcache.replay_buffers import ModelForTokenGeneration
from keys_values.kvcache.test_utils import (
    create_kv_cache,
    available_backends,
)
from keys_values.model import GPT
from keys_values.utils import randint_torch


def args_loss_after_replay():
    return [
        b + c + (a,)
        for a, b, c in product(
            available_backends(),
            [
                ("lastrec", dict()),
                ("h2o", {"replay_log_blocksize": 64}),
                ("h2o", {"grace_period": 10, "replay_log_blocksize": 64}),
            ],
            [
                ([128, 128],),
                ([96, 128],),
            ],
        )
    ]


@pytest.mark.parametrize(
    "cache_name, cache_kwargs, cache_lengths, device",
    args_loss_after_replay(),
)
def test_loss_after_replay(
    cache_name,
    cache_kwargs,
    cache_lengths,
    device,
):
    seed = 31415927
    torch.random.manual_seed(seed)

    dtype = torch.float16
    torch.set_default_dtype(dtype)  # Set default dtype

    qname = "torch-quantized8"
    batch_sizes = [5] + [4] * (len(cache_lengths) - 1)
    n_layer = len(cache_lengths)
    n_head = 8
    n_query_groups = 4
    head_size = 64
    vocab_size = 128
    chunk_size = 16
    max_returned_tokens = 8
    max_sequence_length = max(cache_lengths) * 8
    min_sequence_length = max(cache_lengths) * 2

    # Create model and KV caches
    config = Config(
        n_layer=n_layer,
        n_head=n_head,
        n_query_groups=n_query_groups,
        n_embd=n_head * head_size,
        block_size=max_sequence_length,
        vocab_size=vocab_size,
        rotary_percentage=1,
    )
    params = KVCacheParams.from_config(
        config=config,
        max_batch_size=max(batch_sizes),
        cache_length=cache_lengths[0],
        dtype=dtype,
    )
    if device == torch.device("cpu"):
        mha_kwargs = dict()
    else:
        mha_kwargs = dict(flexatt_args=FlexAttentionArgs())
    cache_kwargs.update(mha_kwargs)
    with torch.device(device):
        gpt_model = GPT(config, **mha_kwargs)
        gpt_model.apply(gpt_model._init_weights)  # Initialization
    gpt_model.assign_kv_caches(
        [
            create_kv_cache(
                name=cache_name + "-" + qname,
                params=replace(params, cache_length=cache_length),
                block_idx=block_idx,
                **cache_kwargs,
            )
            for block_idx, cache_length in enumerate(cache_lengths)
        ]
    )

    # Create data batches
    head_model_name = CrossEntropyOnLogits.NAME
    all_input_ids = []
    all_targets = []
    for batch_size in batch_sizes:
        seq_length = randint_torch(min_sequence_length, max_sequence_length)
        token_ids = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(batch_size, seq_length),
            device=device,
        )
        num_output_tokens = randint_torch(4, int(seq_length * 0.5))
        all_input_ids.append(token_ids[:, :-1])
        all_targets.append(token_ids[:, (-num_output_tokens):])

    # Create model
    head_model = HeadModelFactory.create(name=head_model_name, config=config)
    model = LongContextInferenceModel(
        gpt_model=gpt_model,
        head_model=head_model,
        chunk_size=chunk_size,
    )
    model.eval()
    gen_wrapper = ModelForTokenGeneration(gpt_model)

    # We compare loss values (1) compute directly and (2) computed by first
    # processing the prompt, then generating tokens with replay buffers,
    # finally computing the loss value.
    loss_direct = []
    for input_ids, targets in zip(all_input_ids, all_targets):
        loss = model(input_ids, targets)
        loss_direct.append(loss.mean().detach())
    loss_indirect = []
    for input_ids, targets in zip(all_input_ids, all_targets):
        logits = model(input_ids, targets, mode="inputs")
        gen_wrapper.switch_status(True)
        result = list(
            batched_generate_fn(
                model=model,
                prompts_or_logits=logits,
                max_returned_tokens=max_returned_tokens,
                sample_args=dict(),
                deallocate_cache_buffers=False,
            )
        )
        print("\n".join(["", "Generated:"] + [str(x) for x in result]))
        gen_wrapper.switch_status(False)
        loss = model(input_ids, targets, mode="targets")
        loss_indirect.append(loss.mean().detach())

    # Compare the two
    print("Comparing loss values")
    for l_dir, l_ind in zip(loss_direct, loss_indirect):
        torch.testing.assert_close(l_dir, l_ind)


@pytest.mark.parametrize(
    "cache_name, cache_kwargs, cache_lengths, device",
    args_loss_after_replay(),
)
def test_generate_several_times(
    cache_name,
    cache_kwargs,
    cache_lengths,
    device,
):
    seed = 31415927
    torch.random.manual_seed(seed)

    dtype = torch.float16
    torch.set_default_dtype(dtype)  # Set default dtype

    qname = "torch-quantized8"
    batch_sizes = [5] + [4] * (len(cache_lengths) - 1)
    n_layer = len(cache_lengths)
    n_head = 8
    n_query_groups = 4
    head_size = 64
    vocab_size = 128
    chunk_size = 16
    max_returned_tokens = 64
    max_sequence_length = max(cache_lengths) * 4
    min_sequence_length = max(cache_lengths) * 2

    # Create model and KV caches
    config = Config(
        n_layer=n_layer,
        n_head=n_head,
        n_query_groups=n_query_groups,
        n_embd=n_head * head_size,
        block_size=max_sequence_length,
        vocab_size=vocab_size,
        rotary_percentage=1,
    )
    params = KVCacheParams.from_config(
        config=config,
        max_batch_size=max(batch_sizes),
        cache_length=cache_lengths[0],
        dtype=dtype,
    )
    if device == torch.device("cpu"):
        mha_kwargs = dict()
    else:
        mha_kwargs = dict(flexatt_args=FlexAttentionArgs())
    cache_kwargs.update(mha_kwargs)
    with torch.device(device):
        gpt_model = GPT(config, **mha_kwargs)
        gpt_model.apply(gpt_model._init_weights)  # Initialization
    gpt_model.assign_kv_caches(
        [
            create_kv_cache(
                name=cache_name + "-" + qname,
                params=replace(params, cache_length=cache_length),
                block_idx=block_idx,
                **cache_kwargs,
            )
            for block_idx, cache_length in enumerate(cache_lengths)
        ]
    )

    # Create data batches
    head_model_name = CrossEntropyOnLogits.NAME
    all_input_ids = []
    all_targets = []
    for batch_size in batch_sizes:
        seq_length = randint_torch(min_sequence_length, max_sequence_length)
        all_input_ids.append(
            torch.randint(
                low=0,
                high=config.vocab_size,
                size=(batch_size, seq_length),
                device=device,
            )
        )
        num_targets = randint_torch(max_returned_tokens // 4, max_returned_tokens)
        all_targets.append(
            torch.randint(
                low=0,
                high=config.vocab_size,
                size=(batch_size, num_targets),
                device=device,
            )
        )

    # Create model
    head_model = HeadModelFactory.create(name=head_model_name, config=config)
    model = LongContextInferenceModel(
        gpt_model=gpt_model,
        head_model=head_model,
        chunk_size=chunk_size,
    )
    model.eval()
    gen_wrapper = ModelForTokenGeneration(gpt_model)

    data_idx = [0]
    gen_logits = []

    def batched_sample(
        logits_stack: torch.Tensor,
        kwargs: Union[dict, List[dict]],
    ) -> torch.Tensor:
        pos = len(gen_logits)
        gen_logits.append(logits_stack)
        return all_targets[data_idx[0]][:, pos].unsqueeze(-1)

    # We process the targets 2x and check whether the same logits are
    # produced
    for i, input_ids in enumerate(all_input_ids):
        data_idx[0] = i
        num_targets = all_targets[i].shape[-1]
        init_logits = model(input_ids, targets=None)
        results = []
        for _ in range(2):
            gen_wrapper.switch_status(True)
            gen_logits.clear()
            with mock.patch(
                "keys_values.generate.base.batched_sample",
                batched_sample,
            ):
                dummy = list(
                    batched_generate_fn(
                        model=model,
                        prompts_or_logits=init_logits,
                        max_returned_tokens=num_targets,
                        sample_args=dict(),
                        deallocate_cache_buffers=False,
                    )
                )
            assert len(gen_logits) == num_targets
            results.append(gen_logits.copy())
            gen_wrapper.switch_status(False)
        # Compare the two
        print("Comparing logits")
        for j, (logits1, logits2) in enumerate(zip(results[0], results[1])):
            print(f"Token position {j}")
            torch.testing.assert_close(logits1, logits2)
