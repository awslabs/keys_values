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
from dataclasses import replace

import torch
import pytest

from litgpt.config import Config

from keys_values.head_model import CrossEntropyOnLogits
from keys_values.head_model_factory import HeadModelFactory
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.gradient.main import LongContextGradientModel
from keys_values.kvcache.test_utils import create_kv_cache, copy_gradients
from keys_values.model import GPT


def args_complete_gradient_computation():
    return [
        a + b
        for a, b in product(
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
    "cache_name, cache_kwargs, cache_lengths",
    args_complete_gradient_computation(),
)
def test_complete_gradient_computation(cache_name, cache_kwargs, cache_lengths):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)

    device = torch.device("cpu")
    dtype = torch.float32
    torch.set_default_dtype(dtype)  # Set default dtype

    qname = "default"  # No quantization
    batch_sizes = [5] + [4] * (len(cache_lengths) - 1)
    n_layer = len(cache_lengths)
    n_head = 8
    n_query_groups = 4
    head_size = 64
    vocab_size = 48
    layers_per_cell = 1
    chunk_size = 8
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
        device=device,
        dtype=dtype,
    )
    gpt_model = GPT(config)
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
            total_loss = loss.mean().detach() + total_loss
        eval_loss_values.append(total_loss)
        model.train()
        total_loss = 0
        for input_ids, targets in zip(all_input_ids, all_targets):
            loss = model(input_ids, targets)
            loss.backward()
            # Check whether there are unmatched pack arguments
            some_unmatched = False
            for (fli, fci), logs in model.annotation_usage_logs().items():
                num_unmatched = len(logs.unmatched_pack_args)
                if num_unmatched > 0:
                    print(f"\nUnmatched pack arguments for first_layer_index={fli}, first_chunk_index={fci}")
                    print(logs.report())
                    some_unmatched = True
            assert not some_unmatched
            total_loss = loss.detach() + total_loss
        gradients.append(copy_gradients(gpt_model, device=torch.device("cpu")))
        train_loss_values.append(total_loss)

    # Compare the two
    print("\nComparing total loss values")
    torch.testing.assert_close(train_loss_values[0], train_loss_values[1])
    torch.testing.assert_close(eval_loss_values[0], eval_loss_values[1])
    print("Comparing training and evaluation losses")
    torch.testing.assert_close(train_loss_values[0], eval_loss_values[0])
    kwargs = {"atol": 2e-5, "rtol": 4e-5}
    for name, value in gradients[0].items():
        value_comp = gradients[1].get(name)
        if value_comp is None:
            raise IndexError(f"name = {name} is in gradients[0], but not in gradients[1]")
        print(f"Comparing gradient for {name}")
        torch.testing.assert_close(value, value_comp, **kwargs)
