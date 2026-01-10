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

from keys_values.head_model import CrossEntropyOnLogits, SequenceClassification
from keys_values.head_model_factory import HeadModelFactory
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.gradient.main import LongContextGradientModel
from keys_values.kvcache.test_utils import (
    create_kv_cache,
    copy_gradients,
    available_backends,
    cache_names_and_devices,
)
from keys_values.model import GPT


def args_complete_gradient_computation():
    return [
        b + c + (d, a)
        for a, b, c, d in product(
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
            [False, True],
        )
    ]


@pytest.mark.parametrize(
    "cache_name, cache_kwargs, cache_lengths, use_new_cache, device",
    args_complete_gradient_computation(),
)
def test_complete_gradient_computation(
    cache_name, cache_kwargs, cache_lengths, use_new_cache, device,
):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)

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
        dtype=dtype,
    )
    with torch.device(device):
        gpt_model = GPT(config)
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
            train_cache_kwargs=dict(use_new_cache=use_new_cache),
            autograd_hooks_kwargs=dict(max_match_trials_pack_arg=4),
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


def args_copy_model_to_device():
    return [
        (device, dtype, name)
        for device, dtype, name in product(
            available_backends(do_mps=False),
            [torch.bfloat16, torch.float16, torch.float32],
            [
                name
                for name, _ in cache_names_and_devices(only_cpu=True)
                if not name.startswith("dense")
            ],
        )
    ]

@pytest.mark.parametrize(
    "cpu_offload_device, dtype, cache_name", args_copy_model_to_device(),
)
def test_copy_model_to_device(cpu_offload_device, dtype, cache_name):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.set_default_dtype(dtype)

    device = torch.device("cpu")
    cache_lengths = [128, 128]
    batch_size = 5
    n_layer = len(cache_lengths)
    n_head = 8
    n_query_groups = 4
    head_size = 64
    vocab_size = 48
    layers_per_cell = 1
    chunk_size = 8
    max_sequence_length = max(cache_lengths) * 8
    head_model_name = SequenceClassification.NAME

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
        max_batch_size=batch_size,
        cache_length=cache_lengths[0],
        dtype=dtype,
    )
    gpt_model = GPT(config)
    gpt_model.apply(gpt_model._init_weights)  # Initialization
    gpt_model.assign_kv_caches(
        [
            create_kv_cache(
                name=cache_name,
                params=replace(params, cache_length=cache_length),
                block_idx=block_idx,
            )
            for block_idx, cache_length in enumerate(cache_lengths)
        ]
    )
    for l_ix, kv_cache in enumerate(gpt_model.get_kv_caches()):
        assert kv_cache.device in (device, None), (l_ix, kv_cache.device, device)
    with torch.device(cpu_offload_device):
        head_model = HeadModelFactory.create(name=head_model_name, config=config)
    model = LongContextGradientModel(
        gpt_model=gpt_model,
        head_model=head_model,
        layers_per_cell=layers_per_cell,
        chunk_size=chunk_size,
        qname="default",
        offload_device=cpu_offload_device,
    )
    model.zero_grad()

    # Deep copy of model
    model_copy = model.copy_model_for_evaluation()
    # Compare all params
    _model = model.gpt_model
    _model_copy = model_copy.gpt_model
    state_dict = _model_copy.state_dict()
    for name, param in _model.named_parameters():
        assert name in state_dict, f"name='{name}' in original, but not copy"
        param_copy = state_dict[name]
        if param is None:
            assert param_copy is None, f"name='{name}': param is None, param_copy is not None"
        else:
            assert param.data.device == device, (param.data.device, device)
            assert param_copy.data.device == cpu_offload_device, (param_copy.data.device, cpu_offload_device)
            torch.testing.assert_close(param.data, param_copy.data.to(device=device))
    copy_names = _model_copy.state_dict().keys()
    names = _model.state_dict().keys()
    diff = set(copy_names).difference(names)
    assert len(diff) == 0, f"Entries in copy but not in original:\n{diff}"
    # All KV caches exist
    assert model.gpt_model.transformer.wte.weight.device == device
    assert model_copy.gpt_model.transformer.wte.weight.device == cpu_offload_device
    for l_ix, (kv_cache, kv_cache_copy) in enumerate(
        zip(
            model.gpt_model.get_kv_caches(),
            model_copy.gpt_model.get_kv_caches(),
        )
    ):
        assert kv_cache is not None and kv_cache_copy is not None, (l_ix, kv_cache, kv_cache_copy)
        assert type(kv_cache) == type(kv_cache_copy), (l_ix, type(kv_cache), type(kv_cache_copy))
        assert kv_cache_copy.device in (cpu_offload_device, None), (l_ix, kv_cache_copy.device, cpu_offload_device)


if __name__ == "__main__":
    test_complete_gradient_computation(
        cache_name="h2o",
        cache_kwargs={"grace_period": 10, "replay_log_blocksize": 64},
        cache_lengths=[128, 128],
        use_new_cache=False,
        device=torch.device("cuda", 0),
    )
