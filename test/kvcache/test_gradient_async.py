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
from dataclasses import replace
from itertools import product

import torch
import pytest

from litgpt.utils import _RunIf

from keys_values.attention.flex_attention import FlexAttentionArgs
from keys_values.config import Config
from keys_values.finetune.utils import may_match_twice_flex_attention_sdpa
from keys_values.head_model import CrossEntropyOnLogits
from keys_values.head_model_factory import HeadModelFactory
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.gradient.main import LongContextGradientModel
from keys_values.kvcache.test_utils import (
    create_kv_cache,
    copy_gradients,
)
from keys_values.model import GPT
from keys_values.utils import randint_torch


def args_async_cpu_transfer():
    return [
        (cache_name, cachecp_qname)
        for cache_name, cachecp_qname in product(
            ["lastrec", "h2o"],
            ["default", "torch-quantized8"],
        )
    ]


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("cache_name, cachecp_qname", args_async_cpu_transfer())
def test_async_cpu_transfer_gradients(cache_name, cachecp_qname):
    """
    Compares losses and gradients between `async_cpu_transfer=False`
    (single-stream, the default) and `async_cpu_transfer=True` (CPU-GPU
    transfers on separate CUDA streams, in parallel with GPU computation).
    Both must give the same results.

    We run two batches per setting, since some races only corrupt state
    for subsequent batches.
    """
    seed = 31415927
    torch.random.manual_seed(seed)
    device = torch.device("cuda", 0)

    dtype = torch.float32
    torch.set_default_dtype(dtype)

    cache_length = 128
    num_batches = 2
    batch_size = 4
    n_layer = 2
    n_head = 8
    n_query_groups = 4
    head_size = 64
    vocab_size = 48
    layers_per_cell = 1
    chunk_size = 32
    max_sequence_length = cache_length * 6
    min_sequence_length = cache_length * 4

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
        cache_length=cache_length,
        dtype=dtype,
    )
    mha_kwargs = dict(flexatt_args=FlexAttentionArgs(q_lens=[8, 32, 96, 128]))
    cache_kwargs = dict(mha_kwargs)
    if cache_name == "h2o":
        cache_kwargs["replay_log_blocksize"] = 64
    with torch.device(device):
        gpt_model = GPT(config, **mha_kwargs)
        gpt_model.apply(gpt_model._init_weights)  # Initialization
    gpt_model.assign_kv_caches(
        [
            create_kv_cache(
                name=cache_name + "-default",
                params=replace(params, cache_length=cache_length),
                block_idx=block_idx,
                **cache_kwargs,
            )
            for block_idx in range(n_layer)
        ]
    )
    autograd_hooks_kwargs = dict(
        max_match_trials_pack_arg=4,
        may_match_twice=may_match_twice_flex_attention_sdpa,
    )

    # Create data batches. Sequences are long enough (>= 4 * cache_length)
    # for rows to have several cells, so that prefetching (CPU -> GPU) and
    # delayed writes (GPU -> CPU) are really exercised.
    all_input_ids = []
    all_targets = []
    for _ in range(num_batches):
        seq_length = randint_torch(min_sequence_length, max_sequence_length)
        token_ids = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(batch_size, seq_length),
            device=device,
        )
        num_output_tokens = randint_torch(4, int(seq_length * 0.75))
        all_input_ids.append(token_ids[:, :-1])
        all_targets.append(token_ids[:, (-num_output_tokens):])
    head_model = HeadModelFactory.create(
        name=CrossEntropyOnLogits.NAME, config=config
    )

    # Main loop: First with async_cpu_transfer=False (single stream), then
    # with async_cpu_transfer=True (multiple streams)
    gradients = []
    train_loss_values = []
    for async_cpu_transfer in [False, True]:
        if not async_cpu_transfer:
            print("\n*** Gradient computation with async_cpu_transfer=False ***")
        else:
            print("\n*** Gradient computation with async_cpu_transfer=True ***")
        model = LongContextGradientModel(
            gpt_model=gpt_model,
            head_model=head_model,
            layers_per_cell=layers_per_cell,
            chunk_size=chunk_size,
            layercp_qname="default",
            cachecp_qname=cachecp_qname,
            autograd_hooks_kwargs=autograd_hooks_kwargs,
            async_cpu_transfer=async_cpu_transfer,
            layercp_pin_memory=True,
            cachecp_pin_memory=True,
        )
        model.zero_grad()
        model.train()
        loss_values = []
        for input_ids, targets in zip(all_input_ids, all_targets):
            loss = model(input_ids, targets)
            loss.backward()
            loss_values.append(loss.detach())
        gradients.append(copy_gradients(gpt_model, device=torch.device("cpu")))
        train_loss_values.append(loss_values)

    # Compare the two
    print("\nComparing loss values per batch")
    for loss, loss_comp in zip(*train_loss_values):
        torch.testing.assert_close(loss, loss_comp)
    kwargs = {"atol": 2e-5, "rtol": 4e-5}
    for name, value in gradients[0].items():
        value_comp = gradients[1].get(name)
        if value_comp is None:
            raise IndexError(
                f"name = {name} is in gradients[0], but not in gradients[1]"
            )
        print(f"Comparing gradient for {name}")
        assert not torch.isnan(value_comp).any(), f"NaNs in gradient for {name}"
        torch.testing.assert_close(value, value_comp, **kwargs)


@_RunIf(min_cuda_gpus=1)
def test_async_cpu_transfer_requires_pinned_memory():
    """
    `async_cpu_transfer=True` requires pinned CPU memory for both layer
    input and KV cache checkpoints.
    """
    config = Config(
        n_layer=2,
        n_head=8,
        n_query_groups=4,
        n_embd=8 * 64,
        block_size=256,
        vocab_size=48,
        rotary_percentage=1,
    )
    device = torch.device("cuda", 0)
    with torch.device(device):
        gpt_model = GPT(config)
    params = KVCacheParams.from_config(
        config=config,
        max_batch_size=2,
        cache_length=64,
        dtype=torch.float32,
    )
    gpt_model.assign_kv_caches(
        [
            create_kv_cache(name="lastrec-default", params=params, block_idx=i)
            for i in range(config.n_layer)
        ]
    )
    head_model = HeadModelFactory.create(
        name=CrossEntropyOnLogits.NAME, config=config
    )
    for layercp_pin, cachecp_pin in [(False, True), (True, False), (False, False)]:
        with pytest.raises(ValueError, match="async_cpu_transfer"):
            LongContextGradientModel(
                gpt_model=gpt_model,
                head_model=head_model,
                layers_per_cell=1,
                async_cpu_transfer=True,
                layercp_pin_memory=layercp_pin,
                cachecp_pin_memory=cachecp_pin,
            )
