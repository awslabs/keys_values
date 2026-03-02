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
from typing import List, Tuple
from itertools import product
import re

import torch
import pytest

from keys_values.config import Config

from keys_values.flex_attention import FlexAttentionArgs
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.kvcache.consts import split_name
from keys_values.kvcache.quantize.bitsandbytes import determine_blocksize
from keys_values.kvcache.test_utils import (
    available_backends,
    create_kv_cache,
    tensor_is_simple,
    cache_name_gpu_only,
    cache_names_and_devices,
    product_with_devices,
    random_args_cache_forward,
    range_from_args,
)
from keys_values.utils import randint_torch


def args_compare_forward() -> Tuple[str, List[tuple]]:
    names = [
        name
        for name in KVCacheFactory.supported_names()
        if name.startswith("lastrec") and not name.endswith("-default")
    ]
    setups = [
        (128, 16, 256),
        (128, 32, 281),
    ]
    return (
        "device, name, cache_length, chunk_size, seq_length",
        [
            (a, b) + c for a, b, c in product(
                available_backends(),
                names,
                setups,
            )
        ]
    )

@pytest.mark.parametrize(*args_compare_forward())
def test_compare_forward(device, name, cache_length, chunk_size, seq_length):
    seed = 31415927
    torch.random.manual_seed(seed)

    dtype = torch.float16
    torch.set_default_dtype(dtype)  # Set default dtype

    batch_size = 2
    n_layer = 4
    n_head = 8
    n_query_groups = 4
    head_size = 32
    vocab_size = 48

    # Create model and KV caches
    config = Config(
        n_layer=n_layer,
        n_head=n_head,
        n_query_groups=n_query_groups,
        n_embd=n_head * head_size,
        block_size=seq_length,
        vocab_size=vocab_size,
        rotary_percentage=1,
    )
    params = KVCacheParams.from_config(
        config=config,
        max_batch_size=batch_size,
        cache_length=cache_length,
        dtype=dtype,
    )
    if device == torch.device("cpu"):
        mha_kwargs = dict()
    else:
        mha_kwargs = dict(flexatt_args=FlexAttentionArgs())
    # HIER!
    with torch.device(device):
        gpt_model = GPT(config, **mha_kwargs)
        gpt_model.apply(gpt_model._init_weights)  # Initialization
    gpt_model.assign_kv_caches(
        [
            create_kv_cache(
                name=cache_name + "-" + qname,
                params=replace(params, cache_length=cache_length),
                block_idx=block_idx,
                **mha_kwargs,
            )
            for block_idx, cache_length in enumerate(cache_lengths)
        ]
    )
    may_match_twice = (
        may_match_twice_fused_eager_sdpa
        if use_old_cache
        else may_match_twice_flex_attention_sdpa
    )
    autograd_hooks_kwargs = dict(
        max_match_trials_pack_arg=4,
        may_match_twice=may_match_twice,
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
        num_output_tokens = randint_torch(4, int(seq_length * 0.75))
        all_input_ids.append(token_ids[:, :-1])
        all_targets.append(token_ids[:, (-num_output_tokens):])
    head_model = HeadModelFactory.create(name=head_model_name, config=config)
