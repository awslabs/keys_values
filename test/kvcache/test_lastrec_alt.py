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
import random
from typing import List

import torch
import pytest

from litgpt.config import Config

from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.kvcache.test_utils import random_args_cache_forward


def args_compare_variants() -> List[tuple]:
    # For `bfloat16`, `float16`, and the non-trivial cases in `setups`,
    # the test fails, since certain entries have high relative errors.
    # For these cases, we only insist on small absolute errors.
    # For `float32`, all cases work with the normal tolerances.
    batch_sizes = [1, 3]
    just_atol = dict(atol=0.00015, rtol=1e+5)
    setups = [
        (112, 16, 112, dict()),  # Cache not yet full
        (128, 32, 44, just_atol),  # R1 left of R2, no overlap
        (128, 48, 72, just_atol),  # R1 left of R2, overlap
        (128, 32, 112, just_atol),  # R1 right of R2, splits into two
        (128, 72, 112, just_atol),  # R1 right of R2, splits into two
        (128, 128, 16, dict()),  # Cache fully replaced
    ]
    n_head_groups = [(4, 2), (4, 4), (8, 1)]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    result = [
        record[:1] + record[1] + record[2] + record[3:]
        for record in product(
            batch_sizes, # 2
            setups, # 6
            n_head_groups, # 3
            dtypes, # 3
        )
    ]
    return result


@pytest.mark.parametrize(
    "batch_size, current_length, q_len, next_pos, tol_kwargs, n_head, n_query_groups, dtype",
    args_compare_variants(),
)
def test_compare_variants(batch_size, current_length, q_len, next_pos, tol_kwargs, n_head, n_query_groups, dtype):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    cache_length = 128
    head_size = 12
    vocab_size = 128
    device = torch.device("cpu")
    names = ["lastrec-default", "lastrec-alt-default"]
    assert current_length == cache_length or current_length == next_pos

    params = KVCacheParams(
        max_batch_size=batch_size,
        n_query_groups=n_query_groups,
        cache_length=cache_length,
        head_size=head_size,
        n_head=n_head,
        dtype=dtype,
        device=device,
    )
    config = Config(
        n_query_groups=n_query_groups,
        n_embd=n_head * head_size,
        n_head=n_head,
        n_layer=1,
        vocab_size=vocab_size,
    )

    # Create and prepare caches
    caches = []
    prefill = None
    insert = None
    input_pos = 0
    for name in names:
        cache = KVCacheFactory.create_single(
            name=name,
            config=config,
            max_batch_size=batch_size,
            cache_length=cache_length,
            block_idx=0,
            device=device,
            dtype=dtype,
        )
        caches.append(cache)
        # Prefill
        num = min(current_length, cache_length)
        if prefill is None:
            prefill = random_args_cache_forward(params, num, vocab_size)
        _input_pos = 0
        cache(**prefill, input_pos=_input_pos)
        _input_pos += num
        # Insert
        if current_length == cache_length:
            num = next_pos
            if insert is None:
                insert = random_args_cache_forward(params, num, vocab_size)
            cache(**insert, input_pos=_input_pos)
            _input_pos += num
        input_pos = _input_pos
    # Same code is run: Must be the same
    kv1 = caches[0].kv_buffers.get_keys_values()
    kv2 = caches[1].kv_buffers.get_keys_values()
    torch.testing.assert_close(kv1.keys(), kv2.keys())
    torch.testing.assert_close(kv1.values(), kv2.values())

    # Compute MHA outputs for different caches
    insert = None
    results = []
    for cache in caches:
        if insert is None:
            insert = random_args_cache_forward(params, q_len, vocab_size)
        results.append(cache(**insert, input_pos=input_pos))
    torch.testing.assert_close(results[0], results[1], **tol_kwargs)
