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

import torch
import pytest

from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.test_utils import (
    create_kv_cache,
    tensor_is_simple,
    random_keys_values,
    random_tensor,
)


@pytest.mark.parametrize(
    "name", ["lastrec-default", "lastrec-torch-quantized8"],
)
def test_last_recent(name):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    vocab_size = 128

    params = KVCacheParams(
        max_batch_size=3,
        n_query_groups=4,
        cache_length=32,
        head_size=8,
        n_head=4,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    cache_length = params.cache_length
    kv_cache = create_kv_cache(name, params)
    num_insert = random.randint(cache_length, 3 * cache_length)
    max_prefill_length = kv_cache.max_prefill_length
    num_prefill = random.randint(num_insert // 3, int(num_insert * 0.75))
    if max_prefill_length is not None and num_prefill > max_prefill_length:
        num_prefill = max_prefill_length

    keys, values = random_keys_values(params, num=num_insert)
    queries = random_tensor(params, num=num_insert, is_query=True)
    token_idx = torch.randint(
        low=0,
        high=vocab_size,
        size=(params.max_batch_size, num_insert),
    )
    kv_cache(
        query=queries[:, :, :num_prefill, :],
        key=keys[:, :, :num_prefill, :],
        value=values[:, :, :num_prefill, :],
        token_idx=token_idx[:, :num_prefill],
        input_pos=0,
    )
    for pos in range(num_prefill, num_insert):
        kv_cache(
            query=queries[:, :, pos:(pos + 1), :],
            key=keys[:, :, pos:(pos + 1), :],
            value=values[:, :, pos:(pos + 1), :],
            token_idx=token_idx[:, pos:(pos + 1)],
            input_pos=pos,
        )

    current_length = min(cache_length, num_insert)
    assert kv_cache.current_length == current_length
    token_positions = kv_cache.token_positions().to(dtype=torch.int64)
    assert token_positions.shape == (params.max_batch_size, params.n_query_groups, current_length)
    assert tensor_is_simple(token_positions)
    positions = token_positions[0, 0, :].tolist()
    assert len(set(positions)) == current_length
    assert all(num_insert - current_length <= x < num_insert for x in positions)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32],
)
def test_incremental_versus_singlepass(dtype):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    vocab_size = 128
    name = "dense-default"

    params = KVCacheParams(
        max_batch_size=3,
        n_query_groups=2,
        cache_length=128,
        head_size=8,
        n_head=4,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    cache_length = params.cache_length
    kv_cache = create_kv_cache(name, params)
    num_prefill = random.randint(cache_length // 3, int(cache_length * 0.75))
    max_prefill_length = kv_cache.max_prefill_length
    if max_prefill_length is not None and num_prefill > max_prefill_length:
        num_prefill = max_prefill_length
    num_insert = max_prefill_length if max_prefill_length is not None else cache_length

    keys, values = random_keys_values(params, num=num_insert)
    queries = random_tensor(params, num=num_insert, is_query=True)
    token_idx = torch.randint(
        low=0,
        high=vocab_size,
        size=(params.max_batch_size, num_insert),
    )
    # Compute MHA in a single shot
    y_sshot = kv_cache(
        query=queries,
        key=keys,
        value=values,
        token_idx=token_idx,
        input_pos=0,
    )
    should_be = torch.arange(
        num_insert, dtype=kv_cache.token_positions().dtype,
    ).view(1, 1, -1).expand(params.max_batch_size, params.n_query_groups, -1)
    assert (should_be == kv_cache.token_positions()[:, :, :num_insert]).all().item()
    # Compute MHA in steps
    y_parts = []
    y_parts.append(
        kv_cache(
            query=queries[:, :, :num_prefill, :],
            key=keys[:, :, :num_prefill, :],
            value=values[:, :, :num_prefill, :],
            token_idx=token_idx[:, :num_prefill],
            input_pos=0,
        )
    )
    for pos in range(num_prefill, num_insert):
        y_parts.append(
            kv_cache(
                query=queries[:, :, pos:(pos + 1), :],
                key=keys[:, :, pos:(pos + 1), :],
                value=values[:, :, pos:(pos + 1), :],
                token_idx=token_idx[:, pos:(pos + 1)],
                input_pos=pos,
            )
        )

    assert kv_cache.current_length == num_insert
    assert (should_be == kv_cache.token_positions()[:, :, :num_insert]).all().item()
    print(f"0:{num_prefill}")
    torch.testing.assert_close(y_parts[0], y_sshot[:, :num_prefill, :])
    # Incremental computation is not very close to single-shot for 16-bit
    # data types. This is because different code is used (PyTorch kernels with
    # `is_causal=True` for single-shot, own code for incremental)
    if dtype == torch.float32:
        for pos, yp in zip(range(num_prefill, num_insert), y_parts[1:]):
            print(f"{pos}:{pos + 1}")
            torch.testing.assert_close(yp, y_sshot[:, pos:(pos + 1), :])
