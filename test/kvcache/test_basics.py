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

import torch
import pytest

from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.test_utils import (
    create_kv_cache,
    tensor_is_simple,
    random_args_cache_forward,
    range_from_args,
    available_backends,
    product_with_devices,
)


@pytest.mark.parametrize(
    "name, device",
    product(
        ["lastrec-default", "lastrec-torch-quantized8"],
        available_backends(),
    )
)
def test_last_recent(name, device):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    vocab_size = 128
    dtype = torch.bfloat16

    params = KVCacheParams(
        max_batch_size=3,
        n_query_groups=4,
        cache_length=32,
        head_size=8,
        n_head=4,
        device=device,
        dtype=dtype,
    )
    cache_length = params.cache_length
    kv_cache = create_kv_cache(name, params)
    num_insert = random.randint(cache_length, 3 * cache_length)
    max_prefill_length = kv_cache.max_prefill_length
    num_prefill = random.randint(num_insert // 3, int(num_insert * 0.75))
    if max_prefill_length is not None and num_prefill > max_prefill_length:
        num_prefill = max_prefill_length

    data = random_args_cache_forward(params, num_insert, vocab_size)
    kv_cache(**range_from_args(data, 0, num_prefill), input_pos=0)
    for pos in range(num_prefill, num_insert):
        kv_cache(**range_from_args(data, pos, pos + 1), input_pos=pos)

    current_length = min(cache_length, num_insert)
    assert kv_cache.current_length == current_length
    token_positions = kv_cache.token_positions().to(dtype=torch.int64)
    assert token_positions.shape == (params.max_batch_size, params.n_query_groups, current_length)
    assert tensor_is_simple(token_positions)
    positions = token_positions[0, 0, :].tolist()
    assert len(set(positions)) == current_length
    assert all(num_insert - current_length <= x < num_insert for x in positions)


@pytest.mark.parametrize(
    "dtype, tol_kwargs, device",
    product_with_devices(
        [
            (torch.bfloat16, dict(atol=0.0005, rtol=0.03)),
            (torch.float16, dict(atol=0.00015, rtol=0.01)),
            (torch.float32, dict()),
        ],
    ),
)
def test_incremental_versus_singlepass(dtype, tol_kwargs, device):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    vocab_size = 128
    name = "dense-default"
    print(f"dtype = {dtype}, device = {device}")

    params = KVCacheParams(
        max_batch_size=3,
        n_query_groups=2,
        cache_length=128,
        head_size=8,
        n_head=4,
        device=device,
        dtype=dtype,
    )
    cache_length = params.cache_length
    kv_cache = create_kv_cache(name, params)
    num_prefill = random.randint(cache_length // 3, int(cache_length * 0.75))
    max_prefill_length = kv_cache.max_prefill_length
    if max_prefill_length is not None and num_prefill > max_prefill_length:
        num_prefill = max_prefill_length
    num_insert = max_prefill_length if max_prefill_length is not None else cache_length

    data = random_args_cache_forward(params, num_insert, vocab_size)
    # Compute MHA in a single shot
    y_sshot = kv_cache(**data, input_pos=0)
    should_be = torch.arange(
        num_insert, dtype=kv_cache.token_positions().dtype, device=device,
    ).view(1, 1, -1).expand(params.max_batch_size, params.n_query_groups, -1)
    assert (should_be == kv_cache.token_positions()[:, :, :num_insert]).all().item()
    # Compute MHA in steps
    y_parts = []
    y_parts.append(
        kv_cache(**range_from_args(data, 0, num_prefill), input_pos=0)
    )
    for pos in range(num_prefill, num_insert):
        y_parts.append(
            kv_cache(**range_from_args(data, pos, pos + 1), input_pos=pos)
        )

    assert kv_cache.current_length == num_insert
    assert (should_be == kv_cache.token_positions()[:, :, :num_insert]).all().item()
    print(f"0:{num_prefill}")
    torch.testing.assert_close(
        y_parts[0], y_sshot[:, :num_prefill, :], **tol_kwargs,
    )
    # Incremental computation is not very close to single-shot for 16-bit
    # data types. This is because different code is used (PyTorch kernels with
    # `is_causal=True` for single-shot, own code for incremental)
    if dtype == torch.float32:
        for pos, yp in zip(range(num_prefill, num_insert), y_parts[1:]):
            print(f"{pos}:{pos + 1}")
            torch.testing.assert_close(
                yp, y_sshot[:, pos:(pos + 1), :], **tol_kwargs,
            )
