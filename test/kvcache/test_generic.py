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
from typing import List
from itertools import product
import random
import re

import torch
import pytest

from litgpt.config import Config

from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.factory import KVCacheFactory, split_name
from keys_values.kvcache.quantize.bitsandbytes import determine_blocksize
from keys_values.kvcache.test_utils import (
    create_kv_cache,
    tensor_is_simple,
    random_keys_values,
    random_tensor,
    device_for_cache_name,
    cache_name_gpu_only,
)


def args_store_retrieve() -> List[tuple]:
    return [
        (name, dict()) for name in KVCacheFactory.supported_names()
        if name.endswith("-default")
    ] + [
        ("h2o-default", dict(grace_period=3)),
        ("h2o-vlen-default", dict(grace_period=3)),
    ]


@pytest.mark.parametrize("name, kwargs", args_store_retrieve())
def test_store_retrieve(name, kwargs):
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
    kv_cache = create_kv_cache(name, params, **kwargs)
    if name.startswith("dense"):
        num_insert = random.randint(cache_length // 2, cache_length)
    else:
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
    if "h2o" not in name:
        assert tensor_is_simple(token_positions)
    # Positions for every (b, h) must be different
    for b, h in zip(range(params.max_batch_size), range(params.n_query_groups)):
        token_pos = token_positions[b, h, :].tolist()
        assert all(0 <= x < num_insert for x in token_pos)
        err_msg = f"num_insert = {num_insert}, b = {b}, h = {h}, current_length = {current_length}, num_prefill = {num_prefill}"
        assert len(set(token_pos)) == current_length, err_msg
    # Test cache content slice by slice
    keys_and_values = kv_cache.get_keys_values()
    for pos in range(current_length):
        index = token_positions[:, :, pos][:, :, None, None].expand(-1, -1, 1, params.head_size)
        # `index[i, j, 0, k] = next_position[i, j]`
        k_expected = keys.gather(-2, index).squeeze(-2)
        v_expected = values.gather(-2, index).squeeze(-2)
        torch.testing.assert_close(
            k_expected, keys_and_values.keys()[:, :, pos, :]
        )
        torch.testing.assert_close(
            v_expected, keys_and_values.values()[:, :, pos, :]
        )


def args_prefill() -> List[str]:
    if torch.cuda.is_available():
        return KVCacheFactory.supported_names()
    else:
        return [
            name
            for name in KVCacheFactory.supported_names()
            if not cache_name_gpu_only(name)
        ]


@pytest.mark.parametrize("name", args_prefill())
def test_prefill(name):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    num_compares = 3
    vocab_size = 128
    device = device_for_cache_name(name)

    params = KVCacheParams(
        max_batch_size=2,
        n_query_groups=2,
        cache_length=32,
        head_size=64,
        n_head=2,
        device=device,
        dtype=torch.bfloat16,
    )
    cache_length = params.cache_length
    kv_cache = create_kv_cache(name, params)

    keys, values = random_keys_values(params, num=cache_length)
    queries = random_tensor(params, num=cache_length, is_query=True)
    token_idx = torch.randint(
        low=0,
        high=vocab_size,
        size=(params.max_batch_size, cache_length),
    )
    keys_cached = []
    values_cached = []
    max_prefill_length = kv_cache.max_prefill_length
    for _ in range(num_compares):
        num_prefill = random.randint(cache_length // 8, cache_length)
        if max_prefill_length is not None and num_prefill > max_prefill_length:
            num_prefill = max_prefill_length
        kv_cache(
            query=queries[:, :, :num_prefill, :],
            key=keys[:, :, :num_prefill, :],
            value=values[:, :, :num_prefill, :],
            token_idx=token_idx[:, :num_prefill],
            input_pos=0,
        )
        for pos in range(num_prefill, cache_length):
            kv_cache(
                query=queries[:, :, pos:(pos + 1), :],
                key=keys[:, :, pos:(pos + 1), :],
                value=values[:, :, pos:(pos + 1), :],
                token_idx=token_idx[:, pos:(pos + 1)],
                input_pos=pos,
            )
        keys_and_values = kv_cache.get_keys_values()
        if keys_and_values is not None:
            keys_cached.append(keys_and_values.keys().clone())
            values_cached.append(keys_and_values.values().clone())
        else:
            keys_cached.append(None)
            values_cached.append(None)

    num_none = 0
    for k, v in zip(keys_cached[1:], values_cached[1:]):
        if k is not None:
            torch.testing.assert_close(k, keys_cached[0])
            torch.testing.assert_close(v, values_cached[0])
        else:
            num_none += 1
    assert num_none < num_compares - 1


def _normalize_key(name: str) -> str:
    regex = r"layer[0-9]*_(.*)$"
    result = re.match(regex, name)
    if result is not None:
        name = "layer_" + result.group(1)
    return name


def _filter_func(record: tuple) -> bool:
    name, batch_size, _, (_, n_query_groups), head_size, _, _ = record
    shape = (batch_size, n_query_groups, 1, head_size)
    return cache_name_gpu_only(name) or determine_blocksize(shape) is not None


# 480 cases on CPU, 1056 cases on GPU
def args_size_estimate() -> List[tuple]:
    excludes = {"h2o-vlen", "qh2o-vlen", "h2o-orig"}
    names = [
        name for name in args_prefill() if split_name(name)[0] not in excludes
    ]  # 7 on CPU, 15 on GPU
    batch_sizes = [1, 3]  # 2
    cache_lengths = [32, 28]  # 2
    n_head_groups = [(4, 2), (4, 4), (8, 1)]  # 3
    head_sizes = [16, 32]  # 2
    dtypes = [torch.bfloat16, torch.float32]  # 2
    boh_lst = [False, True]  # 2
    result = [
        record[:3] + record[3] + record[4:]
        for record in product(
            names,
            batch_sizes,
            cache_lengths,
            n_head_groups,
            head_sizes,
            dtypes,
            boh_lst
        )
        if _filter_func(record)
    ]
    return result


@pytest.mark.parametrize(
    "name, batch_size, cache_length, n_head, n_query_groups, head_size, dtype, blocks_over_heads",
    args_size_estimate(),
)
def test_size_estimate(
    name,
    batch_size,
    cache_length,
    n_head,
    n_query_groups,
    head_size,
    dtype,
    blocks_over_heads,
):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    vocab_size = 128
    n_layer = 4
    device = device_for_cache_name(name)

    params = KVCacheParams(
        max_batch_size=batch_size,
        n_query_groups=n_query_groups,
        cache_length=cache_length,
        head_size=head_size,
        n_head=n_head,
        device=device,
        dtype=dtype,
    )
    config = Config(
        n_query_groups=n_query_groups,
        n_embd=n_head * head_size,
        n_head=n_head,
        n_layer=n_layer,
        vocab_size=vocab_size,
    )

    if name.endswith("default"):
        cache_kwargs = dict()
    else:
        cache_kwargs = dict(blocks_over_heads=blocks_over_heads)
    kv_caches = [
        KVCacheFactory.create_single(
            name=name,
            config=config,
            max_batch_size=batch_size,
            cache_length=cache_length,
            block_idx=block_idx,
            device=device,
            dtype=dtype,
            cache_kwargs=cache_kwargs,
        )
        for block_idx in range(config.n_layer)
    ]

    # Need to prefill caches so that `size_estimate` works
    keys, values = random_keys_values(params, num=cache_length)
    queries = random_tensor(params, num=cache_length, is_query=True)
    token_idx = torch.randint(
        low=0,
        high=vocab_size,
        size=(params.max_batch_size, cache_length),
    )
    max_prefill_length = kv_caches[0].max_prefill_length
    for kv_cache in kv_caches:
        kv_cache(
            query=queries[:, :, :max_prefill_length, :],
            key=keys[:, :, :max_prefill_length, :],
            value=values[:, :, :max_prefill_length, :],
            token_idx=token_idx[:, :max_prefill_length],
            input_pos=0,
        )
    num_bits_total1, bits_by_part1 = KVCacheFactory.size_estimate(kv_caches)
    num_bits_total2, bits_by_part2 = KVCacheFactory.size_estimate_apriori(
        name=name,
        config=config,
        max_batch_size=batch_size,
        cache_length=cache_length,
        device=device,
        dtype=dtype,
        cache_kwargs=cache_kwargs,
    )
    print(f"name={name}, batch_size={batch_size}, cache_length={cache_length}, n_head={n_head}, n_query_groups={n_query_groups}, head_size={head_size}, dtype={dtype}, blocks_over_heads={blocks_over_heads}")
    print(bits_by_part1)
    print(bits_by_part2)
    assert num_bits_total1 == num_bits_total2
    # Some entries in `bits_by_part1` have names "layer<l>_*" for layer
    # numbers "<l>". In `bits_by_part2`, there is only one corresponding
    # entry "layer_*" with the sum of sizes.
    bits_by_part1_accum = {k: 0 for k in bits_by_part2.keys()}
    for k, v in bits_by_part1.items():
        bits_by_part1_accum[_normalize_key(k)] += v
    for k, v in bits_by_part2.items():
        assert bits_by_part1_accum[k] == v, (k, v, bits_by_part1_accum[k])
