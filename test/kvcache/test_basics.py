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

import torch
import pytest

from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.basics import DEFAULT_INIT_GRACE_TOKENS
from keys_values.kvcache.buffers import DEFAULT_GROW_BUFFERS_INITIAL_SLOTS
from keys_values.kvcache.test_utils import (
    create_kv_cache,
    tensor_is_simple,
    random_args_cache_forward,
    range_from_args,
    available_backends,
    product_with_devices,
)
from keys_values.utils import randint_torch, index_to_3d, is_index_1d


@pytest.mark.parametrize(
    "device, name",
    product(
        available_backends(),
        ["lastrec-default", "lastrec-torch-quantized8"],
    ),
)
def test_last_recent(device, name):
    seed = 31415927
    torch.random.manual_seed(seed)
    vocab_size = 128
    dtype = torch.bfloat16

    params = KVCacheParams(
        max_batch_size=3,
        n_query_groups=4,
        cache_length=32,
        head_size=8,
        n_head=4,
        dtype=dtype,
    )
    cache_length = params.cache_length
    kv_cache = create_kv_cache(name, params)
    assert kv_cache.init_grace_tokens == min(
        DEFAULT_INIT_GRACE_TOKENS,
        max(params.cache_length // 8, 1),
    )
    num_insert = randint_torch(cache_length, 3 * cache_length)
    max_prefill_length = kv_cache.max_prefill_length
    num_prefill = randint_torch(num_insert // 3, int(num_insert * 0.75))
    if max_prefill_length is not None and num_prefill > max_prefill_length:
        num_prefill = max_prefill_length

    data = random_args_cache_forward(
        params,
        num_insert,
        vocab_size,
        device=device,
    )
    kv_cache(**range_from_args(data, 0, num_prefill))
    for pos in range(num_prefill, num_insert):
        kv_cache(**range_from_args(data, pos, pos + 1))

    current_length = min(cache_length, num_insert)
    assert kv_cache.current_length == current_length
    token_positions = kv_cache.token_positions().to(dtype=torch.int64)
    assert token_positions.shape == (
        params.max_batch_size,
        params.n_query_groups,
        current_length,
    )
    assert tensor_is_simple(token_positions)
    positions = token_positions[0, 0, :].tolist()
    assert len(set(positions)) == current_length
    assert all(
        x < kv_cache.init_grace_tokens or num_insert - current_length <= x < num_insert
        for x in positions
    )


@pytest.mark.parametrize(
    *product_with_devices(
        [
            (torch.bfloat16, dict(atol=0.0005, rtol=0.03)),
            (torch.float16, dict(atol=0.00015, rtol=0.01)),
            (torch.float32, dict()),
        ],
        "dtype, tol_kwargs",
    ),
)
def test_incremental_versus_singlepass(device, dtype, tol_kwargs):
    seed = 31415927
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
        dtype=dtype,
    )
    cache_length = params.cache_length
    kv_cache = create_kv_cache(name, params)
    num_prefill = randint_torch(cache_length // 3, int(cache_length * 0.75))
    max_prefill_length = kv_cache.max_prefill_length
    if max_prefill_length is not None and num_prefill > max_prefill_length:
        num_prefill = max_prefill_length
    num_insert = max_prefill_length if max_prefill_length is not None else cache_length

    data = random_args_cache_forward(
        params,
        num_insert,
        vocab_size,
        device=device,
    )
    # Compute MHA in a single shot
    y_sshot = kv_cache(**data)
    should_be = (
        torch.arange(
            num_insert,
            dtype=kv_cache.token_positions().dtype,
            device=device,
        )
        .view(1, 1, -1)
        .expand(params.max_batch_size, params.n_query_groups, -1)
    )
    assert (should_be == kv_cache.token_positions()[:, :, :num_insert]).all().item()
    # Compute MHA in steps
    kv_cache.reset()
    y_parts = [kv_cache(**range_from_args(data, 0, num_prefill))] + [
        kv_cache(**range_from_args(data, pos, pos + 1))
        for pos in range(num_prefill, num_insert)
    ]

    assert kv_cache.current_length == num_insert
    assert (should_be == kv_cache.token_positions()[:, :, :num_insert]).all().item()
    print(f"0:{num_prefill}")
    torch.testing.assert_close(
        y_parts[0],
        y_sshot[:, :num_prefill, :],
        **tol_kwargs,
    )
    # Incremental computation is not very close to single-shot for 16-bit
    # data types. This is because different code is used (PyTorch kernels with
    # `is_causal=True` for single-shot, own code for incremental)
    if dtype == torch.float32:
        for pos, yp in zip(range(num_prefill, num_insert), y_parts[1:]):
            print(f"{pos}:{pos + 1}")
            torch.testing.assert_close(
                yp,
                y_sshot[:, pos : (pos + 1), :],
                **tol_kwargs,
            )


def test_index_to_3d():
    for shape in [
        (2, 3, 4),
        (1, 2, 3),
        (2, 1, 3),
        (1, 1, 2),
        (3, 1, 1),
        (1, 2, 1),
        (1, 1, 1),
    ]:
        size = shape[2]
        index = torch.arange(size)
        result = index_to_3d(index, *shape[:-1])
        assert is_index_1d(result), ("type 1", shape, result.shape, result.stride())
        if size > 1:
            # Can change the 1D row, tensor remains 1D
            result[0, 0, :size] = 321
            assert is_index_1d(result), ("type 3", shape, result.shape, result.stride())
        index = torch.arange(size * 2)[:size]
        result = index_to_3d(index, *shape[:-1])
        assert is_index_1d(result), ("type 2", shape, result.shape, result.stride())


def _physical_slots(cache) -> int:
    return cache.kv_buffers.k.shape[2]


def _assert_same_keys_values(growing, fixed):
    grown = growing.get_keys_values()
    full = fixed.get_keys_values()
    torch.testing.assert_close(grown.keys(), full.keys())
    torch.testing.assert_close(grown.values(), full.values())


def _grow_buffers_params(cache_length: int) -> KVCacheParams:
    return KVCacheParams(
        max_batch_size=3,
        n_query_groups=2,
        cache_length=cache_length,
        head_size=4,
        n_head=2,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("device", available_backends())
def test_grow_buffers_doubles_until_cache_length(device):
    torch.manual_seed(0)
    initial = DEFAULT_GROW_BUFFERS_INITIAL_SLOTS
    cache_length = initial * 4
    params = _grow_buffers_params(cache_length)
    growing = create_kv_cache(
        "lastrec-default", params, grow_buffers=True, init_grace_tokens=0
    )
    fixed = create_kv_cache(
        "lastrec-default", params, grow_buffers=False, init_grace_tokens=0
    )
    num_insert = cache_length + 4
    data = random_args_cache_forward(params, num_insert, vocab_size=32, device=device)

    prefill = range_from_args(data, 0, 1)
    growing(**prefill)
    fixed(**prefill)
    assert _physical_slots(growing) == initial
    assert _physical_slots(fixed) == cache_length
    assert growing.token_pos.shape == (cache_length,)
    assert fixed.token_pos.shape == (cache_length,)
    _assert_same_keys_values(growing, fixed)

    seen = [initial]
    for pos in range(1, num_insert):
        step = range_from_args(data, pos, pos + 1)
        growing(**step)
        fixed(**step)
        slots = _physical_slots(growing)
        if slots != seen[-1]:
            seen.append(slots)
        assert _physical_slots(fixed) == cache_length
        _assert_same_keys_values(growing, fixed)
    assert seen == [initial, initial * 2, cache_length]
    assert _physical_slots(growing) == cache_length


@pytest.mark.parametrize("device", available_backends())
def test_grow_buffers_prefill_longer_than_initial(device):
    torch.manual_seed(1)
    initial = DEFAULT_GROW_BUFFERS_INITIAL_SLOTS
    cache_length = initial * 4
    prefill_length = initial + 4
    params = _grow_buffers_params(cache_length)
    growing = create_kv_cache("lastrec-default", params, grow_buffers=True)
    fixed = create_kv_cache("lastrec-default", params, grow_buffers=False)
    data = random_args_cache_forward(
        params, prefill_length + 1, vocab_size=32, device=device
    )
    prefill = range_from_args(data, 0, prefill_length)
    growing(**prefill)
    fixed(**prefill)
    assert _physical_slots(growing) == initial * 2
    assert _physical_slots(fixed) == cache_length
    _assert_same_keys_values(growing, fixed)
    step = range_from_args(data, prefill_length, prefill_length + 1)
    growing(**step)
    fixed(**step)
    assert _physical_slots(growing) == initial * 2
    _assert_same_keys_values(growing, fixed)


@pytest.mark.parametrize("device", available_backends())
def test_grow_buffers_off_allocates_full_length(device):
    cache_length = DEFAULT_GROW_BUFFERS_INITIAL_SLOTS * 4
    params = _grow_buffers_params(cache_length)
    cache = create_kv_cache("lastrec-default", params, init_grace_tokens=0)
    data = random_args_cache_forward(params, 2, vocab_size=32, device=device)
    cache(**range_from_args(data, 0, 1))
    assert not cache.kv_buffers.grow_buffers
    assert _physical_slots(cache) == cache_length
    cache(**range_from_args(data, 1, 2))
    assert _physical_slots(cache) == cache_length
    assert cache.token_pos.shape == (cache_length,)


@pytest.mark.parametrize("device", available_backends())
def test_grow_buffers_batch_size_change(device):
    torch.manual_seed(2)
    initial = DEFAULT_GROW_BUFFERS_INITIAL_SLOTS
    cache_length = initial * 4
    params = _grow_buffers_params(cache_length)
    growing = create_kv_cache(
        "lastrec-default", params, grow_buffers=True, init_grace_tokens=0
    )
    fixed = create_kv_cache(
        "lastrec-default", params, grow_buffers=False, init_grace_tokens=0
    )

    first = random_args_cache_forward(params, 8, vocab_size=32, device=device)
    first = {name: tensor[:1] for name, tensor in first.items()}
    growing(**range_from_args(first, 0, 4))
    fixed(**range_from_args(first, 0, 4))
    for pos in range(4, 8):
        step = range_from_args(first, pos, pos + 1)
        growing(**step)
        fixed(**step)
    assert growing.kv_buffers.k.shape[0] == 1
    assert _physical_slots(growing) == initial
    _assert_same_keys_values(growing, fixed)

    growing.reset()
    fixed.reset()
    second_len = initial + 4
    second = random_args_cache_forward(
        params, second_len + 3, vocab_size=32, device=device
    )
    second = {name: tensor[:2] for name, tensor in second.items()}
    growing(**range_from_args(second, 0, second_len))
    fixed(**range_from_args(second, 0, second_len))
    assert growing.kv_buffers.k.shape[0] == 2
    assert fixed.kv_buffers.k.shape[0] == 2
    assert _physical_slots(growing) == initial * 2
    assert _physical_slots(fixed) == cache_length
    _assert_same_keys_values(growing, fixed)
    for pos in range(second_len, second_len + 3):
        step = range_from_args(second, pos, pos + 1)
        growing(**step)
        fixed(**step)
        _assert_same_keys_values(growing, fixed)
    assert growing.kv_buffers.k.shape[0] == 2
    assert growing.token_pos.shape == (cache_length,)
