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
from keys_values.kvcache.test_utils import (
    create_kv_cache,
    tensor_is_simple,
    random_args_cache_forward,
    range_from_args,
    available_backends,
    product_with_devices,
)
from keys_values.kvcache.test_utils_advanced import cache_kwargs_for_smart_lastrec
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


# Default-buffer caches. Quantized buffers do not grow.
_DEFAULT_GROW_CACHE_NAMES = (
    "dense-default",
    "lastrec-default",
    "smart-lastrec-default",
    "h2o-default",
    "h2o-vlen-default",
    "h2o-orig-default",
)


def _physical_slots(cache) -> int:
    return cache.kv_buffers.k.shape[2]


def _assert_same_keys_values(growing, fixed):
    grown = growing.get_keys_values()
    full = fixed.get_keys_values()
    assert torch.equal(grown.keys(), full.keys())
    assert torch.equal(grown.values(), full.values())


def _grow_buffers_params(cache_length: int) -> KVCacheParams:
    return KVCacheParams(
        max_batch_size=3,
        n_query_groups=2,
        cache_length=cache_length,
        head_size=4,
        n_head=2,
        dtype=torch.float32,
    )


def _grow_cache_kwargs(name: str, grow_buffers_init_length=None, include_grow=True):
    kwargs = {}
    if include_grow:
        kwargs["grow_buffers_init_length"] = grow_buffers_init_length
    if name.startswith("lastrec-"):
        kwargs["init_grace_tokens"] = 0
    elif name.startswith("smart-lastrec"):
        kwargs.update(cache_kwargs_for_smart_lastrec())
    return kwargs


@pytest.mark.parametrize("name", _DEFAULT_GROW_CACHE_NAMES)
@pytest.mark.parametrize("device", available_backends())
def test_grow_buffers_init_length_none_allocates_full_length(device, name):
    cache_length = 16
    params = _grow_buffers_params(cache_length)
    omitted = create_kv_cache(
        name, params, **_grow_cache_kwargs(name, include_grow=False)
    )
    explicit = create_kv_cache(
        name,
        params,
        **_grow_cache_kwargs(name, grow_buffers_init_length=None),
    )
    data = random_args_cache_forward(params, 2, vocab_size=32, device=device)
    for cache in (omitted, explicit):
        assert cache.kv_buffers.grow_buffers_init_length is None
        cache(**range_from_args(data, 0, 1))
        cache(**range_from_args(data, 1, 2))
        assert _physical_slots(cache) == cache_length
        if name.startswith("lastrec-"):
            assert cache.token_pos.shape == (cache_length,)


@pytest.mark.parametrize("name", _DEFAULT_GROW_CACHE_NAMES)
@pytest.mark.parametrize("device", available_backends())
def test_grow_buffers_doubles_and_matches_full_cache(device, name):
    torch.manual_seed(0)
    init_length = 4
    cache_length = init_length * 4
    params = _grow_buffers_params(cache_length)
    growing = create_kv_cache(
        name,
        params,
        **_grow_cache_kwargs(name, grow_buffers_init_length=init_length),
    )
    fixed = create_kv_cache(
        name,
        params,
        **_grow_cache_kwargs(name, grow_buffers_init_length=None),
    )
    # Dense raises once `input_pos` reaches `cache_length`. The others keep
    # accepting tokens and must still match the full-size cache.
    num_insert = cache_length if name.startswith("dense") else cache_length + 4
    data = random_args_cache_forward(params, num_insert, vocab_size=32, device=device)

    # Prefill the initial slot count so a protected range (smart-lastrec)
    # lies inside the first chunk, and the first write still fits.
    prefill = range_from_args(data, 0, init_length)
    growing(**prefill)
    fixed(**prefill)
    assert _physical_slots(growing) == init_length
    assert _physical_slots(fixed) == cache_length
    if name.startswith("lastrec-"):
        assert growing.token_pos.shape == (cache_length,)
    _assert_same_keys_values(growing, fixed)

    seen = [init_length]
    for pos in range(init_length, num_insert):
        step = range_from_args(data, pos, pos + 1)
        growing(**step)
        fixed(**step)
        slots = _physical_slots(growing)
        if slots != seen[-1]:
            seen.append(slots)
        assert _physical_slots(fixed) == cache_length
        _assert_same_keys_values(growing, fixed)
    assert seen == [init_length, init_length * 2, cache_length]
    assert _physical_slots(growing) == cache_length


def test_grow_buffers_init_length_rejected_for_quantized_buffers():
    params = _grow_buffers_params(16)
    with pytest.raises(ValueError, match="default buffers only"):
        create_kv_cache(
            "lastrec-torch-quantized8",
            params,
            grow_buffers_init_length=4,
            init_grace_tokens=0,
        )


@pytest.mark.parametrize("device", available_backends())
def test_set_slots_doubles_when_write_does_not_fit(device):
    torch.manual_seed(1)
    init_length = 4
    cache_length = 32
    params = _grow_buffers_params(cache_length)
    cache = create_kv_cache(
        "lastrec-default",
        params,
        grow_buffers_init_length=init_length,
        init_grace_tokens=0,
    )
    # 4 -> 8 is still short of this write, so set_slots doubles twice.
    span = 8
    data = random_args_cache_forward(
        params, init_length + span, vocab_size=32, device=device
    )
    cache(**range_from_args(data, 0, init_length))
    assert _physical_slots(cache) == init_length
    prefix = cache.kv_buffers.k[:, :, :init_length, :].clone()
    step = range_from_args(data, init_length, init_length + span)
    cache.kv_buffers.forward(
        (init_length, init_length + span),
        step["key"],
        step["value"],
    )
    assert _physical_slots(cache) == init_length * 4
    assert torch.equal(cache.kv_buffers.k[:, :, :init_length, :], prefix)
