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
from typing import List, Tuple, Dict, Callable
from unittest import mock

import pytest
import torch

from keys_values.attention.sdpa_wrapper import (
    reorder_key_value,
    reorder_buffer_given_extra_info,
)
from keys_values.kvcache.attn_weights import AttnWeightsKVCache
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.basics import KVCacheWithBuffers
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.kvcache.smart_lastrec import SmartInitialLastRecentlyInsertedKVCache
from keys_values.kvcache.parallel.equalize import equalize_cache_content
from keys_values.kvcache.test_utils import (
    create_kv_cache,
    product_with_devices,
    random_args_cache_forward,
    range_from_args,
    random_token_positions,
)
from keys_values.kvcache.test_utils_advanced import cache_kwargs_for_smart_lastrec
from keys_values.utils import randint_torch, random_choices, index_to_3d


def args_equalize_before_after() -> Tuple[str, List[tuple]]:
    names = [
        (name, dict())
        for name in KVCacheFactory.supported_names()
        if name.endswith("-default") and not name.startswith("dense")
    ]
    return product_with_devices(names[1:2], "name, kwargs")


def _extract_content(
    kv_caches: List[KVCacheWithBuffers],
) -> Dict[str, torch.Tensor]:
    essentially_1d = kv_caches[0].active_dimensions() == ()
    parts = {
        "key": [],
        "value": [],
        "token_pos": [],
    }
    for kv_cache in kv_caches:
        k_and_v = kv_cache.kv_buffers.get_keys_values()
        parts["key"].append(k_and_v.keys())
        parts["value"].append(k_and_v.values())
        tp = kv_cache.token_positions()
        parts["token_pos"].append(
            tp[0, 0, :] if essentially_1d else tp
        )
    return {
        k: torch.cat(v, dim=0 if k == "token_pos" else 2)
        for k, v in parts.items()
    }


def _compare_contents(
    contents: List[Dict[str, torch.Tensor]],
    essentially_1d: bool,
) -> None:
    sorted_contents = []
    for content in contents:
        token_pos = content["token_pos"]
        if essentially_1d:
            token_pos = index_to_3d(token_pos, *content["key"].shape[:2])
        sorted_key, sorted_value, extra_info = reorder_key_value(
            key=content["key"],
            value=content["value"],
            token_positions=token_pos,
            input_pos=8,  # not used
            q_len=4,  # not used
            sort_if_3d=True,
        )
        if essentially_1d:
            sorted_token_pos = token_pos[0, 0, extra_info["sort_index"]]
        else:
            sorted_token_pos = reorder_buffer_given_extra_info(
                buffer=token_pos, **extra_info,
            )
        sorted_contents.append(
            {
                "key": sorted_key,
                "value": sorted_value,
                "token_pos": sorted_token_pos,
            }
        )
    for name in ("token_pos", "key", "value"):
        print(name)
        torch.testing.assert_close(sorted_contents[0][name], sorted_contents[1][name])


# TODO: Variant where communications run in parallel
@pytest.mark.parametrize(*args_equalize_before_after())
def test_equalize_before_after(device, name, kwargs):
    """
    We mock the equalization, where communications run in parallel. Here,
    we call :func:`equalize_cache_content` sequentially for each rank, and
    we repeat this twice.
    - First phase: `_receive` does nothing, and its callback returns False, so
        results are not received or written back. `_send` stores all transfers
        in a dictionary
    - Second phase: `_send` does nothing. `_receive` fetches data from the
        dictionary, its callback returns True.

    """
    seed = 31415927
    torch.random.manual_seed(seed)
    vocab_size = 128
    dtype = torch.bfloat16
    num_devices = 8
    index_kwargs = dict(dtype=torch.int64, device=device)

    params = KVCacheParams(
        max_batch_size=3,
        n_query_groups=4,
        cache_length=32,
        head_size=8,
        n_head=4,
        dtype=dtype,
    )
    cache_length = params.cache_length
    virtual_length = cache_length * num_devices
    if name.startswith("smart-lastrec"):
        kwargs = {**kwargs, **cache_kwargs_for_smart_lastrec()}
    kv_caches = [
        create_kv_cache(name, params, **kwargs) for _ in range(num_devices)
    ]
    active_dimensions = kv_caches[0].active_dimensions()
    essentially_1d = active_dimensions == ()

    q_len = randint_torch(virtual_length // 8, (virtual_length * 3) // 4)
    input_pos = randint_torch(virtual_length, virtual_length * 2)
    data = random_args_cache_forward(
        params,
        virtual_length,
        vocab_size,
        device=device,
    )
    # Prefill caches
    start = 0
    for i, kv_cache in enumerate(kv_caches):
        end = start + cache_length
        kv_cache(**range_from_args(data, start, end))
        start = end
    # Set token_positions
    all_token_positions = random_token_positions(
        batch_size=params.max_batch_size,
        n_query_groups=params.n_query_groups,
        cache_length=virtual_length,
        input_pos=input_pos,
        essentially_1d=essentially_1d,
        device=device,
    )
    for off, kv_cache in enumerate(kv_caches):
        sel_ind = torch.arange(off, virtual_length, num_devices, **index_kwargs)
        assert sel_ind.numel() == cache_length  # Sanity check
        if essentially_1d:
            kv_cache.set_token_positions(
                index=torch.arange(cache_length, **index_kwargs),
                tp_values=all_token_positions[0, 0, sel_ind],
            )
        elif isinstance(kv_cache, AttnWeightsKVCache):
            kv_cache.token_pos.copy_(all_token_positions[:, :, sel_ind])
        elif isinstance(kv_cache, SmartInitialLastRecentlyInsertedKVCache):
            kv_cache.token_pos.copy_(all_token_positions[:, 0, sel_ind])
        else:
            raise NotImplementedError(f"type(kv_cache) = {type(kv_cache)} not supported")
    # Content before equalization
    contents = [_extract_content(kv_caches)]

    # Equalization (parallel computation is mocked, see above)
    shape = (params.max_batch_size, params.n_query_groups, q_len)
    inner_shape = tuple(shape[i] for i in active_dimensions + (2,))
    _inner = random_choices(inner_shape, size_range=virtual_length, device=device)
    view_dims = list(shape)
    exp_dims = list(shape[:-1]) + [-1]
    for i in range(2):
        if i in active_dimensions:
            exp_dims[i] = -1
        else:
            view_dims[i] = 1
    overwrite_pos = _inner.view(*view_dims).expand(*exp_dims)

    # Mock proceeds in two phases. First phase 1 (send)
    source_stats = []
    target_stats = []
    communications = dict()

    def mock_send_phase1(
        src_rank: int,
        trg_rank: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        token_pos: torch.Tensor,
    ) -> Callable[[], bool]:
        communications[(src_rank, trg_rank)] = {
            "keys": keys,
            "values": values,
            "token_pos": token_pos,
        }
        return lambda: True

    def mock_receive_phase1(
        src_rank: int,
        trg_rank: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        token_pos: torch.Tensor,
    ) -> Callable[[], bool]:
        return lambda: False

    with mock.patch("keys_values.kvcache.parallel.equalize._send", mock_send_phase1):
        with mock.patch("keys_values.kvcache.parallel.equalize._receive", mock_receive_phase1):
            for rank, kv_cache in enumerate(kv_caches):
                _source_stats, _ = equalize_cache_content(
                    rank=rank,
                    num_devices=num_devices,
                    input_pos=input_pos,
                    kv_cache=kv_cache,
                    overwrite_pos=overwrite_pos,
                )
                source_stats.append(_source_stats)

    # Second phase 2 (receive)
    def mock_send_phase2(
        src_rank: int,
        trg_rank: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        token_pos: torch.Tensor,
    ) -> Callable[[], bool]:
        return lambda: True

    def mock_receive_phase2(
        src_rank: int,
        trg_rank: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        token_pos: torch.Tensor,
    ) -> Callable[[], bool]:
        entry = communications[(src_rank, trg_rank)]
        keys.copy_(entry["keys"])
        values.copy_(entry["values"])
        token_pos.copy_(entry["token_pos"])
        return lambda: True

    with mock.patch("keys_values.kvcache.parallel.equalize._send", mock_send_phase2):
        with mock.patch("keys_values.kvcache.parallel.equalize._receive", mock_receive_phase2):
            for rank, kv_cache in enumerate(kv_caches):
                _, _target_stats = equalize_cache_content(
                    rank=rank,
                    num_devices=num_devices,
                    input_pos=input_pos,
                    kv_cache=kv_cache,
                    overwrite_pos=overwrite_pos,
                )
                target_stats.append(_target_stats)

    for rank, (src, trg) in enumerate(zip(source_stats, target_stats)):
        num_sent = sum(src.values())
        num_recv = sum(trg.values())
        print(f"{rank}: Sent {num_sent:3d}, received {num_recv:3d}")

    contents.append(_extract_content(kv_caches))
    _compare_contents(
        contents=contents,
        essentially_1d=essentially_1d,
    )
