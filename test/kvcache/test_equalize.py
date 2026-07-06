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
from collections import Counter
from functools import partial
from typing import List, Tuple, Dict, Callable, Set
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
from keys_values.kvcache.parallel.equalize import (
    equalize_cache_content,
    _get_delta_per_rank,
    _get_communication_plan,
    _get_allocations,
    _get_q_len_for_rank,
    _append_local_overwrite_pos,
    _remove_local_overwrite_pos,
)
from keys_values.kvcache.test_utils import (
    create_kv_cache,
    product_with_devices,
    random_args_cache_forward,
    range_from_args,
    random_token_positions,
)
from keys_values.kvcache.test_utils_advanced import cache_kwargs_for_smart_lastrec
from keys_values.utils import randint_torch, random_choices, index_to_3d


def _sample_overwrite_pos(
    batch_size: int,
    n_query_groups: int,
    q_len: int,
    cache_length: int,
    num_devices: int,
    active_dimensions: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    shape = (batch_size, n_query_groups, q_len)
    virtual_length = cache_length * num_devices
    inner_shape = tuple(shape[i] for i in active_dimensions + (2,))
    _inner = random_choices(inner_shape, size_range=virtual_length, device=device)
    view_dims = list(shape)
    exp_dims = list(shape[:-1]) + [-1]
    for i in range(2):
        if i in active_dimensions:
            exp_dims[i] = -1
        else:
            view_dims[i] = 1
    return _inner.view(*view_dims).expand(*exp_dims)


@pytest.mark.parametrize(
    "batch_size, n_query_groups, q_len, cache_length, input_pos, num_devices, essentially_1d",
    [
        (2, 4, 8, 256, 256 * 2 + 11, 2, False),
        (4, 2, 64, 128, 128 * 4 + 5, 4, False),
        (3, 4, 16, 512, 512 * 8 + 127, 8, False),
        (5, 8, 13, 256, 256 * 3 + 15, 3, False),
        (1, 4, 21, 256, 256 * 5 + 15, 5, False),
        (4, 2, 15, 256, 256 * 7 + 15, 7, False),
        (5, 8, 13, 256, 256 * 3 + 15, 3, True),
        (1, 4, 21, 256, 256 * 5 + 15, 5, True),
        (4, 2, 15, 256, 256 * 7 + 15, 7, True),
    ],
)
def test_communication_plan(batch_size, n_query_groups, q_len, cache_length, input_pos, num_devices, essentially_1d):
    active_dimensions = () if essentially_1d else (0, 1)
    device = torch.device("cpu")

    overwrite_pos = _sample_overwrite_pos(
        batch_size,
        n_query_groups,
        q_len,
        cache_length,
        num_devices,
        active_dimensions,
        device,
    )
    comm_plan = _get_communication_plan(
        num_devices=num_devices,
        input_pos=input_pos,
        overwrite_pos=overwrite_pos,
        active_dimensions=active_dimensions,
    )

    # Test correctness of plan
    # (1) Does the plan really equalize?
    delta_per_rank = _get_delta_per_rank(
        num_devices=num_devices,
        q_len=q_len,
        input_pos=input_pos,
        overwrite_pos=overwrite_pos,
        essentially_1d=essentially_1d,
    )
    if essentially_1d:
        delta_per_rank = delta_per_rank.view(1, 1, -1)
    print("delta_per_rank:")
    if essentially_1d:
        bh_range = [(0, 0)]
    else:
        bh_range = [(b, h) for b in range(batch_size) for h in range(n_query_groups)]
    print(
        "\n".join(
            f"{(b, h)}: {delta_per_rank[b, h, :].tolist()}"
            for b, h in bh_range
        )
    )
    print("\ncommunication_plan:")
    print("\n".join(f"{k}: {v.numpy()}" for k, v in comm_plan.items()))
    for (src_rank, trg_rank), plan in comm_plan.items():
        for row in plan:
            b, h, num = tuple(int(x) for x in row)
            assert num > 0, (src_rank, trg_rank, b, h, num)
            delta_per_rank[b, h, src_rank] += num
            delta_per_rank[b, h, trg_rank] -= num
    assert torch.all(delta_per_rank == 0).item()
    # (2) Communication must not go in both directions for any (b, h)
    check: Dict[Tuple[int, int], Set[Tuple[int, int]]] = dict()
    for (src_rank, trg_rank), plan in comm_plan.items():
        ranks = (src_rank, trg_rank) if src_rank < trg_rank else (trg_rank, src_rank)
        for row in plan:
            b_h = (int(row[0]), int(row[1]))
            vals = check.get(b_h, set())
            assert ranks not in vals, (b_h, ranks, vals)
            vals.add(ranks)
            check[b_h] = vals


@pytest.mark.parametrize(
    "batch_size, n_query_groups, q_len, cache_length, input_pos, num_devices, essentially_1d",
    [
        (2, 4, 8, 256, 256 * 2 + 11, 2, False),
        (4, 2, 64, 128, 128 * 4 + 5, 4, False),
        (3, 4, 16, 512, 512 * 8 + 127, 8, False),
        (5, 8, 13, 256, 256 * 3 + 15, 3, False),
        (1, 4, 21, 256, 256 * 5 + 15, 5, False),
        (4, 2, 15, 256, 256 * 7 + 15, 7, False),
        (5, 8, 13, 256, 256 * 3 + 15, 3, True),
        (1, 4, 21, 256, 256 * 5 + 15, 5, True),
        (4, 2, 15, 256, 256 * 7 + 15, 7, True),
    ],
)
def test_allocations_from_plan(batch_size, n_query_groups, q_len, cache_length, input_pos, num_devices, essentially_1d):
    active_dimensions = () if essentially_1d else (0, 1)
    device = torch.device("cpu")

    overwrite_pos = _sample_overwrite_pos(
        batch_size,
        n_query_groups,
        q_len,
        cache_length,
        num_devices,
        active_dimensions,
        device,
    )
    kwargs = dict(dtype=overwrite_pos.dtype, device=device)
    comm_plan = _get_communication_plan(
        num_devices=num_devices,
        input_pos=input_pos,
        overwrite_pos=overwrite_pos,
        active_dimensions=active_dimensions,
    )
    source_allocations, target_allocations = zip(
        *[
            _get_allocations(
                rank=rank,
                num_devices=num_devices,
                input_pos=input_pos,
                cache_length=cache_length,
                overwrite_pos=overwrite_pos,
                comm_plan=comm_plan,
                active_dimensions=active_dimensions,
            )
            for rank in range(num_devices)
        ]
    )

    # Test correctness of allocations.
    # (1) Match sources with targets
    for rank1 in range(num_devices):
        for rank2 in range(num_devices):
            if rank1 != rank2:
                if rank2 in source_allocations[rank1]:
                    assert rank1 in target_allocations[rank2]
                    src = source_allocations[rank1][rank2]
                    trg = target_allocations[rank2][rank1]
                    assert src.shape == trg.shape
                    if essentially_1d:
                        assert src.ndim == trg.ndim == 1
                    else:
                        assert src.ndim == trg.ndim == 2
                        assert src.shape[0] == 3
                        cnt_src = Counter(
                            (int(row[0]), int(row[1])) for row in src.T
                        )
                        cnt_trg = Counter(
                            (int(row[0]), int(row[1])) for row in trg.T
                        )
                        assert cnt_src == cnt_trg, (cnt_src, cnt_trg)
                else:
                    assert rank1 not in target_allocations[rank2]

    # (2) We run over all communications, modifying a copy of local overwrite
    # positions. We must end up with a balanced situations, where for all
    # `(b, h)`, there are as many local overwrite positions as stated by
    # `_get_q_len_for_rank`.
    bs = 1 if essentially_1d else batch_size
    nh = 1 if essentially_1d else n_query_groups
    local_overwrite_pos = []
    for rank in range(num_devices):
        is_for_me = (overwrite_pos % num_devices) == rank
        local_overwrite_pos.append(
            {
                (b, h): overwrite_pos[b, h, is_for_me[b, h, :]] // num_devices
                for b in range(bs)
                for h in range(nh)
            }
        )
    for src_rank, src_allocs in enumerate(source_allocations):
        for trg_rank, src_alloc in src_allocs.items():
            trg_alloc = target_allocations[trg_rank][src_rank]
            if essentially_1d:
                src_alloc = torch.cat(
                    (
                        torch.zeros((1, 1), **kwargs).expand(2, src_alloc.shape[-1]),
                        src_alloc.unsqueeze(0),
                    ),
                    dim=0,
                )
                trg_alloc = torch.cat(
                    (
                        torch.zeros((1, 1), **kwargs).expand(2, trg_alloc.shape[-1]),
                        trg_alloc.unsqueeze(0),
                    ),
                    dim=0,
                )
            # src_rank -> trg_rank
            _append_local_overwrite_pos(
                local_overwrite_pos=local_overwrite_pos[src_rank],
                index=src_alloc,
                essentially_1d=essentially_1d,
                **kwargs,
            )
            _remove_local_overwrite_pos(
                local_overwrite_pos=local_overwrite_pos[trg_rank],
                index=trg_alloc,
                essentially_1d=essentially_1d,
                src_rank=src_rank,
                trg_rank=trg_rank,
            )
    q_len_for_rank = _get_q_len_for_rank(
        num_devices, q_len, input_pos, **kwargs,
    )
    for rank, local_opos in enumerate(local_overwrite_pos):
        should_be = q_len_for_rank[rank].item()
        for b_h, vals in local_opos.items():
            assert vals.numel() == should_be, (rank, should_be, b_h, vals)


def _mock_send_phase1(
    src_rank: int,
    trg_rank: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    token_pos: torch.Tensor,
    communications: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
) -> Callable[[], bool]:
    communications[(src_rank, trg_rank)] = {
        "keys": keys,
        "values": values,
        "token_pos": token_pos,
    }
    return lambda: True


def _mock_receive_phase1(
    src_rank: int,
    trg_rank: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    token_pos: torch.Tensor,
) -> Callable[[], bool]:
    return lambda: False


def _mock_send_phase2(
    src_rank: int,
    trg_rank: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    token_pos: torch.Tensor,
) -> Callable[[], bool]:
    return lambda: True

def _mock_receive_phase2(
    src_rank: int,
    trg_rank: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    token_pos: torch.Tensor,
    communications: Dict[Tuple[int, int], Dict[str, torch.Tensor]],
) -> Callable[[], bool]:
    entry = communications[(src_rank, trg_rank)]
    keys.copy_(entry["keys"])
    values.copy_(entry["values"])
    token_pos.copy_(entry["token_pos"])
    return lambda: True


def _run_equalization(
    kv_caches: List[KVCacheWithBuffers],
    overwrite_pos: torch.Tensor,
    input_pos: int,
    verbose: bool = True,
):
    num_devices = len(kv_caches)
    # Mock proceeds in two phases. First phase 1 (send)
    source_stats = []
    target_stats = []
    communications = dict()

    with mock.patch(
        "keys_values.kvcache.parallel.equalize._send",
        partial(_mock_send_phase1, communications=communications),
    ):
        with mock.patch(
            "keys_values.kvcache.parallel.equalize._receive", _mock_receive_phase1
        ):
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
    with mock.patch("keys_values.kvcache.parallel.equalize._send", _mock_send_phase2):
        with mock.patch(
            "keys_values.kvcache.parallel.equalize._receive",
            partial(_mock_receive_phase2, communications=communications),
        ):
            for rank, kv_cache in enumerate(kv_caches):
                _, _target_stats = equalize_cache_content(
                    rank=rank,
                    num_devices=num_devices,
                    input_pos=input_pos,
                    kv_cache=kv_cache,
                    overwrite_pos=overwrite_pos,
                )
                target_stats.append(_target_stats)

    if verbose:
        for rank, (src, trg) in enumerate(zip(source_stats, target_stats)):
            num_sent = sum(src.values())
            num_recv = sum(trg.values())
            print(f"{rank}: Sent {num_sent:3d}, received {num_recv:3d}")


@pytest.mark.parametrize(
    "batch_size, n_query_groups, q_len, cache_length, input_pos, num_devices, essentially_1d",
    [
        (2, 4, 8, 256, 256 * 2 + 11, 2, False),
        (4, 2, 64, 128, 128 * 4 + 5, 4, False),
        (3, 4, 16, 512, 512 * 8 + 127, 8, False),
        (5, 8, 13, 256, 256 * 3 + 15, 3, False),
        (1, 4, 21, 256, 256 * 5 + 15, 5, False),
        (4, 2, 15, 256, 256 * 7 + 15, 7, False),
        (2, 4, 8, 32, 32 * 2 + 11, 2, True),
        (5, 8, 13, 256, 256 * 3 + 15, 3, True),
        (1, 4, 21, 256, 256 * 5 + 15, 5, True),
        (4, 2, 15, 256, 256 * 7 + 15, 7, True),
    ][6:7],
)
def test_retain_content(batch_size, n_query_groups, q_len, cache_length, input_pos, num_devices, essentially_1d):
    # Idea:
    # - Cache dependent on essentially_1d, no quantization
    # - Use regular token_pos, keys, values, random overwrite_pos
    # - Mock communication
    # - Check that all content is retained afterwards in union of caches,
    #   just different ordering
    seed = 31415927
    torch.random.manual_seed(seed)
    vocab_size = cache_length * num_devices * 2
    dtype = torch.bfloat16
    device = torch.device("cpu")
    if essentially_1d:
        name = "lastrec-default"
    else:
        name = "h2o-default"

    head_size = 4
    n_head = n_query_groups
    params = KVCacheParams(
        max_batch_size=batch_size,
        n_query_groups=n_query_groups,
        cache_length=cache_length,
        head_size=head_size,
        n_head=n_head,
        dtype=dtype,
    )
    virtual_length = cache_length * num_devices
    kv_caches = [create_kv_cache(name, params) for _ in range(num_devices)]
    index_kwargs = dict(dtype=torch.int, device=device)
    active_dimensions = kv_caches[0].active_dimensions()
    assert essentially_1d == (active_dimensions == ())
    # Assign content to buffers
    for rank, kv_cache in enumerate(kv_caches):
        token_pos = torch.arange(
            rank * cache_length,
            (rank + 1) * cache_length,
            **index_kwargs,
        )
        key = token_pos.to(dtype=dtype).view(1, 1, -1, 1).expand(
            batch_size, n_query_groups, -1, head_size)
        value = (
            token_pos.unsqueeze(-1) * torch.arange(head_size, **index_kwargs)
        ).to(dtype=dtype).view(1, 1, -1, head_size).expand(batch_size, n_query_groups, -1, -1)
        query = torch.randn(
            (batch_size, n_query_groups, cache_length, head_size),
            dtype=dtype,
            device=device,
        )
        kv_cache(
            query=query,
            key=key,
            value=value,
            token_idx=token_pos.unsqueeze(0).expand(batch_size, -1),
        )
        if essentially_1d:
            kv_cache.token_pos.copy_(token_pos)
        else:
            kv_cache.token_pos.copy_(index_to_3d(token_pos, batch_size, n_query_groups))
    # Sample overwrite positions at random
    q_len = randint_torch(virtual_length // 8, (virtual_length * 3) // 4)
    input_pos = randint_torch(virtual_length, virtual_length * 2)
    overwrite_pos = _sample_overwrite_pos(
        batch_size,
        n_query_groups,
        q_len,
        cache_length,
        num_devices,
        active_dimensions,
        device,
    )
    # Equalization (parallel computation is mocked)
    _run_equalization(kv_caches, overwrite_pos, input_pos)

    # Check consistency:
    # - token_pos must cover all of range(virtual_length)
    # - keys, values must be consistent with token_pos
    if essentially_1d:
        all_token_pos = torch.cat(
            [kv_cache.token_positions()[0, 0, :] for kv_cache in kv_caches]
        ).cpu().view(1, 1, -1)
        for rank, kv_cache in enumerate(kv_caches):
            print(f"Rank {rank}:")
            print(kv_cache.token_positions()[0, 0, :].tolist())
    else:
        all_token_pos = torch.cat(
            [kv_cache.token_positions().cpu() for kv_cache in kv_caches],
            dim=-1,
        )
    assert all_token_pos.shape[-1] == virtual_length
    torch.testing.assert_close(
        torch.sort(all_token_pos, dim=-1).values,
        torch.arange(
            virtual_length, device=torch.device("cpu"), dtype=all_token_pos.dtype,
        ).view(1, 1, -1).expand(*all_token_pos.shape),
    )
    all_keys = []
    all_values = []
    for kv_cache in kv_caches:
        k_and_v = kv_cache.get_keys_values()
        assert k_and_v is not None
        all_keys.append(k_and_v.keys())
        all_values.append(k_and_v.values())
    all_keys = torch.cat(all_keys, dim=2)
    all_values = torch.cat(all_values, dim=2)
    cmp_keys = all_token_pos.to(dtype=dtype).unsqueeze(-1).expand(*all_keys.shape)
    torch.testing.assert_close(all_keys, cmp_keys)
    cmp_values = (
        all_token_pos.unsqueeze(-1) * torch.arange(head_size, **index_kwargs).view(1, 1, 1, -1)
    ).expand(*all_values.shape)
    torch.testing.assert_close(all_values, cmp_values)


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
        parts["token_pos"].append(tp[0, 0, :] if essentially_1d else tp)
    return {k: torch.cat(v, dim=0 if k == "token_pos" else 2) for k, v in parts.items()}


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
                buffer=token_pos,
                **extra_info,
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


def args_equalize_before_after() -> Tuple[str, List[tuple]]:
    names = [
        (name, dict())
        for name in KVCacheFactory.supported_names()
        if name.endswith("-default") and not name.startswith("dense")
    ]
    return product_with_devices(names[1:2], "name, kwargs")


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
    index_kwargs = dict(dtype=torch.int64, device=device)

    num_devices = 8
    batch_size = 3
    n_query_groups = 4
    cache_length = 32
    head_size = 8
    n_head = 4
    params = KVCacheParams(
        max_batch_size=batch_size,
        n_query_groups=n_query_groups,
        cache_length=cache_length,
        head_size=head_size,
        n_head=n_head,
        dtype=dtype,
    )
    virtual_length = cache_length * num_devices
    if name.startswith("smart-lastrec"):
        kwargs = {**kwargs, **cache_kwargs_for_smart_lastrec()}
    kv_caches = [create_kv_cache(name, params, **kwargs) for _ in range(num_devices)]
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
        batch_size=batch_size,
        n_query_groups=n_query_groups,
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
            raise NotImplementedError(
                f"type(kv_cache) = {type(kv_cache)} not supported"
            )
    # Content before equalization
    contents = [_extract_content(kv_caches)]

    # Equalization (parallel computation is mocked, see above)
    overwrite_pos = _sample_overwrite_pos(
        batch_size,
        n_query_groups,
        q_len,
        cache_length,
        num_devices,
        active_dimensions,
        device,
    )
    _run_equalization(kv_caches, overwrite_pos, input_pos)

    contents.append(_extract_content(kv_caches))
    _compare_contents(
        contents=contents,
        essentially_1d=essentially_1d,
    )


if __name__ == "__main__":
    test_equalize_before_after(torch.device("cpu"), "lastrec-default", dict())
