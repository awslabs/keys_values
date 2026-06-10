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
from dataclasses import dataclass
from itertools import islice
from typing import Dict, Tuple, Optional, Callable

import pandas as pd
import torch
import torch.distributed as dist

from keys_values.kvcache.basics import KVCacheWithBuffers
from keys_values.utils import index_to_3d, is_index_compatible


def _get_q_len_for_rank(
    num_devices: int,
    q_len: int,
    input_pos: int,
    **kwargs,
) -> torch.Tensor:
    min_ql = q_len // num_devices
    num_plus1 = q_len - min_ql * num_devices
    q_len_for_rank = torch.full((num_devices,), min_ql, **kwargs)
    if num_plus1 > 0:
        start = input_pos % num_devices
        end = min(start + num_plus1, num_devices)
        q_len_for_rank[start:end] = min_ql + 1
        rem = num_plus1 + start - end
        q_len_for_rank[:rem] = min_ql + 1
    return q_len_for_rank


def _get_communication_plan(
    num_devices: int,
    cache_length: int,
    input_pos: int,
    overwrite_pos: torch.Tensor,
    active_dimensions: Tuple[int, ...],
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    The communication plan maps `(from_device, to_device)` to a matrix whose
    rows are `[b, h, num_vecs]`, meaning that for `(b, h)`, `num_vecs` KV

    If the KV cache is essentially 1D (`active_dimensions == ()`), matrices are
    single rows `[0, 0, num_vecs]`. But in other cases of
    `active_dimensions != (0, 1)`, we "flatten" content into the form above.

    Args:
        num_devices: Total number of ranks
        cache_length: Length of cache on each rank
        input_pos: Token position where new content starts
        overwrite_pos: See :func:`equalize_cache_content`
        active_dimensions: Property of KV cache logic

    Returns:
        Communication plan

    """
    assert overwrite_pos.ndim == 3
    essentially_1d = active_dimensions == ()
    q_len = overwrite_pos.shape[-1]
    kwargs = dict(dtype=overwrite_pos.dtype, device=overwrite_pos.device)
    q_len_for_rank = _get_q_len_for_rank(num_devices, q_len, input_pos, **kwargs)
    # Determine the Delta numbers
    if essentially_1d:
        shape = ()
        overwrite_pos = overwrite_pos[0, 0, :]
    else:
        shape = (1, 1)
    num_per_rank = (
        (overwrite_pos // cache_length).unsqueeze(-1)
        == torch.arange(num_devices, **kwargs).view(*shape, 1, -1)
    ).sum(dim=-2)
    delta_per_rank = num_per_rank - q_len_for_rank.view(*shape, -1)
    if not torch.all(delta_per_rank.sum(dim=-1) == 0).item():
        raise ValueError(
            f"delta_per_rank must sum to 0 for each (b, h), but got:\n{delta_per_rank}"
        )
    # Shapes: (bs, n_kv, num_devices) or (num_devices,)
    # Note: If `len(active_dimensions) == 1`, we do redundant work along the
    # inactive dimension, but this does not matter.
    # Determine the communications
    communications = None
    while not torch.all(delta_per_rank == 0).item():
        # (bs, n_kv, 1) or (1,):
        ranks_smallest = torch.argmin(delta_per_rank, dim=-1, keepdim=True)
        ranks_largest = torch.argmax(delta_per_rank, dim=-1, keepdim=True)
        smallest = delta_per_rank.gather(-1, ranks_smallest)
        largest = delta_per_rank.gather(-1, ranks_largest)
        min_vals = torch.minimum(largest, -smallest)
        args1 = torch.cat(
            (ranks_smallest, ranks_largest),
            dim=-1,
        )
        args2 = torch.cat(
            (ranks_largest, ranks_smallest),
            dim=-1,
        )
        # (bs, n_kv, 3, 1) or (3, 1):
        extra = torch.cat(
            (torch.where(largest >= -smallest, args1, args2), min_vals),
            dim=-1,
        ).unsqueeze(-1)
        if communications is None:
            communications = extra
        else:
            communications = torch.cat((communications, extra), dim=-1)
        largest -= min_vals
        smallest += min_vals
        delta_per_rank.scatter_(-1, ranks_smallest, smallest)
        delta_per_rank.scatter_(-1, ranks_largest, largest)

    # `communications` is `(bs, n_kv, 3, num_rounds)` or `(3, num_rounds)`
    # Transform into plan: Needs groupby and filter out zero mass items
    if communications is None:
        return dict()
    if not essentially_1d:
        batch_size, n_query_groups, _, num_rounds = communications.shape
        b_idx, h_idx = torch.meshgrid(
            torch.arange(batch_size, **kwargs),
            torch.arange(n_query_groups, **kwargs),
            indexing="ij",
        )
        shape = (batch_size, n_query_groups, num_rounds)
        df = pd.DataFrame(
            {
                "b": b_idx.unsqueeze(-1).expand(*shape).flatten().numpy(),
                "h": h_idx.unsqueeze(-1).expand(*shape).flatten().numpy(),
                "from": communications[:, :, 0, :].flatten().numpy(),
                "to": communications[:, :, 1, :].flatten().numpy(),
                "mass": communications[:, :, 2, :].flatten().numpy(),
            }
        )
        df = df[df["mass"] != 0]
        return {
            (from_, to_): torch.tensor(group[["b", "h", "mass"]].to_numpy(), **kwargs)
            for (from_, to_), group in df.groupby(["from", "to"])
        }
    else:
        df = pd.DataFrame(
            {
                "from": communications[0].flatten().numpy(),
                "to": communications[1].flatten().numpy(),
                "mass": communications[2].flatten().numpy(),
            }
        )
        df = df[df["mass"] != 0]
        return {
            (from_, to_): torch.tensor(
                [0, 0, group["mass"].item()], **kwargs
            ).unsqueeze(0)
            for (from_, to_), group in df.groupby(["from", "to"])
        }


def _get_allocations(
    rank: int,
    num_devices: int,
    input_pos: int,
    cache_length: int,
    overwrite_pos: torch.Tensor,
    comm_plan: Dict[Tuple[int, int], torch.Tensor],
    active_dimensions: Tuple[int, ...],
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Return source and target allocations for rank `rank`.

    Source allocations apply when `rank` is the source in a communication,
    listing cache positions which are not to be overwritten. They are indexed
    by `trg_rank`, the target rank in the communication.
    Target allocations apply when `rank` is the target in a communication,
    listing overwriting positions. They are indexed by `src_rank`, the source
    rank in the communication.

    Args:
        rank: Rank of device
        num_devices: Total number of ranks
        input_pos: Token position where new content starts
        cache_length: Length of cache on each rank
        overwrite_pos: See :func:`equalize_cache_content`
        comm_plan: Communication plan
        active_dimensions: Property of KV cache logic

    Returns:
        `source_allocations, target_allocations`

    """
    kwargs = dict(dtype=overwrite_pos.dtype, device=overwrite_pos.device)
    essentially_1d = active_dimensions == ()
    is_for_me = (overwrite_pos // cache_length) == rank
    if not essentially_1d:
        batch_size, n_query_groups, q_len = overwrite_pos.shape
    else:
        # In this case, the dictionaries have a single entry with key (0, 0)
        batch_size, n_query_groups, q_len = 1, 1, overwrite_pos.shape[-1]
    # Lists of size two: First target, then source
    positions = [
        {
            (b, h): overwrite_pos[b, h, is_for_me[b, h, :]] % cache_length
            for b in range(batch_size)
            for h in range(n_query_groups)
        }
    ]
    q_len_for_me = _get_q_len_for_rank(num_devices, q_len, input_pos, **kwargs)[rank]
    _other_pos = dict()
    for b_h, ow_pos in positions[0].items():
        num = q_len_for_me - ow_pos.shape[0]
        if num > 0:
            ow_set = set(ow_pos.tolist())
            _other_pos[b_h] = torch.tensor(
                list(islice((x for x in range(cache_length) if x not in ow_set), num)),
                **kwargs,
            )
    positions.append(_other_pos)
    rel_pos = [
        {(b, h): 0 for b in range(batch_size) for h in range(n_query_groups)}
        for _ in range(2)
    ]
    allocations = [dict(), dict()]
    for (src_rank, trg_rank), plan in comm_plan.items():
        if trg_rank == rank:
            ind = 0
            other_rank = src_rank
        elif src_rank == rank:
            ind = 1
            other_rank = trg_rank
        else:
            continue
        for row in plan:
            b_h = (int(row[0]), int(row[1]))
            mass = row[-1]
            print(f"({src_rank}, {trg_rank}): ind={ind}, {b_h} -- {mass}")  # DEBUG
            pos = rel_pos[ind][b_h]
            new_cols = torch.tensor(
                positions[ind][b_h][pos : (pos + mass)],
                **kwargs,
            )
            if not essentially_1d:
                new_cols = torch.cat(
                    (
                        torch.tensor(b_h, **kwargs).unsqueeze(-1).expand(-1, mass),
                        new_cols.unsqueeze(0),
                    ),
                    dim=0,
                )
            if other_rank in allocations[ind]:
                allocations[ind][other_rank] = torch.cat(
                    (allocations[ind][other_rank], new_cols),
                    dim=-1,
                )
            else:
                allocations[ind][other_rank] = new_cols
            rel_pos[ind][b_h] += mass
    return allocations[1], allocations[0]


@dataclass(frozen=True)
class WriteBackToBuffer:
    src_rank: int
    keys: torch.Tensor
    values: torch.Tensor
    token_pos: torch.Tensor


# We use `_send` and `_receive` in order to be able to mock things during
# testing, see `test.kvcache.test_equalize.test_equalize_before_after`.
# The callback returned is called at the end of the function. It returns
# a flag. If this is `False`, the results are not written back.


def _send(
    src_rank: int,
    trg_rank: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    token_pos: torch.Tensor,
) -> Callable[[], bool]:
    req_keys = dist.isend(keys, dst=trg_rank)
    req_values = dist.isend(values, dst=trg_rank)
    req_tps = dist.isend(token_pos, dst=trg_rank)

    def wait() -> bool:
        req_keys.wait()
        req_values.wait()
        req_tps.wait()
        return True

    return wait


def _receive(
    src_rank: int,
    trg_rank: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    token_pos: torch.Tensor,
) -> Callable[[], bool]:
    req_keys = dist.irecv(keys, src=src_rank)
    req_values = dist.irecv(values, src=src_rank)
    req_tps = dist.irecv(token_pos, src=src_rank)

    def wait() -> bool:
        req_keys.wait()
        req_values.wait()
        req_tps.wait()
        return True

    return wait


def _execute_communication(
    rank: int,
    src_rank: int,
    trg_rank: int,
    source_allocations: Dict[int, torch.Tensor],
    num_vecs: int,
    kv_cache: KVCacheWithBuffers,
    source_stats: Dict[int, int],
) -> Optional[WriteBackToBuffer]:
    """
    Note: If `len(active_dimensions) == 1`, one dimension in `token_pos` has
    stride 0, we could exploit this to transmit less data. But this is dwarfed
    by `keys, values`, so we don't bother here.

    """
    assert rank in (src_rank, trg_rank)
    active_dimensions = kv_cache.active_dimensions()
    essentially_1d = active_dimensions == ()
    result = None
    dtype = kv_cache.dtype
    device = kv_cache.device
    head_size = kv_cache.head_size
    int_dtype = kv_cache.token_positions().dtype
    kwargs = dict(dtype=dtype, device=device)
    shape = (kv_cache.batch_size, kv_cache.n_query_groups)
    sz = num_vecs * shape[0] * shape[1] if essentially_1d else num_vecs
    keys = torch.zeros((sz, head_size), **kwargs)
    values = torch.zeros((sz, head_size), **kwargs)
    token_pos = torch.zeros((num_vecs,), dtype=int_dtype, device=device)

    if rank == src_rank:
        # Collect and send to `trg_rank`
        index = source_allocations[trg_rank]
        if not essentially_1d:
            kv_cache.kv_buffers.get_vectors(
                index=index,
                out_key=keys,
                out_value=values,
            )
            token_pos.copy_(kv_cache.token_positions()[index[0], index[1], index[2]])
        else:
            kv_cache.kv_buffers.get_slots(
                positions=index_to_3d(index, *shape),
                out_key=keys.view(*shape, num_vecs, head_size),
                out_value=values.view(*shape, num_vecs, head_size),
            )
            token_pos.copy_(kv_cache.token_positions()[0, 0, index])
        source_stats[trg_rank] = source_stats.get(trg_rank, 0) + sz
        callback = _send(src_rank, trg_rank, keys, values, token_pos)
    else:
        # Receive from `src_rank`
        callback = _receive(src_rank, trg_rank, keys, values, token_pos)
    do_write_back = callback()

    # Write-back to`trg_rank` cache is delayed
    if do_write_back and rank == trg_rank:
        result = WriteBackToBuffer(
            src_rank=src_rank,
            keys=keys,
            values=values,
            token_pos=token_pos,
        )
    return result


def equalize_cache_content(
    rank: int,
    num_devices: int,
    input_pos: int,
    kv_cache: KVCacheWithBuffers,
    overwrite_pos: torch.Tensor,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    In context parallelism, a "virtual" KV cache of length
    `num_devices * cache_length` is realized on `num_devices` devices, each
    maintaining a "physical" cache of length `cache_length`.

    At the start of an update with new KV information for `q_len` tokens,
    `batch_size * n_query_groups * q_len` KV vectors are evicted (i.e.,
    overwritten by new information). The new information is distributed
    across devices, so that rank `r` receives `q_len(r)` of it, where
    `sum(q_len(r)) == q_len`. Here, `q_len(r)` is either `M` or `M - 1`, where
    `M = ceil(q_len / num_devices)`.

    Equalization is needed because each physical cache `r` needs to overwrite
    `batch_size * n_query_groups * q_len(r)` KV vectors, but the KV cache
    policy may evict slots in different proportions. In a nutshell, equalization
    moves cache content from source ranks (less overwrite positions than new
    information) to target ranks (more overwrite positions than new information),
    so that afterwards each rank has exactly the right number of slots to
    overwrite.

    `overwrite_pos` is an index tensor of shape
    `(batch_size, n_query_groups, q_len)`, containing the overwrite positions
    for the virtual cache. Entries are in `range(num_devices * cache_length)`.
    A position `p` is assigned to rank `p // cache_length`, referring to local
    position `p % cache_length` there.

    `overwrite_pos` must be compatible with `kv_cache.active_dimensions()`. With
    less active dimensions, transfers are more constrained.

    Args:
        rank: Rank of device
        num_devices: Total number of ranks
        input_pos: Token position where new content starts
        kv_cache: KV cache on rank `rank`
        overwrite_pos: See above

    Returns:
        `source_stats`, `target_stats`, which are dictionaries mapping
        rank of communication partner to number of KV vectors transferred.

    """
    if num_devices <= 1:
        raise ValueError(f"num_devices must be > 1, but got {num_devices}")
    if overwrite_pos.ndim != 3:
        raise ValueError("overwrite_pos must have 3 dimensions")
    cache_length = kv_cache.cache_length
    head_size = kv_cache.head_size
    if kv_cache.current_length < cache_length:
        raise ValueError(
            f"kv_cache must be full (but current_length = {kv_cache.current_length} < {cache_length} = cache_length)"
        )
    active_dimensions = kv_cache.active_dimensions()
    essentially_1d = active_dimensions == ()
    if not is_index_compatible(overwrite_pos, active_dimensions):
        raise ValueError(
            "overwrite_pos not compatible with kv_cache.active_dimensions:\n"
            f"stride = {overwrite_pos.stride()}\n"
            f"shape = {overwrite_pos.shape}\n"
            f"active_dimensions = {active_dimensions}"
        )
    # Create communication plan
    comm_plan = _get_communication_plan(
        num_devices,
        cache_length,
        input_pos,
        overwrite_pos,
        active_dimensions,
    )
    # Rank needs to determine:
    # - Which slots to read and free up as source
    # - Which slots to write to as target
    # Note: Given the allocations, we do not need `comm_plan.values()` anymore.
    source_allocations, target_allocations = _get_allocations(
        rank,
        num_devices,
        input_pos,
        cache_length,
        overwrite_pos,
        comm_plan,
        active_dimensions,
    )

    # Communication (each transfer can run in parallel)
    # Note: Results are first collected, then written back en bulk below.
    # While `KVCacheBuffers.set_vectors` allows to write specific KV vectors
    # into buffers, and these are non-overlapping for different transfers, we
    # cannot be sure whether PyTorch supports concurrent writing into the
    # same buffers.
    results = []
    source_stats = dict()
    for src_rank, trg_rank in comm_plan.keys():
        if rank in (src_rank, trg_rank):
            if rank == src_rank:
                num_vecs = source_allocations[trg_rank].shape[-1]
            else:
                num_vecs = target_allocations[src_rank].shape[-1]
            result = _execute_communication(
                rank,
                src_rank,
                trg_rank,
                source_allocations,
                num_vecs,
                kv_cache,
                source_stats,
            )
            if result is not None:
                results.append(result)

    # Write back to cache buffer
    target_stats = dict()
    shape = (kv_cache.batch_size, kv_cache.n_query_groups)
    for result in results:
        src_rank = result.src_rank
        index = target_allocations[src_rank]
        if not essentially_1d:
            kv_cache.kv_buffers.set_vectors(
                index=index,
                key=result.keys,
                value=result.values,
            )
        else:
            kv_cache.kv_buffers.set_slots(
                positions=index_to_3d(index, *shape),
                key=result.keys.view(*shape, -1, head_size),
                value=result.values.view(*shape, -1, head_size),
            )
        if len(active_dimensions) == 1:
            # We transmitted `token_pos` as if active dimensions were (0, 1).
            adim = 1 - active_dimensions[0]
            assert adim in (0, 1)  # Sanity check
            rel_ind = index[1 - adim] == 0
            token_pos_2d = result.token_pos[rel_ind]
            if token_pos_2d.numel() * shape[adim] != result.token_pos.numel():
                raise ValueError(
                    f"token_pos received not compatible with active_dimensions = {active_dimensions}:\n"
                    + str(result.token_pos.view(*shape, -1))
                )
            index_2d = index[rel_ind, [adim, 2]]
            kv_cache.set_token_positions(
                index=index_2d,
                tp_values=token_pos_2d,
            )
        else:
            kv_cache.set_token_positions(
                index=index,
                tp_values=result.token_pos,
            )
        target_stats[src_rank] = target_stats.get(src_rank, 0) + result.keys.shape[0]

    return source_stats, target_stats
