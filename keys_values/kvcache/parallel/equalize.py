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
    """
    Consider the values `j % num_devices`, where
    `j in range(input_pos, input_pos + q_len)`. Return tensor
    `q_len_for_rank`, so that `q_len_for_rank[r]` is the number of values
    equal to `r`, for `r in range(num_devices)`.

    """
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


def _get_delta_per_rank(
    num_devices: int,
    q_len: int,
    input_pos: int,
    overwrite_pos: torch.Tensor,
    essentially_1d: bool,
) -> torch.Tensor:
    kwargs = dict(dtype=overwrite_pos.dtype, device=overwrite_pos.device)
    q_len_for_rank = _get_q_len_for_rank(num_devices, q_len, input_pos, **kwargs)
    # Determine the Delta numbers
    if essentially_1d:
        shape = ()
        overwrite_pos = overwrite_pos[0, 0, :]
    else:
        shape = (1, 1)
    num_per_rank = (
        (overwrite_pos % num_devices).unsqueeze(-1)
        == torch.arange(num_devices, **kwargs).view(*shape, 1, -1)
    ).sum(dim=-2)
    delta_per_rank = num_per_rank - q_len_for_rank.view(*shape, -1)
    if not torch.all(delta_per_rank.sum(dim=-1) == 0).item():
        raise ValueError(
            f"delta_per_rank must sum to 0 for each (b, h), but got:\n{delta_per_rank}"
        )
    return delta_per_rank


def _get_communication_plan(
    num_devices: int,
    input_pos: int,
    overwrite_pos: torch.Tensor,
    active_dimensions: Tuple[int, ...],
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    The communication plan maps `(from_device, to_device)` to a matrix whose
    rows are `[b, h, num_vecs]`, meaning that for `(b, h)`, `num_vecs` KV
    vectors need to be transferred.

    If the KV cache is essentially 1D (`active_dimensions == ()`), matrices are
    single rows `[0, 0, num_vecs]`, which means that `num_vecs` KV vectors need
    to be transferred for every `(b, h)`. But in other cases of
    `active_dimensions != (0, 1)`, we "flatten" content into the form above.

    Args:
        num_devices: Total number of ranks
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
    delta_per_rank = _get_delta_per_rank(
        num_devices=num_devices,
        q_len=q_len,
        input_pos=input_pos,
        overwrite_pos=overwrite_pos,
        essentially_1d=essentially_1d,
    )
    # Shapes: (bs, n_kv, num_devices) or (num_devices,)
    # Note: If `len(active_dimensions) == 1`, we do redundant work along the
    # inactive dimension.
    # Determine the communications (using greedy algorithm)
    communications = None
    while not torch.all(delta_per_rank == 0).item():
        # (bs, n_kv, 1) or (1,):
        ranks_smallest = torch.argmin(delta_per_rank, dim=-1, keepdim=True)
        ranks_largest = torch.argmax(delta_per_rank, dim=-1, keepdim=True)
        smallest = delta_per_rank.gather(-1, ranks_smallest)
        largest = delta_per_rank.gather(-1, ranks_largest)
        min_vals = torch.minimum(largest, -smallest)
        extra = torch.cat(
            (ranks_smallest, ranks_largest, min_vals),
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

    Source allocations apply when `rank` is the source rank in a communication.
    `source_allocations` is a dictionary, mapping target rank to positions of
    shape `(3, num)`, or `(num,)` if `essentially_1d`. These are positions to
    read from cache buffers. The content is sent to the target rank in order
    not to be evicted. Their slots become overwrite positions here,

    Target allocations apply when `rank` is the target rank in a communication.
    `target_allocations` is a dictionary, mapping source rank to positions of
    shape `(3, num)`, or `(num,)` if `essentially_1d`. These are positions to
    write into cache buffers. The content is received from the source rank and
    overwrites slots to be evicted here.

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
    is_for_me = (overwrite_pos % num_devices) == rank
    if not essentially_1d:
        batch_size, n_query_groups, q_len = overwrite_pos.shape
    else:
        # In this case, the dictionaries have a single entry with key (0, 0)
        batch_size, n_query_groups, q_len = 1, 1, overwrite_pos.shape[-1]
    # Lists of size two: First for target, then for source
    # - `positions[0][(b, h)]`: Local overwrite positions for `rank`. These
    #   positions can become target for a communication
    # - `positions[1][(b, h)]`: Local positions for `rank` not to be overwritten.
    #   These positions can become source for a communication
    positions = [
        {
            (b, h): overwrite_pos[b, h, is_for_me[b, h, :]] // num_devices
            for b in range(batch_size)
            for h in range(n_query_groups)
        }
    ]
    q_len_for_me = _get_q_len_for_rank(num_devices, q_len, input_pos, **kwargs)[rank]
    _other_pos = dict()
    for b_h, ow_pos in positions[0].items():
        # Only if local overwrite positions less than `q_len_for_me`, so that
        # `rank` is a source for `b_h`
        num = q_len_for_me - ow_pos.shape[0]
        if num > 0:
            ow_set = set(ow_pos.tolist())
            # `num` positions which can be source
            _other_pos[b_h] = torch.tensor(
                list(islice((x for x in range(cache_length) if x not in ow_set), num)),
                **kwargs,
            )
    positions.append(_other_pos)  # source part
    # `rel_pos[k][b_h]` entries in `positions[k][b_h]` have been used already:
    rel_pos = [
        {(b, h): 0 for b in range(batch_size) for h in range(n_query_groups)}
        for _ in range(2)
    ]
    allocations = [dict(), dict()]
    for (src_rank, trg_rank), plan in comm_plan.items():
        # `positions[ind]`, `rel_pos[ind]` pertains to `rank`
        if trg_rank == rank:
            ind = 0
            other_rank = src_rank
        elif src_rank == rank:
            ind = 1
            other_rank = trg_rank
        else:
            continue
        for row in plan:
            # `b_h`: Transfer `mass` from `src_rank` -> `trg_rank`
            b_h = (int(row[0]), int(row[1]))
            mass = int(row[-1])
            pos = rel_pos[ind][b_h]
            if ind == 1 and b_h not in positions[ind]:
                raise IndexError(
                    f"Internal error: b_h={b_h}, positions[0]={positions[0][b_h]}, q_len_for_me={q_len_for_me}"
                )
            if positions[ind][b_h].numel() < pos + mass:
                raise IndexError(
                    f"Internal error: b_h={b_h}, positions[{ind}]={positions[ind][b_h]}, pos={pos}, mass={mass}"
                )
            # Local positions to read from (`ind == 1`) or write to (`ind == 0`)
            # Shape: `(mass,)`
            new_cols = positions[ind][b_h][pos : (pos + mass)].to(**kwargs)
            if not essentially_1d:
                # Must be `[[b, h, p0], [b, h, p1], ...]`
                # Shape: `(3, mass)`
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
            rel_pos[ind][b_h] = pos + mass
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
    reqs = [dist.isend(x, dst=trg_rank) for x in (keys, values, token_pos)]

    def wait() -> bool:
        for req in reqs:
            req.wait()
        return True

    return wait


def _receive(
    src_rank: int,
    trg_rank: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    token_pos: torch.Tensor,
) -> Callable[[], bool]:
    reqs = [dist.irecv(x, src=src_rank) for x in (keys, values, token_pos)]

    def wait() -> bool:
        for req in reqs:
            req.wait()
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
    else:
        result = None
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

    A virtual cache position `p` maps to physical position `p // num_devices`
    on device `p % num_devices`. Namely, the cache is filled in a round-robin
    fashion, which ensures that the most recent tokens are equally distributed
    among devices, which reduces the amount of equalization (see below).

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
    A position `p` is assigned to rank `p % cache_length`, referring to local
    position `p // cache_length` there. For each `(b, h)`, the overwrite positions
    `overwrite_pos[b, h, :]` are mapped to the different ranks by
    `overwrite_pos[b, h, :] % cache_length`. If the numbers of positions per rank
    are different from `q_len(r)`, we need equalization for `(b, h)`.
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
    # Create communication plan: How many slots need to be transferred?
    comm_plan = _get_communication_plan(
        num_devices,
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
        num_vecs = index.shape[-1]
        sz = num_vecs * shape[0] * shape[1] if essentially_1d else num_vecs
        if result.keys.shape != (sz, head_size) or result.token_pos.shape != (num_vecs,):
            raise IndexError(
                f"src_rank {src_rank}: keys {result.keys.shape} vs {(sz, head_size)}, token_pos {result.token_pos.shape} vs {(num_vecs,)}"
            )
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
