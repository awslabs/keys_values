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
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from keys_values.kvcache.basics import KVCacheWithBuffers


def _get_q_len_for_rank(
    num_devices: int,
    q_len: int,
    input_pos: int,
    int_dtype: torch.dtype,
) -> np.ndarray:
    min_ql = q_len // num_devices
    num_plus1 = q_len - min_ql * num_devices
    q_len_for_rank = np.full((num_devices,), min_ql, dtype=int_dtype)
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
    overwrite_pos: np.ndarray,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    The communication plan maps `(from_device, to_device)` to a matrix whose
    rows are `[b, h, num_vecs]`, meaning that for `(b, h)`, `num_vecs` KV
    vectors are to be transferred.

    Args:
        num_devices: Total number of ranks
        cache_length: Length of cache on each rank
        input_pos: Token position where new content starts
        overwrite_pos: See :class:`CacheContentEqualizer`

    Returns:
        Communication plan

    """
    assert overwrite_pos.ndim == 3
    q_len = overwrite_pos.shape[-1]
    int_dtype = overwrite_pos.dtype
    q_len_for_rank = _get_q_len_for_rank(num_devices, q_len, input_pos, int_dtype)
    # Determine the Delta numbers
    num_per_rank = ((overwrite_pos // cache_length).astype(int_dtype) == np.arange(num_devices, dtype=int_dtype).reshape(1, 1, 1, -1)).sum(axis=-2)
    delta_per_rank = num_per_rank - q_len_for_rank.reshape(1, 1, -1)
    if not np.all(delta_per_rank.sum(axis=-1) == 0):
        raise ValueError(f"delta_per_rank must sum to 0 for each (b,h), but got:\n{delta_per_rank}:\n{delta_per_rank}")
    # Shapes: (bs, n_kv, num_devices)
    ranks = np.argsort(delta_per_rank, axis=-1)
    sorted_delta = np.take_along_axis(delta_per_rank, ranks, axis=-1)

    # Determine the communications
    communications = None
    while not torch.all(sorted_delta == 0):
        smallest = sorted_delta[..., 0].copy()  # (bs, n_kv)
        largest = sorted_delta[..., -1].copy()  # (bs, n_kv)
        min_vals = np.minimum(largest, -smallest)
        args1 = np.concatenate(
            (ranks[..., (num_devices - 1):num_devices], ranks[..., 0:1]),
            axis=-1,
        )
        args2 = np.concatenate(
            (ranks[..., 0:1], ranks[..., (num_devices - 1):num_devices]),
            axis=-1,
        )
        extra = np.concatenate(
            (np.where(largest >= -smallest, args1, args2), min_vals[..., np.newaxis]),
            axis=-1,
        )[..., np.newaxis]
        if communications is None:
            communications = extra
        else:
            communications = np.concatenate((communications, extra), axis=-1)
        largest -= min_vals
        smallest += min_vals
        if num_devices > 2:
            sorted_delta = sorted_delta[...,1:(-1)]
            new_pos_largest = np.searchsorted(sorted_delta, largest)
            sorted_delta = np.insert(
                sorted_delta, new_pos_largest, largest,
            )
            new_pos_smallest = np.searchsorted(sorted_delta, smallest)
            sorted_delta = np.insert(
                sorted_delta, new_pos_smallest, smallest,
            )
        else:
            if not torch.all(largest == 0) or not torch.all(smallest == 0):
                raise IndexError("Internal error for num_devices == 2")
            sorted_delta = np.zeros_like(sorted_delta)

    # Transform into plan: Needs groupby and filter out zero mass items
    if communications is None:
        return dict()

    batch_size, n_query_groups, _, num_rounds = communications.shape
    b_idx, h_idx = np.meshgrid(
        np.arange(batch_size), np.arange(n_query_groups), indexing="ij"
    )
    df = pd.DataFrame(
        {
            "b": np.broadcast_to(b_idx[..., np.newaxis], (batch_size, n_query_groups, num_rounds)).ravel(),
            "h": np.broadcast_to(h_idx[..., np.newaxis], (batch_size, n_query_groups, num_rounds)).ravel(),
            "from": communications[:, :, 0, :].ravel(),
            "to": communications[:, :, 1, :].ravel(),
            "mass": communications[:, :, 2, :].ravel(),
        }
    )
    df = df[df["mass"] != 0]
    return {
        (from_, to_): group[["b", "h", "mass"]].to_numpy().T
        for (from_, to_), group in df.groupby(["from", "to"])
    }


def equalize_cache_content(
    rank: int,
    num_devices: int,
    input_pos: int,
    kv_cache: KVCacheWithBuffers,
    overwrite_pos: torch.Tensor,
):
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

    Args:
        rank: Rank of device
        num_devices: Total number of ranks
        input_pos: Token position where new content starts
        kv_cache: KV cache on rank `rank`
        overwrite_pos: See :class:`CacheContentEqualizer`

    """
    if num_devices <= 1:
        raise ValueError(f"num_devices must be > 1, but got {num_devices}")
    if overwrite_pos.ndim != 3:
        raise ValueError("overwrite_pos must have 3 dimensions")
    q_len = overwrite_pos.shape[-1]
    overwrite_pos = overwrite_pos.numpy()
    # Create communication plan
    comm_plan = _get_communication_plan(
        num_devices, kv_cache.cache_length, input_pos, overwrite_pos,
    )
    # Rank needs to determine:
    # - Which slots to read and free up as source
    # - Which slots to write to as target
    # Since these slots are mutually exclusive across point-to-point transfers,
    # they can run in parallel and in any order
