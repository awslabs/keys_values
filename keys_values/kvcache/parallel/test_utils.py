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
from typing import Dict, Tuple, List, Any, Optional

import torch

from keys_values.attention.sdpa_wrapper import reorder_key_value
from keys_values.utils import random_choices, index_to_3d


def distribute_and_reorder_data(
    data: Dict[str, torch.Tensor],
    num_devices: int,
    input_pos: int,
) -> Tuple[List[Dict[str, Any]], List[torch.Tensor]]:
    """
    Given `data` for a combined (virtual) setup, where `data["key"],
    data["value"]` have length `cache_length == local_cl * num_devices`, and
    `data["token_pos"]` is equalized w.r.t. `input_pos` (see also
    :func:`sample_equalized_token_pos`), split this into data to be kept on
    each rank. For each rank, `result[rank]` is like `data`, but with cache
    length `local_cl`. Moreover, the "key", "value" entries are reordered
    according to their slice of `token_pos`, by calling
    :func:`reorder_key_value` accordingly.

    Args:
        data: Data for combined (virtual) setup
        num_devices: Number of devices
        input_pos: Input position for next update

    Returns:
        `result, q_inds`, where `result` is a list of size `num_devices`, whose
        entries are dictionaries like `data`, but for cache length
        `local_cl`. `q_inds` is a list of indices of how the "query" entry
        from `data` is distributed between the `result` entries.

    """
    shape = tuple(data["key"].shape)
    cache_length = shape[2]
    assert cache_length % num_devices == 0
    local_cl = cache_length // num_devices
    # Use views to do the "round-robin" distribution between ranks
    new_shape = shape[:2] + (local_cl, num_devices, shape[-1])
    _data = {name: data[name].view(*new_shape) for name in ("key", "value")}
    if input_pos > 0:
        _data["token_pos"] = data["token_pos"].view(*new_shape[:-1])
    q_len = data["query"].shape[2]
    u_val = (num_devices - input_pos % num_devices) % num_devices
    result = []
    q_inds = []
    for rank in range(num_devices):
        entry = {
            name: _data[name][:, :, :, rank, :].contiguous()
            for name in ("key", "value")
        }
        start = (u_val + rank) % num_devices
        q_ind = torch.arange(start, q_len, num_devices)
        q_inds.append(q_ind)
        entry["query"] = data["query"][:, :, q_ind, :].contiguous()
        if input_pos > 0:
            entry["token_pos"] = _data["token_pos"][:, :, :, rank].contiguous()
            entry["key"], entry["value"], entry["extra_info"] = reorder_key_value(
                key=entry["key"],
                value=entry["value"],
                token_positions=entry["token_pos"],
                input_pos=input_pos,  # Not used
                q_len=0,  # Not used
                sort_if_3d=True,
            )
        result.append(entry)
    return result, q_inds


def sample_equalized_token_pos(
    batch_size: int,
    n_query_groups: int,
    kv_per_rank: int,
    num_devices: int,
    input_pos: int,
    q_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Randomly draw `token_pos` of shape `(batch_size, n_query_groups, kv_len)`,
    where `kv_len = kv_per_rank * num_devices`. This is done in a way so that
    `token_pos` is equalized w.r.t. `input_pos`, which means that for each
    b, h, values in `token_pos[b, h, :]` which are `>= input_pos`, are
    distributed correctly between the ranks (and entries `< input_pos` are a
    random subset of `range(input_pos)`). For a distributed KV cache in real use,
    `token_pos` fulfils this property after equalization.

    More specifically, denote by `num_to_evict[b, h, rank]` the number of
    values `x in token_pos[b, h, :], x >= input_pos` so that
    `x % num_devices == rank`, for `rank in range(num_devices)`. Also, denote
    by `num_expected[rank]` the number of `x in range(input_pos,
    input_pos + q_len)` so that `x % num_devices == rank`. Then, we must have
    `num_to_evict[b, h, rank] == num_expected[rank]` for all b, h, rank.

    Args:
        batch_size: Batch size
        n_query_groups: Number of query groups
        kv_per_rank: Cache length per rank
        num_devices: Number of devices
        input_pos: Input position for next update
        q_len: Length of next update
        device: Device for `token_pos`

    Returns:
        `token_pos` of shape `(batch_size, n_query_groups, kv_len)` with
        properties detailed above.

    """
    kv_len = kv_per_rank * num_devices
    assert input_pos >= kv_len
    token_pos = random_choices(
        (batch_size, n_query_groups, kv_len),
        size_range=input_pos,
        device=device,
    )
    tp_4d = token_pos.view(batch_size, n_query_groups, kv_per_rank, num_devices)
    kwargs = dict(dtype=token_pos.dtype, device=device)
    uval = (num_devices - input_pos % num_devices) % num_devices
    for rank in range(num_devices):
        start = input_pos + (uval + rank) % num_devices
        new_vals = torch.arange(start, input_pos + q_len, num_devices, **kwargs)
        sz = new_vals.numel()
        rand_pos = random_choices(
            (batch_size, n_query_groups, sz),
            size_range=kv_per_rank,
            device=token_pos.device,
        )
        tp_4d[:, :, :, rank].scatter_(
            2,
            rand_pos,
            index_to_3d(new_vals, batch_size, n_query_groups),
        )
    return token_pos
