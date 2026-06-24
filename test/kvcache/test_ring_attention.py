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
from typing import Dict, List, Any, Tuple

import pytest
import torch
from litgpt.utils import _RunIf

from keys_values.attention.attention_utils import sample_token_positions
from keys_values.attention.flex_attention import (
    scaled_dot_product_attention_flexatt,
    FlexAttentionArgs,
)
from keys_values.attention.sdpa_wrapper import reorder_key_value
from keys_values.config import Config
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.parallel.flex_for_ring import (
    RingOffdiagFlexAttentionArgs,
    RingDiagFlexAttentionArgs,
)
from keys_values.kvcache.parallel.ring_attention import RingAttentionDriver
from keys_values.kvcache.test_utils import random_args_cache_forward


def _distribute_and_reorder_data(
    data: Dict[str, torch.Tensor],
    num_devices: int,
    input_pos: int,
) -> Tuple[List[Dict[str, Any]], List[torch.Tensor]]:
    shape = tuple(data["key"].shape)
    cache_length = shape[2]
    assert cache_length % num_devices == 0
    local_cl = cache_length // num_devices
    new_shape = shape[:2] + (local_cl, num_devices, shape[-1])
    _data = {
        name: data[name].view(*new_shape)
        for name in ("key", "value")
    }
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


def _equalize_token_pos(
    token_pos: torch.Tensor,
    input_pos: int,
    q_len: int,
    num_devices: int,
):
    assert token_pos.ndim == 1
    kv_len = token_pos.numel()
    assert kv_len % num_devices == 0
    kv_per_rank = kv_len // num_devices
    tp_2d = token_pos.view(kv_per_rank, num_devices)
    kwargs = dict(dtype=token_pos.dtype, device=token_pos.device)
    uval = (num_devices - input_pos % num_devices) % num_devices
    for rank in range(num_devices):
        start = input_pos + (uval + rank) % num_devices
        new_vals = torch.arange(start, input_pos + q_len, num_devices, **kwargs)
        sz = new_vals.numel()
        rand_pos = torch.randperm(kv_per_rank)[:sz]
        tp_2d[rand_pos, rank] = new_vals


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "n_head, n_query_groups, q_len, kv_len_per_rank, dtype, input_pos, num_devices, do_q_lens, is_1d",
    [
        (4, 2, 128, 512, torch.float16, 512 * 4, 4, False, False),
        (4, 4, 8, 256, torch.bfloat16, 256 * 2 + 11, 2, False, False),
        (8, 4, 64, 128, torch.float16, 128 * 8 + 5, 8, True, True),
        (12, 4, 16, 512, torch.bfloat16, 512 * 3 + 127, 3, False, True),
        (24, 8, 8, 256, torch.float16, 256 * 5 + 15, 5, True, False),
        (9, 3, 128, 256, torch.bfloat16, 256 * 4 + 27, 4, False, True),
        (12, 4, 16, 256, torch.float16, 256 * 8 + 513, 8, True, False),
    ],
)
def test_sdpa_distributed_vs_single_on_chunk(
    n_head,
    n_query_groups,
    q_len,
    kv_len_per_rank,
    dtype,
    input_pos,
    num_devices,
    do_q_lens,
    is_1d,
):
    seed = 31415927
    torch.manual_seed(seed)
    atol = 0.0005 if dtype == torch.float16 else 0.005
    rtol = 0.1

    batch_size = 2
    head_size = 32
    device = torch.device("cuda", 0)
    kv_len = kv_len_per_rank * num_devices

    config = Config.from_name(
        "gemma-2-27b",
        block_size=3 * kv_len,
        n_layer=1,
        n_query_groups=n_query_groups,
        n_head=n_head,
        n_embd=n_head * head_size,
        intermediate_size=n_head * head_size * 3,
        rotary_percentage=1.0,
    )
    params = KVCacheParams.from_config(
        config=config,
        max_batch_size=batch_size,
        cache_length=kv_len,
        dtype=dtype,
    )
    # Sample data for comparison
    data_all = random_args_cache_forward(
        params,
        num=kv_len,
        vocab_size=config.vocab_size,
        device=device,
    )
    data_all["query"] = data_all["query"][:, :, :q_len, :].contiguous()
    # For `token_pos`, we need to make sure that values `>= input_pos` are
    # distributed correctly between the ranks, so that the new KV information
    # is for the same tokens (per rank) than the query information. This is
    # guaranteed by equalization.
    index_kwargs = dict(dtype=torch.int64, device=device)
    bs = 1 if is_1d else batch_size
    nqg = 1 if is_1d else n_query_groups
    tp = torch.zeros((bs, nqg, kv_len), **index_kwargs)
    for b in range(bs):
        for h in range(nqg):
            tp[b, h, :] = torch.randperm(input_pos, **index_kwargs)[:kv_len]
            _equalize_token_pos(tp[b, h, :], input_pos, q_len, num_devices)
    if not is_1d:
        data_all["token_pos"] = tp
    else:
        data_all["token_pos"] = tp.expand(batch_size, n_query_groups, -1)

    # Distributed computation
    data, q_inds = _distribute_and_reorder_data(data_all, num_devices, input_pos)
    if do_q_lens:
        max_val = max(x["query"].shape[2] for x in data)
        q_lens = [max_val + 4]
    else:
        q_lens = None
    flexatt_args_diag = RingDiagFlexAttentionArgs(q_lens=q_lens)
    flexatt_args_offdiag = RingOffdiagFlexAttentionArgs(
        num_devices=num_devices, q_lens=q_lens,
    )
    drivers = [
        RingAttentionDriver(
            rank_r=r,
            flexatt_args_diag=flexatt_args_diag,
            flexatt_args_offdiag=flexatt_args_offdiag,
        )
        for r in range(num_devices)
    ]
    for entry, driver in zip(data, drivers):
        driver.reset(
            query=entry["query"],
            scale=None,
            input_pos=input_pos,
            num_new_tokens=q_len,
            config=config,
        )
    # Loop around the ring
    for iter in range(num_devices):
        for driver in drivers:
            rank_s = (driver.rank_r - iter) % num_devices
            entry_s = data[rank_s]
            driver(entry_s["key"], entry_s["value"])
    dist_outputs = [driver.results()[0] for driver in drivers]

    # Single computation
    flexatt_args = FlexAttentionArgs()
    output_all = scaled_dot_product_attention_flexatt(
        flexatt_args=flexatt_args,
        query=data_all["query"],
        key=data_all["key"],
        value=data_all["value"],
        scale_factor=None,
        sliding_window_size=None,
        attention_logit_softcapping=None,
        input_pos=input_pos,
        token_positions=data_all["token_pos"],
    )
    single_outputs = [output_all[:, :, q_ind, :] for q_ind in q_inds]

    for rank, (d_output, s_output) in enumerate(
        zip(dist_outputs, single_outputs)
    ):
        print(f"Outputs for rank {rank}")
        torch.testing.assert_close(d_output, s_output, atol=atol, rtol=rtol)


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "n_head, n_query_groups, kv_len_per_rank, dtype, num_devices",
    [
        (4, 2, 512, torch.float16, 4),
        (4, 4, 256, torch.bfloat16, 2),
        (8, 4, 128, torch.float16, 8),
        (12, 4, 512, torch.bfloat16, 3),
        (24, 8, 256, torch.float16, 5),
        (9, 3, 256, torch.bfloat16, 4),
        (12, 4, 256, torch.float16, 8),
    ],
)
def test_sdpa_distributed_vs_single_on_prefill(
    n_head,
    n_query_groups,
    kv_len_per_rank,
    dtype,
    num_devices,
):
    seed = 31415927
    torch.manual_seed(seed)
    atol = 0.0005 if dtype == torch.float16 else 0.005
    rtol = 0.1

    batch_size = 2
    head_size = 32
    device = torch.device("cuda", 0)
    kv_len = kv_len_per_rank * num_devices
    input_pos = 0

    config = Config.from_name(
        "gemma-2-27b",
        block_size=3 * kv_len,
        n_layer=1,
        n_query_groups=n_query_groups,
        n_head=n_head,
        n_embd=n_head * head_size,
        intermediate_size=n_head * head_size * 3,
        rotary_percentage=1.0,
    )
    params = KVCacheParams.from_config(
        config=config,
        max_batch_size=batch_size,
        cache_length=kv_len,
        dtype=dtype,
    )
    # Sample data for comparison
    data_all = random_args_cache_forward(
        params,
        num=kv_len,
        vocab_size=config.vocab_size,
        device=device,
    )

    # Distributed computation
    data, q_inds = _distribute_and_reorder_data(data_all, num_devices, input_pos)
    flexatt_args_diag = RingDiagFlexAttentionArgs()
    flexatt_args_offdiag = RingOffdiagFlexAttentionArgs(num_devices=num_devices)
    drivers = [
        RingAttentionDriver(
            rank_r=r,
            flexatt_args_diag=flexatt_args_diag,
            flexatt_args_offdiag=flexatt_args_offdiag,
        )
        for r in range(num_devices)
    ]
    for entry, driver in zip(data, drivers):
        # print(f"Rank {driver.rank_r}: reset")  # DEBUG
        driver.reset(
            query=entry["query"],
            scale=None,
            input_pos=input_pos,
            num_new_tokens=kv_len,
            config=config,
        )
    # Loop around the ring
    for it in range(num_devices):
        for driver in drivers:
            rank_s = (driver.rank_r - it) % num_devices
            # print(f"Iter {it}, rank {driver.rank_r}: rank_s = {rank_s}")  # DEBUG
            entry_s = data[rank_s]
            driver(entry_s["key"], entry_s["value"])
    dist_outputs = [driver.results()[0] for driver in drivers]

    # Single computation
    flexatt_args = FlexAttentionArgs()
    output_all = scaled_dot_product_attention_flexatt(
        flexatt_args=flexatt_args,
        query=data_all["query"],
        key=data_all["key"],
        value=data_all["value"],
        scale_factor=None,
        sliding_window_size=None,
        attention_logit_softcapping=None,
        input_pos=input_pos,
        token_positions=None,
    )
    single_outputs = [output_all[:, :, q_ind, :] for q_ind in q_inds]

    for rank, (d_output, s_output) in enumerate(
        zip(dist_outputs, single_outputs)
    ):
        print(f"Outputs for rank {rank}")
        torch.testing.assert_close(d_output, s_output, atol=atol, rtol=rtol)
