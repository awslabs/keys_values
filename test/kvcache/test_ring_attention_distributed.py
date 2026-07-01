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
import os
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from litgpt.utils import _RunIf
import lightning as L
from lightning.fabric.strategies import DDPStrategy

from keys_values.attention.flex_attention import (
    scaled_dot_product_attention_flexatt,
    FlexAttentionArgs,
)
from keys_values.config import Config
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.parallel.flex_for_ring import (
    RingOffdiagFlexAttentionArgs,
    RingDiagFlexAttentionArgs,
)
from keys_values.kvcache.parallel.ring_attention import RingAttentionDriver
from keys_values.kvcache.parallel.ring_attention_utils import RingAttentionComputation
from keys_values.kvcache.test_utils import (
    random_args_cache_forward,
    distribute_and_reorder_data,
    equalize_token_pos,
)
from keys_values.utils import random_choices


# TODO: Less cases!
@_RunIf(min_cuda_gpus=3)
@pytest.mark.parametrize(
    "n_head, n_query_groups, q_len, kv_len_per_rank, dtype, input_pos, num_devices, do_q_lens, is_1d",
    [
        ( 4, 2, 128, 512,  torch.float16,       512 * 3, 3, False, False),
        ( 4, 4,   8, 256, torch.bfloat16,  256 * 2 + 11, 2, False, False),
        ( 8, 4,  64, 128,  torch.float16,   128 * 3 + 5, 3,  True,  True),
        (12, 4,  16, 512, torch.bfloat16, 512 * 2 + 127, 2, False,  True),
        (24, 8,   8, 256,  torch.float16,  256 * 3 + 15, 3,  True, False),
        (24, 8,   8, 256,  torch.float16,  256 * 3 + 15, 3,  True, False),
        ( 9, 3, 128, 256, torch.bfloat16,  256 * 2 + 27, 2, False,  True),
        (12, 4,  16, 256,  torch.float16, 256 * 3 + 513, 3,  True, False),
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
    )
    data_all["query"] = data_all["query"][:, :, :q_len, :].contiguous()
    # For `token_pos`, we need to make sure that values `>= input_pos` are
    # distributed correctly between the ranks, so that the new KV information
    # is for the same tokens (per rank) than the query information. This is
    # guaranteed by equalization.
    bs = 1 if is_1d else batch_size
    nqg = 1 if is_1d else n_query_groups
    tp = random_choices(
        (bs, nqg, kv_len),
        size_range=input_pos,
    )
    equalize_token_pos(tp, input_pos, q_len, num_devices)
    if not is_1d:
        data_all["token_pos"] = tp
    else:
        data_all["token_pos"] = tp.expand(batch_size, n_query_groups, -1)

    # Distributed computation
    fabric = L.Fabric(
        devices=num_devices,
        num_nodes=1,
        strategy=DDPStrategy(static_graph=True, broadcast_buffers=False),
        precision="bf16-true",
    )
    fabric.launch(
        run_sdpa_distributed_vs_single_on_chunk,
        config=config,
        q_len=q_len,
        input_pos=input_pos,
        do_q_lens=do_q_lens,
        data_all=data_all,
        atol=atol,
        rtol=rtol,
    )


def run_sdpa_distributed_vs_single_on_chunk(
    fabric: L.Fabric,
    config: Config,
    q_len,
    input_pos,
    do_q_lens,
    data_all,
    atol,
    rtol,
):
    rank = fabric.local_rank
    num_devices = fabric.world_size
    device = torch.device("cuda", rank)
    data, q_inds = distribute_and_reorder_data(data_all, num_devices, input_pos)
    data = {k: v.to(device=device) for k, v in data[rank].items()}
    # Distributed computation
    if do_q_lens:
        max_val = data["query"].shape[2]
        q_lens = [max_val + 4]
    else:
        q_lens = None
    flexatt_args_diag = RingDiagFlexAttentionArgs(q_lens=q_lens)
    flexatt_args_offdiag = RingOffdiagFlexAttentionArgs(
        num_devices=num_devices, q_lens=q_lens,
    )
    ring_att_comp = RingAttentionComputation(
        rank_r=rank,
        flexatt_args_diag=flexatt_args_diag,
        flexatt_args_offdiag=flexatt_args_offdiag,
    )
    driver = RingAttentionDriver(ring_att_comp)
    outputs = driver(
        queries=data["query"],
        keys=data["key"],
        values=data["value"],
        scale=None,
        input_pos=input_pos,
        num_new_tokens=q_len,
        config=config,
    )
    dist_outputs = [x for x in fabric.all_gather(outputs)]

    # Only on master rank
    if rank == 0:
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
        # Comparison
        for _rank, (d_output, s_output) in enumerate(
            zip(dist_outputs, single_outputs)
        ):
            print(f"Outputs for rank {_rank}")
            torch.testing.assert_close(d_output, s_output, atol=atol, rtol=rtol)


# TODO: Less cases
@_RunIf(min_cuda_gpus=3)
@pytest.mark.parametrize(
    "n_head, n_query_groups, kv_len_per_rank, dtype, num_devices",
    [
        (4, 2, 512, torch.float16, 3),
        (4, 4, 256, torch.bfloat16, 2),
        (8, 4, 128, torch.float16, 3),
        (12, 4, 512, torch.bfloat16, 2),
        (24, 8, 256, torch.float16, 3),
        (9, 3, 256, torch.bfloat16, 3),
    ][0:1],
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
    # Generate all data on the main process (CPU) and distribute per rank.
    # Workers receive their slice via the mp.spawn args tuple.
    print("Sample data")
    data_all = random_args_cache_forward(params, num=kv_len, vocab_size=config.vocab_size)

    mp.spawn(
        run_sdpa_distributed_vs_single_on_prefill,
        args=(num_devices, config, data_all, atol, rtol),
        nprocs=num_devices,
        join=True,
    )


def run_sdpa_distributed_vs_single_on_prefill(
    rank: int,
    num_devices: int,
    config: Config,
    data_all,
    atol,
    rtol,
):
    # Set the device BEFORE init_process_group so NCCL registers its
    # communicator under the correct device for this rank.
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=num_devices,
        rank=rank,
    )
    prefix = f"[Rank {rank}]: "
    data_for_rank, q_inds = distribute_and_reorder_data(data_all, num_devices, input_pos=0)
    try:
        kv_len = data_for_rank[rank]["key"].shape[2]
        input_pos = 0
        data = {
            k: v.to(device=device) for k, v in data_for_rank[rank].items()
            if k in ("query", "key", "value")
        }
        del data_for_rank

        print(prefix + "Create driver")
        flexatt_args_diag = RingDiagFlexAttentionArgs()
        flexatt_args_offdiag = RingOffdiagFlexAttentionArgs(num_devices=num_devices)
        ring_att_comp = RingAttentionComputation(
            rank_r=rank,
            flexatt_args_diag=flexatt_args_diag,
            flexatt_args_offdiag=flexatt_args_offdiag,
        )
        driver = RingAttentionDriver(ring_att_comp)
        print(prefix + "Distributed computation")
        outputs = driver(
            queries=data["query"],
            keys=data["key"],
            values=data["value"],
            scale=None,
            input_pos=input_pos,
            num_new_tokens=kv_len * num_devices,
            config=config,
        )[0]

        # Gather all per-rank outputs onto rank 0
        print(prefix + "Gather outputs")
        outputs_gathered = [torch.zeros_like(outputs) for _ in range(num_devices)]
        dist.all_gather(outputs_gathered, outputs)

        if rank == 0:
            print(prefix + "Compute outputs on single node")
            flexatt_args = FlexAttentionArgs()
            output_all = scaled_dot_product_attention_flexatt(
                flexatt_args=flexatt_args,
                query=data_all["query"].to(device=device),
                key=data_all["key"].to(device=device),
                value=data_all["value"].to(device=device),
                scale_factor=None,
                sliding_window_size=None,
                attention_logit_softcapping=None,
                input_pos=input_pos,
                token_positions=None,
            )
            single_outputs = [output_all[:, :, q_ind, :] for q_ind in q_inds]
            for _rank, (d_output, s_output) in enumerate(
                zip(outputs_gathered, single_outputs)
            ):
                print(f"Comparison for rank {_rank}")
                torch.testing.assert_close(d_output, s_output, atol=atol, rtol=rtol)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    test_sdpa_distributed_vs_single_on_prefill(4, 2, 512, torch.float16, 3)
