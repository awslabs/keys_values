# Original Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# Modification Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import math

import pytest
import torch

from keys_values.config import Config
from litgpt.utils import _RunIf

from keys_values.attention import MultiHeadSelfAttention, DefaultKeysAndValues
from keys_values.flex_attention import (
    FlexAttentionArgs,
    scaled_dot_product_attention_flexatt,
)
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.test_utils import (
    random_args_cache_forward,
    random_index,
)
from keys_values.utils import index_to_3d


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "tp_ndim",
    (0, 1, 3, None),
)
def test_flexatt_working(tp_ndim):
    seed = 31415927
    torch.manual_seed(seed)

    batch_size = 2
    n_head = 32
    n_query_groups = 8
    cache_length = 2**12
    head_size = 128
    chunk_size = 2**10
    device = torch.device("cuda", 0)
    dtype = torch.float16
    scale_factor = 1.0 / math.sqrt(head_size)
    shared_manager = tp_ndim is None
    if shared_manager:
        tp_ndims = (0, 1, 3)
    else:
        tp_ndims = (tp_ndim,)

    config = Config(
        n_layer=1,
        n_head=n_head,
        n_query_groups=n_query_groups,
        n_embd=n_head * head_size,
        block_size=cache_length + 2 * chunk_size,
        vocab_size=128,
        rotary_percentage=1,
    )
    params = KVCacheParams.from_config(
        config=config,
        max_batch_size=batch_size,
        cache_length=cache_length,
        dtype=dtype,
    )
    data_prefill = random_args_cache_forward(
        params,
        num=cache_length,
        vocab_size=config.vocab_size,
        device=device,
    )
    data_chunk = random_args_cache_forward(
        params,
        num=chunk_size,
        vocab_size=config.vocab_size,
        device=device,
    )
    diff = cache_length - chunk_size
    for name in ("key", "value"):
        data_chunk[name] = torch.cat(
            (data_prefill[name][:, :, (-diff):, :], data_chunk[name]),
            dim=2,
        )

    if shared_manager:
        flexatt_args_global = FlexAttentionArgs()
    else:
        flexatt_args_global = None
    for tp_ndim in tp_ndims:
        print(f"shared_manager = {shared_manager}, tp_ndim = {tp_ndim}")
        if shared_manager:
            flexatt_args = flexatt_args_global
        else:
            flexatt_args = FlexAttentionArgs()
        # Prefill
        print(f"Computing prefill MHA (cache_length={cache_length})")
        attn_outputs = scaled_dot_product_attention_flexatt(
            flexatt_args=flexatt_args,
            query=data_prefill["query"],
            key=data_prefill["key"],
            value=data_prefill["value"],
            scale_factor=scale_factor,
            sliding_window_size=None,
            attention_logit_softcapping=None,
            input_pos=0,
            token_positions=None,
        )
        print(attn_outputs.sum().item())
        # Process chunk
        if tp_ndim == 0:
            token_positions = None
        elif tp_ndim == 1:
            _ind = random_index(
                params,
                start=0,
                end=cache_length + chunk_size,
                num=cache_length,
                batch_size=1,
                device=device,
            )
            token_positions = index_to_3d(
                _ind[0, 0, :],
                batch_size,
                n_query_groups,
            )
        else:
            token_positions = random_index(
                params,
                start=0,
                end=cache_length + chunk_size,
                num=cache_length,
                device=device,
            )
        print(f"Computing chunk MHA (chunk_size={chunk_size})")
        attn_outputs = scaled_dot_product_attention_flexatt(
            flexatt_args=flexatt_args,
            query=data_chunk["query"],
            key=data_chunk["key"],
            value=data_chunk["value"],
            scale_factor=scale_factor,
            sliding_window_size=None,
            attention_logit_softcapping=None,
            input_pos=cache_length,
            token_positions=token_positions,
        )
        print(attn_outputs.sum().item())


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "n_head, n_query_groups, q_len, kv_len, dtype, sliding_window_size, attention_logit_softcapping, tp_ndim, atol",
    [
        a + (b,)
        for a, b in product(
            [
                (4, 2, 128, 512, torch.float16, None, None, 0.0002),
                (4, 4, 8, 256, torch.bfloat16, None, None, 0.0008),
                (8, 4, 32, 128, torch.float16, None, None, 0.0002),
                (12, 4, 16, 512, torch.bfloat16, None, None, 0.002),
                (24, 8, 2, 512, torch.float16, None, None, 0.0002),
                (9, 3, 128, 512, torch.bfloat16, None, None, 0.002),
                (12, 4, 16, 512, torch.float16, 12, None, 0.0003),
                (24, 8, 2, 512, torch.bfloat16, 64, None, 0.0008),
                (9, 3, 128, 512, torch.float16, 96, None, 0.0002),
                (12, 4, 16, 512, torch.float16, None, 5, 0.0004),
                (24, 8, 2, 512, torch.bfloat16, None, 2, 0.004),
                (12, 4, 16, 512, torch.float16, 64, 5, 0.0004),
                (9, 3, 128, 512, torch.float16, 12, 2, 0.0004),
            ],
            [1, 3],
        )
    ],
)
def test_comparison(
    n_head,
    n_query_groups,
    q_len,
    kv_len,
    dtype,
    sliding_window_size,
    attention_logit_softcapping,
    tp_ndim,
    atol,
):
    seed = 31415927
    torch.manual_seed(seed)

    batch_size = 2
    head_size = 32
    num_chunks = 2
    device = torch.device("cuda", 0)

    config = Config.from_name(
        "gemma-2-27b",
        block_size=3 * kv_len,
        sliding_window_size=sliding_window_size,
        attention_logit_softcapping=attention_logit_softcapping,
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
    data = [
        random_args_cache_forward(
            params,
            num=kv_len,
            vocab_size=config.vocab_size,
            device=device,
        )
    ]
    diff = kv_len - q_len
    token_positions = []
    input_pos = kv_len
    for chunk in range(1, num_chunks + 1):
        data.append(
            random_args_cache_forward(
                params,
                num=q_len,
                vocab_size=config.vocab_size,
                device=device,
            )
        )
        for name in ("key", "value"):
            data[chunk][name] = torch.cat(
                (data[chunk - 1][name][:, :, (-diff):, :], data[chunk][name]),
                dim=2,
            )
            assert data[chunk][name].shape[2] == kv_len
        # We need a valid `token_positions` here, which covers
        # `range(input_pos, input_pos + q_len)`
        end = input_pos + q_len
        start = end - kv_len
        if tp_ndim == 1:
            _ind = random_index(
                params,
                start=start,
                end=end,
                num=kv_len,
                batch_size=1,
                device=device,
            )
            token_positions.append(
                index_to_3d(_ind[0, 0, :], batch_size, n_query_groups)
            )
        else:
            token_positions.append(
                random_index(
                    params,
                    start=start,
                    end=end,
                    num=kv_len,
                    device=device,
                )
            )
        input_pos += q_len

    # Competitors
    flexatt_args = FlexAttentionArgs()
    names = ["no_flexatt", "flexatt"]
    mhas = [
        MultiHeadSelfAttention(config),
        MultiHeadSelfAttention(config, flexatt_args=flexatt_args),
    ]
    attn_outputs = [[] for _ in range(num_chunks + 1)]
    for mha, name in zip(mhas, names):
        input_pos = 0
        print(f"MHA: {name}")
        for i, chunk in enumerate(data):
            print(f"chunk: {i}")
            tp = None if i == 0 else token_positions[i - 1]
            outputs, _ = mha(
                query=chunk["query"],
                k_and_v=DefaultKeysAndValues(chunk["key"], chunk["value"]),
                block_idx=0,
                input_pos=input_pos,
                token_positions=tp,
            )
            attn_outputs[i].append(outputs)
            input_pos += chunk["query"].shape[2]
    # Comparison
    test_kwargs = dict(atol=atol, rtol=1)
    for i, outputs in enumerate(attn_outputs):
        prefix = f"Chunk {i}: "
        print(prefix + "no_flexatt vs flexatt")
        torch.testing.assert_close(outputs[0], outputs[1], **test_kwargs)
