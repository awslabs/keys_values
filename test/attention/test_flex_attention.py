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
from typing import List, Tuple, Dict

import pytest
import torch

from litgpt.utils import _RunIf

from keys_values.attention.attention_utils import sample_token_positions
from keys_values.attention.base import MultiHeadSelfAttention, DefaultKeysAndValues
from keys_values.attention.flex_attention import (
    FlexAttentionArgs,
    scaled_dot_product_attention_flexatt,
)
from keys_values.config import Config
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.test_utils import random_args_cache_forward
from keys_values.utils import index_to_3d, randint_torch


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "tp_ndim, sort_if_3d",
    list(product((0, 1, 3, None), (False, True))),
)
def test_flexatt_working(tp_ndim, sort_if_3d):
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
            sort_if_3d=sort_if_3d,
        )
        print(attn_outputs.sum().item())
        # Process chunk
        if tp_ndim == 0:
            token_positions = None
        elif tp_ndim == 1:
            _ind = sample_token_positions(
                batch_size=1,
                n_query_groups=1,
                q_len=chunk_size,
                kv_len=cache_length,
                input_pos=cache_length,
                device=device,
            ).flatten()
            token_positions = index_to_3d(
                _ind,
                batch_size,
                n_query_groups,
            )
        else:
            token_positions = sample_token_positions(
                batch_size,
                n_query_groups,
                chunk_size,
                cache_length,
                input_pos=cache_length,
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
            sort_if_3d=sort_if_3d,
        )
        print(attn_outputs.sum().item())


def gen_data_for_q_kv_lens(
    q_kv_lens: List[Tuple[int, int]],
    params: KVCacheParams,
    device: torch.device,
    batch_size: int,
    n_query_groups: int,
    vocab_size: int,
    tp_ndim: int,
) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor]]:
    data = [
        random_args_cache_forward(
            params,
            num=q_kv_lens[0][1],
            vocab_size=vocab_size,
            device=device,
        )
    ]
    token_positions = []
    input_pos = kvl_prev = q_kv_lens[0][1]
    for ql, kvl in q_kv_lens[1:]:
        diff = kvl - kvl_prev
        kvl_prev = kvl
        data.append(
            random_args_cache_forward(
                params,
                num=ql,
                vocab_size=vocab_size,
                device=device,
            )
        )
        if diff > 0:
            for name in ("key", "value"):
                data[-1][name] = torch.cat(
                    (data[-2][name], data[-1][name]),
                    dim=2,
                )
                assert data[-1][name].shape[2] == kvl
            token_positions.append(
                index_to_3d(
                    torch.arange(kvl, device=device),
                    batch_size,
                    n_query_groups,
                )
            )
        else:
            for name in ("key", "value"):
                pos = randint_torch(0, kvl - ql)
                new_part = data[-1][name]
                data[-1][name] = data[-2][name]
                data[-1][name][:, :, pos : (pos + ql), :] = new_part
            if tp_ndim == 1:
                _ind = sample_token_positions(
                    batch_size=1,
                    n_query_groups=1,
                    q_len=ql,
                    kv_len=kvl,
                    input_pos=input_pos,
                    device=device,
                ).flatten()
                token_positions.append(index_to_3d(_ind, batch_size, n_query_groups))
            else:
                token_positions.append(
                    sample_token_positions(
                        batch_size,
                        n_query_groups,
                        ql,
                        kvl,
                        input_pos=input_pos,
                        device=device,
                    )
                )
        input_pos += ql

    return data, token_positions


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "n_head, n_query_groups, q_len, kv_len, dtype, attention_logit_softcapping, atol, tp_ndim",
    [
        a + (b,)
        for a, b in product(
            [
                (4, 2, 128, 512, torch.float16, None, 0.0002),
                (4, 4, 8, 256, torch.bfloat16, None, 0.0008),
                (8, 4, 32, 128, torch.float16, None, 0.0002),
                (12, 4, 16, 512, torch.bfloat16, None, 0.002),
                (24, 8, 2, 512, torch.float16, None, 0.0002),
                (9, 3, 128, 512, torch.bfloat16, None, 0.002),
                (12, 4, 16, 512, torch.float16, 5, 0.0004),
                (24, 8, 2, 512, torch.bfloat16, 2, 0.004),
                (12, 4, 16, 512, torch.float16, 5, 0.0004),
                (9, 3, 128, 512, torch.float16, 2, 0.0004),
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
    attention_logit_softcapping,
    atol,
    tp_ndim,
):
    seed = 31415927
    torch.manual_seed(seed)

    batch_size = 2
    head_size = 32
    device = torch.device("cuda", 0)
    q_kv_lens = [
        (kv_len, kv_len),
        (q_len, kv_len),
        (q_len, kv_len),
    ]

    config = Config.from_name(
        "gemma-2-27b",
        block_size=3 * kv_len,
        sliding_window_size=None,
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
    data, token_positions = gen_data_for_q_kv_lens(
        q_kv_lens=q_kv_lens,
        params=params,
        device=device,
        batch_size=batch_size,
        n_query_groups=n_query_groups,
        vocab_size=config.vocab_size,
        tp_ndim=tp_ndim,
    )

    # Competitors
    flexatt_args = FlexAttentionArgs()
    names = ["no_flexatt", "flexatt"]
    mhas = [
        MultiHeadSelfAttention(config),
        MultiHeadSelfAttention(config, flexatt_args=flexatt_args),
    ]
    attn_outputs = [[] for _ in range(len(q_kv_lens))]
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


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "n_head, n_query_groups, q_len, kv_len, dtype, attention_logit_softcapping, atol, tp_ndim",
    [
        a + (b,)
        for a, b in product(
            [
                (4, 2, 128, 512, torch.float16, None, 0.0002),
                (4, 4, 8, 256, torch.bfloat16, None, 0.004),
                (8, 4, 32, 128, torch.float16, None, 0.0008),
                (12, 4, 16, 512, torch.bfloat16, None, 0.004),
                (24, 8, 2, 512, torch.float16, None, 0.0003),
                (9, 9, 128, 512, torch.bfloat16, None, 0.004),
                (12, 4, 16, 512, torch.float16, 5, 0.0004),
                (24, 8, 2, 512, torch.bfloat16, 2, 0.004),
                (12, 4, 16, 512, torch.float16, 5, 0.0004),
                (9, 9, 128, 512, torch.float16, 2, 0.0004),
            ],
            [1, 3],
        )
    ],
)
def test_comparison_with_attn_weights(
    n_head,
    n_query_groups,
    q_len,
    kv_len,
    dtype,
    attention_logit_softcapping,
    atol,
    tp_ndim,
):
    seed = 31415927
    torch.manual_seed(seed)

    batch_size = 2
    head_size = 32
    device = torch.device("cuda", 0)
    q_kv_lens = [
        (kv_len, kv_len),
        (q_len, kv_len),
        (q_len, kv_len),
    ]

    config = Config.from_name(
        "gemma-2-27b",
        block_size=3 * kv_len,
        sliding_window_size=None,
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
    data, token_positions = gen_data_for_q_kv_lens(
        q_kv_lens=q_kv_lens,
        params=params,
        device=device,
        batch_size=batch_size,
        n_query_groups=n_query_groups,
        vocab_size=config.vocab_size,
        tp_ndim=tp_ndim,
    )

    # Competitors
    flexatt_args = FlexAttentionArgs(forward_return_lse=True)
    names = ["no_flexatt", "flexatt"]
    mhas = [
        MultiHeadSelfAttention(config),
        MultiHeadSelfAttention(
            config,
            flexatt_args=flexatt_args,
            use_flexattn_for_prefill=True,
        ),
    ]
    attn_outputs = [[] for _ in range(len(q_kv_lens))]
    attn_weights = [[] for _ in range(len(q_kv_lens))]
    for mha, name in zip(mhas, names):
        input_pos = 0
        print(f"MHA: {name}")
        for i, chunk in enumerate(data):
            print(f"chunk: {i}")
            tp = None if i == 0 else token_positions[i - 1]
            outputs, attn_wgts = mha(
                query=chunk["query"],
                k_and_v=DefaultKeysAndValues(chunk["key"], chunk["value"]),
                block_idx=0,
                input_pos=input_pos,
                return_attn_weights=input_pos > 0,
                token_positions=tp,
            )
            attn_outputs[i].append(outputs)
            if i > 0:
                attn_weights[i].append(attn_wgts)
            input_pos += chunk["query"].shape[2]
    # Comparison
    test_kwargs = dict(atol=atol, rtol=1)
    for i, (outputs, attn_wgts) in enumerate(zip(attn_outputs, attn_weights)):
        prefix = f"Chunk {i}: "
        print(prefix + "no_flexatt vs flexatt: attn_output")
        torch.testing.assert_close(outputs[0], outputs[1], **test_kwargs)
        if i > 0:
            print(prefix + "no_flexatt vs flexatt: attn_weights")
            torch.testing.assert_close(attn_wgts[0], attn_wgts[1], **test_kwargs)


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "n_head, n_query_groups, kv_len, dtype, attention_logit_softcapping, atol",
    [
        (4, 2, 512, torch.float16, None, 0.0004),
        (4, 4, 256, torch.bfloat16, None, 0.005),
        (8, 4, 128, torch.float16, None, 0.0002),
        (12, 4, 512, torch.bfloat16, None, 0.005),
        (24, 8, 512, torch.float16, None, 0.0004),
        (9, 3, 512, torch.bfloat16, None, 0.005),
        (12, 4, 512, torch.float16, 5, 0.0004),
        (24, 8, 512, torch.bfloat16, 2, 0.005),
        (12, 4, 512, torch.float16, 5, 0.0004),
        (9, 3, 512, torch.float16, 2, 0.0004),
    ],
)
def test_comparison_padding_prefill(
    n_head,
    n_query_groups,
    kv_len,
    dtype,
    attention_logit_softcapping,
    atol,
):
    seed = 31415927
    torch.manual_seed(seed)

    batch_size = 2
    head_size = 32
    device = torch.device("cuda", 0)
    pad_lengths = [1, 3, 16, 53]

    config = Config.from_name(
        "gemma-2-27b",
        block_size=3 * kv_len,
        sliding_window_size=None,
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
    datas = [
        random_args_cache_forward(
            params,
            num=kv_len - pad_length,
            vocab_size=config.vocab_size,
            device=device,
        )
        for pad_length in pad_lengths
    ]

    # For a number of lengths < `kv_len`, we compare FlexAttn with KV
    # padding to not using FlexAttn
    flexatt_args = FlexAttentionArgs(kv_lens=[kv_len])
    names = ["no_flexatt", "flexatt"]
    mhas = [
        MultiHeadSelfAttention(config),
        MultiHeadSelfAttention(
            config,
            flexatt_args=flexatt_args,
            use_flexattn_for_prefill=True,
        ),
    ]
    attn_outputs = [[] for _ in range(len(pad_lengths))]
    for mha, name in zip(mhas, names):
        print(f"MHA: {name}")
        for i, (pad_length, data) in enumerate(zip(pad_lengths, datas)):
            print(f"pad_length: {pad_length}")
            outputs, _ = mha(
                query=data["query"],
                k_and_v=DefaultKeysAndValues(data["key"], data["value"]),
                block_idx=0,
                input_pos=0,
            )
            attn_outputs[i].append(outputs)
    # Comparison
    test_kwargs = dict(atol=atol, rtol=1)
    for outputs, pad_length in zip(attn_outputs, pad_lengths):
        prefix = f"pad_length {pad_length}: "
        print(prefix + "no_flexatt vs flexatt")
        torch.testing.assert_close(outputs[0], outputs[1], **test_kwargs)


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "n_head, n_query_groups, q_len, kv_len, dtype, attention_logit_softcapping, atol, tp_ndim",
    [
        a + (b,)
        for a, b in product(
            [
                (4, 2, 128, 512, torch.float16, None, 0.0004),
                (4, 4, 8, 256, torch.bfloat16, None, 0.005),
                (8, 4, 32, 128, torch.float16, None, 0.0002),
                (12, 4, 16, 512, torch.bfloat16, None, 0.005),
                (24, 8, 2, 512, torch.float16, None, 0.0004),
                (9, 9, 128, 512, torch.bfloat16, None, 0.005),
                (12, 4, 16, 512, torch.float16, 5, 0.0004),
                (24, 8, 2, 512, torch.bfloat16, 2, 0.005),
                (12, 4, 16, 512, torch.float16, 5, 0.0004),
                (9, 9, 128, 512, torch.float16, 2, 0.0004),
            ],
            [1, 3],
        )
    ],
)
def test_comparison_padding_chunk(
    n_head,
    n_query_groups,
    q_len,
    kv_len,
    dtype,
    attention_logit_softcapping,
    atol,
    tp_ndim,
):
    seed = 31415927
    torch.manual_seed(seed)

    batch_size = 2
    head_size = 32
    device = torch.device("cuda", 0)
    qstep = max(q_len // 5, 1)
    q_kv_lens = [
        (kv_len - 5, kv_len - 5),  # prefill
        (1, kv_len - 4),
        (2, kv_len - 2),
        (1, kv_len - 1),
    ] + [(ql, kv_len) for ql in range(1, q_len + 1, qstep)]

    config = Config.from_name(
        "gemma-2-27b",
        block_size=3 * kv_len,
        sliding_window_size=None,
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
    data, token_positions = gen_data_for_q_kv_lens(
        q_kv_lens=q_kv_lens,
        params=params,
        device=device,
        batch_size=batch_size,
        n_query_groups=n_query_groups,
        vocab_size=config.vocab_size,
        tp_ndim=tp_ndim,
    )

    # For a number of lengths < `kv_len`, we compare FlexAttn with KV
    # padding to not using FlexAttn. We also use Q padding
    flexatt_args = FlexAttentionArgs(
        kv_lens=[kv_len],
        q_lens=[1, q_len],
    )
    names = ["no_flexatt", "flexatt"]
    mhas = [
        MultiHeadSelfAttention(config),
        MultiHeadSelfAttention(
            config,
            flexatt_args=flexatt_args,
            use_flexattn_for_prefill=True,
        ),
    ]
    attn_outputs = [[] for _ in range(len(q_kv_lens))]
    for mha, name in zip(mhas, names):
        print(f"MHA: {name}")
        input_pos = 0
        for i, (chunk, (ql, kvl)) in enumerate(zip(data, q_kv_lens)):
            print(f"Chunk ql={ql}, kvl={kvl}")
            tp = None if i == 0 else token_positions[i - 1]
            outputs, _ = mha(
                query=chunk["query"],
                k_and_v=DefaultKeysAndValues(chunk["key"], chunk["value"]),
                block_idx=0,
                input_pos=input_pos,
                token_positions=tp,
            )
            attn_outputs[i].append(outputs)
            input_pos += ql
    # Comparison
    test_kwargs = dict(atol=atol, rtol=1)
    for outputs, (ql, kvl) in zip(attn_outputs, q_kv_lens):
        print(f"Chunk ql={ql}, kvl={kvl}: no_flexatt vs flexatt")
        torch.testing.assert_close(outputs[0], outputs[1], **test_kwargs)


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "n_head, n_query_groups, q_len, kv_len, dtype, attention_logit_softcapping, atol, tp_ndim",
    [
        a + (b,)
        for a, b in product(
            [
                (4, 2, 128, 512, torch.float16, None, 0.0004),
                (4, 4, 8, 256, torch.bfloat16, None, 0.005),
                (8, 4, 32, 128, torch.float16, None, 0.0002),
                (12, 4, 16, 512, torch.bfloat16, None, 0.005),
                (24, 8, 2, 512, torch.float16, None, 0.0004),
                (9, 9, 128, 512, torch.bfloat16, None, 0.005),
                (12, 4, 16, 512, torch.float16, 5, 0.0004),
                (24, 8, 2, 512, torch.bfloat16, 2, 0.005),
                (12, 4, 16, 512, torch.float16, 5, 0.0004),
                (9, 9, 128, 512, torch.float16, 2, 0.0004),
            ],
            [1, 3],
        )
    ],
)
def test_comparison_padding_with_attn_weights(
    n_head,
    n_query_groups,
    q_len,
    kv_len,
    dtype,
    attention_logit_softcapping,
    atol,
    tp_ndim,
):
    seed = 31415927
    torch.manual_seed(seed)

    batch_size = 2
    head_size = 32
    device = torch.device("cuda", 0)
    qstep = max(q_len // 5, 1)
    q_kv_lens = [
        (kv_len - 5, kv_len - 5),  # prefill
        (1, kv_len - 4),
        (2, kv_len - 2),
        (1, kv_len - 1),
    ] + [(ql, kv_len) for ql in range(1, q_len + 1, qstep)]

    config = Config.from_name(
        "gemma-2-27b",
        block_size=3 * kv_len,
        sliding_window_size=None,
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
    data, token_positions = gen_data_for_q_kv_lens(
        q_kv_lens=q_kv_lens,
        params=params,
        device=device,
        batch_size=batch_size,
        n_query_groups=n_query_groups,
        vocab_size=config.vocab_size,
        tp_ndim=tp_ndim,
    )

    # Competitors
    flexatt_args = FlexAttentionArgs(
        kv_lens=[kv_len],
        q_lens=[1, q_len],
        forward_return_lse=True,
    )
    names = ["no_flexatt", "flexatt"]
    mhas = [
        MultiHeadSelfAttention(config),
        MultiHeadSelfAttention(
            config,
            flexatt_args=flexatt_args,
            use_flexattn_for_prefill=True,
        ),
    ]
    attn_outputs = [[] for _ in range(len(q_kv_lens))]
    attn_weights = [[] for _ in range(len(q_kv_lens))]
    for mha, name in zip(mhas, names):
        print(f"MHA: {name}")
        input_pos = 0
        for i, (chunk, (ql, kvl)) in enumerate(zip(data, q_kv_lens)):
            print(f"Chunk ql={ql}, kvl={kvl}")
            tp = None if i == 0 else token_positions[i - 1]
            outputs, attn_wgts = mha(
                query=chunk["query"],
                k_and_v=DefaultKeysAndValues(chunk["key"], chunk["value"]),
                block_idx=0,
                input_pos=input_pos,
                return_attn_weights=input_pos > 0,
                token_positions=tp,
            )
            attn_outputs[i].append(outputs)
            if i > 0:
                attn_weights[i].append(attn_wgts)
            input_pos += ql
    # Comparison
    test_kwargs = dict(atol=atol, rtol=1)
    for i, (outputs, attn_wgts) in enumerate(zip(attn_outputs, attn_weights)):
        prefix = f"Chunk {i}: "
        print(prefix + "no_flexatt vs flexatt: attn_output")
        torch.testing.assert_close(outputs[0], outputs[1], **test_kwargs)
        if i > 0:
            print(prefix + "no_flexatt vs flexatt: attn_weights")
            torch.testing.assert_close(attn_wgts[0], attn_wgts[1], **test_kwargs)


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "n_head, n_query_groups, kv_len, dtype, attention_logit_softcapping, atol, tp_ndim",
    [
        a + (b,)
        for a, b in product(
            [
                (4, 2, 256, torch.float16, None, 0.0004),
                (4, 4, 128, torch.bfloat16, None, 0.005),
                (8, 4, 64, torch.float16, None, 0.0002),
                (12, 4, 512, torch.bfloat16, None, 0.005),
                (24, 8, 256, torch.float16, None, 0.0004),
                (9, 9, 256, torch.bfloat16, None, 0.005),
                (12, 4, 128, torch.float16, 5, 0.0004),
                (24, 8, 256, torch.bfloat16, 2, 0.005),
                (12, 4, 128, torch.float16, 5, 0.0004),
                (9, 9, 256, torch.float16, 2, 0.0004),
            ],
            [1, 3],
        )
    ],
)
def test_padding_token_generation(
    n_head,
    n_query_groups,
    kv_len,
    dtype,
    attention_logit_softcapping,
    atol,
    tp_ndim,
):
    seed = 31415927
    torch.manual_seed(seed)

    batch_size = 2
    head_size = 32
    device = torch.device("cuda", 0)
    num_tokens = 16
    q_kv_lens = [(kv_len - num_tokens, kv_len - num_tokens)] + [
        (1, kv_len - i) for i in range(num_tokens - 1, -1, -1)
    ]
    kv_lens = [kv_len - 12, kv_len - 8, kv_len - 4, kv_len]

    config = Config.from_name(
        "gemma-2-27b",
        block_size=3 * kv_len,
        sliding_window_size=None,
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
    data, token_positions = gen_data_for_q_kv_lens(
        q_kv_lens=q_kv_lens,
        params=params,
        device=device,
        batch_size=batch_size,
        n_query_groups=n_query_groups,
        vocab_size=config.vocab_size,
        tp_ndim=tp_ndim,
    )

    # Competitors
    flexatt_args = FlexAttentionArgs(
        kv_lens=kv_lens,
        q_lens=[1],
    )
    names = ["no_flexatt", "flexatt"]
    mhas = [
        MultiHeadSelfAttention(config),
        MultiHeadSelfAttention(
            config,
            flexatt_args=flexatt_args,
            use_flexattn_for_prefill=True,
        ),
    ]
    attn_outputs = [[] for _ in range(len(q_kv_lens))]
    for mha, name in zip(mhas, names):
        print(f"MHA: {name}")
        input_pos = 0
        for i, (chunk, (ql, kvl)) in enumerate(zip(data, q_kv_lens)):
            print(f"Chunk ql={ql}, kvl={kvl}")
            tp = None if i == 0 else token_positions[i - 1]
            outputs, _ = mha(
                query=chunk["query"],
                k_and_v=DefaultKeysAndValues(chunk["key"], chunk["value"]),
                block_idx=0,
                input_pos=input_pos,
                token_positions=tp,
            )
            attn_outputs[i].append(outputs)
            input_pos += ql

    # Comparison
    test_kwargs = dict(atol=atol, rtol=1)
    for outputs, (ql, kvl) in zip(attn_outputs, q_kv_lens):
        print(f"Chunk ql={ql}, kvl={kvl}: no_flexatt vs flexatt")
        torch.testing.assert_close(outputs[0], outputs[1], **test_kwargs)

    # How often has each graph been used?
    print("Testing number of hits for prefill")
    num_hits = flexatt_args.attn_prefill_manager.num_hits
    assert len(num_hits) == 1, num_hits
    for arg, num in num_hits.items():
        assert num == 1, num_hits
        assert arg[0] == kv_lens[0]
    print("Testing number of hits for chunks")
    num_hits = flexatt_args.attn_chunk_manager.num_hits
    assert len(num_hits) == 4, num_hits
    _kv_lens = set()
    for arg, num in num_hits.items():
        assert num == 4, num_hits
        _kv_lens.add(arg[1])
        assert arg[0] == 1
    assert _kv_lens == set(kv_lens), num_hits
