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
import math
import random

import torch
from torch.nn import functional as F
import pytest

from keys_values.attention_utils import (
    build_mask_slice,
    sample_token_positions,
    ENTRIES_PER_GB,
    FUSED_SDPA_DOES_NOT_SUPPORT_ENABLE_GQA,
)
from keys_values.kvcache.gradient.accumulate import copy_requires_grad
from keys_values.kvcache.gradient.sdpa_op import SDPAFunction, sdpa_backward
from keys_values.kvcache.test_utils import product_with_devices


@pytest.mark.parametrize(
    ("n_head", "n_query_groups", "q_len", "kv_len", "dtype", "sliding_window_size", "device"),
    product_with_devices(
        [
            (4, 2, 128, 512, torch.float32, None),
            (4, 4, 1, 256, torch.float32, None),
            (8, 4, 128, 128, torch.float32, None),
            (12, 4, 16, 512, torch.float32, None),
            (24, 8, 2, 512, torch.float16, None),
            (9, 3, 128, 512, torch.bfloat16, None),
            (16, 16, 128, 512, torch.float16, None),
            (12, 4, 16, 512, torch.float32, 12),
            (24, 8, 2, 512, torch.float16, 64),
            (9, 3, 128, 512, torch.bfloat16, 96),
        ],
    ),
)
def test_sdpa_op_gradients(n_head, n_query_groups, q_len, kv_len, dtype, sliding_window_size, device):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    num_repeats = 32
    seq_len = 2 * kv_len
    is_causal = q_len == kv_len
    input_pos = seq_len - q_len if not is_causal else 0
    q_per_kv = n_head // n_query_groups
    enable_gqa = n_query_groups < n_head

    print(f"n_head={n_head}, n_query_groups={n_query_groups}, q_len={q_len}, kv_len={kv_len}, is_causal={is_causal}, dtype={dtype}, device={device}")
    kwargs = dict(device=device, dtype=dtype)
    for repeat in range(num_repeats):
        head_size = 2 ** random.randint(3, 6)
        batch_size = random.randint(1, 5)
        print(f"repeat={repeat}, head_size={head_size}, batch_size={batch_size}")
        if is_causal:
            token_positions = None
        else:
            token_positions = sample_token_positions(
                batch_size,
                n_query_groups,
                q_len,
                kv_len,
                input_pos,
                device=device,
            )
        shape = (batch_size, n_head, q_len, head_size)
        _query = torch.randn(shape, **kwargs)
        shape = (batch_size, n_query_groups, kv_len, head_size)
        _key = torch.randn(shape, **kwargs)
        _value = torch.randn(shape, **kwargs)
        print(f"query {_query.shape}, key {_key.shape}, value {_value.shape}")
        if token_positions is not None:
            print(f"token_positions {token_positions.shape}")
        scale = 1.0 / math.sqrt(head_size)
        gradients = dict()
        for kind in ("op", "noop"):
            query = copy_requires_grad(_query)
            key = copy_requires_grad(_key)
            value = copy_requires_grad(_value)
            if kind == "op":
                y = SDPAFunction.apply(
                    query,
                    key,
                    value,
                    token_positions,
                    input_pos,
                    scale,
                    sliding_window_size,
                )
            else:
                if enable_gqa and FUSED_SDPA_DOES_NOT_SUPPORT_ENABLE_GQA:
                    # Some efficient kernels have not reliably implemented
                    # `enabla_gqa=True`. It is better to extend keys, values in
                    # this case.
                    _dtype = torch.float32
                    key2 = torch.repeat_interleave(
                        key.to(_dtype), q_per_kv, dim=1,
                    )
                    value2 = torch.repeat_interleave(
                        value.to(_dtype), q_per_kv, dim=1,
                    )
                    query2 = query.to(_dtype)
                    _enable_gqa = False
                else:
                    query2 = query
                    key2 = key
                    value2 = value
                    _enable_gqa = enable_gqa
                    _dtype = dtype
                if is_causal:
                    mask = None
                else:
                    mask = build_mask_slice(
                        input_pos=input_pos,
                        num=q_len,
                        token_positions=token_positions,
                        n_head=n_head,
                        dtype=_dtype,
                        sliding_window_size=sliding_window_size,
                    ).detach()
                    print(f"mask {mask.shape}")
                y = F.scaled_dot_product_attention(
                    query=query2,
                    key=key2,
                    value=value2,
                    attn_mask=mask,
                    dropout_p=0.0,
                    scale=scale,
                    is_causal=is_causal,
                    enable_gqa=_enable_gqa,
                )
            loss = y.sum()
            loss.backward()
            gradients[kind] = (query.grad, key.grad, value.grad)
        # Compare
        for name, grad_op, grad_noop in zip(
            ("query", "key", "value"), gradients["op"], gradients["noop"],
        ):
            print(f"Compare gradients for {name}")
            torch.testing.assert_close(
                grad_op, grad_noop, atol=0.0005, rtol=0.05,
            )


@pytest.mark.parametrize(
    ("n_head", "n_query_groups", "q_len", "kv_len", "dtype", "sliding_window_size", "device"),
    product_with_devices(
        [
            (4, 2, 128, 512, torch.float16, None),
            (4, 4, 8, 256, torch.bfloat16, None),
            (8, 4, 128, 128, torch.float16, None),
            (12, 4, 16, 512, torch.bfloat16, None),
            (24, 8, 2, 512, torch.float16, None),
            (9, 3, 128, 512, torch.bfloat16, None),
            (16, 16, 128, 512, torch.bfloat16, None),
            (16, 16, 128, 512, torch.float16, None),
            (12, 4, 16, 512, torch.float16, 12),
            (24, 8, 2, 512, torch.bfloat16, 64),
            (9, 3, 128, 512, torch.float16, 96),
        ],
    ),
)
def test_sdpa_backward(n_head, n_query_groups, q_len, kv_len, dtype, sliding_window_size, device):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    num_repeats = 32
    seq_len = 2 * kv_len
    is_causal = q_len == kv_len
    input_pos = seq_len - q_len if not is_causal else 0

    print(f"n_head={n_head}, n_query_groups={n_query_groups}, q_len={q_len}, kv_len={kv_len}, is_causal={is_causal}, dtype={dtype}, device={device}")
    kwargs = dict(device=device, dtype=dtype)
    for repeat in range(num_repeats):
        head_size = 2 ** random.randint(3, 6)
        batch_size = random.randint(1, 5)
        if q_len % 2 != 0 and batch_size % 2 != 0:
            batch_size += 1
        print(f"repeat={repeat}, head_size={head_size}, batch_size={batch_size}")
        if is_causal:
            token_positions = None
        else:
            token_positions = sample_token_positions(
                batch_size,
                n_query_groups,
                q_len,
                kv_len,
                input_pos,
                device=device,
            )
        shape = (batch_size, n_head, q_len, head_size)
        query = torch.randn(shape, **kwargs)
        grad_attn_output = torch.randn(shape, **kwargs)
        shape = (batch_size, n_query_groups, kv_len, head_size)
        key = torch.randn(shape, **kwargs)
        value = torch.randn(shape, **kwargs)
        print(f"query {query.shape}, key {key.shape}, value {value.shape}")
        if token_positions is not None:
            print(f"token_positions {token_positions.shape}")
        scale_factor = 1.0 / math.sqrt(head_size)
        gradients = {"query": [], "key": [], "value": []}
        numel_tmp = batch_size * n_head * q_len * kv_len
        for kind in ("no", "yes"):
            num_temp_entry_limit = 2 * numel_tmp if kind == "no" else numel_tmp // 2
            grad_query, grad_key, grad_value = sdpa_backward(
                grad_attn_output=grad_attn_output,
                query=query,
                key=key,
                value=value,
                token_positions=token_positions,
                input_pos=input_pos,
                scale_factor=scale_factor,
                sliding_window_size=sliding_window_size,
                need_query=True,
                need_key=True,
                need_value=True,
                tmp_array_limit_gb=num_temp_entry_limit / ENTRIES_PER_GB,
            )
            gradients["query"].append(grad_query)
            gradients["key"].append(grad_key)
            gradients["value"].append(grad_value)
        # Compare
        for name, grads in gradients.items():
            print(f"Compare gradients for {name}")
            torch.testing.assert_close(
                grads[0], grads[1], atol=0.0005, rtol=0.05,
            )
