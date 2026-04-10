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
import math

import torch
import pytest

from keys_values.attention import (
    scaled_dot_product_attention_in_blocks,
    DefaultKeysAndValues,
)
from keys_values.flex_attention import (
    sdpa_flexatt_with_attn_weights,
    FlexAttentionArgs,
)
from keys_values.utils import repeat_interleave, index_to_3d


def pytorch_reference_weights(
    query,
    key,
    scale,
    input_pos,
    token_positions,
):
    """Ground truth: full matmul + softmax + causal mask + sum.

    Args:
        query: [batch, n_head, q_len, head_dim]
        key: [batch, n_kv_heads, kv_len, head_dim]
        scale: softmax scale factor
        input_pos: starting query position
        token_positions: [batch, n_kv_heads, kv_len] int positions
        n_kv_heads, group_size: GQA parameters

    Returns:
        W: [batch, n_kv_heads, kv_len] float32 attention weight sums
    """
    batch, n_head, q_len, hd = query.shape
    _, n_kv_heads, kv_len, _ = key.shape
    q_per_kv = n_head // n_kv_heads
    Q_4d = query.float()
    K_exp = repeat_interleave(key.float(), n_head)

    # Scores: [batch, n_head, q_len, kv_len]
    scores = torch.matmul(Q_4d, K_exp.transpose(-1, -2)) * scale

    # Causal mask
    q_positions = torch.arange(q_len, device=query.device) + input_pos  # [q_len]
    # token_positions: [batch, n_kv_heads, kv_len] -> expand to [batch, n_head, kv_len]
    tp_exp = repeat_interleave(token_positions, n_head)

    # mask[b, h, q, k] = True if kv_pos <= query_pos
    mask = tp_exp[:, :, None, :] <= q_positions[None, None, :, None]
    scores.masked_fill_(~mask, float("-inf"))

    # Softmax + sum
    weights = torch.softmax(scores, dim=-1)  # [B, H, Q, KV]
    # Sum over queries, then group heads -> [B, Hkv, KV]
    if q_per_kv > 1:
        weights = (
            weights.view(batch, n_kv_heads, q_per_kv, q_len, kv_len).sum(dim=(2, 3))
            / q_per_kv
        )
    else:
        weights = weights.sum(dim=2)
    return weights


def flexatt_weights(query, key, scale, input_pos, token_positions):
    flexatt_args = FlexAttentionArgs(
        forward_return_lse=True,
        extend_kv=False,
    )
    _, attn_weights = sdpa_flexatt_with_attn_weights(
        flexatt_args=flexatt_args,
        query=query,
        key=key,
        value=key,  # does not matter
        scale_factor=scale,
        attention_logit_softcapping=None,
        input_pos=input_pos,
        token_positions=token_positions,
    )
    return attn_weights


def eager_weights(query, key, scale, input_pos, token_positions):
    _, attn_weights = scaled_dot_product_attention_in_blocks(
        query=query,
        k_and_v=DefaultKeysAndValues(key, key),
        scale_factor=scale,
        return_attn_weights=True,
        input_pos=input_pos,
        token_positions=token_positions,
        sliding_window_size=None,
    )
    return attn_weights


@torch.inference_mode()
def test_small_comparison():
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    # Qwen3-4B config
    batch_size = 1
    q_len = 16
    n_head = 32
    n_kv_heads = 8
    hd = 128
    kv_len = 64
    input_pos = 48  # simulates non-prefill: query starts at pos 48
    scale = 1.0 / math.sqrt(hd)

    for tp_contiguous in (True, False):
        if tp_contiguous:
            print("\nTest for contiguous token_positions")
            # Contiguous token positions: 0..kv_len-1
            token_positions = index_to_3d(
                torch.arange(kv_len, device=device, dtype=torch.int32),
                batch_size,
                n_kv_heads,
            )
            kv_len2 = kv_len
        else:
            print("\nTest for non-contiguous token_positions")
            kept_positions = torch.tensor(
                [0, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 47] + list(range(48, 64)),
                device=device,
                dtype=torch.int32,
            )
            kv_len2 = len(kept_positions)
            token_positions = index_to_3d(
                kept_positions,
                batch_size,
                n_kv_heads,
            )

        kwargs = dict(device=device, dtype=dtype)
        query = torch.randn(batch_size, n_head, q_len, hd, **kwargs)
        key = torch.randn(batch_size, n_kv_heads, kv_len2, hd, **kwargs)

        # Different variants
        attn_weights = [
            func(query, key, scale, input_pos, token_positions)
            for func in (pytorch_reference_weights, eager_weights, flexatt_weights)
        ]

        # Compare attn weights
        print("Comparing summed attention weights:")
        for ind, name in ((1, "eager"), (2, "FlexAttention")):
            print(f"PyTorch reference vs {name}:")
            torch.testing.assert_close(attn_weights[0], attn_weights[ind])

        # Check eviction decisions: which top-K entries differ?
        keep_ratio = 0.5
        keep_k = int(kv_len2 * keep_ratio)
        print(f"Comparing eviction decisions: Keep {keep_k} of {kv_len2}:")
        topk_ref = attn_weights[0][0, 0, :].topk(keep_k).indices.sort().values
        for ind, name in ((1, "eager"), (2, "FlexAttention")):
            topk_comp = attn_weights[ind][0, 0, :].topk(keep_k).indices.sort().values
            num_match = sum(topk_comp == topk_ref).item()
            print(f"PyTorch reference vs {name}: {num_match} matches")
            assert num_match == keep_k


@pytest.mark.parametrize(
    "q_len, kv_len, input_pos, keep_ratio, description",
    [
        (16, 256, 240, 0.5, "kv_len=256, keep 50%"),
        (16, 512, 496, 0.5, "kv_len=512, keep 50%"),
        (16, 1024, 1008, 0.5, "kv_len=1024, keep 50%"),
        (16, 2048, 2032, 0.5, "kv_len=2048, keep 50%"),
        (16, 1024, 1008, 0.25, "kv_len=1024, keep 25% (aggressive eviction)"),
        (16, 1024, 1008, 0.75, "kv_len=1024, keep 75% (mild eviction)"),
    ],
)
@torch.inference_mode()
def test_larger_comparison(q_len, kv_len, input_pos, keep_ratio, description):
    torch.manual_seed(42 + kv_len)
    device = "cuda"
    dtype = torch.bfloat16

    # Qwen3-4B config
    n_head = 32
    n_kv_heads = 8
    hd = 128
    scale = 1.0 / math.sqrt(hd)
    batch_size = 1

    token_positions = index_to_3d(
        torch.arange(kv_len, device=device, dtype=torch.int32),
        batch_size,
        n_kv_heads,
    )
    kwargs = dict(device=device, dtype=dtype)
    query = torch.randn(batch_size, n_head, q_len, hd, **kwargs)
    key = torch.randn(batch_size, n_kv_heads, kv_len, hd, **kwargs)

    # Different variants
    attn_weights = [
        func(query, key, scale, input_pos, token_positions)
        for func in (pytorch_reference_weights, eager_weights, flexatt_weights)
    ]

    # Compare attn weights
    print(f"\n{description}\nComparing summed attention weights:")
    for ind, name in ((1, "eager"), (2, "FlexAttention")):
        print(f"PyTorch reference vs {name}:")
        torch.testing.assert_close(attn_weights[0], attn_weights[ind])

    # Check eviction decisions: which top-K entries differ?
    keep_k = int(kv_len * keep_ratio)
    print(f"Comparing eviction decisions: Keep {keep_k} of {kv_len}:")
    # Check all kv heads, not just head 0
    mismatches = [0, 0]
    for b in range(batch_size):
        for h in range(n_kv_heads):
            topk_ref = set(attn_weights[0][b, h].topk(keep_k).indices.tolist())
            for ind in range(1, 3):
                topk_comp = set(attn_weights[ind][b, h].topk(keep_k).indices.tolist())
                if topk_ref != topk_comp:
                    mismatches[ind - 1] += 1
    for name, num_mis in zip(("eager", "FlexAttention"), mismatches):
        print(f"PyTorch reference vs {name}: {num_mis} mismatches")
    assert sum(mismatches) == 0
