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

from litgpt.utils import _RunIf

from keys_values.attention.base import (
    scaled_dot_product_attention_in_blocks,
    DefaultKeysAndValues,
)
from keys_values.attention.flex_attention import (
    sdpa_flexatt_with_attn_weights,
    FlexAttentionArgs,
)
from keys_values.utils import repeat_interleave, index_to_3d

# Lazy singleton for FlashInfer wrapper
_flashinfer_sdpa = None
_flashinfer_checked = False


def _get_flashinfer_sdpa():
    """Lazily initialize and return the FlashInfer SDPA wrapper, or None."""
    global _flashinfer_sdpa, _flashinfer_checked
    if not _flashinfer_checked:
        _flashinfer_checked = True
        try:
            from keys_values.attention.flashinfer_wrapper import FlashInferSDPA

            _flashinfer_sdpa = FlashInferSDPA()
        except Exception:
            pass
    return _flashinfer_sdpa


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
    q_positions = torch.arange(
        input_pos, input_pos + q_len, device=query.device
    )  # [q_len]
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


def flashinfer_weights(query, key, scale, input_pos, token_positions):
    flashinfer = _get_flashinfer_sdpa()
    _, attn_weights = flashinfer.scaled_dot_product_attention(
        query=query,
        key=key,
        value=key,  # does not matter
        scale_factor=scale,
        input_pos=input_pos,
        token_positions=token_positions,
        return_attn_weights=True,
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


def get_variants(q_len: int):
    variants = (pytorch_reference_weights, eager_weights, flexatt_weights)
    names = ("PyTorch reference", "eager", "FlexAttention")
    # TODO: Currently, FlashInfer is skipped for q_len = 1, because there is
    # some bug. Fix it and remove this exclusion.
    if _get_flashinfer_sdpa() is not None and q_len > 1:
        variants += (flashinfer_weights,)
        names += ("FlashInfer",)
    return variants, names


@_RunIf(min_cuda_gpus=1)
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
    variants, names = get_variants(q_len)

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
            func(query, key, scale, input_pos, token_positions) for func in variants
        ]

        # Compare attn weights
        ref_name = names[0]
        print("Comparing summed attention weights:")
        for ind, name in enumerate(names):
            if ind > 0:
                print(f"{ref_name} vs {name}:")
                torch.testing.assert_close(
                    attn_weights[0],
                    attn_weights[ind],
                    atol=1e-4,
                    rtol=1e-4,
                )

        # Check eviction decisions: which top-K entries differ?
        keep_ratio = 0.5
        keep_k = int(kv_len2 * keep_ratio)
        print(f"Comparing eviction decisions: Keep {keep_k} of {kv_len2}:")
        topk_ref = attn_weights[0][0, 0, :].topk(keep_k).indices.sort().values
        for ind, name in enumerate(names):
            if ind > 0:
                topk_comp = (
                    attn_weights[ind][0, 0, :].topk(keep_k).indices.sort().values
                )
                num_match = sum(topk_comp == topk_ref).item()
                print(f"{ref_name} vs {name}: {num_match} matches")
                assert num_match == keep_k


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "q_len, kv_len, input_pos, keep_ratio, description",
    [
        (16, 256, 240, 0.5, "kv_len=256, keep 50%"),
        (16, 512, 496, 0.5, "kv_len=512, keep 50%"),
        (1, 256, 255, 0.5, "q_len=1, kv_len=256, keep 50%"),
        (1, 512, 511, 0.5, "q_len=1, kv_len=512, keep 50%"),
        (2, 256, 254, 0.5, "q_len=1, kv_len=256, keep 50%"),
        (2, 512, 510, 0.5, "q_len=1, kv_len=512, keep 50%"),
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
    variants, names = get_variants(q_len)

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
        func(query, key, scale, input_pos, token_positions) for func in variants
    ]

    # Compare attn weights
    ref_name = names[0]
    print(f"\n{description}\nComparing summed attention weights:")
    for ind, name in enumerate(names):
        if ind > 0:
            print(f"{ref_name} vs {name}:")
            torch.testing.assert_close(
                attn_weights[0],
                attn_weights[ind],
                atol=1e-4,
                rtol=1e-4,
            )

    # Check eviction decisions: which top-K entries differ?
    keep_k = int(kv_len * keep_ratio)
    print(f"Comparing eviction decisions: Keep {keep_k} of {kv_len}:")
    # Check all kv heads, not just head 0
    mismatches = [0, 0]
    for b in range(batch_size):
        for h in range(n_kv_heads):
            topk_ref = set(attn_weights[0][b, h].topk(keep_k).indices.tolist())
            for ind in range(1, len(names)):
                topk_comp = set(attn_weights[ind][b, h].topk(keep_k).indices.tolist())
                if topk_ref != topk_comp:
                    mismatches[ind - 1] += 1
    for name, num_mis in zip(names[1:], mismatches):
        print(f"{ref_name} vs {name}: {num_mis} mismatches")
    assert sum(mismatches) == 0
