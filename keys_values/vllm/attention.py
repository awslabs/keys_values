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
"""
Attention-weight extraction for H2O (task 3).

H2O scores each KV slot by the attention mass it receives, summed over the query
axis (and over the query heads sharing a KV head, for GQA). This module:

- defines that score with a pure-torch reference (the contract), and
- dispatches to keys_values' FlashInfer + Triton score-sum path when available
  (CUDA + fp16/bf16 + vendored kernels), matching the wrapper's
  ``(batch, n_query_groups, kv_len)`` float32 output.

The reference is GPU-free and unit-tested as the source of truth; the FlashInfer
branch is validated against it on GPU (see test/vllm/test_attn_weights.py). It is
deliberately scoped to the decode/update case (``q_len <= kv_len``, queries are
the last ``q_len`` positions of the sequence), which is when H2O scores during
generation.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


def reference_summed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch attention with summed attention weights (the H2O contract).

    Queries are assumed to be the last ``q_len`` positions of a length-``kv_len``
    causal sequence (the decode/update setting). Supports grouped-query
    attention: ``n_heads`` must be a multiple of ``n_kv_heads``.

    Args:
        query: ``(batch, n_heads, q_len, head_size)``.
        key:   ``(batch, n_kv_heads, kv_len, head_size)``.
        value: ``(batch, n_kv_heads, kv_len, head_size)``.
        scale: Softmax scale; defaults to ``1/sqrt(head_size)``.
        causal: Apply causal masking aligned to the right (queries are the most
            recent ``q_len`` tokens).

    Returns:
        ``(attn_output, summed_weights)`` where ``attn_output`` is
        ``(batch, n_heads, q_len, head_size)`` and ``summed_weights`` is
        ``(batch, n_kv_heads, kv_len)`` in float32 — the attention mass each KV
        slot receives, summed over the query axis and over the query heads in
        its group.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, value must be 4D")
    batch, n_heads, q_len, head_size = query.shape
    _, n_kv_heads, kv_len, _ = key.shape
    if key.shape[1] != value.shape[1] or key.shape[2] != value.shape[2]:
        raise ValueError("key and value must share n_kv_heads and kv_len")
    if n_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})"
        )
    if q_len > kv_len:
        raise ValueError(f"q_len ({q_len}) must be <= kv_len ({kv_len})")
    if scale is None:
        scale = 1.0 / math.sqrt(head_size)

    group = n_heads // n_kv_heads
    # (batch, n_kv_heads, group, q_len/kv_len, head_size)
    q = query.view(batch, n_kv_heads, group, q_len, head_size).float()
    k = key.unsqueeze(2).float()  # (batch, n_kv_heads, 1, kv_len, head_size)
    v = value.unsqueeze(2).float()

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (.., q_len, kv_len)

    if causal:
        # Query i (0..q_len-1) is absolute position kv_len - q_len + i and may
        # attend to kv positions j <= that.
        offset = kv_len - q_len
        q_idx = torch.arange(q_len, device=query.device).view(q_len, 1) + offset
        k_idx = torch.arange(kv_len, device=query.device).view(1, kv_len)
        mask = k_idx <= q_idx  # (q_len, kv_len)
        scores = scores.masked_fill(~mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)  # (batch, n_kv_heads, group, q_len, kv_len)
    out = torch.matmul(probs, v)  # (batch, n_kv_heads, group, q_len, head_size)
    out = out.reshape(batch, n_heads, q_len, head_size).to(query.dtype)

    # Sum over the query heads in each group and over the query axis.
    summed = probs.sum(dim=(2, 3)).to(torch.float32)  # (batch, n_kv_heads, kv_len)
    return out, summed


def flashinfer_weight_path_available(head_size: int, dtype: torch.dtype) -> bool:
    """Whether keys_values' FlashInfer weight-returning SDPA can run here."""
    try:
        from keys_values.attention.flashinfer_wrapper import (
            can_do_flashinfer,
            get_flashinfer_sdpa,
        )
    except Exception:  # noqa: BLE001 - import/availability is environment-dependent
        return False
    if not can_do_flashinfer(head_size, dtype, return_attn_weights=True):
        return False
    try:
        get_flashinfer_sdpa()
    except Exception:  # noqa: BLE001 - vendored kernels may not be built
        return False
    return True


def _flashinfer_summed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Route through keys_values' FlashInfer + Triton score-sum path.

    NOTE: iteration-1, validated on GPU against the reference. The wrapper
    targets the update case and rejects square prefill (``q_len == kv_len``);
    callers should fall back to the reference there.
    """
    from keys_values.attention.flashinfer_wrapper import get_flashinfer_sdpa

    batch, n_heads, q_len, head_size = query.shape
    kv_len = key.shape[2]
    n_kv_heads = key.shape[1]
    input_pos = kv_len - q_len
    if input_pos <= 0:
        raise NotImplementedError(
            "FlashInfer weight path needs q_len < kv_len (update case)."
        )
    token_positions = (
        torch.arange(kv_len, device=query.device)
        .view(1, 1, kv_len)
        .expand(batch, n_kv_heads, kv_len)
    )
    sdpa = get_flashinfer_sdpa()
    out, weights = sdpa.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        scale_factor=scale,
        input_pos=input_pos,
        token_positions=token_positions,
        return_attn_weights=True,
        output_transposed=False,
    )
    return out, weights


def summed_attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = True,
    prefer_flashinfer: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention output + per-KV-slot summed attention weights for H2O.

    Uses keys_values' FlashInfer/Triton path when available (and applicable),
    otherwise the pure-torch reference. Both return ``summed_weights`` of shape
    ``(batch, n_kv_heads, kv_len)`` in float32.
    """
    head_size = query.shape[-1]
    if (
        prefer_flashinfer
        and query.shape[2] < key.shape[2]  # update case only
        and flashinfer_weight_path_available(head_size, query.dtype)
    ):
        try:
            return _flashinfer_summed(query, key, value, scale)
        except NotImplementedError:
            pass
    return reference_summed_attention(query, key, value, scale, causal)
