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
Fused Triton kernels for attention weight computation.

This module provides a Triton kernel that computes accumulated attention weights
from Q, K, and LSE (log-sum-exp) values. The kernel fuses the Q·K^T matmul with
exp2 normalization and Q-axis reduction into a single kernel, avoiding
materialization of the full score matrix in global memory.
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl


def _get_autotune_configs():
    return [
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_Q": 128, "BLOCK_KV": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 64}, num_stages=3, num_warps=4),
    ]


@triton.autotune(configs=_get_autotune_configs(), key=["q_len", "kv_len", "HEAD_DIM", "q_per_kv"])
@triton.jit
def _attn_weights_from_lse_kernel(
    # Pointers
    Q_ptr,
    K_ptr,
    LSE_ptr,
    W_ptr,
    # Dimensions
    n_kv_heads,
    q_per_kv,
    q_len,
    kv_len,
    # Strides for Q [batch, n_head, q_len, head_dim]
    stride_q_batch,
    stride_q_head,
    stride_q_seq,
    stride_q_dim,
    # Strides for K [batch, n_kv_heads, kv_len, head_dim]
    stride_k_batch,
    stride_k_head,
    stride_k_seq,
    stride_k_dim,
    # Strides for LSE [batch, q_len, n_head]
    stride_lse_batch,
    stride_lse_seq,
    stride_lse_head,
    # Strides for W [batch, n_kv_heads, kv_len]
    stride_w_batch,
    stride_w_head,
    stride_w_seq,
    # Scalars
    sm_scale_log2,
    input_pos,
    sliding_window_size,
    # Compile-time constants
    HAS_SLIDING_WINDOW: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    # Decode program ID into (batch, kv_head, kv_tile)
    pid = tl.program_id(0)
    kv_tiles = tl.cdiv(kv_len, BLOCK_KV)
    batch_kv_head_idx = pid // kv_tiles
    kv_tile_idx = pid % kv_tiles
    kv_head_idx = batch_kv_head_idx % n_kv_heads
    batch_idx = batch_kv_head_idx // n_kv_heads

    # KV tile range
    kv_start = kv_tile_idx * BLOCK_KV
    kv_offs = kv_start + tl.arange(0, BLOCK_KV)
    kv_mask = kv_offs < kv_len

    # Head dim offsets
    d_offs = tl.arange(0, HEAD_DIM)

    # Load K tile transposed: [HEAD_DIM, BLOCK_KV] for tl.dot(Q, K_T)
    k_base = K_ptr + batch_idx * stride_k_batch + kv_head_idx * stride_k_head
    k_ptrs = k_base + d_offs[:, None] * stride_k_dim + kv_offs[None, :] * stride_k_seq
    k_tile = tl.load(k_ptrs, mask=kv_mask[None, :], other=0.0)  # [HEAD_DIM, BLOCK_KV]

    # Weight accumulator
    w_acc = tl.zeros([BLOCK_KV], dtype=tl.float32)

    # Iterate over GQA heads in this group
    for h in range(q_per_kv):
        head_idx = kv_head_idx * q_per_kv + h
        q_base = Q_ptr + batch_idx * stride_q_batch + head_idx * stride_q_head
        lse_base = LSE_ptr + batch_idx * stride_lse_batch + head_idx * stride_lse_head

        # Iterate over Q tiles
        q_tiles = tl.cdiv(q_len, BLOCK_Q)
        for q_tile_idx in range(0, q_tiles):
            q_start = q_tile_idx * BLOCK_Q
            q_offs = q_start + tl.arange(0, BLOCK_Q)
            q_mask = q_offs < q_len

            # Load Q tile: [BLOCK_Q, HEAD_DIM], pre-scale by sm_scale_log2
            q_ptrs = q_base + q_offs[:, None] * stride_q_seq + d_offs[None, :] * stride_q_dim
            q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
            q_tile = (q_tile * sm_scale_log2).to(q_tile.dtype)

            # TensorCore matmul: [BLOCK_Q, HEAD_DIM] @ [HEAD_DIM, BLOCK_KV] -> [BLOCK_Q, BLOCK_KV]
            scores = tl.dot(q_tile, k_tile)  # fp32 accumulation

            # Load LSE tile: [BLOCK_Q]
            lse_ptrs = lse_base + q_offs * stride_lse_seq
            lse_tile = tl.load(lse_ptrs, mask=q_mask, other=0.0)

            # Causal mask: kv_idx <= input_pos + q_idx
            q_positions = input_pos + q_offs  # [BLOCK_Q]
            causal_mask = kv_offs[None, :] <= q_positions[:, None]

            # Optional sliding window
            if HAS_SLIDING_WINDOW:
                window_mask = q_positions[:, None] - kv_offs[None, :] < sliding_window_size
                causal_mask = causal_mask & window_mask

            # Combined mask
            full_mask = causal_mask & q_mask[:, None] & kv_mask[None, :]
            scores = tl.where(full_mask, scores, float("-inf"))

            # Normalize: exp2(scores - LSE)
            weights = tl.exp2(scores - lse_tile[:, None])

            # Sum over Q dimension and accumulate
            w_acc += tl.sum(weights, axis=0)

    # Store result
    w_ptrs = W_ptr + batch_idx * stride_w_batch + kv_head_idx * stride_w_head + kv_offs * stride_w_seq
    tl.store(w_ptrs, w_acc, mask=kv_mask)


def compute_weights_from_lse_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    lse: torch.Tensor,
    scale_factor: float,
    input_pos: int,
    sliding_window_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute accumulated attention weights from Q, K, and LSE using a fused Triton kernel.

    This replaces the PyTorch-based weight computation with a single kernel that
    performs Q·K^T matmul, LSE subtraction, exp2, and Q-axis reduction without
    materializing the score matrix in global memory.

    Args:
        query: Query tensor, shape (batch_size, n_head, q_len, head_dim), fp16/bf16
        key: Key tensor, shape (batch_size, n_kv_heads, kv_len, head_dim), fp16/bf16
        lse: Log-sum-exp values from FlashInfer, shape (batch_size, q_len, n_head),
             float32, log base 2
        scale_factor: Attention scale factor (typically 1/sqrt(head_dim))
        input_pos: Starting position of the query in the full sequence
        sliding_window_size: Optional sliding window size for attention masking

    Returns:
        Accumulated attention weights, shape (batch_size, n_kv_heads, kv_len), float32
    """
    batch_size, n_head, q_len, head_dim = query.shape
    _, n_kv_heads, kv_len, _ = key.shape
    q_per_kv = n_head // n_kv_heads

    log2e = math.log2(math.e)
    sm_scale_log2 = scale_factor * log2e

    # Ensure contiguous tensors for correct stride computation
    query = query.contiguous()
    key = key.contiguous()
    lse = lse.contiguous()

    # Allocate output
    weights = torch.zeros(
        batch_size, n_kv_heads, kv_len, device=query.device, dtype=torch.float32
    )

    has_sliding_window = sliding_window_size is not None and sliding_window_size > 0
    sw_size = sliding_window_size if has_sliding_window else 0

    # Launch kernel
    grid = lambda meta: (batch_size * n_kv_heads * triton.cdiv(kv_len, meta["BLOCK_KV"]),)

    _attn_weights_from_lse_kernel[grid](
        query,
        key,
        lse,
        weights,
        n_kv_heads,
        q_per_kv,
        q_len,
        kv_len,
        # Q strides
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        # K strides
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        # LSE strides
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        # W strides
        weights.stride(0),
        weights.stride(1),
        weights.stride(2),
        # Scalars
        sm_scale_log2,
        input_pos,
        sw_size,
        # Compile-time constants
        HAS_SLIDING_WINDOW=has_sliding_window,
        HEAD_DIM=head_dim,
    )

    return weights
