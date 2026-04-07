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
FlashInfer wrapper module for optimized CUDA kernels.

This module provides a clean abstraction layer for FlashInfer's optimized
attention kernels with graceful fallback to eager implementations when
FlashInfer is not available or not applicable.
"""

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_triton_available = False
try:
    import triton
    import triton.language as tl
    _triton_available = True
except ImportError:
    pass


# ============================================================================
# Triton kernel: Score-sum without V (attention weight accumulation)
#
# Computes W[kv_head, k] = Σ_q Σ_{h∈group} softmax(Q·K·scale)[q,h,k]
#                         = Σ_q Σ_{h∈group} exp2(Q[q,h]·K[k]·scale·log2e - LSE_log2[q,h])
#
# This is like flash attention but WITHOUT reading V or writing O — only the
# Q·K dot products and weight accumulation. Saves ~40-50% bandwidth vs a full
# reverse attention call.
# ============================================================================
if _triton_available:
    @triton.jit
    def _score_sum_kernel(
        Q_ptr, K_ptr, LSE_ptr, W_ptr,
        TP_ptr,
        total_q, kv_len,
        Q_stride_bh, Q_stride_q, Q_stride_d,
        K_stride_bh, K_stride_k, K_stride_d,
        LSE_stride_bh, LSE_stride_q,
        W_stride_bh,
        TP_stride_bh,
        sm_scale_log2,
        input_pos,
        BLOCK_KV: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        HAS_CAUSAL: tl.constexpr,
    ):
        """Score-sum kernel: Q·K → exp2(score·scale·log2e - LSE) → sum over Q.

        Grid: (cdiv(kv_len, BLOCK_KV), batch_size * n_kv_heads)

        Inputs (all pre-reshaped for contiguous access):
          Q: [batch*n_kv_heads, q_len*group_size, head_dim]  (fp16/bf16)
          K: [batch*n_kv_heads, kv_len, head_dim]             (fp16/bf16)
          LSE: [batch*n_kv_heads, q_len*group_size]           (fp32, log2 scale)
          TP: [batch*n_kv_heads, kv_len]                      (int32, token positions)
          W: [batch*n_kv_heads, kv_len]                       (fp32, output)

        When HAS_CAUSAL=True, applies causal masking:
          query q (absolute pos = input_pos + q // GROUP_SIZE) only attends
          to KV entry k if token_positions[k] <= query_pos.
        """
        kv_block_id = tl.program_id(0)
        bh_id = tl.program_id(1)

        kv_start = kv_block_id * BLOCK_KV
        kv_offsets = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = kv_offsets < kv_len
        d_offsets = tl.arange(0, HEAD_DIM)

        # Accumulated weights for this K tile: [BLOCK_KV]
        w_acc = tl.zeros([BLOCK_KV], dtype=tl.float32)

        # Load K tile: [BLOCK_KV, HEAD_DIM] — stays in SRAM for all Q iterations
        k_base = K_ptr + bh_id * K_stride_bh
        k_ptrs = k_base + kv_offsets[:, None] * K_stride_k + d_offsets[None, :]
        k_tile = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)

        # Load token positions for this KV block (for causal masking)
        if HAS_CAUSAL:
            tp_ptrs = TP_ptr + bh_id * TP_stride_bh + kv_offsets
            tp_tile = tl.load(tp_ptrs, mask=kv_mask, other=2147483647).to(tl.int32)

        # Iterate over Q tiles
        q_base = Q_ptr + bh_id * Q_stride_bh
        lse_base = LSE_ptr + bh_id * LSE_stride_bh

        for q_start in range(0, total_q, BLOCK_Q):
            q_offsets = q_start + tl.arange(0, BLOCK_Q)
            q_mask = q_offsets < total_q

            # Load Q tile: [BLOCK_Q, HEAD_DIM]
            q_ptrs = q_base + q_offsets[:, None] * Q_stride_q + d_offsets[None, :]
            q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

            # Load LSE: [BLOCK_Q] (log2 scale)
            lse_ptrs = lse_base + q_offsets * LSE_stride_q
            lse_tile = tl.load(lse_ptrs, mask=q_mask, other=0.0).to(tl.float32)

            # Score = Q @ K^T: [BLOCK_Q, BLOCK_KV] via tensor cores
            scores = tl.dot(q_tile, tl.trans(k_tile)).to(tl.float32)

            # log2 space: score * scale * log2(e) - LSE_log2
            scores = scores * sm_scale_log2 - lse_tile[:, None]

            # exp2, mask invalid Q positions, sum over Q axis
            weights = tl.exp2(scores)
            weights = tl.where(q_mask[:, None], weights, 0.0)

            # Causal masking: zero out weights where kv_pos > query_pos
            if HAS_CAUSAL:
                q_pos = input_pos + (q_offsets // GROUP_SIZE)  # [BLOCK_Q]
                causal_ok = tp_tile[None, :] <= q_pos[:, None]  # [BLOCK_Q, BLOCK_KV]
                weights = tl.where(causal_ok, weights, 0.0)

            w_acc += tl.sum(weights, axis=0)  # [BLOCK_KV]

        # Write accumulated weights
        w_ptrs = W_ptr + bh_id * W_stride_bh + kv_offsets
        tl.store(w_ptrs, w_acc, mask=kv_mask)


def triton_score_sum(
    Q: torch.Tensor,
    K: torch.Tensor,
    LSE: torch.Tensor,
    scale: float,
    n_kv_heads: int,
    group_size: int,
    token_positions: Optional[torch.Tensor] = None,
    input_pos: int = 0,
) -> torch.Tensor:
    """Compute attention weight sums using Triton (no V needed).

    Args:
        Q: [batch, q_len, n_head, head_dim] (fp16/bf16)
        K: [batch, kv_len, n_kv_heads, head_dim] (fp16/bf16)
        LSE: [batch, q_len, n_head] (fp32, log2 scale from FlashInfer)
        scale: softmax scale factor (1/sqrt(head_dim))
        n_kv_heads: number of KV heads
        group_size: GQA group size (n_head // n_kv_heads)
        token_positions: [batch, n_kv_heads, kv_len] (int32) absolute sequence
            positions for each KV cache entry. Required for causal masking.
            If None, no causal masking is applied.
        input_pos: starting absolute position of the query chunk

    Returns:
        W: [batch, n_kv_heads, kv_len] (fp32) attention weight sums
    """
    batch, q_len, n_head, head_dim = Q.shape
    _, kv_len, _, _ = K.shape

    # Reshape Q by KV head groups → contiguous [batch*n_kv_heads, q_len*group_size, head_dim]
    Q_grouped = (
        Q.reshape(batch, q_len, n_kv_heads, group_size, head_dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * n_kv_heads, q_len * group_size, head_dim)
        .contiguous()
    )

    # Reshape K → contiguous [batch*n_kv_heads, kv_len, head_dim]
    K_flat = (
        K.permute(0, 2, 1, 3)
        .reshape(batch * n_kv_heads, kv_len, head_dim)
        .contiguous()
    )

    # Reshape LSE → contiguous [batch*n_kv_heads, q_len*group_size]
    LSE_grouped = (
        LSE.reshape(batch, q_len, n_kv_heads, group_size)
        .permute(0, 2, 1, 3)
        .reshape(batch * n_kv_heads, q_len * group_size)
        .contiguous()
    )

    # Reshape token_positions → contiguous [batch*n_kv_heads, kv_len]
    has_causal = token_positions is not None
    if has_causal:
        TP_flat = (
            token_positions.to(dtype=torch.int32)
            .reshape(batch * n_kv_heads, kv_len)
            .contiguous()
        )
    else:
        # Dummy tensor — not accessed when HAS_CAUSAL=False
        TP_flat = torch.empty(1, device=Q.device, dtype=torch.int32)

    total_q = q_len * group_size
    W = torch.zeros(batch * n_kv_heads, kv_len, device=Q.device, dtype=torch.float32)

    # Block sizes tuned for A100 + head_dim=128
    BLOCK_KV = 128
    BLOCK_Q = 32
    NUM_WARPS = 4
    NUM_STAGES = 2
    if head_dim <= 64:
        BLOCK_KV = 256
        BLOCK_Q = 64

    sm_scale_log2 = scale * 1.4426950408889634  # scale * log2(e)

    grid = (triton.cdiv(kv_len, BLOCK_KV), batch * n_kv_heads)
    _score_sum_kernel[grid](
        Q_grouped, K_flat, LSE_grouped, W,
        TP_flat,
        total_q, kv_len,
        Q_grouped.stride(0), Q_grouped.stride(1), Q_grouped.stride(2),
        K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
        LSE_grouped.stride(0), LSE_grouped.stride(1),
        W.stride(0),
        TP_flat.stride(0) if has_causal else 0,
        sm_scale_log2,
        input_pos,
        BLOCK_KV=BLOCK_KV,
        BLOCK_Q=BLOCK_Q,
        HEAD_DIM=head_dim,
        GROUP_SIZE=group_size,
        HAS_CAUSAL=has_causal,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    return W.reshape(batch, n_kv_heads, kv_len)


class FlashInferSDPA:
    """
    Wrapper for FlashInfer CUDA kernels with fallback support.
    
    This class encapsulates FlashInfer's optimized attention kernels and provides
    a unified interface compatible with existing keys_values code. It gracefully
    falls back to eager implementations when FlashInfer is not available or when
    a configuration is not supported.
    """

    def __init__(self, use_fused_prefill: bool = True):
        """Initialize wrapper and detect vendored kernel availability.

        Args:
            use_fused_prefill: If True (default), use the fused prefill kernel
                that accumulates attention weights during the tiling loop. If False,
                use the old two-phase approach (FlashInfer for O+LSE, then Q@K matmul).
                Set to False for A/B comparison.
        """
        self.available = self._check_vendored_kernels_available()
        self.use_fused_prefill = use_fused_prefill
        if self.available:
            logger.info("Vendored FlashInfer kernels are available and will be used for SDPA computation")
            if use_fused_prefill:
                logger.info("Using fused prefill kernel for attention weight accumulation")
        else:
            logger.debug("Vendored FlashInfer kernels are not available, will use eager SDPA implementation")

    def _check_vendored_kernels_available(self) -> bool:
        """
        Check if vendored FlashInfer kernels are available.
        
        Returns:
            True if vendored kernels are available and compatible, False otherwise
        """
        try:
            from keys_values import flashinfer_ops
            available = flashinfer_ops.is_available()
            if available:
                logger.debug("Vendored FlashInfer kernels loaded successfully")
            else:
                error = flashinfer_ops.get_load_error()
                logger.debug(f"Vendored FlashInfer kernels not available: {error}")
            return available
        except ImportError as e:
            logger.debug(f"Failed to import flashinfer_ops module: {e}")
            return False
        except Exception as e:
            logger.debug(f"Error checking vendored kernel availability: {e}")
            return False

    def _should_use_chunk_processing(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> bool:
        """
        Determine if decode-kernel processing should be used.

        Returns True only for single-token decode (q_len == 1, kv_len > 1).
        For non-square attention with q_len > 1 (e.g., chunked prefill),
        the prefill kernel is used instead.

        Args:
            query: Query tensor, shape (batch_size, n_head, q_len, head_size)
            key: Key tensor, shape (batch_size, n_query_groups, kv_len, head_size)

        Returns:
            True if decode-kernel processing should be used, False otherwise
        """
        q_len = query.shape[2]
        kv_len = key.shape[2]
        return q_len == 1 and kv_len > 1

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_factor: float,
        return_attn_weights: bool = False,
        token_positions: Optional[torch.Tensor] = None,
        input_pos: int = 0,
        sliding_window_size: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute SDPA using FlashInfer kernels with fallback.
        
        Args:
            query: Query tensor, shape (batch_size, n_head, q_len, head_size)
            key: Key tensor, shape (batch_size, n_query_groups, kv_len, head_size)
            value: Value tensor, shape (batch_size, n_query_groups, kv_len, head_size)
            scale_factor: Scale factor for attention scores
            return_attn_weights: Whether to return attention weights
            token_positions: Token positions in KV cache, shape (batch_size, n_query_groups, kv_len)
            input_pos: Position in input sequence
            sliding_window_size: Size of sliding window for attention masking
            chunk_size: Optional chunk size for processing long sequences. When provided
                and query length exceeds chunk_size, the query is split into chunks
                and processed sequentially to manage GPU memory.
        
        Returns:
            Tuple of (attention_output, attention_weights)
            - attention_output: shape (batch_size, n_head, q_len, head_size)
            - attention_weights: shape (batch_size, n_query_groups, kv_len) if return_attn_weights=True, else None
        """
        if not self.available:
            return self._fallback_sdpa(
                query, key, value, scale_factor, return_attn_weights,
                token_positions, input_pos, sliding_window_size, chunk_size
            )
        
        try:
            return self._flashinfer_sdpa(
                query, key, value, scale_factor, return_attn_weights,
                token_positions, input_pos, sliding_window_size, chunk_size
            )
        except Exception as e:
            logger.warning(f"FlashInfer SDPA failed with error: {e}. Falling back to eager implementation.")
            return self._fallback_sdpa(
                query, key, value, scale_factor, return_attn_weights,
                token_positions, input_pos, sliding_window_size, chunk_size
            )

    def _flashinfer_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_factor: float,
        return_attn_weights: bool = False,
        token_positions: Optional[torch.Tensor] = None,
        input_pos: int = 0,
        sliding_window_size: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Call vendored FlashInfer kernels.
        
        This method translates parameters from keys_values format to vendored kernel format
        and calls the appropriate kernels.
        
        Args:
            query: Query tensor, shape (batch_size, n_head, q_len, head_size)
            key: Key tensor, shape (batch_size, n_query_groups, kv_len, head_size)
            value: Value tensor, shape (batch_size, n_query_groups, kv_len, head_size)
            scale_factor: Scale factor for attention scores
            return_attn_weights: Whether to return attention weights
            token_positions: Token positions in KV cache
            input_pos: Position in input sequence
            sliding_window_size: Size of sliding window for attention masking
            chunk_size: Optional chunk size for processing long sequences
        
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        from keys_values import flashinfer_ops
        
        batch_size, n_head, q_len, head_size = query.shape
        _, n_query_groups, kv_len, _ = key.shape
        
        # Validate input shapes
        assert query.shape[0] == key.shape[0] == value.shape[0], "Batch size mismatch"
        assert query.shape[3] == key.shape[3] == value.shape[3], "Head size mismatch"
        assert key.shape[1] == value.shape[1], "Key and value must have same number of heads"
        assert n_head % n_query_groups == 0, "n_head must be divisible by n_query_groups"
        
        # Routing:
        # 1. q_len == 1 (single-token decode): use optimized decode kernel
        #    (has efficient logits caching for attention weights)
        # 2. q_len > 1 with return_attn_weights + FlashInfer eligible: two-phase approach
        #    Phase 1: FlashInfer prefill for O + LSE (fast, no large intermediates)
        #    Phase 2: Compute weights from Q, K, LSE via chunked matmuls
        # 3. q_len > 1 with return_attn_weights but FlashInfer not eligible: eager fallback
        # 4. q_len > 1 without weights: use FlashInfer prefill kernel (fastest)
        #    With chunk_size: chunk queries for memory management
        use_decode_kernel = q_len == 1 and kv_len > 1

        # Check if FlashInfer fast prefill can be used (no token_positions, supported dtype/head_dim)
        can_use_flashinfer_fast = (
            token_positions is None
            and query.dtype in (torch.float16, torch.bfloat16)
            and head_size in (64, 128, 256)
        )

        # Check if Triton score-sum kernel can be used for weight accumulation.
        # Requires: fp16/bf16 (for tensor-core tl.dot), supported head_dim,
        # and input_pos > 0 (at input_pos=0 it's a prefill where
        # token_positions is None, which we need for causal masking).
        input_pos_val = input_pos if isinstance(input_pos, int) else (
            input_pos[0].item() if hasattr(input_pos, 'item') or isinstance(input_pos, torch.Tensor) else 0
        )
        can_use_fused_prefill = (
            self.use_fused_prefill
            and _triton_available
            and query.dtype in (torch.float16, torch.bfloat16)
            and head_size in (64, 128, 256)
            and input_pos_val > 0
        )

        if use_decode_kernel:
            # Single-token decode: use optimized decode kernel
            return self._flashinfer_sdpa_chunk_processing(
                query, key, value, scale_factor, return_attn_weights,
                token_positions, input_pos, sliding_window_size
            )
        elif q_len > 1 and return_attn_weights and can_use_fused_prefill:
            # FlashInfer forward + Triton score-sum (no large intermediate, tensor cores)
            return self._flashinfer_sdpa_fused_prefill(
                query, key, value, scale_factor,
                token_positions, input_pos, sliding_window_size
            )
        elif q_len > 1 and return_attn_weights and can_use_flashinfer_fast:
            # Old two-phase: FlashInfer prefill for O+LSE, then compute weights from LSE
            return self._flashinfer_sdpa_two_phase_weights(
                query, key, value, scale_factor,
                token_positions, input_pos, sliding_window_size, chunk_size
            )
        elif q_len > 1 and return_attn_weights:
            # Fallback for cases FlashInfer can't handle (token_positions, float32, etc.)
            return self._fallback_sdpa(
                query, key, value, scale_factor, return_attn_weights,
                token_positions, input_pos, sliding_window_size, chunk_size
            )
        elif chunk_size is not None and q_len > chunk_size:
            # Chunk queries via prefill kernel for memory management (no weights)
            return self._flashinfer_sdpa_long_sequence_chunking(
                query, key, value, scale_factor, return_attn_weights,
                token_positions, input_pos, sliding_window_size, chunk_size
            )
        else:
            # Standard FlashInfer prefill kernel (no weights needed)
            return self._flashinfer_sdpa_standard(
                query, key, value, scale_factor, return_attn_weights,
                token_positions, input_pos, sliding_window_size
            )

    def _flashinfer_sdpa_standard(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_factor: float,
        return_attn_weights: bool = False,
        token_positions: Optional[torch.Tensor] = None,
        input_pos: int = 0,
        sliding_window_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Standard vendored kernel SDPA using the prefill kernel.

        Handles both square (q_len == kv_len) and non-square (q_len != kv_len)
        attention. For non-square cases where q_len < kv_len, the kernel uses
        input_pos to correctly offset causal masking.
        
        Args:
            query: Query tensor, shape (batch_size, n_head, q_len, head_size)
            key: Key tensor, shape (batch_size, n_query_groups, kv_len, head_size)
            value: Value tensor, shape (batch_size, n_query_groups, kv_len, head_size)
            scale_factor: Scale factor for attention scores
            return_attn_weights: Whether to return attention weights
            token_positions: Token positions in KV cache, shape (batch_size, n_query_groups, kv_len)
            input_pos: Position in input sequence
            sliding_window_size: Size of sliding window for attention masking
        
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        from keys_values import flashinfer_ops
        
        batch_size, n_head, q_len, head_size = query.shape
        _, n_query_groups, kv_len, _ = key.shape
        
        # Transform tensors to vendored kernel format
        # Vendored kernel expects:
        # - query: [batch_size, q_len, num_qo_heads, head_dim]
        # - key: [batch_size, kv_len, num_kv_heads, head_dim]
        # - value: [batch_size, kv_len, num_kv_heads, head_dim]
        
        # Transpose query from (batch_size, n_head, q_len, head_size) to (batch_size, q_len, n_head, head_size)
        query_transformed = query.transpose(1, 2).contiguous()
        
        # Transpose key and value from (batch_size, n_query_groups, kv_len, head_size) to (batch_size, kv_len, n_query_groups, head_size)
        key_transformed = key.transpose(1, 2).contiguous()
        value_transformed = value.transpose(1, 2).contiguous()
        
        # Prepare input_pos as tensor if it's an integer
        if isinstance(input_pos, int):
            input_pos_tensor = torch.tensor([input_pos] * batch_size, device=query.device, dtype=torch.int32)
        else:
            input_pos_tensor = input_pos.to(dtype=torch.int32) if input_pos.dtype != torch.int32 else input_pos

        # Prepare token_positions if provided
        # FlashInfer sdpa_prefill expects 2D: [batch_size, kv_len]
        # KV cache provides 3D: [batch_size, n_query_groups, kv_len]
        # Collapse by taking first head (positions are same across heads)
        token_positions_transformed = None
        if token_positions is not None:
            if token_positions.ndim == 3:
                token_positions_transformed = token_positions[:, 0, :].to(dtype=torch.int32).contiguous()
            else:
                token_positions_transformed = token_positions.to(dtype=torch.int32).contiguous()

        # Prepare sliding window size
        window_size = sliding_window_size if sliding_window_size is not None else -1

        # Call vendored prefill kernel
        output_transformed, weights_transformed, _ = flashinfer_ops.sdpa_prefill(
            query=query_transformed,
            key=key_transformed,
            value=value_transformed,
            scale=scale_factor,
            token_positions=token_positions_transformed,
            input_pos=input_pos_tensor,
            sliding_window_size=window_size,
            causal=True,
            return_weights=return_attn_weights,
        )
        
        # Transform output back to keys_values format
        # From (batch_size, q_len, n_head, head_size) to (batch_size, n_head, q_len, head_size)
        output = output_transformed.transpose(1, 2).contiguous()
        
        # Transform weights if returned
        # Vendored kernel returns: (batch_size, num_kv_heads, kv_len)
        # We need: (batch_size, n_query_groups, kv_len)
        # These should already match since num_kv_heads == n_query_groups
        weights = weights_transformed
        
        # Ensure weights are float32 if returned
        if weights is not None:
            weights = weights.float()
        
        return output, weights

    def _flashinfer_sdpa_fused_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_factor: float,
        token_positions: Optional[torch.Tensor] = None,
        input_pos: int = 0,
        sliding_window_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        FlashInfer forward + Triton score-sum for O + attention weights.

        Call 1: FlashInfer prefill(Q, K, V) -> O + LSE  (causal, flash speed)
        Call 2: Triton score-sum kernel computes
                W[kv_head, k] = Σ_q Σ_{h∈group} exp2(Q·K·scale·log2e - LSE_log2)
                with causal masking via token_positions.

        O(1) extra memory (no large intermediates), tensor-core dot products.
        Only works when input_pos > 0 (token_positions is available).

        Args:
            query: Query tensor, shape (batch_size, n_head, q_len, head_size)
            key: Key tensor, shape (batch_size, n_query_groups, kv_len, head_size)
            value: Value tensor, shape (batch_size, n_query_groups, kv_len, head_size)
            scale_factor: Scale factor for attention scores
            token_positions: Token positions in KV cache (not yet supported, must be None)
            input_pos: Position in input sequence (must be > 0)
            sliding_window_size: Size of sliding window for attention masking

        Returns:
            Tuple of (attention_output, attention_weights)
        """
        from keys_values import flashinfer_ops

        batch_size, n_head, q_len, head_size = query.shape
        _, n_kv_heads, kv_len, _ = key.shape
        group_size = n_head // n_kv_heads

        # Transform to kernel format: (batch, seq, heads, dim)
        q_t = query.transpose(1, 2).contiguous()    # [bs, q_len, n_head, head_size]
        k_t = key.transpose(1, 2).contiguous()      # [bs, kv_len, n_kv_heads, head_size]
        v_t = value.transpose(1, 2).contiguous()    # [bs, kv_len, n_kv_heads, head_size]

        if isinstance(input_pos, int):
            input_pos_tensor = torch.tensor(
                [input_pos] * batch_size, device=query.device, dtype=torch.int32
            )
        else:
            input_pos_tensor = input_pos.to(dtype=torch.int32) if input_pos.dtype != torch.int32 else input_pos

        window_size = sliding_window_size if sliding_window_size is not None else -1

        # ================================================================
        # Call 1: Forward attention -> O + LSE
        #
        # Use causal=True with input_pos only (no token_positions).
        # The current chunk's K/V is in the cache, so causal masking is
        # needed. FlashInfer's token_positions path is ~100x slower, but
        # causal=True + input_pos uses the fast kernel and is correct for
        # contiguous positions [0..kv_len-1].
        # ================================================================
        o_t, _, lse = flashinfer_ops.sdpa_prefill(
            query=q_t,
            key=k_t,
            value=v_t,
            scale=scale_factor,
            token_positions=None,
            input_pos=input_pos_tensor,
            sliding_window_size=window_size,
            causal=True,
            return_weights=False,
            return_lse=True,
        )
        # o_t: [bs, q_len, n_head, head_size]
        # lse: [bs, q_len, n_head] (log2 scale)

        output = o_t.transpose(1, 2).contiguous()  # [bs, n_head, q_len, head_size]

        # ================================================================
        # Phase 2: Compute attention weight sums using Triton score-sum kernel
        #
        # W[kv_head, k] = Σ_q Σ_{h∈group} exp2(Q·K·scale·log2e - LSE_log2)
        #
        # This is like flash attention but WITHOUT V — only Q·K dot products
        # and weight accumulation. Uses tensor cores via tl.dot.
        # ================================================================
        # Derive scalar input_pos for the Triton kernel
        input_pos_val = input_pos if isinstance(input_pos, int) else input_pos[0].item()

        weights = triton_score_sum(
            Q=q_t,           # [bs, q_len, n_head, head_size]
            K=k_t,           # [bs, kv_len, n_kv_heads, head_size]
            LSE=lse.float(), # [bs, q_len, n_head] (log2 scale)
            scale=scale_factor,
            n_kv_heads=n_kv_heads,
            group_size=group_size,
            token_positions=token_positions,  # [bs, n_kv_heads, kv_len]
            input_pos=input_pos_val,
        )



        return output, weights

    def _flashinfer_sdpa_two_phase_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_factor: float,
        token_positions: Optional[torch.Tensor] = None,
        input_pos: int = 0,
        sliding_window_size: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Two-phase attention: FlashInfer prefill for O+LSE, then weight accumulation.

        Phase 1: Run FlashInfer's fast prefill kernel to get output O and LSE
                 (log-sum-exp per query position). This is Flash Attention speed
                 and never materializes the full attention matrix.

        Phase 2: Compute accumulated attention weights using Q, K, and LSE.
                 Uses chunked matmuls to control memory usage. Never materializes
                 the full [batch, heads, q_len, kv_len] attention matrix.

        Args:
            query: Query tensor, shape (batch_size, n_head, q_len, head_size)
            key: Key tensor, shape (batch_size, n_query_groups, kv_len, head_size)
            value: Value tensor, shape (batch_size, n_query_groups, kv_len, head_size)
            scale_factor: Scale factor for attention scores
            token_positions: Token positions (must be None for this path)
            input_pos: Position in input sequence
            sliding_window_size: Size of sliding window for attention masking
            chunk_size: Optional chunk size for Phase 2 query chunking

        Returns:
            Tuple of (attention_output, attention_weights)
        """
        from keys_values import flashinfer_ops

        batch_size, n_head, q_len, head_size = query.shape
        _, n_query_groups, kv_len, _ = key.shape

        # Phase 1: FlashInfer prefill for output + LSE
        # Transform tensors to vendored kernel format
        query_transformed = query.transpose(1, 2).contiguous()
        key_transformed = key.transpose(1, 2).contiguous()
        value_transformed = value.transpose(1, 2).contiguous()

        if isinstance(input_pos, int):
            input_pos_tensor = torch.tensor(
                [input_pos] * batch_size, device=query.device, dtype=torch.int32
            )
        else:
            input_pos_tensor = input_pos.to(dtype=torch.int32) if input_pos.dtype != torch.int32 else input_pos

        window_size = sliding_window_size if sliding_window_size is not None else -1

        output_transformed, _, lse = flashinfer_ops.sdpa_prefill(
            query=query_transformed,
            key=key_transformed,
            value=value_transformed,
            scale=scale_factor,
            token_positions=None,
            input_pos=input_pos_tensor,
            sliding_window_size=window_size,
            causal=True,
            return_weights=False,
            return_lse=True,
        )

        # Transform output back
        output = output_transformed.transpose(1, 2).contiguous()

        # Phase 2: Compute accumulated weights from Q, K, LSE
        weights = self._compute_weights_from_lse(
            query, key, scale_factor, lse, input_pos,
            sliding_window_size, chunk_size
        )

        return output, weights

    def _compute_weights_from_lse(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        scale_factor: float,
        lse: torch.Tensor,
        input_pos: int,
        sliding_window_size: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute accumulated attention weights from LSE values.

        For each KV position k, computes:
            W[b, kv_head, k] = Σ_{q, h∈group} exp2(Q[b,h,q]·K[b,kv_head,k] × scale × log2(e) − LSE[b,q,h])

        Uses a fused Triton kernel when available (computes Q·K^T, exp2, and sum
        in SRAM without materializing the score matrix). Falls back to PyTorch ops
        when Triton is not available.

        Args:
            query: Query tensor, shape (batch_size, n_head, q_len, head_size)
            key: Key tensor, shape (batch_size, n_query_groups, kv_len, head_size)
            scale_factor: Scale factor for attention scores
            lse: Log-sum-exp values from FlashInfer, shape (batch_size, q_len, n_head), log base 2
            input_pos: Starting position of the query in the full sequence
            sliding_window_size: Size of sliding window for attention masking
            chunk_size: Chunk size for query dimension. If None, computed automatically.
                        Only used by the PyTorch fallback path.

        Returns:
            Accumulated attention weights, shape (batch_size, n_query_groups, kv_len), dtype float32
        """
        # Use fused Triton kernel when available (avoids materializing score matrix)
        if _triton_available and query.is_cuda:
            try:
                from keys_values.triton_kernels import compute_weights_from_lse_triton
                return compute_weights_from_lse_triton(
                    query, key, lse, scale_factor, input_pos, sliding_window_size
                )
            except ImportError:
                pass  # Fall through to PyTorch implementation

        # Fallback: PyTorch implementation (materializes score matrix in chunks)
        import math

        batch_size, n_head, q_len, head_size = query.shape
        _, n_kv_heads, kv_len, _ = key.shape
        q_per_kv = n_head // n_kv_heads

        log2e = math.log2(math.e)
        sm_scale_log2 = scale_factor * log2e

        # Determine chunk size for query dimension
        # Memory per chunk: batch * n_kv_heads * q_per_kv * chunk_q * kv_len * 6 bytes
        # (fp16 bmm output = 2 bytes, fp32 after cast = 4 bytes)
        if chunk_size is not None:
            chunk_q = chunk_size
        else:
            target_bytes = 2 * 1024 * 1024 * 1024  # 2 GB
            bytes_per_q = batch_size * n_kv_heads * q_per_kv * kv_len * 6
            chunk_q = max(1, int(target_bytes / bytes_per_q)) if bytes_per_q > 0 else q_len
            chunk_q = min(chunk_q, q_len)

        # Pre-scale Q in native dtype (fp16/bf16) for TensorCore matmul
        # Reshape for GQA: [batch, n_head, q_len, D] -> [batch, n_kv_heads, q_per_kv, q_len, D]
        query_scaled = (query * sm_scale_log2).view(
            batch_size, n_kv_heads, q_per_kv, q_len, head_size
        )

        # Reshape K for bmm (no GQA expansion needed):
        # [batch, n_kv_heads, kv_len, D] -> [batch * n_kv_heads, D, kv_len]
        key_T = key.reshape(batch_size * n_kv_heads, kv_len, head_size).transpose(-2, -1)

        weights = torch.zeros(
            batch_size, n_kv_heads, kv_len,
            device=query.device, dtype=torch.float32
        )

        kv_indices = torch.arange(kv_len, device=query.device)

        for q_start in range(0, q_len, chunk_q):
            q_end = min(q_start + chunk_q, q_len)
            actual_chunk = q_end - q_start

            # Q chunk: [batch, n_kv_heads, q_per_kv, chunk_q, D]
            q_chunk = query_scaled[:, :, :, q_start:q_end, :]

            # Flatten for bmm: [batch * n_kv_heads, q_per_kv * chunk_q, D]
            q_flat = q_chunk.reshape(batch_size * n_kv_heads, q_per_kv * actual_chunk, head_size)

            # fp16/bf16 TensorCore matmul: [batch * n_kv_heads, q_per_kv * chunk_q, kv_len]
            scores_native = torch.bmm(q_flat, key_T)

            # Cast to fp32 for precision-sensitive exp2 subtraction
            # Reshape: [batch, n_kv_heads, q_per_kv, chunk_q, kv_len]
            scores = scores_native.float().view(
                batch_size, n_kv_heads, q_per_kv, actual_chunk, kv_len
            )

            # Apply causal mask: kv_idx <= input_pos + q_idx
            q_positions = torch.arange(q_start, q_end, device=query.device)
            causal_mask = kv_indices[None, :] <= (input_pos + q_positions[:, None])

            if sliding_window_size is not None and sliding_window_size > 0:
                window_mask = (input_pos + q_positions[:, None]) - kv_indices[None, :] < sliding_window_size
                causal_mask = causal_mask & window_mask

            # Broadcast: [1, 1, 1, chunk_q, kv_len]
            causal_mask = causal_mask[None, None, None, :, :]
            scores.masked_fill_(~causal_mask, float('-inf'))

            # LSE chunk: [batch, chunk_q, n_head] -> [batch, n_kv_heads, q_per_kv, chunk_q, 1]
            lse_chunk = lse[:, q_start:q_end, :]
            lse_expanded = lse_chunk.view(
                batch_size, actual_chunk, n_kv_heads, q_per_kv
            ).permute(0, 2, 3, 1).unsqueeze(-1)

            # Normalized weights: exp2(scores - LSE)
            norm_weights = torch.exp2(scores - lse_expanded)

            # Sum over q_per_kv (dim=2) and chunk_q (dim=3): [batch, n_kv_heads, kv_len]
            weights += norm_weights.sum(dim=(2, 3))

        return weights

    def _flashinfer_sdpa_chunk_processing(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_factor: float,
        return_attn_weights: bool = False,
        token_positions: Optional[torch.Tensor] = None,
        input_pos: int = 0,
        sliding_window_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Decode-kernel variant for single-token decode (q_len == 1).

        Uses the vendored sdpa_decode kernel for single-token attention.
        For multi-token non-square attention (q_len > 1, q_len < kv_len),
        use _flashinfer_sdpa_standard or _flashinfer_sdpa_long_sequence_chunking instead.
        
        Args:
            query: Query tensor, shape (batch_size, n_head, q_len, head_size)
            key: Key tensor, shape (batch_size, n_query_groups, kv_len, head_size)
            value: Value tensor, shape (batch_size, n_query_groups, kv_len, head_size)
            scale_factor: Scale factor for attention scores
            return_attn_weights: Whether to return attention weights
            token_positions: Token positions in KV cache, shape (batch_size, n_query_groups, kv_len)
            input_pos: Position in input sequence
            sliding_window_size: Size of sliding window for attention masking
        
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        from keys_values import flashinfer_ops
        
        batch_size, n_head, q_len, head_size = query.shape
        _, n_query_groups, kv_len, _ = key.shape
        
        # Transform key and value to vendored kernel format
        # From (batch_size, n_query_groups, kv_len, head_size) to (batch_size, kv_len, n_query_groups, head_size)
        key_transformed = key.transpose(1, 2).contiguous()
        value_transformed = value.transpose(1, 2).contiguous()

        # Prepare token_positions if provided
        # FlashInfer sdpa_prefill expects 2D: [batch_size, kv_len]
        # KV cache provides 3D: [batch_size, n_query_groups, kv_len]
        token_positions_transformed = None
        if token_positions is not None:
            if token_positions.ndim == 3:
                token_positions_transformed = token_positions[:, 0, :].to(dtype=torch.int32).contiguous()
            else:
                token_positions_transformed = token_positions.to(dtype=torch.int32).contiguous()
        
        # Prepare sliding window size
        window_size = sliding_window_size if sliding_window_size is not None else -1
        
        # Process each query token separately
        output_list = []
        weights_list = []
        
        for q_idx in range(q_len):
            # Get single query token: (batch_size, n_head, head_size)
            query_token = query[:, :, q_idx, :].contiguous()
            
            # Prepare input_pos for this query token
            current_pos = input_pos + q_idx
            if isinstance(current_pos, int):
                input_pos_tensor = torch.tensor([current_pos] * batch_size, device=query.device, dtype=torch.int32)
            else:
                input_pos_tensor = current_pos
            
            # Call vendored decode kernel
            # Expected query shape: [batch_size, num_qo_heads, head_dim]
            output_token, weights_token = flashinfer_ops.sdpa_decode(
                query=query_token,
                key=key_transformed,
                value=value_transformed,
                scale=scale_factor,
                token_positions=token_positions_transformed,
                input_pos=input_pos_tensor,
                sliding_window_size=window_size,
                causal=True,
                return_weights=return_attn_weights,
            )
            
            # output_token shape: (batch_size, n_head, head_size)
            # Add q_len dimension back
            output_list.append(output_token.unsqueeze(2))
            
            if return_attn_weights and weights_token is not None:
                # weights_token shape: (batch_size, n_query_groups, kv_len)
                weights_list.append(weights_token)
        
        # Concatenate outputs along q_len dimension
        # From list of (batch_size, n_head, 1, head_size) to (batch_size, n_head, q_len, head_size)
        output = torch.cat(output_list, dim=2)
        
        # Accumulate weights if requested
        weights = None
        if return_attn_weights and weights_list:
            # Sum weights across all query tokens
            # Each weights_token: (batch_size, n_query_groups, kv_len)
            # Result: (batch_size, n_query_groups, kv_len)
            weights = torch.stack(weights_list, dim=0).sum(dim=0)
            
            # Ensure weights are float32
            weights = weights.float()
        
        return output, weights

    def _flashinfer_sdpa_long_sequence_chunking(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_factor: float,
        return_attn_weights: bool = False,
        token_positions: Optional[torch.Tensor] = None,
        input_pos: int = 0,
        sliding_window_size: Optional[int] = None,
        chunk_size: int = 1024,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process long query sequences in chunks for memory management.
        
        This method splits a long query sequence into smaller chunks and processes
        each chunk sequentially using the vendored prefill kernel. This enables
        processing of sequences that would otherwise exceed GPU memory limits.
        
        Each chunk correctly applies causal masking based on its position in the
        original sequence, ensuring that queries only attend to appropriate key
        positions.
        
        Args:
            query: Query tensor, shape (batch_size, n_head, q_len, head_size)
            key: Key tensor, shape (batch_size, n_query_groups, kv_len, head_size)
            value: Value tensor, shape (batch_size, n_query_groups, kv_len, head_size)
            scale_factor: Scale factor for attention scores
            return_attn_weights: Whether to return attention weights
            token_positions: Token positions in KV cache, shape (batch_size, n_query_groups, kv_len)
            input_pos: Position in input sequence
            sliding_window_size: Size of sliding window for attention masking
            chunk_size: Size of each chunk for processing
        
        Returns:
            Tuple of (attention_output, attention_weights)
            - attention_output: shape (batch_size, n_head, q_len, head_size)
            - attention_weights: shape (batch_size, n_query_groups, kv_len) if return_attn_weights=True
              The weights are accumulated (summed) across all chunks.
        """
        from keys_values import flashinfer_ops
        
        batch_size, n_head, q_len, head_size = query.shape
        _, n_query_groups, kv_len, _ = key.shape
        
        # Calculate number of chunks needed
        num_chunks = (q_len + chunk_size - 1) // chunk_size
        
        # Transform key and value to vendored kernel format once (shared across chunks)
        # From (batch_size, n_query_groups, kv_len, head_size) to (batch_size, kv_len, n_query_groups, head_size)
        key_transformed = key.transpose(1, 2).contiguous()
        value_transformed = value.transpose(1, 2).contiguous()
        
        # Prepare token_positions if provided
        token_positions_transformed = None
        if token_positions is not None:
            # Ensure int32 dtype as required by the kernel
            token_positions_transformed = token_positions.to(dtype=torch.int32).contiguous()
        
        # Prepare sliding window size
        window_size = sliding_window_size if sliding_window_size is not None else -1
        
        # Process each chunk
        output_chunks = []
        weights_chunks = []
        
        for chunk_idx in range(num_chunks):
            # Calculate chunk boundaries
            chunk_start = chunk_idx * chunk_size
            chunk_end = min((chunk_idx + 1) * chunk_size, q_len)
            current_chunk_size = chunk_end - chunk_start
            
            # Extract query chunk
            # From (batch_size, n_head, q_len, head_size) to (batch_size, n_head, chunk_size, head_size)
            query_chunk = query[:, :, chunk_start:chunk_end, :].contiguous()
            
            # Transform query chunk to vendored kernel format
            # From (batch_size, n_head, chunk_size, head_size) to (batch_size, chunk_size, n_head, head_size)
            query_chunk_transformed = query_chunk.transpose(1, 2).contiguous()
            
            # Calculate input_pos for this chunk
            # Each chunk's queries have positions starting at input_pos + chunk_start
            chunk_input_pos = input_pos + chunk_start
            input_pos_tensor = torch.tensor(
                [chunk_input_pos] * batch_size, 
                device=query.device, 
                dtype=torch.int32
            )
            
            # Call vendored prefill kernel for this chunk
            output_chunk_transformed, weights_chunk, _ = flashinfer_ops.sdpa_prefill(
                query=query_chunk_transformed,
                key=key_transformed,
                value=value_transformed,
                scale=scale_factor,
                token_positions=token_positions_transformed,
                input_pos=input_pos_tensor,
                sliding_window_size=window_size,
                causal=True,
                return_weights=return_attn_weights,
            )
            
            # Transform output chunk back to keys_values format
            # From (batch_size, chunk_size, n_head, head_size) to (batch_size, n_head, chunk_size, head_size)
            output_chunk = output_chunk_transformed.transpose(1, 2).contiguous()
            output_chunks.append(output_chunk)
            
            # Collect weights if requested
            if return_attn_weights and weights_chunk is not None:
                weights_chunks.append(weights_chunk)
        
        # Concatenate output chunks along the query length dimension
        # Result: (batch_size, n_head, q_len, head_size)
        output = torch.cat(output_chunks, dim=2)
        
        # Accumulate weights across chunks if requested
        weights = None
        if return_attn_weights and weights_chunks:
            # Sum weights across all chunks
            # Each weights_chunk: (batch_size, n_query_groups, kv_len)
            # Result: (batch_size, n_query_groups, kv_len)
            weights = torch.stack(weights_chunks, dim=0).sum(dim=0)
            
            # Ensure weights are float32
            weights = weights.float()
        
        return output, weights

    def _fallback_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_factor: float,
        return_attn_weights: bool = False,
        token_positions: Optional[torch.Tensor] = None,
        input_pos: int = 0,
        sliding_window_size: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fall back to eager implementation using attention_utils.
        
        This method uses the existing eager SDPA implementations from attention_utils
        to compute attention when FlashInfer is not available or not applicable.
        
        Args:
            query: Query tensor, shape (batch_size, n_head, q_len, head_size)
            key: Key tensor, shape (batch_size, n_query_groups, kv_len, head_size)
            value: Value tensor, shape (batch_size, n_query_groups, kv_len, head_size)
            scale_factor: Scale factor for attention scores
            return_attn_weights: Whether to return attention weights
            token_positions: Token positions in KV cache
            input_pos: Position in input sequence
            sliding_window_size: Size of sliding window for attention masking
            chunk_size: Optional chunk size for processing long sequences. If provided,
                overrides the automatic chunking based on memory limits.
        
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        from keys_values.attention_utils import (
            attention_compute_scores,
            attention_compute_weighted_values,
            sdpa_attention_weights,
            create_temp_array,
            slice_as_flat,
        )
        
        batch_size, n_head, q_len, head_size = query.shape
        _, n_query_groups, kv_len, _ = key.shape
        
        # Determine chunking parameters
        # If chunk_size is provided, use it; otherwise use automatic memory-based chunking
        if chunk_size is not None and q_len > chunk_size:
            # Use provided chunk_size
            tmp_len = chunk_size
            num_splits = (q_len + chunk_size - 1) // chunk_size
            # Create temp array with the specified chunk size
            shape = (batch_size, n_head, tmp_len, kv_len)
            tmp_array = torch.empty(shape, device=query.device, dtype=torch.float32)
        else:
            # Use automatic memory-based chunking
            tmp_array, num_splits, tmp_len = create_temp_array(
                batch_size=batch_size,
                n_head=n_head,
                q_len=q_len,
                kv_len=kv_len,
                device=query.device,
            )
        
        # Convert tmp_array to match query dtype for intermediate computations
        tmp_array = tmp_array.to(dtype=query.dtype)
        
        # Compute attention weights using eager implementation
        if return_attn_weights:
            # Compute attention weights in chunks if necessary
            attn_weights_list = []
            attn_output_list = []
            
            for split_idx in range(num_splits):
                start_idx = split_idx * tmp_len
                end_idx = min((split_idx + 1) * tmp_len, q_len)
                chunk_len = end_idx - start_idx
                
                # Get query chunk
                query_chunk = query[:, :, start_idx:end_idx, :]
                
                # Compute attention weights for this chunk
                tmp_array_chunk = slice_as_flat(tmp_array, chunk_len)
                attn_weights_chunk = sdpa_attention_weights(
                    query=query_chunk,
                    key=key,
                    tmp_array=tmp_array_chunk,
                    token_positions=token_positions,
                    input_pos=input_pos + start_idx,
                    scale_factor=scale_factor,
                    sliding_window_size=sliding_window_size,
                )
                attn_weights_list.append(attn_weights_chunk)
                
                # Compute attention output for this chunk
                attn_output_chunk = attention_compute_weighted_values(
                    scores=attn_weights_chunk,
                    value=value,
                )
                attn_output_list.append(attn_output_chunk)
            
            # Concatenate chunks
            attn_output = torch.cat(attn_output_list, dim=2)
            attn_weights = torch.cat(attn_weights_list, dim=2)
            
            # Sum attention weights over query axis to get (batch_size, n_query_groups, kv_len)
            # attn_weights shape: (batch_size, n_head, q_len, kv_len)
            # We need to sum over query axis (dim=2) and aggregate heads to n_query_groups
            
            if n_head > n_query_groups:
                # Handle GQA: n_head > n_query_groups
                q_per_kv = n_head // n_query_groups
                # Reshape to (batch_size, n_query_groups, q_per_kv, q_len, kv_len)
                attn_weights = attn_weights.view(
                    batch_size, n_query_groups, q_per_kv, q_len, kv_len
                )
                # Sum over query axis (dim=3) and query heads (dim=2)
                # Result: (batch_size, n_query_groups, kv_len)
                attn_weights = attn_weights.sum(dim=(2, 3))
            else:
                # No GQA: n_head == n_query_groups
                # Sum over query axis: (batch_size, n_head, q_len, kv_len) -> (batch_size, n_head, kv_len)
                attn_weights = attn_weights.sum(dim=2)
            
            # Convert to float32 for numerical stability
            attn_weights = attn_weights.float()
            
            return attn_output, attn_weights
        else:
            # Compute attention without weights
            attn_output_list = []
            
            for split_idx in range(num_splits):
                start_idx = split_idx * tmp_len
                end_idx = min((split_idx + 1) * tmp_len, q_len)
                chunk_len = end_idx - start_idx
                
                # Get query chunk
                query_chunk = query[:, :, start_idx:end_idx, :]
                
                # Compute attention weights for this chunk
                tmp_array_chunk = slice_as_flat(tmp_array, chunk_len)
                attn_weights_chunk = sdpa_attention_weights(
                    query=query_chunk,
                    key=key,
                    tmp_array=tmp_array_chunk,
                    token_positions=token_positions,
                    input_pos=input_pos + start_idx,
                    scale_factor=scale_factor,
                    sliding_window_size=sliding_window_size,
                )
                
                # Compute attention output for this chunk
                attn_output_chunk = attention_compute_weighted_values(
                    scores=attn_weights_chunk,
                    value=value,
                )
                attn_output_list.append(attn_output_chunk)
            
            # Concatenate chunks
            attn_output = torch.cat(attn_output_list, dim=2)
            
            return attn_output, None


# Global instance of FlashInferSDPA wrapper
_flashinfer_sdpa_instance: Optional[FlashInferSDPA] = None


def get_flashinfer_sdpa() -> FlashInferSDPA:
    """
    Get the global FlashInferSDPA instance.
    
    Returns:
        FlashInferSDPA instance
    """
    global _flashinfer_sdpa_instance
    if _flashinfer_sdpa_instance is None:
        _flashinfer_sdpa_instance = FlashInferSDPA()
    return _flashinfer_sdpa_instance


# =============================================================================
# Backend Equivalence Verification Utilities
# =============================================================================

class BackendEquivalenceResult:
    """
    Result of backend equivalence verification.
    
    This class holds the results of comparing vendored kernel outputs
    against eager implementation outputs.
    """
    
    def __init__(
        self,
        is_equivalent: bool,
        output_max_diff: float,
        output_mean_diff: float,
        weights_max_diff: Optional[float] = None,
        weights_mean_diff: Optional[float] = None,
        rtol: float = 1e-4,
        atol: float = 1e-6,
        message: str = "",
    ):
        """
        Initialize equivalence result.
        
        Args:
            is_equivalent: Whether outputs are numerically equivalent
            output_max_diff: Maximum absolute difference in attention outputs
            output_mean_diff: Mean absolute difference in attention outputs
            weights_max_diff: Maximum absolute difference in attention weights (if computed)
            weights_mean_diff: Mean absolute difference in attention weights (if computed)
            rtol: Relative tolerance used for comparison
            atol: Absolute tolerance used for comparison
            message: Human-readable message describing the result
        """
        self.is_equivalent = is_equivalent
        self.output_max_diff = output_max_diff
        self.output_mean_diff = output_mean_diff
        self.weights_max_diff = weights_max_diff
        self.weights_mean_diff = weights_mean_diff
        self.rtol = rtol
        self.atol = atol
        self.message = message
    
    def __repr__(self) -> str:
        weights_max_str = f"{self.weights_max_diff:.2e}" if self.weights_max_diff is not None else "None"
        weights_mean_str = f"{self.weights_mean_diff:.2e}" if self.weights_mean_diff is not None else "None"
        return (
            f"BackendEquivalenceResult("
            f"is_equivalent={self.is_equivalent}, "
            f"output_max_diff={self.output_max_diff:.2e}, "
            f"output_mean_diff={self.output_mean_diff:.2e}, "
            f"weights_max_diff={weights_max_str}, "
            f"weights_mean_diff={weights_mean_str})"
        )
    
    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.is_equivalent


def check_numerical_equivalence(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> Tuple[bool, float, float]:
    """
    Check if two tensors are numerically equivalent within tolerance.
    
    Args:
        tensor_a: First tensor
        tensor_b: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        Tuple of (is_equivalent, max_diff, mean_diff)
    """
    if tensor_a.shape != tensor_b.shape:
        raise ValueError(
            f"Shape mismatch: {tensor_a.shape} vs {tensor_b.shape}"
        )
    
    # Compute differences
    diff = torch.abs(tensor_a.float() - tensor_b.float())
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Check equivalence using torch.allclose logic
    is_equivalent = torch.allclose(
        tensor_a.float(), tensor_b.float(), rtol=rtol, atol=atol
    )
    
    return is_equivalent, max_diff, mean_diff


def verify_backend_equivalence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: float,
    return_attn_weights: bool = False,
    token_positions: Optional[torch.Tensor] = None,
    input_pos: int = 0,
    sliding_window_size: Optional[int] = None,
    chunk_size: Optional[int] = None,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    log_results: bool = True,
) -> BackendEquivalenceResult:
    """
    Verify that vendored kernels produce equivalent results to eager implementation.
    
    This function computes SDPA using both the vendored FlashInfer kernels and
    the eager fallback implementation, then compares the results to verify
    numerical equivalence within the specified tolerance.
    
    Args:
        query: Query tensor, shape (batch_size, n_head, q_len, head_size)
        key: Key tensor, shape (batch_size, n_query_groups, kv_len, head_size)
        value: Value tensor, shape (batch_size, n_query_groups, kv_len, head_size)
        scale_factor: Scale factor for attention scores
        return_attn_weights: Whether to compare attention weights
        token_positions: Token positions in KV cache
        input_pos: Position in input sequence
        sliding_window_size: Size of sliding window for attention masking
        chunk_size: Optional chunk size for processing long sequences
        rtol: Relative tolerance for numerical comparison
        atol: Absolute tolerance for numerical comparison
        log_results: Whether to log verification results
    
    Returns:
        BackendEquivalenceResult containing comparison metrics and equivalence status
    
    Raises:
        RuntimeError: If vendored kernels are not available
    
    Example:
        >>> wrapper = FlashInferSDPA()
        >>> query = torch.randn(2, 4, 8, 64, device='cuda')
        >>> key = torch.randn(2, 2, 16, 64, device='cuda')
        >>> value = torch.randn(2, 2, 16, 64, device='cuda')
        >>> result = verify_backend_equivalence(
        ...     query, key, value, scale_factor=0.125,
        ...     return_attn_weights=True
        ... )
        >>> if result.is_equivalent:
        ...     print("Backends produce equivalent results")
        ... else:
        ...     print(f"Backends differ: {result.message}")
    """
    wrapper = get_flashinfer_sdpa()
    
    if not wrapper.available:
        raise RuntimeError(
            "Cannot verify backend equivalence: vendored kernels are not available. "
            "Ensure CUDA is available and the extension is compiled."
        )
    
    # Compute using vendored kernels
    try:
        vendored_output, vendored_weights = wrapper._flashinfer_sdpa(
            query, key, value, scale_factor, return_attn_weights,
            token_positions, input_pos, sliding_window_size, chunk_size
        )
    except Exception as e:
        message = f"Vendored kernel computation failed: {e}"
        if log_results:
            logger.error(message)
        return BackendEquivalenceResult(
            is_equivalent=False,
            output_max_diff=float('inf'),
            output_mean_diff=float('inf'),
            rtol=rtol,
            atol=atol,
            message=message,
        )
    
    # Compute using eager fallback
    eager_output, eager_weights = wrapper._fallback_sdpa(
        query, key, value, scale_factor, return_attn_weights,
        token_positions, input_pos, sliding_window_size, chunk_size
    )
    
    # Compare outputs
    output_equivalent, output_max_diff, output_mean_diff = check_numerical_equivalence(
        vendored_output, eager_output, rtol=rtol, atol=atol
    )
    
    # Compare weights if requested
    weights_max_diff = None
    weights_mean_diff = None
    weights_equivalent = True
    
    if return_attn_weights and vendored_weights is not None and eager_weights is not None:
        weights_equivalent, weights_max_diff, weights_mean_diff = check_numerical_equivalence(
            vendored_weights, eager_weights, rtol=rtol, atol=atol
        )
    
    # Determine overall equivalence
    is_equivalent = output_equivalent and weights_equivalent
    
    # Build message
    if is_equivalent:
        message = (
            f"Backend equivalence verified: "
            f"output_max_diff={output_max_diff:.2e}, "
            f"output_mean_diff={output_mean_diff:.2e}"
        )
        if weights_max_diff is not None:
            message += f", weights_max_diff={weights_max_diff:.2e}"
    else:
        message = "Backend equivalence FAILED: "
        if not output_equivalent:
            message += (
                f"output differs (max_diff={output_max_diff:.2e}, "
                f"mean_diff={output_mean_diff:.2e}, rtol={rtol}, atol={atol})"
            )
        if not weights_equivalent:
            message += (
                f"weights differ (max_diff={weights_max_diff:.2e}, "
                f"mean_diff={weights_mean_diff:.2e})"
            )
    
    # Log results
    if log_results:
        if is_equivalent:
            logger.debug(message)
        else:
            logger.warning(message)
    
    return BackendEquivalenceResult(
        is_equivalent=is_equivalent,
        output_max_diff=output_max_diff,
        output_mean_diff=output_mean_diff,
        weights_max_diff=weights_max_diff,
        weights_mean_diff=weights_mean_diff,
        rtol=rtol,
        atol=atol,
        message=message,
    )


def verify_backend_equivalence_batch(
    test_cases: list,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    log_results: bool = True,
    stop_on_failure: bool = False,
) -> Tuple[int, int, list]:
    """
    Verify backend equivalence for multiple test cases.
    
    Args:
        test_cases: List of dictionaries containing test parameters.
            Each dictionary should have keys: query, key, value, scale_factor,
            and optionally: return_attn_weights, token_positions, input_pos,
            sliding_window_size, chunk_size
        rtol: Relative tolerance for numerical comparison
        atol: Absolute tolerance for numerical comparison
        log_results: Whether to log verification results
        stop_on_failure: Whether to stop on first failure
    
    Returns:
        Tuple of (passed_count, failed_count, results_list)
    
    Example:
        >>> test_cases = [
        ...     {
        ...         'query': torch.randn(2, 4, 8, 64, device='cuda'),
        ...         'key': torch.randn(2, 2, 16, 64, device='cuda'),
        ...         'value': torch.randn(2, 2, 16, 64, device='cuda'),
        ...         'scale_factor': 0.125,
        ...         'return_attn_weights': True,
        ...     },
        ...     # ... more test cases
        ... ]
        >>> passed, failed, results = verify_backend_equivalence_batch(test_cases)
        >>> print(f"Passed: {passed}, Failed: {failed}")
    """
    passed_count = 0
    failed_count = 0
    results = []
    
    for i, test_case in enumerate(test_cases):
        try:
            result = verify_backend_equivalence(
                query=test_case['query'],
                key=test_case['key'],
                value=test_case['value'],
                scale_factor=test_case['scale_factor'],
                return_attn_weights=test_case.get('return_attn_weights', False),
                token_positions=test_case.get('token_positions'),
                input_pos=test_case.get('input_pos', 0),
                sliding_window_size=test_case.get('sliding_window_size'),
                chunk_size=test_case.get('chunk_size'),
                rtol=rtol,
                atol=atol,
                log_results=log_results,
            )
            
            results.append(result)
            
            if result.is_equivalent:
                passed_count += 1
            else:
                failed_count += 1
                if stop_on_failure:
                    if log_results:
                        logger.warning(f"Stopping at test case {i} due to failure")
                    break
                    
        except Exception as e:
            failed_count += 1
            error_result = BackendEquivalenceResult(
                is_equivalent=False,
                output_max_diff=float('inf'),
                output_mean_diff=float('inf'),
                rtol=rtol,
                atol=atol,
                message=f"Test case {i} raised exception: {e}",
            )
            results.append(error_result)
            
            if log_results:
                logger.error(f"Test case {i} failed with exception: {e}")
            
            if stop_on_failure:
                break
    
    if log_results:
        logger.info(
            f"Backend equivalence verification complete: "
            f"{passed_count} passed, {failed_count} failed out of {len(test_cases)} test cases"
        )
    
    return passed_count, failed_count, results
