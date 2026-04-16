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

import logging
import math
from typing import Optional, Tuple

import torch

from keys_values.attention.sdpa_wrapper import (
    sdpa_check_args,
    reorder_key_value,
    reorder_inverse,
)

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
        Q_ptr,
        K_ptr,
        LSE_ptr,
        W_ptr,
        TP_ptr,
        total_q,
        kv_len,
        Q_stride_bh,
        Q_stride_q,
        Q_stride_d,
        K_stride_bh,
        K_stride_k,
        K_stride_d,
        LSE_stride_bh,
        LSE_stride_q,
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
          Q: [batch*n_kv_heads, q_len*group_size, head_size]  (fp16/bf16)
          K: [batch*n_kv_heads, kv_len, head_size]             (fp16/bf16)
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


# TODO:
# Rewrite `_score_sum_kernel` so that `token_positions`, `input_pos`
# not needed anymore.
def triton_score_sum(
    Q: torch.Tensor,
    K: torch.Tensor,
    LSE: torch.Tensor,
    scale: float,
    n_kv_heads: int,
    group_size: int,
) -> torch.Tensor:
    """Compute attention weight sums using Triton (no V needed).

    We use causal attention masking, where Q and K are aligned on the
    right. For now, this is done by creating `token_positions` and
    `input_pos` which work.

    Args:
        Q: [batch, q_len, n_head, head_size] (fp16/bf16)
        K: [batch, kv_len, n_kv_heads, head_size] (fp16/bf16)
        LSE: [batch, q_len, n_head] (fp32, log2 scale from FlashInfer)
        scale: softmax scale factor (1/sqrt(head_size))
        n_kv_heads: number of KV heads
        group_size: GQA group size (n_head // n_kv_heads)

    Returns:
        W: [batch, n_kv_heads, kv_len] (fp32) attention weight sums
    """
    batch_size, q_len, _, head_size = Q.shape
    _, kv_len, _, _ = K.shape

    # Reshape Q by KV head groups → contiguous [batch*n_kv_heads, q_len*group_size, head_size]
    Q_grouped = (
        Q.reshape(batch_size, q_len, n_kv_heads, group_size, head_size)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch_size * n_kv_heads, q_len * group_size, head_size)
        .contiguous()
    )

    # Reshape K → contiguous [batch*n_kv_heads, kv_len, head_size]
    K_flat = (
        K.permute(0, 2, 1, 3)
        .reshape(batch_size * n_kv_heads, kv_len, head_size)
        .contiguous()
    )

    # Reshape LSE → contiguous [batch*n_kv_heads, q_len*group_size]
    LSE_grouped = (
        LSE.reshape(batch_size, q_len, n_kv_heads, group_size)
        .permute(0, 2, 1, 3)
        .reshape(batch_size * n_kv_heads, q_len * group_size)
        .contiguous()
    )

    # Create `TP_flat`, `input_pos` to make default causal attention masking
    # work
    TP_flat = (
        torch.arange(
            kv_len,
            dtype=torch.int32,
            device=Q.device,
        )[None, :]
        .expand(batch_size * n_kv_heads, kv_len)
        .contiguous()
    )
    input_pos = kv_len - q_len

    total_q = q_len * group_size
    W = torch.zeros(
        batch_size * n_kv_heads, kv_len, device=Q.device, dtype=torch.float32
    )

    # Block sizes tuned for A100 + head_size=128
    BLOCK_KV = 128
    BLOCK_Q = 32
    NUM_WARPS = 4
    NUM_STAGES = 2
    if head_size <= 64:
        BLOCK_KV = 256
        BLOCK_Q = 64

    sm_scale_log2 = scale * 1.4426950408889634  # scale * log2(e)

    grid = (triton.cdiv(kv_len, BLOCK_KV), batch_size * n_kv_heads)
    _score_sum_kernel[grid](
        Q_grouped,
        K_flat,
        LSE_grouped,
        W,
        TP_flat,
        total_q,
        kv_len,
        Q_grouped.stride(0),
        Q_grouped.stride(1),
        Q_grouped.stride(2),
        K_flat.stride(0),
        K_flat.stride(1),
        K_flat.stride(2),
        LSE_grouped.stride(0),
        LSE_grouped.stride(1),
        W.stride(0),
        TP_flat.stride(0),
        sm_scale_log2,
        input_pos,
        BLOCK_KV=BLOCK_KV,
        BLOCK_Q=BLOCK_Q,
        HEAD_DIM=head_size,
        GROUP_SIZE=group_size,
        HAS_CAUSAL=True,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    # Mean over GQA group (divide by group_size) to match codebase convention
    W = W.reshape(batch_size, n_kv_heads, kv_len)
    if group_size > 1:
        W = W / group_size
    return W


ALLOWED_HEAD_SIZES = (64, 128, 256)


def pad_head_size(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[int]]:
    head_size = query.shape[-1]
    assert 0 < head_size <= ALLOWED_HEAD_SIZES[-1]
    assert key.shape[-1] == head_size
    assert value.shape[-1] == head_size
    diff = None
    for a, b in zip((0,) + ALLOWED_HEAD_SIZES[:-1], ALLOWED_HEAD_SIZES):
        if a < head_size < b:
            diff = b - head_size
            break
    if diff is not None:
        kwargs = dict(dtype=query.dtype, device=query.device)
        dims = (1, 1, 1, 1)
        query = torch.cat(
            (query, torch.zeros(dims, **kwargs).expand(*query.shape[:-1], diff)),
            dim=-1,
        )
        key = torch.cat(
            (key, torch.zeros(dims, **kwargs).expand(*key.shape[:-1], diff)),
            dim=-1,
        )
        value = torch.cat(
            (value, torch.zeros(dims, **kwargs).expand(*value.shape[:-1], diff)),
            dim=-1,
        )
    return query, key, value, diff


def can_do_flashinfer(
    head_size: int, dtype: torch.dtype, return_attn_weights: bool
) -> bool:
    return (
        torch.cuda.is_available()
        and dtype in (torch.float16, torch.bfloat16)
        and head_size <= ALLOWED_HEAD_SIZES[-1]
        and (not return_attn_weights or _triton_available)
    )


class FlashInferSDPA:
    """
    Wrapper for FlashInfer CUDA kernels.

    This class encapsulates FlashInfer's optimized attention kernels and provides
    a unified interface compatible with existing keys_values code.
    """

    def __init__(self):
        if not self._check_vendored_kernels_available():
            raise AssertionError(
                "FlashInfer kernels are not available. Installation (at repository root):\n"
                "$ pip install flashinfer-python\n"
                "$ python build_ext.py"
            )
        if not _triton_available:
            logger.warning(
                "Triton is not available. This means that "
                "scaled_dot_product_attention cannot be called with return_attn_weights=True."
            )

    def _check_vendored_kernels_available(self) -> bool:
        """
        Check if vendored FlashInfer kernels are available.

        Returns:
            True if vendored kernels are available and compatible, False otherwise
        """
        try:
            from keys_values.attention import flashinfer_ops

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

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_factor: Optional[float],
        input_pos: int,
        token_positions: Optional[torch.Tensor],
        return_attn_weights: bool = False,
        sort_if_3d: bool = True,
        output_transposed: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute SDPA using FlashInfer kernels.

        Args:
            query: Query tensor, shape `(batch_size, n_head, q_len, head_size)`
            key: Key tensor, shape `(batch_size, n_query_groups, kv_len, head_size)`
            value: Value tensor, shape `(batch_size, n_query_groups, kv_len, head_size)`
            scale_factor: Scale factor for attention scores
            input_pos: Position in input sequence, must be > 0. For square prefill,
                the native PyTorch SDPA is faster anyway
            token_positions: Contains token positions in KV cache, shape
                `(batch_size, n_query_groups, kv_len)`. If not given, it is
                equivalent to `arange(kv_len)`.
            return_attn_weights: Whether to return attention weights
            sort_if_3d: See :func:`reorder_key_value`.
            output_transposed: If `True`, dims 1 and 2 of `attn_outputs` are
                transposed (compared to `query`).

        Returns:
            Tuple `(attn_outputs, attn_weights)`, where `attn_outputs` has shape
            `(batch_size, n_head, q_len, head_size)` if
            `output_transposed == False, shape
            `(batch_size, q_len, n_head, head_size)` otherwise. `attn_weights`
            has shape `(batch_size, n_query_groups, kv_len)` and `dtype == float32`
            if `return_attn_weights == True`. Otherwise, `None` is returned.

        """
        if not isinstance(input_pos, int):
            raise ValueError("input_pos must be int scalar")
        if input_pos <= 0:
            raise ValueError(
                f"input_pos must be positive. Don't use for square prefill"
            )
        batch_size, n_head, n_query_groups, q_len, kv_len, head_size = sdpa_check_args(
            query,
            key,
            value,
        )
        if token_positions is not None and token_positions.shape != key.shape[:-1]:
            raise ValueError(
                f"token_positions.shape = {token_positions.shape}, key.shape = {key.shape}: Not compatible"
            )
        if scale_factor is None:
            scale_factor = 1.0 / math.sqrt(head_size)

        # Check if FlashInfer fast prefill can be used
        if return_attn_weights and not _triton_available:
            raise NotImplementedError("Triton is required for return_attn_weights=True")
        if not can_do_flashinfer(head_size, query.dtype, return_attn_weights):
            raise NotImplementedError(
                "FlashInfer SDPA needs these conditions:\n"
                f"- head_size <= {ALLOWED_HEAD_SIZES[-1]}, but is {head_size}\n"
                f"- query.dtype in (torch.float16, torch.bfloat16), but is {query.dtype}"
            )

        # Routing:
        # 1. q_len == 1 (single-token decode):
        #    Use optimized decode kernel (has efficient logits caching for
        #    attention weights)
        # 2. q_len > 1, return_attn_weights, Triton available:
        #    FlashInfer forward + Triton score-sum (no large intermediate,
        #    tensor cores)
        # 3. q_len > 1, return_attn_weights, FlashInfer eligible:
        #    Two phase approach:
        #    Phase 1: FlashInfer prefill for O + LSE (fast, no large intermediates)
        #    Phase 2: Compute weights from Q, K, LSE via chunked matmuls
        # 4. q_len > 1, no return_attn_weights:
        #    Use FlashInfer prefill kernel (fastest)
        use_decode_kernel = q_len == 1
        if use_decode_kernel and kv_len == 1:
            raise NotImplementedError("Don't use for q_len=1, kv_len=1")

        # Deal with `token_positions, input_pos` here, by reordering.
        # And pad final dimension of inputs with zeros so that head size
        # becomes value in :const:`ALLOWED_HEAD_SIZES`.
        if token_positions is not None and not use_decode_kernel:
            key, value, extra_info = reorder_key_value(
                key,
                value,
                token_positions.detach(),
                input_pos,
                q_len,
                sort_if_3d,
            )
        else:
            extra_info = dict()
        query, key, value, head_size_diff = pad_head_size(
            query,
            key,
            value,
        )

        if use_decode_kernel:
            # Single-token decode: Use optimized decode kernel
            # If `q_len == 1`, we don't need `token_positions, input_pos`,
            # since standard causal masking applies
            attn_outputs, attn_weights = self._flashinfer_sdpa_chunk_processing(
                query,
                key,
                value,
                scale_factor,
                return_attn_weights,
            )
        elif return_attn_weights:
            # Return attention weights, summed over query axis, as well
            # FlashInfer forward + Triton score-sum (no large
            # intermediate, tensor cores)
            attn_outputs, attn_weights = self._flashinfer_sdpa_fused_prefill(
                query,
                key,
                value,
                scale_factor,
            )
            if token_positions is not None:
                # Undo reordering
                attn_weights = reorder_inverse(attn_weights, **extra_info)
        else:
            # q_len > 1, return_attn_weights == False:
            # Use FlashInfer prefill kernel
            attn_outputs = self._flashinfer_sdpa_standard(
                query,
                key,
                value,
                scale_factor,
            )
            attn_weights = None

        if head_size_diff is not None:
            attn_outputs = attn_outputs[:, :, :, :head_size]
        if not output_transposed:
            attn_outputs = attn_outputs.transpose(1, 2).contiguous()
        return attn_outputs, attn_weights

    def _flashinfer_sdpa_standard(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        """
        Standard vendored kernel SDPA using the prefill kernel.

        Args:
            query: Query tensor, shape `(batch_size, n_head, q_len, head_size)`
            key: Key tensor, shape `(batch_size, n_query_groups, kv_len, head_size)`
            value: Value tensor, shape `(batch_size, n_query_groups, kv_len, head_size)`
            scale_factor: Scale factor for attention scores

        Returns:
            Attention outputs, shape `(batch_size, q_len, n_head, head_size)`

        """
        from keys_values.attention import flashinfer_ops

        # Transform tensors to vendored kernel format
        # Vendored kernel expects:
        # - query: [batch_size, q_len, num_qo_heads, head_size]
        # - key: [batch_size, kv_len, num_kv_heads, head_size]
        # - value: [batch_size, kv_len, num_kv_heads, head_size]

        query_transformed = query.transpose(1, 2).contiguous()
        key_transformed = key.transpose(1, 2).contiguous()
        value_transformed = value.transpose(1, 2).contiguous()

        # Call vendored prefill kernel
        output_transformed, _, _ = flashinfer_ops.sdpa_prefill(
            query=query_transformed,
            key=key_transformed,
            value=value_transformed,
            scale=scale_factor,
            return_weights=False,
        )

        return output_transformed

    def _flashinfer_sdpa_fused_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_factor: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        Returns:
            Tuple of (attention_output, attention_weights)

        """
        from keys_values.attention import flashinfer_ops

        n_head = query.shape[1]
        n_kv_heads = key.shape[1]
        group_size = n_head // n_kv_heads

        # Transform to kernel format: (batch, seq, heads, dim)
        q_t = query.transpose(1, 2).contiguous()  # [bs, q_len, n_head, head_size]
        k_t = key.transpose(1, 2).contiguous()  # [bs, kv_len, n_kv_heads, head_size]
        v_t = value.transpose(1, 2).contiguous()  # [bs, kv_len, n_kv_heads, head_size]

        # ================================================================
        # Call 1: Forward attention -> O + LSE
        #
        # Use causal=True with input_pos only (no token_positions).
        # The current chunk's K/V is in the cache, so causal masking is
        # needed. FlashInfer's token_positions path is ~100x slower, but
        # causal=True + input_pos uses the fast kernel and is correct for
        # contiguous positions [0..kv_len-1].
        # ================================================================
        output, _, lse = flashinfer_ops.sdpa_prefill(
            query=q_t,
            key=k_t,
            value=v_t,
            scale=scale_factor,
            return_weights=False,
            return_lse=True,
        )
        # output: [bs, q_len, n_head, head_size]
        # lse: [bs, q_len, n_head] (log2 scale)

        # ================================================================
        # Phase 2: Compute attention weight sums using Triton score-sum kernel
        #
        # W[kv_head, k] = Σ_q Σ_{h∈group} exp2(Q·K·scale·log2e - LSE_log2)
        #
        # This is like flash attention but WITHOUT V — only Q·K dot products
        # and weight accumulation. Uses tensor cores via tl.dot.
        # ================================================================
        weights = triton_score_sum(
            Q=q_t,  # [bs, q_len, n_head, head_size]
            K=k_t,  # [bs, kv_len, n_kv_heads, head_size]
            LSE=lse.float(),  # [bs, q_len, n_head] (log2 scale)
            scale=scale_factor,
            n_kv_heads=n_kv_heads,
            group_size=group_size,
        )

        return output, weights

    def _flashinfer_sdpa_chunk_processing(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_factor: float,
        return_attn_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Decode-kernel variant for single-token decode (`q_len == 1`).
        Uses the vendored sdpa_decode kernel for single-token attention.

        We use default causal attention masking here, where `query` and
        `key` are aligned on the right.

        Args:
            query: Query tensor, shape `(batch_size, n_head, 1, head_size)`
            key: Key tensor, shape
                `(batch_size, n_query_groups, kv_len, head_size)`
            value: Value tensor, shape
                `(batch_size, n_query_groups, kv_len, head_size)`
            scale_factor: Scale factor for attention scores
            return_attn_weights: Whether to return attention weights

        Returns:
            Tuple of (attention_output, attention_weights)
        """
        from keys_values.attention import flashinfer_ops

        q_len = query.shape[2]
        assert q_len == 1, f"Need q_len == 1, but got {q_len}"

        # Transform key and value to vendored kernel format
        # From (batch_size, n_query_groups, kv_len, head_size) to (batch_size, kv_len, n_query_groups, head_size)
        query = query.squeeze(2).contiguous()
        key_transformed = key.transpose(1, 2).contiguous()
        value_transformed = value.transpose(1, 2).contiguous()

        # Call vendored decode kernel
        # Expected query shape: [batch_size, num_qo_heads, head_size]
        output_token, attn_weights = flashinfer_ops.sdpa_decode(
            query=query,
            key=key_transformed,
            value=value_transformed,
            scale=scale_factor,
            return_weights=return_attn_weights,
        )
        attn_outputs = output_token.unsqueeze(1)

        return attn_outputs, attn_weights


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
