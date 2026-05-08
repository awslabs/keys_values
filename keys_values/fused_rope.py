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
"""Fused Triton kernel for rotary position embedding (RoPE).

Replaces the eager apply_rope() sequence (slice + negate + cat + mul + mul +
add + to-dtype = ~5-6 kernels) with a single Triton kernel. Forward and
backward are both implemented as torch.autograd.Function, so the training
cell loop's saved_tensors_hooks see normal autograd functions and continue
to work unchanged.

Forward:
  y = x * cos + rot(x) * sin
where rot(x) = cat(-x[..., half:], x[..., :half], dim=-1).

Backward (see pos_encoding.py derivation):
  dL/dx_j for j < half = dL/dy_j * cos_j + dL/dy_{j+half} * sin_{j+half}
  dL/dx_j for j >= half = dL/dy_j * cos_j - dL/dy_{j-half} * sin_{j-half}
"""

from typing import Tuple

import torch

_triton_available = False
try:
    import triton
    import triton.language as tl

    _triton_available = True
except ImportError:
    pass


if _triton_available:

    @triton.jit
    def _fused_rope_fwd_kernel(
        X_ptr,  # [BH, T, D] element type = bf16/fp16/fp32
        Cos_ptr,  # [T, D]
        Sin_ptr,  # [T, D]
        Out_ptr,  # [BH, T, D]
        x_stride_bh,
        x_stride_t,
        x_stride_d,
        o_stride_bh,
        o_stride_t,
        o_stride_d,
        cs_stride_t,
        cs_stride_d,
        T,
        D: tl.constexpr,
        HALF: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        bh = tl.program_id(0)
        t_block = tl.program_id(1)

        t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        d_offsets = tl.arange(0, D)
        half_mask = d_offsets < HALF
        # rot(x)[j] = -x[j+half] for j<half, x[j-half] for j>=half
        rot_idx = tl.where(half_mask, d_offsets + HALF, d_offsets - HALF)
        rot_sign = tl.where(half_mask, -1.0, 1.0)

        # Load tile [BLOCK_T, D] of x
        x_offsets = (
            bh * x_stride_bh
            + t_offsets[:, None] * x_stride_t
            + d_offsets[None, :] * x_stride_d
        )
        x = tl.load(X_ptr + x_offsets, mask=t_mask[:, None], other=0.0)

        # Load [BLOCK_T, D] rotated positions of x (same rows, swapped halves)
        x_rot_offsets = (
            bh * x_stride_bh
            + t_offsets[:, None] * x_stride_t
            + rot_idx[None, :] * x_stride_d
        )
        x_rot = tl.load(X_ptr + x_rot_offsets, mask=t_mask[:, None], other=0.0)

        # Load cos, sin (shared across BH dimension)
        cs_offsets = t_offsets[:, None] * cs_stride_t + d_offsets[None, :] * cs_stride_d
        cos = tl.load(Cos_ptr + cs_offsets, mask=t_mask[:, None], other=0.0)
        sin = tl.load(Sin_ptr + cs_offsets, mask=t_mask[:, None], other=0.0)

        # Accumulate in fp32, write back in input dtype
        x_f = x.to(tl.float32)
        x_rot_f = x_rot.to(tl.float32) * rot_sign[None, :]
        cos_f = cos.to(tl.float32)
        sin_f = sin.to(tl.float32)
        out_f = x_f * cos_f + x_rot_f * sin_f
        out = out_f.to(x.dtype)

        o_offsets = (
            bh * o_stride_bh
            + t_offsets[:, None] * o_stride_t
            + d_offsets[None, :] * o_stride_d
        )
        tl.store(Out_ptr + o_offsets, out, mask=t_mask[:, None])

    @triton.jit
    def _fused_rope_bwd_kernel(
        GradOut_ptr,  # [BH, T, D]
        Cos_ptr,  # [T, D]
        Sin_ptr,  # [T, D]
        GradX_ptr,  # [BH, T, D]
        go_stride_bh,
        go_stride_t,
        go_stride_d,
        gx_stride_bh,
        gx_stride_t,
        gx_stride_d,
        cs_stride_t,
        cs_stride_d,
        T,
        D: tl.constexpr,
        HALF: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        bh = tl.program_id(0)
        t_block = tl.program_id(1)

        t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        d_offsets = tl.arange(0, D)
        half_mask = d_offsets < HALF
        # Backward: dx_j = dy_j*c_j + sign_j * dy_{swap(j)} * sin_{swap(j)}
        # where swap(j) = j+half for j<half, j-half otherwise
        # and sign_j = +1 for j<half, -1 otherwise
        swap_idx = tl.where(half_mask, d_offsets + HALF, d_offsets - HALF)
        sign = tl.where(half_mask, 1.0, -1.0)

        # Load dL/dy at [bh, t, d]
        go_row_base = bh * go_stride_bh + t_offsets[:, None] * go_stride_t
        go = tl.load(
            GradOut_ptr + go_row_base + d_offsets[None, :] * go_stride_d,
            mask=t_mask[:, None],
            other=0.0,
        )
        go_swap = tl.load(
            GradOut_ptr + go_row_base + swap_idx[None, :] * go_stride_d,
            mask=t_mask[:, None],
            other=0.0,
        )

        # Load cos[t, d] and sin[t, swap(d)]
        cs_row_base = t_offsets[:, None] * cs_stride_t
        cos = tl.load(
            Cos_ptr + cs_row_base + d_offsets[None, :] * cs_stride_d,
            mask=t_mask[:, None],
            other=0.0,
        )
        sin_swap = tl.load(
            Sin_ptr + cs_row_base + swap_idx[None, :] * cs_stride_d,
            mask=t_mask[:, None],
            other=0.0,
        )

        go_f = go.to(tl.float32)
        go_swap_f = go_swap.to(tl.float32)
        cos_f = cos.to(tl.float32)
        sin_swap_f = sin_swap.to(tl.float32)

        gx_f = go_f * cos_f + sign[None, :] * go_swap_f * sin_swap_f
        gx = gx_f.to(go.dtype)

        gx_offsets = (
            bh * gx_stride_bh
            + t_offsets[:, None] * gx_stride_t
            + d_offsets[None, :] * gx_stride_d
        )
        tl.store(GradX_ptr + gx_offsets, gx, mask=t_mask[:, None])


def can_use_fused_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    """Check if `fused_apply_rope` can handle this input."""
    if not _triton_available:
        return False
    if not x.is_cuda:
        return False
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if x.dim() < 2:
        return False
    D = x.shape[-1]
    if D % 2 != 0:
        return False
    if cos.shape != sin.shape:
        return False
    if cos.shape[-1] != D:
        return False
    T = x.shape[-2]
    if cos.numel() != T * D:
        return False
    return True


def _reshape_inputs(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, torch.Size]:
    """Reshape x to [BH, T, D] contiguous and cos/sin to [T, D] contiguous."""
    original_shape = x.shape
    T = x.shape[-2]
    D = x.shape[-1]
    BH = x.numel() // (T * D)
    x_contig = x.contiguous()
    x_view = x_contig.view(BH, T, D)
    cos_view = cos.reshape(T, D).contiguous()
    sin_view = sin.reshape(T, D).contiguous()
    return x_view, cos_view, sin_view, BH, T, D, original_shape


class _FusedRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        if not can_use_fused_rope(x, cos, sin):
            raise ValueError(
                "fused_apply_rope: inputs not compatible. "
                "Use can_use_fused_rope() to check first."
            )
        x_view, cos_view, sin_view, BH, T, D, original_shape = _reshape_inputs(
            x, cos, sin
        )

        out_flat = torch.empty_like(x_view)
        HALF = D // 2
        BLOCK_T = 32
        grid = (BH, triton.cdiv(T, BLOCK_T))

        _fused_rope_fwd_kernel[grid](
            x_view,
            cos_view,
            sin_view,
            out_flat,
            x_view.stride(0),
            x_view.stride(1),
            x_view.stride(2),
            out_flat.stride(0),
            out_flat.stride(1),
            out_flat.stride(2),
            cos_view.stride(0),
            cos_view.stride(1),
            T,
            D=D,
            HALF=HALF,
            BLOCK_T=BLOCK_T,
        )

        ctx.save_for_backward(cos_view, sin_view)
        ctx.BH = BH
        ctx.T = T
        ctx.D = D
        ctx.original_shape = original_shape

        return out_flat.view(original_shape)

    @staticmethod
    def backward(ctx, grad_out):
        cos, sin = ctx.saved_tensors
        BH, T, D = ctx.BH, ctx.T, ctx.D
        HALF = D // 2

        grad_out_contig = grad_out.contiguous()
        grad_out_view = grad_out_contig.view(BH, T, D)
        grad_x = torch.empty_like(grad_out_view)

        BLOCK_T = 32
        grid = (BH, triton.cdiv(T, BLOCK_T))

        _fused_rope_bwd_kernel[grid](
            grad_out_view,
            cos,
            sin,
            grad_x,
            grad_out_view.stride(0),
            grad_out_view.stride(1),
            grad_out_view.stride(2),
            grad_x.stride(0),
            grad_x.stride(1),
            grad_x.stride(2),
            cos.stride(0),
            cos.stride(1),
            T,
            D=D,
            HALF=HALF,
            BLOCK_T=BLOCK_T,
        )

        return grad_x.view(ctx.original_shape), None, None


def fused_apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Fused RoPE: drop-in replacement for litgpt.model.apply_rope.

    Args:
        x: Input tensor, any shape with last two dims (..., T, head_size)
        cos: Cached cosines, shape (1, T, head_size) or (T, head_size)
        sin: Cached sines, same shape as cos

    Returns:
        RoPE-transformed tensor, same shape and dtype as x.
    """
    return _FusedRope.apply(x, cos, sin)
