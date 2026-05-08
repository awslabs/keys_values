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
"""Fused Triton kernel for RMSNorm.

Replaces the eager RMSNorm forward (x.float() + x*x + mean + rsqrt + x*rsqrt
+ mul weight + cast = ~5-6 kernels) with a single Triton kernel. Similarly
for backward. Implemented as torch.autograd.Function so the training cell
loop's saved_tensors_hooks see a normal autograd function.

Forward:
  y = (x / sqrt(mean(x**2) + eps)) * weight        (add_unit_offset=False)
  y = (x / sqrt(mean(x**2) + eps)) * (1 + weight)  (add_unit_offset=True)

Backward (per row, with w' = weight or 1+weight):
  r = rsqrt(mean_sq + eps)
  dL/dw = sum_over_batch( dL/dy * x * r )   (in fp32, reduced across all rows)
  dL/dx = r * w' * dL/dy - (r**3 / D) * (sum_j(dL/dy_j * w'_j * x_j)) * x
where D is the norm dim size (last dim).
"""

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
    def _fused_rmsnorm_fwd_kernel(
        X_ptr,  # [M, D]
        W_ptr,  # [D]
        Out_ptr,  # [M, D]
        Rsqrt_ptr,  # [M]  — saved for backward (fp32)
        x_stride_m,
        x_stride_d,
        o_stride_m,
        o_stride_d,
        eps,
        add_unit_offset: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        x = tl.load(
            X_ptr + row * x_stride_m + d_offsets * x_stride_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)

        # mean of x^2 and rsqrt
        mean_sq = tl.sum(x * x, axis=0) / D
        r = 1.0 / tl.sqrt(mean_sq + eps)

        # load weight, apply add_unit_offset
        w = tl.load(W_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        if add_unit_offset:
            w = w + 1.0

        y = x * r * w

        tl.store(
            Out_ptr + row * o_stride_m + d_offsets * o_stride_d,
            y.to(X_ptr.dtype.element_ty),
            mask=d_mask,
        )
        # Save rsqrt per row for backward
        tl.store(Rsqrt_ptr + row, r)

    @triton.jit
    def _fused_rmsnorm_bwd_dx_kernel(
        X_ptr,  # [M, D]
        W_ptr,  # [D]
        GradOut_ptr,  # [M, D]
        Rsqrt_ptr,  # [M]
        GradX_ptr,  # [M, D]
        x_stride_m,
        x_stride_d,
        go_stride_m,
        go_stride_d,
        gx_stride_m,
        gx_stride_d,
        add_unit_offset: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        x = tl.load(
            X_ptr + row * x_stride_m + d_offsets * x_stride_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        go = tl.load(
            GradOut_ptr + row * go_stride_m + d_offsets * go_stride_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(W_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        if add_unit_offset:
            w = w + 1.0
        r = tl.load(Rsqrt_ptr + row)  # fp32

        # wgo = w' * dL/dy
        wgo = w * go
        # sum_j ( x_j * w'_j * dL/dy_j )
        dot = tl.sum(x * wgo, axis=0)

        # dL/dx = r * (wgo - (r^2 / D) * dot * x)
        gx = r * (wgo - (r * r / D) * dot * x)

        tl.store(
            GradX_ptr + row * gx_stride_m + d_offsets * gx_stride_d,
            gx.to(GradOut_ptr.dtype.element_ty),
            mask=d_mask,
        )

    @triton.jit
    def _fused_rmsnorm_bwd_dw_partial_kernel(
        X_ptr,  # [M, D]
        GradOut_ptr,  # [M, D]
        Rsqrt_ptr,  # [M]
        Partial_ptr,  # [n_m_blocks, D]  (fp32 partial sums)
        x_stride_m,
        x_stride_d,
        go_stride_m,
        go_stride_d,
        p_stride_mb,
        p_stride_d,
        M,
        M_PER_PROGRAM: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Split-K partial-sum pass.

        Grid: (n_col_blocks, n_m_blocks). Each program handles a
        (M_PER_PROGRAM, BLOCK_D) tile of the (M, D) input, summing along M
        and producing a partial dw row of length BLOCK_D. These partials are
        stacked in Partial_ptr[mb, d] for the subsequent reduction pass.
        """
        col_block = tl.program_id(0)
        m_block = tl.program_id(1)

        d_offsets = col_block * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        m_start_global = m_block * M_PER_PROGRAM
        gw = tl.zeros((BLOCK_D,), dtype=tl.float32)

        # Inner loop over small BLOCK_M tiles within this program's slab
        for m_sub in range(0, M_PER_PROGRAM, BLOCK_M):
            m_off = m_start_global + m_sub + tl.arange(0, BLOCK_M)
            m_mask = m_off < M

            x_tile = tl.load(
                X_ptr + m_off[:, None] * x_stride_m + d_offsets[None, :] * x_stride_d,
                mask=m_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            go_tile = tl.load(
                GradOut_ptr
                + m_off[:, None] * go_stride_m
                + d_offsets[None, :] * go_stride_d,
                mask=m_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            r_tile = tl.load(Rsqrt_ptr + m_off, mask=m_mask, other=0.0)

            contrib = go_tile * x_tile * r_tile[:, None]
            gw += tl.sum(contrib, axis=0)

        tl.store(
            Partial_ptr + m_block * p_stride_mb + d_offsets * p_stride_d,
            gw,
            mask=d_mask,
        )

    @triton.jit
    def _fused_rmsnorm_bwd_dw_reduce_kernel(
        Partial_ptr,  # [n_m_blocks, D]
        GradW_ptr,  # [D]
        p_stride_mb,
        p_stride_d,
        N_MB: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Reduce the n_m_blocks partial sums column-wise into dW[D]."""
        col_block = tl.program_id(0)
        d_offsets = col_block * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D

        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for mb in range(0, N_MB):
            p = tl.load(
                Partial_ptr + mb * p_stride_mb + d_offsets * p_stride_d,
                mask=d_mask,
                other=0.0,
            )
            acc += p
        tl.store(GradW_ptr + d_offsets, acc, mask=d_mask)


def can_use_fused_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    dim: int,
) -> bool:
    """Check if `fused_rmsnorm` can handle this input."""
    if not _triton_available:
        return False
    if not x.is_cuda:
        return False
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if x.dim() < 2:
        return False
    # Only support reducing along the last dim (the common case)
    if dim != -1 and dim != x.dim() - 1:
        return False
    D = x.shape[-1]
    if weight.numel() != D:
        return False
    # Triton block size must fit the hidden dim; cap at 16384 to stay within
    # shared memory budgets
    if D > 16384:
        return False
    return True


def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


class _FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, add_unit_offset):
        if not can_use_fused_rmsnorm(x, weight, -1):
            raise ValueError(
                "fused_rmsnorm: inputs not compatible. "
                "Use can_use_fused_rmsnorm() to check first."
            )

        original_shape = x.shape
        D = x.shape[-1]
        M = x.numel() // D
        x_contig = x.contiguous()
        x_view = x_contig.view(M, D)
        w_contig = weight.contiguous()

        out = torch.empty_like(x_view)
        rsqrt = torch.empty(M, device=x.device, dtype=torch.float32)

        BLOCK_D = _next_power_of_two(D)

        _fused_rmsnorm_fwd_kernel[(M,)](
            x_view,
            w_contig,
            out,
            rsqrt,
            x_view.stride(0),
            x_view.stride(1),
            out.stride(0),
            out.stride(1),
            eps,
            add_unit_offset=bool(add_unit_offset),
            D=D,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(x_view, w_contig, rsqrt)
        ctx.add_unit_offset = bool(add_unit_offset)
        ctx.original_shape = original_shape
        ctx.M = M
        ctx.D = D

        return out.view(original_shape)

    @staticmethod
    def backward(ctx, grad_out):
        x_view, weight, rsqrt = ctx.saved_tensors
        M, D = ctx.M, ctx.D
        add_unit_offset = ctx.add_unit_offset

        grad_out_contig = grad_out.contiguous()
        grad_out_view = grad_out_contig.view(M, D)

        grad_x = torch.empty_like(grad_out_view)
        grad_w_fp32 = torch.empty(D, device=x_view.device, dtype=torch.float32)

        BLOCK_D = _next_power_of_two(D)

        # dL/dx
        _fused_rmsnorm_bwd_dx_kernel[(M,)](
            x_view,
            weight,
            grad_out_view,
            rsqrt,
            grad_x,
            x_view.stride(0),
            x_view.stride(1),
            grad_out_view.stride(0),
            grad_out_view.stride(1),
            grad_x.stride(0),
            grad_x.stride(1),
            add_unit_offset=add_unit_offset,
            D=D,
            BLOCK_D=BLOCK_D,
        )

        # dL/dw: split-K reduction.
        # Grid: (n_col_blocks, n_m_blocks). Each program reduces a
        # (M_PER_PROGRAM, BLOCK_D_REDUCE) tile into a partial sum. A second
        # pass sums the n_m_blocks partial sums column-wise into grad_w_fp32.
        BLOCK_M = 32
        BLOCK_D_REDUCE = 64
        # Aim for ~4 programs per SM on an A100 (108 SMs) → ~432 programs
        # total. n_col_blocks = D/BLOCK_D_REDUCE. Pick M_PER_PROGRAM so that
        # n_col_blocks * n_m_blocks lands in that range, while keeping
        # M_PER_PROGRAM a multiple of BLOCK_M and large enough to amortize
        # kernel-launch cost.
        n_col_blocks = (D + BLOCK_D_REDUCE - 1) // BLOCK_D_REDUCE
        target_programs = 432
        target_n_m_blocks = max(1, target_programs // max(1, n_col_blocks))
        # Round M_PER_PROGRAM up to a multiple of BLOCK_M, at least BLOCK_M
        m_per_program = max(
            BLOCK_M,
            ((M + target_n_m_blocks - 1) // target_n_m_blocks + BLOCK_M - 1)
            // BLOCK_M
            * BLOCK_M,
        )
        n_m_blocks = (M + m_per_program - 1) // m_per_program

        partial = torch.empty(
            (n_m_blocks, D), device=x_view.device, dtype=torch.float32
        )

        _fused_rmsnorm_bwd_dw_partial_kernel[(n_col_blocks, n_m_blocks)](
            x_view,
            grad_out_view,
            rsqrt,
            partial,
            x_view.stride(0),
            x_view.stride(1),
            grad_out_view.stride(0),
            grad_out_view.stride(1),
            partial.stride(0),
            partial.stride(1),
            M,
            M_PER_PROGRAM=m_per_program,
            D=D,
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D_REDUCE,
        )

        _fused_rmsnorm_bwd_dw_reduce_kernel[(n_col_blocks,)](
            partial,
            grad_w_fp32,
            partial.stride(0),
            partial.stride(1),
            N_MB=n_m_blocks,
            D=D,
            BLOCK_D=BLOCK_D_REDUCE,
        )

        grad_w = grad_w_fp32.to(weight.dtype)

        return grad_x.view(ctx.original_shape), grad_w, None, None


def fused_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    add_unit_offset: bool = False,
) -> torch.Tensor:
    """Fused RMSNorm: drop-in replacement for eager RMSNorm.forward.

    Args:
        x: Input tensor, any shape with last dim = norm dim.
        weight: Learnable gain, shape (D,).
        eps: Numerical stability epsilon.
        add_unit_offset: If True, use (1 + weight) as the gain (Gemma-style).

    Returns:
        Normalized tensor, same shape and dtype as x.
    """
    return _FusedRMSNorm.apply(x, weight, eps, add_unit_offset)


# Module-level flag controlling whether RMSNorm classes are patched to use the
# fused kernel. Flipped by `set_fused_rmsnorm_enabled()` during model setup.
_USE_FUSED_RMSNORM = False


def set_fused_rmsnorm_enabled(enabled: bool):
    """Toggle whether RMSNorm.forward uses the fused Triton kernel.

    When enabled, both keys_values.model.RMSNorm and litgpt.model.RMSNorm
    dispatch through `fused_rmsnorm` (falling back to eager if
    `can_use_fused_rmsnorm` rejects the input). When disabled, both use
    their original eager forward.

    Idempotent — calling twice with the same value is a no-op.
    """
    global _USE_FUSED_RMSNORM
    if _USE_FUSED_RMSNORM == enabled:
        return
    _USE_FUSED_RMSNORM = enabled
    _install_or_restore_hooks(enabled)


# Handles to restore the eager forwards when disabled
_ORIGINAL_FORWARDS = {}


def _install_or_restore_hooks(install: bool):
    from keys_values.model import RMSNorm as RMSNormNew
    from litgpt.model import RMSNorm as RMSNormOld

    classes = [RMSNormNew, RMSNormOld]

    if install:
        for cls in classes:
            if cls in _ORIGINAL_FORWARDS:
                continue
            _ORIGINAL_FORWARDS[cls] = cls.forward
            cls.forward = _dispatch_forward
    else:
        for cls in classes:
            orig = _ORIGINAL_FORWARDS.pop(cls, None)
            if orig is not None:
                cls.forward = orig


def _dispatch_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Patched RMSNorm.forward: dispatch to fused if eligible, else eager."""
    # dim attribute exists on both classes; defaults to -1
    dim = getattr(self, "dim", -1)
    weight = self.weight
    eps = self.eps
    add_unit_offset = bool(getattr(self, "add_unit_offset", False))
    if can_use_fused_rmsnorm(x, weight, dim):
        return fused_rmsnorm(x, weight, eps, add_unit_offset)
    # Fallback: call the original eager forward. Find which class owns this
    # method by type(self). add_unit_offset handling differs slightly between
    # the two original classes but both fall back cleanly.
    orig = _ORIGINAL_FORWARDS.get(type(self))
    if orig is None:
        # Class wasn't patched (shouldn't happen, but be safe)
        raise RuntimeError(
            f"_dispatch_forward called on unpatched class {type(self).__name__}"
        )
    return orig(self, x)
