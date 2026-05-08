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
"""Fused Triton kernel for SwiGLU (silu(a) * b).

Replaces the eager LLaMAMLP inner sequence `F.silu(x_fc_1) * x_fc_2`
(2 kernels: silu + elementwise multiply) with a single Triton kernel.
Forward and backward are both implemented as torch.autograd.Function so
the training cell loop's saved_tensors_hooks see a normal autograd
function.

Math:
  silu(a) = a * sigmoid(a)
  y = silu(a) * b

Backward (with s = sigmoid(a), f = silu(a) = a * s):
  dL/da = dL/dy * b * s * (1 + a * (1 - s))
        = dL/dy * b * (s + f * (1 - s))
  dL/db = dL/dy * f
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
    def _swiglu_fwd_kernel(
        A_ptr,  # [N]  (flattened)
        B_ptr,  # [N]
        Out_ptr,  # [N]
        N,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N

        a = tl.load(A_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        s = tl.sigmoid(a)
        y = a * s * b  # silu(a) * b
        tl.store(Out_ptr + offsets, y.to(A_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_bwd_kernel(
        A_ptr,  # [N]
        B_ptr,  # [N]
        GradOut_ptr,  # [N]
        GradA_ptr,  # [N]
        GradB_ptr,  # [N]
        N,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N

        a = tl.load(A_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        go = tl.load(GradOut_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        s = tl.sigmoid(a)
        f = a * s  # silu(a)
        # dL/da = go * b * (s + f * (1 - s))
        # Equivalent to: go * b * s * (1 + a * (1 - s))
        ga = go * b * (s + f * (1.0 - s))
        # dL/db = go * f
        gb = go * f

        tl.store(GradA_ptr + offsets, ga.to(A_ptr.dtype.element_ty), mask=mask)
        tl.store(GradB_ptr + offsets, gb.to(B_ptr.dtype.element_ty), mask=mask)


def can_use_fused_swiglu(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Check if `fused_swiglu` can handle these inputs."""
    if not _triton_available:
        return False
    if not a.is_cuda:
        return False
    if a.shape != b.shape:
        return False
    if a.dtype != b.dtype:
        return False
    if a.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    return True


class _FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        if not can_use_fused_swiglu(a, b):
            raise ValueError(
                "fused_swiglu: inputs not compatible. "
                "Use can_use_fused_swiglu() to check first."
            )

        a_contig = a.contiguous()
        b_contig = b.contiguous()
        out = torch.empty_like(a_contig)

        N = a_contig.numel()
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _swiglu_fwd_kernel[grid](a_contig, b_contig, out, N, BLOCK=BLOCK)

        # Save a, b (flattened-contiguous views) for backward
        ctx.save_for_backward(a_contig, b_contig)
        ctx.original_shape = a.shape

        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        grad_out_contig = grad_out.contiguous()

        grad_a = torch.empty_like(a)
        grad_b = torch.empty_like(b)

        N = a.numel()
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _swiglu_bwd_kernel[grid](
            a,
            b,
            grad_out_contig,
            grad_a,
            grad_b,
            N,
            BLOCK=BLOCK,
        )

        return grad_a, grad_b


def fused_swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU: drop-in replacement for `F.silu(a) * b`.

    Args:
        a: First input (gate), any shape.
        b: Second input (up-proj), same shape/dtype as a.

    Returns:
        Tensor `silu(a) * b`, same shape and dtype as inputs.
    """
    return _FusedSwiGLU.apply(a, b)


# Module-level flag + monkey-patch machinery, following the same pattern as
# fused_rmsnorm. When enabled, patches LLaMAMLP.forward to use the fused
# kernel for the silu-mul step.
_USE_FUSED_SWIGLU = False
_ORIGINAL_FORWARDS = {}


def set_fused_swiglu_enabled(enabled: bool):
    """Toggle whether LLaMAMLP.forward uses the fused SwiGLU kernel.

    Idempotent. When enabled, patches both `keys_values.lora.LLaMAMLP.forward`
    and `litgpt.model.LLaMAMLP.forward`. When disabled, restores the
    originals.
    """
    global _USE_FUSED_SWIGLU
    if _USE_FUSED_SWIGLU == enabled:
        return
    _USE_FUSED_SWIGLU = enabled
    _install_or_restore_hooks(enabled)


def _install_or_restore_hooks(install: bool):
    from keys_values.lora import LLaMAMLP as LLaMAMLPLoRA
    from litgpt.model import LLaMAMLP as LLaMAMLPLitgpt

    classes = [LLaMAMLPLoRA, LLaMAMLPLitgpt]

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
    """Patched LLaMAMLP.forward: fuses the silu-mul step."""
    x_fc_1 = self.fc_1(x)
    x_fc_2 = self.fc_2(x)
    if can_use_fused_swiglu(x_fc_1, x_fc_2):
        gated = fused_swiglu(x_fc_1, x_fc_2)
    else:
        gated = torch.nn.functional.silu(x_fc_1) * x_fc_2
    return self.proj(gated)
