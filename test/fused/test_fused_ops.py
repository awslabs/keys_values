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
"""Comparison tests for fused Triton operators vs eager PyTorch expressions.

Each test compares:
- forward: output of the fused op vs the eager expression
- backward: gradients w.r.t. all inputs via a random linear combination loss
"""
from itertools import product

import torch
import torch.nn.functional as F
import pytest

from litgpt.utils import _RunIf
import litgpt.model as litgpt_model

from keys_values import model as kv_model
from keys_values import lora as kv_lora
from keys_values.config import Config
from keys_values.lora import Config as LoraConfig
from keys_values.fused.fused_rmsnorm import fused_rmsnorm
from keys_values.fused.fused_rope import fused_apply_rope
from keys_values.fused.fused_swiglu import fused_swiglu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_loss(y: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """Scalar loss = fixed linear combination of the output elements."""
    return (y * coeff).sum()


def _copy_with_grad(t: torch.Tensor) -> torch.Tensor:
    return t.detach().clone().requires_grad_(True)


def _tolerances(dtype: torch.dtype):
    """Return (atol, rtol) appropriate for comparing fused vs eager."""
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 1e-2, 1e-2


# ---------------------------------------------------------------------------
# fused_rmsnorm
# ---------------------------------------------------------------------------

_RMSNORM_SHAPES = [
    # (batch_dims, D)
    ((4,), 64),
    ((8,), 128),
    ((2, 16), 256),
    ((1, 32), 512),
    ((32,), 1024),
    ((4, 8), 128),
    ((16,), 32),
]
_RMSNORM_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
_RMSNORM_ADD_UNIT_OFFSET = [False, True]

_RMSNORM_PARAMS = [
    (batch_dims, D, dtype, auo)
    for (batch_dims, D), dtype, auo in product(
        _RMSNORM_SHAPES, _RMSNORM_DTYPES, _RMSNORM_ADD_UNIT_OFFSET
    )
]


def _rmsnorm_eager(
    x: torch.Tensor, weight: torch.Tensor, eps: float, add_unit_offset: bool
) -> torch.Tensor:
    """Reference RMSNorm: compute in fp32, return in input dtype."""
    x_f = x.float()
    rms = x_f.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    y = x_f * rms
    w = weight.float()
    if add_unit_offset:
        w = w + 1.0
    return (y * w).to(x.dtype)


def _rmsnorm_via_module(
    module_class,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    add_unit_offset: bool,
) -> torch.Tensor:
    """Run RMSNorm through an nn.Module instance, sharing the given weight."""
    D = x.shape[-1]
    m = module_class(D, eps=eps, add_unit_offset=add_unit_offset).to(x.device)
    with torch.no_grad():
        # Both module classes store weight as float32; cast to match.
        m.weight.copy_(weight.float())
    return m(x)


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("batch_dims, D, dtype, add_unit_offset", _RMSNORM_PARAMS)
def test_fused_rmsnorm_forward(batch_dims, D, dtype, add_unit_offset):
    seed = 31415927
    torch.manual_seed(seed)
    device = torch.device("cuda", 0)
    eps = 1e-5

    shape = (*batch_dims, D)
    x = torch.randn(shape, device=device, dtype=dtype)
    weight = torch.randn(D, device=device, dtype=dtype) * 0.1

    out_fused = fused_rmsnorm(x, weight, eps, add_unit_offset)

    atol, rtol = _tolerances(dtype)
    references = {
        "eager": _rmsnorm_eager(x, weight, eps, add_unit_offset),
        "kv_model.RMSNorm": _rmsnorm_via_module(kv_model.RMSNorm, x, weight, eps, add_unit_offset),
        "litgpt.RMSNorm": _rmsnorm_via_module(litgpt_model.RMSNorm, x, weight, eps, add_unit_offset),
    }
    for name, out_ref in references.items():
        print(f"rmsnorm fwd: shape={shape}, dtype={dtype}, ref={name}")
        torch.testing.assert_close(out_fused, out_ref, atol=atol, rtol=rtol)


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("batch_dims, D, dtype, add_unit_offset", _RMSNORM_PARAMS)
def test_fused_rmsnorm_backward(batch_dims, D, dtype, add_unit_offset):
    seed = 31415927
    torch.manual_seed(seed)
    device = torch.device("cuda", 0)
    eps = 1e-5

    shape = (*batch_dims, D)
    _x = torch.randn(shape, device=device, dtype=dtype)
    _w = torch.randn(D, device=device, dtype=dtype) * 0.1
    coeff = torch.randn(shape, device=device)

    # Collect gradients from fused op and each reference
    def _run(fn):
        x = _copy_with_grad(_x)
        w = _copy_with_grad(_w)
        _linear_loss(fn(x, w), coeff).backward()
        return {"x": x.grad, "w": w.grad}

    grads_fused = _run(lambda x, w: fused_rmsnorm(x, w, eps, add_unit_offset))
    references = {
        "eager": _run(lambda x, w: _rmsnorm_eager(x, w, eps, add_unit_offset)),
        "kv_model.RMSNorm": _run(
            lambda x, w: _rmsnorm_via_module(kv_model.RMSNorm, x, w, eps, add_unit_offset)
        ),
        "litgpt.RMSNorm": _run(
            lambda x, w: _rmsnorm_via_module(litgpt_model.RMSNorm, x, w, eps, add_unit_offset)
        ),
    }

    atol, rtol = _tolerances(dtype)
    for ref_name, grads_ref in references.items():
        for param_name in ("x", "w"):
            print(
                f"rmsnorm bwd: shape={shape}, dtype={dtype},"
                f" add_unit_offset={add_unit_offset}, ref={ref_name}, grad[{param_name}]"
            )
            torch.testing.assert_close(
                grads_fused[param_name],
                grads_ref[param_name],
                atol=atol,
                rtol=rtol,
            )


# ---------------------------------------------------------------------------
# fused_apply_rope
# ---------------------------------------------------------------------------

# 3-D inputs [BH, T, D]; D must be even
_ROPE_SHAPES_3D = [
    # (BH, T, D)
    (4, 16, 64),
    (8, 32, 128),
    (2, 64, 256),
    (16, 8, 32),
    (1, 128, 64),
    (6, 7, 128),   # T not a multiple of BLOCK_T=32
    (4, 33, 64),   # T slightly above BLOCK_T
]

# 4-D inputs [B, n_head, T, D] – tests shape broadcasting in reshape logic
_ROPE_SHAPES_4D = [
    # (B, n_head, T, D)
    (2, 4, 16, 64),
    (3, 8, 32, 128),
    (1, 2, 7, 32),
]

_ROPE_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

_ROPE_PARAMS_3D = [
    (BH, T, D, dtype)
    for (BH, T, D), dtype in product(_ROPE_SHAPES_3D, _ROPE_DTYPES)
]
_ROPE_PARAMS_4D = [
    (B, n_head, T, D, dtype)
    for (B, n_head, T, D), dtype in product(_ROPE_SHAPES_4D, _ROPE_DTYPES)
]


def _build_cos_sin(T: int, D: int, device: torch.device) -> tuple:
    """Build a realistic (cos, sin) pair in float32 from random phase angles.

    cos² + sin² = 1 elementwise, which keeps intermediate products O(1) and
    avoids the large cancellation errors that occur when cos and sin are
    sampled independently with torch.randn.
    """
    theta = torch.rand(T, D // 2, device=device) * 2 * torch.pi
    cos = torch.cat([theta.cos(), theta.cos()], dim=-1)
    sin = torch.cat([theta.sin(), theta.sin()], dim=-1)
    return cos, sin


def _rope_eager(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Reference RoPE: y = x * cos + rot(x) * sin, computed in fp32."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rot_x = torch.cat((-x2, x1), dim=-1)
    return (x.float() * cos.float() + rot_x.float() * sin.float()).to(x.dtype)


def _rope_via_litgpt(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Delegate to litgpt.model.apply_rope, which requires 3-D cos/sin (1, T, D)."""
    cos3d = cos.reshape(1, cos.shape[-2], cos.shape[-1])
    sin3d = sin.reshape(1, sin.shape[-2], sin.shape[-1])
    return litgpt_model.apply_rope(x, cos3d, sin3d)


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("BH, T, D, dtype", _ROPE_PARAMS_3D)
def test_fused_apply_rope_forward(BH, T, D, dtype):
    seed = 31415927
    torch.manual_seed(seed)
    device = torch.device("cuda", 0)

    x = torch.randn(BH, T, D, device=device, dtype=dtype)
    cos, sin = _build_cos_sin(T, D, device)
    cos, sin = cos.to(dtype), sin.to(dtype)

    out_fused = fused_apply_rope(x, cos, sin)

    atol, rtol = _tolerances(dtype)
    references = {
        "eager": _rope_eager(x, cos, sin),
        "litgpt.apply_rope": _rope_via_litgpt(x, cos, sin),
    }
    for name, out_ref in references.items():
        print(f"rope fwd 3D: BH={BH}, T={T}, D={D}, dtype={dtype}, ref={name}")
        torch.testing.assert_close(out_fused, out_ref, atol=atol, rtol=rtol)


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("BH, T, D, dtype", _ROPE_PARAMS_3D)
def test_fused_apply_rope_backward(BH, T, D, dtype):
    seed = 31415927
    torch.manual_seed(seed)
    device = torch.device("cuda", 0)

    _x = torch.randn(BH, T, D, device=device, dtype=dtype)
    cos, sin = _build_cos_sin(T, D, device)
    cos, sin = cos.to(dtype), sin.to(dtype)
    coeff = torch.randn(BH, T, D, device=device)

    def _run(fn):
        x = _copy_with_grad(_x)
        _linear_loss(fn(x), coeff).backward()
        return x.grad

    grad_fused = _run(lambda x: fused_apply_rope(x, cos, sin))
    references = {
        "eager": _run(lambda x: _rope_eager(x, cos, sin)),
        "litgpt.apply_rope": _run(lambda x: _rope_via_litgpt(x, cos, sin)),
    }

    print(f"rope bwd 3D: BH={BH}, T={T}, D={D}, dtype={dtype}")
    atol, rtol = _tolerances(dtype)
    for name, grad_ref in references.items():
        torch.testing.assert_close(grad_fused, grad_ref, atol=atol, rtol=rtol)


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("B, n_head, T, D, dtype", _ROPE_PARAMS_4D)
def test_fused_apply_rope_nd_forward(B, n_head, T, D, dtype):
    seed = 31415927
    torch.manual_seed(seed)
    device = torch.device("cuda", 0)

    x = torch.randn(B, n_head, T, D, device=device, dtype=dtype)
    cos, sin = _build_cos_sin(T, D, device)
    cos, sin = cos.to(dtype), sin.to(dtype)

    out_fused = fused_apply_rope(x, cos, sin)

    atol, rtol = _tolerances(dtype)
    references = {
        "eager": _rope_eager(x, cos, sin),
        "litgpt.apply_rope": _rope_via_litgpt(x, cos, sin),
    }
    for name, out_ref in references.items():
        print(f"rope fwd 4D: B={B}, n_head={n_head}, T={T}, D={D}, dtype={dtype}, ref={name}")
        torch.testing.assert_close(out_fused, out_ref, atol=atol, rtol=rtol)


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("B, n_head, T, D, dtype", _ROPE_PARAMS_4D)
def test_fused_apply_rope_nd_backward(B, n_head, T, D, dtype):
    seed = 31415927
    torch.manual_seed(seed)
    device = torch.device("cuda", 0)

    _x = torch.randn(B, n_head, T, D, device=device, dtype=dtype)
    cos, sin = _build_cos_sin(T, D, device)
    cos, sin = cos.to(dtype), sin.to(dtype)
    coeff = torch.randn(B, n_head, T, D, device=device)

    def _run(fn):
        x = _copy_with_grad(_x)
        _linear_loss(fn(x), coeff).backward()
        return x.grad

    grad_fused = _run(lambda x: fused_apply_rope(x, cos, sin))
    references = {
        "eager": _run(lambda x: _rope_eager(x, cos, sin)),
        "litgpt.apply_rope": _run(lambda x: _rope_via_litgpt(x, cos, sin)),
    }

    print(f"rope bwd 4D: B={B}, n_head={n_head}, T={T}, D={D}, dtype={dtype}")
    atol, rtol = _tolerances(dtype)
    for name, grad_ref in references.items():
        torch.testing.assert_close(grad_fused, grad_ref, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# fused_swiglu
# ---------------------------------------------------------------------------

_SWIGLU_SHAPES = [
    (64,),
    (128,),
    (4, 256),
    (8, 512),
    (2, 16, 128),
    (32, 1024),
    (1, 1, 64),
    (7, 13, 32),   # non-power-of-two dimensions
]
_SWIGLU_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

_SWIGLU_PARAMS = [
    (shape, dtype) for shape, dtype in product(_SWIGLU_SHAPES, _SWIGLU_DTYPES)
]


def _swiglu_eager(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference SwiGLU: silu(a) * b, computed in fp32, returned in input dtype."""
    return (F.silu(a.float()) * b.float()).to(a.dtype)



@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("shape, dtype", _SWIGLU_PARAMS)
def test_fused_swiglu_forward(shape, dtype):
    seed = 31415927
    torch.manual_seed(seed)
    device = torch.device("cuda", 0)

    a = torch.randn(shape, device=device, dtype=dtype)
    b = torch.randn(shape, device=device, dtype=dtype)

    out_fused = fused_swiglu(a, b)

    atol, rtol = _tolerances(dtype)
    references = {
        "eager": _swiglu_eager(a, b),
    }
    for name, out_ref in references.items():
        print(f"swiglu fwd: shape={shape}, dtype={dtype}, ref={name}")
        torch.testing.assert_close(out_fused, out_ref, atol=atol, rtol=rtol)


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("shape, dtype", _SWIGLU_PARAMS)
def test_fused_swiglu_backward(shape, dtype):
    seed = 31415927
    torch.manual_seed(seed)
    device = torch.device("cuda", 0)

    _a = torch.randn(shape, device=device, dtype=dtype)
    _b = torch.randn(shape, device=device, dtype=dtype)
    coeff = torch.randn(shape, device=device)

    def _run(fn):
        a = _copy_with_grad(_a)
        b = _copy_with_grad(_b)
        _linear_loss(fn(a, b), coeff).backward()
        return {"a": a.grad, "b": b.grad}

    grads_fused = _run(fused_swiglu)
    references = {
        "eager": _run(_swiglu_eager),
    }

    print(f"swiglu bwd: shape={shape}, dtype={dtype}")
    atol, rtol = _tolerances(dtype)
    for ref_name, grads_ref in references.items():
        for param_name in ("a", "b"):
            torch.testing.assert_close(
                grads_fused[param_name],
                grads_ref[param_name],
                atol=atol,
                rtol=rtol,
            )


# ---------------------------------------------------------------------------
# fused_swiglu via LLaMAMLP.forward — end-to-end module comparison
#
# Here we run a full LLaMAMLP forward pass with the fused kernel active
# (via set_fused_swiglu_enabled) and compare against the same module with
# the kernel disabled.  We use a fixed 1-D intermediate size so we can
# build a valid Config.
# ---------------------------------------------------------------------------

_SWIGLU_MLP_SHAPES = [
    # (batch_shape, n_embd, intermediate_size)
    ((4,), 64, 128),
    ((2, 8), 128, 256),
    ((16,), 64, 192),
]
_SWIGLU_MLP_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
_SWIGLU_MLP_PARAMS = list(product(_SWIGLU_MLP_SHAPES, _SWIGLU_MLP_DTYPES))


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("shape_cfg, dtype", _SWIGLU_MLP_PARAMS)
def test_fused_swiglu_llamamlp_forward(shape_cfg, dtype):
    batch_shape, n_embd, intermediate_size = shape_cfg
    seed = 31415927
    torch.manual_seed(seed)
    device = torch.device("cuda", 0)

    base_cfg = Config(
        n_layer=1, n_head=4, n_embd=n_embd, block_size=32,
        vocab_size=32, intermediate_size=intermediate_size,
    )
    lora_cfg = LoraConfig(
        n_layer=1, n_head=4, n_embd=n_embd, block_size=32,
        vocab_size=32, intermediate_size=intermediate_size,
    )

    x = torch.randn(*batch_shape, n_embd, device=device, dtype=dtype)

    atol, rtol = _tolerances(dtype)
    for mlp_class, cfg, label in (
        (litgpt_model.LLaMAMLP, base_cfg, "litgpt.LLaMAMLP"),
        (kv_lora.LLaMAMLP, lora_cfg, "kv_lora.LLaMAMLP"),
    ):
        mlp = mlp_class(cfg).to(device=device, dtype=dtype)
        with torch.no_grad():
            out_eager = mlp(x)

        from keys_values.fused.fused_swiglu import set_fused_swiglu_enabled
        set_fused_swiglu_enabled(True)
        try:
            with torch.no_grad():
                out_fused = mlp(x)
        finally:
            set_fused_swiglu_enabled(False)

        print(f"swiglu mlp fwd: shape={(*batch_shape, n_embd)}, dtype={dtype}, ref={label}")
        torch.testing.assert_close(out_fused, out_eager, atol=atol, rtol=rtol)


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize("shape_cfg, dtype", _SWIGLU_MLP_PARAMS)
def test_fused_swiglu_llamamlp_backward(shape_cfg, dtype):
    batch_shape, n_embd, intermediate_size = shape_cfg
    seed = 31415927
    torch.manual_seed(seed)
    device = torch.device("cuda", 0)

    base_cfg = Config(
        n_layer=1, n_head=4, n_embd=n_embd, block_size=32,
        vocab_size=32, intermediate_size=intermediate_size,
    )
    lora_cfg = LoraConfig(
        n_layer=1, n_head=4, n_embd=n_embd, block_size=32,
        vocab_size=32, intermediate_size=intermediate_size,
    )

    _x = torch.randn(*batch_shape, n_embd, device=device, dtype=dtype)

    atol, rtol = _tolerances(dtype)
    from keys_values.fused.fused_swiglu import set_fused_swiglu_enabled

    for mlp_class, cfg, label in (
        (litgpt_model.LLaMAMLP, base_cfg, "litgpt.LLaMAMLP"),
        (kv_lora.LLaMAMLP, lora_cfg, "kv_lora.LLaMAMLP"),
    ):
        mlp = mlp_class(cfg).to(device=device, dtype=dtype)

        # Run eager once to determine output shape, then fix the coefficient.
        with torch.no_grad():
            y_shape = mlp(_x).shape
        coeff = torch.randn(y_shape, device=device)

        grads = {}
        for kind in ("eager", "fused"):
            x = _copy_with_grad(_x)
            if kind == "fused":
                set_fused_swiglu_enabled(True)
            try:
                y = mlp(x)
            finally:
                if kind == "fused":
                    set_fused_swiglu_enabled(False)
            _linear_loss(y, coeff).backward()
            grads[kind] = x.grad

        print(
            f"swiglu mlp bwd: shape={(*batch_shape, n_embd)}, dtype={dtype}, ref={label}"
        )
        torch.testing.assert_close(grads["fused"], grads["eager"], atol=atol, rtol=rtol)
