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

"""Tests for fused Triton kernel for attention weight accumulation."""

import math

import pytest
import torch

# Skip all tests if CUDA or Triton unavailable
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
]

try:
    from keys_values.triton_kernels import compute_weights_from_lse_triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def _compute_reference_weights(query, key, scale_factor, input_pos, sliding_window_size=None):
    """Compute attention weights using explicit PyTorch ops as reference."""
    batch_size, n_head, q_len, head_dim = query.shape
    _, n_kv_heads, kv_len, _ = key.shape
    q_per_kv = n_head // n_kv_heads

    log2e = math.log2(math.e)
    sm_scale_log2 = scale_factor * log2e

    # Compute full scores: [batch, n_head, q_len, kv_len]
    scores = torch.zeros(batch_size, n_head, q_len, kv_len, device=query.device, dtype=torch.float32)
    for h in range(n_head):
        kv_h = h // q_per_kv
        scores[:, h] = torch.matmul(
            query[:, h].float(), key[:, kv_h].float().transpose(-2, -1)
        ) * sm_scale_log2

    # Causal mask
    q_positions = torch.arange(q_len, device=query.device)
    kv_indices = torch.arange(kv_len, device=query.device)
    causal_mask = kv_indices[None, :] <= (input_pos + q_positions[:, None])

    if sliding_window_size is not None and sliding_window_size > 0:
        window_mask = (input_pos + q_positions[:, None]) - kv_indices[None, :] < sliding_window_size
        causal_mask = causal_mask & window_mask

    mask_4d = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, n_head, q_len, kv_len)
    scores = scores.masked_fill(~mask_4d, float("-inf"))

    # LSE in log base 2
    lse = torch.logsumexp(scores * math.log(2), dim=-1) / math.log(2)
    lse = lse.permute(0, 2, 1)  # [batch, q_len, n_head]

    # Accumulated weights
    weights = torch.zeros(batch_size, n_kv_heads, kv_len, device=query.device, dtype=torch.float32)
    for h in range(n_head):
        kv_h = h // q_per_kv
        norm_w = torch.exp2(scores[:, h] - lse[:, :, h].unsqueeze(-1))
        weights[:, kv_h] += norm_w.sum(dim=1)

    return weights, lse


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
class TestTritonWeightsKernel:

    def test_triton_matches_reference_basic(self):
        """Small config: batch=1, heads=4, kv_heads=2, q=16, kv=32, dim=64."""
        B, H, Hkv, D = 1, 4, 2, 64
        q_len, kv_len = 16, 32
        scale = 1.0 / math.sqrt(D)
        input_pos = kv_len - q_len

        Q = torch.randn(B, H, q_len, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, Hkv, kv_len, D, device="cuda", dtype=torch.float16)

        ref_weights, lse = _compute_reference_weights(Q, K, scale, input_pos)
        triton_weights = compute_weights_from_lse_triton(Q, K, lse, scale, input_pos)

        torch.testing.assert_close(triton_weights, ref_weights, rtol=1e-3, atol=1e-4)

    def test_triton_matches_reference_gqa(self):
        """Qwen3-4B GQA config: heads=32, kv_heads=8, q_per_kv=4, dim=128."""
        B, H, Hkv, D = 1, 32, 8, 128
        q_len, kv_len = 64, 256
        scale = 1.0 / math.sqrt(D)
        input_pos = kv_len - q_len

        Q = torch.randn(B, H, q_len, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, Hkv, kv_len, D, device="cuda", dtype=torch.float16)

        ref_weights, lse = _compute_reference_weights(Q, K, scale, input_pos)
        triton_weights = compute_weights_from_lse_triton(Q, K, lse, scale, input_pos)

        torch.testing.assert_close(triton_weights, ref_weights, rtol=1e-3, atol=1e-4)

    def test_triton_output_shape_and_dtype(self):
        """Verify output shape is [batch, n_kv_heads, kv_len] and dtype is float32."""
        B, H, Hkv, D = 2, 8, 2, 64
        q_len, kv_len = 32, 64
        scale = 1.0 / math.sqrt(D)
        input_pos = kv_len - q_len

        Q = torch.randn(B, H, q_len, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, Hkv, kv_len, D, device="cuda", dtype=torch.float16)
        _, lse = _compute_reference_weights(Q, K, scale, input_pos)

        w = compute_weights_from_lse_triton(Q, K, lse, scale, input_pos)
        assert w.shape == (B, Hkv, kv_len)
        assert w.dtype == torch.float32

    def test_triton_nonnegative_weights(self):
        """All output values should be >= 0."""
        B, H, Hkv, D = 1, 8, 2, 64
        q_len, kv_len = 32, 64
        scale = 1.0 / math.sqrt(D)
        input_pos = kv_len - q_len

        Q = torch.randn(B, H, q_len, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, Hkv, kv_len, D, device="cuda", dtype=torch.float16)
        _, lse = _compute_reference_weights(Q, K, scale, input_pos)

        w = compute_weights_from_lse_triton(Q, K, lse, scale, input_pos)
        assert (w >= 0).all()

    def test_triton_causal_mask(self):
        """Future positions should have zero weight."""
        B, H, Hkv, D = 1, 4, 2, 64
        q_len, kv_len = 4, 8
        scale = 1.0 / math.sqrt(D)
        input_pos = 0  # queries at positions 0..3, so kv positions 4..7 are future

        Q = torch.randn(B, H, q_len, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, Hkv, kv_len, D, device="cuda", dtype=torch.float16)
        _, lse = _compute_reference_weights(Q, K, scale, input_pos)

        w = compute_weights_from_lse_triton(Q, K, lse, scale, input_pos)
        # Positions 4..7 should have zero weight (all queries at 0..3)
        assert (w[:, :, 4:] < 1e-6).all(), f"Future positions should be zero, got max={w[:, :, 4:].max()}"

    def test_triton_sliding_window(self):
        """Sliding window should zero out distant positions."""
        B, H, Hkv, D = 1, 4, 2, 64
        q_len, kv_len = 8, 32
        scale = 1.0 / math.sqrt(D)
        input_pos = kv_len - q_len
        sliding_window = 4

        Q = torch.randn(B, H, q_len, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, Hkv, kv_len, D, device="cuda", dtype=torch.float16)

        ref_weights, lse = _compute_reference_weights(Q, K, scale, input_pos, sliding_window)
        triton_weights = compute_weights_from_lse_triton(Q, K, lse, scale, input_pos, sliding_window)

        torch.testing.assert_close(triton_weights, ref_weights, rtol=1e-3, atol=1e-4)
        # Early positions should have zero weight due to sliding window
        assert (triton_weights[:, :, :input_pos - sliding_window] < 1e-6).all()

    def test_triton_fp16(self):
        """Test with fp16 inputs."""
        B, H, Hkv, D = 1, 8, 2, 128
        q_len, kv_len = 32, 64
        scale = 1.0 / math.sqrt(D)
        input_pos = kv_len - q_len

        Q = torch.randn(B, H, q_len, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, Hkv, kv_len, D, device="cuda", dtype=torch.float16)

        ref_weights, lse = _compute_reference_weights(Q, K, scale, input_pos)
        triton_weights = compute_weights_from_lse_triton(Q, K, lse, scale, input_pos)
        torch.testing.assert_close(triton_weights, ref_weights, rtol=1e-3, atol=1e-4)

    def test_triton_bf16(self):
        """Test with bf16 inputs."""
        B, H, Hkv, D = 1, 8, 2, 128
        q_len, kv_len = 32, 64
        scale = 1.0 / math.sqrt(D)
        input_pos = kv_len - q_len

        Q = torch.randn(B, H, q_len, D, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(B, Hkv, kv_len, D, device="cuda", dtype=torch.bfloat16)

        ref_weights, lse = _compute_reference_weights(Q, K, scale, input_pos)
        triton_weights = compute_weights_from_lse_triton(Q, K, lse, scale, input_pos)
        # bf16 has less mantissa precision
        torch.testing.assert_close(triton_weights, ref_weights, rtol=2e-2, atol=1e-3)

    def test_triton_non_multiple_tile_sizes(self):
        """Test with q_len and kv_len that aren't multiples of tile sizes."""
        B, H, Hkv, D = 1, 4, 2, 64
        q_len, kv_len = 17, 33
        scale = 1.0 / math.sqrt(D)
        input_pos = kv_len - q_len

        Q = torch.randn(B, H, q_len, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, Hkv, kv_len, D, device="cuda", dtype=torch.float16)

        ref_weights, lse = _compute_reference_weights(Q, K, scale, input_pos)
        triton_weights = compute_weights_from_lse_triton(Q, K, lse, scale, input_pos)

        torch.testing.assert_close(triton_weights, ref_weights, rtol=1e-3, atol=1e-4)

    def test_triton_large_production_config(self):
        """Production-scale test: Qwen3-4B with q=512, kv=4096."""
        B, H, Hkv, D = 2, 32, 8, 128
        q_len, kv_len = 512, 4096
        scale = 1.0 / math.sqrt(D)
        input_pos = kv_len - q_len

        Q = torch.randn(B, H, q_len, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, Hkv, kv_len, D, device="cuda", dtype=torch.float16)

        ref_weights, lse = _compute_reference_weights(Q, K, scale, input_pos)
        triton_weights = compute_weights_from_lse_triton(Q, K, lse, scale, input_pos)

        torch.testing.assert_close(triton_weights, ref_weights, rtol=5e-3, atol=5e-4)

    def test_triton_batched(self):
        """Test with batch_size > 1."""
        B, H, Hkv, D = 4, 8, 2, 64
        q_len, kv_len = 32, 128
        scale = 1.0 / math.sqrt(D)
        input_pos = kv_len - q_len

        Q = torch.randn(B, H, q_len, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, Hkv, kv_len, D, device="cuda", dtype=torch.float16)

        ref_weights, lse = _compute_reference_weights(Q, K, scale, input_pos)
        triton_weights = compute_weights_from_lse_triton(Q, K, lse, scale, input_pos)

        torch.testing.assert_close(triton_weights, ref_weights, rtol=1e-3, atol=1e-4)

    def test_triton_no_gqa(self):
        """Test with n_head == n_kv_heads (no GQA)."""
        B, H, Hkv, D = 1, 8, 8, 64
        q_len, kv_len = 16, 32
        scale = 1.0 / math.sqrt(D)
        input_pos = kv_len - q_len

        Q = torch.randn(B, H, q_len, D, device="cuda", dtype=torch.float16)
        K = torch.randn(B, Hkv, kv_len, D, device="cuda", dtype=torch.float16)

        ref_weights, lse = _compute_reference_weights(Q, K, scale, input_pos)
        triton_weights = compute_weights_from_lse_triton(Q, K, lse, scale, input_pos)

        torch.testing.assert_close(triton_weights, ref_weights, rtol=1e-3, atol=1e-4)
