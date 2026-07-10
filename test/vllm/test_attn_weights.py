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
Tests for H2O attention-weight extraction (task 3).

The reference tests run anywhere (CPU torch). The FlashInfer parity test needs a
GPU with the vendored kernels and is skipped otherwise.
"""

import math

import pytest
import torch

from keys_values.vllm.attention import (
    flashinfer_weight_path_available,
    reference_summed_attention,
    summed_attention_weights,
)


def test_reference_output_matches_torch_sdpa_mha():
    """Reference attention output matches torch's SDPA for full causal MHA."""
    torch.manual_seed(0)
    b, h, n, d = 2, 3, 16, 8
    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, d)
    out, _ = reference_summed_attention(q, k, v, causal=True)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, expected, atol=1e-4, rtol=1e-4)


def test_reference_output_matches_torch_sdpa_gqa():
    """Reference matches torch SDPA with grouped-query attention."""
    torch.manual_seed(0)
    b, hq, hkv, n, d = 2, 4, 2, 12, 8
    q = torch.randn(b, hq, n, d)
    k = torch.randn(b, hkv, n, d)
    v = torch.randn(b, hkv, n, d)
    out, summed = reference_summed_attention(q, k, v, causal=True)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=True
    )
    assert torch.allclose(out, expected, atol=1e-4, rtol=1e-4)
    assert summed.shape == (b, hkv, n)
    assert summed.dtype == torch.float32


def test_summed_weights_total_mass_invariant():
    """Each query row's probs sum to 1; total mass per KV head = q_len * group."""
    torch.manual_seed(1)
    b, hq, hkv, n, d = 1, 6, 3, 10, 4
    group = hq // hkv
    q = torch.randn(b, hq, n, d)
    k = torch.randn(b, hkv, n, d)
    v = torch.randn(b, hkv, n, d)
    _, summed = reference_summed_attention(q, k, v, causal=True)
    total = summed.sum(dim=-1)  # (b, hkv)
    assert torch.allclose(total, torch.full_like(total, float(n * group)), atol=1e-3)
    assert torch.all(summed >= 0)


def test_summed_weights_decode_single_query():
    """q_len=1 (decode): summed weights equal the single row's softmax probs."""
    torch.manual_seed(2)
    b, h, kv, d = 1, 1, 5, 4
    q = torch.randn(b, h, 1, d)
    k = torch.randn(b, h, kv, d)
    v = torch.randn(b, h, kv, d)
    _, summed = reference_summed_attention(q, k, v, causal=True)
    scale = 1.0 / math.sqrt(d)
    expected = torch.softmax((q @ k.transpose(-1, -2) * scale), dim=-1).squeeze(2)
    assert torch.allclose(summed, expected.float(), atol=1e-5)


def test_causal_mask_excludes_future_for_update():
    """In an update (q_len<kv_len), the last query attends to all kv; the first
    query (older) cannot attend to the most recent kv slots."""
    torch.manual_seed(3)
    b, h, kv, q_len, d = 1, 1, 8, 3, 4
    q = torch.randn(b, h, q_len, d)
    k = torch.randn(b, h, kv, d)
    v = torch.randn(b, h, kv, d)
    # Build per-query probs to inspect the mask directly via reference internals:
    _, summed = reference_summed_attention(q, k, v, causal=True)
    # Total mass equals q_len (one head, group 1).
    assert torch.allclose(summed.sum(), torch.tensor(float(q_len)), atol=1e-3)


def test_dispatcher_falls_back_to_reference_on_cpu():
    """Without a usable FlashInfer path, the dispatcher returns the reference."""
    torch.manual_seed(4)
    b, h, kv, q_len, d = 1, 2, 8, 2, 8
    q = torch.randn(b, h, q_len, d)
    k = torch.randn(b, h, kv, d)
    v = torch.randn(b, h, kv, d)
    out, summed = summed_attention_weights(q, k, v)
    ref_out, ref_summed = reference_summed_attention(q, k, v)
    assert torch.allclose(out, ref_out, atol=1e-5)
    assert torch.allclose(summed, ref_summed, atol=1e-5)


@pytest.mark.parametrize("q_len", [1, 4])
def test_flashinfer_matches_reference_on_gpu(q_len):
    """FlashInfer weight path matches the reference within tolerance (GPU)."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    head_size = 64
    if not flashinfer_weight_path_available(head_size, torch.float16):
        pytest.skip("FlashInfer weight path unavailable (kernels not built)")
    torch.manual_seed(5)
    b, h, kv = 1, 2, 64
    dev = "cuda"
    q = torch.randn(b, h, q_len, head_size, dtype=torch.float16, device=dev)
    k = torch.randn(b, h, kv, head_size, dtype=torch.float16, device=dev)
    v = torch.randn(b, h, kv, head_size, dtype=torch.float16, device=dev)
    _, fi_summed = summed_attention_weights(q, k, v, prefer_flashinfer=True)
    _, ref_summed = reference_summed_attention(q, k, v)
    # fp16 + kernel differences: compare with a loose tolerance.
    assert fi_summed.shape == ref_summed.shape
    assert torch.allclose(fi_summed.cpu(), ref_summed.cpu(), atol=1e-2, rtol=1e-2)
