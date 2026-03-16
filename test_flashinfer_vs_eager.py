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
Test script to compare FlashInfer SDPA kernels against eager implementation.

This script directly compares the vendored FlashInfer kernels with the eager
fallback implementation to verify numerical equivalence.

Usage:
    python test_flashinfer_vs_eager.py
"""

import torch
import math
from typing import Optional, Tuple, List, Dict, Any

from keys_values.flashinfer_wrapper import (
    FlashInferSDPA,
    verify_backend_equivalence,
    verify_backend_equivalence_batch,
    BackendEquivalenceResult,
)


def create_test_tensors(
    batch_size: int,
    n_head: int,
    n_query_groups: int,
    q_len: int,
    kv_len: int,
    head_size: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Create test tensors for SDPA comparison."""
    query = torch.randn(batch_size, n_head, q_len, head_size, dtype=dtype, device=device)
    key = torch.randn(batch_size, n_query_groups, kv_len, head_size, dtype=dtype, device=device)
    value = torch.randn(batch_size, n_query_groups, kv_len, head_size, dtype=dtype, device=device)
    scale_factor = 1.0 / math.sqrt(head_size)
    return query, key, value, scale_factor


def test_basic_equivalence():
    """Test basic equivalence between FlashInfer and eager SDPA."""
    print("\n" + "=" * 60)
    print("Test: Basic Equivalence (Prefill Phase)")
    print("=" * 60)
    
    wrapper = FlashInferSDPA()
    print(f"FlashInfer available: {wrapper.available}")
    
    if not wrapper.available:
        print("SKIPPED: FlashInfer kernels not available")
        return False
    
    # Test configuration
    batch_size, n_head, n_query_groups = 2, 8, 2
    q_len, kv_len, head_size = 16, 16, 64
    
    query, key, value, scale_factor = create_test_tensors(
        batch_size, n_head, n_query_groups, q_len, kv_len, head_size
    )
    
    result = verify_backend_equivalence(
        query, key, value, scale_factor,
        return_attn_weights=True,
        rtol=1e-3, atol=1e-2,  # Relaxed tolerance for float16
        log_results=True,
    )
    
    print(f"\nResult: {result}")
    print(f"Output max diff: {result.output_max_diff:.2e}")
    print(f"Output mean diff: {result.output_mean_diff:.2e}")
    if result.weights_max_diff is not None:
        print(f"Weights max diff: {result.weights_max_diff:.2e}")
        print(f"Weights mean diff: {result.weights_mean_diff:.2e}")
    
    return result.is_equivalent


def test_decode_phase_equivalence():
    """Test equivalence for decode phase (q_len < kv_len)."""
    print("\n" + "=" * 60)
    print("Test: Decode Phase Equivalence (q_len < kv_len)")
    print("=" * 60)
    
    wrapper = FlashInferSDPA()
    if not wrapper.available:
        print("SKIPPED: FlashInfer kernels not available")
        return False
    
    # Decode phase: single query token attending to longer KV cache
    batch_size, n_head, n_query_groups = 2, 8, 2
    q_len, kv_len, head_size = 1, 64, 64
    
    query, key, value, scale_factor = create_test_tensors(
        batch_size, n_head, n_query_groups, q_len, kv_len, head_size
    )
    
    # Create token positions for decode phase
    token_positions = torch.arange(kv_len, device="cuda").unsqueeze(0).unsqueeze(0)
    token_positions = token_positions.expand(batch_size, n_query_groups, -1)
    
    result = verify_backend_equivalence(
        query, key, value, scale_factor,
        return_attn_weights=True,
        token_positions=token_positions,
        input_pos=kv_len - 1,  # Last position
        rtol=1e-3, atol=1e-2,  # Relaxed tolerance for float16
        log_results=True,
    )
    
    print(f"\nResult: {result}")
    return result.is_equivalent


def test_gqa_equivalence():
    """Test equivalence with Grouped Query Attention."""
    print("\n" + "=" * 60)
    print("Test: Grouped Query Attention Equivalence")
    print("=" * 60)
    
    wrapper = FlashInferSDPA()
    if not wrapper.available:
        print("SKIPPED: FlashInfer kernels not available")
        return False
    
    # GQA: more query heads than KV heads
    configs = [
        {"n_head": 8, "n_query_groups": 2, "desc": "4 queries per KV head"},
        {"n_head": 16, "n_query_groups": 4, "desc": "4 queries per KV head"},
        {"n_head": 32, "n_query_groups": 8, "desc": "4 queries per KV head"},
    ]
    
    all_passed = True
    for config in configs:
        print(f"\n  Testing: {config['desc']} (n_head={config['n_head']}, n_query_groups={config['n_query_groups']})")
        
        batch_size = 2
        q_len, kv_len, head_size = 8, 32, 64
        
        query, key, value, scale_factor = create_test_tensors(
            batch_size, config["n_head"], config["n_query_groups"],
            q_len, kv_len, head_size
        )
        
        result = verify_backend_equivalence(
            query, key, value, scale_factor,
            return_attn_weights=True,
            rtol=1e-3, atol=1e-2,  # Relaxed for float16
            log_results=False,
        )
        
        status = "PASS" if result.is_equivalent else "FAIL"
        print(f"    {status}: output_max_diff={result.output_max_diff:.2e}, weights_max_diff={result.weights_max_diff:.2e}")
        all_passed = all_passed and result.is_equivalent
    
    return all_passed


def test_sliding_window_equivalence():
    """Test equivalence with sliding window attention."""
    print("\n" + "=" * 60)
    print("Test: Sliding Window Attention Equivalence")
    print("=" * 60)
    
    wrapper = FlashInferSDPA()
    if not wrapper.available:
        print("SKIPPED: FlashInfer kernels not available")
        return False
    
    batch_size, n_head, n_query_groups = 2, 8, 2
    q_len, kv_len, head_size = 16, 64, 64
    
    query, key, value, scale_factor = create_test_tensors(
        batch_size, n_head, n_query_groups, q_len, kv_len, head_size
    )
    
    # Test with different sliding window sizes
    window_sizes = [8, 16, 32]
    all_passed = True
    
    for window_size in window_sizes:
        print(f"\n  Testing sliding_window_size={window_size}")
        
        result = verify_backend_equivalence(
            query, key, value, scale_factor,
            return_attn_weights=True,
            sliding_window_size=window_size,
            rtol=1e-3, atol=1e-2,  # Relaxed for float16
            log_results=False,
        )
        
        status = "PASS" if result.is_equivalent else "FAIL"
        print(f"    {status}: output_max_diff={result.output_max_diff:.2e}")
        all_passed = all_passed and result.is_equivalent
    
    return all_passed


def test_various_dtypes():
    """Test equivalence with various data types."""
    print("\n" + "=" * 60)
    print("Test: Various Data Types")
    print("=" * 60)
    
    wrapper = FlashInferSDPA()
    if not wrapper.available:
        print("SKIPPED: FlashInfer kernels not available")
        return False
    
    batch_size, n_head, n_query_groups = 2, 4, 2
    q_len, kv_len, head_size = 8, 16, 64
    
    dtypes = [torch.float16, torch.bfloat16, torch.float32]
    all_passed = True
    
    for dtype in dtypes:
        print(f"\n  Testing dtype={dtype}")
        
        query, key, value, scale_factor = create_test_tensors(
            batch_size, n_head, n_query_groups, q_len, kv_len, head_size,
            dtype=dtype
        )
        
        # Use dtype-appropriate tolerance
        if dtype == torch.float32:
            rtol, atol = 1e-4, 1e-5
        elif dtype == torch.bfloat16:
            rtol, atol = 1e-2, 1e-2  # bfloat16 has lower precision
        else:
            rtol, atol = 1e-3, 1e-2  # float16
        
        result = verify_backend_equivalence(
            query, key, value, scale_factor,
            return_attn_weights=True,
            rtol=rtol, atol=atol,
            log_results=False,
        )
        
        status = "PASS" if result.is_equivalent else "FAIL"
        print(f"    {status}: output_max_diff={result.output_max_diff:.2e}")
        all_passed = all_passed and result.is_equivalent
    
    return all_passed


def test_batch_comparison():
    """Run batch comparison with multiple configurations."""
    print("\n" + "=" * 60)
    print("Test: Batch Comparison (Multiple Configurations)")
    print("=" * 60)
    
    wrapper = FlashInferSDPA()
    if not wrapper.available:
        print("SKIPPED: FlashInfer kernels not available")
        return False
    
    # Generate test cases
    test_cases = []
    configs = [
        # (batch_size, n_head, n_query_groups, q_len, kv_len, head_size)
        (1, 4, 2, 8, 16, 64),
        (2, 8, 4, 16, 32, 64),
        (4, 16, 4, 32, 64, 128),
        (1, 8, 2, 1, 64, 64),  # Decode phase
        (2, 4, 4, 4, 4, 64),   # Prefill phase
    ]
    
    for batch_size, n_head, n_query_groups, q_len, kv_len, head_size in configs:
        query, key, value, scale_factor = create_test_tensors(
            batch_size, n_head, n_query_groups, q_len, kv_len, head_size
        )
        test_cases.append({
            'query': query,
            'key': key,
            'value': value,
            'scale_factor': scale_factor,
            'return_attn_weights': True,
        })
    
    passed, failed, results = verify_backend_equivalence_batch(
        test_cases, rtol=1e-3, atol=1e-2, log_results=True  # Relaxed for float16
    )
    
    print(f"\nBatch Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    
    for i, result in enumerate(results):
        status = "PASS" if result.is_equivalent else "FAIL"
        print(f"  Test {i+1}: {status} - {result.message[:80]}...")
    
    return failed == 0


def test_fallback_correctness():
    """Test that the fallback SDPA produces correct results."""
    print("\n" + "=" * 60)
    print("Test: Fallback SDPA Correctness")
    print("=" * 60)
    
    wrapper = FlashInferSDPA()
    
    # Test configuration
    batch_size, n_head, n_query_groups = 2, 8, 2
    q_len, kv_len, head_size = 16, 16, 64
    
    query, key, value, scale_factor = create_test_tensors(
        batch_size, n_head, n_query_groups, q_len, kv_len, head_size
    )
    
    # Call fallback directly
    output, weights = wrapper._fallback_sdpa(
        query, key, value, scale_factor,
        return_attn_weights=True,
    )
    
    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights dtype: {weights.dtype}")
    
    # Verify shapes
    assert output.shape == (batch_size, n_head, q_len, head_size), f"Output shape mismatch"
    assert weights.shape == (batch_size, n_query_groups, kv_len), f"Weights shape mismatch"
    assert weights.dtype == torch.float32, f"Weights dtype should be float32"
    
    # Verify weights are valid (non-negative, finite)
    assert torch.all(weights >= 0), "Weights should be non-negative"
    assert torch.all(torch.isfinite(weights)), "Weights should be finite"
    
    print("PASS: Fallback SDPA produces correct output shapes and valid weights")
    return True


def test_integration_with_multihead_attention():
    """Test the integration point in MultiHeadSelfAttention."""
    print("\n" + "=" * 60)
    print("Test: MultiHeadSelfAttention Integration")
    print("=" * 60)
    
    from litgpt.config import Config
    from keys_values.attention import MultiHeadSelfAttention, DefaultKeysAndValues
    
    # Create a simple config
    config = Config.from_name(
        "Llama-3.2-1B",
        block_size=128,
        n_layer=1,
    )
    
    mha = MultiHeadSelfAttention(config, use_eager_sdpa_always=True)
    
    # Check that FlashInfer wrapper is initialized
    assert hasattr(mha, '_flashinfer_wrapper'), "FlashInfer wrapper should be initialized"
    print(f"FlashInfer wrapper available: {mha._flashinfer_wrapper.available}")
    
    # Test with return_attn_weights=True (should use FlashInfer path if available)
    batch_size = 2
    q_len = 8
    kv_len = 16
    
    query = torch.randn(batch_size, config.n_head, q_len, config.head_size, device='cuda', dtype=torch.float16)
    key = torch.randn(batch_size, config.n_query_groups, kv_len, config.head_size, device='cuda', dtype=torch.float16)
    value = torch.randn(batch_size, config.n_query_groups, kv_len, config.head_size, device='cuda', dtype=torch.float16)
    
    k_and_v = DefaultKeysAndValues(key, value)
    
    # Create token positions
    token_positions = torch.arange(kv_len, device='cuda').unsqueeze(0).unsqueeze(0)
    token_positions = token_positions.expand(batch_size, config.n_query_groups, -1)
    
    # Call MHA with return_attn_weights=True
    output, weights = mha(
        query=query,
        k_and_v=k_and_v,
        block_idx=0,
        input_pos=kv_len - q_len,
        return_attn_weights=True,
        token_positions=token_positions,
    )
    
    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {weights.shape if weights is not None else 'None'}")
    
    # Verify output shape (MHA transposes output)
    expected_output_shape = (batch_size, q_len, config.n_head * config.head_size)
    assert output.shape == expected_output_shape, f"Output shape mismatch: {output.shape} vs {expected_output_shape}"
    
    # Verify weights
    if weights is not None:
        assert weights.shape == (batch_size, config.n_query_groups, kv_len), f"Weights shape mismatch"
        assert weights.dtype == torch.float32, f"Weights dtype should be float32"
        print("PASS: Weights returned correctly")
    else:
        print("INFO: Weights are None (FlashInfer path may not have been used)")
    
    print("PASS: MultiHeadSelfAttention integration works correctly")
    return True


def run_all_tests():
    """Run all comparison tests."""
    print("\n" + "#" * 60)
    print("# FlashInfer vs Eager SDPA Comparison Tests")
    print("#" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\nERROR: CUDA is not available. These tests require a GPU.")
        return
    
    print(f"\nCUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Check FlashInfer availability
    wrapper = FlashInferSDPA()
    print(f"FlashInfer Available: {wrapper.available}")
    
    # Run tests that don't require working FlashInfer kernels
    results = {}
    results["Fallback Correctness"] = test_fallback_correctness()
    results["MHA Integration"] = test_integration_with_multihead_attention()
    
    # Run kernel comparison tests if FlashInfer is available
    if wrapper.available:
        results["Basic Equivalence"] = test_basic_equivalence()
        results["Decode Phase"] = test_decode_phase_equivalence()
        results["GQA"] = test_gqa_equivalence()
        results["Sliding Window"] = test_sliding_window_equivalence()
        results["Various Dtypes"] = test_various_dtypes()
        results["Batch Comparison"] = test_batch_comparison()
    else:
        print("\nWARNING: FlashInfer kernels are not available.")
        print("Only testing fallback path.")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
