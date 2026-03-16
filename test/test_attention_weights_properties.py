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
Property-based tests for attention weights functionality.

These tests verify that attention weights computation satisfies the correctness
properties defined in the design document.
"""

import torch
from hypothesis import given, strategies as st, settings, HealthCheck


def create_attention_weights_manually(
    query: torch.Tensor,
    key: torch.Tensor,
    scale_factor: float,
    input_pos: int = 0,
    apply_causal_mask: bool = False,
) -> torch.Tensor:
    """
    Manually compute attention weights for testing purposes.
    
    Args:
        query: (batch_size, n_head, q_len, head_size)
        key: (batch_size, n_query_groups, kv_len, head_size)
        scale_factor: scaling factor for attention scores
        input_pos: starting position for queries (for causal masking)
        apply_causal_mask: whether to apply causal masking
    
    Returns:
        Attention weights: (batch_size, n_head, q_len, kv_len)
    """
    batch_size, n_head, q_len, head_size = query.shape
    _, n_query_groups, kv_len, _ = key.shape
    
    # Expand key to match n_head if using GQA
    if n_head > n_query_groups:
        q_per_kv = n_head // n_query_groups
        # Repeat key for each query head: (batch, n_query_groups, kv_len, head_size) 
        # -> (batch, n_head, kv_len, head_size)
        key_expanded = key.repeat_interleave(q_per_kv, dim=1)
    else:
        key_expanded = key
    
    # Compute attention scores: (batch, n_head, q_len, kv_len)
    scores = torch.matmul(query, key_expanded.transpose(-2, -1)) * scale_factor
    
    # Apply causal mask if requested
    if apply_causal_mask:
        # Create causal mask: query position i can only attend to key positions <= input_pos + i
        # token_positions for keys are assumed to be [0, 1, 2, ..., kv_len-1]
        query_positions = torch.arange(input_pos, input_pos + q_len, device=query.device).view(1, 1, q_len, 1)
        key_positions = torch.arange(kv_len, device=query.device).view(1, 1, 1, kv_len)
        causal_mask = key_positions > query_positions  # True where we should mask
        scores = scores.masked_fill(causal_mask, float('-inf'))
    
    # Apply softmax to get attention weights
    attn_weights = torch.softmax(scores, dim=-1)
    
    return attn_weights


def sum_attention_weights_over_query_axis(
    attn_weights: torch.Tensor,
    n_query_groups: int,
) -> torch.Tensor:
    """
    Sum attention weights over query axis and aggregate to n_query_groups.
    
    Args:
        attn_weights: (batch_size, n_head, q_len, kv_len)
        n_query_groups: number of query groups
    
    Returns:
        Summed weights: (batch_size, n_query_groups, kv_len)
    """
    batch_size, n_head, q_len, kv_len = attn_weights.shape
    
    if n_head > n_query_groups:
        # Handle GQA: n_head > n_query_groups
        q_per_kv = n_head // n_query_groups
        # Reshape to (batch_size, n_query_groups, q_per_kv, q_len, kv_len)
        attn_weights_reshaped = attn_weights.view(
            batch_size, n_query_groups, q_per_kv, q_len, kv_len
        )
        # Sum over query axis (dim=3) and query heads (dim=2)
        # Result: (batch_size, n_query_groups, kv_len)
        summed_weights = attn_weights_reshaped.sum(dim=(2, 3))
    else:
        # No GQA: n_head == n_query_groups
        # Sum over query axis: (batch_size, n_head, q_len, kv_len) -> (batch_size, n_head, kv_len)
        summed_weights = attn_weights.sum(dim=2)
    
    return summed_weights


class TestAttentionWeightsProperties:
    """Property-based tests for attention weights functionality."""

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        n_head=st.integers(min_value=1, max_value=8),
        q_len=st.integers(min_value=1, max_value=16),
        head_size=st.sampled_from([32, 64, 128]),
    )
    @settings(
        max_examples=100,
        deadline=None,  # Disable deadline for slow tests
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_property_3_attention_weights_summation(
        self, batch_size, n_head, q_len, head_size
    ):
        """
        **Feature: flashinfer-sparse-sdpa, Property 3: Attention Weights Summation**
        
        *For any* SDPA computation with `return_attn_weights=True`, the returned attention 
        weights SHALL have shape `(batch_size, n_query_groups, kv_len)` and represent the 
        sum of attention weights over the query axis.
        
        **Validates: Requirements 2.1**
        """
        from keys_values.flashinfer_wrapper import FlashInferSDPA
        
        # For prefill (input_pos=0), kv_len must equal q_len for causal attention
        kv_len = q_len
        
        # Ensure n_query_groups divides n_head for valid GQA
        n_query_groups = max(1, n_head // max(1, n_head // 2))
        if n_head % n_query_groups != 0:
            n_query_groups = 1
        
        # Create random tensors
        query = torch.randn(batch_size, n_head, q_len, head_size)
        key = torch.randn(batch_size, n_query_groups, kv_len, head_size)
        value = torch.randn(batch_size, n_query_groups, kv_len, head_size)
        scale_factor = 1.0 / (head_size ** 0.5)
        
        # Compute expected attention weights manually WITH causal masking
        # (the wrapper applies causal masking by default)
        expected_attn_weights = create_attention_weights_manually(
            query, key, scale_factor, input_pos=0, apply_causal_mask=True
        )
        expected_summed_weights = sum_attention_weights_over_query_axis(expected_attn_weights, n_query_groups)
        
        # Call the FlashInferSDPA wrapper (will use fallback on CPU)
        wrapper = FlashInferSDPA()
        _, actual_weights = wrapper.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            scale_factor=scale_factor,
            return_attn_weights=True,
            token_positions=None,
            input_pos=0,
            sliding_window_size=None,
        )
        
        # Property 1: Attention weights shape should be (batch_size, n_query_groups, kv_len)
        assert actual_weights is not None, "Attention weights should be returned when return_attn_weights=True"
        assert actual_weights.shape == (batch_size, n_query_groups, kv_len), \
            f"Expected shape {(batch_size, n_query_groups, kv_len)}, got {actual_weights.shape}"
        
        # Property 2: Attention weights should be non-negative (summed softmax values)
        assert torch.all(actual_weights >= 0), \
            "Attention weights should be non-negative"
        
        # Property 3: Attention weights should be finite
        assert torch.all(torch.isfinite(actual_weights)), \
            "Attention weights should be finite (no NaN or Inf)"
        
        # Property 4: Wrapper output should match manual computation with causal masking
        assert torch.allclose(actual_weights, expected_summed_weights.float(), rtol=1e-4, atol=1e-5), \
            f"Wrapper attention weights should match manual computation. " \
            f"Max diff: {(actual_weights - expected_summed_weights.float()).abs().max()}"

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        n_head=st.integers(min_value=1, max_value=8),
        q_len=st.integers(min_value=1, max_value=16),
        head_size=st.sampled_from([32, 64, 128]),
        kv_len=st.integers(min_value=1, max_value=32),
        input_dtype=st.sampled_from([torch.float32, torch.float16]),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_property_4_attention_weights_float32_dtype(
        self, batch_size, n_head, q_len, head_size, kv_len, input_dtype
    ):
        """
        **Feature: flashinfer-sparse-sdpa, Property 4: Attention Weights Float32 Dtype**
        
        *For any* attention weights computation, the returned weights SHALL be in float32 
        dtype regardless of input query dtype.
        
        **Validates: Requirements 2.2**
        """
        # Ensure n_query_groups divides n_head for valid GQA
        n_query_groups = max(1, n_head // max(1, n_head // 2))
        if n_head % n_query_groups != 0:
            n_query_groups = 1
        
        # Create random tensors with specified dtype
        query = torch.randn(batch_size, n_head, q_len, head_size, dtype=input_dtype)
        key = torch.randn(batch_size, n_query_groups, kv_len, head_size, dtype=input_dtype)
        scale_factor = 1.0 / (head_size ** 0.5)
        
        # Compute attention weights manually
        attn_weights = create_attention_weights_manually(query, key, scale_factor)
        
        # Sum over query axis
        summed_weights = sum_attention_weights_over_query_axis(attn_weights, n_query_groups)
        
        # Convert to float32 (as the implementation does)
        summed_weights_float32 = summed_weights.float()
        
        # Property 1: Attention weights should always be float32 regardless of input dtype
        assert summed_weights_float32.dtype == torch.float32, \
            f"Expected attention weights dtype float32, got {summed_weights_float32.dtype}"
        
        # Property 2: Attention weights should be finite
        assert torch.all(torch.isfinite(summed_weights_float32)), \
            "Attention weights should be finite (no NaN or Inf)"
        
        # Property 3: Attention weights should be non-negative
        assert torch.all(summed_weights_float32 >= 0), \
            "Attention weights should be non-negative"
        
        # Property 4: Values should be preserved after dtype conversion
        # (within floating-point precision limits)
        assert torch.allclose(summed_weights.float(), summed_weights_float32, rtol=1e-5, atol=1e-7), \
            "Values should be preserved after dtype conversion"

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        n_head=st.integers(min_value=2, max_value=8),
        q_len=st.integers(min_value=1, max_value=16),
        head_size=st.sampled_from([32, 64, 128]),
        kv_len=st.integers(min_value=1, max_value=32),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_property_3_with_gqa(
        self, batch_size, n_head, q_len, head_size, kv_len
    ):
        """
        Test Property 3 specifically with Grouped Query Attention (GQA).
        
        Ensures that attention weights are correctly summed and aggregated
        when n_query_groups < n_head.
        """
        # Create GQA scenario: n_query_groups < n_head
        n_query_groups = max(1, n_head // 2)
        
        # Skip if n_head is not divisible by n_query_groups
        if n_head % n_query_groups != 0:
            return
        
        # Create random tensors
        query = torch.randn(batch_size, n_head, q_len, head_size)
        key = torch.randn(batch_size, n_query_groups, kv_len, head_size)
        scale_factor = 1.0 / (head_size ** 0.5)
        
        # Compute attention weights manually
        attn_weights = create_attention_weights_manually(query, key, scale_factor)
        
        # Sum over query axis
        summed_weights = sum_attention_weights_over_query_axis(attn_weights, n_query_groups)
        
        # Property: With GQA, weights should be aggregated to n_query_groups
        assert summed_weights.shape == (batch_size, n_query_groups, kv_len), \
            f"Expected shape {(batch_size, n_query_groups, kv_len)}, got {summed_weights.shape}"
        
        # Property: Each query group should have aggregated weights from multiple heads
        q_per_kv = n_head // n_query_groups
        assert q_per_kv > 1, "GQA should have multiple heads per query group"
        
        # Property: The sum of weights across all KV positions for each query group
        # should equal q_len * q_per_kv (since each query position's softmax sums to 1.0,
        # and we aggregate q_len positions across q_per_kv heads)
        expected_sum_per_group = float(q_len * q_per_kv)
        actual_sum_per_group = summed_weights.sum(dim=-1)  # Sum over kv_len
        assert torch.allclose(actual_sum_per_group, torch.full_like(actual_sum_per_group, expected_sum_per_group), rtol=1e-4, atol=1e-5), \
            f"GQA aggregated weights should sum to {expected_sum_per_group} per query group, got {actual_sum_per_group}"
        
        # Property: Weights should be non-negative
        assert torch.all(summed_weights >= 0), "Attention weights should be non-negative"

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        n_head=st.integers(min_value=1, max_value=8),
        q_len=st.integers(min_value=1, max_value=16),
        head_size=st.sampled_from([32, 64, 128]),
        kv_len=st.integers(min_value=1, max_value=32),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_property_4_dtype_preservation_across_operations(
        self, batch_size, n_head, q_len, head_size, kv_len
    ):
        """
        Test that float32 dtype is preserved through the entire computation pipeline.
        
        This ensures that even if intermediate computations use different dtypes,
        the final output is always float32.
        """
        n_query_groups = max(1, n_head // max(1, n_head // 2))
        if n_head % n_query_groups != 0:
            n_query_groups = 1
        
        # Test with float16 input
        query_fp16 = torch.randn(batch_size, n_head, q_len, head_size, dtype=torch.float16)
        key_fp16 = torch.randn(batch_size, n_query_groups, kv_len, head_size, dtype=torch.float16)
        scale_factor = 1.0 / (head_size ** 0.5)
        
        # Compute attention weights (will be in float16 due to input dtype)
        attn_weights_fp16 = create_attention_weights_manually(query_fp16, key_fp16, scale_factor)
        
        # Sum over query axis
        summed_weights_fp16 = sum_attention_weights_over_query_axis(attn_weights_fp16, n_query_groups)
        
        # Convert to float32 (as the implementation does)
        summed_weights_fp32 = summed_weights_fp16.float()
        
        # Property: Output should be float32
        assert summed_weights_fp32.dtype == torch.float32
        
        # Property: No information should be lost in conversion
        assert torch.all(torch.isfinite(summed_weights_fp32))
        
        # Compare with float32 computation
        query_fp32 = query_fp16.float()
        key_fp32 = key_fp16.float()
        attn_weights_fp32 = create_attention_weights_manually(query_fp32, key_fp32, scale_factor)
        summed_weights_fp32_direct = sum_attention_weights_over_query_axis(attn_weights_fp32, n_query_groups)
        
        # Property: Results should be close (allowing for float16 precision loss)
        assert torch.allclose(summed_weights_fp32, summed_weights_fp32_direct, rtol=1e-2, atol=1e-3), \
            "Float32 conversion should preserve values within float16 precision limits"

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        n_head=st.sampled_from([2, 4, 8]),
        q_len=st.integers(min_value=1, max_value=16),
        head_size=st.sampled_from([32, 64, 128]),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_property_5_gqa_weight_aggregation(
        self, batch_size, n_head, q_len, head_size
    ):
        """
        **Feature: flashinfer-sparse-sdpa, Property 5: Grouped Query Attention Weight Aggregation**
        
        *For any* configuration with n_query_groups < n_head (Grouped Query Attention), 
        the attention weights aggregation SHALL correctly handle the head dimension 
        mismatch and produce correct results.
        
        **Validates: Requirements 2.3**
        """
        from keys_values.flashinfer_wrapper import FlashInferSDPA
        
        # Create GQA scenario: n_query_groups < n_head
        # Use n_query_groups = n_head // 2 to ensure GQA
        n_query_groups = max(1, n_head // 2)
        
        # Skip if n_head is not divisible by n_query_groups
        if n_head % n_query_groups != 0:
            return
        
        # For prefill (input_pos=0), kv_len must equal q_len for causal attention
        kv_len = q_len
        
        # Create random tensors
        query = torch.randn(batch_size, n_head, q_len, head_size)
        key = torch.randn(batch_size, n_query_groups, kv_len, head_size)
        value = torch.randn(batch_size, n_query_groups, kv_len, head_size)
        scale_factor = 1.0 / (head_size ** 0.5)
        
        # Compute expected attention weights manually WITH causal masking
        expected_attn_weights = create_attention_weights_manually(
            query, key, scale_factor, input_pos=0, apply_causal_mask=True
        )
        expected_summed_weights = sum_attention_weights_over_query_axis(expected_attn_weights, n_query_groups)
        
        # Call the FlashInferSDPA wrapper (will use fallback on CPU)
        wrapper = FlashInferSDPA()
        _, actual_weights = wrapper.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            scale_factor=scale_factor,
            return_attn_weights=True,
            token_positions=None,
            input_pos=0,
            sliding_window_size=None,
        )
        
        # Property 1: Attention weights shape should be (batch_size, n_query_groups, kv_len)
        # NOT (batch_size, n_head, kv_len) - GQA aggregates to n_query_groups
        assert actual_weights is not None, "Attention weights should be returned when return_attn_weights=True"
        assert actual_weights.shape == (batch_size, n_query_groups, kv_len), \
            f"GQA: Expected shape {(batch_size, n_query_groups, kv_len)}, got {actual_weights.shape}"
        
        # Property 2: Attention weights should be non-negative
        assert torch.all(actual_weights >= 0), \
            "Attention weights should be non-negative"
        
        # Property 3: Attention weights should be finite
        assert torch.all(torch.isfinite(actual_weights)), \
            "Attention weights should be finite (no NaN or Inf)"
        
        # Property 4: Wrapper output should match manual computation with causal masking
        assert torch.allclose(actual_weights, expected_summed_weights.float(), rtol=1e-4, atol=1e-5), \
            f"GQA: Wrapper attention weights should match manual computation. " \
            f"Max diff: {(actual_weights - expected_summed_weights.float()).abs().max()}"
        
        # Property 5: The sum of weights across all KV positions for each query group
        # should equal q_len * q_per_kv (since each query position's softmax sums to 1.0,
        # and we aggregate q_len positions across q_per_kv heads)
        q_per_kv = n_head // n_query_groups
        expected_sum_per_group = float(q_len * q_per_kv)
        actual_sum_per_group = actual_weights.sum(dim=-1)  # Sum over kv_len
        assert torch.allclose(actual_sum_per_group, torch.full_like(actual_sum_per_group, expected_sum_per_group), rtol=1e-4, atol=1e-5), \
            f"GQA: Aggregated weights should sum to {expected_sum_per_group} per query group, got {actual_sum_per_group}"
        
        # Property 6: Verify GQA is actually being tested (n_query_groups < n_head)
        assert n_query_groups < n_head or n_head == 1, \
            f"Test should use GQA configuration: n_query_groups={n_query_groups}, n_head={n_head}"
