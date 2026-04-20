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
from hypothesis import given, settings, strategies as st, HealthCheck
from typing import Optional, Tuple
from unittest.mock import patch

import pytest
import torch

from litgpt.utils import _RunIf

from keys_values.attention.flashinfer_wrapper import (
    FlashInferSDPA,
    get_flashinfer_sdpa,
)
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.test_utils import (
    random_args_cache_forward,
    range_from_args,
)


@_RunIf(min_cuda_gpus=1)
class TestFlashInferSDPAInitialization:
    """Test FlashInferSDPA class initialization."""

    def test_initialization_creates_instance(self):
        """Test that FlashInferSDPA can be instantiated."""
        wrapper = FlashInferSDPA()
        assert wrapper is not None
        assert isinstance(wrapper, FlashInferSDPA)

    def test_availability_detection_with_flashinfer_available(self):
        """Test availability detection when FlashInfer is available."""
        with patch(
            "keys_values.attention.flashinfer_wrapper.FlashInferSDPA._check_vendored_kernels_available"
        ) as mock_check:
            mock_check.return_value = True
            wrapper = FlashInferSDPA()
            assert wrapper is not None
            assert isinstance(wrapper, FlashInferSDPA)

    def test_availability_detection_without_flashinfer(self):
        """Test availability detection when FlashInfer is not installed."""
        with patch(
            "keys_values.attention.flashinfer_wrapper.FlashInferSDPA._check_vendored_kernels_available"
        ) as mock_check:
            mock_check.return_value = False
            with pytest.raises(AssertionError):
                wrapper = FlashInferSDPA()

    def test_availability_detection_without_cuda(self):
        """Test availability detection when CUDA is not available."""
        with patch("keys_values.attention.flashinfer_ops.is_available", return_value=False):
            with pytest.raises(AssertionError):
                wrapper = FlashInferSDPA()

    def test_check_flashinfer_available_handles_import_error(self):
        """Test that _check_vendored_kernels_available handles ImportError gracefully."""
        with patch(
            "keys_values.attention.flashinfer_ops.is_available",
            side_effect=ImportError("No module named 'flashinfer_ops'"),
        ):
            with pytest.raises(AssertionError):
                wrapper = FlashInferSDPA()

    def test_check_flashinfer_available_handles_generic_exception(self):
        """Test that _check_vendored_kernels_available handles generic exceptions gracefully."""
        with patch(
            "keys_values.attention.flashinfer_ops.is_available",
            side_effect=RuntimeError("Some error"),
        ):
            with pytest.raises(AssertionError):
                wrapper = FlashInferSDPA()

    def test_global_instance_creation(self):
        """Test that get_flashinfer_sdpa returns a singleton instance."""
        instance1 = get_flashinfer_sdpa()
        instance2 = get_flashinfer_sdpa()
        assert instance1 is instance2

    def test_global_instance_is_flashinfer_sdpa(self):
        """Test that global instance is of correct type."""
        instance = get_flashinfer_sdpa()
        assert isinstance(instance, FlashInferSDPA)


def call_sdpa_with_random_args(
    wrapper: FlashInferSDPA,
    kv_len: int = 16,
    q_len: int = 8,
    input_pos: int = 4,
    token_positions: Optional[torch.Tensor] = None,
    max_batch_size: int = 2,
    n_query_groups: int = 2,
    cache_length: int = 16,
    head_size: int = 64,
    n_head: int = 4,
    dtype: torch.dtype = torch.float16,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    params = KVCacheParams(
        max_batch_size=max_batch_size,
        n_query_groups=n_query_groups,
        cache_length=cache_length,
        head_size=head_size,
        n_head=n_head,
        dtype=dtype,
    )
    data = random_args_cache_forward(
        params,
        kv_len,
        vocab_size=None,
        device=torch.device("cuda", 0),
    )

    return wrapper.scaled_dot_product_attention(
        **range_from_args(data, 0, q_len, only_query=True),
        scale_factor=None,
        input_pos=input_pos,
        token_positions=token_positions,
        **kwargs,
    )


@_RunIf(min_cuda_gpus=1)
class TestFlashInferSDPAInterface:
    """Test FlashInferSDPA interface and method signatures."""

    def test_scaled_dot_product_attention_method_exists(self):
        """Test that scaled_dot_product_attention method exists."""
        wrapper = FlashInferSDPA()
        assert hasattr(wrapper, "scaled_dot_product_attention")
        assert callable(wrapper.scaled_dot_product_attention)

    def test_scaled_dot_product_attention_accepts_required_parameters(self):
        """Test that scaled_dot_product_attention accepts required parameters."""
        wrapper = FlashInferSDPA()

        # This should not raise an exception for parameter acceptance
        # (it will raise NotImplementedError for the actual computation)
        try:
            call_sdpa_with_random_args(wrapper)
        except NotImplementedError:
            # Expected since fallback is not yet implemented
            pass

    def test_scaled_dot_product_attention_accepts_optional_parameters(self):
        """Test that scaled_dot_product_attention accepts optional parameters."""
        wrapper = FlashInferSDPA()
        max_batch_size = 2
        n_query_groups = 2
        kv_len = 16
        token_positions = (
            torch.arange(kv_len)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(max_batch_size, n_query_groups, -1)
        )

        # This should not raise an exception for parameter acceptance
        try:
            call_sdpa_with_random_args(
                wrapper,
                token_positions=token_positions,
                return_attn_weights=True,
                sort_if_3d=True,
                output_transposed=False,
            )
        except NotImplementedError:
            # Expected since fallback is not yet implemented
            pass


@_RunIf(min_cuda_gpus=1)
class TestFlashInferKernelWrapping:
    """Test FlashInfer kernel wrapping interface."""

    # HIER: Shape returned by mock is wrong!
    def test_flashinfer_sdpa_routes_to_chunk_processing(self):
        """Test that scaled_dot_product_attention routes to chunk processing for decode phase."""
        with patch.object(
            FlashInferSDPA, "_check_vendored_kernels_available", return_value=True
        ):
            attn_outputs = torch.zeros(
                (2, 4, 64), dtype=torch.float16, device=torch.device("cuda", 0),
            )
            with patch.object(
                FlashInferSDPA,
                "_flashinfer_sdpa_chunk_processing",
                return_value=(attn_outputs, None),
            ) as mock_chunk:
                wrapper = FlashInferSDPA()

                call_sdpa_with_random_args(wrapper, q_len=1)
                mock_chunk.assert_called_once()

    def test_flashinfer_sdpa_routes_nonsquare_no_weights_to_standard(self):
        """Test that non-square attention without weights routes to standard prefill."""
        with patch.object(
            FlashInferSDPA, "_check_vendored_kernels_available", return_value=True
        ):
            attn_outputs = torch.zeros(
                (2, 1, 4, 64), dtype=torch.float16, device=torch.device("cuda", 0),
            )
            with patch.object(
                FlashInferSDPA,
                "_flashinfer_sdpa_standard",
                return_value=attn_outputs,
            ) as mock_standard:
                wrapper = FlashInferSDPA()

                call_sdpa_with_random_args(
                    wrapper,
                    q_len=2048,
                    kv_len=32768,
                )
                mock_standard.assert_called_once()

    def test_flashinfer_sdpa_routes_nonsquare_with_weights_to_fused_prefill(self):
        """Test that non-square attention with weights routes to fused_prefill."""
        with patch.object(
            FlashInferSDPA, "_check_vendored_kernels_available", return_value=True
        ):
            q_len = 2048
            kv_len = 32768
            device = torch.device("cuda", 0)
            attn_outputs = torch.zeros(
                (2, q_len, 4, 64), dtype=torch.float16, device=device,
            )
            attn_weights = torch.zeros(
                (2, 2, kv_len), dtype=torch.float32, device=device,
            )
            with patch.object(
                FlashInferSDPA,
                "_flashinfer_sdpa_fused_prefill",
                return_value=(attn_outputs, attn_weights),
            ) as mock_fallback:
                wrapper = FlashInferSDPA()

                call_sdpa_with_random_args(
                    wrapper,
                    q_len=q_len,
                    kv_len=kv_len,
                    return_attn_weights=True,
                )
                mock_fallback.assert_called_once()

    def test_parameter_translation_validates_shapes(self):
        """Test that parameter translation validates input shapes."""
        with patch.object(
            FlashInferSDPA, "_check_vendored_kernels_available", return_value=True
        ):
            wrapper = FlashInferSDPA()

            params = KVCacheParams(
                max_batch_size=2,
                n_query_groups=2,
                cache_length=16,
                head_size=64,
                n_head=4,
                dtype=torch.float16,
            )
            q_len = 8
            kv_len = 16
            data = random_args_cache_forward(
                params,
                kv_len,
                vocab_size=None,
                device=torch.device("cuda", 0),
            )

            # Should raise AssertionError due to head size mismatch
            with pytest.raises(ValueError):
                wrapper.scaled_dot_product_attention(
                    query=data["query"][:, :, :q_len, :],
                    key=data["key"],
                    value=data["value"][:, :, :, :(params.head_size // 2)],
                    scale_factor=None,
                    input_pos=4,
                    token_positions=None,
                )

    def test_parameter_translation_validates_gqa_divisibility(self):
        """Test that parameter translation validates GQA divisibility."""
        with patch.object(
            FlashInferSDPA, "_check_vendored_kernels_available", return_value=True
        ):
            wrapper = FlashInferSDPA()

            # Should raise AssertionError due to n_head not divisible by n_query_groups
            with pytest.raises(ValueError):
                call_sdpa_with_random_args(
                    wrapper,
                    n_head=5,
                    n_query_groups=2,
                )


@_RunIf(min_cuda_gpus=1)
class TestAttentionWeightsReturn:
    """Test attention weights return functionality."""

    def test_attention_weights_shape_without_gqa(self):
        """Test that attention weights have correct shape without GQA."""
        wrapper = FlashInferSDPA()

        batch_size = 2
        n_query_groups = 4
        n_head = 4
        kv_len = 16
        _, attn_weights = call_sdpa_with_random_args(
            wrapper,
            max_batch_size=batch_size,
            n_query_groups=n_query_groups,
            n_head=n_head,
            kv_len=kv_len,
            return_attn_weights=True,
        )
        # Weights should be summed over query axis: (batch_size, n_query_groups, kv_len)
        assert attn_weights.shape == (batch_size, n_query_groups, kv_len)

    def test_attention_weights_shape_with_gqa(self):
        """Test that attention weights have correct shape with GQA."""
        wrapper = FlashInferSDPA()

        batch_size = 2
        n_query_groups = 2
        n_head = 4
        kv_len = 16
        _, attn_weights = call_sdpa_with_random_args(
            wrapper,
            max_batch_size=batch_size,
            n_query_groups=n_query_groups,
            n_head=n_head,
            kv_len=kv_len,
            return_attn_weights=True,
        )
        # Weights should be aggregated to n_query_groups: (batch_size, n_query_groups, kv_len)
        assert attn_weights.shape == (batch_size, n_query_groups, kv_len)
        # Attention weights are returned in float32 dtype
        assert attn_weights.dtype == torch.float32
        # Each weight should be the sum of q_len softmax values
        # With causal masking, some positions may have 0 weight (future positions)
        # So we just check that weights are non-negative and some are positive
        assert torch.all(attn_weights >= 0)
        assert torch.any(attn_weights > 0)

    def test_attention_weights_are_valid_probabilities(self):
        """Test that attention weights are valid probability distributions."""
        wrapper = FlashInferSDPA()

        batch_size = 2
        n_query_groups = 2
        q_len = 8
        _, attn_weights = call_sdpa_with_random_args(
            wrapper,
            q_len = q_len,
            n_query_groups=n_query_groups,
            max_batch_size=batch_size,
            return_attn_weights=True,
        )

        # Weights should be non-negative (they're summed softmax values)
        assert torch.all(attn_weights >= 0)
        # Entries must sum to `q_len`
        must_be = torch.tensor(
            q_len, device=attn_weights.device, dtype=attn_weights.dtype,
        ).view(1, 1).expand(batch_size, n_query_groups)
        sum_over_kv = attn_weights.sum(dim=-1)
        torch.testing.assert_close(
            sum_over_kv, must_be, atol=0.0003, rtol=5e-5,
        )

    def test_attention_weights_none_when_not_requested(self):
        """Test that attention weights are None when not requested."""
        wrapper = FlashInferSDPA()

        batch_size = 2
        n_head = 4
        q_len = 8
        head_size = 64
        attn_output, attn_weights = call_sdpa_with_random_args(
            wrapper,
            max_batch_size=batch_size,
            n_head=n_head,
            q_len=q_len,
            head_size=head_size,
        )
        assert attn_weights is None
        assert attn_output.shape == (batch_size, n_head, q_len, head_size)


@_RunIf(min_cuda_gpus=1)
class TestAttentionWeightsProperties:
    """Property-based tests for attention weights functionality."""

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        n_head=st.integers(min_value=1, max_value=8),
        q_len=st.integers(min_value=1, max_value=16),
        head_size=st.sampled_from([32, 64, 128]),
        kv_len=st.integers(min_value=4, max_value=32),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        deadline=None,
    )
    def test_property_3_attention_weights_summation(
        self, batch_size, n_head, q_len, head_size, kv_len
    ):
        """
        **Feature: flashinfer-sparse-sdpa, Property 3: Attention Weights Summation**

        *For any* SDPA computation with `return_attn_weights=True`, the returned attention
        weights SHALL have shape `(batch_size, n_query_groups, kv_len)` and represent the
        sum of attention weights over the query axis.

        **Validates: Requirements 2.1**
        """
        # Ensure n_query_groups divides n_head for valid GQA
        n_query_groups = max(1, n_head // max(1, n_head // 2))
        if n_head % n_query_groups != 0:
            n_query_groups = 1
        if q_len >= kv_len:
            q_len = kv_len // 2

        wrapper = FlashInferSDPA()

        attn_output, attn_weights = call_sdpa_with_random_args(
            wrapper,
            max_batch_size=batch_size,
            n_head=n_head,
            n_query_groups=n_query_groups,
            q_len=q_len,
            kv_len=kv_len,
            head_size=head_size,
            return_attn_weights=True,
        )

        # Property: Attention weights shape should be (batch_size, n_query_groups, kv_len)
        assert attn_weights.shape == (
            batch_size,
            n_query_groups,
            kv_len,
        ), f"Expected shape {(batch_size, n_query_groups, kv_len)}, got {attn_weights.shape}"

        # Property: Attention weights should be non-negative (summed softmax values)
        assert torch.all(attn_weights >= 0), "Attention weights should be non-negative"
        # Entries must sum to `q_len`
        must_be = torch.tensor(
            q_len, device=attn_weights.device, dtype=attn_weights.dtype,
        ).view(1, 1).expand(batch_size, n_query_groups)
        sum_over_kv = attn_weights.sum(dim=-1)
        torch.testing.assert_close(
            sum_over_kv, must_be, atol=0.0003, rtol=5e-5,
        )

    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        n_head=st.integers(min_value=1, max_value=8),
        q_len=st.integers(min_value=1, max_value=16),
        head_size=st.sampled_from([32, 64, 128]),
        kv_len=st.integers(min_value=4, max_value=32),
        input_dtype=st.sampled_from([torch.bfloat16, torch.float16]),
    )
    @settings(
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
        deadline=None,
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
        if q_len >= kv_len:
            q_len = kv_len // 2

        wrapper = FlashInferSDPA()

        attn_output, attn_weights = call_sdpa_with_random_args(
            wrapper,
            max_batch_size=batch_size,
            n_head=n_head,
            n_query_groups=n_query_groups,
            q_len=q_len,
            kv_len=kv_len,
            head_size=head_size,
            return_attn_weights=True,
        )

        # Property: Attention weights should always be float32 regardless of input dtype
        assert (
            attn_weights.dtype == torch.float32
        ), f"Expected attention weights dtype float32, got {attn_weights.dtype}"

        # Property: Attention weights should be non-negative (summed softmax values)
        assert torch.all(attn_weights >= 0), "Attention weights should be non-negative"
        # Entries must sum to `q_len`
        must_be = torch.tensor(
            q_len, device=attn_weights.device, dtype=attn_weights.dtype,
        ).view(1, 1).expand(batch_size, n_query_groups)
        sum_over_kv = attn_weights.sum(dim=-1)
        torch.testing.assert_close(
            sum_over_kv, must_be, atol=0.0003, rtol=5e-5,
        )


@_RunIf(min_cuda_gpus=1)
@pytest.mark.skip("To be fixed")
class TestChunkProcessingForLongSequences:
    """Test chunk processing for long sequences (Requirement 4.1, 4.2)."""

    def test_chunk_processing_with_chunk_size_parameter(self):
        """Test that chunk_size parameter is accepted and used."""
        wrapper = FlashInferSDPA()

        batch_size, n_head, q_len, head_size = 2, 4, 32, 64
        n_query_groups = 2
        kv_len = 32
        chunk_size = 8  # Process in chunks of 8

        query = torch.randn(batch_size, n_head, q_len, head_size, device="cpu")
        key = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        value = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        scale_factor = 1.0 / (head_size**0.5)

        # Should work with chunk_size parameter
        attn_output, attn_weights = wrapper._fallback_sdpa(
            query,
            key,
            value,
            scale_factor,
            return_attn_weights=True,
            chunk_size=chunk_size,
        )

        # Verify output shapes
        assert attn_output.shape == (batch_size, n_head, q_len, head_size)
        assert attn_weights.shape == (batch_size, n_query_groups, kv_len)

    def test_chunk_processing_consistency_with_full_processing(self):
        """Test that chunked processing produces same results as full processing."""
        wrapper = FlashInferSDPA()

        batch_size, n_head, q_len, head_size = 1, 2, 16, 32
        n_query_groups = 2
        kv_len = 16

        # Use same random seed for reproducibility
        torch.manual_seed(42)
        query = torch.randn(batch_size, n_head, q_len, head_size, device="cpu")
        key = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        value = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        scale_factor = 1.0 / (head_size**0.5)

        # Process without chunking (chunk_size=None means automatic)
        output_full, weights_full = wrapper._fallback_sdpa(
            query, key, value, scale_factor, return_attn_weights=True, chunk_size=None
        )

        # Process with explicit chunking
        output_chunked, weights_chunked = wrapper._fallback_sdpa(
            query,
            key,
            value,
            scale_factor,
            return_attn_weights=True,
            chunk_size=4,  # Small chunk size
        )

        # Results should be numerically equivalent
        assert torch.allclose(
            output_full, output_chunked, rtol=1e-4, atol=1e-6
        ), "Chunked output should match full output"
        assert torch.allclose(
            weights_full, weights_chunked, rtol=1e-4, atol=1e-6
        ), "Chunked weights should match full weights"

    def test_chunk_processing_causal_masking_across_chunks(self):
        """Test that causal masking is correctly applied across chunk boundaries."""
        wrapper = FlashInferSDPA()

        batch_size, n_head, q_len, head_size = 1, 2, 8, 32
        n_query_groups = 2
        kv_len = 8
        chunk_size = 4

        query = torch.randn(batch_size, n_head, q_len, head_size, device="cpu")
        key = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        value = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        scale_factor = 1.0 / (head_size**0.5)

        # Process with chunking
        output, weights = wrapper._fallback_sdpa(
            query,
            key,
            value,
            scale_factor,
            return_attn_weights=True,
            input_pos=0,
            chunk_size=chunk_size,
        )

        # Verify output shape
        assert output.shape == (batch_size, n_head, q_len, head_size)

        # Verify weights are valid (non-negative, finite)
        assert torch.all(weights >= 0), "Weights should be non-negative"
        assert torch.all(torch.isfinite(weights)), "Weights should be finite"

    def test_chunk_processing_weight_accumulation(self):
        """Test that attention weights are correctly accumulated across chunks."""
        wrapper = FlashInferSDPA()

        batch_size, n_head, q_len, head_size = 1, 2, 12, 32
        n_query_groups = 2
        kv_len = 12
        chunk_size = 4  # 3 chunks

        query = torch.randn(batch_size, n_head, q_len, head_size, device="cpu")
        key = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        value = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        scale_factor = 1.0 / (head_size**0.5)

        output, weights = wrapper._fallback_sdpa(
            query,
            key,
            value,
            scale_factor,
            return_attn_weights=True,
            chunk_size=chunk_size,
        )

        # Weights should be accumulated across all chunks
        # Shape should be (batch_size, n_query_groups, kv_len)
        assert weights.shape == (batch_size, n_query_groups, kv_len)

        # Weights should be non-negative (sum of softmax values)
        assert torch.all(weights >= 0)

        # Some weights should be positive (not all masked)
        assert torch.any(weights > 0)

    def test_chunk_processing_with_token_positions(self):
        """Test chunk processing with custom token positions."""
        wrapper = FlashInferSDPA()

        batch_size, n_head, q_len, head_size = 1, 2, 8, 32
        n_query_groups = 2
        kv_len = 8
        chunk_size = 4

        query = torch.randn(batch_size, n_head, q_len, head_size, device="cpu")
        key = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        value = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        token_positions = (
            torch.arange(kv_len, device="cpu")
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, n_query_groups, -1)
        )
        scale_factor = 1.0 / (head_size**0.5)

        output, weights = wrapper._fallback_sdpa(
            query,
            key,
            value,
            scale_factor,
            return_attn_weights=True,
            token_positions=token_positions,
            input_pos=0,
            chunk_size=chunk_size,
        )

        # Verify output shape
        assert output.shape == (batch_size, n_head, q_len, head_size)
        assert weights.shape == (batch_size, n_query_groups, kv_len)

    def test_chunk_processing_with_sliding_window(self):
        """Test chunk processing with sliding window attention."""
        wrapper = FlashInferSDPA()

        batch_size, n_head, q_len, head_size = 1, 2, 16, 32
        n_query_groups = 2
        kv_len = 16
        chunk_size = 4
        sliding_window_size = 4

        query = torch.randn(batch_size, n_head, q_len, head_size, device="cpu")
        key = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        value = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        scale_factor = 1.0 / (head_size**0.5)

        output, weights = wrapper._fallback_sdpa(
            query,
            key,
            value,
            scale_factor,
            return_attn_weights=True,
            sliding_window_size=sliding_window_size,
            chunk_size=chunk_size,
        )

        # Verify output shape
        assert output.shape == (batch_size, n_head, q_len, head_size)
        assert weights.shape == (batch_size, n_query_groups, kv_len)

        # Weights should be valid
        assert torch.all(weights >= 0)
        assert torch.all(torch.isfinite(weights))

    def test_scaled_dot_product_attention_accepts_chunk_size(self):
        """Test that the main interface accepts chunk_size parameter."""
        wrapper = FlashInferSDPA()

        batch_size, n_head, q_len, head_size = 2, 4, 16, 64
        n_query_groups = 2
        kv_len = 16

        query = torch.randn(batch_size, n_head, q_len, head_size, device="cpu")
        key = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        value = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        scale_factor = 1.0 / (head_size**0.5)

        # Should accept chunk_size parameter without error
        output, weights = wrapper.scaled_dot_product_attention(
            query, key, value, scale_factor, return_attn_weights=True, chunk_size=4
        )

        assert output.shape == (batch_size, n_head, q_len, head_size)
        assert weights.shape == (batch_size, n_query_groups, kv_len)

    def test_chunk_size_larger_than_query_length(self):
        """Test that chunk_size larger than q_len works correctly (no chunking needed)."""
        wrapper = FlashInferSDPA()

        batch_size, n_head, q_len, head_size = 1, 2, 8, 32
        n_query_groups = 2
        kv_len = 8
        chunk_size = 16  # Larger than q_len

        query = torch.randn(batch_size, n_head, q_len, head_size, device="cpu")
        key = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        value = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        scale_factor = 1.0 / (head_size**0.5)

        output, weights = wrapper._fallback_sdpa(
            query,
            key,
            value,
            scale_factor,
            return_attn_weights=True,
            chunk_size=chunk_size,
        )

        # Should still work correctly
        assert output.shape == (batch_size, n_head, q_len, head_size)
        assert weights.shape == (batch_size, n_query_groups, kv_len)

    def test_chunk_processing_without_weights(self):
        """Test chunk processing when attention weights are not requested."""
        wrapper = FlashInferSDPA()

        batch_size, n_head, q_len, head_size = 2, 4, 16, 64
        n_query_groups = 2
        kv_len = 16
        chunk_size = 4

        query = torch.randn(batch_size, n_head, q_len, head_size, device="cpu")
        key = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        value = torch.randn(batch_size, n_query_groups, kv_len, head_size, device="cpu")
        scale_factor = 1.0 / (head_size**0.5)

        output, weights = wrapper._fallback_sdpa(
            query,
            key,
            value,
            scale_factor,
            return_attn_weights=False,
            chunk_size=chunk_size,
        )

        assert output.shape == (batch_size, n_head, q_len, head_size)
        assert weights is None


@_RunIf(min_cuda_gpus=1)
@pytest.mark.skip("To be fixed")
class TestBackendEquivalenceVerification:
    """Test backend equivalence verification utilities (Requirement 1.3)."""

    def test_backend_equivalence_result_initialization(self):
        """Test BackendEquivalenceResult initialization."""
        from keys_values.attention.flashinfer_verification import BackendEquivalenceResult

        result = BackendEquivalenceResult(
            is_equivalent=True,
            output_max_diff=1e-5,
            output_mean_diff=1e-6,
            weights_max_diff=1e-5,
            weights_mean_diff=1e-6,
            rtol=1e-4,
            atol=1e-6,
            message="Test message",
        )

        assert result.is_equivalent is True
        assert result.output_max_diff == 1e-5
        assert result.output_mean_diff == 1e-6
        assert result.weights_max_diff == 1e-5
        assert result.weights_mean_diff == 1e-6
        assert result.rtol == 1e-4
        assert result.atol == 1e-6
        assert result.message == "Test message"

    def test_backend_equivalence_result_bool_conversion(self):
        """Test BackendEquivalenceResult boolean conversion."""
        from keys_values.attention.flashinfer_verification import BackendEquivalenceResult

        # Equivalent result should be truthy
        result_true = BackendEquivalenceResult(
            is_equivalent=True,
            output_max_diff=1e-5,
            output_mean_diff=1e-6,
        )
        assert bool(result_true) is True

        # Non-equivalent result should be falsy
        result_false = BackendEquivalenceResult(
            is_equivalent=False,
            output_max_diff=1.0,
            output_mean_diff=0.5,
        )
        assert bool(result_false) is False

    def test_backend_equivalence_result_repr(self):
        """Test BackendEquivalenceResult string representation."""
        from keys_values.attention.flashinfer_verification import BackendEquivalenceResult

        result = BackendEquivalenceResult(
            is_equivalent=True,
            output_max_diff=1e-5,
            output_mean_diff=1e-6,
            weights_max_diff=1e-5,
            weights_mean_diff=1e-6,
        )

        repr_str = repr(result)
        assert "BackendEquivalenceResult" in repr_str
        assert "is_equivalent=True" in repr_str
        assert "output_max_diff" in repr_str

    def test_check_numerical_equivalence_identical_tensors(self):
        """Test check_numerical_equivalence with identical tensors."""
        from keys_values.attention.flashinfer_verification import check_numerical_equivalence

        tensor = torch.randn(2, 4, 8, 64)
        is_equiv, max_diff, mean_diff = check_numerical_equivalence(tensor, tensor)

        assert is_equiv is True
        assert max_diff == 0.0
        assert mean_diff == 0.0

    def test_check_numerical_equivalence_within_tolerance(self):
        """Test check_numerical_equivalence with tensors within tolerance."""
        from keys_values.attention.flashinfer_verification import check_numerical_equivalence

        tensor_a = torch.randn(2, 4, 8, 64)
        # Add small noise within tolerance
        tensor_b = tensor_a + torch.randn_like(tensor_a) * 1e-7

        is_equiv, max_diff, mean_diff = check_numerical_equivalence(
            tensor_a, tensor_b, rtol=1e-4, atol=1e-6
        )

        assert is_equiv is True
        assert max_diff < 1e-5
        assert mean_diff < 1e-6

    def test_check_numerical_equivalence_outside_tolerance(self):
        """Test check_numerical_equivalence with tensors outside tolerance."""
        from keys_values.attention.flashinfer_verification import check_numerical_equivalence

        tensor_a = torch.randn(2, 4, 8, 64)
        # Add large noise outside tolerance
        tensor_b = tensor_a + torch.ones_like(tensor_a) * 0.1

        is_equiv, max_diff, mean_diff = check_numerical_equivalence(
            tensor_a, tensor_b, rtol=1e-4, atol=1e-6
        )

        assert is_equiv is False
        assert max_diff > 0.05
        assert mean_diff > 0.05

    def test_check_numerical_equivalence_shape_mismatch(self):
        """Test check_numerical_equivalence raises error on shape mismatch."""
        from keys_values.attention.flashinfer_verification import check_numerical_equivalence

        tensor_a = torch.randn(2, 4, 8, 64)
        tensor_b = torch.randn(2, 4, 16, 64)  # Different shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            check_numerical_equivalence(tensor_a, tensor_b)

    def test_verify_backend_equivalence_raises_when_unavailable(self):
        """Test verify_backend_equivalence raises error when kernels unavailable."""
        from keys_values.attention.flashinfer_verification import verify_backend_equivalence

        with patch.object(
            FlashInferSDPA, "_check_vendored_kernels_available", return_value=False
        ):
            # Reset the global instance to pick up the mock
            import keys_values.attention.flashinfer_wrapper as wrapper_module

            wrapper_module._flashinfer_sdpa_instance = None

            query = torch.randn(2, 4, 8, 64)
            key = torch.randn(2, 2, 16, 64)
            value = torch.randn(2, 2, 16, 64)

            with pytest.raises(
                RuntimeError, match="vendored kernels are not available"
            ):
                verify_backend_equivalence(query, key, value, scale_factor=0.125)

            # Reset for other tests
            wrapper_module._flashinfer_sdpa_instance = None

    def test_verify_backend_equivalence_handles_kernel_error(self):
        """Test verify_backend_equivalence handles kernel computation errors."""
        from keys_values.attention.flashinfer_verification import verify_backend_equivalence

        with patch.object(
            FlashInferSDPA, "_check_vendored_kernels_available", return_value=True
        ):
            with patch.object(
                FlashInferSDPA,
                "scaled_dot_product_attention",
                side_effect=RuntimeError("Kernel error"),
            ):
                # Reset the global instance
                import keys_values.attention.flashinfer_wrapper as wrapper_module

                wrapper_module._flashinfer_sdpa_instance = None

                wrapper = get_flashinfer_sdpa()
                wrapper.available = True

                query = torch.randn(2, 4, 8, 64)
                key = torch.randn(2, 2, 16, 64)
                value = torch.randn(2, 2, 16, 64)

                result = verify_backend_equivalence(
                    query, key, value, scale_factor=0.125, log_results=False
                )

                assert result.is_equivalent is False
                assert "Kernel error" in result.message

                # Reset for other tests
                wrapper_module._flashinfer_sdpa_instance = None

    def test_verify_backend_equivalence_batch_empty_list(self):
        """Test verify_backend_equivalence_batch with empty list."""
        from keys_values.attention.flashinfer_verification import verify_backend_equivalence_batch

        passed, failed, results = verify_backend_equivalence_batch(
            [], log_results=False
        )

        assert passed == 0
        assert failed == 0
        assert len(results) == 0

    def test_verify_backend_equivalence_batch_handles_exceptions(self):
        """Test verify_backend_equivalence_batch handles exceptions in test cases."""
        from keys_values.attention.flashinfer_verification import verify_backend_equivalence_batch

        # Test case with missing required key
        test_cases = [
            {
                "query": torch.randn(2, 4, 8, 64),
                "key": torch.randn(2, 2, 16, 64),
                # Missing 'value' and 'scale_factor'
            }
        ]

        passed, failed, results = verify_backend_equivalence_batch(
            test_cases, log_results=False
        )

        assert passed == 0
        assert failed == 1
        assert len(results) == 1
        assert results[0].is_equivalent is False


@_RunIf(min_cuda_gpus=1)
@pytest.mark.skip("To be fixed")
class TestBackendEquivalenceVerificationIntegration:
    """Integration tests for backend equivalence verification."""

    def test_fallback_produces_consistent_results(self):
        """Test that fallback SDPA produces consistent results across calls."""
        wrapper = FlashInferSDPA()

        batch_size, n_head, q_len, head_size = 2, 4, 8, 64
        n_query_groups = 2
        kv_len = 16

        # Use fixed seed for reproducibility
        torch.manual_seed(42)
        query = torch.randn(batch_size, n_head, q_len, head_size)
        key = torch.randn(batch_size, n_query_groups, kv_len, head_size)
        value = torch.randn(batch_size, n_query_groups, kv_len, head_size)
        scale_factor = 1.0 / (head_size**0.5)

        # Call fallback twice
        output1, weights1 = wrapper._fallback_sdpa(
            query, key, value, scale_factor, return_attn_weights=True
        )
        output2, weights2 = wrapper._fallback_sdpa(
            query, key, value, scale_factor, return_attn_weights=True
        )

        # Results should be identical
        assert torch.allclose(output1, output2)
        assert torch.allclose(weights1, weights2)

    def test_numerical_tolerance_checking_with_different_tolerances(self):
        """Test numerical tolerance checking with various tolerance levels."""
        from keys_values.attention.flashinfer_verification import check_numerical_equivalence

        tensor_a = torch.randn(2, 4, 8, 64)
        # Add noise of known magnitude
        noise = torch.randn_like(tensor_a) * 1e-5
        tensor_b = tensor_a + noise

        # Should pass with loose tolerance
        is_equiv_loose, _, _ = check_numerical_equivalence(
            tensor_a, tensor_b, rtol=1e-3, atol=1e-4
        )
        assert is_equiv_loose is True

        # May fail with tight tolerance
        is_equiv_tight, max_diff, _ = check_numerical_equivalence(
            tensor_a, tensor_b, rtol=1e-8, atol=1e-9
        )
        # The result depends on the actual noise magnitude
        # Just verify the function runs and returns sensible values
        assert isinstance(is_equiv_tight, bool)
        assert max_diff >= 0

    def test_equivalence_result_message_contains_useful_info(self):
        """Test that equivalence result message contains useful debugging info."""
        from keys_values.attention.flashinfer_verification import BackendEquivalenceResult

        # Test failure message
        result_fail = BackendEquivalenceResult(
            is_equivalent=False,
            output_max_diff=0.1,
            output_mean_diff=0.05,
            rtol=1e-4,
            atol=1e-6,
            message="Backend equivalence FAILED: output differs (max_diff=1.00e-01)",
        )

        assert "FAILED" in result_fail.message
        assert "max_diff" in result_fail.message

        # Test success message
        result_pass = BackendEquivalenceResult(
            is_equivalent=True,
            output_max_diff=1e-6,
            output_mean_diff=1e-7,
            rtol=1e-4,
            atol=1e-6,
            message="Backend equivalence verified: output_max_diff=1.00e-06",
        )

        assert "verified" in result_pass.message


@_RunIf(min_cuda_gpus=1)
@pytest.mark.skip("To be fixed")
class TestTwoPhaseWeightAccumulation:
    """Tests for two-phase attention weight accumulation (FlashInfer O+LSE, then PyTorch weights)."""

    def test_fused_prefill_routing_with_fp16(self):
        """Test that non-square attention with weights + fp16 + input_pos>0 routes to fused prefill."""
        with patch.object(
            FlashInferSDPA, "_check_vendored_kernels_available", return_value=True
        ):
            with patch.object(
                FlashInferSDPA,
                "_flashinfer_sdpa_fused_prefill",
                return_value=(torch.tensor([]), None),
            ) as mock_fused:
                wrapper = FlashInferSDPA(use_fused_prefill=True)
                wrapper.available = True

                batch_size, n_head, q_len, head_size = 2, 4, 2048, 64
                n_query_groups = 2
                kv_len = 32768

                # fp16 + input_pos>0 triggers fused prefill path (Triton score-sum)
                query = torch.randn(
                    batch_size, n_head, q_len, head_size, dtype=torch.float16
                )
                key = torch.randn(
                    batch_size, n_query_groups, kv_len, head_size, dtype=torch.float16
                )
                value = torch.randn(
                    batch_size, n_query_groups, kv_len, head_size, dtype=torch.float16
                )
                scale_factor = 1.0 / (head_size**0.5)

                wrapper.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    scale_factor,
                    return_attn_weights=True,
                    input_pos=q_len,
                )
                mock_fused.assert_called_once()

    def test_fused_prefill_routing_with_bf16(self):
        """Test that non-square attention with weights + bf16 + input_pos>0 routes to fused prefill."""
        with patch.object(
            FlashInferSDPA, "_check_vendored_kernels_available", return_value=True
        ):
            with patch.object(
                FlashInferSDPA,
                "_flashinfer_sdpa_fused_prefill",
                return_value=(torch.tensor([]), None),
            ) as mock_fused:
                wrapper = FlashInferSDPA(use_fused_prefill=True)
                wrapper.available = True

                batch_size, n_head, q_len, head_size = 2, 4, 2048, 128
                n_query_groups = 2
                kv_len = 32768

                query = torch.randn(
                    batch_size, n_head, q_len, head_size, dtype=torch.bfloat16
                )
                key = torch.randn(
                    batch_size, n_query_groups, kv_len, head_size, dtype=torch.bfloat16
                )
                value = torch.randn(
                    batch_size, n_query_groups, kv_len, head_size, dtype=torch.bfloat16
                )
                scale_factor = 1.0 / (head_size**0.5)

                wrapper.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    scale_factor,
                    return_attn_weights=True,
                    input_pos=q_len,
                )
                mock_fused.assert_called_once()

    def test_fused_prefill_fallback_at_input_pos_zero(self):
        """Test that fp16 with input_pos=0 falls back (fused prefill requires input_pos>0)."""
        with patch.object(
            FlashInferSDPA, "_check_vendored_kernels_available", return_value=True
        ):
            with patch.object(
                FlashInferSDPA,
                "_flashinfer_sdpa_two_phase_weights",
                return_value=(torch.tensor([]), None),
            ) as mock_two_phase:
                wrapper = FlashInferSDPA(use_fused_prefill=True)
                wrapper.available = True

                batch_size, n_head, q_len, head_size = 2, 4, 16, 64
                n_query_groups = 2
                kv_len = 32

                query = torch.randn(
                    batch_size, n_head, q_len, head_size, dtype=torch.float16
                )
                key = torch.randn(
                    batch_size, n_query_groups, kv_len, head_size, dtype=torch.float16
                )
                value = torch.randn(
                    batch_size, n_query_groups, kv_len, head_size, dtype=torch.float16
                )
                scale_factor = 1.0 / (head_size**0.5)

                # input_pos=0 -> fused prefill can't be used (needs non-causal) -> two-phase
                wrapper.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    scale_factor,
                    return_attn_weights=True,
                    input_pos=0,
                )
                mock_two_phase.assert_called_once()

    def test_fallback_with_float32(self):
        """Test that float32 with weights falls back to eager (not fused/two-phase)."""
        with patch.object(
            FlashInferSDPA, "_check_vendored_kernels_available", return_value=True
        ):
            with patch.object(
                FlashInferSDPA, "_fallback_sdpa", return_value=(torch.tensor([]), None)
            ) as mock_fallback:
                wrapper = FlashInferSDPA(use_fused_prefill=True)
                wrapper.available = True

                batch_size, n_head, q_len, head_size = 2, 4, 16, 64
                n_query_groups = 2
                kv_len = 32

                # float32 dtype -> can't use FlashInfer or Triton -> falls to eager
                query = torch.randn(
                    batch_size, n_head, q_len, head_size, dtype=torch.float32
                )
                key = torch.randn(
                    batch_size, n_query_groups, kv_len, head_size, dtype=torch.float32
                )
                value = torch.randn(
                    batch_size, n_query_groups, kv_len, head_size, dtype=torch.float32
                )
                scale_factor = 1.0 / (head_size**0.5)

                wrapper.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    scale_factor,
                    return_attn_weights=True,
                    input_pos=q_len,
                )
                mock_fallback.assert_called_once()

    def test_compute_weights_from_lse_shape(self):
        """Test that _compute_weights_from_lse returns correct shape."""
        with patch.object(
            FlashInferSDPA, "_check_vendored_kernels_available", return_value=True
        ):
            wrapper = FlashInferSDPA()

            batch_size, n_head, q_len, head_size = 2, 8, 16, 64
            n_kv_heads = 2
            kv_len = 32

            query = torch.randn(batch_size, n_head, q_len, head_size)
            key = torch.randn(batch_size, n_kv_heads, kv_len, head_size)
            # LSE shape: [batch, q_len, n_head] (FlashInfer layout)
            lse = torch.randn(batch_size, q_len, n_head)

            weights = wrapper._compute_weights_from_lse(
                query,
                key,
                scale_factor=1.0 / (head_size**0.5),
                lse=lse,
                input_pos=0,
            )

            assert weights.shape == (batch_size, n_kv_heads, kv_len)
            assert weights.dtype == torch.float32

    def test_compute_weights_from_lse_nonnegative(self):
        """Test that computed weights are non-negative."""
        with patch.object(
            FlashInferSDPA, "_check_vendored_kernels_available", return_value=True
        ):
            wrapper = FlashInferSDPA()

            batch_size, n_head, q_len, head_size = 1, 4, 8, 64
            n_kv_heads = 2
            kv_len = 16

            query = torch.randn(batch_size, n_head, q_len, head_size)
            key = torch.randn(batch_size, n_kv_heads, kv_len, head_size)
            # Use a realistic LSE (positive values, as log2 of softmax denominator)
            lse = torch.ones(batch_size, q_len, n_head) * 5.0

            weights = wrapper._compute_weights_from_lse(
                query,
                key,
                scale_factor=1.0 / (head_size**0.5),
                lse=lse,
                input_pos=0,
            )

            assert (weights >= 0).all(), "Attention weights should be non-negative"

    def test_compute_weights_from_lse_causal_mask(self):
        """Test that causal masking is correctly applied in weight computation."""
        with patch.object(
            FlashInferSDPA, "_check_vendored_kernels_available", return_value=True
        ):
            wrapper = FlashInferSDPA()

            batch_size, n_head, q_len, head_size = 1, 2, 4, 64
            n_kv_heads = 2
            kv_len = 8

            query = torch.randn(batch_size, n_head, q_len, head_size)
            key = torch.randn(batch_size, n_kv_heads, kv_len, head_size)
            lse = torch.ones(batch_size, q_len, n_head) * 5.0

            # With input_pos=0: q_idx in [0,3], causal mask: kv_idx <= q_idx
            # So kv_idx 4,5,6,7 should have zero weight (no query can attend to them)
            weights = wrapper._compute_weights_from_lse(
                query,
                key,
                scale_factor=1.0 / (head_size**0.5),
                lse=lse,
                input_pos=0,
            )

            # Positions beyond q_len-1 should have zero accumulated weight
            # (no query attended there with input_pos=0)
            assert (
                weights[:, :, q_len:].sum().item() < 1e-6
            ), "Positions beyond causal boundary should have ~zero weight"

    def test_compute_weights_from_lse_with_chunk_size(self):
        """Test that chunked computation matches unchunked."""
        with patch.object(
            FlashInferSDPA, "_check_vendored_kernels_available", return_value=True
        ):
            wrapper = FlashInferSDPA()

            batch_size, n_head, q_len, head_size = 1, 4, 16, 64
            n_kv_heads = 2
            kv_len = 32

            query = torch.randn(batch_size, n_head, q_len, head_size)
            key = torch.randn(batch_size, n_kv_heads, kv_len, head_size)
            lse = torch.ones(batch_size, q_len, n_head) * 5.0

            scale = 1.0 / (head_size**0.5)

            # Compute without chunking (chunk_q = q_len)
            weights_full = wrapper._compute_weights_from_lse(
                query,
                key,
                scale_factor=scale,
                lse=lse,
                input_pos=16,
            )

            # Compute with small chunk size
            weights_chunked = wrapper._compute_weights_from_lse(
                query,
                key,
                scale_factor=scale,
                lse=lse,
                input_pos=16,
                chunk_size=4,
            )

            torch.testing.assert_close(
                weights_full, weights_chunked, rtol=1e-4, atol=1e-5
            )


@_RunIf(min_cuda_gpus=1)
@pytest.mark.skip("To be fixed")
class TestCausalMaskingCorrectness:
    """Verify causal masking in fused prefill matches eager fallback.

    When input_pos > 0, the current chunk's K/V is already in the cache,
    so KV positions overlap with query positions. The fused prefill path
    (FlashInfer forward + Triton score-sum) must apply causal masking
    to produce correct output and attention weights.
    """

    def _eager_reference(self, query, key, value, scale, input_pos, token_positions):
        """Compute reference O and W using eager SDPA with explicit causal mask."""
        wrapper = FlashInferSDPA(use_fused_prefill=False)
        wrapper.available = False  # force eager fallback
        return wrapper.scaled_dot_product_attention(
            query,
            key,
            value,
            scale,
            return_attn_weights=True,
            token_positions=token_positions,
            input_pos=input_pos,
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_fused_prefill_causal_masking_vs_eager(self, dtype):
        """Fused prefill output and weights must match eager with causal mask."""
        batch_size = 2
        n_head = 8
        n_kv_heads = 2
        q_len = 64
        input_pos = 128
        kv_len = input_pos + q_len  # current chunk included in cache
        head_size = 128

        torch.manual_seed(42)
        query = torch.randn(
            batch_size, n_head, q_len, head_size, device="cuda", dtype=dtype
        )
        key = torch.randn(
            batch_size, n_kv_heads, kv_len, head_size, device="cuda", dtype=dtype
        )
        value = torch.randn(
            batch_size, n_kv_heads, kv_len, head_size, device="cuda", dtype=dtype
        )
        scale = 1.0 / (head_size**0.5)

        # Contiguous positions: [0, 1, ..., kv_len-1]
        token_positions = torch.arange(kv_len, device="cuda", dtype=torch.int32)
        token_positions = (
            token_positions.view(1, 1, -1)
            .expand(batch_size, n_kv_heads, -1)
            .contiguous()
        )

        # Eager reference
        ref_output, ref_weights = self._eager_reference(
            query, key, value, scale, input_pos, token_positions
        )

        # Fused prefill path
        wrapper = FlashInferSDPA(use_fused_prefill=True)
        if not wrapper.available:
            pytest.skip("FlashInfer kernels not available")

        fused_output, fused_weights = wrapper.scaled_dot_product_attention(
            query,
            key,
            value,
            scale,
            return_attn_weights=True,
            token_positions=token_positions,
            input_pos=input_pos,
        )

        # Output O must match
        torch.testing.assert_close(
            fused_output.float(),
            ref_output.float(),
            rtol=1e-2,
            atol=1e-2,
            msg="Fused prefill output does not match eager reference (causal masking bug?)",
        )

        # Attention weights W must match
        assert fused_weights is not None
        assert ref_weights is not None
        torch.testing.assert_close(
            fused_weights.float(),
            ref_weights.float(),
            rtol=1e-2,
            atol=1e-2,
            msg="Fused prefill weights do not match eager reference (causal masking bug?)",
        )

    def test_fused_prefill_weights_zero_for_future_positions(self):
        """Verify that attention weights for future KV positions are zero."""
        batch_size = 1
        n_head = 4
        n_kv_heads = 2
        q_len = 32
        input_pos = 64
        kv_len = input_pos + q_len
        head_size = 128

        torch.manual_seed(123)
        query = torch.randn(
            batch_size, n_head, q_len, head_size, device="cuda", dtype=torch.float16
        )
        key = torch.randn(
            batch_size,
            n_kv_heads,
            kv_len,
            head_size,
            device="cuda",
            dtype=torch.float16,
        )
        value = torch.randn(
            batch_size,
            n_kv_heads,
            kv_len,
            head_size,
            device="cuda",
            dtype=torch.float16,
        )
        scale = 1.0 / (head_size**0.5)

        token_positions = torch.arange(kv_len, device="cuda", dtype=torch.int32)
        token_positions = (
            token_positions.view(1, 1, -1)
            .expand(batch_size, n_kv_heads, -1)
            .contiguous()
        )

        wrapper = FlashInferSDPA(use_fused_prefill=True)
        if not wrapper.available:
            pytest.skip("FlashInfer kernels not available")

        _, weights = wrapper.scaled_dot_product_attention(
            query,
            key,
            value,
            scale,
            return_attn_weights=True,
            token_positions=token_positions,
            input_pos=input_pos,
        )

        assert weights is not None
        # The last KV position (kv_len-1) should only get weight from the last
        # query (input_pos + q_len - 1 = kv_len - 1). All earlier queries should
        # NOT attend to it. So the weight at the last position should be much less
        # than the weight at position 0 (which all queries attend to).
        # More importantly: if causal masking is broken, the weight at the LAST
        # position would be as large as early positions.
        w = weights[0]  # [n_kv_heads, kv_len]
        # Weight at position 0: all q_len queries attend → sum ≈ q_len
        # Weight at position kv_len-1: only 1 query attends → sum ≈ 1
        ratio = w[:, 0].mean() / (w[:, -1].mean() + 1e-10)
        assert ratio > 2.0, (
            f"Expected first-position weight >> last-position weight (ratio={ratio:.2f}). "
            f"Causal masking may not be working."
        )


def _torch_attention_weights(query, key, scale, input_pos, token_positions):
    """Pure PyTorch reference: compute attention weight sums with causal mask.

    Args:
        query: [batch, n_head, q_len, head_dim]
        key: [batch, n_kv_heads, kv_len, head_dim]
        scale: softmax scale
        input_pos: int, starting query position
        token_positions: [batch, n_kv_heads, kv_len] int32

    Returns:
        W: [batch, n_kv_heads, kv_len] float32 — sum of softmax weights over queries
    """
    batch, n_head, q_len, hd = query.shape
    _, n_kv_heads, kv_len, _ = key.shape
    group_size = n_head // n_kv_heads

    # Expand key for GQA: [batch, n_head, kv_len, hd]
    key_expanded = key.unsqueeze(2).expand(-1, -1, group_size, -1, -1)
    key_expanded = key_expanded.reshape(batch, n_head, kv_len, hd)

    # Scores: [batch, n_head, q_len, kv_len]
    scores = torch.matmul(query.float(), key_expanded.float().transpose(-1, -2)) * scale

    # Causal mask: query pos >= kv pos
    q_pos = torch.arange(input_pos, input_pos + q_len, device=query.device)  # [q_len]
    kv_pos = token_positions[:, 0, :]  # [batch, kv_len] — same across heads
    # mask[b, q, k] = True if q_pos[q] >= kv_pos[b, k]
    mask = q_pos[None, :, None] >= kv_pos[:, None, :]  # [batch, q_len, kv_len]
    scores.masked_fill_(~mask[:, None, :, :], float("-inf"))

    # Softmax over kv_len
    weights = torch.softmax(scores, dim=-1)  # [batch, n_head, q_len, kv_len]

    # Sum over queries and group heads → [batch, n_kv_heads, kv_len]
    weights = weights.reshape(batch, n_kv_heads, group_size, q_len, kv_len)
    W = weights.sum(dim=(2, 3))  # sum over group heads and queries
    # Mean over GQA group (divide by group_size) to match codebase convention
    W = W / group_size
    return W.float()


@_RunIf(min_cuda_gpus=1)
@pytest.mark.skip("To be fixed")
class TestTritonScoreSumKernel:
    """Test the Triton score-sum kernel directly."""

    def test_no_causal_matches_pytorch(self):
        """Without causal masking, Triton kernel matches PyTorch matmul-based weights."""
        from keys_values.attention.flashinfer_wrapper import triton_score_sum

        batch, q_len, n_head, n_kv, hd = 1, 32, 8, 2, 128
        kv_len = 64
        group_size = n_head // n_kv
        scale = 1.0 / (hd**0.5)

        torch.manual_seed(0)
        Q = torch.randn(batch, q_len, n_head, hd, device="cuda", dtype=torch.float16)
        K = torch.randn(batch, kv_len, n_kv, hd, device="cuda", dtype=torch.float16)

        # Compute LSE via PyTorch (no causal mask)
        Q_4d = Q.permute(0, 2, 1, 3)  # [batch, n_head, q_len, hd]
        # Expand K: each kv head repeated group_size times along head dim
        # so heads 0..group_size-1 share kv_head_0, etc.
        K_exp = K.permute(0, 2, 1, 3)  # [batch, n_kv, kv_len, hd]
        K_exp = K_exp.unsqueeze(2).expand(-1, -1, group_size, -1, -1)
        K_exp = K_exp.reshape(batch, n_head, kv_len, hd)
        scores = torch.matmul(Q_4d.float(), K_exp.float().transpose(-1, -2)) * scale
        lse = torch.logsumexp(scores, dim=-1) / 0.6931471805599453  # convert to log2
        # lse: [batch, n_head, q_len] → transpose to [batch, q_len, n_head]
        lse = lse.permute(0, 2, 1).float()

        # Reference weights (no causal mask)
        weights_ref = torch.softmax(scores, dim=-1)
        W_ref = (
            weights_ref.reshape(batch, n_kv, group_size, q_len, kv_len).sum(dim=(2, 3))
            / group_size
        )

        # Triton kernel (no causal)
        W_triton = triton_score_sum(Q, K, lse, scale, n_kv, group_size)

        torch.testing.assert_close(
            W_triton,
            W_ref.float(),
            rtol=1e-2,
            atol=1e-2,
            msg="Triton score-sum without causal mask doesn't match PyTorch reference",
        )

    def test_causal_masking_zeros_future(self):
        """With causal masking, weights for KV positions > query position are zero."""
        from keys_values.attention.flashinfer_wrapper import triton_score_sum

        batch, q_len, n_head, n_kv, hd = 1, 16, 4, 2, 128
        input_pos = 32
        kv_len = input_pos + q_len  # 48
        group_size = n_head // n_kv
        scale = 1.0 / (hd**0.5)

        torch.manual_seed(7)
        Q = torch.randn(batch, q_len, n_head, hd, device="cuda", dtype=torch.float16)
        K = torch.randn(batch, kv_len, n_kv, hd, device="cuda", dtype=torch.float16)

        # Compute LSE with causal mask via PyTorch
        Q_4d = Q.permute(0, 2, 1, 3).float()
        K_exp = K.permute(0, 2, 1, 3).unsqueeze(1).expand(-1, group_size, -1, -1, -1)
        K_exp = K_exp.reshape(batch, n_head, kv_len, hd).float()
        scores = torch.matmul(Q_4d, K_exp.transpose(-1, -2)) * scale
        q_pos = torch.arange(input_pos, input_pos + q_len, device="cuda")
        kv_pos = torch.arange(kv_len, device="cuda")
        causal_mask = q_pos[:, None] >= kv_pos[None, :]  # [q_len, kv_len]
        scores.masked_fill_(~causal_mask[None, None, :, :], float("-inf"))
        lse = torch.logsumexp(scores, dim=-1) / 0.6931471805599453
        lse = lse.permute(0, 2, 1).float()

        # Token positions: contiguous [0, 1, ..., kv_len-1]
        tp = torch.arange(kv_len, device="cuda", dtype=torch.int32)
        tp = tp.view(1, 1, -1).expand(batch, n_kv, -1).contiguous()

        W = triton_score_sum(Q, K, lse, scale, n_kv, group_size, tp, input_pos)

        # The LAST KV position (kv_len-1 = input_pos + q_len - 1) should only
        # receive weight from the last query. All prior queries can't attend to it.
        # The FIRST KV position (0) should receive weight from ALL queries.
        for h in range(n_kv):
            assert (
                W[0, h, -1] < W[0, h, 0]
            ), f"Head {h}: W[last]={W[0,h,-1]:.4f} should be < W[first]={W[0,h,0]:.4f}"

    def test_causal_matches_pytorch_reference(self):
        """Triton causal score-sum matches full PyTorch reference computation."""
        from keys_values.attention.flashinfer_wrapper import triton_score_sum

        batch, q_len, n_head, n_kv, hd = 2, 32, 8, 2, 128
        input_pos = 64
        kv_len = input_pos + q_len
        group_size = n_head // n_kv
        scale = 1.0 / (hd**0.5)

        torch.manual_seed(42)
        Q = torch.randn(batch, q_len, n_head, hd, device="cuda", dtype=torch.float16)
        K = torch.randn(batch, kv_len, n_kv, hd, device="cuda", dtype=torch.float16)

        query_4d = Q.permute(0, 2, 1, 3)  # [batch, n_head, q_len, hd]
        key_4d = K.permute(0, 2, 1, 3)  # [batch, n_kv, kv_len, hd]

        tp = torch.arange(kv_len, device="cuda", dtype=torch.int32)
        tp = tp.view(1, 1, -1).expand(batch, n_kv, -1).contiguous()

        # PyTorch reference
        W_ref = _torch_attention_weights(query_4d, key_4d, scale, input_pos, tp)

        # Compute causal LSE via PyTorch
        K_exp = key_4d.unsqueeze(2).expand(-1, -1, group_size, -1, -1)
        K_exp = K_exp.reshape(batch, n_head, kv_len, hd)
        scores = torch.matmul(query_4d.float(), K_exp.float().transpose(-1, -2)) * scale
        q_pos = torch.arange(input_pos, input_pos + q_len, device="cuda")
        kv_pos = torch.arange(kv_len, device="cuda")
        causal = q_pos[:, None] >= kv_pos[None, :]
        scores.masked_fill_(~causal[None, None, :, :], float("-inf"))
        lse = torch.logsumexp(scores, dim=-1) / 0.6931471805599453
        lse = lse.permute(0, 2, 1).float()

        # Triton kernel
        W_triton = triton_score_sum(Q, K, lse, scale, n_kv, group_size, tp, input_pos)

        torch.testing.assert_close(
            W_triton,
            W_ref,
            rtol=1e-2,
            atol=1e-2,
            msg="Triton causal score-sum doesn't match PyTorch reference",
        )

    def test_noncontiguous_positions(self):
        """Simulates H2O eviction: non-contiguous token_positions."""
        from keys_values.attention.flashinfer_wrapper import triton_score_sum

        batch, q_len, n_head, n_kv, hd = 1, 16, 4, 2, 128
        kv_len = 64
        input_pos = 100  # well past cache filling
        group_size = n_head // n_kv
        scale = 1.0 / (hd**0.5)

        torch.manual_seed(99)
        Q = torch.randn(batch, q_len, n_head, hd, device="cuda", dtype=torch.float16)
        K = torch.randn(batch, kv_len, n_kv, hd, device="cuda", dtype=torch.float16)

        # Non-contiguous positions: simulate evicted slots replaced with new positions
        positions = list(range(kv_len))
        # Slots 10, 20, 30 were evicted and now hold positions 100, 101, 102
        positions[10] = 100
        positions[20] = 101
        positions[30] = 102
        tp = torch.tensor(positions, device="cuda", dtype=torch.int32)
        tp = tp.view(1, 1, -1).expand(batch, n_kv, -1).contiguous()

        # PyTorch reference with this position mapping
        query_4d = Q.permute(0, 2, 1, 3)
        key_4d = K.permute(0, 2, 1, 3)
        W_ref = _torch_attention_weights(query_4d, key_4d, scale, input_pos, tp)

        # Compute LSE with the correct causal mask
        K_exp = key_4d.unsqueeze(2).expand(-1, -1, group_size, -1, -1)
        K_exp = K_exp.reshape(batch, n_head, kv_len, hd)
        scores = torch.matmul(query_4d.float(), K_exp.float().transpose(-1, -2)) * scale
        q_pos = torch.arange(input_pos, input_pos + q_len, device="cuda")
        kv_pos = tp[0, 0, :]  # [kv_len]
        causal = q_pos[:, None] >= kv_pos[None, :]
        scores.masked_fill_(~causal[None, None, :, :], float("-inf"))
        lse = torch.logsumexp(scores, dim=-1) / 0.6931471805599453
        lse = lse.permute(0, 2, 1).float()

        W_triton = triton_score_sum(Q, K, lse, scale, n_kv, group_size, tp, input_pos)

        torch.testing.assert_close(
            W_triton,
            W_ref,
            rtol=1e-2,
            atol=1e-2,
            msg="Triton score-sum with non-contiguous positions doesn't match reference",
        )

        # Specifically: slot 30 has position 102 > query positions [100..115],
        # so only queries at pos 102..115 (14 queries) should attend to it.
        # Slot 0 has position 0 <= all queries, so all 16 queries attend.
        # Weight at slot 0 should be > weight at slot 30.
        for h in range(n_kv):
            assert W_triton[0, h, 0] > W_triton[0, h, 30], (
                f"Head {h}: slot 0 (pos=0, all queries attend) should have more weight "
                f"than slot 30 (pos=102, only 14 queries attend)"
            )


@_RunIf(min_cuda_gpus=1)
@pytest.mark.skip("To be fixed")
class TestFusedPrefillEndToEnd:
    """End-to-end tests: fused prefill path vs eager fallback."""

    def _run_both_paths(self, batch, n_head, n_kv, q_len, input_pos, hd, dtype):
        """Run fused prefill and eager fallback, return both results."""
        kv_len = input_pos + q_len
        scale = 1.0 / (hd**0.5)

        torch.manual_seed(42)
        query = torch.randn(batch, n_head, q_len, hd, device="cuda", dtype=dtype)
        key = torch.randn(batch, n_kv, kv_len, hd, device="cuda", dtype=dtype)
        value = torch.randn(batch, n_kv, kv_len, hd, device="cuda", dtype=dtype)

        tp = torch.arange(kv_len, device="cuda", dtype=torch.int32)
        tp = tp.view(1, 1, -1).expand(batch, n_kv, -1).contiguous()

        # Eager reference
        eager = FlashInferSDPA(use_fused_prefill=False)
        eager.available = False
        ref_o, ref_w = eager.scaled_dot_product_attention(
            query,
            key,
            value,
            scale,
            return_attn_weights=True,
            token_positions=tp,
            input_pos=input_pos,
        )

        # Fused prefill
        fused = FlashInferSDPA(use_fused_prefill=True)
        if not fused.available:
            pytest.skip("FlashInfer not available")
        fused_o, fused_w = fused.scaled_dot_product_attention(
            query,
            key,
            value,
            scale,
            return_attn_weights=True,
            token_positions=tp,
            input_pos=input_pos,
        )

        return ref_o, ref_w, fused_o, fused_w

    def test_small_config(self):
        """Small config: 2 heads, 1 KV head, short sequences."""
        ref_o, ref_w, fused_o, fused_w = self._run_both_paths(
            batch=1,
            n_head=2,
            n_kv=1,
            q_len=16,
            input_pos=32,
            hd=128,
            dtype=torch.float16,
        )
        torch.testing.assert_close(fused_o.float(), ref_o.float(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(fused_w.float(), ref_w.float(), rtol=1e-2, atol=1e-2)

    def test_qwen3_config(self):
        """Qwen3-4B config: 32 heads, 8 KV heads, head_dim=128."""
        ref_o, ref_w, fused_o, fused_w = self._run_both_paths(
            batch=2,
            n_head=32,
            n_kv=8,
            q_len=64,
            input_pos=128,
            hd=128,
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(fused_o.float(), ref_o.float(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(fused_w.float(), ref_w.float(), rtol=1e-2, atol=1e-2)

    def test_head_dim_64(self):
        """Smaller head_dim=64 (different Triton block sizes)."""
        ref_o, ref_w, fused_o, fused_w = self._run_both_paths(
            batch=1,
            n_head=8,
            n_kv=2,
            q_len=32,
            input_pos=64,
            hd=64,
            dtype=torch.float16,
        )
        torch.testing.assert_close(fused_o.float(), ref_o.float(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(fused_w.float(), ref_w.float(), rtol=1e-2, atol=1e-2)

    def test_weights_are_nonnegative(self):
        """Attention weight sums must always be >= 0."""
        _, _, _, fused_w = self._run_both_paths(
            batch=2,
            n_head=8,
            n_kv=2,
            q_len=64,
            input_pos=128,
            hd=128,
            dtype=torch.bfloat16,
        )
        assert (fused_w >= 0).all(), "Attention weight sums must be non-negative"

    def test_weights_monotonic_by_position(self):
        """Earlier KV positions should have >= total weight than later ones.

        This is because earlier positions are attended to by MORE queries
        (all queries with position >= kv_pos can attend), while later
        positions are only attended to by fewer queries.
        """
        _, _, _, fused_w = self._run_both_paths(
            batch=1,
            n_head=8,
            n_kv=2,
            q_len=64,
            input_pos=128,
            hd=128,
            dtype=torch.bfloat16,
        )
        # Compare first quarter vs last quarter of the within-chunk region
        w = fused_w[0]  # [n_kv, kv_len]
        chunk_start = 128  # input_pos
        first_quarter = w[:, chunk_start : chunk_start + 16].mean()
        last_quarter = w[:, -16:].mean()
        assert first_quarter > last_quarter, (
            f"Earlier chunk positions ({first_quarter:.4f}) should have more weight "
            f"than later ones ({last_quarter:.4f}) due to causal masking"
        )
