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
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

from keys_values.attention import (
    eager_scaled_dot_product_attention,
    DefaultKeysAndValues,
)
from keys_values.flashinfer_wrapper import get_flashinfer_sdpa

# =============================================================================
# Backend Equivalence Verification Utilities
# =============================================================================


class BackendEquivalenceResult:
    """
    Result of backend equivalence verification.

    This class holds the results of comparing vendored kernel outputs
    against eager implementation outputs.
    """

    def __init__(
        self,
        is_equivalent: bool,
        output_max_diff: float,
        output_mean_diff: float,
        weights_max_diff: Optional[float] = None,
        weights_mean_diff: Optional[float] = None,
        rtol: float = 1e-4,
        atol: float = 1e-6,
        message: str = "",
    ):
        """
        Initialize equivalence result.

        Args:
            is_equivalent: Whether outputs are numerically equivalent
            output_max_diff: Maximum absolute difference in attention outputs
            output_mean_diff: Mean absolute difference in attention outputs
            weights_max_diff: Maximum absolute difference in attention weights (if computed)
            weights_mean_diff: Mean absolute difference in attention weights (if computed)
            rtol: Relative tolerance used for comparison
            atol: Absolute tolerance used for comparison
            message: Human-readable message describing the result
        """
        self.is_equivalent = is_equivalent
        self.output_max_diff = output_max_diff
        self.output_mean_diff = output_mean_diff
        self.weights_max_diff = weights_max_diff
        self.weights_mean_diff = weights_mean_diff
        self.rtol = rtol
        self.atol = atol
        self.message = message

    def __repr__(self) -> str:
        weights_max_str = (
            f"{self.weights_max_diff:.2e}"
            if self.weights_max_diff is not None
            else "None"
        )
        weights_mean_str = (
            f"{self.weights_mean_diff:.2e}"
            if self.weights_mean_diff is not None
            else "None"
        )
        return (
            f"BackendEquivalenceResult("
            f"is_equivalent={self.is_equivalent}, "
            f"output_max_diff={self.output_max_diff:.2e}, "
            f"output_mean_diff={self.output_mean_diff:.2e}, "
            f"weights_max_diff={weights_max_str}, "
            f"weights_mean_diff={weights_mean_str})"
        )

    def __bool__(self) -> bool:
        """Allow using result in boolean context."""
        return self.is_equivalent


def check_numerical_equivalence(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> Tuple[bool, float, float]:
    """
    Check if two tensors are numerically equivalent within tolerance.

    Args:
        tensor_a: First tensor
        tensor_b: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Tuple of (is_equivalent, max_diff, mean_diff)
    """
    if tensor_a.shape != tensor_b.shape:
        raise ValueError(f"Shape mismatch: {tensor_a.shape} vs {tensor_b.shape}")

    # Compute differences
    diff = torch.abs(tensor_a.float() - tensor_b.float())
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Check equivalence using torch.allclose logic
    is_equivalent = torch.allclose(
        tensor_a.float(), tensor_b.float(), rtol=rtol, atol=atol
    )

    return is_equivalent, max_diff, mean_diff


def verify_backend_equivalence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: float,
    return_attn_weights: bool = False,
    token_positions: Optional[torch.Tensor] = None,
    input_pos: int = 0,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    log_results: bool = True,
) -> BackendEquivalenceResult:
    """
    Verify that vendored kernels produce equivalent results to eager implementation.

    This function computes SDPA using both the vendored FlashInfer kernels and
    the eager fallback implementation, then compares the results to verify
    numerical equivalence within the specified tolerance.

    Args:
        query: Query tensor, shape (batch_size, n_head, q_len, head_size)
        key: Key tensor, shape (batch_size, n_query_groups, kv_len, head_size)
        value: Value tensor, shape (batch_size, n_query_groups, kv_len, head_size)
        scale_factor: Scale factor for attention scores
        return_attn_weights: Whether to compare attention weights
        token_positions: Token positions in KV cache
        input_pos: Position in input sequence
        rtol: Relative tolerance for numerical comparison
        atol: Absolute tolerance for numerical comparison
        log_results: Whether to log verification results

    Returns:
        BackendEquivalenceResult containing comparison metrics and equivalence status

    Raises:
        RuntimeError: If vendored kernels are not available

    Example:
        >>> wrapper = FlashInferSDPA()
        >>> query = torch.randn(2, 4, 8, 64, device='cuda')
        >>> key = torch.randn(2, 2, 16, 64, device='cuda')
        >>> value = torch.randn(2, 2, 16, 64, device='cuda')
        >>> result = verify_backend_equivalence(
        ...     query, key, value, scale_factor=0.125,
        ...     return_attn_weights=True
        ... )
        >>> if result.is_equivalent:
        ...     print("Backends produce equivalent results")
        ... else:
        ...     print(f"Backends differ: {result.message}")
    """
    wrapper = get_flashinfer_sdpa()

    # Compute using vendored kernels
    try:
        vendored_output, vendored_weights = wrapper.scaled_dot_product_attention(
            query,
            key,
            value,
            scale_factor,
            input_pos=input_pos,
            token_positions=token_positions,
            return_attn_weights=return_attn_weights,
        )
    except Exception as e:
        message = f"Vendored kernel computation failed: {e}"
        if log_results:
            logger.error(message)
        return BackendEquivalenceResult(
            is_equivalent=False,
            output_max_diff=float("inf"),
            output_mean_diff=float("inf"),
            rtol=rtol,
            atol=atol,
            message=message,
        )

    # Compute using eager fallback
    eager_output, eager_weights = eager_scaled_dot_product_attention(
        query=query,
        k_and_v=DefaultKeysAndValues(key, value),
        scale_factor=scale_factor,
        use_blocking=True,
        return_attn_weights=return_attn_weights,
        input_pos=input_pos,
        token_positions=token_positions,
        sliding_window_size=None,
    )

    # Compare outputs
    output_equivalent, output_max_diff, output_mean_diff = check_numerical_equivalence(
        vendored_output, eager_output, rtol=rtol, atol=atol
    )

    # Compare weights if requested
    weights_max_diff = None
    weights_mean_diff = None
    weights_equivalent = True

    if (
        return_attn_weights
        and vendored_weights is not None
        and eager_weights is not None
    ):
        weights_equivalent, weights_max_diff, weights_mean_diff = (
            check_numerical_equivalence(
                vendored_weights, eager_weights, rtol=rtol, atol=atol
            )
        )

    # Determine overall equivalence
    is_equivalent = output_equivalent and weights_equivalent

    # Build message
    if is_equivalent:
        message = (
            f"Backend equivalence verified: "
            f"output_max_diff={output_max_diff:.2e}, "
            f"output_mean_diff={output_mean_diff:.2e}"
        )
        if weights_max_diff is not None:
            message += f", weights_max_diff={weights_max_diff:.2e}"
    else:
        message = "Backend equivalence FAILED: "
        if not output_equivalent:
            message += (
                f"output differs (max_diff={output_max_diff:.2e}, "
                f"mean_diff={output_mean_diff:.2e}, rtol={rtol}, atol={atol})"
            )
        if not weights_equivalent:
            message += (
                f"weights differ (max_diff={weights_max_diff:.2e}, "
                f"mean_diff={weights_mean_diff:.2e})"
            )

    # Log results
    if log_results:
        if is_equivalent:
            logger.debug(message)
        else:
            logger.warning(message)

    return BackendEquivalenceResult(
        is_equivalent=is_equivalent,
        output_max_diff=output_max_diff,
        output_mean_diff=output_mean_diff,
        weights_max_diff=weights_max_diff,
        weights_mean_diff=weights_mean_diff,
        rtol=rtol,
        atol=atol,
        message=message,
    )


def verify_backend_equivalence_batch(
    test_cases: list,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    log_results: bool = True,
    stop_on_failure: bool = False,
) -> Tuple[int, int, list]:
    """
    Verify backend equivalence for multiple test cases.

    Args:
        test_cases: List of dictionaries containing test parameters.
            Each dictionary should have keys: query, key, value, scale_factor,
            and optionally: return_attn_weights, token_positions, input_pos,
            sliding_window_size, chunk_size
        rtol: Relative tolerance for numerical comparison
        atol: Absolute tolerance for numerical comparison
        log_results: Whether to log verification results
        stop_on_failure: Whether to stop on first failure

    Returns:
        Tuple of (passed_count, failed_count, results_list)

    Example:
        >>> test_cases = [
        ...     {
        ...         'query': torch.randn(2, 4, 8, 64, device='cuda'),
        ...         'key': torch.randn(2, 2, 16, 64, device='cuda'),
        ...         'value': torch.randn(2, 2, 16, 64, device='cuda'),
        ...         'scale_factor': 0.125,
        ...         'return_attn_weights': True,
        ...     },
        ...     # ... more test cases
        ... ]
        >>> passed, failed, results = verify_backend_equivalence_batch(test_cases)
        >>> print(f"Passed: {passed}, Failed: {failed}")
    """
    passed_count = 0
    failed_count = 0
    results = []

    for i, test_case in enumerate(test_cases):
        try:
            result = verify_backend_equivalence(
                query=test_case["query"],
                key=test_case["key"],
                value=test_case["value"],
                scale_factor=test_case["scale_factor"],
                return_attn_weights=test_case.get("return_attn_weights", False),
                token_positions=test_case.get("token_positions"),
                input_pos=test_case.get("input_pos", 0),
                rtol=rtol,
                atol=atol,
                log_results=log_results,
            )

            results.append(result)

            if result.is_equivalent:
                passed_count += 1
            else:
                failed_count += 1
                if stop_on_failure:
                    if log_results:
                        logger.warning(f"Stopping at test case {i} due to failure")
                    break

        except Exception as e:
            failed_count += 1
            error_result = BackendEquivalenceResult(
                is_equivalent=False,
                output_max_diff=float("inf"),
                output_mean_diff=float("inf"),
                rtol=rtol,
                atol=atol,
                message=f"Test case {i} raised exception: {e}",
            )
            results.append(error_result)

            if log_results:
                logger.error(f"Test case {i} failed with exception: {e}")

            if stop_on_failure:
                break

    if log_results:
        logger.info(
            f"Backend equivalence verification complete: "
            f"{passed_count} passed, {failed_count} failed out of {len(test_cases)} test cases"
        )

    return passed_count, failed_count, results
