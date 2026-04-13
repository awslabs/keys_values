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
Python interface to load vendored FlashInfer CUDA kernels.

This module attempts to load the compiled CUDA extension (_flashinfer_ops)
and provides a clean Python interface with graceful fallback when the
extension is not available.
"""

import logging
import warnings
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Global state for the compiled extension
_ops = None
_available = False
_load_attempted = False
_load_error: Optional[str] = None


def _load_ops():
    """
    Attempt to load the compiled CUDA extension.

    This function tries to import the _flashinfer_ops extension module
    that was compiled from the vendored CUDA kernels. If the import fails,
    it logs the error and sets _available to False.

    The function is called automatically on module import and caches the result.
    """
    global _ops, _available, _load_attempted, _load_error

    if _load_attempted:
        return

    _load_attempted = True

    try:
        # Try to import the compiled extension
        from keys_values import _flashinfer_ops

        # Verify CUDA is available through the extension
        if not _flashinfer_ops.is_available():
            _load_error = "CUDA is not available on this system"
            logger.debug(_load_error)
            _available = False
            return

        # Check device count
        device_count = _flashinfer_ops.get_device_count()
        if device_count == 0:
            _load_error = "No CUDA devices found"
            logger.debug(_load_error)
            _available = False
            return

        # Success - extension is loaded and CUDA is available
        _ops = _flashinfer_ops
        _available = True

        # Log device information
        logger.info(
            f"Vendored FlashInfer kernels loaded successfully with {device_count} CUDA device(s)"
        )
        for device_id in range(device_count):
            device_info = _ops.get_device_info(device_id)
            logger.debug(f"  {device_info}")

    except ImportError as e:
        _load_error = f"Failed to import _flashinfer_ops extension: {e}"
        logger.debug(_load_error)
        warnings.warn(
            "Vendored FlashInfer CUDA kernels are not available. "
            "The extension may not have been compiled. "
            "Falling back to eager SDPA implementation. "
            f"Error: {e}",
            RuntimeWarning,
            stacklevel=2,
        )
        _available = False
    except Exception as e:
        _load_error = f"Unexpected error loading _flashinfer_ops extension: {e}"
        logger.debug(_load_error)
        warnings.warn(
            f"Unexpected error loading vendored FlashInfer kernels: {e}. "
            "Falling back to eager SDPA implementation.",
            RuntimeWarning,
            stacklevel=2,
        )
        _available = False


# Attempt to load the extension on module import
_load_ops()


def is_available() -> bool:
    """
    Check if vendored FlashInfer CUDA kernels are available.

    Returns:
        True if the compiled extension is loaded and CUDA is available,
        False otherwise.

    Example:
        >>> import keys_values.flashinfer_ops as ops
        >>> if ops.is_available():
        ...     # Use vendored kernels
        ...     output, weights = ops.sdpa_decode(...)
        ... else:
        ...     # Fall back to eager implementation
        ...     pass
    """
    return _available


def get_load_error() -> Optional[str]:
    """
    Get the error message from the last load attempt, if any.

    Returns:
        Error message string if loading failed, None if loading succeeded
        or has not been attempted yet.

    Example:
        >>> import keys_values.flashinfer_ops as ops
        >>> if not ops.is_available():
        ...     print(f"Kernels unavailable: {ops.get_load_error()}")
    """
    return _load_error


def get_device_count() -> int:
    """
    Get the number of available CUDA devices.

    Returns:
        Number of CUDA devices if extension is available, 0 otherwise.

    Raises:
        RuntimeError: If vendored kernels are not available.
    """
    if not _available:
        raise RuntimeError(
            "Vendored FlashInfer kernels are not available. " f"Error: {_load_error}"
        )
    return _ops.get_device_count()


def get_device_info(device_id: int = 0) -> str:
    """
    Get CUDA device properties as a string.

    Args:
        device_id: CUDA device ID (default: 0)

    Returns:
        String with device properties (name, compute capability, memory)

    Raises:
        RuntimeError: If vendored kernels are not available.

    Example:
        >>> import keys_values.flashinfer_ops as ops
        >>> if ops.is_available():
        ...     print(ops.get_device_info(0))
        Device 0: NVIDIA A100-SXM4-40GB (SM 8.0), 40960 MB
    """
    if not _available:
        raise RuntimeError(
            "Vendored FlashInfer kernels are not available. " f"Error: {_load_error}"
        )
    return _ops.get_device_info(device_id)


def sdpa_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    token_positions: Optional[torch.Tensor] = None,
    input_pos: Optional[torch.Tensor] = None,
    sliding_window_size: int = -1,
    causal: bool = True,
    return_weights: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Scaled Dot Product Attention for decode phase (single query token).

    This function calls the vendored CUDA kernel for efficient decode-phase
    attention computation. It supports causal masking, sliding window attention,
    and optional attention weight return.

    Args:
        query: Query tensor with shape:
            - Batched: [batch_size, num_qo_heads, head_dim]
            - Non-batched: [num_qo_heads, head_dim]
        key: Key tensor with shape:
            - Batched: [batch_size, kv_len, num_kv_heads, head_dim]
            - Non-batched: [kv_len, num_kv_heads, head_dim]
        value: Value tensor, same shape as key
        scale: Softmax scale factor (typically 1/sqrt(head_dim))
        token_positions: Optional token positions for causal masking. Supports:
            - Shared across heads:
              - Batched: [batch_size, kv_len]
              - Non-batched: [kv_len]
            - Per-head (heterogeneous orderings):
              - Batched: [batch_size, num_kv_heads, kv_len]
              - Non-batched: [num_kv_heads, kv_len]
        input_pos: Current query position(s):
            - Batched: [batch_size] tensor
            - Non-batched: scalar tensor
        sliding_window_size: Sliding window size (-1 for no window)
        causal: Whether to apply causal masking (default: True)
        return_weights: Whether to return attention weights (default: False)

    Returns:
        Tuple of (output, attention_weights):
        - output: Attention output with same shape as query
        - attention_weights: If return_weights=True, tensor with shape:
            - Batched: [batch_size, num_kv_heads, kv_len]
            - Non-batched: [num_kv_heads, kv_len]
          If return_weights=False, None

    Raises:
        RuntimeError: If vendored kernels are not available.

    Example:
        >>> import torch
        >>> import keys_values.flashinfer_ops as ops
        >>>
        >>> # Batched decode with per-head token positions
        >>> query = torch.randn(2, 32, 128, device='cuda', dtype=torch.float16)
        >>> key = torch.randn(2, 1024, 8, 128, device='cuda', dtype=torch.float16)
        >>> value = torch.randn(2, 1024, 8, 128, device='cuda', dtype=torch.float16)
        >>> token_positions = torch.randint(0, 1024, (2, 8, 1024), device='cuda', dtype=torch.int32)
        >>> scale = 1.0 / (128 ** 0.5)
        >>>
        >>> output, weights = ops.sdpa_decode(
        ...     query, key, value, scale,
        ...     token_positions=token_positions,
        ...     return_weights=True
        ... )
        >>> print(output.shape)  # [2, 32, 128]
        >>> print(weights.shape)  # [2, 8, 1024]
    """
    if not _available:
        raise RuntimeError(
            "Vendored FlashInfer kernels are not available. " f"Error: {_load_error}"
        )

    return _ops.sdpa_decode(
        query,
        key,
        value,
        scale,
        token_positions,
        input_pos,
        sliding_window_size,
        causal,
        return_weights,
    )


def sdpa_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    token_positions: Optional[torch.Tensor] = None,
    input_pos: Optional[torch.Tensor] = None,
    sliding_window_size: int = -1,
    causal: bool = True,
    return_weights: bool = False,
    return_lse: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Scaled Dot Product Attention for prefill phase (multiple query tokens).

    This function calls the vendored CUDA kernel for efficient prefill-phase
    attention computation. It supports causal masking, sliding window attention,
    and optional attention weight return summed over the query axis.

    Args:
        query: Query tensor with shape:
            - Batched: [batch_size, q_len, num_qo_heads, head_dim]
            - Non-batched: [q_len, num_qo_heads, head_dim]
        key: Key tensor with shape:
            - Batched: [batch_size, kv_len, num_kv_heads, head_dim]
            - Non-batched: [kv_len, num_kv_heads, head_dim]
        value: Value tensor, same shape as key
        scale: Softmax scale factor (typically 1/sqrt(head_dim))
        token_positions: Optional token positions for causal masking with shape:
            - Batched: [batch_size, kv_len]
            - Non-batched: [kv_len]
        input_pos: Starting query position(s):
            - Batched: [batch_size] tensor
            - Non-batched: scalar tensor
        sliding_window_size: Sliding window size (-1 for no window)
        causal: Whether to apply causal masking (default: True)
        return_weights: Whether to return attention weights summed over query axis (default: False)
        return_lse: Whether to return log-sum-exp values per query position (default: False).
            When True, uses the fast FlashInfer kernel and returns LSE instead of computing
            per-query attention weights. LSE values are in log base 2.

    Returns:
        Tuple of (output, attention_weights, lse):
        - output: Attention output with same shape as query
        - attention_weights: If return_weights=True and return_lse=False, tensor with shape:
            - Batched: [batch_size, num_kv_heads, kv_len]
            - Non-batched: [num_kv_heads, kv_len]
          Represents the sum of attention weights over the query axis. None otherwise.
        - lse: If return_lse=True, tensor with shape:
            - Batched: [batch_size, q_len, num_qo_heads]
            - Non-batched: [q_len, num_qo_heads]
          Values are log-sum-exp in log base 2. None otherwise.

    Raises:
        RuntimeError: If vendored kernels are not available.

    Example:
        >>> import torch
        >>> import keys_values.flashinfer_ops as ops
        >>>
        >>> # Batched prefill with LSE
        >>> query = torch.randn(2, 512, 32, 128, device='cuda', dtype=torch.float16)
        >>> key = torch.randn(2, 1024, 8, 128, device='cuda', dtype=torch.float16)
        >>> value = torch.randn(2, 1024, 8, 128, device='cuda', dtype=torch.float16)
        >>> scale = 1.0 / (128 ** 0.5)
        >>>
        >>> output, _, lse = ops.sdpa_prefill(
        ...     query, key, value, scale,
        ...     return_lse=True
        ... )
        >>> print(output.shape)  # [2, 512, 32, 128]
        >>> print(lse.shape)  # [2, 512, 32]
    """
    if not _available:
        raise RuntimeError(
            "Vendored FlashInfer kernels are not available. " f"Error: {_load_error}"
        )

    return _ops.sdpa_prefill(
        query,
        key,
        value,
        scale,
        token_positions,
        input_pos,
        sliding_window_size,
        causal,
        return_weights,
        return_lse,
    )


def sdpa_prefill_fused_v2(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    token_positions: Optional[torch.Tensor] = None,
    input_pos: Optional[torch.Tensor] = None,
    sliding_window_size: int = -1,
    causal: bool = True,
    return_weights: bool = False,
    return_lse: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Fused v2 Scaled Dot Product Attention for prefill phase.

    Single kernel that computes O via tiled attention, then re-reads K with Q
    cached in shared memory to compute summed attention weights. No intermediate
    storage — works for ALL q_len/kv_len combinations.

    When return_weights=False or return_lse=True, delegates to sdpa_prefill.

    Args:
        Same as sdpa_prefill.

    Returns:
        Same as sdpa_prefill.
    """
    if not _available:
        raise RuntimeError(
            "Vendored FlashInfer kernels are not available. " f"Error: {_load_error}"
        )

    return _ops.sdpa_prefill_fused_v2(
        query,
        key,
        value,
        scale,
        token_positions,
        input_pos,
        sliding_window_size,
        causal,
        return_weights,
        return_lse,
    )


def sdpa_prefill_fused(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    token_positions: Optional[torch.Tensor] = None,
    input_pos: Optional[torch.Tensor] = None,
    sliding_window_size: int = -1,
    causal: bool = True,
    return_weights: bool = False,
    return_lse: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Fused Scaled Dot Product Attention for prefill phase.

    Same interface as sdpa_prefill but uses a fused kernel that accumulates
    attention weights during the tiling loop. This avoids allocating the large
    [q_len, num_qo_heads, kv_len] intermediate tensor and the separate
    accumulation kernel pass.

    When return_weights=False or return_lse=True, delegates to sdpa_prefill.

    Args:
        Same as sdpa_prefill.

    Returns:
        Same as sdpa_prefill.
    """
    if not _available:
        raise RuntimeError(
            "Vendored FlashInfer kernels are not available. " f"Error: {_load_error}"
        )

    return _ops.sdpa_prefill_fused(
        query,
        key,
        value,
        scale,
        token_positions,
        input_pos,
        sliding_window_size,
        causal,
        return_weights,
        return_lse,
    )
