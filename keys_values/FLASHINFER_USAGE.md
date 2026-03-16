# FlashInfer Kernels Usage Guide

This document describes how to use the vendored FlashInfer CUDA kernels in the keys_values module.

## Overview

The keys_values module includes vendored FlashInfer CUDA kernels for optimized Scaled Dot Product Attention (SDPA) computation. These kernels are compiled as part of the keys_values package, eliminating any runtime dependency on the FlashInfer package.

**Key Benefits:**
- 9-29x faster than eager implementation for prefill operations
- Efficient attention weight computation for sparse attention policies (H2O)
- No external FlashInfer dependency required

## When FlashInfer Kernels Are Used

The FlashInfer kernels are **automatically selected** when all of the following conditions are met:

1. **Attention weights are requested** (`return_attn_weights=True`)
2. **FlashInfer wrapper is available** (kernels compiled successfully)
3. **No attention logit softcapping** (`config.attention_logit_softcapping` is `None`)

### Decision Flow in `attention.py`

```
MultiHeadSelfAttention.__call__()
    │
    ├── _should_use_flashinfer() returns True?
    │   ├── Yes → Use FlashInferSDPA.scaled_dot_product_attention()
    │   └── No  → Use PyTorch SDPA or eager implementation
```

The integration point is in `MultiHeadSelfAttention.scaled_dot_product_attention()` at lines 400-413:

```python
if self._should_use_flashinfer(sdpa_mode, return_attn_weights):
    attn_outputs, attn_weights = self._flashinfer_wrapper.scaled_dot_product_attention(
        query=query,
        key=k_and_v.keys(),
        value=k_and_v.values(),
        scale_factor=scale_factor,
        return_attn_weights=return_attn_weights,
        token_positions=token_positions,
        input_pos=input_pos,
        sliding_window_size=sliding_window_size,
    )
    return attn_outputs.transpose(1, 2), attn_weights
```

## API Reference

### FlashInferSDPA Class

Located in `keys_values/flashinfer_wrapper.py`.

#### Constructor

```python
from keys_values.flashinfer_wrapper import FlashInferSDPA

wrapper = FlashInferSDPA()
print(wrapper.available)  # True if kernels loaded successfully
```

#### Main Method: `scaled_dot_product_attention`

```python
def scaled_dot_product_attention(
    self,
    query: torch.Tensor,           # (batch_size, n_head, q_len, head_size)
    key: torch.Tensor,             # (batch_size, n_query_groups, kv_len, head_size)
    value: torch.Tensor,           # (batch_size, n_query_groups, kv_len, head_size)
    scale_factor: float,           # Typically 1/sqrt(head_size)
    return_attn_weights: bool = False,
    token_positions: Optional[torch.Tensor] = None,  # (batch_size, n_query_groups, kv_len)
    input_pos: int = 0,
    sliding_window_size: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
        attention_output: (batch_size, n_head, q_len, head_size)
        attention_weights: (batch_size, n_query_groups, kv_len) if return_attn_weights=True
    """
```

### Standalone Usage

```python
import torch
from keys_values.flashinfer_wrapper import FlashInferSDPA

# Initialize wrapper
wrapper = FlashInferSDPA()

if not wrapper.available:
    print("FlashInfer kernels not available, will use fallback")

# Create test tensors
batch_size, n_head, q_len, head_size = 2, 32, 128, 64
n_query_groups, kv_len = 8, 256  # GQA: 32 query heads, 8 KV heads

query = torch.randn(batch_size, n_head, q_len, head_size, device='cuda', dtype=torch.float16)
key = torch.randn(batch_size, n_query_groups, kv_len, head_size, device='cuda', dtype=torch.float16)
value = torch.randn(batch_size, n_query_groups, kv_len, head_size, device='cuda', dtype=torch.float16)

# Compute attention with weights
output, weights = wrapper.scaled_dot_product_attention(
    query=query,
    key=key,
    value=value,
    scale_factor=1.0 / (head_size ** 0.5),
    return_attn_weights=True,
)

print(f"Output shape: {output.shape}")    # (2, 32, 128, 64)
print(f"Weights shape: {weights.shape}")  # (2, 8, 256)
print(f"Weights dtype: {weights.dtype}")  # torch.float32
```

### Integration with MultiHeadSelfAttention

The typical usage is through `MultiHeadSelfAttention` which handles kernel selection automatically:

```python
from litgpt.config import Config
from keys_values.attention import MultiHeadSelfAttention, DefaultKeysAndValues

# Create attention module
config = Config(...)  # Your model config
attention = MultiHeadSelfAttention(config)

# Prepare inputs
query = ...  # (batch_size, n_heads, q_len, head_size)
keys_values = DefaultKeysAndValues(keys, values)

# Call attention - FlashInfer used automatically when return_attn_weights=True
output, attn_weights = attention(
    query=query,
    k_and_v=keys_values,
    block_idx=0,
    input_pos=0,
    return_attn_weights=True,  # This triggers FlashInfer usage
)
```

## Processing Modes

The wrapper automatically selects the appropriate processing mode:

| Condition | Mode | Kernel Used |
|-----------|------|-------------|
| `q_len >= kv_len` | Standard (Prefill) | `sdpa_prefill` |
| `q_len < kv_len` | Chunk Processing (Decode) | `sdpa_decode` |
| `chunk_size` provided and `q_len > chunk_size` | Long Sequence Chunking | `sdpa_prefill` (chunked) |

### Prefill Mode (Standard)
Used during the initial prompt processing when query length equals or exceeds KV length.

### Decode Mode (Chunk Processing)
Used during token generation when processing few query tokens against a large KV cache.

### Long Sequence Chunking
For memory management when processing very long sequences:

```python
output, weights = wrapper.scaled_dot_product_attention(
    query=query,
    key=key,
    value=value,
    scale_factor=scale_factor,
    return_attn_weights=True,
    chunk_size=1024,  # Process in chunks of 1024 tokens
)
```

## Supported Features

| Feature | Supported |
|---------|-----------|
| float16 / bfloat16 | Yes |
| float32 | No (falls back to eager) |
| Grouped Query Attention (GQA) | Yes |
| Causal masking | Yes |
| Sliding window attention | Yes |
| Attention weights return | Yes |
| Token positions (sparse attention) | Yes |
| Attention logit softcapping | No (falls back to eager) |

## Fallback Behavior

The wrapper automatically falls back to the eager implementation when:

1. Kernels fail to load (CUDA unavailable, compilation error)
2. Input dtype is float32 (kernels only support 16-bit)
3. Kernel execution fails for any reason

Fallback is transparent - the same API returns correct results:

```python
# This works regardless of kernel availability
output, weights = wrapper.scaled_dot_product_attention(...)
```

## Verifying Backend Equivalence

Use the built-in verification utilities to ensure kernels produce correct results:

```python
from keys_values.flashinfer_wrapper import verify_backend_equivalence

query = torch.randn(2, 4, 8, 64, device='cuda', dtype=torch.float16)
key = torch.randn(2, 2, 16, 64, device='cuda', dtype=torch.float16)
value = torch.randn(2, 2, 16, 64, device='cuda', dtype=torch.float16)

result = verify_backend_equivalence(
    query=query,
    key=key,
    value=value,
    scale_factor=0.125,
    return_attn_weights=True,
    rtol=1e-4,
    atol=1e-6,
)

if result.is_equivalent:
    print(f"Backends match! Max diff: {result.output_max_diff:.2e}")
else:
    print(f"Backends differ: {result.message}")
```

## Performance Characteristics

Based on benchmarks with context length 32768:

| Operation | vs Eager | vs PyTorch SDPA |
|-----------|----------|-----------------|
| Prefill | 9-29x faster | ~0.85x (competitive) |
| Decode | 0.5x slower | 0.16x slower |

**Note:** Decode performance is slower due to per-batch-item iteration in the current implementation. This is a known issue tracked in `sdpa_decode.cu:462-463`.

## Building the Extension

The kernels are built automatically during package installation:

```bash
# Activate virtualenv
source /fsx/pvihang/virtualenvs/keyval_venv/bin/activate

# Build extension
python setup.py build_ext --inplace
```

Required build flags (already in `setup.py`):
```python
"-U__CUDA_NO_HALF_OPERATORS__",
"-U__CUDA_NO_HALF_CONVERSIONS__",
"-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
"-U__CUDA_NO_HALF2_OPERATORS__",
```

## Key Files

| File | Description |
|------|-------------|
| `keys_values/flashinfer_wrapper.py` | Python wrapper with fallback logic |
| `keys_values/attention.py` | Integration point (`MultiHeadSelfAttention`) |
| `keys_values/csrc/kernels/sdpa_prefill.cu` | Prefill kernel dispatch |
| `keys_values/csrc/kernels/sdpa_decode.cu` | Decode kernel dispatch |
| `setup.py` | Build configuration |

## Known Limitations

1. **Decode performance**: Currently slower than eager due to per-batch iteration
2. **float32 not supported**: Kernels only support 16-bit types (float16/bfloat16)
3. **Attention logit softcapping**: Not supported, falls back to eager
4. **Serialization**: Attention state serialization (Requirement 6) not yet implemented

## Troubleshooting

### Kernels not loading

Check if CUDA is available and extension is compiled:

```python
from keys_values.flashinfer_wrapper import FlashInferSDPA
wrapper = FlashInferSDPA()
print(f"Available: {wrapper.available}")

# Check for load errors
from keys_values import flashinfer_ops
if not flashinfer_ops.is_available():
    print(f"Load error: {flashinfer_ops.get_load_error()}")
```

### Numerical differences

If you see numerical differences between FlashInfer and eager:

```python
# Use tighter tolerances for debugging
result = verify_backend_equivalence(
    ...,
    rtol=1e-3,  # Relax tolerance for float16
    atol=1e-5,
)
```

### Memory issues

For long sequences, use chunk processing:

```python
output, weights = wrapper.scaled_dot_product_attention(
    ...,
    chunk_size=512,  # Smaller chunks use less memory
)
```
