# FlashInfer CUDA Kernel Integration

## Motivation

Standard attention libraries (Flash Attention, PyTorch SDPA) do not expose per-position attention weights during decode. The keys_values package needs these weights for H2O (Heavy-Hitter Oracle) cache eviction. This work vendors FlashInfer's CUDA primitives and builds custom kernels that return attention weights efficiently.

## Architecture

### Vendored Files

FlashInfer header files are vendored into `keys_values/csrc/flashinfer/`. These provide low-level primitives used by our custom kernels:

| Primitive | Purpose |
|-----------|---------|
| `vec_t<T, N>` | Vectorized 128-bit loads/stores for Q, K, V |
| `state_t<VEC_SIZE>` | Online softmax state (output, max, denominator) |
| `ptx_exp2(x)` | Single PTX instruction base-2 exponential |
| `ptx_rcp(x)` | Fast reciprocal |
| `shfl_xor_sync()` | Warp-level butterfly reduction (no shared memory) |

### Custom Kernels

| File | Purpose |
|------|---------|
| `keys_values/csrc/kernels/sdpa_prefill.cu` | Prefill kernel dispatch |
| `keys_values/csrc/kernels/sdpa_decode.cu` | Decode kernel with attention weights |
| `keys_values/csrc/kernels/sdpa_decode.cuh` | Decode parameter structures |
| `keys_values/csrc/bindings.cpp` | PyTorch/PyBind11 bindings |

### Python Layer

| File | Purpose |
|------|---------|
| `keys_values/flashinfer_ops.py` | Kernel loading module |
| `keys_values/flashinfer_wrapper.py` | `FlashInferSDPA` class with eager fallback |
| `keys_values/attention.py` | Integration point (`MultiHeadSelfAttention`) |

### Dispatch Strategy

```
launch_batch_decode_attention()
  |
  +-> Path A: Real FlashInfer kernels (per-batch loop)
  |   When: no token_positions, no attn_weights, causal, supported head_dim
  |
  +-> Path B: optimized_batched_decode_kernel
  |   When: GQA divisible, head_dim in {64, 128, 256}, fp16/bf16
  |
  +-> Path C: batched_tiled_decode_attention_kernel (generic fallback)
      When: float32, unsupported head_dim, any other edge case
```

FlashInfer kernels are auto-selected when `return_attn_weights=True`, the wrapper is available, and `config.attention_logit_softcapping` is `None`.

## Decode Kernel Design

The optimized decode kernel (`optimized_batched_decode_kernel`) was the main engineering effort.

### Thread Model

```cpp
Grid:  dim3(batch_size, num_qo_heads)
Block: dim3(BDX, 1, BDZ)   // BDX = head_dim / VEC_SIZE

head_dim=64:  VEC_SIZE=8, BDX=8,  BDZ=16 -> 128 threads
head_dim=128: VEC_SIZE=8, BDX=16, BDZ=8  -> 128 threads
head_dim=256: VEC_SIZE=8, BDX=32, BDZ=4  -> 128 threads
```

BDX threads span the head dimension. BDZ threads tile over the KV sequence in parallel. GQA is handled via L2 cache reuse across blocks (no shared memory).

### Key Optimizations

1. **Q in registers** — loaded once, reused across all KV tiles. Eliminates redundant global reads (128K per head at long context).
2. **Single shared memory buffer** — K and V share the same smem buffer (loaded sequentially). Halves smem usage from 64KB to 32KB, improving occupancy.
3. **Vectorized dot product + warp reduction** — `vec_t` loads + `shfl_xor_sync` butterfly reduction. No `__syncthreads()` for the dot product.
4. **`ptx_exp2` softmax** — single PTX instruction vs multi-instruction `expf()`.
5. **`state_t` online softmax** — running output/max/denominator merged across BDZ via `state_t::merge()`.
6. **Grid sizing for occupancy** — `grid(batch_size, num_qo_heads)` gives 4x more blocks than `grid(batch_size, num_kv_heads)` for GQA-4.

### Logits Caching for Attention Weights

The attention weights pass previously accounted for 75-80% of decode time because it recomputed Q*K from global memory. The fix: cache pre-softmax logits (`float* logits_tmp`, shape `[batch, qo_heads, kv_len]`) during the main attention pass, then read them sequentially in the weights pass.

Memory overhead: `batch * qo_heads * kv_len * 4 bytes`. At bs=4, 128K context, 32 heads: 64 MB (acceptable on a 40GB GPU).

### Kernel Phases

```
Phase 0: Load Q into registers, read input_pos[batch_idx]
Phase 1: Tiled attention loop over KV
          - Cooperative K load -> smem
          - Q*K dot product + warp reduction
          - Causal mask via token_positions
          - Cache logits (if return_weights)
          - Online softmax update via ptx_exp2
          - Cooperative V load -> smem (reuse K buffer)
          - Accumulate weighted V into state_t
Phase 2: Cross-BDZ merge via shared memory (state_t::merge)
Phase 3: Normalize output, write to global memory
Phase 4: Attention weights from cached logits + atomicAdd across QO heads
```

## Building

```bash
# Activate virtualenv
source /fsx/pvihang/virtualenvs/keyval_venv/bin/activate

# Build extension in-place
python setup.py build_ext --inplace

# Or install editable
pip install -e .
```

The build uses PyTorch's `cpp_extension` with conditional CUDA compilation. If CUDA is unavailable, the package falls back to eager implementations.

Key build flags (in `setup.py`):
- `-U__CUDA_NO_HALF_OPERATORS__` etc. — re-enables half-precision ops that PyTorch disables
- `--expt-relaxed-constexpr`, `--expt-extended-lambda` — required by FlashInfer headers
- Targets SM 70, 75, 80, 86, 89, 90

## Benchmark Results

Target: NVIDIA A100-SXM4-40GB, Qwen3-4B (32 QO heads, 8 KV heads, head_dim=128, GQA ratio=4), float16.

`flashinfer` and `eager` return attention weights; `pytorch` does not.

### Prefill

9-29x faster than eager, competitive with PyTorch SDPA (~0.85x).

### Decode — Batch Size 4 (primary serving target)

| Context | flashinfer | eager | pytorch | fi/eager | fi/pytorch |
|--------:|-----------:|------:|--------:|---------:|-----------:|
|      4K |     0.76ms | 1.36ms|  0.74ms |   1.79x  |     0.97x  |
|      8K |     1.44ms | 2.42ms|  1.38ms |   1.68x  |     0.96x  |
|     16K |     2.74ms | 4.62ms|  2.64ms |   1.69x  |     0.96x  |
|     32K |     5.47ms | 9.17ms|  5.37ms |   1.68x  |     0.98x  |
|     64K |    10.88ms |19.61ms| 12.65ms |   1.80x  |     1.16x  |
|    128K |    21.79ms |   N/A | 25.39ms |      N/A |     1.17x  |

At 64K-128K context, flashinfer is faster than PyTorch SDPA despite also returning attention weights.

### Decode — Batch Size 1

| Context | flashinfer | eager | pytorch | fi/eager | fi/pytorch |
|--------:|-----------:|------:|--------:|---------:|-----------:|
|      4K |     0.55ms | 0.47ms|  0.24ms |    0.85x |     0.44x  |
|     32K |     4.12ms | 2.46ms|  1.40ms |    0.60x |     0.34x  |
|    128K |    16.15ms | 9.99ms|  6.24ms |    0.62x |     0.39x  |

bs=1 is slower due to GPU underutilization: `grid(1, 32)` = 32 blocks for 108 SMs.

## Supported Features

| Feature | Supported |
|---------|-----------|
| float16 / bfloat16 | Yes |
| float32 | No (falls back to eager) |
| Grouped Query Attention (GQA) | Yes |
| Causal masking | Yes |
| Token positions (sparse attention) | Yes |
| Sliding window attention | Yes |
| Attention weights return | Yes |
| Chunk processing (long sequences) | Yes |
| Attention logit softcapping | No (falls back to eager) |

## API

```python
from keys_values.flashinfer_wrapper import FlashInferSDPA

wrapper = FlashInferSDPA()

output, weights = wrapper.scaled_dot_product_attention(
    query=query,           # (batch, n_head, q_len, head_dim)
    key=key,               # (batch, n_kv_heads, kv_len, head_dim)
    value=value,           # (batch, n_kv_heads, kv_len, head_dim)
    scale_factor=1.0 / (head_dim ** 0.5),
    return_attn_weights=True,
    token_positions=None,  # (batch, n_kv_heads, kv_len), optional
    input_pos=0,
    sliding_window_size=None,
    chunk_size=None,       # for long sequence chunking
)
# output: (batch, n_head, q_len, head_dim)
# weights: (batch, n_kv_heads, kv_len), float32
```

Typically used through `MultiHeadSelfAttention` which selects FlashInfer automatically when `return_attn_weights=True`.

## Known Limitations and Future Work

1. **Decode bs=1 performance** — 0.34-0.85x of PyTorch due to low SM occupancy. Split-KV parallelism could help but adds complexity. Less important for batched serving.
2. **Serialization of attention state** — not yet implemented. Needed for resumable/distributed inference.
3. **Sorting overhead** — when token positions are out of order in the KV cache, we currently sort first then pass in. Sorting is expensive. Smaller chunk sizes help (2048-4096 range).
