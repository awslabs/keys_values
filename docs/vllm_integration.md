# vLLM Integration

Status: **early spike** (branch `vllm-integration`)

Target vLLM version: **0.23.0** (latest), running on the **V1 engine**.

> History: this was originally scoped against vLLM 0.6.5 (the version in the
> reference docs). We retargeted to 0.23 because 0.6.5 hard-pins
> `torch==2.5.1`, which conflicts with LitGPT (`torch>=2.7`) and would have
> forced a second environment. vLLM 0.23 resolves onto a recent torch (2.11
> in our setup), so **a single env can hold vLLM + LitGPT + keys_values**. The
> tradeoff: the engine internals are completely different (V0 → V1), so the
> old `AttentionBackend`/`AttentionImpl` extension map no longer applies.

## Goal

Bring this library's *selective KV cache policies* (H2O, qH2O, smart-lastrec,
lastrec, ...) to vLLM, so users can run long-context inference with advanced
cache eviction while benefiting from vLLM's serving stack. Reuse vLLM's fast
kernels and multi-device machinery; plug in our cache abstractions and eviction
policies.

This is **not** primarily a "register a new model" task. vLLM ships its own
implementations of Qwen2/Llama/etc. (in `vllm/model_executor/models/`) and loads
HF weights into them — LitGPT is never in the serving path. The work lives at
the **KV cache + attention layer**.

## Environment

One combined env (no more split):

```bash
pip install -U vllm          # 0.23.0, pulls torch 2.11
pip install -e .             # keys_values (torch-free deps)
pip install -U outlines      # clears a stale 0.6.5-era pin
```

Verify: `python -c "import torch, vllm, litgpt, keys_values.kvcache.base"`.
LitGPT (`torch>=2.7`) and vLLM 0.23 (torch 2.11) coexist; the keys_values
`kvcache/` module is pure torch (zero litgpt imports), so it slots into either.

## The impedance mismatch (still the core problem)

| Concern | keys_values | vLLM V1 |
|---|---|---|
| Cache layout | dense `(batch, n_query_groups, cache_length, head_size)` + `token_pos` | paged blocks; per-layer `KVCacheSpec` (block_size, num_kv_heads, head_size, dtype) |
| Eviction granularity | per `(batch, head, slot)` | per block, via a `KVCacheManager` / block pool |
| Eviction decision | inside `KVCache.forward`, H2O needs **attention weights** | block managers decide by position/recency; kernels don't return weights |
| Attention call | `KVCache.forward(query, key, value, token_idx)` does evict + SDPA | model calls `Attention` layer; V1 attention backend + metadata builder |

Two hard constraints persist:
1. **Attention weights.** H2O scores slots by summed attention weight; vLLM
   kernels don't expose them. keys_values already solves this for its own stack
   (`keys_values/attention/`, `kvcache/attn_weights.py` — vendored FlashInfer +
   Triton score-sum). That has to be reachable from the vLLM attention path.
2. **Granularity.** vLLM evicts whole blocks; our policies score per `(head,
   slot)`. Block-level eviction can't reproduce per-head H2O exactly.

## vLLM 0.23 / V1 extension points (mapped from source)

KV cache description — `vllm/v1/kv_cache_interface.py`:
- `KVCacheSpec` (frozen dataclass, base): `block_size`, `page_size_bytes`,
  `max_memory_usage_bytes(...)`. Custom specs register via
  `@register_kv_cache_spec`.
- `AttentionSpec(KVCacheSpec)`: `num_kv_heads`, `head_size`, `dtype`,
  `kv_quant_mode`.
- `FullAttentionSpec`, `SlidingWindowSpec`, `ChunkedLocalAttentionSpec`,
  `MLAAttentionSpec`, `MambaSpec`, ... — **these are the precedent**: vLLM V1
  already expresses non-full-attention cache patterns as first-class specs.
- `KVCacheConfig`: `num_blocks`, `kv_cache_tensors`, `kv_cache_groups`.
- `KVCacheGroupSpec`: layers sharing one block table.

KV cache management — `vllm/v1/core/`:
- `kv_cache_manager.py` (`KVCacheManager`), `block_pool.py`,
  `single_type_kv_cache_manager.py` with `SlidingWindowManager`,
  `ChunkedLocalAttentionManager` — per-pattern managers that free/reuse blocks
  according to the policy. **This is where selective eviction lives in V1.**

Attention — `vllm/v1/attention/backends/`:
- Per-backend metadata builders (flash_attn, flashinfer, triton_attn, ...) plus
  the `Attention` layer the model calls. Backend choice via
  `VLLM_ATTENTION_BACKEND`.

Model / registration:
- `ModelRegistry` / out-of-tree plugins via entry points (only needed if we
  register a custom model; likely we reuse vLLM's Qwen2 + a custom cache path).

## Candidate architectures (revised for V1)

**Option A — custom `KVCacheSpec` + `SingleTypeKVCacheManager`.** Model the
keys_values policy as a V1 cache spec (à la `SlidingWindowSpec`) with a manager
(à la `SlidingWindowManager`) that frees blocks per the eviction policy. This is
the idiomatic V1 path and reuses vLLM's paged kernels. Works cleanly for
**positional** policies (lastrec, sliding-window-like). Limitation: eviction is
per-block, not per-head, and there are no attention weights — so it cannot do
faithful H2O on its own.

**Option B — custom attention backend owning a dense cache.** A V1 attention
backend that bypasses paged storage and keeps keys_values' dense buffers +
`token_pos`, running our SDPA + per-head eviction (and, for H2O, our
attention-weight kernels). Highest fidelity, but fights vLLM's memory manager
and CUDA-graph/scheduler assumptions. Highest effort.

**Recommendation:** start with **Option A + `lastrec`** — positional, no
attention weights, maps directly onto the `SlidingWindowManager` pattern. This
proves the data flow end-to-end with idiomatic V1 machinery. Then evaluate
whether H2O can be approximated at block granularity (Option A) or needs the
dense-cache backend (Option B).

## Spike results (Qwen2.5-0.5B-Instruct, 1× A10G, vLLM 0.23.0 / V1)

Captured by `examples/vllm_spike.py`:
- Architecture: `Qwen2ForCausalLM`; engine: V1; dtype: bfloat16.
- Resolved attention backend: **FLASH_ATTN** (FlashAttention 2). Available on
  this GPU: `['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION']`.
- Cache geometry: `num_kv_heads=2`, `head_size=64`, `num_layers=24`,
  `block_size=16`, `num_gpu_blocks=102197`, `cache_dtype=auto`.
- KV cache capacity: 1,635,152 tokens; max concurrency 798x at 2048 tokens.
- Generation works; engine init (profile + warmup) took ~161 s (one-time).

Implications:
- FlashAttention 2 does **not** return attention weights → confirms H2O (phase 3)
  needs a different path. FLASHINFER is available, which aligns with the
  FlashInfer machinery keys_values already uses for the weight-sum path.
- GQA: 2 KV heads, head_size 64, 24 layers → the per-layer `KVCacheSpec` we
  define must match `FullAttentionSpec(block_size=16, num_kv_heads=2,
  head_size=64, dtype=bf16)`.

## Phased plan

1. **Feasibility spike** (DONE): see results above. Script: `examples/vllm_spike.py`.
2. **lastrec via Option A**: custom `KVCacheSpec` + manager; parity-check output
   vs. the LitGPT path on short sequences.
3. **H2O**: decide block-granularity approximation vs. dense-cache backend;
   route attention-weight computation through the existing FlashInfer + Triton
   score-sum machinery.
4. **Eval + parity tests** vs. the LitGPT inference path, then long-context
   benchmarks.

## Serve-time wiring result (task 2.3)

`lastrec` now runs end-to-end in vLLM 0.23 (V1) and is at output parity with
native sliding window. Mechanism (see `examples/vllm_lastrec_experiment.py`):

- Run the engine in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) so monkeypatches
  apply to the worker.
- Set each `Attention` layer's `sliding_window` to the window size (an **int** —
  a tuple breaks `SlidingWindowSpec.max_admission_blocks_per_request`). vLLM then
  derives a `SlidingWindowSpec` for each layer and masks attention to the window.
- Wrap `GPUModelRunner.get_kv_cache_spec` to convert each derived
  `SlidingWindowSpec` into our `LastRecSpec`, which routes to `LastRecManager`.

Verified on 1x A10G, Qwen2.5-0.5B, window 256: `lastrec` output is byte-identical
to `sliding` for the same prompt/seed. Note: a needle-recall test does NOT
discriminate at window 256, because info diffuses forward ~`window x num_layers`
(~6k tokens) through the residual stream; use a small window (e.g. 32) to make
eviction observable.

Open follow-up: this uses monkeypatches in an experiment harness. A production
path should register via the `register_custom_kv_cache_specs` platform hook and a
config flag rather than patching `get_kv_cache_spec`/layer attributes.

## Open questions

- Can a `SingleTypeKVCacheManager` express keys_values eviction without touching
  the block pool internals?
- H2O at block granularity — acceptable accuracy, or is per-head eviction
  (Option B) required?
- Getting summed attention weights in V1 without a custom kernel per backend.
- CUDA-graph capture with a dynamically-evicting custom manager.
