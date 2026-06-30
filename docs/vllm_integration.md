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

### Why target the latest stable, and what it costs

Agreed: we track the most recent stable vLLM and do not try to stay compatible
with older lines. Integrating against the latest is *easier* on the axis that
matters most here — dependencies. A recent vLLM resolves onto a recent torch,
so vLLM + LitGPT + keys_values share one environment (the original driver for
the retarget).

The drawback is not difficulty of integration but **stability of the seams we
hook**. The V1 extension points we rely on (`KVCacheSpec`, the
`SingleTypeKVCacheManager` family, `GPUModelRunner.get_kv_cache_spec`, the
`register_custom_kv_cache_specs` platform hook) are engine internals, not a
frozen public plugin API. They can and do shift between minor releases, so:

- We pin a known-good vLLM (currently 0.23.0) and bump deliberately, not
  automatically.
- We keep the bridge surface small and isolated in `keys_values/vllm/` so a
  vLLM bump touches few files.
- Parity tests (phase 4) guard against silent behavior drift after a bump.

So "harder against the recent one" is really "we accept a moving-target
internal API in exchange for a single, modern environment" — a good trade for
this library, which has no reason to serve on an old vLLM.

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

**Design principle — minimize what we impose on users.** Users should get only
the part that is genuinely ours and not already in vLLM: the selective
eviction policies (H2O, qH2O, smart-lastrec, ...). Anything vLLM already does
well — paging, scheduling, kernels, model loading — we reuse rather than
re-implement or wrap. Concretely: no forked vLLM, no custom model registration
where vLLM's own model works, and the bridge stays torch-only and litgpt-free
so it adds no LitGPT footprint to a serving environment. The ideal user-facing
surface is a single policy/config flag.

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

## vLLM paged KV layout: what a "block" is

This answers the questions raised in review about how vLLM stores KV and how it
differs from the dense keys_values layout.

**Dense (keys_values).** One dense tensor per layer,
`(batch, n_query_groups, cache_length, head_size)`, plus `token_pos` of shape
`(batch, n_query_groups, cache_length)` recording which logical token sits in
each slot. Eviction is per `(batch, head, slot)`: any single slot can be
overwritten independently.

**Paged (vLLM V1).** KV is stored in fixed-size **blocks** drawn from a shared
pool. For one KV cache group (a set of layers with the same spec) the physical
storage is, per layer, a tensor shaped like:

```
(2, num_blocks, block_size, num_kv_heads, head_size)
 ^K/V  ^pool     ^tokens/blk ^heads       ^feature
```

In vLLM's **V1 implementation** a block is a contiguous slab holding
`block_size` consecutive tokens' keys and values for **all** KV heads of that
layer (heads are the `num_kv_heads` axis *inside* the block, addressed by a
single per-group block table — not a separate table per head). In the spike
(Qwen2.5-0.5B) `block_size = 16`, `num_kv_heads = 2`, `head_size = 64`,
`num_gpu_blocks = 102197`.

**Paper design space vs. vLLM's implementation (review point).** The original
PagedAttention paper notes two valid layouts and is explicit that they are
interchangeable: KV for all heads can live in *one* block, **or** "the key and
value vectors at different heads and layers can each have a separate block and
be managed in separate block tables ... we choose the second one for easy
implementation." So per-head block tables are *not* ruled out in principle, and
the reviewer is right to flag the earlier categorical claim. The constraint we
rely on is narrower and empirical: **vLLM's V1 attention specs as shipped**
(`FullAttentionSpec`, `SlidingWindowSpec`, ...) use the all-heads-in-one-block
layout with a single block table per KV-cache group — confirmed by the spike's
per-layer tensor shape `(2, num_blocks, block_size, num_kv_heads, head_size)`
and the single per-request block table. We integrate against that
implementation, not the paper's full design space, so for our purposes the
block is the smallest unit across all heads. Splitting per head would mean
defining per-head KV-cache groups / specs (a much larger change to vLLM's
manager and metadata machinery), which is out of scope for this bridge.

**Indexing.** To address a single KV vector you need:
`(layer/group, physical block_id, offset ∈ [0, block_size), kv_head, feature)`.
A request does not own a dense range; it owns a **block table** per group — an
ordered list of physical `block_id`s. A logical token at position `t` maps as:

```
logical_block = t // block_size
physical_block = block_table[logical_block]
offset         = t % block_size
```

Blocks are reference-counted in the `BlockPool`; "eviction" means returning a
block to the pool (and, for prefix caching, possibly sharing it across
requests).

**Granularity (the line-56 question).** The smallest allocatable / evictable
unit is **one block = `block_size` tokens (16 here), for an entire layer across
all KV heads**, uniform within a KV cache group. It is a per-model config value,
not something we choose per policy. Two consequences for us:

- **Per-request length can vary** — each sequence has its own block table and
  its own block count, so effective `cache_length` differs per request. This is
  the one axis where vLLM is *more* flexible than our dense form.
- **Per-head length cannot vary** — in vLLM V1 as shipped, the block table and
  `block_size` are shared across all KV heads in a layer, so you cannot free a
  block for one head while keeping it for another. (This is an implementation
  property of vLLM's standard specs, not a hard law of PagedAttention — see the
  paper-design-space note above — but it is the reality the bridge targets.)
  This is why faithful per-head H2O is not expressible at block granularity
  under the existing specs and pushes H2O toward either a block-level
  approximation (Option A) or the dense-cache backend (Option B).

Positional policies (lastrec, smart-lastrec) only need to decide *which logical
positions to drop*, which maps cleanly onto freeing blocks — so they fit the
paged model well. Score-based per-head policies (H2O) are the hard case.

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
proves the data flow end-to-end with idiomatic V1 machinery (done in task 2.3).
Next, extend to **`smart-lastrec`** — still positional and weight-free, but with
per-request (batch-dependent, not head-dependent) initial regions, which is the
first policy that is genuinely unique to keys_values. Only then evaluate
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
2. **lastrec via Option A** (DONE, task 2.3): custom `KVCacheSpec` + manager;
   output parity vs. native sliding window confirmed (see below).
3. **smart-lastrec via Option A** (NEXT): the recommended next policy. It is
   genuinely *not* something vLLM already provides, and — crucially — it needs
   **no attention weights**, so it sidesteps the hardest constraint for now. Its
   eviction decisions depend on the **batch** dimension (per-request initial /
   "grace" regions) but **not** on head position, which fits the paged
   block model: per-request block tables already give us per-request variation,
   and we never need to diverge across heads. Implement `SmartLastRecManager` as
   a subclass of `LastRecManager` (`keys_values/vllm/managers.py` is structured
   to extend here). Upstream `smart-lastrec` is being refactored for left
   padding; align the bridge with that once it lands.
4. **H2O**: only after the positional policies land. Decide block-granularity
   approximation vs. dense-cache backend; route attention-weight computation
   through the existing FlashInfer + Triton score-sum machinery.
5. **Eval + parity tests** vs. the LitGPT inference path, then long-context
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
