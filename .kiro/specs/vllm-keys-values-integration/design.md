# Design Document

## Overview

This design integrates keys_values' selective KV cache policies into vLLM 0.23
(V1 engine). It is organized around one pivotal decision — **how to implement
H2O** — because that choice determines the module boundaries, the attention
path, and how much of vLLM's machinery we keep.

The guiding principle: keep as much of vLLM (paged attention, scheduling,
batching, kernels) as possible, and only replace the parts that selective
eviction genuinely requires. Positional policies (lastrec) fit vLLM's existing
abstractions almost exactly; H2O does not, and most of this document is about
closing that gap responsibly.

All work targets the single combined env (vLLM 0.23 + LitGPT + keys_values on
torch ≥2.7), eager mode first (no CUDA graphs), single GPU first.

## Architecture

Per-step flow relevant to us:

1. Scheduler picks requests and token budgets.
2. The **KV cache manager** (`SingleTypeKVCacheManager` per attention type)
   allocates/frees blocks via the `BlockPool`. Eviction today is **positional**
   (`get_num_skipped_tokens(num_computed_tokens)`) and **block-granular**
   (block_size=16). Managers never see attention scores.
3. The **model runner** runs the forward pass; each attention layer calls the
   resolved **attention backend** (FlashAttention 2 here) over the block table /
   slot mapping the manager produced.
4. The sampler produces tokens.

Registration: custom cache formats register via
`KVCacheSpecRegistry.register(spec_cls, manager_cls, uniform_type_base_spec=...)`.
The platform hook `register_custom_kv_cache_specs(vllm_config)` is the seam to
register ours without patching vLLM. `get_manager_for_kv_cache_spec` then maps a
spec instance to our manager.

Key consequence: **the manager decides block layout, the backend computes
attention.** They are decoupled and communicate only through block tables. Any
score-based policy must (a) get scores out of the attention step and (b) feed
them into the manager before the next step — a channel that does not exist today
and which we must add.

## The H2O decision (centerpiece)

H2O evicts the KV entries with the lowest **accumulated attention weight**. In
keys_values this is computed per `(batch, head, slot)`, requires SDPA to return
summed attention weights, and evicts individual slots. Three properties clash
with vLLM V1:

- **Scores, not positions.** The manager API frees a contiguous skipped prefix;
  H2O frees arbitrary low-score entries.
- **Per-head.** vLLM frees a block for *all* KV heads at once; H2O scores and
  evicts per head.
- **Attention weights.** FlashAttention 2 (the resolved backend) does not return
  weights. This is a hard dependency shared by every option below.

### Shared sub-problem: getting attention weights

Both options need summed attention weights per KV position. Options, best first:

1. **FlashInfer with score output.** FLASHINFER is available on this GPU, and
   keys_values already vendors modified FlashInfer kernels + a Triton score-sum
   kernel (`keys_values/attention/`, `kvcache/attn_weights.py`) precisely for
   H2O. Plan: force the FlashInfer backend and route its attention call through
   the keys_values weight-returning path.
2. **Custom Triton attention** producing both output and per-position score sum
   (reuse the existing score-sum kernel; pay a second pass).
3. **Recompute scores** separately from the attention output (correct, slow;
   fallback only).

This sub-problem is on the critical path for H2O regardless of A vs B, so it is
prototyped first within phase 4.

### Option A — block-level H2O inside vLLM's paged cache (recommended)

Approximate H2O at **block granularity**, keeping vLLM's paged storage and
kernels.

- A custom `KVCacheSpec` (subclass of `AttentionSpec`) carries the H2O config
  (recent-window size R, target cache length L).
- A custom `SingleTypeKVCacheManager` overrides eviction: instead of
  `get_num_skipped_tokens` (positional), it frees the **lowest-scoring blocks**
  among the non-recent region once the per-request block count exceeds L/block.
  The most-recent R tokens are always kept (a positional "recent window",
  exactly like `SlidingWindowManager` keeps the tail).
- **Per-block score** = sum over the block's tokens and over KV heads of the
  accumulated attention weight. Head dimension is collapsed (this is the
  approximation vs. keys_values' per-head H2O).
- **Score feedback channel**: the attention step writes per-(request, block)
  scores into a side buffer keyed by block id; the manager reads it at the start
  of the next step to choose evictions. Implemented as a small registry object
  shared between the custom backend wrapper and the custom manager.

Pros: keeps paged attention, batching, multi-request scheduling, and vLLM's
kernels; far more maintainable; survives vLLM upgrades better. Cons:
block-granular + head-collapsed, so it is an *approximation* — accuracy vs.
keys_values H2O must be measured (Requirement 4.3/4.4). Prefix caching must be
disabled when active (score eviction breaks content-addressed block sharing).

### Option B — custom attention backend owning a dense cache (fallback)

A V1 attention backend that ignores vLLM's paged `kv_cache` and maintains
keys_values-style dense per-request buffers + `token_pos`, running keys_values
SDPA + per-(b,h) eviction directly.

Pros: **faithful** H2O — per-head, per-slot, exact; reuses keys_values cache
classes almost verbatim. Cons: fights vLLM's memory accounting, scheduler,
prefix caching, and CUDA graphs; effectively re-implements keys_values inference
inside a vLLM shell, so the practical gain over just running keys_values is
small; highest effort and most fragile across versions.

### Decision

**Pursue Option A (block-level), measure accuracy against keys_values H2O, and
keep Option B as a documented fallback** if block-level accuracy is unacceptable
for the target tasks. Rationale: Option A delivers the actual integration value
(advanced eviction *inside vLLM's serving stack*), whereas Option B mostly
relocates keys_values into vLLM without gaining vLLM's strengths. The per-head
vs. per-block fidelity question (Requirement: Open Decisions) is resolved
empirically in phase 4; if per-head is mandatory, escalate to Option B.

## Components and Interfaces

New package `keys_values/vllm/` (torch-only; no litgpt import):

- `specs.py` — `LastRecSpec`, `H2OSpec` (subclasses of vLLM `AttentionSpec` /
  `FullAttentionSpec`), carrying policy config.
- `managers.py` — `LastRecManager` (mirrors `SlidingWindowManager` via
  `get_num_skipped_tokens`), `H2OManager` (score-based block eviction + recent
  window).
- `scores.py` — the score feedback channel: per-(request, block) accumulated
  attention weights, written by the attention path, read by `H2OManager`.
- `attention.py` — the FlashInfer/Triton weight-returning attention hook that
  feeds `scores.py`; bridges to `keys_values/attention/` and
  `kvcache/attn_weights.py`.
- `registration.py` — registers specs/managers via `KVCacheSpecRegistry` and the
  `register_custom_kv_cache_specs` platform hook; maps a CLI/config string
  (`lastrec`, `h2o`) to a spec + config.
- `config.py` — parse policy name + params (window, cache length) into specs.

keys_values core (`kvcache/`, `attention/`) is reused, not modified, except
where a thin adapter is needed.

## lastrec design (first real slice, Requirement 3)

`lastrec` = keep the most-recently-inserted `cache_length` tokens. This is a
sliding window. Implementation:

- `LastRecSpec(AttentionSpec)` with `cache_length`.
- `LastRecManager(SingleTypeKVCacheManager)` overriding
  `get_num_skipped_tokens(n) = max(0, n - cache_length + 1)` — identical in
  spirit to `SlidingWindowManager`, reusing its `remove_skipped_blocks` and
  `free` semantics.
- Register via `KVCacheSpecRegistry.register(LastRecSpec, LastRecManager, ...)`.
- Validate against vLLM native sliding window with the same window (outputs
  match within tolerance) — this proves the registration + wiring path that
  H2O will reuse. smart-lastrec (grace tokens / regex prefix) layers on top
  later by always-keeping an initial segment in addition to the recent window.

## Data flow for H2O (per step)

1. Manager start-of-step: read per-block scores from `scores.py` for each
   running request; if block count > L/block, free the lowest-scoring
   non-recent blocks (via `BlockPool.free_blocks`), keep the recent-R tail.
2. Forward: attention layer runs through `attention.py` (FlashInfer weight
   path), computes output AND per-position weight sums.
3. End-of-step: aggregate weight sums to per-block scores, accumulate into
   `scores.py` keyed by block id.
4. Sampler proceeds as normal.

Note the one-step delay (score at t influences eviction at t+1); this matches
keys_values H2O semantics.

## Testing Strategy

The full engine cold start is ~160s, far too slow for an inner loop. Therefore:

- **Manager unit tests** drive `LastRecManager` / `H2OManager` against a fake
  `BlockPool` and synthetic token/score streams — no engine, millisecond
  iteration. Assert block-keep/free sets match expected eviction.
- **Score-channel tests** verify weight sums aggregate to the correct per-block
  scores.
- **Parity tests** compare vLLM-path generations and (for H2O) eviction
  decisions against saved keys_values/LitGPT reference outputs on fixed prompts,
  within a documented tolerance. References are generated once with the LitGPT
  path and checked in as fixtures.
- **End-to-end smoke** (slow, run sparingly): `examples/vllm_spike.py`-style
  full generation with each policy enabled.

## Data Models

### Policy specs (`specs.py`)
- `LastRecSpec(AttentionSpec)`: fields from `AttentionSpec` (`block_size`,
  `num_kv_heads`, `head_size`, `dtype`) plus `cache_length: int`.
- `H2OSpec(FullAttentionSpec)`: above plus `cache_length: int` (target L),
  `recent_window: int` (R, always-kept tail), optional `grace_tokens: int`
  (initial always-kept segment for smart-lastrec reuse).

### Per-block score record (`scores.py`)
- Keyed by `(request_id, block_id)`.
- Value: `score: float32` = running sum of attention weight mass landing on the
  block's tokens, summed over KV heads. Updated each step (accumulate), reset on
  block free.
- Auxiliary: `recent_block_ids: list[int]` per request, marking the protected
  recent-window tail that is never score-evicted.

### Manager bookkeeping (reused from base)
- `req_to_blocks: dict[str, list[KVCacheBlock]]` (from
  `SingleTypeKVCacheManager`).
- `num_cached_block: dict[str, int]` (from base; prefix caching disabled for
  H2O, so this stays at 0 for H2O groups).

## Correctness Properties

These are the executable properties the test suite (PBT where practical)
asserts:

### Property 1: Bounded footprint
For any token stream, after each step the number of live (non-null) blocks for a
request under `lastrec`/H2O is `<= cdiv(cache_length, block_size)` (plus the
recent window for H2O).

**Validates: Requirements 3.2, 4.2**

### Property 2: Recent-window retention
The most-recent `recent_window` tokens are never evicted by H2O; only positional
aging beyond the window applies.

**Validates: Requirements 4.2**

### Property 3: Score monotonicity
A block's accumulated score only increases between allocation and free; freeing
a block clears its record.

**Validates: Requirements 4.2**

### Property 4: Eviction selects minimum score
When H2O must free k blocks, it frees the k lowest-scoring eligible (non-recent,
non-null) blocks.

**Validates: Requirements 4.2, 4.3**

### Property 5: lastrec equals sliding window
With equal window size and prefix caching off, `LastRecManager`'s kept/freed
block set equals `SlidingWindowManager`'s for every step of any token stream.

**Validates: Requirements 3.3**

### Property 6: Parity tolerance
vLLM-path generations match the keys_values/LitGPT reference within a documented
tolerance for the same policy and seed.

**Validates: Requirements 3.3, 4.3, 5.2**

## Error Handling

- When H2O is active, **disable prefix caching** (`enable_prefix_caching=False`)
  — score-based eviction is incompatible with content-addressed block sharing in
  the first cut; document and revisit.
- Eager mode only initially (no CUDA graphs) since the cache size changes
  dynamically; document the constraint.
- If the FlashInfer weight path is unavailable on a platform, fall back to the
  Triton score-sum kernel; if neither, H2O raises a clear configuration error
  rather than silently degrading to positional eviction.

## Risks and mitigations

- **Attention-weight extraction is the critical-path unknown.** Mitigate by
  prototyping `attention.py` against FlashInfer first, in isolation, before the
  manager work.
- **Block-level accuracy may be insufficient.** Mitigate by measuring early
  (phase 4) against keys_values H2O; Option B is the documented escalation.
- **vLLM internal churn.** Mitigate by confining contact to the registry +
  platform hook + a single attention wrapper; avoid patching vLLM source.
- **Score-feedback plumbing has no native seam.** Mitigate with a narrow,
  well-tested `scores.py` channel and unit tests that don't need the engine.

## Phasing (maps to tasks)

1. `keys_values/vllm/` skeleton + registration plumbing (no policy yet).
2. lastrec spec + manager + parity vs. native sliding window.
3. Attention-weight extraction prototype (FlashInfer) in isolation.
4. H2O block-level manager + score channel; accuracy vs. keys_values H2O.
5. Parity/eval harness + long-context benchmark.
6. Quantized buffers / qH2O / smart-lastrec (stretch).
