# Requirements Document

## Introduction

Bring keys_values' selective KV cache policies (lastrec, smart-lastrec, H2O,
qH2O, and quantized buffers) to vLLM so users can run long-context inference
with advanced cache eviction while using vLLM's serving stack (fast kernels,
scheduling, multi-device). The integration targets **vLLM 0.23 on the V1
engine** in a single combined environment (vLLM + LitGPT + keys_values on one
torch).

This is a multi-phase effort. The phases are ordered so each delivers a working,
testable slice and de-risks the next.

### Key technical constraints (from V1 source review)

- vLLM V1 `SingleTypeKVCacheManager` makes eviction decisions from **token
  positions only** (`get_num_skipped_tokens`), at **block granularity**. It
  never sees attention weights.
- Positional policies (lastrec / sliding-window-like) map directly onto this
  abstraction.
- H2O-family policies score by **accumulated attention weight, per token/head**.
  This does not fit the position-based, block-granular manager API, and the
  resolved backend (FlashAttention 2) does not return attention weights.
  Faithful H2O therefore requires either (a) a custom V1 attention backend that
  owns a dense cache and runs keys_values SDPA + weight kernels, or (b) a
  block-level score approximation. Choosing between these is a design decision.
- Custom specs/managers register via `KVCacheSpecRegistry.register(spec_cls,
  manager_cls, ...)`; platforms hook `register_custom_kv_cache_specs`.

## Glossary

- **V1 engine**: vLLM's current execution engine (scheduler + KV cache manager
  + model runner) used in 0.23.
- **KVCacheSpec**: per-layer description of the cache format (block_size,
  num_kv_heads, head_size, dtype) in `vllm/v1/kv_cache_interface.py`.
- **SingleTypeKVCacheManager**: per-attention-type block manager that decides
  which blocks to keep/free.
- **lastrec**: keys_values policy keeping the most-recently-inserted
  `cache_length` tokens.
- **H2O**: Heavy-Hitter Oracle; evicts KV entries with the lowest accumulated
  attention weight.
- **Parity**: matching outputs/eviction decisions against the keys_values
  (LitGPT) reference implementation within a tolerance.

## Requirements

### Requirement 1: Combined environment

**User Story:** As a researcher, I want one environment with vLLM, LitGPT, and
keys_values, so I can run integration code and parity references without
switching envs.

#### Acceptance Criteria
1. WHEN the env is built THEN vLLM 0.23, LitGPT, and keys_values SHALL import
   together on a single torch version (>=2.7).
2. WHEN `keys_values.kvcache` is imported in this env THEN it SHALL NOT require
   any conflicting torch downgrade.
3. The setup SHALL be documented in `docs/vllm_integration.md`.

### Requirement 2: Feasibility spike

**User Story:** As a developer, I want the V1 extension points and cache
geometry confirmed at runtime, so phase 2 builds on facts.

#### Acceptance Criteria
1. WHEN the spike runs against Qwen2.5-0.5B THEN it SHALL report the resolved
   attention backend, per-layer cache geometry, and a successful generation.
2. The findings SHALL be recorded in `docs/vllm_integration.md`.

### Requirement 3: lastrec policy executing in vLLM

**User Story:** As a user, I want to run vLLM with the keys_values `lastrec`
policy, so KV cache is bounded by a fixed window during generation.

#### Acceptance Criteria
1. WHEN a custom `KVCacheSpec` + `SingleTypeKVCacheManager` for `lastrec` is
   registered THEN vLLM SHALL load and generate with it enabled.
2. WHEN generating with a window smaller than the sequence length THEN the
   cache footprint SHALL stay bounded by the configured window.
3. WHEN compared against vLLM's native sliding window on an equivalent config
   THEN outputs SHALL match within tolerance (validates the wiring).
4. The policy SHALL be selectable via a documented configuration path.

### Requirement 4: H2O architecture decision and prototype

**User Story:** As a researcher, I want H2O eviction available in vLLM, because
it is the differentiating capability over stock vLLM.

#### Acceptance Criteria
1. The design SHALL document a chosen approach (custom attention backend owning
   a dense cache, OR block-level score approximation) with tradeoffs.
2. WHEN H2O is enabled THEN attention-weight-based eviction SHALL be computed,
   using keys_values' existing FlashInfer/Triton score-sum path where possible.
3. WHEN compared against the keys_values (LitGPT) H2O implementation on a fixed
   prompt THEN eviction decisions and outputs SHALL match within a documented
   tolerance.
4. IF a faithful per-head implementation is infeasible at acceptable cost THEN
   the approximation and its accuracy impact SHALL be documented.

### Requirement 5: Parity and evaluation harness

**User Story:** As a developer, I want automated parity tests against the
LitGPT path, so regressions are caught without manual inspection.

#### Acceptance Criteria
1. There SHALL be tests that exercise the manager/eviction logic WITHOUT
   starting the full vLLM engine (the engine cold start is ~160s).
2. WHEN a policy is changed THEN parity tests against saved LitGPT reference
   outputs SHALL run and report pass/fail.
3. Long-context behavior SHALL be evaluated on at least one benchmark task.

### Requirement 6: Quantized buffers and additional policies

**User Story:** As a user, I want quantized KV buffers and the qH2O / smart-
lastrec variants in vLLM, to reduce memory further.

#### Acceptance Criteria
1. WHEN buffer quantization is enabled THEN it SHALL interoperate with the
   chosen H2O architecture.
2. Additional policies SHALL reuse the registration mechanism from Requirement 3.

## Out of Scope (initial)

- Multi-device / context-parallel execution of the custom policies.
- CUDA-graph capture with dynamically-evicting custom caches (start eager-only).
- Registering keys_values' LitGPT models in vLLM (vLLM uses its own models).

## Open Decisions

- H2O: custom attention backend (Option B) vs. block-level approximation.
- Target models beyond Qwen2.5 (Llama? larger Qwen?).
- Whether per-head eviction is a hard requirement or block-level is acceptable.
