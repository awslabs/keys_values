# Implementation Plan

## Overview

Incremental, test-first build of the vLLM ↔ keys_values integration. Each task
produces a working slice and is verifiable without the ~160s engine cold start
where possible (manager/score logic is unit-tested against a fake `BlockPool`).
Order de-risks the hard part (H2O attention weights) before the full H2O
manager. Tasks map to the phases in `design.md`.

## Tasks

- [ ] 1. Scaffold `keys_values/vllm/` bridge package and registration plumbing
  - Create `keys_values/vllm/__init__.py`, `config.py`, `registration.py` with
    no policy logic yet; ensure import works in the combined env without
    importing litgpt.
  - Implement a `register_policies()` entry that calls
    `KVCacheSpecRegistry.register(...)` and is invokable from vLLM's
    `register_custom_kv_cache_specs` platform hook.
  - Add a smoke test asserting `import keys_values.vllm` succeeds alongside
    `import vllm` and registration runs without error.
  - _Requirements: 1.1, 1.2, 3.1_

- [ ] 2. Implement the `lastrec` policy as a custom spec + manager
- [ ] 2.1 Define `LastRecSpec` and `LastRecManager`
  - `LastRecSpec(AttentionSpec)` carrying `cache_length`.
  - `LastRecManager(SingleTypeKVCacheManager)` overriding
    `get_num_skipped_tokens(n) = max(0, n - cache_length + 1)`, reusing base
    `remove_skipped_blocks`/`free`.
  - Register the pair via `registration.py`.
  - _Requirements: 3.1, 3.4_
- [ ] 2.2 No-engine unit tests for `LastRecManager`
  - Drive the manager against a fake `BlockPool` and synthetic token streams.
  - Assert bounded footprint and that kept/freed block sets equal
    `SlidingWindowManager` for equal window (Property 1, Property 5).
  - _Requirements: 3.2, 5.1_
- [ ] 2.3 End-to-end lastrec generation + sliding-window parity
  - Enable `lastrec` via config; generate with Qwen2.5-0.5B.
  - Compare outputs to vLLM native sliding window (same window) within
    tolerance; record the config path in `docs/vllm_integration.md`.
  - _Requirements: 3.2, 3.3, 3.4_

- [ ] 3. Prototype attention-weight extraction in isolation
- [ ] 3.1 FlashInfer weight-returning attention hook
  - In `attention.py`, force the FLASHINFER backend and route attention through
    the keys_values weight-returning path (`keys_values/attention/`,
    `kvcache/attn_weights.py`) to obtain per-position summed weights.
  - Unit-test the hook on small synthetic Q/K/V against a reference softmax
    weight-sum (no engine).
  - _Requirements: 4.2_
- [ ] 3.2 Triton score-sum fallback and capability detection
  - Wire the existing Triton score-sum kernel as a fallback; detect
    availability and raise a clear error if no weight path exists.
  - _Requirements: 4.2_

- [ ] 4. Implement block-level H2O (Option A)
- [ ] 4.1 Score feedback channel (`scores.py`)
  - Per-`(request_id, block_id)` accumulated score buffer; write from the
    attention path, read by the manager; clear on block free.
  - Unit-test aggregation of per-position weights to per-block scores
    (Property 3).
  - _Requirements: 4.2, 5.1_
- [ ] 4.2 `H2OSpec` + `H2OManager` with score-based eviction
  - `H2OSpec(FullAttentionSpec)` with `cache_length`, `recent_window`,
    optional `grace_tokens`.
  - `H2OManager` frees lowest-scoring non-recent blocks when over budget; always
    retain the recent-R tail; disable prefix caching for the group.
  - No-engine unit tests for Property 1, 2, 4.
  - _Requirements: 4.1, 4.2, 5.1_
- [ ] 4.3 End-to-end H2O generation and accuracy vs. keys_values H2O
  - Run H2O in vLLM on fixed prompts; compare eviction decisions and outputs to
    the keys_values/LitGPT H2O reference within a documented tolerance.
  - Document the block-level approximation and its accuracy impact; record
    whether per-head (Option B escalation) is needed.
  - _Requirements: 4.1, 4.3, 4.4_

- [ ] 5. Parity and evaluation harness
- [ ] 5.1 Reference-fixture generation from the LitGPT path
  - Script to generate and check in reference outputs/eviction traces for fixed
    prompts under each policy.
  - _Requirements: 5.2_
- [ ] 5.2 Parity test suite (Property 6)
  - Tests comparing vLLM-path outputs to fixtures within tolerance; runnable in
    CI without a full engine cold start where possible.
  - _Requirements: 5.1, 5.2_
- [ ] 5.3 Long-context benchmark run
  - Evaluate at least one long-context task; record results in
    `docs/BENCHMARK_RUNS.md`.
  - _Requirements: 5.3_

- [ ] 6. Stretch: quantized buffers and additional policies
- [ ] 6.1 smart-lastrec via grace-token retention on top of `LastRecManager`
  - _Requirements: 6.2_
- [ ] 6.2 Quantized KV buffers interoperating with the chosen H2O path (qH2O)
  - _Requirements: 6.1, 6.2_

## Task Dependency Graph

```json
{
  "waves": [
    { "wave": 1, "tasks": ["1"], "parallel": false },
    { "wave": 2, "tasks": ["2.1", "3.1"], "parallel": true },
    { "wave": 3, "tasks": ["2.2", "3.2"], "parallel": true },
    { "wave": 4, "tasks": ["2.3", "4.1"], "parallel": true },
    { "wave": 5, "tasks": ["4.2"], "parallel": false },
    { "wave": 6, "tasks": ["4.3", "5.1"], "parallel": true },
    { "wave": 7, "tasks": ["5.2", "5.3"], "parallel": true },
    { "wave": 8, "tasks": ["6.1", "6.2"], "parallel": true }
  ]
}
```

```
1 (scaffold + registration)
├── 2.1 LastRecSpec/Manager ──> 2.2 unit tests ──> 2.3 e2e + parity
└── 3.1 FlashInfer weight hook ──> 3.2 Triton fallback
                                      │
            (3.x + 2.x both feed) ────┼──> 4.1 score channel ──> 4.2 H2OManager ──> 4.3 e2e accuracy
                                                                                       │
                                                          5.1 fixtures ──> 5.2 parity ─┤
                                                                           5.3 benchmark
                                                                                       │
                                                                 6.1 smart-lastrec ────┤
                                                                 6.2 quantized/qH2O ───┘
```

- Task 1 blocks everything (registration is the shared entry point).
- Tasks 2 and 3 are independent and can proceed in parallel after 1.
- Task 4 (H2O) requires both the weight path (3) and the manager mechanics
  proven by lastrec (2).
- Task 5 depends on at least one policy (2.3) existing; full parity needs 4.3.
- Task 6 builds on 2 and 4.

## Notes

- All manager/score logic must be unit-testable without starting the vLLM
  engine (cold start ~160s); reserve full-engine runs for the e2e tasks.
- Eager mode only initially (no CUDA graphs); single GPU first.
- Disable prefix caching whenever H2O is active (see design Error Handling).
- Confine vLLM contact to the registry + platform hook + the single attention
  wrapper to limit exposure to vLLM internal churn.
- The H2O block-level vs. per-head (Option B) decision is resolved empirically
  in task 4.3.
