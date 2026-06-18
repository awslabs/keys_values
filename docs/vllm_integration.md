# vLLM Integration

Status: **early spike** (branch `vllm-integration`)

Target vLLM version: **0.6.5** (pinned to match the reference docs:
<https://docs.vllm.ai/en/v0.6.5/models/adding_model.html>). This is a V0-engine
release; we deliberately avoid the experimental V1 engine for the first pass.

## Goal

Bring this library's *selective KV cache policies* (H2O, qH2O, smart-lastrec,
lastrec, ...) to vLLM, so users can run long-context inference with advanced
cache eviction while benefiting from vLLM's serving stack. The README states the
strategy directly: reuse vLLM's fast kernels and multi-device machinery, but
plug in our KV cache abstractions and eviction policies.

This is **not** primarily a "register a new model" task (which is what the linked
doc describes). A Qwen2.5 model already runs in vLLM unchanged. The hard part is
the **KV cache / attention layer**, because the two systems represent the cache
in fundamentally different ways.

## The core impedance mismatch

| Concern | keys_values | vLLM 0.6.5 |
|---|---|---|
| Cache layout | dense `(batch, n_query_groups, cache_length, head_size)` + `token_pos` book-keeping | paged blocks, `get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)` |
| Eviction granularity | per `(batch, head, slot)` | per block, managed by the scheduler / block manager |
| Eviction decision | inside `KVCache.forward`, sometimes needs **attention weights** back from SDPA | none; PagedAttention kernels do not return attention weights |
| Attention call | `KVCache.forward(query, key, value, token_idx)` does eviction + SDPA together | `AttentionImpl.forward(query, key, value, kv_cache, attn_metadata, ...)` |

Two consequences drive the design:

1. **Attention weights.** H2O-family policies score slots by summed attention
   weight. vLLM's PagedAttention kernels don't expose weights. The library
   already solved this for its own stack via vendored FlashInfer kernels + a
   Triton score-sum kernel (`keys_values/attention/`, `kvcache/attn_weights.py`).
   That machinery has to be reachable from the vLLM attention path.
2. **Eviction granularity.** vLLM's block manager owns the cache and evicts whole
   blocks for *all* heads at once. Our policies evict per `(head, slot)`. These
   models do not compose cleanly.

## vLLM 0.6.5 extension points (mapped)

From `vllm/attention/backends/abstract.py` (v0.6.5):

- `AttentionBackend` — static factory: `get_name`, `get_impl_cls`,
  `get_metadata_cls`, `get_state_cls`, `get_builder_cls`,
  `get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)`,
  `swap_blocks`, `copy_blocks`.
- `AttentionImpl.forward(query, key, value, kv_cache, attn_metadata, k_scale,
  v_scale, attn_type, output)` — the per-layer hot path. `query/key/value` are
  flattened token tensors; `kv_cache` is the paged buffer; `attn_metadata`
  carries `slot_mapping`, the prefill/decode split, block tables, seq lens.
- `AttentionMetadata` — `num_prefills`, `num_prefill_tokens`,
  `num_decode_tokens`, `slot_mapping`, plus `prefill_metadata` /
  `decode_metadata` splits.
- `AttentionMetadataBuilder.build(...)` — builds on-device metadata per step.
- `AttentionState` — CUDA-graph lifecycle hooks.

Related (not in abstract.py, to confirm against source during the spike):
- `vllm/attention/selector.py` — backend selection (`VLLM_ATTENTION_BACKEND`).
- `vllm/worker/cache_engine.py` — allocates KV tensors from `get_kv_cache_shape`.
- `vllm/core/block_manager*` + scheduler — paged block lifecycle / eviction.
- `ModelRegistry` / out-of-tree model plugins — only relevant if we register a
  custom attention-bearing model.

## Candidate architectures

**Option A — custom AttentionBackend that owns its own dense cache.**
Implement an `AttentionImpl.forward` that ignores vLLM's paged `kv_cache` and
instead maintains keys_values' dense KV buffers + `token_pos` per sequence,
running our SDPA + eviction. Most faithful to the policies (keeps per-head
eviction and attention-weight scoring). Cost: fights vLLM's memory model and
scheduler, which assume they own the cache; CUDA graphs and block swapping need
care. Highest fidelity, highest effort.

**Option B — block-level eviction hook on top of PagedAttention.**
Keep PagedAttention and add scoring/eviction at the block-manager level. Much
less invasive, but loses per-head granularity and cannot use attention-weight
scoring (no weights from the kernel), so it cannot faithfully reproduce H2O.

**Recommendation:** pursue Option A, validated incrementally. Start with the
`lastrec` policy because it is **positional** — eviction depends only on
`token_pos`, needs no attention weights — so it exercises the full data flow
without the hardest dependency. Add H2O (attention-weight scoring) only after the
dense-cache data flow is proven end to end.

## Phased plan

1. **Feasibility spike** (current): get baseline Qwen2.5-0.5B running in vLLM
   0.6.5 on a GPU box; dump the resolved attention backend, KV cache shape, and
   `attn_metadata` structure at runtime. Confirm the extension points above
   against the installed source. Script: `examples/vllm_spike.py`.
2. **Bridge module** `keys_values/vllm/`: a custom `AttentionBackend` /
   `AttentionImpl` hosting `lastrec` over a dense per-sequence cache. Validate
   output parity vs. the LitGPT path on short sequences.
3. **H2O**: route attention-weight computation through the existing FlashInfer +
   Triton score-sum machinery; wire summed weights into the H2O scorer.
4. **Eval + parity tests** against the existing LitGPT inference path, then
   long-context benchmarks.

## Environment notes

- Local dev machine is macOS / no CUDA: vLLM cannot run here. Use it for source
  reading, design, and writing scripts only.
- Real runs happen on a Linux + NVIDIA GPU box (the repo's docs reference AWS
  A100 instances; see `docs/launch_instance.md`).
- vLLM 0.6.5 pins an older torch (~2.5.x); keep it in a **separate venv** from
  the main keys_values env (torch 2.12 here) to avoid clobbering it.

## Open questions

- Can Option A coexist with vLLM's scheduler without rewriting the block
  manager, or do we need a custom `cache_engine` too?
- CUDA-graph capture with a non-paged, dynamically-evicting cache — feasible in
  0.6.5, or run eager-only first?
- Batch handling: keys_values fixes batch size at prefill; vLLM batches
  heterogeneous requests. How do per-sequence dense caches map onto a batched
  `AttentionImpl.forward`?
