# GRPO context scaling: dense vs. H2O (sparse) KV cache

Measures what the KeysAndValues sparse KV cache buys for GRPO as prompt/context
length grows: full attention (`dense-default`, cache holds every token) vs. H2O
(`h2o-torch-quantized8`, cache capped at a fixed budget so it evicts once the
prompt exceeds it).

Reproduce with `examples/rl/grpo_context_sweep.py`.

## Setup

- **Hardware**: AWS g5.8xlarge, 1× NVIDIA A10G (24 GB), 124 GB RAM, driver 595, CUDA 13.2.
- **Model**: Qwen2.5-3B-Instruct (bf16), GQA group_size 8 (FlashInfer-compatible).
- **Attention kernels**: vendored FlashInfer SDPA (built for sm_86), returning
  attention weights for H2O scoring.
- **GRPO step**: batch 2 (1 prompt × group 2), `max_new_tokens=32`,
  `chunk_size=1024`, `layers_per_cell=1`, single-epoch (detach) path.
- **H2O cache budget**: 4096 slots (fixed). Dense cache = full sequence length.
- **Timing**: 1 warmup + 3 measured steps per config; per-phase wall-clock (ms),
  CUDA-synchronized. Peak = `torch.cuda.max_memory_allocated`.

## Results

| context | cache | gen (ms) | grad (ms) | total (ms) | peak (GB) |
|--------:|-------|---------:|----------:|-----------:|----------:|
| 2,048   | dense            | 1712 | 3030  | 4767  | 14.34 |
| 2,048   | H2O (cl=2080\*)  | 3222 | 3039  | 6289  | 14.33 |
| 8,192   | dense            | 3893 | 11118 | 15041 | 16.39 |
| 8,192   | H2O (cl=4096)    | 6121 | 16352 | 22501 | 14.84 |
| 16,384  | dense            | 7844 | 24706 | 32619 | 19.77 |
| 16,384  | H2O (cl=4096)    | 9119 | 34196 | 43384 | 14.86 |
| 24,576  | dense            | —    | —     | **OOM** | **OOM** |
| 24,576  | H2O (cl=4096)    | 12115 | 52404 | 64629 | 19.21† |
| 32,768  | dense            | —    | —     | **OOM** | **OOM** |
| 32,768  | H2O (cl=4096)    | 15169 | 70693 | 86007 | 14.89‡ |

\* At 2k the H2O budget (4096) ≥ sequence, so no eviction occurs — H2O runs as a
dense cache plus scoring overhead (same memory, slower). Eviction only kicks in
once context exceeds the budget.

† In-run peaks for configs that execute **after** a dense OOM are inflated by
residual allocation from the failed config (all configs share one process): the
24,576 and 32,768 H2O rows reported 19.21 / 19.23 GB in the sweep.

‡ Re-measured in a **fresh process**, 32,768 H2O peaks at **14.89 GB** — i.e.
H2O's footprint is essentially flat (~14.9 GB) from 2k to 32k. The fixed 4096-slot
cache means memory does not grow with context.

## Takeaways

- **Memory (headline)**: dense peak grows with context (14.3 → 16.4 → 19.8 GB)
  and **OOMs at 24k and 32k on the 24 GB A10G**. H2O's footprint stays **flat at
  ~14.9 GB from 2k to 32k** (fixed 4096-slot cache), so it **completes at every
  length, including 32k** — i.e. sparse attention runs where full attention does
  not fit at all on the same GPU.
- **Speed**: where both fit (≤16k), H2O is ~1.3–1.5× slower than dense — the cost
  of attention-weight scoring, eviction, and the quantized-cache backward, only
  partly offset by FlashInfer. This is far from a pathological slowdown; the
  gradient pass dominates step time in both.
- **Crossover**: below the dense OOM threshold, full attention is faster and fits,
  so you would use it. At/above that threshold (here ~24k on 24 GB), dense is not
  an option and H2O is the enabler: bounded memory, ~1.5× compute premium, and it
  actually finishes. The value proposition is **longer context (or larger batch)
  on the same GPU**.

## Caveats / future tightening

- **Per-config memory measurement**: all (context, cache) configs run in one
  process, so a config following a dense OOM inherits un-freed allocation (why
  H2O's peak jumps from ~14.9 GB to ~19.2 GB right after the first dense OOM).
  For clean per-config peak memory, run each config in a fresh process.
- Single A10G, batch 2, `max_new_tokens=32` (short completions — the long axis is
  the prompt/context). Larger batch or longer completions would shift absolutes.
- H2O cache fixed at 4096; sweeping the budget would trace the memory/quality
  (and speed) tradeoff.

## Command

```bash
python examples/rl/grpo_context_sweep.py --device cuda \
    --model Qwen/Qwen2.5-3B-Instruct \
    --context-lengths 2048,8192,16384,24576,32768 \
    --h2o-cache-length 4096 --warmup 1 --iters 3 --no-reset
```

## Qwen2.5-7B on L40S 48GB (flagship regime)

Same protocol (group 2, 32 new tokens, chunk 1024, 1 iter), Qwen2.5-7B-Instruct
bf16 on a single g6e.2xlarge (L40S, 48 GB):

| context | cache | gen (ms) | grad (ms) | total (ms) | peak (GB) |
|--------:|-------|---------:|----------:|-----------:|----------:|
| 8,192  | dense                | 4377  | 8666   | 13363  | 34.99 |
| 16,384 | dense                | 7754  | 18111  | 26039  | 40.84 |
| 24,576 | dense                | —     | —      | **OOM** | **OOM** |
| 16,384 | H2O-q8 (full cover)  | 9138  | 18392  | 27962  | 40.88 |
| 32,768 | H2O-q8 (full cover)  | —     | —      | **OOM** | **OOM** |
| 24,576 | H2O-q8 (cl=8192)     | 20101 | 54502  | 75121  | 35.47 |
| 32,768 | H2O-q8 (cl=8192)     | 27474 | 76942  | 104731 | 35.66 |
| 49,152 | H2O-q8 (cl=8192)     | 41762 | 120650 | 162854 | 35.68 |
| 65,536 | H2O-q8 (cl=8192)     | 56181 | 163835 | 220905 | 35.70 |
| 24,576 | H2O-q8 (cl=12288)    | 22720 | 64646  | 87881  | 38.46 |
| 32,768 | H2O-q8 (cl=12288)    | 32300 | 95161  | 127729 | 38.47 |

Takeaways:

- **Dense GRPO on 7B dies at 24k** on a 48 GB GPU. A *full-coverage* sparse
  cache dies soon after (32k) — a big cache costs memory in the gradient pass
  too. The regime that works is a **bounded, actively evicting cache**.
- With `cl=8192` the footprint is **flat ~35.7 GB from 24k to 65k** — GRPO on
  7B at 65k context runs on one L40S, 2.7x past the dense OOM point.
- Trade-off knob: `cl=12288` costs ~3 GB and ~20% step time over `cl=8192`
  but retains 50% more of the prompt (accuracy lever per
  `docs/EVICTION_SWEEP_A2.md`: retention determines accuracy).
- Step times grow with context (the chunked gradient pass is linear in
  prompt length): ~105 s/step at 32k, ~221 s/step at 65k, group 2. A 300-step
  run at 32k, group 8, is a multi-day single-GPU job — hence the worker
  fleet (`terraform/`, `scripts/rl/worker_loop.sh`).
