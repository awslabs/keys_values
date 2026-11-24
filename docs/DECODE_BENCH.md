# Decode throughput vs. context: dense vs. bounded evicting cache

Run `aa0_decode_bench_7b` (2026-10-01). Qwen2.5-7B-Instruct bf16, one L40S
48 GB (g6e.2xlarge), batch 8 (= GRPO group), 256 decoded tokens after a
random prompt of the given length, FlashInfer not compiled (eager SDPA).
Reproduce with `examples/decode_bench.py`; raw numbers in
`s3://keys-values-rl-results/runs/aa0_decode_bench_7b/results.json`.

## Throughput (decode tokens/s, batch 8) and peak memory

| context | dense          | H2O int8 K=8192 | H2O int8 + evict_every=64 | H2O bf16 K=8192 | H2O bf16 + evict_every=64 |
|--------:|---------------:|----------------:|--------------------------:|----------------:|--------------------------:|
|  4,096  | 59.1 / 21.8 GB | 47.6 / 21.0 GB  | 47.7 / 21.0 GB            | 58.3 / 21.8 GB  | 58.3 / 21.8 GB            |
|  8,192  | 32.4 / 29.3 GB | 22.5 / 27.7 GB  | 22.8 / 27.7 GB            | 32.3 / 29.3 GB  | 32.8 / 29.2 GB            |
| 16,384  | **OOM**        | 21.9 / 27.7 GB  | 22.1 / 27.7 GB            | 32.3 / 29.3 GB  | 32.7 / 29.2 GB            |
| 32,768  | **OOM**        | 21.9 / 27.8 GB  | 22.1 / 27.7 GB            | 32.1 / 29.3 GB  | 32.7 / 29.2 GB            |

Prefill (chunked, 1024) is the same for all caches at a given context:
~3 s at 4k, ~6 s at 8k, ~38 s at 16k, ~101 s at 32k.

## Takeaways

1. **Dense decode cost grows with context, then dies.** 59 -> 32 tok/s from
   4k to 8k (every token reads the whole KV cache of every layer), OOM at
   16k with batch 8. The bounded cache is flat in both tokens/s and memory
   from 8k to 32k: once the budget is reached, context length stops
   mattering for decode.
2. **Block eviction (`evict_every`) is a no-op here** (<1%). Recomputing the
   argsort every token was not the per-token cost.
3. **The int8 KV cache costs a third of decode throughput** (22 vs 32 tok/s)
   for a 1.5 GB saving at K=8192. The profile (16k, int8) is dominated by
   `aten::copy_` (27% of CUDA time, 162k calls) and elementwise kernels
   (dequantize / write back each step) plus `masked_fill_`, `_softmax`,
   `mean` from the eager attention path that returns weights for H2O
   scoring. The bf16 H2O cache matches dense tokens/s where dense fits.
4. **Where the remaining overhead is:** the eager attention that returns
   per-slot attention weights (needed for H2O scores). A fused decode kernel
   that emits the score sums, or skipping score updates on some steps, is the
   next lever. Not the eviction ranking.

## Decision

RL drivers default to `h2o-default` (bf16) for the decode-heavy RLVR setting.
int8 stays available (`h2o-torch-quantized8`) for the gradient pass where
2x slots at equal memory buys accuracy (`docs/EVICTION_SWEEP_A2.md`).

## Qwen3-1.7B base probes (same day, 8k generation cutoff, avg@1)

| cache              | MATH500 (n=50) | AIME24 (n=30) | truncated in `<think>` | wall |
|--------------------|---------------:|--------------:|-----------------------:|-----:|
| dense              | 0.76           | 0.167         | 20% / 80%              | 5.8 h |
| H2O int8 K=4096    | 0.78           | 0.167         | 14% / 83%              | 14.6 h |

No eviction gap on either set at K=4096 with 5-8k-token completions. The
8k cutoff is the problem: 80% of AIME samples run out of budget mid-thought.
Default cutoff raised to 16k.

## Caveat found 2026-10-05: the whole table above is eager SDPA

The vendored FlashInfer extension (`python build_ext.py`) had never been
built on the workers, and every job script passed `--disable-flashinfer`.
For the H2O path, which needs per-slot attention weights, the fallback is the
eager blocked SDPA (`masked_fill` / `softmax` / `mean` in the profile). The
dense arm at `q_len=1` was on the q-padded PyTorch SDPA, also not the fast
path. So the table measures eager-vs-eager; the *shape* of the result (dense
grows with context then OOMs, bounded cache flat) stands, the absolute
tokens/s do not.

Fixed: `scripts/rl/provision_worker.sh` now builds the extension with the host
GPU's arch (L40S is sm_89; build_ext.py only listed sm_80/sm_90), both idle
workers have it, and `queue_rlvr_jobs.sh` only passes `--disable-flashinfer`
when the kernels are missing. `decode_bench.py` has `--attn
{eager,flashinfer,flex}` (also per config, `attn=`). Job
`aa0_decode_bench_7b_backends` reran the table with all three backends plus
the int8 torch and BnB quantizers; its results are in the next section.

Side finding while validating the build: the FlashInfer decode kernel returns
NaN attention weights for `kv_len < 128` (split-KV merge over empty chunks);
fine for every real cache size. Filed as awslabs/keys_values#156.

## Attention backends (FlashInfer vs eager vs Flex)

Run `aa0_decode_bench_7b_backends` (2026-10-06), same setup as above:
Qwen2.5-7B-Instruct bf16, one L40S 48 GB, batch 8, 256 decoded tokens,
FlashInfer built. Raw numbers in
`s3://keys-values-rl-results/runs/aa0_decode_bench_7b_backends/results.json`.
Cells are decode tok/s (batch 8) / peak GB; `not run` means the combination
was not in the job's config list. Flex here is the `attn=flex` of that date,
i.e. `FlexAttentionArgs` **without** `kv_lens` padding.

| context | cache | eager | flashinfer | flex (no kv_lens) |
|--------:|-------|------:|-----------:|------------------:|
|  4,096  | dense-default              | 58.8 / 21.8 GB | 219.2 / 21.8 GB | 7.0 / 26.8 GB   |
|  4,096  | h2o-default@8192           | 58.2 / 27.7 GB | 202.8 / 21.8 GB | 0.7 / 37.7 GB   |
|  4,096  | h2o-torch-quantized8@8192  | not run        | 106.9 / 21.0 GB | not run         |
|  4,096  | h2o-bnb-quantized8@8192    | not run        | 128.5 / 21.0 GB | not run         |
|  8,192  | dense-default              | 32.3 / 29.3 GB | 165.7 / 29.3 GB | **OOM**         |
|  8,192  | h2o-default@8192           | 32.3 / 31.2 GB | 136.0 / 29.3 GB | 9.7 / 30.5 GB   |
|  8,192  | h2o-torch-quantized8@8192  | not run        | 44.4 / 27.7 GB  | not run         |
|  8,192  | h2o-bnb-quantized8@8192    | not run        | 62.1 / 27.7 GB  | not run         |
| 16,384  | dense-default              | **OOM**        | **OOM**         | **OOM**         |
| 16,384  | h2o-default@8192           | 32.1 / 31.2 GB | 138.4 / 29.3 GB | 8.7 / 34.5 GB   |
| 16,384  | h2o-torch-quantized8@8192  | not run        | 43.4 / 27.7 GB  | not run         |
| 16,384  | h2o-bnb-quantized8@8192    | not run        | 61.5 / 27.7 GB  | not run         |
| 32,768  | dense-default              | **OOM**        | **OOM**         | **OOM**         |
| 32,768  | h2o-default@8192           | 32.1 / 31.2 GB | 137.3 / 29.3 GB | 8.0 / 33.3 GB   |
| 32,768  | h2o-torch-quantized8@8192  | not run        | 42.9 / 27.8 GB  | not run         |
| 32,768  | h2o-bnb-quantized8@8192    | not run        | 60.7 / 27.7 GB  | not run         |

Prefill (chunked, 1024) with FlashInfer: ~3 s at 4k, ~6 s at 8k, ~15 s at
16k, ~29-32 s at 32k, same for all caches. The H2O prefill under `attn=eager`
is 7.6 / 27 / 59 / 122 s (the weight-returning eager path; the Oct 1 table's
38 s / 101 s at 16k / 32k were on it too). The Flex rows include compile time
in their prefill and decode seconds. The job ended with a `--profile` pass of
the three FlashInfer H2O configs at 16k (short profiled decode: 143.0 / 31.4
/ 50.7 tok/s for bf16 / torch int8 / BnB int8, same peaks); those rows are
not in the table.

Conclusions, from these numbers only. FlashInfer is 3.5-5x eager at
`q_len=1`: 219 vs 59 tok/s (dense, 4k), 203 vs 58 (H2O, 4k), 166 vs 32
(dense, 8k), 136-138 vs 32 (H2O, 8k-32k), at the same peak memory for dense
and 2-6 GB less for H2O. H2O K=8192 under FlashInfer is flat at 136-138 tok/s
and 29.3 GB from 8k to 32k while dense goes 219 -> 166 -> OOM -> OOM, so the
Oct 1 shape stands at 4x the absolute throughput. int8 KV costs decode
throughput: at 8k-32k bf16 gives 136-138 tok/s, BnB int8 61-62, torch int8
43-44 (at 4k: 203 / 128.5 / 107), for 1.6 GB less peak memory at K=8192,
which is why the RL drivers default to `h2o-default` (bf16). Flex *without*
`kv_lens` padding is 7-10 tok/s (dense 4k 7.0; H2O 8k-32k 8.0-9.7; H2O 4k
0.7, where 4k + 256 tokens never fills K=8192 and every step saw a new
`kv_len`) with 1-16 GB more peak memory, because the compiled graph is keyed
on the exact `kv_len`, which changes every decode step until the cache is
full. Upstream #160 later added `kv_lens` padding (graphs keyed on one of
`num_kv_lens` buckets); `decode_bench.py --attn flex` and the RL drivers'
`--attn auto` fallback now use it, but the padded variant has not been
benchmarked.
