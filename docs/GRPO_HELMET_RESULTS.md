# RL through an evicting sparse KV cache: HELMET results

Consolidates the experimental findings from the GRPO-on-HELMET campaign
(PR #142): training GRPO, RLOO, and SFT end-to-end through an actively
evicting H2O cache, cross-evaluated against dense-attention arms.

Reproduce with `examples/grpo_helmet.py` (training arms) and
`examples/grpo_helmet_crosseval.py` (evaluation matrix).

## Setup

- **Hardware**: single NVIDIA A10G (24 GB), bf16, eager SDPA.
- **Model**: Qwen2.5-0.5B-Instruct.
- **Tasks**: HELMET `nq` and `hotpot_qa`, 8k bucket (real RAG-QA prompts,
  ~7k tokens average).
- **Arms**: identical except the KV cache during training —
  **H2O @ 4096 slots** (`h2o-torch-quantized8`, actively evicting during
  rollouts and the gradient pass; `grace_period = cache_length / 16` set at
  application level per the #140 discussion) vs **dense** (full attention).
- **Evaluation**: cross-evaluation matrix — every checkpoint evaluated under
  *both* inference caches, n=100 held-out records (SE ≈ 5pp on EM).
- **Metrics**: `sub_exact_match` (EM, substring match) and token-F1.

## Round 1: stability and parity (lr 1e-6, ≤300 steps)

Reward = `sub_exact_match`, then `max(EM, token-F1)` partial credit (v2).

**nq @ 8k** (EM / F1):

| checkpoint ↓ · eval cache → | dense | H2O@4096 |
|---|---:|---:|
| base (untrained) | 0.330 / 0.178 | 0.320 / 0.142 |
| GRPO-trained under H2O | 0.350 / 0.181 | 0.340 / 0.144 |
| GRPO-trained under dense | 0.340 / 0.180 | 0.300 / 0.132 |

v2 arms (F1 partial-credit reward, group 8, 300 steps) reproduce the same
pattern: all cells within noise of base.

**Findings**

1. **Sparse inference is essentially free on nq.** Base under H2O@4096 vs
   dense: 0.320 vs 0.330 EM (~1pp, within noise at n=100). A cache at ~55%
   of the prompt length costs no measurable quality. (Before the #140 fix,
   every evicting configuration scored 0.000.)
2. **Training under the sparse cache does not damage the policy.** All
   checkpoints are within noise under matched eval conditions.
3. **Cross-eval matters.** Per-arm evals (each arm under its own cache,
   small n) had suggested a large H2O-vs-dense gap (0.292 vs 0.417); under
   matched inference at n=100 the gap disappears — it was inference-cache
   conflation plus small-sample noise.
4. At this learning rate neither arm improves over base: round 1 validates
   stability and parity, not learning gains.
5. The F1 partial-credit reward raised the fraction of steps with nonzero
   group-relative advantage ("signal rate") to ~0.7–0.8, from
   mostly-degenerate groups under binary EM.

Footprint: ~13–24 s/step at ~7k-token prompts, ~8 GB peak — roughly half
the dense footprint, consistent with `docs/GRPO_CONTEXT_SCALING.md`.

## Round 2: learning gains (lr 5e-6, group 8 × 2 accumulated prompts × 400 updates)

Reward = `max(EM, token-F1)`; two seeds; n=100 cross-eval.

**nq @ 8k, seed 0** (EM / F1):

| checkpoint ↓ · eval → | dense | H2O@4096 |
|---|---:|---:|
| base | 0.330 / 0.178 | 0.320 / 0.142 |
| GRPO-trained under H2O | **0.450** / 0.166 | **0.420** / 0.167 |
| GRPO-trained under dense | 0.270 / **0.397** | 0.220 / 0.396 |

**hotpot_qa @ 8k, seed 0**:

| checkpoint ↓ · eval → | dense | H2O@4096 |
|---|---:|---:|
| base | 0.290 / 0.145 | 0.180 / 0.109 |
| GRPO-trained under H2O | 0.300 / 0.316 | 0.230 / 0.271 |
| GRPO-trained under dense | **0.410** / **0.436** | 0.320 / 0.390 |

**hotpot_qa @ 8k, seed 1** (seed-0 in parentheses):

| checkpoint ↓ · eval → | dense | H2O@4096 |
|---|---:|---:|
| base | 0.290 / 0.145 | 0.180 / 0.109 |
| H2O-trained s1 | 0.280 / 0.122 (0.300 / 0.316) | 0.290 / 0.120 (0.230 / 0.271) |
| dense-trained s1 | **0.380** / 0.381 (0.410 / 0.436) | 0.330 / 0.335 (0.320 / 0.390) |

**Two-seed summary** (EM under dense eval, vs base):

| arm | nq | hotpot_qa |
|---|---:|---:|
| H2O-trained | **+12pp / +11pp** (0.45, 0.44) | ~flat (0.30, 0.28) |
| dense-trained | −6pp / −2pp (0.27, 0.31) | **+12pp / +9pp** (0.41, 0.38) |

### The dense/nq EM collapse is an answer-style artifact, not reward hacking

Sample-generation inspection (dense eval, distinct NQ questions):

- **base**: verbose scaffolded answers — "The last time the Philadelphia
  Eagles played the New England Patriots was **in Super Bowl LII**, which
  took place..."
- **h2o-trained**: same verbose style, largely intact.
- **dense-trained**: *terse* answers — "Super Bowl LII in 2017.",
  "Mr Carson.", "March 10, 2017."

The F1 partial credit taught the dense arm to strip scaffolding (terse
answers score higher token-F1). HELMET NQ targets often include leading
prepositions ("**in** Super Bowl LII"), and `sub_exact_match` requires the
target as a substring: the verbose style contains it, the terse style does
not, even when semantically correct. Train reward rising (0.70) while eval
EM collapsed (0.44 → 0.12) and F1 doubled is substantially this
metric-style interaction. Both arms shifted style under the F1 reward; the
dense arm shifted further per update.

**Findings**

1. **Training through the evicting H2O cache produces real gains**: +11–12pp
   EM on nq across two seeds — but the "H2O-trained beats dense-trained"
   EM delta is confounded by the style artifact above and is *not* a safe
   claim.
2. **Robust claim**: GRPO through an actively evicting cache trains stably
   and moves the policy as much as dense training does, at ~half the memory.
3. **Task dependence**: on multi-hop hotpot, H2O *inference* costs ~11pp EM
   (dispersed evidence gets evicted) and the H2O training arm stays ~flat
   under dense eval (though it gains under matched H2O inference:
   0.18 → 0.23/0.29 across seeds). Dispersed-evidence tasks are where
   eviction hurts, both at inference and in training rollouts.
4. **Fix adopted for round 3**: reward = EM + 0.2·F1 (`em_f1`) keeps the
   reward anchored to the eval metric so answer style cannot drift.

## Round 3: algorithms × caches × tasks (consolidated)

Same recipe (400 updates × 2 accumulated prompts, lr 5e-6, n=100
cross-eval), `em_f1` reward for RL, single seed. EM / F1 shown as
dense-eval | H2O-eval.

**nq @ 8k** (base: 0.33/0.18 | 0.32/0.14):

| method (training cache) | dense-eval | H2O-eval |
|---|---:|---:|
| SFT (H2O) | 0.16 / 0.32 | 0.19 / 0.30 |
| SFT (dense) | 0.16 / 0.35 | 0.23 / 0.36 |
| GRPO em_f1 (H2O) | 0.28 / 0.33 | 0.26 / 0.32 |
| GRPO em_f1 (dense) | 0.29 / 0.39 | 0.29 / 0.42 |
| **RLOO em_f1 (H2O)** | **0.43 / 0.15** | **0.42 / 0.12** |

**hotpot_qa @ 8k** (base: 0.29/0.15 | 0.18/0.11):

| method (training cache) | dense-eval | H2O-eval |
|---|---:|---:|
| SFT (H2O) | 0.31 / 0.39 | 0.28 / 0.39 |
| SFT (dense) | 0.37 / 0.43 | 0.31 / 0.40 |
| GRPO em_f1 (H2O) | 0.38 / 0.44 | 0.33 / 0.40 |
| GRPO em_f1 (dense) | **0.42 / 0.46** | 0.39 / 0.43 |

**Findings**

1. **Training through the actively evicting cache works across all three
   algorithms** (GRPO, RLOO, SFT) — stable end-to-end, ~half the dense
   memory footprint.
2. **RL > SFT** on both tasks (hotpot EM 0.42 vs 0.37; nq 0.29–0.43 vs 0.16).
3. **Sparse-trained ≈ dense-trained**: GRPO H2O arm within 1–4pp EM of
   dense (nq 0.28 vs 0.29; hotpot 0.38 vs 0.42), and the **RLOO H2O arm is
   the best nq result overall (0.43 EM)** — a sparse-trained model beating
   every dense-trained one.
4. **RLOO's nq gain is the most credible EM improvement of the campaign**:
   its F1 stays at base level (0.15 vs 0.18), i.e. answer *style* did not
   drift — it gets more answers right within the base model's own style.
   RLOO also showed the highest signal rate (0.93–0.94): leave-one-out
   advantages don't divide by group std, so small reward spreads still
   carry gradient.
5. **Training shrinks the sparse-inference penalty**: base loses 11pp EM
   under H2O eval on hotpot; trained models lose 3–5pp.

## Caveats

- nq's substring-EM remains style-sensitive (F1-heavy arms drop nq EM while
  F1 rises); report both metrics.
- Round-3/SFT/RLOO are single-seed (the round-2 pattern replicated across
  2 seeds); n=100 (SE ≈ 5pp), so ±12pp deltas are ~2σ.
- 0.5B model, two tasks, 8k bucket; larger models and longer contexts are
  the natural next axis (see `docs/GRPO_CONTEXT_SCALING.md` for the memory
  scaling that motivates them).
- RLOO mode and the SFT driver live on stacked branches (`rl-rloo`,
  `sft-helmet`); follow-up PRs once the base PR lands.

## Commands

```bash
# Training arm (H2O @ 4096 slots, em_f1 reward, gradient accumulation)
python examples/grpo_helmet.py --device cuda \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset-key nq --max-length 8k \
    --kv-cache-name h2o-torch-quantized8 --cache-length 4096 \
    --reward em_f1 --group-size 8 \
    --prompts-per-update 2 --steps 400 --lr 5e-6

# Cross-evaluation matrix (train-cache x eval-cache, n=100)
python examples/grpo_helmet_crosseval.py --device cuda \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset-key nq --max-length 8k --n-eval 100 \
    --h2o-cache-length 4096 \
    --checkpoints base <h2o_ckpt_dir> <dense_ckpt_dir>
```
