# RL-through-sparse-attention: status report

*2026-08-10. Covers everything since the RL effort started; written for the
paper-planning discussion (option 1: tangible gains on a hard RL problem
under resource constraints).*

## What works today

**Training loop.** Standalone GRPO/RLOO (`keys_values/rl/grpo/`) on
KeysAndValues primitives: chunked prefill + bounded-memory chunked backward
through an actively evicting KV cache. Single-epoch cache reuse (the
sampling pass's log-probs are reused; no separate scoring forward), gradient
accumulation, completion masking, RLOO leave-one-out advantages, SFT
baseline driver. PR #142 (review comments addressed, awaiting merge);
RLOO/SFT on stacked branches. Shared-prompt prefill (LongStraw-style
response-only schedule) implemented and token-exact vs the naive schedule:
1.5x generation speedup at 7k prompts (`prompt-reuse` branch).

**Scale.** Two upstream bugs found and fixed along the way (#139 ln_f
before head; #140 eviction defaults destroying generation quality --
measured 0.00 -> 0.90 recovery). 8-bit paged optimizer support added for
7B+ full fine-tuning on 48GB cards.

## Results so far (all HELMET, question-diverse eval)

**0.5B campaign @8k** (`docs/GRPO_HELMET_RESULTS.md`): training through the
evicting cache is stable across GRPO/RLOO/SFT and matches dense-attention
training at ~half the memory (parity within noise on every paired
comparison, 2 seeds on the round-2 recipe). Best single result: RLOO+EM
through evicting H2O, nq 0.43 EM vs 0.33 base, with no answer-style drift.
RL > SFT throughout. Known artifact documented: substring-EM is
style-sensitive; the EM-anchored `em_f1` reward avoids it.

**Accuracy-gap analysis (A1/A2)** (`docs/POSITION_COMPACTION_A1.md`,
`docs/EVICTION_SWEEP_A2.md`): the sparse-INFERENCE gap is information loss
from eviction, full stop. Position compaction (exact RoPE re-indexing):
no recovery. Every retention knob at fixed slots: no recovery. What works:
8-bit KV quantization is accuracy-free (verified with dense-q8 == h2o-q8
controls), so at equal memory you afford 2x slots -- when the cache covers
the prompt, accuracy equals dense at ~half the memory. Under genuine
eviction the gap is real and degrades gracefully (~25% retention costs
~9pp EM on trivia).

**Memory frontier, 7B on 48GB L40S** (`docs/rl/GRPO_CONTEXT_SCALING.md`):
dense GRPO OOMs at 24k context. Full-coverage sparse OOMs at 32k. Bounded
evicting cache (q8@8192): flat ~35.7GB from 24k to **65k** -- the only
configuration that trains at all in this regime.

## Benchmark hygiene

The HELMET dev/eval split was drawn unseeded (our port dropped the
harness-level seeding HELMET itself uses) -- found by cross-host
replication, fixed in PR #145 (merged). Cross-host splits shared only
8-10/100 questions; all campaign numbers came from one box/one cache, so
paired comparisons stand. Canonical split caches (8k campaign draw + new
seeded 32k draws) now live on S3 + a GitHub release, with SHA-256
manifests over the record IDs. Upstream HELMET has a related (smaller)
demo-sampling nondeterminism; issue filed (princeton-nlp/HELMET#43) with
a fix ready.

## In flight: 7B @32k flagship (first "dense can't do this" gains attempt)

Base-model probes (n=50): nq 0.80 dense-eval / 0.52 through-cache;
hotpot 0.40 / 0.24 -- a 16-28pp eviction gap for training to close, on a
model/context where dense TRAINING cannot run. Four RLOO runs (nq, hotpot
x 2 seeds; em_f1 reward; q8@8192 cache; 200 steps) are running on a
3-worker L40S fleet, ~350s/step, ETA ~2 days. Infra: S3 job queue +
self-provisioning workers + optional Terraform module (`terraform/`,
autoscaling across all AZs/types -- single-GPU capacity is scarce and
manual placement was costing hours).

## Open question: what counts as a genuinely hard task

nq at 32k is real long-context RL but the base model is already strong
(0.80 dense). Candidates for a harder flagship, to be gated by base-model
probes (capability > 0 but low; verifiable reward; not style-gameable):

1. **LongProc** (long procedural generation, HELMET add-on): models score
   low, outputs are 2-8k tokens and rule-checkable, and long *decode*
   stresses eviction differently than long prefill. Strongest candidate.
2. **infinite_bench_qa / narrative_qa at 64k+**: harder contexts, existing
   loaders; 64k+ is deep in the dense-impossible regime.
3. **RULER multi-key needle variants**: difficulty is a dial (keys,
   depth), fully verifiable, and directly measures what eviction destroys.
4. **ALCE citation generation**: verifiable citation grounding; hard, but
   reward engineering is heavier.

json_kv remains capability-null for the base models (0.00) and stays
excluded (cold-start gate).
