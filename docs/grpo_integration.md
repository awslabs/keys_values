# GRPO Fine-tuning with KeysAndValues

This document explains how to run GRPO (Group Relative Policy Optimization)
fine-tuning on top of KeysAndValues, what you need to install, and exactly
where the KeysAndValues KV cache is used in the pipeline.

The code lives under [`keys_values/rl/`](../keys_values/rl):

```
keys_values/rl/
  logprobs.py            # memory-efficient per-token log-prob computation
  grpo/
    trainer.py           # GRPOLongContextTrainer (TRL GRPOTrainer subclass)
    loop.py              # grpo_step: standalone GRPO loop (no TRL needed)
    loss.py              # GRPOLossHeadModel: GRPO loss as a HeadModel
    rollout.py           # generate_completions: KV-cache generation adapter
```

## Why a KV cache for RL?

GRPO repeatedly (1) generates completions, (2) scores them under the sampling
policy, and (3) computes a policy gradient. For long prompts/completions, each
of these steps would normally hold the activations or attention state for the
entire sequence in GPU memory.

KeysAndValues processes sequences in **chunks** through a bounded KV cache, so
peak memory stays flat as sequence length grows. The GRPO integration routes
the memory-heavy steps through that infrastructure.

## Where the KV cache is used

This is the key question, and the answer depends on which entry point you use.

### Standalone loop (`keys_values.rl.grpo.loop.grpo_step`)

The KV cache is used at **every** memory-heavy stage — both inference and the
gradient update:

| Stage | Component | KV cache role |
|-------|-----------|---------------|
| 1. Generation | `rollout.generate_completions` → `LongContextInferenceModel` | inference (chunked prefill + decode) |
| 2. Old (sampling) log-probs | `logprobs.compute_logprobs` (under `no_grad`) | inference (scoring) |
| 3. Policy gradient | `loss.GRPOLossHeadModel` + `LongContextGradientModel` | gradient updates (memory-bounded backward) |
| 4. Optimizer step | `torch.optim.Optimizer` | — |

So for the standalone loop the answer is **both**: the cache backs generation
and old-log-prob scoring (inference) *and* the gradient/backward pass.

### TRL trainer (`keys_values.rl.grpo.trainer.GRPOLongContextTrainer`)

This subclass plugs into TRL's `GRPOTrainer` and overrides only the per-token
log-probability computation (`_get_per_token_logps_and_entropies`), routing it
through `compute_logprobs` when a sequence is longer than `kv_cache_length`.
TRL still owns generation and the optimizer.

| Stage | Owner | KV cache role |
|-------|-------|---------------|
| Generation | TRL (transformers / vLLM) | not used |
| Per-token log-probs (reference and policy) | `compute_logprobs` | inference scoring for long sequences; short sequences fall through to TRL's default with zero overhead |
| Optimizer step | TRL | — |

So for the TRL path the cache is used specifically for the **log-prob
computation** TRL depends on, on long sequences only.

## Installation

Start from a working KeysAndValues install (see the top-level
[README](../README.md) for the base setup), then:

```bash
# Base package (editable install from the repo root)
pip install -e .

# The standalone loop (keys_values.rl.grpo.loop) and logprobs need nothing
# beyond the base install — they run anywhere KeysAndValues runs, incl. CPU.

# The TRL trainer additionally needs TRL:
pip install -e .[trl]      # installs trl>=1.0.0
```

GPU is recommended for real runs but not required for the standalone loop on a
small model (the unit tests run on CPU).

## Running

### Option A — standalone loop (no TRL)

`grpo_step` runs one full GRPO optimization step on a `keys_values.model.GPT`
that has non-dense KV caches assigned. You supply prompts and a reward
function.

```python
import torch
from keys_values.model import GPT
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.rl.grpo.loop import grpo_step

# model must have (non-dense) KV caches assigned
gpt_model.assign_kv_caches(
    KVCacheFactory.create(
        gpt_model=gpt_model,
        name="lastrec-default",
        max_batch_size=num_prompts * group_size,
        cache_length=cache_length,
        dtype=torch.float32,
    )
)
optimizer = torch.optim.SGD(gpt_model.parameters(), lr=1e-2)

def reward_fn(prompt_ids, completion_ids):
    # return a reward tensor of shape (num_prompts * group_size,)
    ...

metrics = grpo_step(
    gpt_model=gpt_model,
    prompt_ids=prompt_ids,            # (num_prompts, prompt_len), left-padded
    reward_fn=reward_fn,
    optimizer=optimizer,
    group_size=group_size,            # completions sampled per prompt
    max_new_tokens=64,
    chunk_size=16,
)
print(metrics)   # loss, mean_reward, mean_advantage, ...
```

A runnable, annotated walkthrough is in
[`examples/trl_grpo_demo.ipynb`](../examples/trl_grpo_demo.ipynb).

### Option B — TRL trainer

A drop-in `GRPOTrainer` for users already on TRL who have a KeysAndValues
model. Long sequences get the bounded-memory log-prob path; short ones use
TRL's default.

```python
from keys_values.rl.grpo.trainer import GRPOLongContextTrainer

trainer = GRPOLongContextTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=my_reward_func,
    train_dataset=dataset,
    kv_cache_name="h2o-torch-quantized8",
    kv_cache_length=16384,   # sequences longer than this use the chunked path
    kv_chunk_size=1024,
)
trainer.train()
```

## Tests

```bash
pytest test/rl/
```

The suite runs on CPU with a tiny model and exercises the full pipeline:
generation → reward → advantages → old log-probs → policy gradient →
optimizer step.
