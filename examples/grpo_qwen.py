# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Replicate the TRL GRPO quickstart using KeysAndValues' native GRPO loop.

This mirrors the TRL quickstart
(https://huggingface.co/docs/trl/en/grpo_trainer): train
``Qwen/Qwen2.5-0.5B-Instruct`` with a length-based reward over a set of
prompts. The TRL reference is::

    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_len(completions, **kwargs):
        return [-abs(20 - len(c)) for c in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=reward_len,
        args=GRPOConfig(output_dir="Qwen2.5-0.5B-GRPO"),
        train_dataset=dataset,
    )
    trainer.train()

Unlike TRL's trainer, here *every* memory-heavy step — generation, old-log-prob
scoring, and the policy-gradient backward — runs through the KeysAndValues KV
cache via :func:`keys_values.rl.grpo.loop.grpo_step`, so GPU memory stays
bounded as prompts/completions grow.

How does this differ from :class:`keys_values.rl.grpo.trainer.GRPOLongContextTrainer`
(the TRL ``GRPOTrainer`` subclass)? TRL's trainer loads a HuggingFace model and
drives generation itself, so it never holds a ``keys_values.model.GPT``. The
subclass can therefore only override the *log-prob scoring* pass, and only when
a ``GPT`` is reachable inside the model. Rollout generation still runs on TRL's
side, i.e. not through the KeysAndValues KV cache. The standalone loop below is
the path that exercises KeysAndValues end-to-end (generation, scoring, and the
policy-gradient backward), which is what you want for long-context RL.

Setup
-----
::

    pip install -e .            # keys_values + litgpt
    # a HuggingFace token may be required to download Qwen weights:
    export HF_TOKEN=...

Run
---
::

    python examples/grpo_qwen.py --max-steps 20
    python examples/grpo_qwen.py --device cuda --kv-cache-name h2o-torch-quantized8

A small built-in prompt list is used by default so the script is self
contained. Swap in a real dataset (e.g. ``trl-lib/tldr``) by editing
:func:`load_prompts`.
"""

from __future__ import annotations

import argparse
from typing import Callable, List

import lightning as L
import torch
from litgpt.prompts import PromptStyle, has_prompt_style, load_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    auto_download_checkpoint,
    check_valid_checkpoint_dir,
    load_checkpoint,
)

from keys_values.config import Config
from keys_values.data.constants import LIT_MODEL_FNAME
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.model import GPT
from keys_values.rl.grpo.loop import grpo_step

# A small, self-contained prompt set (stands in for `trl-lib/tldr`).
_DEFAULT_PROMPTS: List[str] = [
    "Explain what a transformer neural network is.",
    "Summarize the benefits of unit testing.",
    "Describe how a KV cache speeds up generation.",
    "What is reinforcement learning, in one paragraph?",
    "Give a short tip for writing readable code.",
    "Explain gradient descent to a beginner.",
    "What makes a good commit message?",
    "Describe the purpose of a learning rate.",
]


def load_prompts() -> List[str]:
    """Return the list of training prompts.

    Replace this with a real dataset, e.g.::

        from datasets import load_dataset
        ds = load_dataset("trl-lib/tldr", split="train")
        return [row["prompt"] for row in ds.select(range(256))]
    """
    return _DEFAULT_PROMPTS


def left_pad(sequences: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    """Left-pad 1D token tensors to a common length (required by the rollout)."""
    max_len = max(int(s.size(0)) for s in sequences)
    out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        out[i, max_len - seq.size(0):] = seq
    return out


def make_reward_len(
    tokenizer: Tokenizer,
    target_len: int,
    pad_id: int,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the length reward: ``-|target_len - len(decoded_completion)|``.

    Faithful to TRL's ``reward_len`` — the length is the character count of the
    decoded completion text, with padding stripped.
    """

    def reward_fn(
        prompt_ids: torch.Tensor, completion_ids: torch.Tensor
    ) -> torch.Tensor:
        rewards = []
        for row in completion_ids:
            toks = row[row != pad_id]
            text = tokenizer.decode(toks) if toks.numel() else ""
            rewards.append(-abs(target_len - len(text)))
        return torch.tensor(rewards, dtype=torch.float32)

    return reward_fn


def build_model(
    checkpoint_dir,
    fabric: L.Fabric,
    batch_size: int,
    cache_length: int,
    kv_cache_name: str,
    dtype: torch.dtype,
) -> GPT:
    """Load a keys_values GPT from a litgpt checkpoint and assign KV caches."""
    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    with fabric.init_module(empty_init=True):
        gpt_model = GPT(config)
    load_checkpoint(fabric, gpt_model, checkpoint_dir / LIT_MODEL_FNAME)

    # Non-dense caches are required by the gradient pass.
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            gpt_model=gpt_model,
            name=kv_cache_name,
            max_batch_size=batch_size,
            cache_length=cache_length,
            dtype=dtype,
        )
    )
    return gpt_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--kv-cache-name", default="lastrec-default")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompts-per-step", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--target-len", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--access-token", default=None)
    parser.add_argument(
        "--rescore-old-logps",
        action="store_true",
        help="Recompute old log-probs with a separate scoring forward pass "
        "(multi-epoch / comparison). Default reuses the gradient pass.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print per-phase timings (generation / scoring / gradient) and, "
        "on CUDA, peak memory per step.",
    )
    args = parser.parse_args()

    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    fabric = L.Fabric(
        devices=1,
        accelerator=args.device,
        precision="32-true" if args.device == "cpu" else "bf16-true",
    )

    # 1. Download + convert the HF checkpoint to litgpt format.
    checkpoint_dir = auto_download_checkpoint(
        model_name=args.model, access_token=args.access_token
    )
    tokenizer = Tokenizer(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    prompt_style = (
        load_prompt_style(checkpoint_dir)
        if has_prompt_style(checkpoint_dir)
        else PromptStyle.from_config(config)
    )

    pad_id = tokenizer.processor.token_to_id("<|endoftext|>")
    if pad_id is None:
        pad_id = int(tokenizer.eos_id) if tokenizer.eos_id is not None else 0

    # 2. Tokenize prompts (chat template applied), left-padded per step batch.
    prompts = load_prompts()
    encoded_prompts = [
        tokenizer.encode(prompt_style.apply(p), device=fabric.device) for p in prompts
    ]

    batch_size = args.prompts_per_step * args.group_size
    max_prompt_len = max(int(e.size(0)) for e in encoded_prompts)
    cache_length = max_prompt_len + args.max_new_tokens

    # 3. Build model with KV caches assigned.
    gpt_model = build_model(
        checkpoint_dir=checkpoint_dir,
        fabric=fabric,
        batch_size=batch_size,
        cache_length=cache_length,
        kv_cache_name=args.kv_cache_name,
        dtype=dtype,
    )
    gpt_model.to(fabric.device)

    reward_fn = make_reward_len(tokenizer, args.target_len, pad_id)
    optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=args.lr)

    # 4. GRPO training loop (each step samples `prompts_per_step` prompts).
    n = len(encoded_prompts)
    for step in range(args.max_steps):
        idx = torch.randperm(n)[: args.prompts_per_step]
        batch = [encoded_prompts[int(i)] for i in idx]
        prompt_ids = left_pad(batch, pad_id).to(fabric.device)

        if args.profile and args.device == "cuda":
            torch.cuda.reset_peak_memory_stats(fabric.device)

        metrics = grpo_step(
            gpt_model=gpt_model,
            prompt_ids=prompt_ids,
            reward_fn=reward_fn,
            optimizer=optimizer,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            chunk_size=args.chunk_size,
            temperature=1.0,
            eos_token_id=int(tokenizer.eos_id)
            if tokenizer.eos_id is not None
            else None,
            pad_token_id=pad_id,
            rescore_old_logps=args.rescore_old_logps,
            profile=args.profile,
        )
        line = (
            f"step {step:3d} | loss {metrics['loss']:+.4f} | "
            f"reward {metrics['mean_reward']:+.3f} | "
            f"adv_std {metrics['advantage_std']:.3f} | "
            f"compl_len {metrics['completion_len']}"
        )
        if args.profile:
            line += (
                f" | gen {metrics['gen_time_ms']:.0f}ms"
                f" score {metrics['score_time_ms']:.0f}ms"
                f" grad {metrics['grad_time_ms']:.0f}ms"
            )
            if args.rescore_old_logps:
                line += f" skew {metrics['logp_skew_decode_vs_forward']:.3f}"
            if args.device == "cuda":
                peak_gb = torch.cuda.max_memory_allocated(fabric.device) / 1e9
                line += f" | peak {peak_gb:.2f}GB"
        print(line)

    print("Done.")


if __name__ == "__main__":
    main()
