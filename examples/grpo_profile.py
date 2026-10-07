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
Profile the KeysAndValues GRPO step on a real model, sweeping completion length.

For each sequence length we run several real GRPO steps (generation + policy
gradient on a ``keys_values`` ``GPT``) in two modes and compare:

* ``optimized`` -- old log-probs reuse the gradient pass (``policy_logp.detach()``),
  so no separate scoring forward pass runs (single-epoch GRPO).
* ``rescore``   -- old log-probs recomputed with a separate ``compute_logprobs``
  pass (multi-epoch / baseline). Also reports the rollout-vs-forward log-prob
  skew.

The eliminated scoring pass is a full forward over prompt+completion, so its
cost grows with sequence length -- which is exactly what this sweep shows.

Real model + real prompts (this is not a synthetic benchmark):

    # quick run on an A10
    python examples/grpo_profile.py --device cuda --max-new-tokens 64,128,256

    # longer sweep
    python examples/grpo_profile.py --device cuda \
        --max-new-tokens 128,256,512,1024 --group-size 8 --iters 5

Prompts come from a real dataset (``trl-lib/tldr`` by default) when the
``datasets`` package is available; otherwise a small built-in prompt list is
used with a warning.
"""

from __future__ import annotations

import argparse
import statistics
from typing import Callable, List

import lightning as L
import torch
from litgpt.prompts import PromptStyle, has_prompt_style, load_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import auto_download_checkpoint, check_valid_checkpoint_dir, load_checkpoint

from keys_values.config import Config
from keys_values.data.constants import LIT_MODEL_FNAME
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.model import GPT
from keys_values.rl.grpo.loop import grpo_step

_FALLBACK_PROMPTS: List[str] = [
    "Explain what a transformer neural network is, in detail.",
    "Summarize the tradeoffs between throughput and latency in LLM inference.",
    "Describe how a KV cache speeds up autoregressive generation.",
    "Walk through how reinforcement learning from human feedback works.",
    "Explain gradient descent and why the learning rate matters.",
    "What is sparse attention and when does it help?",
    "Describe the difference between prefill and decode in LLM serving.",
    "Explain group-relative policy optimization at a high level.",
]


def load_prompts(dataset: str, n: int, split: str) -> List[str]:
    """Load real prompts from a HF dataset, falling back to a built-in list."""
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset, split=split)
        field = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
        prompts = [row[field] for row in ds.select(range(min(n, len(ds))))]
        if prompts:
            return prompts
    except Exception as exc:  # noqa: BLE001 - profiling helper, degrade gracefully
        print(f"[warn] could not load dataset '{dataset}' ({exc}); using fallback prompts")
    return _FALLBACK_PROMPTS


def left_pad(sequences: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(int(s.size(0)) for s in sequences)
    out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        out[i, max_len - seq.size(0):] = seq
    return out


def make_reward_len(
    tokenizer: Tokenizer, target_len: int, pad_id: int
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Toy length reward -- reward quality is irrelevant for profiling compute."""

    def reward_fn(prompt_ids: torch.Tensor, completion_ids: torch.Tensor) -> torch.Tensor:
        rewards = []
        for row in completion_ids:
            toks = row[row != pad_id]
            text = tokenizer.decode(toks) if toks.numel() else ""
            rewards.append(-abs(target_len - len(text)))
        return torch.tensor(rewards, dtype=torch.float32)

    return reward_fn


def build_model(checkpoint_dir, fabric, batch_size, cache_length, kv_cache_name, dtype) -> GPT:
    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    with fabric.init_module(empty_init=True):
        gpt_model = GPT(config)
    load_checkpoint(fabric, gpt_model, checkpoint_dir / LIT_MODEL_FNAME)
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            gpt_model=gpt_model, name=kv_cache_name, max_batch_size=batch_size,
            cache_length=cache_length, dtype=dtype,
        )
    )
    return gpt_model


def _agg(metrics_list, key):
    vals = [m[key] for m in metrics_list if key in m]
    return statistics.mean(vals) if vals else 0.0


def run_config(
    gpt_model, snapshot, prompt_batches, reward_fn, args, fabric, dtype,
    max_prompt_len, max_new_tokens, cache_length, kv_cache_name, rescore, pad_id, eos_id,
):
    """Reset weights, (re)assign caches, run warmup + measured steps for one mode."""
    from keys_values.kvcache.factory import deallocate_kv_cache_buffers_of_model

    gpt_model.load_state_dict(snapshot, strict=True)
    deallocate_kv_cache_buffers_of_model(gpt_model)
    batch_size = args.prompts_per_step * args.group_size
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            gpt_model=gpt_model, name=kv_cache_name, max_batch_size=batch_size,
            cache_length=cache_length, dtype=dtype,
        )
    )
    optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=args.lr)

    is_cuda = fabric.device.type == "cuda"
    step_fn = lambda pb: grpo_step(  # noqa: E731
        gpt_model=gpt_model, prompt_ids=pb, reward_fn=reward_fn, optimizer=optimizer,
        group_size=args.group_size, max_new_tokens=max_new_tokens,
        chunk_size=args.chunk_size, layers_per_cell=args.layers_per_cell,
        temperature=1.0, eos_token_id=eos_id, pad_token_id=pad_id,
        rescore_old_logps=rescore, profile=True,
    )

    for i in range(args.warmup):
        step_fn(prompt_batches[i % len(prompt_batches)])

    if is_cuda:
        torch.cuda.synchronize(fabric.device)
        torch.cuda.reset_peak_memory_stats(fabric.device)
    import time as _t
    metrics, wall = [], 0.0
    for i in range(args.iters):
        pb = prompt_batches[(args.warmup + i) % len(prompt_batches)]
        if is_cuda:
            torch.cuda.synchronize(fabric.device)
        t0 = _t.perf_counter()
        m = step_fn(pb)
        if is_cuda:
            torch.cuda.synchronize(fabric.device)
        wall += (_t.perf_counter() - t0) * 1000.0
        metrics.append(m)

    peak_gb = torch.cuda.max_memory_allocated(fabric.device) / 1e9 if is_cuda else 0.0
    total_ms = wall / args.iters
    compl = _agg(metrics, "mean_completion_tokens") * args.prompts_per_step * args.group_size
    tok_s = compl / (total_ms / 1000.0) if total_ms > 0 else 0.0
    return {
        "gen": _agg(metrics, "gen_time_ms"),
        "score": _agg(metrics, "score_time_ms"),
        "grad": _agg(metrics, "grad_time_ms"),
        "total": total_ms,
        "peak_gb": peak_gb,
        "tok_s": tok_s,
        "skew": _agg(metrics, "logp_skew_decode_vs_forward") if rescore else 0.0,
        "loss": _agg(metrics, "loss"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--kv-cache-name", default="h2o-torch-quantized8")
    parser.add_argument("--dataset", default="trl-lib/tldr")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--prompts-per-step", type=int, default=2)
    parser.add_argument(
        "--max-new-tokens", default="64,128,256",
        help="Comma-separated completion lengths to sweep.",
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--layers-per-cell", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--access-token", default=None)
    args = parser.parse_args()

    sweep = [int(x) for x in args.max_new_tokens.split(",")]
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    fabric = L.Fabric(
        devices=1, accelerator=args.device,
        precision="32-true" if args.device == "cpu" else "bf16-true",
    )

    checkpoint_dir = auto_download_checkpoint(model_name=args.model, access_token=args.access_token)
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
    eos_id = int(tokenizer.eos_id) if tokenizer.eos_id is not None else None

    raw_prompts = load_prompts(args.dataset, args.num_prompts, args.dataset_split)
    encoded = [tokenizer.encode(prompt_style.apply(p), device=fabric.device) for p in raw_prompts]
    max_prompt_len = max(int(e.size(0)) for e in encoded)

    # Pre-build fixed prompt batches (same across modes for a fair comparison).
    torch.manual_seed(0)
    n_batches = args.warmup + args.iters
    prompt_batches = []
    for _ in range(n_batches):
        idx = torch.randperm(len(encoded))[: args.prompts_per_step]
        batch = [encoded[int(i)] for i in idx]
        prompt_batches.append(left_pad(batch, pad_id).to(fabric.device))

    batch_size = args.prompts_per_step * args.group_size
    max_cache_len = max_prompt_len + max(sweep)
    gpt_model = build_model(checkpoint_dir, fabric, batch_size, max_cache_len, args.kv_cache_name, dtype)
    gpt_model.to(fabric.device)
    snapshot = {k: v.detach().clone() for k, v in gpt_model.state_dict().items()}
    reward_fn = make_reward_len(tokenizer, target_len=20 * 4, pad_id=pad_id)

    print(f"\nmodel={args.model}  cache={args.kv_cache_name}  device={args.device}  "
          f"prompts={len(encoded)} (max_len={max_prompt_len})  "
          f"batch={batch_size} (={args.prompts_per_step}x{args.group_size})  "
          f"chunk={args.chunk_size}\n")
    hdr = (f"{'compl':>6} {'mode':>9} | {'gen':>7} {'score':>7} {'grad':>7} {'total':>8} (ms)"
           f" | {'tok/s':>7} | {'peakGB':>7} | {'skew':>6}")
    print(hdr)
    print("-" * len(hdr))

    for mnt in sweep:
        cache_len = max_prompt_len + mnt
        rows = {}
        for mode, rescore in (("optimized", False), ("rescore", True)):
            rows[mode] = run_config(
                gpt_model, snapshot, prompt_batches, reward_fn, args, fabric, dtype,
                max_prompt_len, mnt, cache_len, args.kv_cache_name, rescore, pad_id, eos_id,
            )
        for mode in ("optimized", "rescore"):
            r = rows[mode]
            skew = f"{r['skew']:.3f}" if mode == "rescore" else "  -  "
            print(f"{mnt:>6} {mode:>9} | {r['gen']:>7.0f} {r['score']:>7.0f} "
                  f"{r['grad']:>7.0f} {r['total']:>8.0f} | {r['tok_s']:>7.0f} | "
                  f"{r['peak_gb']:>7.2f} | {skew:>6}")
        opt, res = rows["optimized"], rows["rescore"]
        if res["total"] > 0:
            speedup = 100.0 * (res["total"] - opt["total"]) / res["total"]
            print(f"{'':>6} {'Δ':>9} | scoring pass removed -> {speedup:+.1f}% faster/step "
                  f"(saved ~{res['score']:.0f}ms)")
    print("\nDone.")


if __name__ == "__main__":
    main()
