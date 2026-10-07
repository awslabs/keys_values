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
Dense vs. sparse (H2O) GRPO across growing context length.

This is the test that shows what KeysAndValues actually buys you. It runs GRPO
steps with a *long prompt* and compares two KV caches at each context length:

* ``dense-default``       -- full attention; the cache holds every token, so
  memory grows with context (this is the "SotA full attention" baseline).
* ``h2o-torch-quantized8`` -- sparse; the cache is capped at a fixed budget
  (``--h2o-cache-length``), so once the prompt exceeds it, tokens are evicted
  and memory stays bounded. This is the regime H2O is designed for.

For each (context length, cache) we report per-phase time and peak GPU memory.
The headline is peak memory: dense grows ~linearly with context while H2O stays
flat, so H2O keeps running (or fits a bigger batch / smaller GPU) where dense
OOMs. OOMs are caught and reported rather than crashing the sweep.

Example (A10G, 23 GB):

    python examples/grpo_context_sweep.py --device cuda \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --context-lengths 2048,8192,16384,32768 --h2o-cache-length 4096
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, List

import lightning as L
import torch
from litgpt.prompts import PromptStyle, has_prompt_style, load_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import auto_download_checkpoint, check_valid_checkpoint_dir, load_checkpoint

from keys_values.config import Config
from keys_values.data.constants import LIT_MODEL_FNAME
from keys_values.kvcache.factory import KVCacheFactory, deallocate_kv_cache_buffers_of_model
from keys_values.model import GPT
from keys_values.rl.grpo.loop import grpo_step

_BASE_TEXT = (
    "In modern machine learning systems, attention mechanisms let a model weigh "
    "the relevance of every token against every other token. As context length "
    "grows, the key-value cache and the attention computation dominate both "
    "memory and latency, which is why sparse attention and cache eviction "
    "policies such as heavy-hitter oracle (H2O) matter for long-context work. "
)


def build_long_prompt(tokenizer: Tokenizer, target_len: int, device) -> torch.Tensor:
    """Real-token prompt tiled/truncated to exactly ``target_len`` tokens."""
    ids = tokenizer.encode(_BASE_TEXT, device=device)
    while ids.size(0) < target_len:
        ids = torch.cat([ids, ids], dim=0)
    return ids[:target_len]


def make_reward_len(tokenizer, target_len, pad_id) -> Callable:
    def reward_fn(prompt_ids, completion_ids):
        out = []
        for row in completion_ids:
            toks = row[row != pad_id]
            out.append(-abs(target_len - (toks.numel())))
        return torch.tensor(out, dtype=torch.float32)

    return reward_fn


def build_model(checkpoint_dir, fabric, dtype) -> GPT:
    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    with fabric.init_module(empty_init=True):
        gpt_model = GPT(config)
    load_checkpoint(fabric, gpt_model, checkpoint_dir / LIT_MODEL_FNAME)
    return gpt_model


def run_one(gpt_model, snapshot, fabric, dtype, prompt_ids, reward_fn,
            cache_name, cache_length, args, eos_id, pad_id):
    """Assign the given cache and run warmup + measured GRPO steps. Returns
    a metrics dict, or {'oom': True} if the config runs out of memory."""
    is_cuda = fabric.device.type == "cuda"
    try:
        if snapshot is not None:
            gpt_model.load_state_dict(snapshot, strict=True)
        deallocate_kv_cache_buffers_of_model(gpt_model)
        batch_size = args.prompts_per_step * args.group_size
        gpt_model.assign_kv_caches(
            KVCacheFactory.create(
                gpt_model=gpt_model, name=cache_name, max_batch_size=batch_size,
                cache_length=cache_length, dtype=dtype,
            )
        )
        opt = torch.optim.SGD(gpt_model.parameters(), lr=1e-7)
        pb = prompt_ids.repeat(args.prompts_per_step, 1).to(fabric.device)

        def step():
            return grpo_step(
                gpt_model=gpt_model, prompt_ids=pb, reward_fn=reward_fn, optimizer=opt,
                group_size=args.group_size, max_new_tokens=args.max_new_tokens,
                chunk_size=args.chunk_size, layers_per_cell=args.layers_per_cell,
                temperature=1.0, eos_token_id=eos_id, pad_token_id=pad_id, profile=True,
            )

        for _ in range(args.warmup):
            step()
        if is_cuda:
            torch.cuda.synchronize(fabric.device)
            torch.cuda.reset_peak_memory_stats(fabric.device)
        metrics, wall = [], 0.0
        for _ in range(args.iters):
            if is_cuda:
                torch.cuda.synchronize(fabric.device)
            t0 = time.perf_counter()
            metrics.append(step())
            if is_cuda:
                torch.cuda.synchronize(fabric.device)
            wall += (time.perf_counter() - t0) * 1000.0
        peak = torch.cuda.max_memory_allocated(fabric.device) / 1e9 if is_cuda else 0.0
        agg = lambda k: statistics.mean(m[k] for m in metrics)  # noqa: E731
        return {
            "gen": agg("gen_time_ms"), "grad": agg("grad_time_ms"),
            "total": wall / args.iters, "peak_gb": peak, "oom": False,
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if is_cuda:
                torch.cuda.empty_cache()
            return {"oom": True}
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--context-lengths", default="2048,8192,16384,32768")
    p.add_argument("--h2o-cache-length", type=int, default=4096)
    p.add_argument("--caches", default="dense-default,h2o-torch-quantized8")
    p.add_argument("--group-size", type=int, default=2)
    p.add_argument("--prompts-per-step", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--layers-per-cell", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument(
        "--no-reset",
        action="store_true",
        help="Skip the per-config weight-reset snapshot. Saves ~model-size of "
        "CPU RAM (needed for large models on small-RAM hosts); weights drift "
        "negligibly at the tiny profiling LR.",
    )
    p.add_argument("--access-token", default=None)
    args = p.parse_args()

    lengths = [int(x) for x in args.context_lengths.split(",")]
    caches = [c.strip() for c in args.caches.split(",")]
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    fabric = L.Fabric(devices=1, accelerator=args.device,
                      precision="32-true" if args.device == "cpu" else "bf16-true")

    checkpoint_dir = auto_download_checkpoint(model_name=args.model, access_token=args.access_token)
    tokenizer = Tokenizer(checkpoint_dir)
    pad_id = tokenizer.processor.token_to_id("<|endoftext|>")
    if pad_id is None:
        pad_id = int(tokenizer.eos_id) if tokenizer.eos_id is not None else 0
    eos_id = int(tokenizer.eos_id) if tokenizer.eos_id is not None else None

    gpt_model = build_model(checkpoint_dir, fabric, dtype).to(fabric.device)
    # Keep the reset snapshot on CPU so it doesn't consume GPU memory. Skip it
    # entirely with --no-reset to save host RAM (a full state-dict clone is
    # ~model-size; on small-RAM hosts a large model otherwise triggers the OS
    # OOM-killer). Weights drift negligibly at the profiling LR.
    snapshot = (
        None
        if args.no_reset
        else {k: v.detach().cpu().clone() for k, v in gpt_model.state_dict().items()}
    )
    reward_fn = make_reward_len(tokenizer, args.max_new_tokens, pad_id)

    print(f"\nmodel={args.model}  device={args.device}  batch={args.prompts_per_step*args.group_size}"
          f"  h2o_cache={args.h2o_cache_length}  chunk={args.chunk_size}\n")
    hdr = f"{'ctx':>7} {'cache':>22} | {'gen':>8} {'grad':>8} {'total':>9} (ms) | {'peakGB':>7}"
    print(hdr); print("-" * len(hdr))

    for L_ctx in lengths:
        prompt_ids = build_long_prompt(tokenizer, L_ctx, fabric.device).unsqueeze(0)
        rows = {}
        for cache_name in caches:
            # dense must hold the whole sequence; H2O is capped at a fixed budget.
            if cache_name.startswith("h2o") or cache_name.startswith("qh2o"):
                cache_length = min(args.h2o_cache_length, L_ctx + args.max_new_tokens)
            else:
                cache_length = L_ctx + args.max_new_tokens
            r = run_one(gpt_model, snapshot, fabric, dtype, prompt_ids, reward_fn,
                        cache_name, cache_length, args, eos_id, pad_id)
            rows[cache_name] = r
            tag = f"{cache_name}(cl={cache_length})"
            if r.get("oom"):
                print(f"{L_ctx:>7} {tag:>22} | {'OOM':>8} {'OOM':>8} {'OOM':>9}      | {'OOM':>7}")
            else:
                print(f"{L_ctx:>7} {tag:>22} | {r['gen']:>8.0f} {r['grad']:>8.0f} "
                      f"{r['total']:>9.0f}      | {r['peak_gb']:>7.2f}")
        d, h = rows.get("dense-default"), rows.get("h2o-torch-quantized8")
        if d and h and not d.get("oom") and not h.get("oom"):
            mem = 100.0 * (d["peak_gb"] - h["peak_gb"]) / d["peak_gb"] if d["peak_gb"] else 0.0
            spd = 100.0 * (d["total"] - h["total"]) / d["total"] if d["total"] else 0.0
            print(f"{'':>7} {'Δ H2O vs dense':>22} | mem {mem:+.0f}%  time {spd:+.0f}%")
    print("\nDone.")


if __name__ == "__main__":
    main()
