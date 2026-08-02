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
GRPO training on HELMET long-context tasks, with periodic held-out evaluation.

Trains a policy with :func:`keys_values.rl.grpo.loop.grpo_step` on a HELMET
dataset (default: ``nq`` RAG-QA at the 8k context bucket). The reward is
``sub_exact_match`` of the generated answer against the reference answers --
the same metric the repo's HELMET evaluation harness uses -- so training
reward and evaluation measure the same thing.

Run one arm per invocation; compare a sparse arm against a dense arm:

    # sparse (H2O) arm: cache budget < context, so eviction is real
    python examples/grpo_helmet.py --device cuda --kv-cache-name h2o-torch-quantized8 \
        --cache-length 4096 --steps 150 --out-dir runs/h2o

    # dense (full attention) arm at the same settings
    python examples/grpo_helmet.py --device cuda --kv-cache-name dense-default \
        --cache-length 0 --steps 150 --out-dir runs/dense   # 0 = fit longest prompt

Metrics stream to ``<out-dir>/metrics.jsonl``; the final weights are saved to
``<out-dir>/final.pt``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import lightning as L
import torch
from litgpt.prompts import PromptStyle, has_prompt_style, load_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import auto_download_checkpoint, check_valid_checkpoint_dir, load_checkpoint

from keys_values.config import Config
from keys_values.data.constants import LIT_MODEL_FNAME
from keys_values.data.load_helmet_dev_eval import load_helmet_dev_eval
from keys_values.evaluation.metrics import rouge_n_f1, sub_exact_match
from keys_values.kvcache.factory import KVCacheFactory, deallocate_kv_cache_buffers_of_model
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT
from keys_values.rl.grpo.loop import grpo_step
from keys_values.rl.grpo.rollout import generate_completions
from keys_values.utils import VerbosityLevels


def targets_of(record) -> list[str]:
    out = record["output"]
    return [str(x) for x in out] if isinstance(out, (list, tuple)) else [str(out)]


def shaped_reward(text: str, targets: list[str], kind: str) -> float:
    """Reward for one completion.

    ``em``: binary substring exact match (sparse signal).
    ``f1``: max(EM, token-level F1) -- partial credit densifies the
    within-group variance that GRPO's group-relative advantages need.
    """
    em = float(any(sub_exact_match(text, t) for t in targets))
    if kind == "em" or em == 1.0:
        return em
    return max((rouge_n_f1(text, t, n=1) for t in targets), default=0.0)


def decode_row(tokenizer, row: torch.Tensor, pad_id: int) -> str:
    toks = row[row != pad_id]
    return tokenizer.decode(toks) if toks.numel() else ""


@torch.no_grad()
def evaluate(gpt_model, records, tokenizer, prompt_style, pad_id, eos_id,
             max_new, chunk_size, fabric) -> float:
    """Greedy generation + sub_exact_match over held-out records."""
    gpt_model.eval()
    correct = 0
    for rec in records:
        ids = tokenizer.encode(prompt_style.apply(rec["input"]), device=fabric.device)
        prompt = ids.unsqueeze(0)
        gpt_model.max_seq_length = int(prompt.shape[1]) + max_new
        caps = [kvc.cache_length - (getattr(kvc, "grace_period", 0)
                or getattr(kvc, "init_grace_tokens", 0) or 0)
                for kvc in gpt_model.get_kv_caches() if kvc is not None]
        inf = LongContextInferenceModel(
            gpt_model, head_model=None,
            chunk_size=max(min([chunk_size] + caps), 1),
            verbose=VerbosityLevels.NONE)
        comp = generate_completions(
            model=inf, prompt_ids=prompt, max_new_tokens=max_new,
            temperature=1.0, top_k=1, top_p=1.0,
            eos_token_id=eos_id, pad_token_id=pad_id,
            # The cache buffers are shared with the training rollout/backward,
            # which updates them in place: inference_mode would taint them.
            no_inference_mode=True)
        text = decode_row(tokenizer, comp[0], pad_id)
        correct += int(any(sub_exact_match(text, t) for t in targets_of(rec)))
    return correct / max(len(records), 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--dataset-key", default="nq")
    p.add_argument("--max-length", default="8k")
    p.add_argument("--dataset-parent-dir", default=None)
    p.add_argument("--kv-cache-name", default="h2o-torch-quantized8")
    p.add_argument("--cache-length", type=int, default=4096,
                   help="KV cache budget; 0 = size to the longest prompt (dense).")
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--prompts-per-update", type=int, default=1,
                   help="Gradient accumulation: prompts (each with group-size "
                        "rollouts) folded into one optimizer update.")
    p.add_argument("--reward", choices=["em", "f1"], default="f1",
                   help="'f1' = max(exact-match, token-F1) partial credit (denser signal).")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--layers-per-cell", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--n-eval", type=int, default=24)
    p.add_argument("--n-train", type=int, default=512, help="Training prompts drawn from the dev split.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/grpo_helmet")
    p.add_argument("--disable-flashinfer", action="store_true",
                   help="Force eager SDPA (needed for GQA group sizes FlashInfer rejects, e.g. Qwen-0.5B).")
    p.add_argument("--access-token", default=None)
    args = p.parse_args()

    if args.disable_flashinfer:
        # Example-script escape hatch: FlashInfer's decode kernel rejects some
        # GQA group sizes (e.g. 7 for Qwen2.5-0.5B); fall back to eager SDPA.
        from keys_values.attention import flashinfer_ops
        flashinfer_ops._available = False

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    fabric = L.Fabric(devices=1, accelerator=args.device,
                      precision="32-true" if args.device == "cpu" else "bf16-true")

    checkpoint_dir = auto_download_checkpoint(model_name=args.model, access_token=args.access_token)
    tokenizer = Tokenizer(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    prompt_style = (load_prompt_style(checkpoint_dir) if has_prompt_style(checkpoint_dir)
                    else PromptStyle.from_config(config))
    pad_id = tokenizer.processor.token_to_id("<|endoftext|>")
    if pad_id is None:
        pad_id = int(tokenizer.eos_id) if tokenizer.eos_id is not None else 0
    eos_id = int(tokenizer.eos_id) if tokenizer.eos_id is not None else None

    # HELMET data: dev split for training prompts, eval split held out.
    load_kwargs = dict(tokenizer=tokenizer.processor, max_length=args.max_length)
    if args.dataset_parent_dir:
        load_kwargs["dataset_parent_dir"] = args.dataset_parent_dir
    dev_data, eval_data = load_helmet_dev_eval(args.dataset_key, **load_kwargs)
    rng = random.Random(args.seed)
    train_records = list(dev_data)[: args.n_train]
    eval_records = list(eval_data)[: args.n_eval]
    print(f"HELMET {args.dataset_key}/{args.max_length}: {len(train_records)} train, "
          f"{len(eval_records)} eval records")

    # Tokenized prompt lengths (needed for cache sizing).
    enc = [tokenizer.encode(prompt_style.apply(r["input"])) for r in train_records]
    lens = [int(e.size(0)) for e in enc]
    max_len = max(lens)
    cache_length = args.cache_length or (max_len + args.max_new_tokens + 8)
    print(f"prompt tokens: min={min(lens)} avg={sum(lens)//len(lens)} max={max_len}; "
          f"cache={args.kv_cache_name}@{cache_length} "
          f"({'EVICTING' if cache_length < max_len else 'no eviction'})")

    # Model + caches + optimizer.
    check_valid_checkpoint_dir(checkpoint_dir)
    with fabric.init_module(empty_init=True):
        gpt_model = GPT(config)
    load_checkpoint(fabric, gpt_model, checkpoint_dir / LIT_MODEL_FNAME)
    gpt_model.to(fabric.device)
    gpt_model.assign_kv_caches(KVCacheFactory.create(
        gpt_model=gpt_model, name=args.kv_cache_name, max_batch_size=args.group_size,
        cache_length=cache_length, dtype=dtype))
    optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=args.lr)

    def reward_fn(prompt_ids: torch.Tensor, completion_ids: torch.Tensor) -> torch.Tensor:
        rec = reward_fn.current_record
        tgts = targets_of(rec)
        vals = [shaped_reward(decode_row(tokenizer, row, pad_id), tgts, args.reward)
                for row in completion_ids]
        return torch.tensor(vals, dtype=torch.float32)

    # Initial eval.
    acc0 = evaluate(gpt_model, eval_records, tokenizer, prompt_style, pad_id, eos_id,
                    args.max_new_tokens, args.chunk_size, fabric)
    print(f"step 0 | eval_acc {acc0:.3f}")
    with metrics_path.open("a") as f:
        f.write(json.dumps({"step": 0, "eval_acc": acc0}) + "\n")

    # Training loop: each update accumulates K prompts x group_size rollouts.
    torch.manual_seed(args.seed)
    n_signal = 0
    K = max(args.prompts_per_update, 1)
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        micro = []
        for k in range(K):
            i = rng.randrange(len(train_records))
            reward_fn.current_record = train_records[i]
            prompt_ids = enc[i].unsqueeze(0).to(fabric.device)
            micro.append(grpo_step(
                gpt_model=gpt_model, prompt_ids=prompt_ids, reward_fn=reward_fn,
                optimizer=optimizer, group_size=args.group_size,
                max_new_tokens=args.max_new_tokens, chunk_size=args.chunk_size,
                layers_per_cell=args.layers_per_cell, temperature=args.temperature,
                eos_token_id=eos_id, pad_token_id=pad_id,
                zero_grad=(k == 0), optimizer_step=(k == K - 1),
                grad_scale=1.0 / K))
            micro[-1]["prompt_len"] = int(prompt_ids.shape[1])
        m = {
            "step": step,
            "mean_reward": sum(x["mean_reward"] for x in micro) / K,
            "advantage_std": max(x["advantage_std"] for x in micro),
            "loss": sum(x["loss"] for x in micro) / K,
            "prompt_len": int(sum(x["prompt_len"] for x in micro) / K),
        }
        m["step_time_s"] = round(time.perf_counter() - t0, 2)
        # An update where every group has zero reward variance produces zero
        # advantage -> no gradient. Track how often updates carry signal.
        m["has_signal"] = int(any(x["advantage_std"] > 1e-6 for x in micro))
        n_signal += m["has_signal"]
        m["signal_rate"] = round(n_signal / step, 3)
        if step % args.eval_every == 0 or step == args.steps:
            m["eval_acc"] = evaluate(gpt_model, eval_records, tokenizer, prompt_style,
                                     pad_id, eos_id, args.max_new_tokens,
                                     args.chunk_size, fabric)
        with metrics_path.open("a") as f:
            f.write(json.dumps(m) + "\n")
        line = (f"step {step:4d} | reward {m['mean_reward']:.3f} "
                f"| adv_std {m['advantage_std']:.3f} | signal {m['signal_rate']:.2f} "
                f"| ctx {m['prompt_len']} | {m['step_time_s']}s")
        if "eval_acc" in m:
            line += f" | eval_acc {m['eval_acc']:.3f}"
        print(line, flush=True)

    torch.save(gpt_model.state_dict(), out_dir / "final.pt")
    print(f"Done. Metrics: {metrics_path}  Weights: {out_dir/'final.pt'}")


if __name__ == "__main__":
    main()
