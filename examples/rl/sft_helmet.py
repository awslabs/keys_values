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
SFT baseline on HELMET long-context tasks, matched to the GRPO driver.

Supervised fine-tuning on (prompt, reference-answer) pairs from a HELMET
dataset, through the same memory-bounded chunked forward/backward and the
same (optionally evicting) KV cache as :mod:`grpo_helmet` -- so SFT-vs-GRPO
comparisons are apples-to-apples: same data, cache, eval, and step budget.

Uses the repo's existing SFT machinery (:class:`CrossEntropyOnLogits` +
:class:`LongContextGradientModel`); this driver only adds the HELMET plumbing
and matched evaluation.

    python examples/rl/sft_helmet.py --device cuda --dataset-key nq \
        --kv-cache-name h2o-torch-quantized8 --cache-length 4096 \
        --steps 400 --out-dir runs/nq_sft_h2o
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from litgpt.prompts import PromptStyle, has_prompt_style, load_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    auto_download_checkpoint,
    check_valid_checkpoint_dir,
)

sys.path.insert(0, str(Path(__file__).parent))
from grpo_helmet import evaluate, targets_of  # noqa: E402

from keys_values.finetune.utils import load_checkpoint
from keys_values.rl.runtime import SingleDevice
from keys_values.config import Config
from keys_values.data.constants import LIT_MODEL_FNAME
from keys_values.data.load_helmet_dev_eval import load_helmet_dev_eval
from keys_values.head_model import CrossEntropyOnLogits
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.kvcache.gradient.main import LongContextGradientModel
from keys_values.model import GPT
from keys_values.utils import VerbosityLevels


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--dataset-key", default="nq")
    p.add_argument("--max-length", default="8k")
    p.add_argument("--dataset-parent-dir", default=None)
    p.add_argument("--kv-cache-name", default="h2o-torch-quantized8")
    p.add_argument(
        "--cache-length",
        type=int,
        default=4096,
        help="KV cache budget; 0 = size to the longest sequence (dense).",
    )
    p.add_argument("--prompts-per-update", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--layers-per-cell", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--n-eval", type=int, default=50)
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/sft_helmet")
    p.add_argument("--disable-flashinfer", action="store_true")
    p.add_argument("--access-token", default=None)
    args = p.parse_args()

    if args.disable_flashinfer:
        from keys_values.attention import flashinfer_ops

        flashinfer_ops._available = False

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    fabric = SingleDevice.create(args.device)

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
    eos_id = int(tokenizer.eos_id) if tokenizer.eos_id is not None else None

    load_kwargs = dict(tokenizer=tokenizer.processor, max_length=args.max_length)
    if args.dataset_parent_dir:
        load_kwargs["dataset_parent_dir"] = args.dataset_parent_dir
    dev_data, eval_data = load_helmet_dev_eval(args.dataset_key, **load_kwargs)
    rng = random.Random(args.seed)
    train_records = list(dev_data)[: args.n_train]
    eval_records = list(eval_data)[: args.n_eval]

    # Pre-tokenize: full = prompt + reference answer (+ eos); SFT targets are
    # the answer tokens, right-aligned (same alignment convention as GRPO:
    # model input is full[:, :-1]).
    full_seqs, tgt_seqs = [], []
    for r in train_records:
        p_ids = tokenizer.encode(prompt_style.apply(r["input"]))
        a_ids = tokenizer.encode(" " + targets_of(r)[0], eos=True)
        # cross_entropy requires Long targets
        full_seqs.append(torch.cat([p_ids, a_ids]).long())
        tgt_seqs.append(a_ids.long())
    lens = [int(s.size(0)) for s in full_seqs]
    max_len = max(lens)
    cache_length = args.cache_length or (max_len + 8)
    print(
        f"HELMET {args.dataset_key}/{args.max_length}: {len(train_records)} train, "
        f"{len(eval_records)} eval; seq tokens avg={sum(lens)//len(lens)} max={max_len}; "
        f"cache={args.kv_cache_name}@{cache_length} "
        f"({'EVICTING' if cache_length < max_len else 'no eviction'})"
    )

    check_valid_checkpoint_dir(checkpoint_dir)
    with fabric.init_module(empty_init=True):
        gpt_model = GPT(config)
    load_checkpoint(gpt_model, checkpoint_dir / LIT_MODEL_FNAME)
    gpt_model.to(fabric.device)
    cache_kwargs = {}
    if (
        args.kv_cache_name.startswith(("h2o", "qh2o"))
        and "orig" not in args.kv_cache_name
    ):
        cache_kwargs["grace_period"] = cache_length // 16
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            gpt_model=gpt_model,
            name=args.kv_cache_name,
            max_batch_size=1,
            cache_length=cache_length,
            dtype=dtype,
            cache_kwargs=cache_kwargs,
        )
    )
    optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=args.lr)

    caps = [
        kvc.cache_length
        - (getattr(kvc, "grace_period", 0) or getattr(kvc, "init_grace_tokens", 0) or 0)
        for kvc in gpt_model.get_kv_caches()
        if kvc is not None
    ]
    chunk_size = max(min([args.chunk_size] + caps), 1)

    acc0 = evaluate(
        gpt_model,
        eval_records,
        tokenizer,
        prompt_style,
        pad_id,
        eos_id,
        args.max_new_tokens,
        chunk_size,
        fabric,
    )
    print(f"step 0 | eval_acc {acc0:.3f}", flush=True)
    with metrics_path.open("a") as f:
        f.write(json.dumps({"step": 0, "eval_acc": acc0}) + "\n")

    torch.manual_seed(args.seed)
    K = max(args.prompts_per_update, 1)
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for k in range(K):
            i = rng.randrange(len(train_records))
            full = full_seqs[i].unsqueeze(0).to(fabric.device)
            tgt = tgt_seqs[i].unsqueeze(0).to(fabric.device)
            gpt_model.eval()  # forward pass caches run in inference layout
            gpt_model.max_seq_length = int(full.shape[1])
            head = CrossEntropyOnLogits(gpt_model.config)
            grad_model = LongContextGradientModel(
                gpt_model=gpt_model,
                head_model=head,
                layers_per_cell=args.layers_per_cell,
                chunk_size=chunk_size,
                verbose=VerbosityLevels.NONE,
            )
            grad_model.train()
            loss = grad_model(full[:, :-1], tgt, scale_factor=1.0 / K)
            loss.backward()
            losses.append(float(loss.detach().mean().item()))
        optimizer.step()
        m = {
            "step": step,
            "loss": sum(losses) / K,
            "seq_len": int(full.shape[1]),
            "step_time_s": round(time.perf_counter() - t0, 2),
        }
        if step % args.eval_every == 0 or step == args.steps:
            m["eval_acc"] = evaluate(
                gpt_model,
                eval_records,
                tokenizer,
                prompt_style,
                pad_id,
                eos_id,
                args.max_new_tokens,
                chunk_size,
                fabric,
            )
        with metrics_path.open("a") as f:
            f.write(json.dumps(m) + "\n")
        line = f"step {step:4d} | loss {m['loss']:.4f} | seq {m['seq_len']} | {m['step_time_s']}s"
        if "eval_acc" in m:
            line += f" | eval_acc {m['eval_acc']:.3f}"
        print(line, flush=True)

    torch.save(gpt_model.state_dict(), out_dir / "final.pt")
    print(f"Done. Metrics: {metrics_path}  Weights: {out_dir/'final.pt'}")


if __name__ == "__main__":
    main()
