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
Cross-evaluation of GRPO checkpoints: training cache x inference cache.

Evaluates each checkpoint (plus the untrained base model) under BOTH a dense
and a sparse (H2O) inference cache on a HELMET eval split. This separates the
two effects that a plain per-arm eval conflates:

- Did training under a sparse cache damage the *policy*?
  (compare rows under the same eval cache)
- Does sparse *inference* cost accuracy, independent of training?
  (compare columns for the same checkpoint)

    python examples/grpo_helmet_crosseval.py --device cuda --dataset-key nq \
        --checkpoints base h2o=runs/nq_h2o/final.pt dense=runs/nq_dense/final.pt \
        --h2o-cache-length 4096 --n-eval 100 --disable-flashinfer \
        --out runs/crosseval_nq.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightning as L
import torch
from litgpt.prompts import PromptStyle, has_prompt_style, load_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import auto_download_checkpoint, check_valid_checkpoint_dir, load_checkpoint

sys.path.insert(0, str(Path(__file__).parent))
from grpo_helmet import decode_row, targets_of  # noqa: E402

from keys_values.config import Config
from keys_values.data.constants import LIT_MODEL_FNAME
from keys_values.data.load_helmet_dev_eval import load_helmet_dev_eval
from keys_values.evaluation.metrics import rouge_n_f1, sub_exact_match
from keys_values.kvcache.factory import KVCacheFactory, deallocate_kv_cache_buffers_of_model
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT
from keys_values.rl.grpo.rollout import generate_completions
from keys_values.utils import VerbosityLevels


@torch.no_grad()
def eval_records(gpt_model, records, tokenizer, prompt_style, pad_id, eos_id,
                 max_new, chunk_size, fabric):
    gpt_model.eval()
    em_sum, f1_sum = 0.0, 0.0
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
            eos_token_id=eos_id, pad_token_id=pad_id, no_inference_mode=True)
        text = decode_row(tokenizer, comp[0], pad_id)
        tgts = targets_of(rec)
        em_sum += float(any(sub_exact_match(text, t) for t in tgts))
        f1_sum += max((rouge_n_f1(text, t, n=1) for t in tgts), default=0.0)
    n = max(len(records), 1)
    return em_sum / n, f1_sum / n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--dataset-key", default="nq")
    p.add_argument("--max-length", default="8k")
    p.add_argument("--dataset-parent-dir", default=None)
    p.add_argument("--checkpoints", nargs="+", default=["base"],
                   help="Entries 'name=path/to/state_dict.pt' or 'base' (untrained).")
    p.add_argument("--h2o-cache-length", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--out", default=None)
    p.add_argument("--disable-flashinfer", action="store_true")
    p.add_argument("--access-token", default=None)
    args = p.parse_args()

    if args.disable_flashinfer:
        from keys_values.attention import flashinfer_ops
        flashinfer_ops._available = False

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

    load_kwargs = dict(tokenizer=tokenizer.processor, max_length=args.max_length)
    if args.dataset_parent_dir:
        load_kwargs["dataset_parent_dir"] = args.dataset_parent_dir
    _, eval_data = load_helmet_dev_eval(args.dataset_key, **load_kwargs)
    # Maximize question diversity: the eval split contains several context
    # variants per question (e.g. 600 instances over 100 questions for the
    # RAG tasks), and taking the first n in file order covers only a
    # handful of distinct questions (17 at n=100 on nq), inflating variance
    # far beyond the nominal-n standard error. Round-robin over questions
    # instead: first one instance per distinct question, then seconds, etc.
    by_qid: dict = {}
    for rec in eval_data:
        by_qid.setdefault(rec["query_id"], []).append(rec)
    records = []
    rank = 0
    while len(records) < args.n_eval and any(rank < len(v) for v in by_qid.values()):
        for variants in by_qid.values():
            if rank < len(variants) and len(records) < args.n_eval:
                records.append(variants[rank])
        rank += 1
    print(f"Selected {len(records)} records covering "
          f"{len(set(r['query_id'] for r in records))} distinct questions")
    lens = [int(tokenizer.encode(prompt_style.apply(r["input"])).size(0)) for r in records]
    max_len = max(lens)
    print(f"HELMET {args.dataset_key}/{args.max_length}: {len(records)} eval records, "
          f"prompt tokens max={max_len}")

    check_valid_checkpoint_dir(checkpoint_dir)
    with fabric.init_module(empty_init=True):
        gpt_model = GPT(config)
    load_checkpoint(fabric, gpt_model, checkpoint_dir / LIT_MODEL_FNAME)
    gpt_model.to(fabric.device)
    base_sd = {k: v.detach().cpu().clone() for k, v in gpt_model.state_dict().items()}

    eval_caches = [
        ("dense", "dense-default", max_len + args.max_new_tokens + 8),
        ("h2o", "h2o-torch-quantized8", args.h2o_cache_length),
    ]

    results = {}
    hdr = f"{'checkpoint':>12} {'eval-cache':>10} | {'EM':>6} {'F1':>6}"
    print("\n" + hdr); print("-" * len(hdr))
    for entry in args.checkpoints:
        if entry == "base":
            name, sd = "base", base_sd
        else:
            name, path = entry.split("=", 1)
            sd = torch.load(path, map_location="cpu", weights_only=True)
        gpt_model.load_state_dict(sd, strict=True)
        for cache_tag, cache_name, cl in eval_caches:
            deallocate_kv_cache_buffers_of_model(gpt_model)
            ckw = {"grace_period": cl // 16} if cache_name.startswith("h2o") else {}
            gpt_model.assign_kv_caches(KVCacheFactory.create(
                gpt_model=gpt_model, name=cache_name, max_batch_size=1,
                cache_length=cl, dtype=dtype, cache_kwargs=ckw))
            em, f1 = eval_records(gpt_model, records, tokenizer, prompt_style,
                                  pad_id, eos_id, args.max_new_tokens,
                                  args.chunk_size, fabric)
            results[f"{name}/{cache_tag}"] = {"em": round(em, 4), "f1": round(f1, 4)}
            print(f"{name:>12} {cache_tag:>10} | {em:>6.3f} {f1:>6.3f}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"dataset": args.dataset_key, "max_length": args.max_length,
                       "n_eval": len(records), "results": results}, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
