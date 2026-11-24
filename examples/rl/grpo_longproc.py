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
GRPO/RLOO on LongProc (long procedural generation) through a KeysAndValues
sparse KV cache.

LongProc (arXiv:2501.05414; princeton-pli/LongProc) tasks require *generating*
long, rule-checkable outputs (0.5k/2k/8k-token tiers) from long inputs:
html_to_tsv (extract tables; scored by row F1), travel_planning (constraint
satisfaction), countdown, tom_tracking, path_traversal, pseudo_to_code. The
programmatic evaluators double as verifiable RL rewards, and the long
*decode* exercises cache eviction during generation -- a regime the HELMET QA
campaigns (long prefill, 32-token decode) never touch.

Setup: clone LongProc next to the repo (data ships in-repo, ~100MB):

    git clone --depth 1 https://github.com/princeton-pli/LongProc.git repos/longproc

Example (7B on a 48GB card, known-good memory recipe):

    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python examples/rl/grpo_longproc.py --device cuda \
        --model Qwen/Qwen2.5-7B-Instruct --dataset html_to_tsv_2k \
        --kv-cache-name h2o-torch-quantized8 --cache-length 8192 \
        --group-size 4 --prompts-per-update 2 --optimizer paged_adamw8bit \
        --disable-flashinfer --out-dir runs/longproc_html2k
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

from keys_values.finetune.utils import load_checkpoint
from keys_values.rl.runtime import SingleDevice
from keys_values.config import Config
from keys_values.data.constants import LIT_MODEL_FNAME
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.long_context import LongContextInferenceModel
from keys_values.lora import (
    GPT as GPTLoRA,
    Config as ConfigLoRA,
    mark_only_lora_as_trainable,
)
from keys_values.model import GPT
from keys_values.rl.grpo.attention import (
    ATTN_BACKENDS,
    attention_mha_kwargs,
    gradient_train_cache_kwargs,
)
from keys_values.rl.grpo.loop import grpo_step
from keys_values.rl.grpo.rollout import generate_completions
from keys_values.utils import VerbosityLevels

# Primary metric per task family: the scalar the evaluator returns that we
# use as the reward (and eval score). Verified against each evaluator's
# actual output on reference completions (countdown/travel_planning return
# "accuracy", not "success").
PRIMARY_METRIC = {
    "html_to_tsv": "f1",
    "pseudo_to_code": "accuracy",
    "path_traversal": "accuracy",
    "tom_tracking": "accuracy",
    "countdown": "accuracy",
    "travel_planning": "accuracy",
}


def load_longproc(dataset: str, path: Path):
    sys.path.insert(0, str(path))
    from longproc.longproc_data import load_longproc_data  # noqa: E402

    data, eval_fn = load_longproc_data(dataset, str(path / "data"))
    family = dataset.rsplit("_", 1)[0]
    metric = PRIMARY_METRIC.get(family)
    if metric is None:
        raise ValueError(f"No primary metric registered for {family}")
    return data, eval_fn, metric


def primary_score(eval_fn, metric: str, prediction: str, record: dict) -> float:
    try:
        metrics, _ = eval_fn(prediction, record)
        return float(metrics.get(metric, 0.0))
    except Exception:
        # Evaluators are robust to garbage, but belt-and-braces: a crash in
        # the checker is a zero-reward completion, not a dead training run.
        return 0.0


def shaped_score(
    eval_fn, metric: str, prediction: str, record: dict, format_bonus: float
) -> float:
    """Training reward: primary metric plus a small format bonus.

    Sampled rollouts at temperature 1.0 almost never hit the strict output
    format the checkers require (e.g. a ```tsv fenced block), so the primary
    metric is 0.0 for every group member and RLOO gets no gradient (observed:
    30+ steps of all-zero rewards while greedy eval scores 0.157). The
    evaluators expose format adherence (``extraction_rate`` etc.); paying a
    small bonus for parseable output gives the group a reward *spread* to
    climb out of the cold start. Eval always uses the unshaped primary
    metric.
    """
    primary, fmt = score_components(eval_fn, metric, prediction, record)
    return primary + format_bonus * fmt


def score_components(
    eval_fn, metric: str, prediction: str, record: dict
) -> tuple[float, float]:
    """(primary metric, format adherence) for one completion, 0.0 on error.

    Logged separately per step: the shaped reward alone cannot distinguish
    "the model solves the task" from "the model learned to emit the fenced
    block". Observed on countdown_8k: shaped reward flat at 0.20 while greedy
    eval accuracy went 0.125 -> 0.000 -- the policy collected the format
    bonus and lost the answer.
    """
    try:
        metrics, _ = eval_fn(prediction, record)
        return (
            float(metrics.get(metric, 0.0)),
            float(metrics.get("extraction_rate", 0.0)),
        )
    except Exception:
        return 0.0, 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--dataset", default="html_to_tsv_2k")
    p.add_argument("--longproc-path", default="repos/longproc")
    p.add_argument("--kv-cache-name", default="h2o-torch-quantized8")
    p.add_argument("--cache-length", type=int, default=8192)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--prompts-per-update", type=int, default=2)
    p.add_argument("--adv-mode", choices=["grpo", "rloo"], default="rloo")
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=2600,
        help="Output budget; the _2k tiers need ~2200 tokens.",
    )
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument(
        "--optimizer", choices=["adamw", "paged_adamw8bit"], default="paged_adamw8bit"
    )
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument(
        "--attn",
        default="auto",
        choices=ATTN_BACKENDS,
        help="Attention backend. auto: FlashInfer decode if built, else "
        "FlexAttention decode with kv_lens padding (bounded compiles), and "
        "FlexAttention in the gradient cells on CUDA (C-long queries, no "
        "K^2/C memory term); flex: FlexAttention everywhere, decode with "
        "kv_lens padding; eager: eager decode and query-padded PyTorch SDPA "
        "in the gradient cells (the pre-Flex path, A/B baseline). "
        "--disable-flashinfer maps auto to eager.",
    )
    p.add_argument(
        "--backward-tmp-gb",
        type=float,
        default=2.0,
        help="Limit (GiB) for temporary device arrays in the "
        "chunked backward (0 disables). Needed at 32k+.",
    )
    p.add_argument(
        "--lora-r",
        type=int,
        default=0,
        help="LoRA rank (0 = full fine-tuning). Adapters on "
        "q/k/v/proj/mlp, alpha=2r. Applies to sparse and "
        "dense arms alike.",
    )
    p.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Also save a full state_dict at every eval step "
        "(15 GB each for 7B). Off by default: this filled "
        "worker disks. final.pt is always written.",
    )
    p.add_argument(
        "--dense-baseline",
        action="store_true",
        help="Dense-RL baseline: compute the gradient with one "
        "full-sequence backward instead of the memory-bounded "
        "chunked path. Same loss and gradients (see "
        "test_dense_baseline_matches_chunked_gradient); much "
        "higher peak memory. Use for paper baselines.",
    )
    p.add_argument(
        "--truncate-prompt",
        type=int,
        default=0,
        help="Prompt token budget (0 = off), applied in training "
        "AND evaluation. Instruction-preserving: keeps the "
        "head and tail of the prompt and cuts the document "
        "in the middle. For the dense-truncated baseline.",
    )
    p.add_argument("--layers-per-cell", type=int, default=1)
    p.add_argument(
        "--evict-every",
        type=int,
        default=1,
        help="Block eviction for H2O caches: rank slots once per "
        "B decoded tokens instead of every token (1 = off).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Rollout sampling temperature. Long structured outputs "
        "derail badly at 1.0 (zero parseable rollouts observed).",
    )
    p.add_argument(
        "--format-bonus",
        type=float,
        default=0.2,
        help="Training-reward bonus per unit of format adherence "
        "(extraction_rate); 0 disables shaping.",
    )
    p.add_argument(
        "--format-gate",
        type=float,
        default=0.5,
        help="Pay the format bonus only while the fraction of "
        "parseable rollouts in the group is below this. "
        ">= 1.0 disables the gate (old always-on bonus, which "
        "the policy locked onto: countdown 0.125 -> 0.0).",
    )
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--n-eval", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/grpo_longproc")
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Base-model probe: score n-eval records, no training.",
    )
    p.add_argument("--disable-flashinfer", action="store_true")
    p.add_argument("--access-token", default=None)
    args = p.parse_args()

    if args.disable_flashinfer:
        from keys_values.attention import flashinfer_ops

        flashinfer_ops._available = False

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    fabric = SingleDevice.create(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = auto_download_checkpoint(
        model_name=args.model, access_token=args.access_token
    )
    tokenizer = Tokenizer(checkpoint_dir)
    if args.lora_r > 0:
        # LoRA variant (same pattern as finetune_long_lora). Applied to BOTH
        # the sparse and dense arms, so a LoRA comparison stays apples to
        # apples; the point is a dense baseline that fits where full FT does
        # not (optimizer state and gradients shrink to the adapter).
        config = ConfigLoRA.from_file(
            checkpoint_dir / "model_config.yaml",
            lora_r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            lora_dropout=0.0,
            lora_query=True,
            lora_key=True,
            lora_value=True,
            lora_projection=True,
            lora_mlp=True,
            lora_head=False,
        )
    else:
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

    records, eval_fn, metric = load_longproc(args.dataset, Path(args.longproc_path))
    # Deterministic eval/train partition (records ship in fixed order).
    rng = random.Random(42)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    eval_idx = set(idx[: args.n_eval])
    eval_records = [records[i] for i in sorted(eval_idx)]
    train_records = [records[i] for i in idx[args.n_eval :]]
    print(
        f"{args.dataset}: {len(records)} records -> "
        f"{len(train_records)} train / {len(eval_records)} eval; "
        f"reward metric = {metric}"
    )

    check_valid_checkpoint_dir(checkpoint_dir)
    # Attention backend. Must reach both the model and the caches (the caches
    # build their own MHA, and the gradient cells reuse kv_cache.mha).
    attn_backend = (
        "eager" if args.disable_flashinfer and args.attn == "auto" else args.attn
    )
    mha_kwargs = attention_mha_kwargs(
        backend=attn_backend,
        kv_cache_name=args.kv_cache_name,
        chunk_size=args.chunk_size,
        device=fabric.device,
        cache_length=args.cache_length,
    )
    # Gradient cells only: FlexAttention under auto/flex (CUDA, own kv_lens=None
    # args), query-padded SDPA under eager. Reaches TrainingAttnWeightsReplayCache
    # via grpo_step; the decode MHAs above are not touched.
    train_cache_kwargs = gradient_train_cache_kwargs(
        backend=attn_backend,
        chunk_size=args.chunk_size,
        device=fabric.device,
    )
    with fabric.init_module(empty_init=True):
        gpt_model = (
            GPTLoRA(config, **mha_kwargs)
            if args.lora_r > 0
            else GPT(config, **mha_kwargs)
        )
    load_checkpoint(
        gpt_model, checkpoint_dir / LIT_MODEL_FNAME, strict=(args.lora_r == 0)
    )
    if args.lora_r > 0:
        mark_only_lora_as_trainable(gpt_model)
        n_train = sum(p.numel() for p in gpt_model.parameters() if p.requires_grad)
        print(
            f"LoRA r={args.lora_r}: {n_train / 1e6:.1f}M trainable params", flush=True
        )
    gpt_model.to(fabric.device)

    cache_kwargs = dict(mha_kwargs)
    if (
        args.kv_cache_name.startswith(("h2o", "qh2o"))
        and "orig" not in args.kv_cache_name
    ):
        cache_kwargs["grace_period"] = args.cache_length // 16
    if args.evict_every > 1:
        cache_kwargs["evict_every"] = args.evict_every
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            gpt_model=gpt_model,
            name=args.kv_cache_name,
            max_batch_size=args.group_size,
            cache_length=args.cache_length,
            dtype=dtype,
            cache_kwargs=cache_kwargs,
        )
    )

    def encode(rec):
        ids = tokenizer.encode(
            prompt_style.apply(rec["input_prompt"]), device=fabric.device
        )
        return truncate_prompt(ids, args.truncate_prompt)

    def truncate_prompt(ids: torch.Tensor, budget: int) -> torch.Tensor:
        """
        Dense-truncated baseline: what dense RL can afford at a matched memory
        budget. Applied identically in training and evaluation (a truncated
        arm evaluated on full prompts would be a different system).

        Instruction-preserving: LongProc prompts carry the task instructions
        at the head AND the output-format instructions at the tail, with the
        long document in between. Tail-only truncation would delete the task
        statement; so we keep the first `budget // 4` and last
        `budget - budget // 4` tokens and cut the middle of the document.
        """
        if budget <= 0 or ids.shape[-1] <= budget:
            return ids
        head_n = budget // 4
        return torch.cat((ids[:head_n], ids[-(budget - head_n) :]), dim=-1)

    @torch.no_grad()
    def eval_model(tag: str) -> float:
        gpt_model.eval()
        scores = []
        for rec in eval_records:
            ids = encode(rec).unsqueeze(0)
            gpt_model.max_seq_length = int(ids.shape[1]) + args.max_new_tokens
            inf = LongContextInferenceModel(
                gpt_model,
                head_model=None,
                chunk_size=args.chunk_size,
                verbose=VerbosityLevels.NONE,
            )
            comp = generate_completions(
                model=inf,
                prompt_ids=ids,
                max_new_tokens=args.max_new_tokens,
                temperature=1.0,
                top_k=1,
                top_p=1.0,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                no_inference_mode=True,
            )
            text = tokenizer.decode(comp[0][comp[0] != pad_id])
            scores.append(primary_score(eval_fn, metric, text, rec))
        mean = sum(scores) / max(len(scores), 1)
        print(f"[eval @ {tag}] {metric} = {mean:.3f} (n={len(scores)})", flush=True)
        return mean

    if args.eval_only:
        eval_model("base")
        return

    trainable = [p for p in gpt_model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    else:
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=args.lr)

    eval_model("step 0")
    history = []
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        micro_metrics = []
        step_primary: list[float] = []  # unshaped task metric per rollout
        step_format: list[float] = []  # format adherence per rollout
        step_spread: list[float] = []  # 1.0 per group with non-identical rewards
        for micro in range(args.prompts_per_update):
            rec = train_records[
                (step * args.prompts_per_update + micro) % len(train_records)
            ]
            prompt_ids = encode(rec).unsqueeze(0)  # truncation (if any) inside

            def reward_fn(p_ids, completion_ids):
                comps = []
                for row in completion_ids:
                    text = tokenizer.decode(row[row != pad_id])
                    comps.append(score_components(eval_fn, metric, text, rec))
                primaries = [c[0] for c in comps]
                fmts = [c[1] for c in comps]
                step_primary.extend(primaries)
                step_format.extend(fmts)
                # Gated format bonus. The bonus exists to create reward
                # SPREAD inside a group during cold start (some rollouts hit
                # the fenced format, some do not). Once most of the group
                # formats, a flat bonus is a constant that RLOO's baseline
                # removes -- but on the way there it is a standing reward the
                # policy can lock onto (observed: countdown accuracy
                # 0.125 -> 0.0 while shaped reward sat at exactly 0.20). So
                # pay it only while fewer than half the group is parseable.
                frac_fmt = sum(fmts) / max(len(fmts), 1)
                gated_off = args.format_gate < 1.0 and frac_fmt >= args.format_gate
                bonus = 0.0 if gated_off else args.format_bonus
                vals = [p + bonus * f for p, f in zip(primaries, fmts)]
                # Is there any gradient in this group? RLOO has one iff the
                # rewards are not all identical.
                step_spread.append(1.0 if max(vals) - min(vals) > 1e-9 else 0.0)
                return torch.tensor(vals, dtype=torch.float32)

            micro_metrics.append(
                grpo_step(
                    gpt_model=gpt_model,
                    prompt_ids=prompt_ids,
                    reward_fn=reward_fn,
                    optimizer=optimizer,
                    group_size=args.group_size,
                    max_new_tokens=args.max_new_tokens,
                    chunk_size=args.chunk_size,
                    layers_per_cell=args.layers_per_cell,
                    temperature=args.temperature,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                    advantage_mode=args.adv_mode,
                    zero_grad=(micro == 0),
                    optimizer_step=(micro == args.prompts_per_update - 1),
                    grad_scale=1.0 / args.prompts_per_update,
                    backward_tmp_gb=args.backward_tmp_gb,
                    dense_baseline=args.dense_baseline,
                    train_cache_kwargs=train_cache_kwargs,
                )
            )
        mean_r = sum(m["mean_reward"] for m in micro_metrics) / len(micro_metrics)
        dt = time.perf_counter() - t0
        n_roll = max(len(step_primary), 1)
        mean_primary = sum(step_primary) / n_roll
        mean_format = sum(step_format) / n_roll
        frac_spread = sum(step_spread) / max(len(step_spread), 1)
        entry = {
            "step": step,
            "reward": mean_r,
            "primary": mean_primary,
            "format": mean_format,
            "spread": frac_spread,
            "sec": dt,
        }
        mem_msg = (
            f" | primary {mean_primary:.3f} fmt {mean_format:.2f}"
            f" spread {frac_spread:.2f}"
        )
        for key in ("grad_peak_device_mib", "parked_peak_mib", "parked_peak_count"):
            if key in micro_metrics[0]:
                entry[key] = max(m[key] for m in micro_metrics)
        if "grad_peak_device_mib" in entry:
            mem_msg += f" | dev peak {entry['grad_peak_device_mib'] / 1024:.1f}G"
        if "parked_peak_mib" in entry:
            mem_msg += (
                f" | parked {entry['parked_peak_mib']:.0f}MiB"
                f"/{int(entry['parked_peak_count'])}"
            )
        if "grad_sdpa_mode" in micro_metrics[0]:
            # SDPA mode of the gradient cells (flexattention / qpadded-pytorch)
            entry["grad_sdpa_mode"] = micro_metrics[0]["grad_sdpa_mode"]
            mem_msg += f" | grad {entry['grad_sdpa_mode']}"
        print(f"step {step:4d} | reward {mean_r:.3f} | {dt:.1f}s{mem_msg}", flush=True)
        history.append(entry)
        if step % args.eval_every == 0:
            score = eval_model(f"step {step}")
            history.append({"step": step, "eval": score})
            if args.save_intermediate:
                # 15 GB per checkpoint for a 7B model; with --eval-every 50
                # this filled worker disks and killed a whole batch of runs.
                # Off by default; final.pt is always written.
                torch.save(gpt_model.state_dict(), out_dir / f"step{step}.pt")

    torch.save(gpt_model.state_dict(), out_dir / "final.pt")
    eval_model("final")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"done; artifacts in {out_dir}")


if __name__ == "__main__":
    main()
