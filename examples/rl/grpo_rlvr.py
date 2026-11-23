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
RLVR on competition math through a bounded, evicting KV cache.

Follows the benchmark protocol of Sparrow (Sadhukhan et al., 2026, arXiv
2606.08446) so results line up with the sparse-rollout literature:

  train : DeepScaleR-Preview (40k problems) or Polaris-53K, binary reward =
          final boxed answer matches the reference
  eval  : MATH500, AIME 2024, AIME 2025, AMC 2023 at avg@k (sampled, Qwen3
          recommended temperature 0.6 / top-p 0.95), reported per set

The long axis here is the *generation* (thinking-model CoT), not the prompt:
prompts are a few hundred tokens and the completion is evicted from the cache
once it exceeds ``--cache-length``. Both rollout and the chunked gradient run
through the same cache, so there is no actor/policy mismatch to correct.

Reward uses ``math_verify`` when installed (symbolic equivalence; the
standard RLVR verifier) and falls back to normalised string match otherwise.

Typical probe (base-model accuracy and timing, no training):

    python examples/rl/grpo_rlvr.py --model Qwen/Qwen3-1.7B --eval-only \\
        --eval-sets math500 --n-eval 50 --eval-samples 1 \\
        --max-new-tokens 8192 --cache-length 4096

Training run:

    python examples/rl/grpo_rlvr.py --model Qwen/Qwen3-1.7B \\
        --train-dataset deepscaler --steps 200 --group-size 8 \\
        --max-new-tokens 8192 --cache-length 4096 --optimizer paged_adamw8bit
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
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

# Standard RLVR instruction (DeepSeek-R1 / DeepScaleR / Polaris all use a
# variant of this sentence).
INSTRUCTION = (
    "\n\nPlease reason step by step, and put your final answer " "within \\boxed{}."
)

# (HF repo, config, split, problem column, answer column)
TRAIN_SETS = {
    "deepscaler": (
        "agentica-org/DeepScaleR-Preview-Dataset",
        None,
        "train",
        "problem",
        "answer",
    ),
    "polaris": (
        "POLARIS-Project/Polaris-Dataset-53K",
        None,
        "train",
        "problem",
        "answer",
    ),  # has "difficulty" = k/8 solved by the ref model
}
EVAL_SETS = {
    "math500": ("HuggingFaceH4/MATH-500", None, "test", "problem", "answer"),
    "aime24": ("HuggingFaceH4/aime_2024", None, "train", "problem", "answer"),
    "aime25": ("yentinglin/aime_2025", "default", "train", "problem", "answer"),
    "amc23": ("math-ai/amc23", None, "test", "question", "answer"),
}


def load_problems(
    key: str,
    table: dict,
    limit: int = 0,
    seed: int = 42,
    difficulty_max: int = 0,
    difficulty_min: int = 0,
) -> list[dict]:
    repo, cfg, split, q_col, a_col = table[key]
    ds = (
        load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
    )
    recs = [
        {
            "problem": r[q_col],
            "answer": str(r[a_col]),
            "id": f"{key}:{i}",
            "difficulty": r.get("difficulty"),
        }
        for i, r in enumerate(ds)
    ]
    if difficulty_max:
        # Polaris: "k/8" = reference-model solve rate; keep the hard end.
        def hard(r):
            d = r.get("difficulty")
            if d is None or "/" not in str(d):
                return False
            return difficulty_min <= int(str(d).split("/")[0]) <= difficulty_max

        n0 = len(recs)
        recs = [r for r in recs if hard(r)]
        print(
            f"{key}: difficulty in [{difficulty_min}, {difficulty_max}]/8 "
            f"keeps {len(recs)}/{n0}",
            flush=True,
        )
    if limit and limit < len(recs):
        # Deterministic subset so paired comparisons across runs share
        # problems (same discipline as the HELMET split manifests).
        rng = random.Random(seed)
        rng.shuffle(recs)
        recs = recs[:limit]
    return recs


# ---------------------------------------------------------------- reward ---


def extract_boxed(text: str) -> str | None:
    """Last \\boxed{...} content, brace-balanced."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None


def strip_think(text: str) -> str:
    """Qwen3 thinking models wrap CoT in <think>...</think>; the answer is
    read from the text after the LAST close tag. An unterminated think block
    (budget exhausted mid-thought) yields no answer, which is the intended
    reward of 0."""
    if "<think>" in text and "</think>" not in text:
        return ""
    return text.rsplit("</think>", 1)[-1]


def normalize_answer(ans: str) -> str:
    ans = ans.strip().strip("$")
    ans = re.sub(r"\\left|\\right", "", ans)
    ans = re.sub(r"\\!|\\,|\\;|\\ ", "", ans)
    ans = re.sub(r"^\\text\{(.+)\}$", r"\1", ans)
    ans = re.sub(r"\\dfrac|\\tfrac", r"\\frac", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = ans.rstrip(".")
    return ans


try:  # symbolic verifier (pip install math-verify); optional
    from math_verify import parse as _mv_parse, verify as _mv_verify

    def _symbolic_match(pred: str, gold: str) -> bool:
        try:
            return bool(_mv_verify(_mv_parse(f"${gold}$"), _mv_parse(f"${pred}$")))
        except Exception:
            return False

    HAVE_MATH_VERIFY = True
except Exception:  # pragma: no cover - depends on the environment

    def _symbolic_match(pred: str, gold: str) -> bool:
        return False

    HAVE_MATH_VERIFY = False


def answer_reward(completion: str, gold: str) -> float:
    """1.0 iff the final boxed answer matches the reference."""
    boxed = extract_boxed(strip_think(completion))
    if boxed is None:
        return 0.0
    if normalize_answer(boxed) == normalize_answer(gold):
        return 1.0
    return 1.0 if _symbolic_match(boxed, gold) else 0.0


def is_overlong(
    row: torch.Tensor, pad_id: int, max_new_tokens: int, eos_id: int | None = None
) -> bool:
    """True iff a completion row used the full ``max_new_tokens`` budget
    without stopping: no padding (padding only follows an early stop) and the
    last token is not EOS (a sequence may legitimately emit EOS exactly at
    the cutoff; the stop token is kept in the row)."""
    if int((row != pad_id).sum().item()) != max_new_tokens:
        return False
    return eos_id is None or int(row[-1].item()) != eos_id


def overlong_reward(reward: float, overlong: bool, penalty: float) -> float:
    """Subtract ``penalty`` from the verifier reward of a truncated rollout
    (``penalty > 0`` only). The verifier result is kept, so a truncated
    rollout that somehow verifies scores ``1 - penalty``, an ordinary
    truncated one ``-penalty``: the reward separates "ran out of budget" from
    "finished and wrong", which the binary verifier alone cannot."""
    if overlong and penalty > 0:
        return reward - penalty
    return reward


# ------------------------------------------------------------------ main ---


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument(
        "--train-dataset",
        default="polaris",
        choices=sorted(TRAIN_SETS),
        help="Polaris by default: Qwen3-1.7B solves most of DeepScaleR "
        "outright (MATH500 0.94 @16k), so groups have no reward "
        "spread and GRPO gets no gradient.",
    )
    p.add_argument(
        "--train-limit",
        type=int,
        default=0,
        help="Deterministic subset of the training set (0 = all).",
    )
    p.add_argument(
        "--difficulty-max",
        type=int,
        default=0,
        help="Polaris only: keep problems with difficulty <= N/8 " "(0 = no filter).",
    )
    p.add_argument(
        "--difficulty-min",
        type=int,
        default=0,
        help="Polaris only: and >= N/8. Use 1 to drop the 0/8 problems the "
        "reference model never solved: all-zero groups, no gradient.",
    )
    p.add_argument(
        "--eval-sets",
        default="math500,aime24,aime25,amc23",
        help="Comma-separated subset of " + ",".join(sorted(EVAL_SETS)),
    )
    p.add_argument(
        "--n-eval",
        type=int,
        default=0,
        help="Problems per eval set (0 = whole set; MATH500 is 500).",
    )
    p.add_argument(
        "--eval-samples",
        type=int,
        default=1,
        help="k for avg@k. Sparrow: 16 on AIME, 4 on AMC.",
    )
    p.add_argument("--eval-temperature", type=float, default=0.6)
    p.add_argument("--eval-top-p", type=float, default=0.95)
    p.add_argument(
        "--kv-cache-name",
        default="h2o-default",
        help="bf16 H2O by default: int8 (h2o-torch-quantized8) costs "
        "~1/3 of decode throughput for 1.5 GB at K=8192 (see "
        "decode_bench, 7B/L40S).",
    )
    p.add_argument("--cache-length", type=int, default=4096)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--prompts-per-update", type=int, default=2)
    p.add_argument("--adv-mode", choices=["grpo", "rloo"], default="grpo")
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=16384,
        help="Generation cutoff. At 8k, 80%% of Qwen3-1.7B AIME samples "
        "end inside <think>. Sparrow: 37k thinking, 12k base.",
    )
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument(
        "--optimizer", choices=["adamw", "paged_adamw8bit"], default="paged_adamw8bit"
    )
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument(
        "--attn",
        default="auto",
        choices=ATTN_BACKENDS,
        help="Attention backend. auto: FlashInfer decode if built (else eager "
        "with a loud warning) and FlexAttention in the gradient cells on CUDA "
        "(C-long queries, no K^2/C memory term); flex: FlexAttention "
        "everywhere, decode recompiles per kv_len (experiments only); eager: "
        "eager decode and query-padded PyTorch SDPA in the gradient cells "
        "(the pre-Flex path, A/B baseline).",
    )
    p.add_argument("--backward-tmp-gb", type=float, default=2.0)
    p.add_argument("--lora-r", type=int, default=0)
    p.add_argument("--save-intermediate", action="store_true")
    p.add_argument(
        "--dense-baseline",
        action="store_true",
        help="Dense-RL baseline (one full-sequence backward). Pair "
        "with --kv-cache-name dense-default and a cache "
        "length >= prompt + max-new-tokens.",
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
        default=1.0,
        help="Rollout sampling temperature (RLVR standard: 1.0).",
    )
    p.add_argument(
        "--overlong-penalty",
        type=float,
        default=0.0,
        help="Subtract this from the reward of a rollout that used the full "
        "--max-new-tokens budget without EOS (truncated mid-reasoning), "
        "so 'ran out of budget' scores below 'finished and wrong'. "
        "0 = off (default). overlong_frac is logged regardless.",
    )
    p.add_argument(
        "--per-sequence-norm",
        action="store_true",
        help="Normalize each rollout's loss by its own token count instead "
        "of the group-mean count, so a 25k-token rollout no longer "
        "carries 3x the gradient mass of an 8k one (length drift).",
    )
    p.add_argument(
        "--grad-micro-batch",
        type=int,
        default=None,
        help="Sequences per chunked backward pass (default: whole group in "
        "one pass). Every memory term of the chunked backward scales with "
        "the batch size, so e.g. 1 or 2 divides the gradient-phase peak by "
        "G or G/2 and lets H2O K=16384/32768 fit a 44 GiB GPU for "
        "Qwen3-1.7B. Exact: gradients and loss equal the single-pass ones.",
    )
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/grpo_rlvr")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument(
        "--skip-initial-eval",
        action="store_true",
        help="Skip the step-0 eval (reuse an earlier base eval of the "
        "same model/cache/cutoff; saves 15-30 h at 32k).",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="state_dict (final.pt) to load before evaluating; with "
        "--eval-only this scores a trained policy.",
    )
    p.add_argument(
        "--final-eval-samples",
        type=int,
        default=0,
        help="k for the final avg@k (0 = same as --eval-samples). "
        "Lets training use cheap avg@1 checks and finish with "
        "e.g. avg@16 on AIME.",
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
    print(
        f"math_verify: {'yes' if HAVE_MATH_VERIFY else 'NO (string match only)'}",
        flush=True,
    )

    checkpoint_dir = auto_download_checkpoint(
        model_name=args.model, access_token=args.access_token
    )
    tokenizer = Tokenizer(checkpoint_dir)
    if args.lora_r > 0:
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

    eval_sets = {
        k: load_problems(k, EVAL_SETS, args.n_eval)
        for k in args.eval_sets.split(",")
        if k
    }
    for k, v in eval_sets.items():
        print(f"eval {k}: {len(v)} problems x {args.eval_samples} samples", flush=True)
    train_records = []
    if not args.eval_only:
        train_records = load_problems(
            args.train_dataset,
            TRAIN_SETS,
            args.train_limit,
            difficulty_max=args.difficulty_max,
            difficulty_min=args.difficulty_min,
        )
        random.Random(args.seed).shuffle(train_records)
        print(f"train {args.train_dataset}: {len(train_records)} problems", flush=True)

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
    )
    # Gradient cells only: FlexAttention under auto (CUDA), query-padded SDPA
    # under eager. Reaches TrainingAttnWeightsReplayCache via grpo_step; the
    # decode MHAs above are not touched.
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
    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        gpt_model.load_state_dict(sd, strict=True)
        print(f"loaded policy from {args.checkpoint}", flush=True)
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
            max_batch_size=max(
                args.group_size, args.eval_samples, args.final_eval_samples
            ),
            cache_length=args.cache_length,
            dtype=dtype,
            cache_kwargs=cache_kwargs,
        )
    )

    def encode(rec):
        return tokenizer.encode(
            prompt_style.apply(rec["problem"] + INSTRUCTION), device=fabric.device
        )

    @torch.no_grad()
    def eval_model(tag: str, k: int = 0) -> dict[str, float]:
        """avg@k per eval set; also logs mean completion length and the
        fraction of samples that ran out of budget inside <think>."""
        k = k or args.eval_samples
        gpt_model.eval()
        out = {}
        for name, recs in eval_sets.items():
            t0 = time.perf_counter()
            scores, lengths, truncated = [], [], 0
            for rec in recs:
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
                    prompt_ids=ids.repeat(k, 1),
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.eval_temperature,
                    top_k=None,
                    top_p=args.eval_top_p,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                    no_inference_mode=True,
                )
                for row in comp:
                    row = row[row != pad_id]
                    text = tokenizer.decode(row)
                    lengths.append(int(row.numel()))
                    if "<think>" in text and "</think>" not in text:
                        truncated += 1
                    scores.append(answer_reward(text, rec["answer"]))
            acc = sum(scores) / max(len(scores), 1)
            out[name] = acc
            print(
                f"[eval @ {tag}] {name} avg@{k} = {acc:.3f} "
                f"(n={len(recs)}) | mean len {sum(lengths) / max(len(lengths), 1):.0f} "
                f"| truncated {truncated / max(len(scores), 1):.2f} "
                f"| {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
        return out

    if args.eval_only:
        res = eval_model(
            "checkpoint" if args.checkpoint else "base", k=args.final_eval_samples
        )
        with open(out_dir / "eval_base.json", "w") as f:
            json.dump(res, f, indent=2)
        return

    trainable = [q for q in gpt_model.parameters() if q.requires_grad]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    else:
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=args.lr)

    history = (
        [] if args.skip_initial_eval else [{"step": 0, "eval": eval_model("step 0")}]
    )
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        micro_metrics = []
        step_len: list[int] = []
        step_spread: list[float] = []
        step_overlong: list[float] = []
        step_raw: list[float] = []
        for micro in range(args.prompts_per_update):
            rec = train_records[
                ((step - 1) * args.prompts_per_update + micro) % len(train_records)
            ]
            prompt_ids = encode(rec).unsqueeze(0)

            def reward_fn(p_ids, completion_ids):
                vals = []
                for row in completion_ids:
                    overlong = is_overlong(row, pad_id, args.max_new_tokens, eos_id)
                    step_overlong.append(1.0 if overlong else 0.0)
                    row = row[row != pad_id]
                    step_len.append(int(row.numel()))
                    r = answer_reward(tokenizer.decode(row), rec["answer"])
                    step_raw.append(float(r))
                    vals.append(overlong_reward(r, overlong, args.overlong_penalty))
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
                    per_sequence_norm=args.per_sequence_norm,
                    grad_micro_batch=args.grad_micro_batch,
                    train_cache_kwargs=train_cache_kwargs,
                )
            )
        mean_r = sum(m["mean_reward"] for m in micro_metrics) / len(micro_metrics)
        dt = time.perf_counter() - t0
        entry = {
            "step": step,
            "reward": mean_r,
            "sec": dt,
            "mean_len": sum(step_len) / max(len(step_len), 1),
            "spread": sum(step_spread) / max(len(step_spread), 1),
            "overlong_frac": sum(step_overlong) / max(len(step_overlong), 1),
            # Correctness-only reward (no overlong penalty): comparable
            # across runs with and without --overlong-penalty.
            "reward_raw": sum(step_raw) / max(len(step_raw), 1),
        }
        msg = (
            f" | len {entry['mean_len']:.0f} spread {entry['spread']:.2f}"
            f" overlong_frac {entry['overlong_frac']:.2f}"
            f" raw {entry['reward_raw']:.3f}"
        )
        for key in ("grad_peak_device_mib", "parked_peak_mib", "parked_peak_count"):
            if key in micro_metrics[0]:
                entry[key] = max(m[key] for m in micro_metrics)
        if "grad_peak_device_mib" in entry:
            msg += f" | dev peak {entry['grad_peak_device_mib'] / 1024:.1f}G"
        if "grad_sdpa_mode" in micro_metrics[0]:
            # SDPA mode of the gradient cells (flexattention / qpadded-pytorch)
            entry["grad_sdpa_mode"] = micro_metrics[0]["grad_sdpa_mode"]
            msg += f" | grad {entry['grad_sdpa_mode']}"
        print(f"step {step:4d} | reward {mean_r:.3f} | {dt:.1f}s{msg}", flush=True)
        history.append(entry)
        if step % args.eval_every == 0 and step < args.steps:
            history.append({"step": step, "eval": eval_model(f"step {step}")})
            if args.save_intermediate:
                torch.save(gpt_model.state_dict(), out_dir / f"step{step}.pt")
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    torch.save(gpt_model.state_dict(), out_dir / "final.pt")
    history.append(
        {"step": args.steps, "eval": eval_model("final", k=args.final_eval_samples)}
    )
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"done; artifacts in {out_dir}")


if __name__ == "__main__":
    main()
