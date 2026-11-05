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
Compares generation with and without speculative decoding on a set of
prompts, and profiles where time is spent.

Example (models downloaded with `litgpt download`, prompts from a Hugging
Face dataset):

```
python -m keys_values.scripts.benchmark_speculative_decoding \
    --target_checkpoint_dir checkpoints/Qwen/Qwen3-8B \
    --draft_checkpoint_dir checkpoints/Qwen/Qwen3-0.6B \
    --hf_dataset openai/openai_humaneval --hf_split test --hf_field prompt \
    --num_prompts 20 --max_new_tokens 512 --num_repeats 3 \
    --cache_name dense-default --cache_length 4096 \
    --speculative_k 2 4 6 --temperature 0 --output results.json
```

Prompts can also come from `--prompt_file` (one prompt per line, or JSON
lines with `--prompt_field`). `--max_prompt_tokens` truncates the text of
each prompt before `--prompt_template` and the chat format of the model are
applied. For each prompt, configuration (plain
generation, or speculative decoding for each value of `speculative_k`) and
repeat, we measure the time until the first generated token (prompt
processing of all models involved, plus sampling) and the decoding time
after it. Each configuration is warmed up once before it is measured.
Results are medians over repeats.

With `--profile`, one generation per configuration is run under
`torch.profiler`. We print the time spent in the phases of speculative
decoding (draft, verify, accept, keep), in the forward calls of each
model, and the most expensive operators, and export a trace which can be
opened in Perfetto or `chrome://tracing`. With `--sync_debug` (CUDA only),
PyTorch warns about every device-to-host synchronization in one generation
per configuration.

With `temperature=0`, we also report the first position where the output
differs from plain generation. For `dense` caches, outputs agree up to
floating point effects. For caches with eviction, they can differ once
tokens are evicted, see :mod:`keys_values.generate.speculative_decoding`.
This needs LitGPT 0.5.13 or later: before, plain generation samples for
`temperature=0` (see :func:`keys_values.generate.speculative_decoding.is_greedy`).

"""

import argparse
import json
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from lightning.fabric.connector import _convert_precision_to_unified_args
from torch.profiler import ProfilerActivity, profile, record_function

from litgpt.prompts import PromptStyle, has_prompt_style, load_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    check_file_size_on_cpu_and_warn,
    check_valid_checkpoint_dir,
    extend_checkpoint_dir,
)

from keys_values.config import Config
from keys_values.data.constants import LIT_MODEL_FNAME
from keys_values.distributed.fabric import Fabric
from keys_values.finetune.utils import init_module, load_checkpoint
from keys_values.generate.base import generate_fn
from keys_values.generate.speculative_decoding import (
    SpeculativeDecodingStats,
    speculative_generate_fn,
)
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT
from keys_values.utils import fabric_precision_to_dtype, seed_everything


def _normalize_inference_precision(precision: str) -> str:
    """Return an explicit, supported model dtype, never implicit mixed AMP."""
    precision = _convert_precision_to_unified_args(precision)
    if precision in ("16-mixed", "bf16-mixed"):
        raise ValueError(
            f"Mixed precision {precision!r} requires autocast, which this "
            "benchmark does not use. Choose 16-true, bf16-true or 32-true."
        )
    # Also validates unsupported modes (for example, 64-true).
    fabric_precision_to_dtype(precision)
    return precision


def _load_inference_model(
    checkpoint_dir: Path,
    cache_factory_kwargs: Dict[str, Any],
    max_returned_tokens: int,
    prompt_chunksize: int,
    precision: str,
    device: torch.device,
) -> LongContextInferenceModel:
    """Load an inference model and its KV caches with the requested dtype."""
    precision = _normalize_inference_precision(precision)
    dtype = fabric_precision_to_dtype(precision)
    checkpoint_dir = extend_checkpoint_dir(Path(checkpoint_dir))
    check_valid_checkpoint_dir(checkpoint_dir, ignore_tokenizer_files=True)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    checkpoint_path = checkpoint_dir / LIT_MODEL_FNAME
    check_file_size_on_cpu_and_warn(checkpoint_path, device)
    print(
        f"Loading model {str(checkpoint_path)!r} with {config.__dict__}",
        file=sys.stderr,
    )
    with init_module(empty_init=True, precision=precision, device=device):
        gpt_model = GPT(config)

    # The factory may mutate the nested cache_kwargs; do not pass the caller's
    # own dictionary (including its nested dictionary) directly to it.
    factory_kwargs = dict(cache_factory_kwargs)
    if factory_kwargs.get("cache_kwargs") is not None:
        factory_kwargs["cache_kwargs"] = dict(factory_kwargs["cache_kwargs"])
    if "quantized" in factory_kwargs["name"] and "dtype" not in factory_kwargs:
        factory_kwargs["dtype"] = dtype
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            **factory_kwargs,
            gpt_model=gpt_model,
            max_batch_size=1,
        )
    )
    # This setter can create RoPE/position-encoding buffers; initialize them
    # on the same device and at the same precision as the model parameters.
    with init_module(precision=precision, device=device):
        gpt_model.max_seq_length = max_returned_tokens
    gpt_model.eval()
    model = LongContextInferenceModel(
        gpt_model=gpt_model,
        head_model=None,
        chunk_size=prompt_chunksize,
    )
    t0 = time.perf_counter()
    load_checkpoint(gpt_model, checkpoint_path, strict=True)
    print(
        f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.",
        file=sys.stderr,
    )
    return model


DEFAULT_PROMPTS = [
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a detailed summary of the causes of the French Revolution.",
    "List ten tips for writing clean, maintainable code and explain each one.",
    "What is speculative decoding in large language models, and why does it "
    "speed up inference?",
]

PHASE_PREFIX = "speculative_decoding::"


def _synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _first_divergence(a: List[int], b: List[int]) -> int:
    for pos, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return pos
    return min(len(a), len(b)) if len(a) != len(b) else -1


def load_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompt_file is not None:
        path = Path(args.prompt_file)
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if path.suffix == ".jsonl":
            prompts = [json.loads(line)[args.prompt_field] for line in lines]
        else:
            prompts = [line.strip() for line in lines]
    elif args.hf_dataset is not None:
        from datasets import load_dataset

        dataset = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split)
        prompts = [row[args.hf_field] for row in dataset]
    else:
        print(
            "No prompts given, using a few built-in prompts. For a meaningful "
            "profile, use --hf_dataset or --prompt_file"
        )
        prompts = DEFAULT_PROMPTS
    if args.num_prompts is not None:
        prompts = prompts[: args.num_prompts]
    return prompts


def build_prompt_text(
    text: str,
    args: argparse.Namespace,
    tokenizer: Tokenizer,
    prompt_style: PromptStyle,
) -> str:
    """
    Truncates `text` to `args.max_prompt_tokens` tokens (if given), then
    applies `args.prompt_template` and the prompt style of the model.
    Truncation happens first, so that the end of the template and the chat
    format are kept. It is approximate, since decoding and encoding again
    need not give the same number of tokens.

    """
    if args.max_prompt_tokens is not None:
        ids = tokenizer.encode(text)
        if ids.numel() > args.max_prompt_tokens:
            text = tokenizer.decode(ids[: args.max_prompt_tokens])
    # Allow "\n" in the template from the command line
    template = args.prompt_template.replace("\\n", "\n")
    text = template.format(text=text)
    return text if args.no_prompt_style else prompt_style.apply(text)


@contextmanager
def label_forward_calls(models: Dict[str, torch.nn.Module]):
    """
    Annotates forward calls of each model with `record_function`, so that
    the profiler attributes time to the target and draft model.

    """
    handles = []
    for name, model in models.items():
        state = {}

        def pre_hook(module, inputs, _name=name, _state=state):
            _state["rf"] = record_function(f"model_forward::{_name}")
            _state["rf"].__enter__()

        def post_hook(module, inputs, output, _state=state):
            _state.pop("rf").__exit__(None, None, None)

        handles.append(model.register_forward_pre_hook(pre_hook))
        handles.append(model.register_forward_hook(post_hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def timed_run(
    token_iter: Iterator[torch.Tensor],
    device: torch.device,
) -> Tuple[List[int], float, float]:
    """
    Consumes a generator which yields generated tokens.

    Returns:
        `(tokens, time_to_first_token, decode_time)` in seconds, where
        `decode_time` is the time after the first token

    """
    _synchronize(device)
    t_start = time.perf_counter()
    t_first = None
    chunks = []
    for token in token_iter:
        # Both generators synchronize before yielding a token, so the time
        # stamp is accurate. Tokens are converted after timing.
        chunks.append(token)
        if t_first is None:
            t_first = time.perf_counter()
    _synchronize(device)
    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end
    tokens = torch.cat(chunks).tolist() if chunks else []
    return tokens, t_first - t_start, t_end - t_first


def run_record(
    tokens: List[int],
    ttft: float,
    decode: float,
    stats: Optional[SpeculativeDecodingStats],
) -> Dict[str, Any]:
    """
    Measurements of a single generation. Rates are computed per run, since
    the number of generated tokens can differ between repeats (sampling with
    a stop token).

    """
    num_decoded = max(len(tokens) - 1, 0)
    record = dict(
        num_tokens=len(tokens),
        ttft_s=ttft,
        decode_s=decode,
        decode_tok_per_s=num_decoded / decode if decode > 0 else 0.0,
    )
    if stats is not None:
        record.update(
            num_steps=stats.num_steps,
            num_draft_tokens=stats.num_draft_tokens,
            num_accepted_tokens=stats.num_accepted_tokens,
            num_generated_tokens=stats.num_generated_tokens,
        )
    return record


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates run records: medians of per-run values, and pooled values
    (totals over all runs), which weight runs by their number of tokens.

    """
    total_decoded = sum(max(r["num_tokens"] - 1, 0) for r in runs)
    total_decode_s = sum(r["decode_s"] for r in runs)
    result = dict(
        num_runs=len(runs),
        decode_tok_per_s=statistics.median(r["decode_tok_per_s"] for r in runs),
        pooled_decode_tok_per_s=(
            total_decoded / total_decode_s if total_decode_s > 0 else 0.0
        ),
        ttft_s=statistics.median(r["ttft_s"] for r in runs),
        median_num_tokens=statistics.median(r["num_tokens"] for r in runs),
    )
    if "num_steps" in runs[0]:
        drafted = sum(r["num_draft_tokens"] for r in runs)
        accepted = sum(r["num_accepted_tokens"] for r in runs)
        steps = sum(r["num_steps"] for r in runs)
        generated = sum(r["num_generated_tokens"] for r in runs)
        result["acceptance_rate"] = accepted / drafted if drafted > 0 else 0.0
        result["tokens_per_target_call"] = generated / steps if steps > 0 else 0.0
    return result


def print_profile(prof: profile, title: str, use_cuda: bool, top: int):
    events = prof.key_averages()
    time_attr = "device_time_total" if use_cuda else "cpu_time_total"
    if use_cuda and not hasattr(events[0], time_attr):
        time_attr = "cuda_time_total"
    rows = []
    for event in events:
        if event.key.startswith(PHASE_PREFIX) or event.key.startswith(
            "model_forward::"
        ):
            rows.append(
                (
                    event.key,
                    event.count,
                    event.cpu_time_total / 1000,
                    getattr(event, time_attr, 0.0) / 1000,
                )
            )
    print(f"\n=== Profile: {title} ===")
    print(f"{'region':48s} {'calls':>6s} {'CPU ms':>10s} {'device ms':>10s}")
    for key, count, cpu_ms, dev_ms in sorted(rows):
        print(f"{key:48s} {count:6d} {cpu_ms:10.1f} {dev_ms:10.1f}")
    sort_keys = (
        ["self_device_time_total", "self_cuda_time_total"]
        if use_cuda
        else ["self_cpu_time_total"]
    )
    for sort_by in sort_keys:
        try:
            print(events.table(sort_by=sort_by, row_limit=top))
            break
        except (AttributeError, KeyError, ValueError):
            continue


def main(args: argparse.Namespace):
    precision = _normalize_inference_precision(args.precision)
    device = Fabric.device()
    target_dir = extend_checkpoint_dir(Path(args.target_checkpoint_dir))
    draft_dir = extend_checkpoint_dir(Path(args.draft_checkpoint_dir))
    tokenizer = Tokenizer(target_dir)
    target_config = Config.from_file(target_dir / "model_config.yaml")
    prompt_style = (
        load_prompt_style(target_dir)
        if has_prompt_style(target_dir)
        else PromptStyle.from_config(target_config)
    )
    prompts = load_prompts(args)
    encoded = [
        tokenizer.encode(
            build_prompt_text(p, args, tokenizer, prompt_style), device=device
        )
        for p in prompts
    ]
    max_returned = max(e.numel() for e in encoded) + args.max_new_tokens
    cache_kwargs = dict(name=args.cache_name, cache_length=args.cache_length)
    draft_cache_kwargs = dict(
        name=args.draft_cache_name or args.cache_name,
        cache_length=args.draft_cache_length or args.cache_length,
    )
    load_kwargs = dict(
        precision=precision,
        device=device,
        max_returned_tokens=max_returned,
        prompt_chunksize=args.prompt_chunksize,
    )
    target = _load_inference_model(
        checkpoint_dir=target_dir, cache_factory_kwargs=cache_kwargs, **load_kwargs
    )
    draft = _load_inference_model(
        checkpoint_dir=draft_dir, cache_factory_kwargs=draft_cache_kwargs, **load_kwargs
    )
    sample_kwargs = dict(
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p
    )
    stop_tokens = () if args.ignore_eos else ([tokenizer.eos_id],)
    use_cuda = device.type == "cuda"

    def make_iter(prompt: torch.Tensor, k: Optional[int], stats=None):
        kwargs = dict(
            prompt=prompt,
            max_returned_tokens=prompt.numel() + args.max_new_tokens,
            stop_tokens=stop_tokens,
            include_prompt=False,
            include_eos=True,
            **sample_kwargs,
        )
        if k is None:
            return generate_fn(model=target, **kwargs)
        return speculative_generate_fn(
            target_model=target,
            draft_model=draft,
            speculative_k=k,
            stats=stats,
            **kwargs,
        )

    configs = [None] + list(args.speculative_k)

    def config_name(k: Optional[int]) -> str:
        return "plain" if k is None else f"speculative_k{k}"

    # Warm-up, once per configuration
    for k in configs:
        timed_run(make_iter(encoded[0], k), device)

    results: List[Dict[str, Any]] = []
    all_runs: Dict[str, List[Dict[str, Any]]] = {config_name(k): [] for k in configs}
    for prompt_idx, prompt in enumerate(encoded):
        row = dict(prompt=prompt_idx, prompt_tokens=prompt.numel(), configs={})
        plain_tokens = None
        for k in configs:
            runs = []
            first_tokens = None
            for repeat in range(args.num_repeats):
                seed_everything(args.seed + repeat)
                stats = SpeculativeDecodingStats() if k is not None else None
                tokens, ttft, decode = timed_run(make_iter(prompt, k, stats), device)
                if first_tokens is None:
                    first_tokens = tokens
                runs.append(run_record(tokens, ttft, decode, stats))
            all_runs[config_name(k)].extend(runs)
            entry = summarize_runs(runs)
            entry["runs"] = runs
            if k is None:
                plain_tokens = first_tokens
            elif args.temperature == 0:
                # Greedy decoding is deterministic, so the first run suffices
                entry["first_divergence"] = _first_divergence(
                    plain_tokens, first_tokens
                )
            row["configs"][config_name(k)] = entry
        plain = row["configs"]["plain"]
        msg = (
            f"prompt {prompt_idx} ({prompt.numel()} tokens): plain "
            f"{plain['decode_tok_per_s']:.1f} tok/s"
        )
        for k in args.speculative_k:
            e = row["configs"][config_name(k)]
            speedup = e["decode_tok_per_s"] / max(plain["decode_tok_per_s"], 1e-9)
            e["decode_speedup"] = speedup
            msg += (
                f" | k={k}: {e['decode_tok_per_s']:.1f} tok/s ({speedup:.2f}x), "
                f"acc {e['acceptance_rate']:.2f}, "
                f"{e['tokens_per_target_call']:.2f} tok/call"
            )
            if "first_divergence" in e:
                msg += f", div {e['first_divergence']}"
        print(msg)
        results.append(row)

    # Summary: pooled over all prompts and repeats
    summary = {name: summarize_runs(runs) for name, runs in all_runs.items()}
    plain_rate = summary["plain"]["pooled_decode_tok_per_s"]
    for k in args.speculative_k:
        entry = summary[config_name(k)]
        entry["pooled_decode_speedup"] = entry["pooled_decode_tok_per_s"] / max(
            plain_rate, 1e-9
        )
    print("\n=== Summary over all prompts ===")
    for name, s in summary.items():
        print(name, {key: round(value, 3) for key, value in s.items()})

    if args.profile:
        profile_dir = Path(args.profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        activities = [ProfilerActivity.CPU] + (
            [ProfilerActivity.CUDA] if use_cuda else []
        )
        models = {"target": target.gpt_model, "draft": draft.gpt_model}
        for k in configs:
            seed_everything(args.seed)
            with label_forward_calls(models):
                with profile(
                    activities=activities, with_stack=args.profile_with_stack
                ) as prof:
                    timed_run(make_iter(encoded[0], k), device)
            name = config_name(k)
            print_profile(prof, name, use_cuda, args.profile_top)
            trace_path = profile_dir / f"trace_{name}.json"
            prof.export_chrome_trace(str(trace_path))
            print(f"Trace written to {trace_path}")

    if args.sync_debug:
        if not use_cuda:
            print("--sync_debug requires CUDA, skipping")
        else:
            for k in configs:
                print(f"\n=== Synchronizations: {config_name(k)} ===")
                torch.cuda.set_sync_debug_mode("warn")
                try:
                    timed_run(make_iter(encoded[0], k), device)
                finally:
                    torch.cuda.set_sync_debug_mode("default")

    if args.output is not None:
        output = dict(
            target=str(target_dir),
            draft=str(draft_dir),
            cache=cache_kwargs,
            draft_cache=draft_cache_kwargs,
            sample_kwargs=sample_kwargs,
            max_new_tokens=args.max_new_tokens,
            num_repeats=args.num_repeats,
            device=torch.cuda.get_device_name() if use_cuda else str(device),
            torch_version=torch.__version__,
            summary=summary,
            results=results,
        )
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"Results written to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--target_checkpoint_dir", type=str, required=True)
    parser.add_argument("--draft_checkpoint_dir", type=str, required=True)
    parser.add_argument("--cache_name", type=str, default="dense-default")
    parser.add_argument("--cache_length", type=int, default=4096)
    parser.add_argument("--draft_cache_name", type=str, default=None)
    parser.add_argument("--draft_cache_length", type=int, default=None)
    parser.add_argument("--speculative_k", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--prompt_chunksize", type=int, default=16)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--prompt_field", type=str, default="prompt")
    parser.add_argument("--hf_dataset", type=str, default=None)
    parser.add_argument("--hf_config", type=str, default=None)
    parser.add_argument("--hf_split", type=str, default="test")
    parser.add_argument("--hf_field", type=str, default="prompt")
    parser.add_argument("--prompt_template", type=str, default="{text}")
    parser.add_argument("--no_prompt_style", action="store_true")
    parser.add_argument("--num_prompts", type=int, default=None)
    parser.add_argument("--max_prompt_tokens", type=int, default=None)
    parser.add_argument("--num_repeats", type=int, default=3)
    parser.add_argument("--precision", type=str, default="bf16-true")
    parser.add_argument("--ignore_eos", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile_dir", type=str, default="spec_profiles")
    parser.add_argument("--profile_with_stack", action="store_true")
    parser.add_argument("--profile_top", type=int, default=25)
    parser.add_argument("--sync_debug", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
