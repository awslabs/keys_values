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
Decode throughput vs. context length: dense cache vs. bounded evicting cache.

The question this answers: in RL rollouts, does per-token generation cost
grow with context (dense: every token reads the whole KV cache of every
layer) and does a bounded cache keep it flat? And what does the evicting
cache pay per token for its bookkeeping (scores, ranking, slot moves)?

Per (context length, cache config) the script

  1. builds a random prompt of `L` tokens, prefills it in chunks,
  2. decodes `--new-tokens` tokens with a batch of `--batch` sequences,
  3. reports prefill time, decode wall time, decode tokens/s
     (batch * new_tokens / seconds), ms per decode step, and peak memory.

Cache configs are given as `name[@K][:key=val;key=val]` (semicolon-separated
kwargs, since the config list itself is comma-separated), e.g.

    dense-default
    h2o-torch-quantized8@8192
    h2o-torch-quantized8@8192:evict_every=64
    h2o-default@8192:grace_period=512;evict_every=64
    h2o-default@8192:attn=flex          (per-config attention backend)

The attention backend is `--attn` (default) or per config via `attn=`:
  eager      eager SDPA (what runs when FlashInfer is not built)
  flashinfer vendored FlashInfer decode kernel + Triton score sum (needs
             `python build_ext.py`)
  flex       FlexAttention baseline with kv_lens padding, 2x call to return
             LSE for H2O scores

With `--profile`, a torch.profiler trace of the decode loop is taken for each
config and the top CUDA kernels by self time are printed, which tells us
*where* the per-token time goes (attention, sort, scatter, dequant, ...).

Example (one L40S):

    python examples/decode_bench.py --model Qwen/Qwen2.5-7B-Instruct \\
        --context-lengths 4096,8192,16384,32768 --new-tokens 256 --batch 8 \\
        --caches dense-default,h2o-torch-quantized8@8192,h2o-torch-quantized8@8192:evict_every=64
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    auto_download_checkpoint,
    check_valid_checkpoint_dir,
)

from keys_values.finetune.utils import load_checkpoint
from keys_values.rl.runtime import SingleDevice
from keys_values.config import Config
from keys_values.data.constants import LIT_MODEL_FNAME
from keys_values.kvcache.factory import (
    KVCacheFactory,
    deallocate_kv_cache_buffers_of_model,
)
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT
from keys_values.attention.flex_attention import (
    FlexAttentionArgs,
    choose_kv_lens,
    choose_q_lens,
)
from keys_values.rl.grpo.rollout import generate_completions
from keys_values.utils import VerbosityLevels


def parse_cache_spec(spec: str):
    """`name[@K][:k=v;k=v]` -> (name, K or None, kwargs)."""
    kwargs = {}
    if ":" in spec:
        spec, kv = spec.split(":", 1)
        for item in kv.split(";"):
            k, v = item.split("=")
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    v = {"true": True, "false": False}.get(v.lower(), v)
            kwargs[k] = v
    budget = None
    if "@" in spec:
        spec, b = spec.split("@", 1)
        budget = int(b)
    return spec, budget, kwargs


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--context-lengths", default="4096,8192,16384,32768")
    p.add_argument(
        "--caches",
        default="dense-default,h2o-torch-quantized8@8192,"
        "h2o-torch-quantized8@8192:evict_every=64",
    )
    p.add_argument("--new-tokens", type=int, default=256)
    p.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Sequences decoded in parallel (= GRPO group size).",
    )
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument(
        "--warmup-tokens",
        type=int,
        default=16,
        help="Decode steps excluded from timing (kernel warmup).",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="torch.profiler on the decode loop; print top kernels.",
    )
    p.add_argument("--profile-top", type=int, default=15)
    p.add_argument(
        "--attn",
        default="flashinfer",
        choices=["eager", "flashinfer", "flex"],
        help="Attention backend (overridable per config with attn=).",
    )
    p.add_argument("--out", default="runs/decode_bench/results.json")
    p.add_argument("--disable-flashinfer", action="store_true")
    p.add_argument("--access-token", default=None)
    args = p.parse_args()

    from keys_values.attention import flashinfer_ops

    if args.disable_flashinfer:
        args.attn = "eager"
    flashinfer_built = flashinfer_ops._available
    if not flashinfer_built:
        print(
            "NOTE: FlashInfer extension not built; 'flashinfer' configs run eager",
            flush=True,
        )
    torch._dynamo.config.cache_size_limit = 32
    torch._dynamo.config.accumulated_cache_size_limit = 128

    torch.manual_seed(0)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    fabric = SingleDevice.create(args.device)
    device = fabric.device
    checkpoint_dir = auto_download_checkpoint(
        model_name=args.model, access_token=args.access_token
    )
    check_valid_checkpoint_dir(checkpoint_dir)
    tokenizer = Tokenizer(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    with fabric.init_module(empty_init=True):
        gpt_model = GPT(config)
    load_checkpoint(gpt_model, checkpoint_dir / LIT_MODEL_FNAME)
    gpt_model.to(device)
    gpt_model.eval()
    vocab = tokenizer.vocab_size
    eos_id = int(tokenizer.eos_id) if tokenizer.eos_id is not None else None

    ctx_lens = [int(x) for x in args.context_lengths.split(",")]
    specs = [parse_cache_spec(s) for s in args.caches.split(",")]
    results = []
    total_new = args.warmup_tokens + args.new_tokens
    print(
        f"{'context':>8} {'cache':<52} {'prefill s':>10} {'decode s':>9} "
        f"{'tok/s':>8} {'ms/step':>8} {'peak GB':>8}",
        flush=True,
    )
    for L_ctx in ctx_lens:
        # Random prompt: content does not matter for timing; avoid EOS so no
        # row stops early (all rows must decode the full budget).
        prompt = torch.randint(0, vocab, (args.batch, L_ctx), device=device)
        if eos_id is not None:
            prompt[prompt == eos_id] = (eos_id + 1) % vocab
        for name, budget, kwargs in specs:
            label = (
                name
                + (f"@{budget}" if budget else "")
                + (
                    ":" + ";".join(f"{k}={v}" for k, v in kwargs.items())
                    if kwargs
                    else ""
                )
            )
            seq_total = L_ctx + total_new
            if name.startswith("dense"):
                cache_length = seq_total
            else:
                cache_length = min(budget or seq_total, seq_total)
            cache_kwargs = dict(kwargs)
            attn = cache_kwargs.pop("attn", args.attn)
            label = f"{label} [{attn}]"
            needs_weights = name.startswith(("h2o", "qh2o"))
            if (
                name.startswith(("h2o", "qh2o"))
                and "orig" not in name
                and "grace_period" not in cache_kwargs
            ):
                cache_kwargs["grace_period"] = cache_length // 16
            # The caches build their own MultiHeadSelfAttention from
            # cache_kwargs, so the backend selection has to go in here. Select
            # via the MHA kwarg: toggling flashinfer_ops._available poisons the
            # lazily-cached FlashInferSDPA singleton for every later config.
            if attn != "flashinfer":
                cache_kwargs["use_flashinfer"] = False
            elif not flashinfer_built:
                label = label.replace("[flashinfer]", "[eager: not built]")
            if attn == "flex":
                # kv_lens padding (upstream #160) bounds the decode compiles
                # to one graph per bucket instead of one per kv_len.
                cache_kwargs["flexatt_args"] = FlexAttentionArgs(
                    extend_kv=False,
                    q_lens=choose_q_lens(chunk_size=args.chunk_size, num_q_lens=4),
                    kv_lens=choose_kv_lens(cache_length=cache_length, num_kv_lens=4),
                    forward_return_lse=needs_weights,
                )
            elif attn == "eager":
                cache_kwargs["use_eager_sdpa_always"] = needs_weights
            deallocate_kv_cache_buffers_of_model(gpt_model)
            try:
                gpt_model.assign_kv_caches(
                    KVCacheFactory.create(
                        gpt_model=gpt_model,
                        name=name,
                        max_batch_size=args.batch,
                        cache_length=cache_length,
                        dtype=dtype,
                        cache_kwargs=cache_kwargs,
                    )
                )
                gpt_model.max_seq_length = seq_total
                inf = LongContextInferenceModel(
                    gpt_model,
                    head_model=None,
                    chunk_size=args.chunk_size,
                    verbose=VerbosityLevels.NONE,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)

                # --- timing: prefill (chunked) vs decode -------------------
                # generate_completions does prefill + decode in one call; we
                # time a warmup-length call to isolate prefill, then the full
                # call, and attribute the difference to decode. The prefill
                # is deterministic (same prompt) so this is exact up to noise.
                sync(device)
                t0 = time.perf_counter()
                generate_completions(
                    model=inf,
                    prompt_ids=prompt,
                    max_new_tokens=args.warmup_tokens,
                    temperature=1.0,
                    eos_token_id=None,
                    pad_token_id=0,
                )
                sync(device)
                t_short = time.perf_counter() - t0
                prof = None
                if args.profile:
                    from torch.profiler import ProfilerActivity, profile

                    acts = [ProfilerActivity.CPU]
                    if device.type == "cuda":
                        acts.append(ProfilerActivity.CUDA)
                    prof = profile(activities=acts, record_shapes=False)
                    prof.__enter__()
                sync(device)
                t0 = time.perf_counter()
                comp = generate_completions(
                    model=inf,
                    prompt_ids=prompt,
                    max_new_tokens=total_new,
                    temperature=1.0,
                    eos_token_id=None,
                    pad_token_id=0,
                )
                sync(device)
                t_long = time.perf_counter() - t0
                if prof is not None:
                    prof.__exit__(None, None, None)
                n_gen = int(comp.shape[1])
                assert n_gen == total_new, (n_gen, total_new)
                # decode time for the extra `new_tokens` steps
                t_decode = t_long - t_short
                t_prefill = t_short - (t_decode / args.new_tokens) * args.warmup_tokens
                tok_s = args.batch * args.new_tokens / max(t_decode, 1e-9)
                ms_step = 1000.0 * t_decode / args.new_tokens
                peak = (
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda"
                    else float("nan")
                )
                row = dict(
                    context=L_ctx,
                    cache=label,
                    cache_length=cache_length,
                    batch=args.batch,
                    new_tokens=args.new_tokens,
                    prefill_s=t_prefill,
                    decode_s=t_decode,
                    tok_s=tok_s,
                    ms_per_step=ms_step,
                    peak_gb=peak,
                )
                print(
                    f"{L_ctx:>8} {label:<52} {t_prefill:>10.2f} {t_decode:>9.2f} "
                    f"{tok_s:>8.1f} {ms_step:>8.1f} {peak:>8.2f}",
                    flush=True,
                )
                if prof is not None:
                    # torch >= 2.x renamed self_cuda_time_total -> self_device_time_total
                    key = "self_cpu_time_total"
                    if device.type == "cuda":
                        key = next(
                            k
                            for k in ("self_device_time_total", "self_cuda_time_total")
                            if hasattr(next(iter(prof.key_averages())), k)
                        )
                    print(
                        prof.key_averages().table(
                            sort_by=key, row_limit=args.profile_top
                        ),
                        flush=True,
                    )
                    top = []
                    for ev in sorted(
                        prof.key_averages(), key=lambda e: -getattr(e, key)
                    ):
                        top.append(
                            dict(
                                name=ev.key,
                                us=float(getattr(ev, key)),
                                count=int(ev.count),
                            )
                        )
                        if len(top) >= args.profile_top:
                            break
                    row["profile_top"] = top
                results.append(row)
            except (
                torch.cuda.OutOfMemoryError if device.type == "cuda" else MemoryError
            ):
                print(f"{L_ctx:>8} {label:<52} {'OOM':>10}", flush=True)
                results.append(dict(context=L_ctx, cache=label, oom=True))
            # Release everything from this config before the next one. The
            # OOM traceback keeps the generation frames (and their tensors)
            # alive until the except block is left, so collect afterwards; a
            # later config inheriting that memory would OOM spuriously.
            comp = prof = inf = None
            deallocate_kv_cache_buffers_of_model(gpt_model)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(args=vars(args), results=results), f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
