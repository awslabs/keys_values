"""
Issue #148: search for a SMALL model configuration on which the pre-fix
autograd hooks raise the annotation-adjacency error.

All three reproductions so far are on a 7B model. A small repro would give a
fast GPU regression test. This sweeps the axes that differ between the
passing `test_gradient_row_of_cells` configs and the crashing GRPO runs:
GQA ratio, chunks per cell, generated-region length, cache quantization,
grace period, eos-ragged loss mask, dtype.

Run with `autograd_hooks.py` reverted to the last pre-fix commit (the job
script does that). Each config runs one forward/backward through
`LongContextGradientModel` with the GRPO loss head and records: OK,
ADJACENCY (the #148 error), or OTHER (any other exception).

    python scripts/rl/x148_small_config_search.py --device cuda --out results.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
import traceback

import torch

from keys_values.config import Config
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.kvcache.gradient.main import LongContextGradientModel
from keys_values.model import GPT
from keys_values.rl.grpo.loss import GRPOLossHeadModel
from keys_values.utils import VerbosityLevels

HEAD_SIZE = 64  # torch-quantized8 needs a "real" head size; matches test_gradient


def run_config(
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_head: int,
    n_query_groups: int,
    cache_length: int,
    chunk_size: int,
    completion_chunks: int,
    quant: str,
    grace: bool,
    ragged: bool,
    batch_size: int = 4,
    n_layer: int = 2,
    seed: int = 0,
) -> str:
    torch.manual_seed(seed)
    torch.set_default_dtype(dtype)
    prompt_len = cache_length + chunk_size // 2  # cache is full before generation
    completion_len = completion_chunks * chunk_size - chunk_size // 4  # misaligned
    seq_len = prompt_len + completion_len
    config = Config(
        block_size=seq_len + 64,
        vocab_size=128,
        padded_vocab_size=128,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_head * HEAD_SIZE,
        n_query_groups=n_query_groups,
        intermediate_size=2 * n_head * HEAD_SIZE,
        rotary_percentage=1,
    )
    with torch.device(device):
        gpt_model = GPT(config)
        gpt_model.apply(gpt_model._init_weights)
    cache_kwargs = {}
    if grace:
        cache_kwargs["grace_period"] = max(cache_length // 16, 1)
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            gpt_model=gpt_model,
            name=f"h2o-{quant}",
            max_batch_size=batch_size,
            cache_length=cache_length,
            dtype=dtype,
            cache_kwargs=cache_kwargs,
        )
    )
    full_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    model_input_ids = full_ids[:, :-1]
    completions = full_ids[:, prompt_len:]
    mask = torch.ones_like(completions, dtype=dtype)
    if ragged:
        # eos-ragged: rows stop at different points inside the generated region
        for b in range(batch_size):
            stop = completion_len * (b + 1) // (batch_size + 1)
            mask[b, stop:] = 0
    advantages = torch.randn(batch_size, device=device, dtype=dtype)
    advantages = advantages - advantages.mean()

    head = GRPOLossHeadModel(config)
    head.set_batch(advantages=advantages, old_logps=None, mask=mask)
    grad_model = LongContextGradientModel(
        gpt_model=gpt_model,
        head_model=head,
        layers_per_cell=1,
        chunk_size=chunk_size,
        verbose=VerbosityLevels.NONE,
    )
    grad_model.train()
    try:
        loss = grad_model(model_input_ids, completions)
        loss.backward()
        return "OK"
    except ValueError as ex:
        msg = str(ex)
        if "final chunk_idx" in msg:
            return (
                "ADJACENCY: "
                + msg.split(":")[0]
                + msg[msg.find("final chunk_idx") - 2 :]
            )
        return "OTHER: " + msg[:120]
    except Exception as ex:  # noqa: BLE001
        return "OTHER: " + type(ex).__name__ + ": " + str(ex)[:100]
    finally:
        del grad_model, gpt_model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="x148_search.csv")
    p.add_argument("--quick", action="store_true", help="tiny grid, smoke test")
    args = p.parse_args()
    device = torch.device(args.device)

    if args.quick:
        grid = dict(
            dtype=[torch.float32],
            heads=[(4, 2)],
            cache_length=[64],
            chunk_size=[16],
            completion_chunks=[3],
            quant=["default"],
            grace=[False],
            ragged=[True],
        )
    else:
        grid = dict(
            dtype=[torch.bfloat16, torch.float32],
            heads=[(4, 4), (8, 4), (14, 2)],  # GQA ratios 1, 2, 7 (7 = Qwen2.5-7B)
            cache_length=[128, 256],
            chunk_size=[16, 32],
            completion_chunks=[3, 6, 12],
            quant=["torch-quantized8", "default"],
            grace=[True, False],
            ragged=[True, False],
        )
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    print(f"{len(combos)} configs on {device}", flush=True)
    n_adj = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dtype",
                "n_head",
                "n_query_groups",
                "cache_length",
                "chunk_size",
                "completion_chunks",
                "quant",
                "grace",
                "ragged",
                "seconds",
                "result",
            ]
        )
        for i, combo in enumerate(combos):
            cfg = dict(zip(keys, combo))
            n_head, n_qg = cfg.pop("heads")
            t0 = time.perf_counter()
            try:
                res = run_config(
                    device=device, n_head=n_head, n_query_groups=n_qg, **cfg
                )
            except Exception:  # noqa: BLE001 - never let one config kill the sweep
                res = "OTHER: " + traceback.format_exc().splitlines()[-1][:120]
            dt = time.perf_counter() - t0
            n_adj += res.startswith("ADJACENCY")
            row = [
                str(cfg["dtype"]).replace("torch.", ""),
                n_head,
                n_qg,
                cfg["cache_length"],
                cfg["chunk_size"],
                cfg["completion_chunks"],
                cfg["quant"],
                cfg["grace"],
                cfg["ragged"],
                f"{dt:.1f}",
                res,
            ]
            w.writerow(row)
            f.flush()
            print(f"[{i + 1}/{len(combos)}] {row[:-2]} -> {res}", flush=True)
    print(f"done: {n_adj} ADJACENCY hits out of {len(combos)}; results in {args.out}")
    sys.exit(0)


if __name__ == "__main__":
    main()
