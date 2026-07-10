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
Task 2.3 experiment: run a model under the keys_values ``lastrec`` policy in
vLLM, and check parity against vLLM's native sliding window.

How it works (and why):
- vLLM builds each layer's ``KVCacheSpec`` in the worker via
  ``GPUModelRunner.get_kv_cache_spec`` (which calls each ``Attention``
  layer's ``get_kv_cache_spec``). We wrap that method to convert the model's
  full-attention specs into ``LastRecSpec`` (or ``SlidingWindowSpec``), so the
  block manager recycles old blocks.
- IMPORTANT: the block manager and the attention *masking* must agree. The
  backend masks based on each layer's ``sliding_window`` attribute, NOT the
  KVCacheSpec. So we also set ``attn_module.sliding_window`` on every layer.
  If only the spec is changed, the backend would attend over blocks the
  manager has already freed. This coupling is the crux of task 2.3 and the
  most likely thing to need per-version adjustment.
- We force the engine in-process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) so this
  process's monkeypatches actually apply to the worker.

This is an experiment harness meant to be iterated on a GPU box, not a final
API. See docs/vllm_integration.md and the spec under .kiro/ (local).

Usage:
    python examples/vllm_lastrec_experiment.py --mode both --window 256
"""

from __future__ import annotations

import argparse
import os

# Must be set before importing vllm so the engine runs in-process and our
# monkeypatches take effect in the same interpreter as the worker.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
# Belt-and-suspenders for the CUDA-fork issue if multiprocessing is re-enabled.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--window", type=int, default=256, help="cache_length / window")
    p.add_argument(
        "--mode",
        choices=["lastrec", "sliding", "full", "both"],
        default="both",
        help="Policy to run. 'full' is the no-window baseline; 'both' prints "
        "guidance to compare lastrec vs sliding in separate processes.",
    )
    p.add_argument(
        "--needle",
        action="store_true",
        help="Use a recall prompt: a secret code early, then filler past the "
        "window, then a question. A windowed policy cannot recall it; the "
        "'full' baseline can. Makes windowing observable.",
    )
    p.add_argument(
        "--prompt",
        default=(
            "Summarize the following in one sentence. "
            + ("The quick brown fox jumps over the lazy dog. " * 200)
        ),
        help="Long-ish prompt so the window (< prompt length) actually evicts.",
    )
    p.add_argument("--max-tokens", type=int, default=48)
    return p.parse_args()


_SECRET_CODE = "42819"


def _build_prompt(args) -> str:
    if not args.needle:
        return args.prompt
    # Secret code up front, then filler that pushes it out of a small window,
    # then the question at the end (inside the window).
    filler = "The quick brown fox jumps over the lazy dog. " * 400
    return (
        f"Remember this: the secret code is {_SECRET_CODE}. "
        f"{filler}"
        "What is the secret code? Answer with only the number. Answer:"
    )


def _check_env() -> None:
    try:
        import torch
    except ImportError:
        sys.exit("torch not installed.")
    if not torch.cuda.is_available():
        sys.exit("No CUDA device. Run this on a GPU box (see docstring).")
    try:
        import vllm  # noqa: F401
    except ImportError:
        sys.exit("vllm not installed.")


def _install_spec_override(window: int, use_lastrec: bool) -> None:
    """Wrap GPUModelRunner.get_kv_cache_spec to emit our windowed spec.

    Converts each FullAttentionSpec the model would produce into either
    LastRecSpec (our policy) or SlidingWindowSpec (vLLM native, for parity),
    preserving block_size / num_kv_heads / head_size / dtype.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    from keys_values.vllm.specs import LastRecSpec

    if getattr(GPUModelRunner.get_kv_cache_spec, "_kv_patched", False):
        return
    original = GPUModelRunner.get_kv_cache_spec

    def patched(self):
        specs = original(self)
        converted = {}
        for name, spec in specs.items():
            win = getattr(spec, "sliding_window", None)
            is_windowable = isinstance(
                spec, (FullAttentionSpec, SlidingWindowSpec)
            ) and not isinstance(spec, LastRecSpec)
            if use_lastrec and is_windowable:
                converted[name] = LastRecSpec(
                    block_size=spec.block_size,
                    num_kv_heads=spec.num_kv_heads,
                    head_size=spec.head_size,
                    dtype=spec.dtype,
                    sliding_window=win if isinstance(win, int) else window,
                )
                print(
                    f"[spec] {name}: {type(spec).__name__} -> "
                    f"LastRecSpec(window={converted[name].sliding_window})"
                )
            else:
                converted[name] = spec
        return converted

    patched._kv_patched = True
    GPUModelRunner.get_kv_cache_spec = patched


def _install_attention_window(window: int) -> None:
    """Set each Attention layer's sliding_window so masking matches the spec.

    NOTE: the attribute/format vLLM expects for sliding_window can vary by
    version (int vs (left, right) tuple). This is the most likely spot to
    adjust during box iteration; we log every layer we touch.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner.load_model, "_kv_window_patched", False):
        return
    original_load = GPUModelRunner.load_model

    def _get_attention_layers(vllm_config):
        # get_layers_from_vllm_config and the Attention class have moved
        # between versions; try the known locations.
        get_layers = None
        for modpath in ("vllm.config", "vllm.model_executor.models.utils"):
            try:
                mod = __import__(modpath, fromlist=["get_layers_from_vllm_config"])
                get_layers = mod.get_layers_from_vllm_config
                break
            except (ImportError, AttributeError):
                continue
        if get_layers is None:
            raise ImportError("could not locate get_layers_from_vllm_config")

        attention_cls = None
        for modpath in ("vllm.model_executor.layers.attention", "vllm.attention"):
            try:
                attention_cls = __import__(modpath, fromlist=["Attention"]).Attention
                break
            except (ImportError, AttributeError):
                continue
        if attention_cls is None:
            raise ImportError("could not locate the Attention layer class")

        return get_layers(vllm_config, attention_cls)

    def patched_load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        try:
            layers = _get_attention_layers(self.vllm_config)
            for name, attn in layers.items():
                if getattr(attn, "sliding_window", None) in (None, (-1, -1)):
                    attn.sliding_window = window
                    print(f"[mask] {name}: sliding_window <- {window}")
        except Exception as exc:  # noqa: BLE001 - experiment: surface it
            print(f"[mask][warn] could not set sliding_window: {exc}")
        return result

    patched_load._kv_window_patched = True
    GPUModelRunner.load_model = patched_load


def _generate(args, mode: str) -> str:
    from keys_values.vllm.registration import register_policies

    if mode in ("lastrec", "sliding"):
        register_policies()
        _install_spec_override(args.window, use_lastrec=(mode == "lastrec"))
        _install_attention_window(args.window)
    # mode == "full": no overrides -> stock full-attention baseline.

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        enable_prefix_caching=False,  # windowed eviction + prefix cache don't mix
    )
    out = llm.generate(
        [_build_prompt(args)],
        SamplingParams(max_tokens=args.max_tokens, temperature=0.0),
    )
    return out[0].outputs[0].text


def main() -> None:
    args = parse_args()
    _check_env()

    if args.mode == "both":
        print(
            "For a clean comparison, run each mode in a separate process "
            "(monkeypatch state does not reset within one process):\n"
            f"  python examples/vllm_lastrec_experiment.py --mode lastrec "
            f"--window {args.window}"
            f"{' --needle' if args.needle else ''}\n"
            f"  python examples/vllm_lastrec_experiment.py --mode sliding "
            f"--window {args.window}"
            f"{' --needle' if args.needle else ''}\n"
            f"  python examples/vllm_lastrec_experiment.py --mode full"
            f"{' --needle' if args.needle else ''}\n"
            "Parity: lastrec vs sliding outputs should match (Property 5).\n"
            "Effect: with --needle, full recalls the code; lastrec/sliding "
            "should not."
        )
        text = _generate(args, mode="lastrec")
        print(f"\n[lastrec] Output: {text!r}")
        return

    text = _generate(args, mode=args.mode)
    print(f"\n[{args.mode}] Output: {text!r}")


if __name__ == "__main__":
    main()
