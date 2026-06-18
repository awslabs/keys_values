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
vLLM integration feasibility spike (vLLM 0.23 / V1 engine).

Goal: stand up a baseline model in vLLM and dump the runtime extension points
the keys_values KV-cache bridge needs (resolved attention backend, per-layer
KV cache spec, KV cache config: num_blocks / block_size / num_kv_heads /
head_size). See docs/vllm_integration.md for the design context.

Run on a Linux + NVIDIA GPU box, in the combined env (vLLM 0.23 + keys_values):

    python examples/vllm_spike.py --model Qwen/Qwen2.5-0.5B-Instruct

V1 internals move between minor releases, so the introspection below is
best-effort and defensive: it prints whatever it can reach and never hard-fails
the run on a missing attribute.
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--prompt", default="The key idea behind selective KV caching is"
    )
    return parser.parse_args()


def check_environment() -> None:
    try:
        import torch
    except ImportError:
        sys.exit("torch is not installed in this environment.")
    if not torch.cuda.is_available():
        sys.exit(
            "No CUDA device available. vLLM needs a Linux + NVIDIA GPU box. "
            "This spike cannot run on macOS / CPU. See docs/vllm_integration.md."
        )
    try:
        import vllm  # noqa: F401
    except ImportError:
        sys.exit("vllm is not installed. `pip install -U vllm` in this env.")


def _first_attr(obj, *names):
    """Return the first attribute in *names* that exists on *obj*, else None."""
    for name in names:
        if obj is not None and hasattr(obj, name):
            return getattr(obj, name)
    return None


def report_vllm_config(llm) -> None:
    """Best-effort dump of the V1 engine's model + KV cache configuration.

    These are the values phase 2 needs: the per-layer cache geometry that a
    custom KVCacheSpec / KVCacheManager (Option A) must reproduce.
    """
    import vllm

    print(f"\n=== vLLM version: {vllm.__version__} ===")

    engine = _first_attr(llm, "llm_engine", "engine")
    vllm_config = _first_attr(engine, "vllm_config")
    model_config = _first_attr(vllm_config, "model_config") or _first_attr(
        engine, "model_config"
    )
    cache_config = _first_attr(vllm_config, "cache_config") or _first_attr(
        engine, "cache_config"
    )
    parallel_config = _first_attr(vllm_config, "parallel_config") or _first_attr(
        engine, "parallel_config"
    )

    print("\n=== Model / cache geometry ===")
    try:
        if model_config is not None:
            # Signatures vary; try with parallel_config, then without.
            def _call(fn_name, *args):
                fn = getattr(model_config, fn_name, None)
                if fn is None:
                    return "n/a"
                try:
                    return fn(*args)
                except TypeError:
                    try:
                        return fn()
                    except Exception:
                        return "n/a"

            print(f"  num_kv_heads : {_call('get_num_kv_heads', parallel_config)}")
            print(f"  head_size    : {_call('get_head_size')}")
            print(f"  num_layers   : {_call('get_num_layers', parallel_config)}")
            print(f"  dtype        : {getattr(model_config, 'dtype', 'n/a')}")
        if cache_config is not None:
            print(f"  block_size   : {getattr(cache_config, 'block_size', 'n/a')}")
            print(
                f"  num_gpu_blocks: "
                f"{getattr(cache_config, 'num_gpu_blocks', 'n/a')}"
            )
            print(f"  cache_dtype  : {getattr(cache_config, 'cache_dtype', 'n/a')}")
    except Exception as exc:  # noqa: BLE001 - spike: surface, don't crash
        print(f"  [warn] config introspection partial: {exc}")

    # The KVCacheSpec types live here; phase 2 subclasses one of these.
    try:
        from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

        print("\n=== V1 KV cache interface present ===")
        print(f"  KVCacheSpec base : {KVCacheSpec}")
        print(f"  FullAttentionSpec: {FullAttentionSpec}")
        print("  (phase 2 registers a keys_values spec + manager alongside these)")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not import v1 kv_cache_interface: {exc}")


def smoke_test_generation(args: argparse.Namespace):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        enforce_eager=True,  # disable CUDA graphs for clearer introspection
    )
    print("\n=== Smoke-test generation ===")
    out = llm.generate([args.prompt], SamplingParams(max_tokens=32, temperature=0.0))
    print(f"Prompt: {args.prompt!r}")
    print(f"Output: {out[0].outputs[0].text!r}")
    return llm


def main() -> None:
    args = parse_args()
    check_environment()
    llm = smoke_test_generation(args)
    report_vllm_config(llm)
    print(
        "\nSpike complete. Next: phase 2 — a custom KVCacheSpec + "
        "SingleTypeKVCacheManager hosting `lastrec` (see "
        "docs/vllm_integration.md)."
    )


if __name__ == "__main__":
    main()
