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
vLLM integration feasibility spike.

Goal: stand up a baseline model in vLLM 0.6.5 and dump the runtime extension
points that the keys_values KV-cache bridge will need to hook into (resolved
attention backend, KV cache shape, attention metadata structure).

This script does NOT run on macOS / CPU-only machines. Run it on a Linux box
with an NVIDIA GPU, in a venv with vLLM 0.6.5 installed:

    python3 -m venv vllm_venv && . vllm_venv/bin/activate
    pip install --upgrade pip
    pip install "vllm==0.6.5"
    python examples/vllm_spike.py --model Qwen/Qwen2.5-0.5B-Instruct

See docs/vllm_integration.md for the design context.
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF model id to load (default: Qwen/Qwen2.5-0.5B-Instruct).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum model/context length for the vLLM engine.",
    )
    parser.add_argument(
        "--prompt",
        default="The key idea behind selective KV caching is",
        help="Prompt used for the smoke-test generation.",
    )
    return parser.parse_args()


def check_environment() -> None:
    """Fail fast with a helpful message off-GPU instead of deep in vLLM."""
    try:
        import torch
    except ImportError:
        sys.exit("torch is not installed in this environment.")

    if not torch.cuda.is_available():
        sys.exit(
            "No CUDA device available. vLLM 0.6.5 needs a Linux + NVIDIA GPU "
            "box. This spike cannot run on macOS / CPU. See "
            "docs/vllm_integration.md (Environment notes)."
        )

    try:
        import vllm  # noqa: F401
    except ImportError:
        sys.exit(
            "vllm is not installed. Create a dedicated venv and "
            '`pip install "vllm==0.6.5"` (see this file\'s docstring).'
        )


def report_attention_backend() -> None:
    """Print which attention backend vLLM resolves to for this platform.

    This is the class we will subclass / shadow for the keys_values cache
    bridge (see docs/vllm_integration.md, Option A).
    """
    import vllm

    print(f"\n=== vLLM version: {vllm.__version__} ===")
    try:
        from vllm.attention.selector import get_attn_backend

        # Signature is version-sensitive; this matches the 0.6.x family. If it
        # changes, the failure message tells us exactly what to update.
        backend = get_attn_backend(
            head_size=64,
            dtype="float16",
            kv_cache_dtype="auto",
            block_size=16,
            is_attention_free=False,
        )
        print(f"Resolved attention backend: {backend}")
        print(f"  name              : {backend.get_name()}")
        print(f"  impl class        : {backend.get_impl_cls()}")
        print(f"  metadata class    : {backend.get_metadata_cls()}")
        kv_shape = backend.get_kv_cache_shape(
            num_blocks=1024, block_size=16, num_kv_heads=2, head_size=64
        )
        print(f"  kv_cache_shape    : {kv_shape}")
    except Exception as exc:  # noqa: BLE001 - spike: surface whatever breaks
        print(f"[warn] could not introspect attention backend directly: {exc}")
        print("       (inspect vllm/attention/selector.py for this version)")


def smoke_test_generation(args: argparse.Namespace) -> None:
    """Confirm the baseline model generates, and dump engine KV cache config."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        enforce_eager=True,  # disable CUDA graphs for clearer introspection
    )

    # Dig out the cache configuration the engine actually allocated.
    try:
        engine = llm.llm_engine
        cache_config = engine.cache_config
        model_config = engine.model_config
        print("\n=== Engine cache config ===")
        print(f"  block_size        : {cache_config.block_size}")
        print(f"  num_gpu_blocks    : {cache_config.num_gpu_blocks}")
        print(f"  num_cpu_blocks    : {cache_config.num_cpu_blocks}")
        print(f"  cache_dtype       : {cache_config.cache_dtype}")
        print(
            f"  num_kv_heads      : "
            f"{model_config.get_num_kv_heads(engine.parallel_config)}"
        )
        print(f"  head_size         : {model_config.get_head_size()}")
        print(
            f"  num_layers        : "
            f"{model_config.get_num_layers(engine.parallel_config)}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not read engine cache config: {exc}")

    print("\n=== Smoke-test generation ===")
    out = llm.generate(
        [args.prompt],
        SamplingParams(max_tokens=32, temperature=0.0),
    )
    print(f"Prompt: {args.prompt!r}")
    print(f"Output: {out[0].outputs[0].text!r}")


def main() -> None:
    args = parse_args()
    check_environment()
    report_attention_backend()
    smoke_test_generation(args)
    print(
        "\nSpike complete. Next: scaffold keys_values/vllm/ with a custom "
        "AttentionBackend hosting the `lastrec` policy (see "
        "docs/vllm_integration.md, phase 2)."
    )


if __name__ == "__main__":
    main()
