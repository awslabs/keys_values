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
Profile SDPA backends: naive (blocked), padded-query, eager+weights, FlashInfer+Triton.

Compares four backends across model configs, batch sizes, cache lengths, and chunk sizes:
  1. eager (no weights)    — scaled_dot_product_attention_in_blocks, return_attn_weights=False
  2. eager (with weights)  — scaled_dot_product_attention_in_blocks, return_attn_weights=True
  3. padded-query          — PyTorch SDPA with query padding (no weights)
  4. flashinfer+triton     — Two-phase: FlashInfer prefill + Triton weight kernel
"""

import gc
import math
import random
import time
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.nn.attention import SDPBackend

from litgpt.config import Config

from keys_values.attention import scaled_dot_product_attention_in_blocks, DefaultKeysAndValues
from keys_values.flashinfer_wrapper import FlashInferSDPA
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.test_utils import random_keys_values, random_tensor
from keys_values.sdpa_wrapper import scaled_dot_product_attention


def _cleanup():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()


def sample_inputs(
    config: Config,
    params: KVCacheParams,
    chunk_size: int,
    input_pos: int,
) -> Dict[str, torch.Tensor]:
    batch_size = params.max_batch_size
    cache_length = params.cache_length
    n_query_groups = params.n_query_groups
    query = random_tensor(params, num=chunk_size, is_query=True)
    key, value = random_keys_values(params, num=cache_length)
    index_kwargs = dict(dtype=torch.int64, device=params.device)
    token_positions = torch.randint(
        low=0,
        high=input_pos - 1,
        size=(batch_size, n_query_groups, cache_length),
        **index_kwargs,
    )
    for b in range(batch_size):
        for h in range(config.n_query_groups):
            randpos = torch.randperm(cache_length, **index_kwargs)[:chunk_size]
            token_positions[b, h, randpos] = torch.arange(
                input_pos, input_pos + chunk_size, **index_kwargs,
            )
    return {
        "query": query,
        "key": key,
        "value": value,
        "token_positions": token_positions,
    }


@torch.inference_mode()
def measure_eager_time(
    params: KVCacheParams,
    config: Config,
    chunk_size: int,
    return_attn_weights: bool,
    num_repeats: int,
    warmup_steps: int = 2,
    tmp_array_limit_gb: float = 4,
) -> Optional[float]:
    """Measure eager blocked SDPA time."""
    input_pos = 2 * params.cache_length
    scale_factor = 1.0 / math.sqrt(params.head_size)
    try:
        for _ in range(warmup_steps):
            data = sample_inputs(config, params, chunk_size, input_pos)
            scaled_dot_product_attention_in_blocks(
                query=data["query"],
                k_and_v=DefaultKeysAndValues(data["key"], data["value"]),
                scale_factor=scale_factor,
                return_attn_weights=return_attn_weights,
                input_pos=input_pos,
                token_positions=data["token_positions"],
                sliding_window_size=None,
                tmp_array_limit_gb=tmp_array_limit_gb,
            )
            torch.cuda.synchronize()

        sum_ms = 0.0
        for _ in range(num_repeats):
            data = sample_inputs(config, params, chunk_size, input_pos)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            scaled_dot_product_attention_in_blocks(
                query=data["query"],
                k_and_v=DefaultKeysAndValues(data["key"], data["value"]),
                scale_factor=scale_factor,
                return_attn_weights=return_attn_weights,
                input_pos=input_pos,
                token_positions=data["token_positions"],
                sliding_window_size=None,
                tmp_array_limit_gb=tmp_array_limit_gb,
            )
            torch.cuda.synchronize()
            sum_ms += (time.perf_counter() - t0) * 1000
        return sum_ms / num_repeats
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        _cleanup()
        return None


@torch.inference_mode()
def measure_padded_time(
    params: KVCacheParams,
    config: Config,
    chunk_size: int,
    num_repeats: int,
    warmup_steps: int = 2,
) -> Optional[float]:
    """Measure padded-query PyTorch SDPA time (no weights)."""
    input_pos = 2 * params.cache_length
    scale_factor = 1.0 / math.sqrt(params.head_size)
    sdpa_kernels = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.MATH,
    ]
    try:
        for _ in range(warmup_steps):
            data = sample_inputs(config, params, chunk_size, input_pos)
            scaled_dot_product_attention(
                query=data["query"],
                key=data["key"],
                value=data["value"],
                scale_factor=scale_factor,
                input_pos=input_pos,
                token_positions=data["token_positions"],
                sdpa_kernels=sdpa_kernels,
            )
            torch.cuda.synchronize()

        sum_ms = 0.0
        for _ in range(num_repeats):
            data = sample_inputs(config, params, chunk_size, input_pos)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            scaled_dot_product_attention(
                query=data["query"],
                key=data["key"],
                value=data["value"],
                scale_factor=scale_factor,
                input_pos=input_pos,
                token_positions=data["token_positions"],
                sdpa_kernels=sdpa_kernels,
            )
            torch.cuda.synchronize()
            sum_ms += (time.perf_counter() - t0) * 1000
        return sum_ms / num_repeats
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        _cleanup()
        return None


@torch.inference_mode()
def measure_flashinfer_time(
    params: KVCacheParams,
    chunk_size: int,
    num_repeats: int,
    wrapper: FlashInferSDPA,
    warmup_steps: int = 2,
) -> Optional[Tuple[float, float, float]]:
    """Measure FlashInfer+Triton two-phase time with attention weights.

    Returns (total_ms, phase1_ms, phase2_ms) or None on failure.
    """
    from keys_values import flashinfer_ops
    from keys_values.triton_kernels import compute_weights_from_lse_triton

    batch_size = params.max_batch_size
    cache_length = params.cache_length
    input_pos = cache_length - chunk_size
    scale_factor = 1.0 / math.sqrt(params.head_size)

    # Allocate tensors (no token_positions — FlashInfer uses input_pos for causal mask)
    query = random_tensor(params, num=chunk_size, is_query=True)
    key, value = random_keys_values(params, num=cache_length)

    # Prepare Phase 1 inputs (FlashInfer expects [batch, seq, head, dim])
    query_t = query.transpose(1, 2).contiguous()
    key_t = key.transpose(1, 2).contiguous()
    value_t = value.transpose(1, 2).contiguous()
    input_pos_tensor = torch.tensor(
        [input_pos] * batch_size, device=params.device, dtype=torch.int32
    )

    try:
        # Warmup
        for _ in range(warmup_steps):
            out_t, _, lse = flashinfer_ops.sdpa_prefill(
                query=query_t, key=key_t, value=value_t, scale=scale_factor,
                token_positions=None, input_pos=input_pos_tensor,
                sliding_window_size=-1, causal=True,
                return_weights=False, return_lse=True,
            )
            _ = compute_weights_from_lse_triton(
                query, key, lse, scale_factor, input_pos, None
            )
            torch.cuda.synchronize()

        # Phase 1 timing
        torch.cuda.synchronize()
        p1_start = time.perf_counter()
        for _ in range(num_repeats):
            out_t, _, lse = flashinfer_ops.sdpa_prefill(
                query=query_t, key=key_t, value=value_t, scale=scale_factor,
                token_positions=None, input_pos=input_pos_tensor,
                sliding_window_size=-1, causal=True,
                return_weights=False, return_lse=True,
            )
            torch.cuda.synchronize()
        p1_end = time.perf_counter()

        # Phase 2 timing
        torch.cuda.synchronize()
        p2_start = time.perf_counter()
        for _ in range(num_repeats):
            _ = compute_weights_from_lse_triton(
                query, key, lse, scale_factor, input_pos, None
            )
            torch.cuda.synchronize()
        p2_end = time.perf_counter()

        # Total timing (both phases together)
        torch.cuda.synchronize()
        total_start = time.perf_counter()
        for _ in range(num_repeats):
            out_t, _, lse = flashinfer_ops.sdpa_prefill(
                query=query_t, key=key_t, value=value_t, scale=scale_factor,
                token_positions=None, input_pos=input_pos_tensor,
                sliding_window_size=-1, causal=True,
                return_weights=False, return_lse=True,
            )
            _ = compute_weights_from_lse_triton(
                query, key, lse, scale_factor, input_pos, None
            )
            torch.cuda.synchronize()
        total_end = time.perf_counter()

        phase1_ms = (p1_end - p1_start) * 1000 / num_repeats
        phase2_ms = (p2_end - p2_start) * 1000 / num_repeats
        total_ms = (total_end - total_start) * 1000 / num_repeats
        return total_ms, phase1_ms, phase2_ms
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        _cleanup()
        return None


def _fmt(val):
    if val is None:
        return "OOM"
    return f"{val:.2f}"


def profile_config(
    config: Config,
    batch_size: int,
    cache_lengths: List[int],
    chunk_sizes: List[int],
    dtype: torch.dtype,
    num_repeats: int,
    warmup_steps: int = 2,
    tmp_array_limit_gb: float = 4,
):
    """Profile all backends for a given model config."""
    device = torch.device("cuda", 0)
    wrapper = FlashInferSDPA()
    fi_available = wrapper.available

    import logging
    logging.getLogger('keys_values.flashinfer_wrapper').setLevel(logging.ERROR)

    fp = (config.n_head, config.n_query_groups, config.head_size)
    print(f"\n{'='*110}")
    print(f"  Config: n_head={fp[0]}, n_kv_heads={fp[1]}, head_dim={fp[2]}, "
          f"batch_size={batch_size}, dtype={dtype}")
    print(f"{'='*110}")

    print(f"\n  {'kv_len':>7} {'q_len':>6}  "
          f"{'eager':>8}  {'eager+w':>8}  {'padded':>8}  "
          f"{'fi+tri':>8}  {'fi_p1':>8}  {'fi_p2':>8}  "
          f"{'fi/eager+w':>11}  {'fi/padded':>10}")
    print(f"  {'':->7} {'':->6}  "
          f"{'':->8}  {'':->8}  {'':->8}  "
          f"{'':->8}  {'':->8}  {'':->8}  "
          f"{'':->11}  {'':->10}")

    for cache_length in cache_lengths:
        params = KVCacheParams(
            max_batch_size=batch_size,
            n_query_groups=config.n_query_groups,
            cache_length=cache_length,
            head_size=config.head_size,
            n_head=config.n_head,
            dtype=dtype,
            device=device,
        )

        for chunk_size in chunk_sizes:
            if chunk_size >= cache_length:
                continue

            _cleanup()

            # 1. Eager (no weights)
            eager_ms = measure_eager_time(
                params, config, chunk_size, return_attn_weights=False,
                num_repeats=num_repeats, warmup_steps=warmup_steps,
                tmp_array_limit_gb=tmp_array_limit_gb,
            )

            _cleanup()

            # 2. Eager (with weights)
            eager_w_ms = measure_eager_time(
                params, config, chunk_size, return_attn_weights=True,
                num_repeats=num_repeats, warmup_steps=warmup_steps,
                tmp_array_limit_gb=tmp_array_limit_gb,
            )

            _cleanup()

            # 3. Padded-query (no weights)
            padded_ms = measure_padded_time(
                params, config, chunk_size,
                num_repeats=num_repeats, warmup_steps=warmup_steps,
            )

            _cleanup()

            # 4. FlashInfer+Triton (with weights)
            fi_result = None
            if fi_available:
                fi_result = measure_flashinfer_time(
                    params, chunk_size,
                    num_repeats=num_repeats, wrapper=wrapper,
                    warmup_steps=warmup_steps,
                )

            fi_total = fi_result[0] if fi_result else None
            fi_p1 = fi_result[1] if fi_result else None
            fi_p2 = fi_result[2] if fi_result else None

            # Speedup ratios
            if fi_total and eager_w_ms and fi_total > 0 and eager_w_ms > 0:
                ratio_ew = f"{eager_w_ms / fi_total:.2f}x"
            else:
                ratio_ew = "N/A"
            if fi_total and padded_ms and fi_total > 0 and padded_ms > 0:
                ratio_pd = f"{padded_ms / fi_total:.2f}x"
            else:
                ratio_pd = "N/A"

            print(f"  {cache_length:>7} {chunk_size:>6}  "
                  f"{_fmt(eager_ms):>8}  {_fmt(eager_w_ms):>8}  {_fmt(padded_ms):>8}  "
                  f"{_fmt(fi_total):>8}  {_fmt(fi_p1):>8}  {_fmt(fi_p2):>8}  "
                  f"{ratio_ew:>11}  {ratio_pd:>10}")

            _cleanup()


def fingerprint(config: Config) -> Tuple[int, int, int]:
    return (config.n_head, config.n_query_groups, config.head_size)


if __name__ == "__main__":
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("This script needs to be run on a GPU instance")

    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.0f} GB")
    print(f"PyTorch Version: {torch.__version__}")

    dtype = torch.bfloat16
    num_repeats = 10
    batch_sizes = [1, 2, 4]
    cache_lengths = [4096, 8192, 16384, 32768]
    chunk_sizes = [256, 512, 1024, 2048]

    config_names = [
        "Qwen2.5-0.5B",   # (14, 2, 64)
        "Qwen2.5-1.5B",   # (12, 2, 128)
        "Qwen2.5-3B",     # (16, 2, 128)
        "Qwen2.5-7B",     # (28, 4, 128)
        "Qwen2.5-14B",    # (40, 8, 128)
        "Qwen3-0.6B",     # (16, 8, 128)
        "Qwen3-1.7B",     # (16, 8, 128) [skip]
        "Qwen3-4B",       # (32, 8, 128)
        "Qwen3-8B",       # (32, 8, 128) [skip]
        "Qwen3-14B",      # (40, 8, 128) [skip]
        "Qwen3-32B",      # (64, 8, 128)
        "Llama-2-7b-hf",  # (32, 32, 128)
        "Llama-2-13b-hf", # (40, 40, 128)
        "Llama-3-8B",     # (32, 8, 128) [skip]
        "Llama-3.1-8B",   # (32, 8, 128) [skip]
        "Llama-3.2-1B",   # (32, 8, 64)
        "Llama-3.2-3B",   # (24, 8, 128)
    ]

    fingerprints_done: Set[Tuple[int, int, int]] = set()
    for name in config_names:
        config = Config.from_name(name)
        fp = fingerprint(config)
        if fp not in fingerprints_done:
            for batch_size in batch_sizes:
                print(f"\n>>> Config[{name}], batch_size = {batch_size}")
                profile_config(
                    config,
                    batch_size,
                    cache_lengths,
                    chunk_sizes,
                    dtype,
                    num_repeats,
                )
            fingerprints_done.add(fp)
        else:
            print(f"\nConfig[{name}] already covered: {fp}")
