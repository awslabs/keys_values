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
from functools import partial
import math
from pathlib import Path
import random
import time
from typing import List, Dict, Any, Tuple, Set

from scipy.optimize import root_scalar
import torch
from torch.nn.attention import SDPBackend

from litgpt.config import Config

from keys_values.attention import scaled_dot_product_attention_in_blocks, DefaultKeysAndValues
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.test_utils import random_keys_values, random_tensor
from keys_values.sdpa_wrapper import scaled_dot_product_attention
from keys_values.utils import append_results_to_csv


def sample_inputs(
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
def measure_naive_time(
    chunk_size: float,
    params: KVCacheParams,
    num_repeats: int,
    records: List[dict],
    tmp_array_limit_gb: float,
    fval: float,
) -> float:
    chunk_size = round(chunk_size)
    input_pos = 2 * params.cache_length  # Value should not matter
    scale_factor = 1.0 / math.sqrt(params.head_size)
    sum_time_in_ms = 0
    try:
        for repeat in range(num_repeats):
            data = sample_inputs(params, chunk_size, input_pos)
            torch.cuda.current_stream().synchronize()
            forward_time = time.perf_counter()
            y, _ = scaled_dot_product_attention_in_blocks(
                query=data["query"],
                k_and_v=DefaultKeysAndValues(data["key"], data["value"]),
                scale_factor=scale_factor,
                return_attn_weights=False,
                input_pos=input_pos,
                token_positions=data["token_positions"],
                sliding_window_size=None,
                tmp_array_limit_gb=tmp_array_limit_gb,
            )
            torch.cuda.current_stream().synchronize()
            time_in_ms = (time.perf_counter() - forward_time) * 1000
            sum_time_in_ms += time_in_ms
            records.append(
                {
                    "cache_length": params.cache_length,
                    "chunk_size": chunk_size,
                    "repeat": repeat,
                    "time_in_ms": time_in_ms,
                }
            )
    except RuntimeError:
        # Most like out of memory error
        sum_time_in_ms = 100000 * num_repeats

    ret = (sum_time_in_ms / num_repeats) - fval
    return ret


@torch.inference_mode()
def find_chunk_size(
    params: KVCacheParams,
    num_repeats: int,
    tmp_array_limit_gb: float,
    warmup_steps: int = 2,
    xtol: float = 1,
    maxiter: int = 100,
) -> Dict[str, Any]:
    """
    Given a setup in `params`, in particular `params.cache_length`, we profile
    time for query-padded SDPA first, which does not depend on `chunk_size`.
    Then, we run root finding to search for a `chunk_size` value such that
    `scaled_dot_product_attention_in_blocks` takes about the same time.

    All evaluations (time in ms) are averaged over `num_repeats` repeats. We
    return the root value for `chunk_size`, as well as a list of dictionaries
    containing all the evaluations.

    """
    if params.cache_length < 32:
        raise ValueError("params.cache_length must be greater than 32")
    # Measure time for query-padded SDPA
    chunk_size = params.cache_length // 64  # Should not matter
    input_pos = 2 * params.cache_length
    sdpa_kernels = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.MATH,
    ]
    scale_factor = 1.0 / math.sqrt(params.head_size)
    sum_time_in_ms = 0
    for repeat in [None] * warmup_steps + list(range(num_repeats)):
        data = sample_inputs(params, chunk_size, input_pos)
        torch.cuda.current_stream().synchronize()
        forward_time = time.perf_counter()
        y = scaled_dot_product_attention(
            query=data["query"],
            key=data["key"],
            value=data["value"],
            scale_factor=scale_factor,
            input_pos=input_pos,
            token_positions=data["token_positions"],
            sdpa_kernels=sdpa_kernels,
        )
        torch.cuda.current_stream().synchronize()
        time_in_ms = (time.perf_counter() - forward_time) * 1000
        if repeat is not None:
            sum_time_in_ms += time_in_ms
    time_padded_query = sum_time_in_ms / num_repeats
    print(f"\ncache_length = {params.cache_length}: Time for padded-query SDPA = {time_padded_query} ms")
    # Running root finding
    records = []
    root_func = partial(
        measure_naive_time,
        params=params,
        num_repeats=num_repeats,
        records=records,
        tmp_array_limit_gb=tmp_array_limit_gb,
        fval=time_padded_query,
    )
    # Warm-up:
    for _ in range(warmup_steps):
        root_func(random.randint(1, params.cache_length))
    x0 = max(params.cache_length // 64, 8)
    bracket = (1, params.cache_length)
    try:
        result = root_scalar(
            f=root_func,
            x0=x0,
            bracket=bracket,
            xtol=xtol,
            maxiter=maxiter,
        )
    except ValueError as ex:
        # Usually bracket is not correct
        f_left = root_func(bracket[0])
        f_right = root_func(bracket[1])
        if f_left > 0:
            print(f"f({bracket[0]}) = {f_left} > 0: Always use query-padded SDPA for cache_lenght = {params.cache_length}")
            always_left = True
        elif f_right < 0:
            print(f"f({bracket[1]}) = {f_right} < 0: Never use query-padded SDPA for cache_lenght = {params.cache_length}")
            always_left = False
        else:
            print(f"f({bracket[0]}) = {f_left} <= 0 <= {f_right} = f({bracket[1]}): Seems fine to me. Pick the one closer to 0.")
            always_left = abs(f_left) <= abs(f_right)
        if always_left:
            return {
                "chunk_size": bracket[0],
                "converged": False,
                "flag": "Always use query-padded SDPA",
                "records": records,
            }
        else:
            return {
                "chunk_size": bracket[1],
                "converged": False,
                "flag": "Never use query-padded SDPA",
                "records": records,
            }

    chunk_size = round(result.root)
    if result.converged:
        print(f"Converged with {result.function_calls} evals.\nchunk_size = {chunk_size}")
    else:
        print(f"No convergence: {result.flag}\nchunk_size = {chunk_size}")
    return {
        "chunk_size": chunk_size,
        "converged": result.converged,
        "flag": result.flag,
        "records": records,
    }


def main(
    config: Config,
    batch_size: int,
    cache_lengths: List[int],
    dtype: torch.dtype,
    all_evals_path: Path,
    result_path: Path,
    num_repeats: int,
    warmup_steps: int = 2,
    tmp_array_limit_gb: float = 4,
):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This script needs to be run on a GPU instance")
    device = torch.device("cuda", 0)
    fixed_result = dict(
        batch_size=batch_size,
        n_head=config.n_head,
        n_query_groups=config.n_query_groups,
        dtype=str(dtype),
    )

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
        root = find_chunk_size(
            params=params,
            num_repeats=num_repeats,
            tmp_array_limit_gb=tmp_array_limit_gb,
            warmup_steps=warmup_steps,
        )
        new_result = {
            **{k: v for k, v in root.items() if k != "records"},
            **fixed_result,
            "cache_length": cache_length,
        }
        append_results_to_csv([new_result], result_path)
        # We also log all evaluations
        if root["records"]:
            append_results_to_csv(
                [{**record, **fixed_result} for record in root["records"]],
                all_evals_path,
            )


def fingerprint(config: Config) -> Tuple[int, int, int]:
    return (config.n_head, config.n_query_groups, config.head_size)


if __name__ == "__main__":
    dtype = torch.bfloat16
    num_repeats = 20
    batch_sizes = [1, 2, 3, 4, 8]
    cache_lengths = [4096, 6144, 8192, 12288, 16384, 24576, 32768]
    # We play through quite some config's here, but the calibration only
    # depends on `(n_head, n_query_group, head_size)`, so we skip over
    # configs if this tuple has already been measured.
    config_names = [
        "Qwen2.5-0.5B",
        "Qwen2.5-1.5B",
        "Qwen2.5-3B",
        "Qwen2.5-7B",
        "Qwen2.5-14B",
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
        "Qwen3-14B",
        "Qwen3-32B",
        "Llama-2-7b-hf",
        "Llama-2-13b-hf",
        "Llama-3-8B",
        "Llama-3.1-8B",
        "Llama-3.2-1B",
        "Llama-3.2-3B",
    ]
    result_path = Path("./qlen_thresholds.csv")
    all_evals_path = Path("./qlen_thresholds_all_evals.csv")
    fingerprints_done: Set[Tuple[int, int, int]] = set()
    for name in config_names:
        config = Config.from_name(name)
        fp = fingerprint(config)
        if fp not in fingerprints_done:
            for batch_size in batch_sizes:
                print(f"\nConfig[{name}], batch_size = {batch_size}")
                main(
                    config,
                    batch_size,
                    cache_lengths,
                    dtype,
                    all_evals_path,
                    result_path,
                    num_repeats,
                )
            fingerprints_done.add(fp)
        else:
            print(f"\nConfig[{name}] already covered: {fp}")
