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
import time
from typing import List, Dict, Any, Tuple, Set

from scipy.optimize import root_scalar
import torch

from keys_values.config import Config

from keys_values.attention import (
    scaled_dot_product_attention_in_blocks,
    DefaultKeysAndValues,
)
from keys_values.flex_attention import (
    scaled_dot_product_attention_flexatt,
    FlexAttentionArgs,
)
from keys_values.kvcache.base import KVCacheParams
from keys_values.scripts.profile_naive_vs_padded_sdpa import (
    sample_inputs,
    fingerprint,
)
from keys_values.utils import append_results_to_csv


@torch.inference_mode()
def measure_naive_and_flexatt_time(
    chunk_size: float,
    params: KVCacheParams,
    num_repeats: int,
    warmup_steps: int,
    records: List[dict],
    tmp_array_limit_gb: float,
    device: torch.device,
    _compile: bool,
) -> float:
    chunk_size = round(chunk_size)
    input_pos = 2 * params.cache_length  # Value does not matter
    scale_factor = 1.0 / math.sqrt(params.head_size)
    # In reality, we use different compiled expressions for each chunk size,
    # so need to do this here as well
    flexatt_args = FlexAttentionArgs(_compile=_compile)
    sum_timediff_in_ms = 0
    for repeat in range(num_repeats):
        data = sample_inputs(params, chunk_size, input_pos, device)
        forward_time = None
        try:
            # Naive SDPA
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
        except RuntimeError:
            # Most likely out of memory error
            if repeat >= warmup_steps:
                sum_timediff_in_ms = 100000 * num_repeats
                break
        time_in_ms_naive = (time.perf_counter() - forward_time) * 1000
        if repeat >= warmup_steps:
            sum_timediff_in_ms += time_in_ms_naive
        try:
            # FlexAttention SDPA
            forward_time = time.perf_counter()
            y = scaled_dot_product_attention_flexatt(
                flexatt_args=flexatt_args,
                query=data["query"],
                key=data["key"],
                value=data["value"],
                scale_factor=scale_factor,
                sliding_window_size=None,
                attention_logit_softcapping=None,
                input_pos=input_pos,
                token_positions=data["token_positions"],
            )
            torch.cuda.current_stream().synchronize()
        except RuntimeError:
            # Most likely this one:
            # torch._inductor.exc.InductorError: RuntimeError: No valid triton configs. OutOfMemoryError: out of resource: triton_tem_fused_flex_attention_0 Required: 278528 Hardware limit:166912 Reducing block sizes or `num_stages` may help.
            if repeat >= warmup_steps:
                sum_timediff_in_ms = -100000 * num_repeats
                break
        time_in_ms_flex = (time.perf_counter() - forward_time) * 1000
        if repeat >= warmup_steps:
            sum_timediff_in_ms -= time_in_ms_flex
            records.append(
                {
                    "cache_length": params.cache_length,
                    "chunk_size": chunk_size,
                    "repeat": repeat,
                    "time_in_ms_naive": time_in_ms_naive,
                    "time_in_ms_flex": time_in_ms_flex,
                }
            )

    return sum_timediff_in_ms / num_repeats


@torch.inference_mode()
def find_chunk_size(
    sdpa_type: str,
    params: KVCacheParams,
    num_repeats: int,
    tmp_array_limit_gb: float,
    device: torch.device,
    warmup_steps: int = 2,
    xtol: float = 1,
    maxiter: int = 100,
) -> Dict[str, Any]:
    """
    Given a setup in `params`, in particular `params.cache_length`, we run root
    finding to search for a `chunk_size` value such that
    `scaled_dot_product_attention_in_blocks` takes about the same time as
    `scaled_dot_product_attention_flexatt`. We return the root value for
    `chunk_size`, as well as a list of dictionaries containing all the
    evaluations.

    Note: Different to zero-padded SDPA, FlexAttention SDPA times depend on
    `chunk_size`, so the difference may not be strictly decreasing and have
    a unique root.

    """
    if params.cache_length < 32:
        raise ValueError("params.cache_length must be greater than 32")
    # Running root finding
    records = []
    root_func = partial(
        measure_naive_and_flexatt_time,
        params=params,
        num_repeats=num_repeats,
        warmup_steps=warmup_steps,
        records=records,
        tmp_array_limit_gb=tmp_array_limit_gb,
        device=device,
        _compile=sdpa_type.endswith("_comp"),
    )
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
            print(
                f"f({bracket[0]}) = {f_left} > 0: Always use {sdpa_type} SDPA for cache_lenght = {params.cache_length}"
            )
            always_left = True
        elif f_right < 0:
            print(
                f"f({bracket[1]}) = {f_right} < 0: Never use {sdpa_type} SDPA for cache_lenght = {params.cache_length}"
            )
            always_left = False
        else:
            print(
                f"f({bracket[0]}) = {f_left} <= 0 <= {f_right} = f({bracket[1]}): Seems fine to me. Pick the one closer to 0."
            )
            always_left = abs(f_left) <= abs(f_right)
        if always_left:
            return {
                "chunk_size": bracket[0],
                "converged": False,
                "flag": f"Always use {sdpa_type} SDPA",
                "records": records,
            }
        else:
            return {
                "chunk_size": bracket[1],
                "converged": False,
                "flag": f"Never use {sdpa_type} SDPA",
                "records": records,
            }

    chunk_size = round(result.root)
    if result.converged:
        print(
            f"Converged with {result.function_calls} evals.\nchunk_size = {chunk_size}"
        )
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
    sdpa_type: str,
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
    torch.random.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This script needs to be run on a GPU instance")
    device = torch.device("cuda", 0)
    fixed_result = dict(
        batch_size=batch_size,
        n_head=config.n_head,
        n_query_groups=config.n_query_groups,
        head_size=config.head_size,
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
        )
        root = find_chunk_size(
            sdpa_type=sdpa_type,
            params=params,
            num_repeats=num_repeats,
            tmp_array_limit_gb=tmp_array_limit_gb,
            device=device,
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


# Note: It turns out that `flex_attention` is almost always faster than
# our eager blocked variant, so the data obtained by this script is not
# used in the end.
#
# Different to `profile_naive_vs_padded_sdpa.py`, the functions targeted
# here may not have a unique root, so the procedure here is brittle.
if __name__ == "__main__":
    # sdpa_type = "zero_padded"
    # sdpa_type = "flexatt_comp"
    sdpa_type = "flexatt_nocomp"
    dtype = torch.bfloat16
    num_repeats = 20
    batch_sizes = [1, 2, 3, 4, 8]
    cache_lengths = [4096, 6144, 8192, 12288, 16384, 24576, 32768]
    # We play through quite some config's here, but the calibration only
    # depends on `(n_head, n_query_groups, head_size)`, so we skip over
    # configs if this tuple has already been measured.
    config_names = [
        "Qwen2.5-0.5B",  # (14, 2, 64)
        "Qwen2.5-1.5B",  # (12, 2, 128)
        "Qwen2.5-3B",  # (16, 2, 128)
        "Qwen2.5-7B",  # (28, 4, 128)
        "Qwen2.5-14B",  # (40, 8, 128)
        "Qwen3-0.6B",  # (16, 8, 128)
        "Qwen3-1.7B",  # (16, 8, 128) [skip]
        "Qwen3-4B",  # (32, 8, 128)
        "Qwen3-8B",  # (32, 8, 128) [skip]
        "Qwen3-14B",  # (40, 8, 128) [skip]
        "Qwen3-32B",  # (64, 8, 128)
        "Llama-2-7b-hf",  # (32, 32, 128)
        "Llama-2-13b-hf",  # (40, 40, 128)
        "Llama-3-8B",  # (32, 8, 128) [skip]
        "Llama-3.1-8B",  # (32, 8, 128) [skip]
        "Llama-3.2-1B",  # (32, 8, 64)
        "Llama-3.2-3B",  # (24, 8, 128)
    ]
    result_path = Path(f"./{sdpa_type}_qlen_thresholds_extra.csv")
    all_evals_path = Path(f"./{sdpa_type}_qlen_thresholds_all_evals_extra.csv")
    fingerprints_done: Set[Tuple[int, int, int]] = set()
    for name in config_names:
        config = Config.from_name(name)
        fp = fingerprint(config)
        if fp not in fingerprints_done:
            for batch_size in batch_sizes:
                print(f"\nConfig[{name}], batch_size = {batch_size}")
                main(
                    config,
                    sdpa_type,
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
