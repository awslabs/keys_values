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
from dataclasses import replace
import random
from pathlib import Path
import time
from typing import List, Tuple

import torch

from litgpt.config import Config

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.kvcache.test_utils import random_args_cache_forward
from keys_values.utils import append_results_to_csv


def main(
    setups: List[Tuple[int, int]],
    config: Config,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    result_path: Path,
    num_repeats: int,
    warmup_repeats: int = 2,
):
    """
    Profiles times to run :meth:`forward` for caches "lastrec" and
    "lastrec-alt".

    """
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    on_gpu = torch.cuda.is_available()

    params = KVCacheParams(
        max_batch_size=batch_size,
        n_query_groups=config.n_query_groups,
        cache_length=128,  # Will be changed
        head_size=config.head_size,
        n_head=config.n_head,
        dtype=dtype,
    )
    cache_kwargs = {
        "tmp_array_limit_gb": TemporaryArrayLimit(
            init_val=4, name="attention_forward_temp_size_gb",
        )
    }
    names = ["lastrec-default", "lastrec-alt-default"]
    print(f"Comparing caches: {names}")
    result_fixed = dict()
    for key in ("n_head", "n_query_groups", "head_size", "max_batch_size", "dtype", "device"):
        val = getattr(params, key)
        print(f"{key} = {getattr(params, key)}")
        result_fixed[key] = str(val) if key in {"dtype", "device"} else val

    for cache_length, chunk_size in setups:
        print(f"\ncache_length = {cache_length}, chunk_size = {chunk_size}")
        # Create caches
        params = replace(params, cache_length=cache_length)
        caches = [
            KVCacheFactory.create_single(
                name=name,
                config=config,
                max_batch_size=batch_size,
                cache_length=cache_length,
                block_idx=0,
                device=device,
                dtype=dtype,
                cache_kwargs=cache_kwargs,
            )
            for name in names
        ]
        rows = []
        for repeat in [None] * warmup_repeats + list(range(num_repeats)):
            next_pos = random.randint(0, cache_length - 1)
            if repeat is not None:
                result = dict(
                    result_fixed,
                    cache_length=cache_length,
                    chunk_size=chunk_size,
                    next_pos=next_pos,
                    repeat=repeat,
                )
            prefill = None
            insert1 = None
            insert2 = None
            for cache in caches:
                cache.reset()
            with torch.no_grad():
                for name, cache in zip(names, caches):
                    # Prefill (prepare)
                    num = cache_length
                    if prefill is None:
                        prefill = random_args_cache_forward(
                            params, num, config.padded_vocab_size,
                        )
                    cache(**prefill)
                    # Insert (prepare)
                    num = next_pos
                    if insert1 is None:
                        insert1 = random_args_cache_forward(
                            params, num, config.padded_vocab_size,
                        )
                    cache(**insert1)
                    # Insert (measure)
                    num = chunk_size
                    if insert2 is None:
                        insert2 = random_args_cache_forward(
                            params, num, config.padded_vocab_size,
                        )
                    if on_gpu:
                        torch.cuda.current_stream().synchronize()
                    forward_time = time.perf_counter()
                    y = cache(**insert2)
                    if on_gpu:
                        torch.cuda.current_stream().synchronize()
                    time_in_ms = (time.perf_counter() - forward_time) * 1000
                    if repeat is not None:
                        _name = name.replace("-", "_")
                        result[f"time_{_name}"] = time_in_ms

            if repeat is not None:
                rows.append(result)

        # Append results to file
        print(f"Append to {result_path}")
        append_results_to_csv(rows, result_path)


if __name__ == "__main__":
    batch_size = 3
    dtype = torch.bfloat16
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    num_repeats = 10
    result_path = Path("./profile_lastrec.csv")
    cache_length = 2 ** 15
    setups = [
        (cache_length, chunk_size)
        for chunk_size in [2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11] + list(range(2048, 2 ** 14, 2048))
    ]
    model_names = ["Qwen3-4B"]
    for model_name in model_names:
        print(f"\nRunning for {model_name} setup")
        config = Config.from_name(model_name)
        main(
            setups,
            config,
            batch_size,
            dtype,
            device,
            result_path,
            num_repeats,
        )
