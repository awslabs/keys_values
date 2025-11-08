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
import csv
from dataclasses import replace
import random
from pathlib import Path
import time
from typing import List, Tuple, Dict, Any

import torch

from litgpt.config import Config, name_to_config

from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.kvcache.test_utils import random_args_cache_forward


# TODO: Use file locking as well!
def append_results(
    results: List[Dict[str, Any]],
    result_path: Path,
):
    fieldnames = sorted(results[0].keys())
    mode = "a" if result_path.exists() else "w"
    with result_path.open(mode) as fp:
        writer = csv.writer(fp, delimiter=",")
        if mode == "w":
            writer.writerow(fieldnames)
            for record in results:
                row = [record[name] for name in fieldnames]
                writer.writerow(row)


def main(
    setups: List[Tuple[int, int]],
    config: Config,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    result_path: Path,
    num_repeats: int,
):
    """
    Profiles times to run :meth:`forward` for caches "lastrec" and
    "lastrec-alt".

    """
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)

    params = KVCacheParams(
        max_batch_size=batch_size,
        n_query_groups=config.n_query_groups,
        cache_length=128,  # Will be changed
        head_size=config.head_size,
        n_head=config.n_head,
        dtype=dtype,
        device=device,
    )
    names = ["lastrec-default", "lastrec-alt-default"]
    print(f"Comparing caches: {names}")
    result_fixed = dict()
    for key in ("n_head", "n_query_groups", "head_size", "batch_size", "dtype", "device"):
        val = getattr(params, key)
        print(f"{key} = {getattr(params, key)}")
        result_fixed[key] = str(val) if key in {"dtype", "device"} else val

    for cache_length, chunk_size in setups:
        print(f"\ncache_length = {cache_length}, chunk_size = {chunk_size}")
        params = replace(params, cache_length=cache_length)
        rows = []
        for repeat in range(num_repeats):
            next_pos = random.randint(0, cache_length - 1)
            # Create and prepare caches
            caches = []
            prefill = None
            insert = None
            input_pos = 0
            for name in names:
                cache = KVCacheFactory.create_single(
                    name=name,
                    config=config,
                    max_batch_size=batch_size,
                    cache_length=cache_length,
                    block_idx=0,
                    device=device,
                    dtype=dtype,
                )
                caches.append(cache)
                # Prefill
                num = cache_length
                if prefill is None:
                    prefill = random_args_cache_forward(
                        params, num, config.padded_vocab_size,
                    )
                _input_pos = 0
                cache(**prefill, input_pos=_input_pos)
                _input_pos += num
                # Insert
                num = next_pos
                if insert is None:
                    insert = random_args_cache_forward(
                        params, num, config.padded_vocab_size,
                    )
                cache(**insert, input_pos=_input_pos)
                _input_pos += num
                input_pos = _input_pos
            # Profile times for forward
            result = dict(
                result_fixed,
                cache_length=cache_length,
                chunk_size=chunk_size,
                next_pos=next_pos,
                repeat=repeat,
            )
            insert = None
            for name, cache in zip(names, caches):
                num = chunk_size
                if insert is None:
                    insert = random_args_cache_forward(
                        params, num, config.padded_vocab_size,
                    )
                forward_time = time.perf_counter()
                y = cache(**insert, input_pos=input_pos)
                time_in_ms = (time.perf_counter() - forward_time) * 1000
                _name = name.replace("-", "_")
                result[f"time_{_name}"] = time_in_ms
            rows.append(result)

        # Append results to file
        print(f"Append to {result_path}")
        append_results(rows, result_path)


if __name__ == "__main__":
    # Setup for Qwen3-4B
    model_name = "Qwen3-4B"
    config = name_to_config[model_name]
    batch_size = 3
    dtype = torch.bfloat16
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    num_repeats = 10
    result_path = Path("./profile_lastrec.csv")
    cache_length = 2 ** 15
    setups = [
        (cache_length, chunk_size)
        for chunk_size in [2 ** 9, 2 ** 10, 2 ** 11] + list(range(2048, 2 ** 14, 2048))
    ]
    main(
        setups,
        config,
        batch_size,
        dtype,
        device,
        result_path,
        num_repeats,
    )
