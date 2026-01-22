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
import math
import time
from typing import List

import torch
import numpy as np

from litgpt.config import Config

from keys_values.attention_utils import SDPA_KERNELS_BEST_ORDERING
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.test_utils import random_args_cache_forward
from keys_values.sdpa_wrapper import scaled_dot_product_attention
from keys_values.sdpa_wrapper_old import scaled_dot_product_attention as wrap_sdpa_old


@torch.inference_mode()
def main(
    config: Config,
    batch_size: int,
    cache_length: int,
    chunk_sizes: List[int],
    dtype: torch.dtype,
    num_repeats: int,
    warmup_repeats: int = 2,
):
    seed = 31415927
    torch.random.manual_seed(seed)
    on_gpu = torch.cuda.is_available()
    device = torch.device("cuda", 0) if on_gpu else torch.device("cpu")

    index_kwargs = dict(dtype=torch.int64, device=device)
    sdpa_kernels = SDPA_KERNELS_BEST_ORDERING.copy()
    params = KVCacheParams(
        max_batch_size=batch_size,
        n_query_groups=config.n_query_groups,
        cache_length=cache_length,
        head_size=config.head_size,
        n_head=config.n_head,
        dtype=dtype,
    )
    input_pos = 2 * cache_length
    scale_factor = 1.0 / math.sqrt(config.head_size)

    print(
        f"config = {config}\ncache_length = {cache_length}\nbatch_size = {batch_size}\ndtype = {dtype}"
    )
    for chunk_size in chunk_sizes:
        print(f"\nchunk_size = {chunk_size}")
        results = np.zeros((3, num_repeats), dtype=np.float64)
        for repeat in [None] * warmup_repeats + list(range(num_repeats)):
            # Sample input data
            data = random_args_cache_forward(
                params,
                cache_length,
                config.padded_vocab_size,
            )
            data["query"] = data["query"][:, :, :chunk_size, :]
            token_positions = torch.randint(
                low=0,
                high=input_pos - 1,
                size=(batch_size, config.n_query_groups, cache_length),
                **index_kwargs,
            )
            for b in range(batch_size):
                for h in range(config.n_query_groups):
                    randpos = torch.randperm(cache_length, **index_kwargs)[:chunk_size]
                    token_positions[b, h, randpos] = torch.arange(
                        input_pos,
                        input_pos + chunk_size,
                        **index_kwargs,
                    )
            sdpa_kwargs = dict(
                query=data["query"],
                key=data["key"],
                value=data["value"],
                scale_factor=scale_factor,
                input_pos=input_pos,
                token_positions=token_positions,
                sdpa_kernels=sdpa_kernels,
            )
            # Loop over kinds. Measure time
            for kind in range(3):
                if on_gpu:
                    torch.cuda.current_stream().synchronize()
                forward_time = time.perf_counter()
                if kind < 2:
                    y = wrap_sdpa_old(**sdpa_kwargs, kind=kind)
                else:
                    y, _ = scaled_dot_product_attention(**sdpa_kwargs)
                if on_gpu:
                    torch.cuda.current_stream().synchronize()
                time_in_ms = (time.perf_counter() - forward_time) * 1000
                if repeat is not None:
                    results[kind, repeat] = time_in_ms
        # Results averaged over repeats
        res_mean = results.mean(axis=1)
        res_std = results.std(axis=1)
        for kind, (mn, st) in enumerate(zip(res_mean, res_std)):
            print(f"kind={kind}: time_ms = {mn:.2f} (+- {st:.3f})")


if __name__ == "__main__":
    batch_size = 8
    dtype = torch.bfloat16
    num_repeats = 50
    cache_length = 32768
    chunk_sizes = [256, 512, 1024, 2048, 4096]
    config = Config.from_name("Qwen3-4B")
    main(
        config,
        batch_size,
        cache_length,
        chunk_sizes,
        dtype,
        num_repeats,
    )
