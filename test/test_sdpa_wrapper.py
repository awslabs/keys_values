# Original Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# Modification Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import random

import pytest
import torch

from litgpt.config import Config

from keys_values.attention import (
    DefaultKeysAndValues,
    scaled_dot_product_attention_in_blocks,
)
from keys_values.attention_utils import SDPA_KERNELS_BEST_ORDERING
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.kvcache.test_utils import (
    product_with_devices,
    random_args_cache_forward,
    range_from_args,
)
from keys_values.sdpa_wrapper import scaled_dot_product_attention as wrapper_sdpa


@pytest.mark.parametrize(
    *product_with_devices(
        [
            (2, 1),
            (4, 1),
            (8, 4),
            (12, 4),
            (24, 8),
            (9, 3),
        ],
        "n_head, n_query_groups",
    ),
)
@torch.inference_mode()
def test_sdpa_wrapper(device, n_head, n_query_groups):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)

    num_repeats = 32
    if device == torch.device("mps"):
        dtype = torch.float32
    else:
        dtype = torch.bfloat16
    gen_kwargs = dict(dtype=dtype, device=device)
    index_kwargs = dict(dtype=torch.int64, device=device)
    if device == torch.device("cpu"):
        assert_kwargs = dict(atol=0.0005, rtol=0.05)
    else:
        assert_kwargs = dict(atol=0.0015, rtol=0.05)
    sdpa_kernels = SDPA_KERNELS_BEST_ORDERING.copy()

    print(f"n_head={n_head}, n_query_groups={n_query_groups}, device={device}")
    for repeat in range(num_repeats):
        head_size = 2 ** random.randint(3, 6)
        batch_size = random.randint(1, 5)
        kv_len = random.randint(64, 256)
        input_pos = random.randint(kv_len, 2 * kv_len)
        # Sample data: For `token_positions`, we iterate between two
        # cases: Overlap / no overlap between the two parts which are
        # exchanged
        token_positions = torch.zeros(
            (batch_size, n_query_groups, kv_len),
            **index_kwargs,
        )
        if repeat % 2 == 0:
            # No overlap case
            q_len = random.randint(1, kv_len // 2)
            left_sz = kv_len - q_len
            s1 = input_pos + q_len - left_sz
            l1 = left_sz
            l2 = s1
            for b in range(batch_size):
                for h in range(n_query_groups):
                    token_positions[b, h, :left_sz] = (
                        torch.randperm(
                            l1,
                            **index_kwargs,
                        )
                        + s1
                    )
                    token_positions[b, h, left_sz:] = torch.randperm(
                        l2,
                        **index_kwargs,
                    )[:q_len]
        else:
            # Overlap case
            q_len = random.randint(kv_len // 10, kv_len // 2)
            s1 = input_pos + q_len - kv_len
            done = False
            for _ in range(50):
                num_overlap = 0
                for b in range(batch_size):
                    for h in range(n_query_groups):
                        slice = torch.randperm(kv_len, **index_kwargs) + s1
                        num_overlap += (slice[(-q_len):] >= input_pos).sum().item()
                        token_positions[b, h, :] = slice
                if num_overlap >= 10:
                    done = True
                    break
            if not done:
                print("Did not manage to reach overlap threshold")
        print(f"repeat {repeat}:")
        print(
            f"head_size={head_size}, batch_size={batch_size}, kv_len={kv_len}, input_pos={input_pos}, q_len={q_len}"
        )
        shape = (batch_size, n_head, q_len, head_size)
        query = torch.randn(shape, **gen_kwargs)
        shape = (batch_size, n_query_groups, kv_len, head_size)
        key = torch.randn(shape, **gen_kwargs)
        value = torch.randn(shape, **gen_kwargs)
        scale_factor = 1.0 / math.sqrt(head_size)

        # Compute in two ways and compare
        output1, sdpa_kernels = wrapper_sdpa(
            query=query,
            key=key,
            value=value,
            scale_factor=scale_factor,
            input_pos=input_pos,
            token_positions=token_positions,
            sdpa_kernels=sdpa_kernels,
            do_filter_kernels=True,
        )
        output2, _ = scaled_dot_product_attention_in_blocks(
            query=query,
            k_and_v=DefaultKeysAndValues(key, value),
            scale_factor=scale_factor,
            return_attn_weights=False,
            input_pos=input_pos,
            token_positions=token_positions,
            sliding_window_size=None,
        )
        torch.testing.assert_close(output1, output2, **assert_kwargs)


@pytest.mark.parametrize(
    *product_with_devices(
        [
            (torch.float32, dict()),
            (torch.bfloat16, dict(atol=0.0005, rtol=0.03)),
            (torch.float16, dict(atol=0.0005, rtol=0.03)),
        ],
        "dtype, tol_kwargs",
    ),
)
@torch.inference_mode()
def test_wrapper_with_lastrec_cache(device, dtype, tol_kwargs):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)

    torch.set_default_dtype(dtype)  # Set default dtype
    batch_size = 5
    n_head = 8
    n_query_groups = 4
    cache_length = 512
    head_size = 64
    vocab_size = 48
    tokens_per_chunk = [cache_length - 1, 1, 8, 4, 8, 2, 8, 2, 8, 8]
    seq_length = sum(tokens_per_chunk)

    config = Config(
        n_layer=1,
        n_head=n_head,
        n_query_groups=n_query_groups,
        n_embd=n_head * head_size,
        block_size=seq_length,
        vocab_size=vocab_size,
        rotary_percentage=1,
    )
    params = KVCacheParams.from_config(
        config=config,
        max_batch_size=batch_size,
        cache_length=cache_length,
        dtype=dtype,
    )
    kwargs = dict(
        name="lastrec-default",
        config=config,
        max_batch_size=batch_size,
        cache_length=cache_length,
        block_idx=0,
        device=device,
        dtype=dtype,
    )
    # Choice between `SDPA_IMPL_EAGER_BLOCKS` and `SDPA_IMPL_QPADDED_PYTORCH`:
    # Always `SDPA_IMPL_EAGER_BLOCKS`:
    cache_old = KVCacheFactory.create_single(
        **kwargs,
        cache_kwargs=dict(
            use_eager_kernel=lambda kv_len, q_len: True,
        ),
    )
    cache_new = KVCacheFactory.create_single(**kwargs)
    assert cache_old.mha._use_eager_kernel is not None
    data = random_args_cache_forward(
        params,
        num=seq_length,
        vocab_size=vocab_size,
    )

    outputs = []
    for cache in (cache_old, cache_new):
        input_pos = 0
        parts = []
        for num in tokens_per_chunk:
            parts.append(cache(**range_from_args(data, input_pos, input_pos + num)))
            input_pos += num
        outputs.append(torch.cat(parts, dim=1))

    input_pos = 0
    for num in tokens_per_chunk:
        end = input_pos + num
        print(f"Comparing {input_pos}:{end}")
        torch.testing.assert_close(
            outputs[0][:, input_pos:end, :],
            outputs[1][:, input_pos:end, :],
            **tol_kwargs,
        )
        input_pos = end
