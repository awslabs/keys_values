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
from itertools import product
import random
from typing import List

import torch
import pytest

from litgpt.config import Config

from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.factory import split_name
from keys_values.kvcache.gradient.checkpoints import LayerInputQuantizedCheckpoints
from keys_values.kvcache.test_utils import (
    create_kv_cache,
    cache_names_and_devices,
)
from keys_values.model import GPT


def args_layer_input_quantized_checkpoints() -> List[tuple]:
    names_devices = [
        tup
        for tup in cache_names_and_devices()
        if split_name(tup[0])[0] == "lastrec" and split_name(tup[0])[1] != "default"
    ]
    setups = [
        (64, 1024),
        (128, 256),
        (256, 1024),
    ]
    return [a + b for a, b in product(names_devices, setups)]


@pytest.mark.parametrize(
    "cache_name, device, chunk_size, max_seq_length",
    args_layer_input_quantized_checkpoints(),
)
def test_layer_input_quantized_checkpoints(
    cache_name,
    device,
    chunk_size,
    max_seq_length,
):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)

    num_set_calls = 20
    batch_size = 3
    dtype = torch.bfloat16
    n_layer = 1
    n_head = 8
    n_query_groups = 4
    head_size = 64
    vocab_size = 48
    qname = split_name(cache_name)[1]

    config = Config(
        n_layer=n_layer,
        n_head=n_head,
        n_query_groups=n_query_groups,
        n_embd=n_head * head_size,
        block_size=max_seq_length,
        vocab_size=vocab_size,
        rotary_percentage=1,
    )
    params = KVCacheParams.from_config(
        config=config,
        max_batch_size=batch_size,
        cache_length=chunk_size,
        dtype=dtype,
    )
    with torch.device(device):
        gpt_model = GPT(config)
        kv_cache = create_kv_cache(
            name=cache_name,
            params=params,
            block_idx=0,
        )
        gpt_model.assign_kv_caches([kv_cache])

    # Create two checkpointers, one with chunk size `chunk_size`, the
    # other with chunk size `max_seq_length` (single chunk)
    kwargs = dict(
        model=gpt_model,
        layer_numbers=[0],
        batch_size=batch_size,
        qname=qname,
    )
    checkpoints = []
    for csize in (chunk_size, max_seq_length):
        checkpoints.append(
            LayerInputQuantizedCheckpoints(chunk_size=csize, **kwargs),
        )

    # Comparison loop
    kwargs = dict(dtype=dtype, device=device)
    for run_no in range(num_set_calls):
        # Sample arguments for `set_checkpoint`
        if run_no == 0:
            set_num = max_seq_length
            set_input_pos = 0
        else:
            set_num = random.randint(1, max_seq_length // 2)
            set_input_pos = random.randint(0, max_seq_length - set_num)
        buffers = torch.randn(batch_size, set_num, config.n_embd, **kwargs)
        get_num = random.randint(chunk_size, max_seq_length)
        get_input_pos = random.randint(0, max_seq_length - get_num)
        # Modify and compare
        results = []
        for cp in checkpoints:
            assert (
                cp.set_checkpoint(
                    layer_idx=0,
                    buffers=buffers,
                    input_pos=set_input_pos,
                )
                == 0
            )
            results.append(
                cp.get_checkpoint(
                    layer_idx=0,
                    input_pos=get_input_pos,
                    num=get_num,
                    device=torch.device("cpu"),
                )
            )
        torch.testing.assert_close(results[0], results[1])
