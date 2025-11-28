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
from itertools import product
import random

import torch
from torch.linalg import vector_norm
import pytest

from keys_values.long_context import LongContextInferenceModel
from litgpt.config import Config

from keys_values.attention import DefaultKeysAndValues
from keys_values.head_model import CrossEntropyOnLogits
from keys_values.head_model_factory import HeadModelFactory
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.test_utils import (
    compute_attn_weights,
    create_kv_cache,
    tensor_is_simple,
    random_keys_values,
    random_tensor,
    available_backends,
    random_args_cache_forward,
    range_from_args,
)
from keys_values.model import GPT
from keys_values.utils import expand_index, copy_parameters


@pytest.mark.parametrize(
    "device, name",
    product(
        available_backends(),
        ["h2o-default", "h2o-vlen-default"],
    )
)
def test_grace_period(device, name):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    vocab_size = 128

    params = KVCacheParams(
        max_batch_size=3,
        n_query_groups=4,
        cache_length=64,
        head_size=8,
        n_head=4,
        dtype=torch.bfloat16,
    )
    cache_length = params.cache_length
    grace_period = 13
    kv_cache = create_kv_cache(name, params, grace_period=grace_period)
    num_insert = random.randint(cache_length, 3 * cache_length)
    num_prefill = min(random.randint(num_insert // 3, int(num_insert * 0.75)), kv_cache.max_prefill_length)

    keys, values = random_keys_values(params, num=num_insert)
    queries = random_tensor(params, num=num_insert, is_query=True)
    token_idx = torch.randint(
        low=0,
        high=vocab_size,
        size=(params.max_batch_size, num_insert),
    )
    kv_cache(
        query=queries[:, :, :num_prefill, :],
        key=keys[:, :, :num_prefill, :],
        value=values[:, :, :num_prefill, :],
        token_idx=token_idx[:, :num_prefill],
    )
    for pos in range(num_prefill, num_insert):
        kv_cache(
            query=queries[:, :, pos:(pos + 1), :],
            key=keys[:, :, pos:(pos + 1), :],
            value=values[:, :, pos:(pos + 1), :],
            token_idx=token_idx[:, pos:(pos + 1)],
        )
        if pos >= cache_length:
            prefix = cache_length - grace_period
            token_positions = kv_cache.token_positions()
            # Positions in grace region
            assert tensor_is_simple(token_positions[:, :, prefix:])
            pos_in = token_positions[0, 0, prefix:].tolist()
            assert len(set(pos_in)) == grace_period
            assert all(pos - grace_period < x <= pos for x in pos_in), (pos - grace_period, pos, pos_in)
            # Positions outside grace region
            pos_out = token_positions[:, :, :prefix].flatten().tolist()
            pos_out = list(set(pos_out))
            assert all(0 <= x <= pos - grace_period for x in pos_out)


def compute_scores(
    queries: torch.Tensor,
    k_and_v: DefaultKeysAndValues,
    num_prefill: int,
    v_length: bool,
) -> torch.Tensor:
    T, head_size = queries.shape[-2:]
    scores = torch.zeros(
        queries.shape[:-1], dtype=torch.float32, device=queries.device,
    )
    keys = k_and_v.keys()
    values = k_and_v.values()
    for pos in range(num_prefill, T):
        k_and_v_red = DefaultKeysAndValues(
            keys=keys[:, :, :(pos + 1), :],
            values=values[:, :, :(pos + 1), :],
        )
        _attn_weights = compute_attn_weights(
            query=queries[:, :, pos:(pos + 1), :],
            k_and_v=k_and_v_red,
        )
        if v_length:
            v_norm = vector_norm(k_and_v_red.values(), dim=-1, dtype=torch.float32)
            _attn_weights *= v_norm
            _attn_weights /= _attn_weights.sum(dim=-1, keepdim=True)
        scores[:, :, :(pos + 1)] += _attn_weights
    return scores


@pytest.mark.parametrize(
    "device, name, dtype",
    product(
        available_backends(),
        ["h2o-default", "h2o-vlen-default"],
        [torch.float32, torch.bfloat16, torch.float16],
    ),
)
def test_h2o_scores(device, name, dtype):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    vocab_size = 128

    v_length = "vlen" in name
    params = KVCacheParams(
        max_batch_size=3,
        n_query_groups=4,
        cache_length=64,
        head_size=8,
        n_head=4,
        dtype=dtype,
    )
    cache_length = params.cache_length
    kv_cache = create_kv_cache(name, params)
    num_prefill = min(random.randint(cache_length // 3, cache_length // 2), kv_cache.max_prefill_length)
    step = (cache_length - num_prefill) // 3
    test_positions = (
        num_prefill, num_prefill + step, num_prefill + 2 * step, cache_length - 1
    )

    keys, values = random_keys_values(params, num=cache_length)
    queries = random_tensor(params, num=cache_length, is_query=True)
    token_idx = torch.randint(
        low=0,
        high=vocab_size,
        size=(params.max_batch_size, cache_length),
    )
    kv_cache(
        query=queries[:, :, :num_prefill, :],
        key=keys[:, :, :num_prefill, :],
        value=values[:, :, :num_prefill, :],
        token_idx=token_idx[:, :num_prefill],
    )
    print(f"test_positions: {test_positions}\nnum_prefill = {num_prefill}, v_length={v_length}, cache_length={cache_length}")
    for pos in range(num_prefill, cache_length - 1):
        kv_cache(
            query=queries[:, :, pos:(pos + 1), :],
            key=keys[:, :, pos:(pos + 1), :],
            value=values[:, :, pos:(pos + 1), :],
            token_idx=token_idx[:, pos:(pos + 1)],
        )
        # Note: `kv_cache.scores` are normalized only once the cache is full!
        if pos in test_positions:
            # Compare scores
            other = compute_scores(
                queries[:, :, :(pos + 1), :],
                k_and_v=DefaultKeysAndValues(
                    keys[:, :, :(pos + 1), :], values[:, :, :(pos + 1), :],
                ),
                num_prefill=num_prefill,
                v_length=v_length,
            )
            print(f"pos={pos}")
            torch.testing.assert_close(
                kv_cache.scores[:, :, :(pos + 1)],
                other,
                rtol=0.005,
                atol=0.005,
            )


@pytest.mark.parametrize(
    "device, dtype",
    product(
        available_backends(),
        [torch.float32, torch.bfloat16, torch.float16],
    ),
)
def test_token_pos_and_pos_log(device, dtype):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    vocab_size = 128
    replay_log_blocksize = 10
    name = "h2o-default"

    params = KVCacheParams(
        max_batch_size=3,
        n_query_groups=4,
        cache_length=64,
        head_size=8,
        n_head=4,
        dtype=dtype,
    )
    cache_length = params.cache_length
    shape = (params.max_batch_size, params.n_query_groups)
    kv_cache = create_kv_cache(name, params, replay_log_blocksize=replay_log_blocksize)
    kv_cache.switch_replay_logging(True)
    num_insert = cache_length + 5
    num_prefill = cache_length - 2
    data = random_args_cache_forward(
        params, num=num_insert, vocab_size=vocab_size, device=device,
    )
    token_chunks = []
    # Prefill up to cache_length - 2
    data_part = range_from_args(data, 0, num_prefill)
    kv_cache(**data_part)
    token_chunks.append(data_part["token_idx"])
    # Checks
    assert kv_cache.input_pos == num_prefill
    other = torch.arange(
        0, num_prefill, dtype=torch.int, device=device,
    ).view(1, 1, -1).expand(*shape, -1)
    torch.testing.assert_close(kv_cache.token_pos[:, :, :num_prefill], other)
    # Try to insert too large of a piece
    assert kv_cache.max_forward_length <= cache_length - num_prefill
    pos = num_prefill
    num = cache_length - num_prefill + 1
    with pytest.raises(ValueError):
        kv_cache(**range_from_args(data, pos, pos + num))
    # Insert to fill up the cache
    num = cache_length - num_prefill
    assert kv_cache.max_forward_length == num
    data_part = range_from_args(data, pos, pos + num)
    kv_cache(**data_part)
    k_and_v = kv_cache.get_keys_values()
    token_chunks.append(data_part["token_idx"])
    pos = cache_length
    # Checks
    assert kv_cache.input_pos == pos
    other = torch.arange(
        0, pos, dtype=torch.int, device=device,
    ).view(1, 1, -1).expand(*shape, -1)
    torch.testing.assert_close(kv_cache.token_pos[:, :, :pos], other)
    torch.testing.assert_close(k_and_v.keys(), data["key"][:, :, :pos, :])
    torch.testing.assert_close(k_and_v.values(), data["value"][:, :, :pos, :])
    # Test eviction decisions based on scores
    scores = compute_scores(
        data["query"][:, :, :pos, :],
        k_and_v=k_and_v,
        num_prefill=num_prefill,
        v_length=False,
    )
    torch.testing.assert_close(kv_cache.scores[:, :, :pos], scores)
    next_positions = scores.argsort(dim=-1)
    torch.testing.assert_close(next_positions, kv_cache.next_positions(pos))
    # Insert chunk and evict due to scores
    num = 4
    data_part = range_from_args(data, pos, pos + num)
    kv_cache(**data_part)
    k_and_v = kv_cache.get_keys_values()
    token_chunks.append(data_part["token_idx"])
    # Checks
    assert kv_cache.input_pos == pos + num
    keys_expected = data["key"][:, :, :cache_length, :].clone()
    values_expected = data["value"][:, :, :cache_length, :].clone()
    index = expand_index(next_positions[:, :, :num], params.head_size)
    keys_expected.scatter_(-2, index, data["key"][:, :, pos:(pos + num), :])
    values_expected.scatter_(-2, index, data["value"][:, :, pos:(pos + num), :])
    torch.testing.assert_close(k_and_v.keys(), keys_expected)
    torch.testing.assert_close(k_and_v.values(), values_expected)
    # Check slot position log
    replay_log = kv_cache.get_replay_log()
    assert len(replay_log) == kv_cache.input_pos
    assert len(replay_log.token_chunks) == len(token_chunks)
    for c1, c2 in zip(token_chunks, replay_log.token_chunks):
        torch.testing.assert_close(c1, c2)
    assert len(replay_log.slot_positions) == 1
    dtype = replay_log.slot_positions[0].dtype
    block = next_positions[:, :, :num].to(dtype=dtype)
    torch.testing.assert_close(
        replay_log.slot_positions[0].to(device=device), block,
    )


def test_max_chunk_size():
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)

    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)  # Set default dtype

    cache_name = "h2o-default"
    batch_size = 3
    n_head = 8
    n_query_groups = 4
    head_size = 64
    vocab_size = 48
    num_repeats = 15
    _config = Config(
        n_layer=1,
        n_head=n_head,
        n_query_groups=n_query_groups,
        n_embd=n_head * head_size,
        block_size=128,
        vocab_size=vocab_size,
        rotary_percentage=1,
    )

    for repeat in range(num_repeats):
        cache_length = random.randint(128, 256)
        chunk_size = random.randint(8, cache_length // 2)
        num_chunks = random.randint(2, 8)
        seq_length = cache_length + (num_chunks - 2) * chunk_size + random.randint(1, chunk_size)
        max_chunk_size = chunk_size
        print(f"\nrepeat {repeat}: cache_length = {cache_length}, chunk_size = {chunk_size}, seq_length = {seq_length}, num_chunks = {num_chunks}")
        # Create model and KV cache
        config = replace(_config, block_size=seq_length)
        params = KVCacheParams.from_config(
            config=config,
            max_batch_size=batch_size,
            cache_length=cache_length,
            dtype=dtype,
        )
        gpt_models = []
        for _ in range(2):
            gpt_model = GPT(config)
            if not gpt_models:
                gpt_model.apply(gpt_model._init_weights)  # Initialization
                cache_kwargs = dict()
            else:
                copy_parameters(gpt_models[0], gpt_model)
                cache_kwargs = dict(max_chunk_size=max_chunk_size)
            kv_cache = create_kv_cache(
                name=cache_name,
                params=params,
                block_idx=0,
                **cache_kwargs,
            )
            kv_cache.debug_next_positions = []
            gpt_model.assign_kv_caches([kv_cache])
            gpt_models.append(gpt_model)
        head_model_name = CrossEntropyOnLogits.NAME
        head_model = HeadModelFactory.create(
            name=head_model_name, config=config,
        )
        models = [
            # Do not deallocate KV cache buffers after forward
            LongContextInferenceModel(
                gpt_model=gpt_model,
                head_model=head_model,
                chunk_size=chunk_size,
                debug_no_deallocate_buffers=True,
            )
            for gpt_model in gpt_models
        ]
        # Create data
        token_ids = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(batch_size, seq_length),
        )
        input_ids = token_ids[:, :-1]
        targets = token_ids[:, 1:]

        # Compute forward for both, compare
        loss_values = [model(input_ids, targets) for model in models]
        print("Loss values")
        torch.testing.assert_close(loss_values[0], loss_values[1])
        print("next_positions used per chunk")
        kv_caches = [model.gpt_model.get_kv_caches()[0] for model in models]
        next_pos = [c.debug_next_positions for c in kv_caches]
        assert(len(next_pos[0]) == len(next_pos[1]))
        for i, (np0, np1) in enumerate(zip(next_pos[0], next_pos[1])):
            print(f"Chunk {i}")
            assert(np0.shape == np1.shape)
            torch.testing.assert_close(np0, np1)
        print("Cache buffer contents")
        k_and_vs = [c.kv_buffers.get_keys_values() for c in kv_caches]
        torch.testing.assert_close(k_and_vs[0].keys(), k_and_vs[1].keys())
        torch.testing.assert_close(k_and_vs[0].values(), k_and_vs[1].values())
