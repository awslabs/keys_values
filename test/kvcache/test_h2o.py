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
import random

import torch
from torch.linalg import vector_norm
import pytest

from keys_values.attention import DefaultKeysAndValues
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.test_utils import (
    compute_attn_weights,
    create_kv_cache,
    tensor_is_simple,
    random_keys_values,
    random_tensor,
)
from keys_values.kvcache.utils import expand_index


@pytest.mark.parametrize("name", ["h2o-default", "h2o-vlen-default"])
def test_grace_period(name):
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
        device=torch.device("cpu"),
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
        input_pos=0,
    )
    for pos in range(num_prefill, num_insert):
        kv_cache(
            query=queries[:, :, pos:(pos + 1), :],
            key=keys[:, :, pos:(pos + 1), :],
            value=values[:, :, pos:(pos + 1), :],
            token_idx=token_idx[:, pos:(pos + 1)],
            input_pos=pos,
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
    normalize_scores: bool = False,
) -> torch.Tensor:
    T, head_size = queries.shape[-2:]
    attn_weights = compute_attn_weights(query=queries, k_and_v=k_and_v)
    # Note that `attn_weights` is lower triangular
    if not v_length:
        scores = attn_weights[:, :, num_prefill:, :]
    else:
        v_norm = vector_norm(
            k_and_v.values(), dim=-1, dtype=torch.float32,
        ).unsqueeze(2).to(dtype=attn_weights.dtype)
        scores = attn_weights[:, :, num_prefill:, :] * v_norm
        scores = scores / scores.sum(dim=-1, keepdim=True)
    scores = scores.sum(dim=2).to(dtype=torch.float32)
    if normalize_scores:
        num_gen = T - num_prefill
        denom = torch.arange(
            T, 0, -1, device=queries.device, dtype=scores.dtype
        ).minimum(num_gen)
        scores = scores / denom.view(1, 1, -1)
    return scores


@pytest.mark.parametrize(
    "name, kwargs, v_length",
    [
        ("h2o-default", dict(normalize_scores=False), False),
        ("h2o-vlen-default", dict(normalize_scores=False), True),
    ]
)
def test_h2o_scores(name, kwargs, v_length):
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
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    cache_length = params.cache_length
    kv_cache = create_kv_cache(name, params, **kwargs)
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
        input_pos=0,
    )
    print(f"test_positions: {test_positions}\nnum_prefill = {num_prefill}")
    for pos in range(num_prefill, cache_length - 1):
        kv_cache(
            query=queries[:, :, pos:(pos + 1), :],
            key=keys[:, :, pos:(pos + 1), :],
            value=values[:, :, pos:(pos + 1), :],
            token_idx=token_idx[:, pos:(pos + 1)],
            input_pos=pos,
        )
        k_and_v = DefaultKeysAndValues(
            keys[:, :, :(pos + 1), :], values[:, :, :(pos + 1), :]
        )
        # Note: `kv_cache.scores` are normalized only once the cache is full!
        if pos in test_positions:
            # Compare scores
            other = compute_scores(
                queries[:, :, :(pos + 1), :],
                k_and_v=k_and_v,
                num_prefill=num_prefill,
                v_length=v_length,
            )
            print(f"v_length={v_length}, pos={pos}, cache_length={cache_length}")
            torch.testing.assert_close(
                kv_cache.scores[:, :, :(pos + 1)],
                other,
                rtol=0.005,
                atol=0.005,
            )


def test_token_pos_and_pos_log():
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
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    cache_length = params.cache_length
    shape = (params.max_batch_size, params.n_query_groups)
    kv_cache = create_kv_cache(name, params, replay_log_blocksize=replay_log_blocksize)
    kv_cache.switch_replay_logging(True)
    num_insert = cache_length + 5
    num_prefill = cache_length - 2
    keys, values = random_keys_values(params, num=num_insert)
    queries = random_tensor(params, num=num_insert, is_query=True)
    token_idxs = torch.randint(
        low=0,
        high=vocab_size,
        size=(params.max_batch_size, num_insert),
    )
    token_chunks = []
    # Prefill up to cache_length - 2
    token_idx = token_idxs[:, :num_prefill]
    kv_cache(
        query=queries[:, :, :num_prefill, :],
        key=keys[:, :, :num_prefill, :],
        value=values[:, :, :num_prefill, :],
        token_idx=token_idx,
        input_pos=0,
    )
    token_chunks.append(token_idx)
    # Checks
    assert kv_cache.next_token_pos == num_prefill
    other = torch.arange(
        0, num_prefill, dtype=torch.int
    ).view(1, 1, -1).expand(*shape, -1)
    torch.testing.assert_close(kv_cache.token_pos[:, :, :num_prefill], other)
    # Try to insert too large of a piece
    assert kv_cache.max_tokens_forward <= cache_length - num_prefill
    pos = num_prefill
    num = cache_length - num_prefill + 1
    with pytest.raises(ValueError):
        kv_cache(
            query=queries[:, :, pos:(pos + num), :],
            key=keys[:, :, pos:(pos + num), :],
            value=values[:, :, pos:(pos + num), :],
            token_idx=token_idxs[:, pos:(pos + num)],
            input_pos=pos,
        )
    # Insert to fill up the cache
    num = cache_length - num_prefill
    assert kv_cache.max_tokens_forward == num
    token_idx = token_idxs[:, pos:(pos + num)]
    kv_cache(
        query=queries[:, :, pos:(pos + num), :],
        key=keys[:, :, pos:(pos + num), :],
        value=values[:, :, pos:(pos + num), :],
        token_idx=token_idx,
        input_pos=pos,
    )
    k_and_v = kv_cache.get_keys_values()
    token_chunks.append(token_idx)
    pos = cache_length
    # Checks
    assert kv_cache.next_token_pos == pos
    other = torch.arange(
        0, pos, dtype=torch.int
    ).view(1, 1, -1).expand(*shape, -1)
    torch.testing.assert_close(kv_cache.token_pos[:, :, :pos], other)
    torch.testing.assert_close(k_and_v.keys(), keys[:, :, :pos, :])
    torch.testing.assert_close(k_and_v.values(), values[:, :, :pos, :])
    # Test eviction decisions based on scores
    scores = compute_scores(
        queries[:, :, :pos, :],
        k_and_v=k_and_v,
        num_prefill=num_prefill,
        v_length=False,
    )
    torch.testing.assert_close(kv_cache.scores[:, :, :pos], scores)
    next_positions = scores.argsort(dim=-1)
    torch.testing.assert_close(next_positions, kv_cache.next_positions(pos))
    # Insert chunk and evict due to scores
    num = 4
    token_idx = token_idxs[:, pos:(pos + num)]
    kv_cache(
        query=queries[:, :, pos:(pos + num), :],
        key=keys[:, :, pos:(pos + num), :],
        value=values[:, :, pos:(pos + num), :],
        token_idx=token_idx,
        input_pos=pos,
    )
    k_and_v = kv_cache.get_keys_values()
    token_chunks.append(token_idx)
    # Checks
    assert kv_cache.next_token_pos == pos + num
    keys_expected = keys[:, :, :cache_length, :].clone()
    values_expected = values[:, :, :cache_length, :].clone()
    index = expand_index(next_positions[:, :, :num], params.head_size)
    keys_expected.scatter_(-2, index, keys[:, :, pos:(pos + num), :])
    values_expected.scatter_(-2, index, values[:, :, pos:(pos + num), :])
    torch.testing.assert_close(k_and_v.keys(), keys_expected)
    torch.testing.assert_close(k_and_v.values(), values_expected)
    # Check slot position log
    replay_log = kv_cache.get_replay_log()
    assert len(replay_log) == kv_cache.next_token_pos
    assert len(replay_log.token_chunks) == len(token_chunks)
    for c1, c2 in zip(token_chunks, replay_log.token_chunks):
        torch.testing.assert_close(c1, c2)
    assert len(replay_log.slot_positions) == 1
    dtype = replay_log.slot_positions[0].dtype
    block = next_positions[:, :, :num].to(dtype=dtype)
    torch.testing.assert_close(replay_log.slot_positions[0], block)
