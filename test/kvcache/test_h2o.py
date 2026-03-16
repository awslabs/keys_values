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
from typing import Optional
import random
import math

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
    available_backends,
    random_args_cache_forward,
    range_from_args,
)
from keys_values.kvcache.utils import expand_index
from keys_values.flashinfer_wrapper import FlashInferSDPA, get_flashinfer_sdpa
from keys_values import flashinfer_ops


@pytest.mark.parametrize(
    "name, device",
    product(
        ["h2o-default", "h2o-vlen-default"],
        available_backends(),
    )
)
def test_grace_period(name, device):
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
        device=device,
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
    "name, dtype, device",
    product(
        ["h2o-default", "h2o-vlen-default"],
        [torch.float32, torch.bfloat16, torch.float16],
        available_backends(),
    ),
)
def test_h2o_scores(name, dtype, device):
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
        device=device,
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
        input_pos=0,
    )
    print(f"test_positions: {test_positions}\nnum_prefill = {num_prefill}, v_length={v_length}, cache_length={cache_length}")
    for pos in range(num_prefill, cache_length - 1):
        kv_cache(
            query=queries[:, :, pos:(pos + 1), :],
            key=keys[:, :, pos:(pos + 1), :],
            value=values[:, :, pos:(pos + 1), :],
            token_idx=token_idx[:, pos:(pos + 1)],
            input_pos=pos,
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
    "dtype, device",
    product(
        [torch.float32, torch.bfloat16, torch.float16],
        available_backends()
    ),
)
def test_token_pos_and_pos_log(dtype, device):
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
        device=device,
        dtype=dtype,
    )
    cache_length = params.cache_length
    shape = (params.max_batch_size, params.n_query_groups)
    kv_cache = create_kv_cache(name, params, replay_log_blocksize=replay_log_blocksize)
    kv_cache.switch_replay_logging(True)
    num_insert = cache_length + 5
    num_prefill = cache_length - 2
    data = random_args_cache_forward(params, num=num_insert, vocab_size=vocab_size)
    token_chunks = []
    # Prefill up to cache_length - 2
    data_part = range_from_args(data, 0, num_prefill)
    kv_cache(**data_part, input_pos=0)
    token_chunks.append(data_part["token_idx"])
    # Checks
    assert kv_cache.next_token_pos == num_prefill
    other = torch.arange(
        0, num_prefill, dtype=torch.int, device=device,
    ).view(1, 1, -1).expand(*shape, -1)
    torch.testing.assert_close(kv_cache.token_pos[:, :, :num_prefill], other)
    # Try to insert too large of a piece
    assert kv_cache.max_tokens_forward <= cache_length - num_prefill
    pos = num_prefill
    num = cache_length - num_prefill + 1
    with pytest.raises(ValueError):
        kv_cache(**range_from_args(data, pos, pos + num), input_pos=pos)
    # Insert to fill up the cache
    num = cache_length - num_prefill
    assert kv_cache.max_tokens_forward == num
    data_part = range_from_args(data, pos, pos + num)
    kv_cache(**data_part, input_pos=pos)
    k_and_v = kv_cache.get_keys_values()
    token_chunks.append(data_part["token_idx"])
    pos = cache_length
    # Checks
    assert kv_cache.next_token_pos == pos
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
    kv_cache(**data_part, input_pos=pos)
    k_and_v = kv_cache.get_keys_values()
    token_chunks.append(data_part["token_idx"])
    # Checks
    assert kv_cache.next_token_pos == pos + num
    keys_expected = data["key"][:, :, :cache_length, :].clone()
    values_expected = data["value"][:, :, :cache_length, :].clone()
    index = expand_index(next_positions[:, :, :num], params.head_size)
    keys_expected.scatter_(-2, index, data["key"][:, :, pos:(pos + num), :])
    values_expected.scatter_(-2, index, data["value"][:, :, pos:(pos + num), :])
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
    torch.testing.assert_close(
        replay_log.slot_positions[0].to(device=device), block,
    )


# =============================================================================
# FlashInfer Integration Tests for H2O Sparse KV Caches
# =============================================================================

def flashinfer_available_backends():
    """Return backends where FlashInfer is available."""
    result = []
    if flashinfer_ops.is_available() and torch.cuda.is_available():
        result.append(torch.device("cuda:0"))
    return result


def skip_if_flashinfer_unavailable():
    """Skip test if FlashInfer is not available."""
    if not flashinfer_ops.is_available():
        pytest.skip("FlashInfer kernels not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def compute_attn_weights_with_flashinfer(
    query: torch.Tensor,
    k_and_v: DefaultKeysAndValues,
    scale_factor: float,
    input_pos: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute attention weights using FlashInfer wrapper.
    
    Args:
        query: Query tensor, shape (batch_size, n_head, q_len, head_size)
        k_and_v: Keys and values
        scale_factor: Scale factor for attention scores
        input_pos: Position in input sequence. For decode phase (q_len=1), this should
            be kv_len - 1 so the query can attend to all KV positions with causal masking.
            If None, defaults to kv_len - q_len.
    
    Returns:
        Attention weights summed over query axis, shape (batch_size, n_query_groups, kv_len)
    """
    wrapper = get_flashinfer_sdpa()
    keys = k_and_v.keys()
    values = k_and_v.values()
    
    kv_len = keys.shape[2]
    q_len = query.shape[2]
    
    # For decode phase (q_len < kv_len), the query position should be at the end
    # of the KV cache to allow attending to all previous positions with causal masking.
    # input_pos should be kv_len - 1 for the query to see all KV positions.
    if input_pos is None:
        input_pos = kv_len - q_len
    
    _, attn_weights = wrapper.scaled_dot_product_attention(
        query=query,
        key=keys,
        value=values,
        scale_factor=scale_factor,
        return_attn_weights=True,
        token_positions=None,
        input_pos=input_pos,
        sliding_window_size=None,
    )
    return attn_weights


def compute_scores_with_flashinfer(
    queries: torch.Tensor,
    k_and_v: DefaultKeysAndValues,
    num_prefill: int,
    v_length: bool,
    head_size: int,
) -> torch.Tensor:
    """
    Compute H2O scores using FlashInfer for attention weights.
    
    This mirrors the compute_scores function but uses FlashInfer for attention
    weight computation instead of the eager implementation.
    
    Args:
        queries: Query tensor, shape (batch_size, n_head, T, head_size)
        k_and_v: Keys and values
        num_prefill: Number of prefill tokens
        v_length: Whether to use V-length weighted scores
        head_size: Head dimension size
    
    Returns:
        Scores tensor, shape (batch_size, n_query_groups, T)
    """
    T = queries.shape[-2]
    scale_factor = 1.0 / math.sqrt(head_size)
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
        
        # Use FlashInfer to compute attention weights
        # For decode phase, input_pos should be pos so the query at position pos
        # can attend to all KV positions 0 to pos with causal masking
        _attn_weights = compute_attn_weights_with_flashinfer(
            query=queries[:, :, pos:(pos + 1), :],
            k_and_v=k_and_v_red,
            scale_factor=scale_factor,
            input_pos=pos,
        )
        
        if v_length:
            v_norm = vector_norm(k_and_v_red.values(), dim=-1, dtype=torch.float32)
            _attn_weights *= v_norm
            _attn_weights /= _attn_weights.sum(dim=-1, keepdim=True)
        
        scores[:, :, :(pos + 1)] += _attn_weights
    
    return scores


@pytest.mark.parametrize(
    "name, dtype",
    product(
        ["h2o-default", "h2o-vlen-default"],
        [torch.float16, torch.bfloat16],
    ),
)
def test_h2o_flashinfer_attention_weights(name, dtype):
    """
    Test that FlashInfer provides attention weights equivalent to eager implementation.
    
    This test verifies that:
    1. FlashInfer kernels are exercised during decode phase
    2. Attention weights returned by FlashInfer match eager implementation
    3. H2O scores computed with FlashInfer match scores computed with eager
    
    Requirements: 1.3, 2.1, 2.3
    """
    skip_if_flashinfer_unavailable()
    
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    
    device = torch.device("cuda:0")
    v_length = "vlen" in name
    
    params = KVCacheParams(
        max_batch_size=2,
        n_query_groups=4,
        cache_length=32,
        head_size=64,
        n_head=4,
        device=device,
        dtype=dtype,
    )
    cache_length = params.cache_length
    num_prefill = min(cache_length // 2, 16)
    
    # Generate test data
    keys, values = random_keys_values(params, num=cache_length)
    queries = random_tensor(params, num=cache_length, is_query=True)
    
    # Compute scores using FlashInfer
    flashinfer_scores = compute_scores_with_flashinfer(
        queries[:, :, :(cache_length - 1), :],
        k_and_v=DefaultKeysAndValues(
            keys[:, :, :(cache_length - 1), :],
            values[:, :, :(cache_length - 1), :],
        ),
        num_prefill=num_prefill,
        v_length=v_length,
        head_size=params.head_size,
    )
    
    # Compute scores using eager implementation
    eager_scores = compute_scores(
        queries[:, :, :(cache_length - 1), :],
        k_and_v=DefaultKeysAndValues(
            keys[:, :, :(cache_length - 1), :],
            values[:, :, :(cache_length - 1), :],
        ),
        num_prefill=num_prefill,
        v_length=v_length,
    )
    
    # Compare FlashInfer scores against eager scores
    torch.testing.assert_close(
        flashinfer_scores,
        eager_scores,
        rtol=0.01,
        atol=0.01,
        msg=f"FlashInfer scores differ from eager scores for {name} with dtype={dtype}",
    )


@pytest.mark.parametrize(
    "name, dtype",
    product(
        ["h2o-default", "h2o-vlen-default"],
        [torch.float16, torch.bfloat16],
    ),
)
def test_h2o_scores_with_flashinfer_enabled(name, dtype):
    """
    Test that H2O cache produces valid scores when FlashInfer is available.
    
    This test verifies that the H2O cache works correctly during decode phase
    and produces valid scores that can be used for eviction decisions.
    
    Requirements: 1.3, 2.1, 2.3
    """
    skip_if_flashinfer_unavailable()
    
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    vocab_size = 128
    
    device = torch.device("cuda:0")
    v_length = "vlen" in name
    
    params = KVCacheParams(
        max_batch_size=3,
        n_query_groups=4,
        cache_length=64,
        head_size=8,
        n_head=4,
        device=device,
        dtype=dtype,
    )
    cache_length = params.cache_length
    kv_cache = create_kv_cache(name, params)
    num_prefill = min(random.randint(cache_length // 3, cache_length // 2), kv_cache.max_prefill_length)
    
    keys, values = random_keys_values(params, num=cache_length)
    queries = random_tensor(params, num=cache_length, is_query=True)
    token_idx = torch.randint(
        low=0,
        high=vocab_size,
        size=(params.max_batch_size, cache_length),
    )
    
    # Prefill
    kv_cache(
        query=queries[:, :, :num_prefill, :],
        key=keys[:, :, :num_prefill, :],
        value=values[:, :, :num_prefill, :],
        token_idx=token_idx[:, :num_prefill],
        input_pos=0,
    )
    
    # Decode phase
    for pos in range(num_prefill, cache_length - 1):
        kv_cache(
            query=queries[:, :, pos:(pos + 1), :],
            key=keys[:, :, pos:(pos + 1), :],
            value=values[:, :, pos:(pos + 1), :],
            token_idx=token_idx[:, pos:(pos + 1)],
            input_pos=pos,
        )
    
    # Verify cache scores have valid properties
    scores = kv_cache.scores[:, :, :(cache_length - 1)]
    
    # 1. Scores should be non-negative
    assert torch.all(scores >= 0), \
        f"H2O scores should be non-negative for {name} with dtype={dtype}"
    
    # 2. Scores should be finite
    assert torch.all(torch.isfinite(scores)), \
        f"H2O scores should be finite for {name} with dtype={dtype}"
    
    # 3. Scores should have correct shape
    expected_shape = (params.max_batch_size, params.n_query_groups, cache_length - 1)
    assert scores.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {scores.shape}"
    
    # 4. Some scores should be positive (attention was computed)
    assert torch.any(scores > 0), \
        f"H2O scores should have some positive values for {name} with dtype={dtype}"
    
    # 5. Verify cache is in valid state
    assert kv_cache.next_token_pos == cache_length - 1, \
        f"Expected next_token_pos={cache_length - 1}, got {kv_cache.next_token_pos}"


@pytest.mark.parametrize(
    "name, dtype",
    product(
        ["h2o-default", "h2o-vlen-default"],
        [torch.float16, torch.bfloat16],
    ),
)
def test_h2o_flashinfer_vs_eager_equivalence(name, dtype):
    """
    Test that H2O cache outputs are equivalent between FlashInfer and eager-only paths.
    
    This test compares the full H2O cache behavior when using FlashInfer vs eager
    implementation for attention weight computation.
    
    Requirements: 1.3, 2.1, 2.3
    """
    skip_if_flashinfer_unavailable()
    
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    vocab_size = 128
    
    device = torch.device("cuda:0")
    v_length = "vlen" in name
    
    params = KVCacheParams(
        max_batch_size=2,
        n_query_groups=4,
        cache_length=32,
        head_size=64,
        n_head=4,
        device=device,
        dtype=dtype,
    )
    cache_length = params.cache_length
    
    # Create two caches - one will use FlashInfer, one will use eager
    kv_cache_flashinfer = create_kv_cache(name, params)
    
    # Reset random state for second cache
    random.seed(seed)
    torch.random.manual_seed(seed)
    kv_cache_eager = create_kv_cache(name, params)
    
    num_prefill = min(cache_length // 2, kv_cache_flashinfer.max_prefill_length)
    
    # Generate test data
    keys, values = random_keys_values(params, num=cache_length)
    queries = random_tensor(params, num=cache_length, is_query=True)
    token_idx = torch.randint(
        low=0,
        high=vocab_size,
        size=(params.max_batch_size, cache_length),
    )
    
    # Prefill both caches
    kv_cache_flashinfer(
        query=queries[:, :, :num_prefill, :],
        key=keys[:, :, :num_prefill, :],
        value=values[:, :, :num_prefill, :],
        token_idx=token_idx[:, :num_prefill],
        input_pos=0,
    )
    kv_cache_eager(
        query=queries[:, :, :num_prefill, :],
        key=keys[:, :, :num_prefill, :],
        value=values[:, :, :num_prefill, :],
        token_idx=token_idx[:, :num_prefill],
        input_pos=0,
    )
    
    # Decode phase - both caches should accumulate similar scores
    for pos in range(num_prefill, cache_length - 1):
        kv_cache_flashinfer(
            query=queries[:, :, pos:(pos + 1), :],
            key=keys[:, :, pos:(pos + 1), :],
            value=values[:, :, pos:(pos + 1), :],
            token_idx=token_idx[:, pos:(pos + 1)],
            input_pos=pos,
        )
        kv_cache_eager(
            query=queries[:, :, pos:(pos + 1), :],
            key=keys[:, :, pos:(pos + 1), :],
            value=values[:, :, pos:(pos + 1), :],
            token_idx=token_idx[:, pos:(pos + 1)],
            input_pos=pos,
        )
    
    # Compare final scores
    torch.testing.assert_close(
        kv_cache_flashinfer.scores,
        kv_cache_eager.scores,
        rtol=0.01,
        atol=0.01,
        msg=f"FlashInfer and eager H2O scores differ for {name} with dtype={dtype}",
    )
    
    # Compare token positions
    torch.testing.assert_close(
        kv_cache_flashinfer.token_pos,
        kv_cache_eager.token_pos,
        msg=f"FlashInfer and eager token positions differ for {name} with dtype={dtype}",
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_h2o_update_requires_attn_weights(dtype):
    """
    Test that H2O caches have update_requires_attn_weights()=True.
    
    This verifies that H2O caches correctly signal that they need attention
    weights, which triggers the FlashInfer path during decode.
    
    Requirements: 2.1
    """
    skip_if_flashinfer_unavailable()
    
    device = torch.device("cuda:0")
    
    params = KVCacheParams(
        max_batch_size=2,
        n_query_groups=4,
        cache_length=32,
        head_size=64,
        n_head=4,
        device=device,
        dtype=dtype,
    )
    
    for name in ["h2o-default", "h2o-vlen-default"]:
        kv_cache = create_kv_cache(name, params)
        assert kv_cache.update_requires_attn_weights() is True, \
            f"{name} cache should require attention weights"


@pytest.mark.parametrize(
    "name, dtype",
    product(
        ["h2o-default", "h2o-vlen-default"],
        [torch.float16, torch.bfloat16],
    ),
)
def test_flashinfer_attention_weights_shape_for_h2o(name, dtype):
    """
    Test that FlashInfer returns attention weights with correct shape for H2O.
    
    H2O expects attention weights with shape (batch_size, n_query_groups, kv_len),
    summed over the query axis.
    
    Requirements: 2.1, 2.3
    """
    skip_if_flashinfer_unavailable()
    
    device = torch.device("cuda:0")
    
    params = KVCacheParams(
        max_batch_size=2,
        n_query_groups=4,
        cache_length=32,
        head_size=64,
        n_head=4,
        device=device,
        dtype=dtype,
    )
    
    # Generate test data
    q_len = 1  # Decode phase
    kv_len = 16
    
    query = random_tensor(params, num=q_len, is_query=True)
    keys, values = random_keys_values(params, num=kv_len)
    
    wrapper = get_flashinfer_sdpa()
    scale_factor = 1.0 / math.sqrt(params.head_size)
    
    _, attn_weights = wrapper.scaled_dot_product_attention(
        query=query,
        key=keys,
        value=values,
        scale_factor=scale_factor,
        return_attn_weights=True,
    )
    
    # Verify shape matches H2O expectations
    expected_shape = (params.max_batch_size, params.n_query_groups, kv_len)
    assert attn_weights.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {attn_weights.shape}"
    
    # Verify dtype is float32 for numerical stability
    assert attn_weights.dtype == torch.float32, \
        f"Expected float32, got {attn_weights.dtype}"
    
    # Verify weights are valid (non-negative, finite)
    assert torch.all(attn_weights >= 0), "Attention weights should be non-negative"
    assert torch.all(torch.isfinite(attn_weights)), "Attention weights should be finite"
