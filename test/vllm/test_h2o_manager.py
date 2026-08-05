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
"""
No-engine unit tests for H2OManager (task 4.2).

Drives the manager against vLLM's real (pure-Python) BlockPool: allocate blocks,
record per-position scores, run eviction, and assert the design properties.
Skipped where vLLM is not installed.
"""

import pytest

vllm = pytest.importorskip("vllm")

import torch  # noqa: E402

from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.single_type_kv_cache_manager import (  # noqa: E402
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec  # noqa: E402

from keys_values.vllm.managers import H2OManager  # noqa: E402
from keys_values.vllm.registration import register_policies  # noqa: E402
from keys_values.vllm.specs import H2OSpec, build_h2o_spec  # noqa: E402

BLOCK_SIZE = 4
NUM_GPU_BLOCKS = 256


def _pool():
    return BlockPool(
        num_gpu_blocks=NUM_GPU_BLOCKS, enable_caching=False, hash_block_size=BLOCK_SIZE
    )


def _spec(cache_length, recent_window, grace_tokens=0):
    return build_h2o_spec(
        cache_length=cache_length,
        recent_window=recent_window,
        grace_tokens=grace_tokens,
        block_size=BLOCK_SIZE,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
    )


def _manager(spec, pool):
    return H2OManager(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=BLOCK_SIZE,
    )


def _allocate(mgr, request_id, num_blocks):
    """Allocate num_blocks blocks for the request via the manager."""
    mgr.allocate_new_blocks(
        request_id, num_blocks * BLOCK_SIZE, num_blocks * BLOCK_SIZE
    )


def test_h2o_spec_is_full_attention_subclass():
    assert issubclass(H2OSpec, FullAttentionSpec)
    spec = _spec(cache_length=32, recent_window=8, grace_tokens=4)
    assert spec.cache_length == 32
    assert spec.recent_window == 8
    assert spec.grace_tokens == 4


def test_registry_routes_h2o_spec_to_h2o_manager():
    register_policies()
    pool = _pool()
    mgr = get_manager_for_kv_cache_spec(
        _spec(cache_length=32, recent_window=8),
        max_num_batched_tokens=8192,
        max_model_len=2048,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=BLOCK_SIZE,
    )
    assert isinstance(mgr, H2OManager)


def test_eviction_bounds_footprint_and_keeps_recent():
    """Properties 1, 2, 4: evict lowest-score non-recent blocks to the budget."""
    pool = _pool()
    # budget 4 blocks (cache_length 16), recent window 1 block (4 tokens).
    mgr = _manager(_spec(cache_length=16, recent_window=4), pool)
    req = "r0"
    n_blocks = 8
    _allocate(mgr, req, n_blocks)
    live = mgr._live_block_ids(req)
    assert len(live) == n_blocks
    ordered_ids = [b.block_id for b in live]

    # Give ascending scores by position: oldest blocks lowest score.
    # position_scores length = n_blocks * BLOCK_SIZE; block i gets score i.
    pos = []
    for i in range(n_blocks):
        pos += [float(i)] + [0.0] * (BLOCK_SIZE - 1)
    mgr.record_block_scores(req, pos)

    evicted = mgr.apply_eviction(req)
    retained = mgr._live_block_ids(req)
    # Budget 4 -> from 8, evict 4.
    assert len(retained) == 4
    # Recent block (newest) retained.
    assert ordered_ids[-1] in [b.block_id for b in retained]
    # The four lowest-scoring (oldest) eligible blocks evicted.
    assert set(evicted) == set(ordered_ids[:4])


def test_no_eviction_when_within_budget():
    pool = _pool()
    mgr = _manager(_spec(cache_length=32, recent_window=4), pool)
    req = "r1"
    _allocate(mgr, req, 4)  # 4 blocks <= budget 8
    mgr.record_block_scores(req, [1.0] * (4 * BLOCK_SIZE))
    assert mgr.apply_eviction(req) == []
    assert len(mgr._live_block_ids(req)) == 4


def test_grace_blocks_are_protected():
    """Property 2: grace prefix never evicted even if lowest-scoring."""
    pool = _pool()
    # budget 3 blocks (cache_length 12), recent 1 block, grace 1 block (4 tokens).
    mgr = _manager(_spec(cache_length=12, recent_window=4, grace_tokens=4), pool)
    req = "r2"
    _allocate(mgr, req, 6)
    ordered_ids = [b.block_id for b in mgr._live_block_ids(req)]
    # Make grace (block 0) the lowest score so it would be evicted if unprotected.
    pos = [0.0] * BLOCK_SIZE + [5.0] * (5 * BLOCK_SIZE)
    mgr.record_block_scores(req, pos)
    evicted = mgr.apply_eviction(req)
    assert ordered_ids[0] not in evicted  # grace protected
    assert ordered_ids[-1] not in evicted  # recent protected
    assert len(mgr._live_block_ids(req)) == 3


def test_free_clears_scores_and_blocks():
    pool = _pool()
    mgr = _manager(_spec(cache_length=16, recent_window=4), pool)
    req = "r3"
    _allocate(mgr, req, 4)
    mgr.record_block_scores(req, [1.0] * (4 * BLOCK_SIZE))
    mgr.free(req)
    assert req not in mgr.req_to_blocks
    assert mgr.scores._scores.get(req) is None
