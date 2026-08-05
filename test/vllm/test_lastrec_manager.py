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
No-engine unit tests for ``LastRecManager`` (task 2.2).

These exercise the manager against vLLM's real (pure-Python, GPU-free)
``BlockPool``, so they validate the eviction bookkeeping in milliseconds without
a ~160s engine cold start. Skipped where vLLM is not installed.

Covers design Correctness Properties:
- Property 1 (bounded footprint)
- Property 5 (lastrec equals sliding window)
"""

import math

import pytest

vllm = pytest.importorskip("vllm")

import torch  # noqa: E402  (after importorskip so it is only needed with vLLM)

from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.single_type_kv_cache_manager import (  # noqa: E402
    SlidingWindowManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import SlidingWindowSpec  # noqa: E402

from keys_values.vllm.managers import LastRecManager  # noqa: E402
from keys_values.vllm.registration import register_policies  # noqa: E402
from keys_values.vllm.specs import LastRecSpec, build_lastrec_spec  # noqa: E402

BLOCK_SIZE = 16
NUM_KV_HEADS = 2
HEAD_SIZE = 64
CACHE_LENGTH = 64  # 4 blocks
NUM_GPU_BLOCKS = 256


def _make_pool() -> BlockPool:
    return BlockPool(
        num_gpu_blocks=NUM_GPU_BLOCKS,
        enable_caching=False,
        hash_block_size=BLOCK_SIZE,
    )


def _make_manager(cls, spec, pool):
    return cls(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=BLOCK_SIZE,
    )


def _lastrec_spec() -> LastRecSpec:
    return build_lastrec_spec(
        cache_length=CACHE_LENGTH,
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
    )


def _sliding_window_spec() -> SlidingWindowSpec:
    return SlidingWindowSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.bfloat16,
        sliding_window=CACHE_LENGTH,
    )


def test_lastrec_spec_is_sliding_window_subclass():
    assert issubclass(LastRecSpec, SlidingWindowSpec)
    spec = _lastrec_spec()
    assert spec.sliding_window == CACHE_LENGTH
    assert spec.block_size == BLOCK_SIZE


def test_get_num_skipped_tokens_matches_sliding_window():
    """Property 5 (core): lastrec's skip rule equals sliding window's."""
    pool = _make_pool()
    lastrec = _make_manager(LastRecManager, _lastrec_spec(), pool)
    swa = _make_manager(SlidingWindowManager, _sliding_window_spec(), pool)
    for n in range(0, 4 * CACHE_LENGTH):
        assert lastrec.get_num_skipped_tokens(n) == swa.get_num_skipped_tokens(n)


def test_bounded_footprint_single_request():
    """Property 1: live (non-null) blocks stay bounded by the window."""
    pool = _make_pool()
    mgr = _make_manager(LastRecManager, _lastrec_spec(), pool)
    req = "r0"
    max_blocks = math.ceil(CACHE_LENGTH / BLOCK_SIZE) + 1  # +1 boundary slack
    total = 8 * CACHE_LENGTH
    for num_tokens in range(1, total + 1):
        mgr.allocate_new_blocks(req, num_tokens, num_tokens)
        mgr.remove_skipped_blocks(req, num_tokens)
        live = sum(1 for b in mgr.req_to_blocks[req] if b is not mgr._null_block)
        assert live <= max_blocks, f"at {num_tokens} tokens: {live} > {max_blocks}"


def test_registry_routes_lastrec_spec_to_lastrec_manager():
    """Registration wires LastRecSpec -> LastRecManager end to end."""
    register_policies()
    pool = _make_pool()
    mgr = get_manager_for_kv_cache_spec(
        _lastrec_spec(),
        max_num_batched_tokens=8192,
        max_model_len=2048,
        block_pool=pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=BLOCK_SIZE,
    )
    assert isinstance(mgr, LastRecManager)
