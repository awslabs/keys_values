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
Unit tests for the H2O score channel + eviction selection (task 4.1).

Engine-free; runs anywhere. Covers design Properties 1-4.
"""

from keys_values.vllm.scores import BlockScoreTracker, select_evictions

BS = 4  # block size


def test_accumulate_aggregates_positions_into_blocks():
    t = BlockScoreTracker(block_size=BS)
    # 3 blocks worth of positions (12); block ids 10, 11, 12.
    pos = [1.0] * 4 + [2.0] * 4 + [0.5] * 4
    t.accumulate("r", pos, [10, 11, 12])
    assert t.block_score("r", 10) == 4.0
    assert t.block_score("r", 11) == 8.0
    assert t.block_score("r", 12) == 2.0


def test_accumulate_is_monotonic_increasing():
    """Property 3: accumulated block scores only increase across steps."""
    t = BlockScoreTracker(block_size=BS)
    t.accumulate("r", [1.0] * 4, [10])
    first = t.block_score("r", 10)
    t.accumulate("r", [0.5] * 4, [10])
    second = t.block_score("r", 10)
    assert second >= first
    assert second == 6.0


def test_accumulate_handles_partial_trailing_block():
    t = BlockScoreTracker(block_size=BS)
    # Only 6 positions: block 10 full (4), block 11 partial (2).
    t.accumulate("r", [1.0] * 6, [10, 11])
    assert t.block_score("r", 10) == 4.0
    assert t.block_score("r", 11) == 2.0


def test_free_blocks_and_request_clear_scores():
    t = BlockScoreTracker(block_size=BS)
    t.accumulate("r", [1.0] * 8, [10, 11])
    t.free_blocks("r", [10])
    assert t.block_score("r", 10) == 0.0
    assert t.block_score("r", 11) == 4.0
    t.free_request("r")
    assert t.block_score("r", 11) == 0.0


def test_select_evictions_within_budget_returns_empty():
    t = BlockScoreTracker(block_size=BS)
    blocks = [1, 2, 3]
    assert select_evictions(blocks, t, "r", max_blocks=4, recent_window_blocks=1) == []


def test_select_evictions_picks_lowest_scoring_eligible():
    """Property 4: evict the lowest-scoring eligible blocks."""
    t = BlockScoreTracker(block_size=BS)
    blocks = [1, 2, 3, 4, 5, 6]
    # Give each block a distinct score.
    for bid, score in zip(blocks, [9, 1, 8, 2, 7, 3]):
        t.accumulate("r", [float(score)] + [0.0] * (BS - 1), [bid])
    # Budget 4 -> evict 2; recent_window keeps block 6; grace keeps block 1.
    evict = select_evictions(
        blocks, t, "r", max_blocks=4, recent_window_blocks=1, grace_blocks=1
    )
    # Eligible = [2,3,4,5] with scores [1,8,2,7]; lowest two -> 2 and 4.
    assert set(evict) == {2, 4}


def test_select_evictions_respects_recent_and_grace():
    """Property 2: recent and grace blocks are never evicted."""
    t = BlockScoreTracker(block_size=BS)
    blocks = [1, 2, 3, 4]
    # Make grace (block 1) and recent (block 4) the lowest-scoring.
    for bid, score in zip(blocks, [0.0, 5.0, 6.0, 0.0]):
        t.accumulate("r", [float(score)] + [0.0] * (BS - 1), [bid])
    evict = select_evictions(
        blocks, t, "r", max_blocks=3, recent_window_blocks=1, grace_blocks=1
    )
    assert 1 not in evict and 4 not in evict
    assert evict == [2]  # only [2,3] eligible; lowest is 2


def test_select_evictions_bounded_footprint():
    """Property 1: after eviction, retained block count == budget."""
    t = BlockScoreTracker(block_size=BS)
    blocks = list(range(20))
    for bid in blocks:
        t.accumulate("r", [float(bid)] + [0.0] * (BS - 1), [bid])
    max_blocks = 8
    evict = select_evictions(
        blocks, t, "r", max_blocks=max_blocks, recent_window_blocks=2, grace_blocks=1
    )
    retained = [b for b in blocks if b not in set(evict)]
    assert len(retained) == max_blocks
