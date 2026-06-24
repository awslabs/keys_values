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
H2O score feedback channel and eviction selection (task 4.1).

The attention path produces per-KV-position summed attention weights (see
``keys_values.vllm.attention``). H2O aggregates those into per-block scores and,
when a request exceeds its block budget, evicts the lowest-scoring blocks that
are neither in the protected recent window nor in the grace prefix.

This module is the pure, engine-free policy core (torch-light) so it can be unit
tested in milliseconds. Wiring it into vLLM's manager/attention is task 4.2/4.3.

Design properties (see the spec):
- Property 1: bounded footprint (eviction brings block count to the budget).
- Property 2: recent-window retention (recent blocks are never evicted).
- Property 3: score monotonicity (accumulated block scores only increase).
- Property 4: eviction selects the lowest-scoring eligible blocks.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

try:  # torch is available in the integration env; keep import optional for tooling
    import torch

    _TensorLike = Union[Sequence[float], "torch.Tensor"]
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    _TensorLike = Sequence[float]


def _to_float_list(position_scores: _TensorLike) -> List[float]:
    if torch is not None and isinstance(position_scores, torch.Tensor):
        if position_scores.ndim != 1:
            raise ValueError("position_scores tensor must be 1D")
        return position_scores.detach().to(torch.float32).cpu().tolist()
    return [float(x) for x in position_scores]


class BlockScoreTracker:
    """Accumulates per-block H2O scores keyed by ``(request_id, block_id)``.

    Scores are running sums of attention mass landing on a block's tokens. Since
    attention weights are non-negative, accumulated scores are non-decreasing
    until a block is freed (Property 3).
    """

    def __init__(self, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self._scores: Dict[str, Dict[int, float]] = {}

    def accumulate(
        self,
        request_id: str,
        position_scores: _TensorLike,
        block_ids: Sequence[int],
    ) -> None:
        """Add this step's per-position weights into per-block running sums.

        Args:
            request_id: The request.
            position_scores: 1D, length ``L`` = number of KV positions; entry
                ``j`` is the attention mass KV position ``j`` received this step
                (summed over query positions and heads).
            block_ids: ``block_ids[i]`` is the block holding KV positions
                ``[i*block_size, (i+1)*block_size)``. Length covers ``L``.
        """
        scores = self._scores.setdefault(request_id, {})
        values = _to_float_list(position_scores)
        n = len(values)
        for i, bid in enumerate(block_ids):
            start = i * self.block_size
            if start >= n:
                break
            block_sum = sum(values[start : start + self.block_size])
            scores[bid] = scores.get(bid, 0.0) + block_sum

    def block_score(self, request_id: str, block_id: int) -> float:
        return self._scores.get(request_id, {}).get(block_id, 0.0)

    def free_blocks(self, request_id: str, block_ids: Sequence[int]) -> None:
        scores = self._scores.get(request_id)
        if scores is None:
            return
        for bid in block_ids:
            scores.pop(bid, None)

    def free_request(self, request_id: str) -> None:
        self._scores.pop(request_id, None)


def select_evictions(
    block_ids: Sequence[int],
    tracker: BlockScoreTracker,
    request_id: str,
    *,
    max_blocks: int,
    recent_window_blocks: int,
    grace_blocks: int = 0,
) -> List[int]:
    """Choose which blocks to evict so the request fits its block budget.

    Args:
        block_ids: The request's blocks, ordered oldest -> newest.
        tracker: Score source.
        request_id: The request.
        max_blocks: Target maximum number of blocks to retain.
        recent_window_blocks: Number of newest blocks that are never evicted.
        grace_blocks: Number of oldest blocks that are never evicted.

    Returns:
        The ``block_ids`` to evict (lowest-scoring eligible first). Empty if the
        request is within budget. Never includes grace or recent blocks; if the
        budget cannot be met without evicting protected blocks, evicts as many
        eligible blocks as possible (footprint may exceed ``max_blocks`` then).
    """
    n = len(block_ids)
    num_to_evict = n - max_blocks
    if num_to_evict <= 0:
        return []
    lo = max(0, grace_blocks)
    hi = n - max(0, recent_window_blocks)
    eligible = list(block_ids[lo:hi]) if hi > lo else []
    if not eligible:
        return []
    eligible_sorted = sorted(eligible, key=lambda b: tracker.block_score(request_id, b))
    return eligible_sorted[: min(num_to_evict, len(eligible_sorted))]
