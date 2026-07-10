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
vLLM V1 ``SingleTypeKVCacheManager`` subclasses for keys_values policies.

Imports vLLM; loaded lazily via the registration entry point.

``LastRecManager`` (task 2): identical eviction behavior to
``SlidingWindowManager`` — keep the most-recent ``cache_length`` (==
``sliding_window``) tokens, recycle older blocks. We subclass it to reuse its
carefully written ``remove_skipped_blocks`` / ``free`` / ``find_longest_cache_hit``
logic, and override only :meth:`get_num_skipped_tokens` to state the lastrec
rule explicitly. A distinct class lets the registry route ``LastRecSpec`` here
and gives smart-lastrec a place to extend.

Next step (see ``docs/vllm_integration.md`` phased plan): a
``SmartLastRecManager`` subclass for the ``smart-lastrec`` policy. It is the
first policy unique to keys_values that still needs no attention weights; its
eviction depends on the batch dimension (per-request initial / grace regions)
but not on head position, so it fits the paged block model (per-request block
tables) without per-head divergence.
"""

from __future__ import annotations

from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager


class LastRecManager(SlidingWindowManager):
    """Manager for the keys_values ``lastrec`` policy."""

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """Tokens outside the retained window are skipped for attention.

        lastrec retains the most-recent ``cache_length`` (== ``sliding_window``)
        tokens, so everything before that window is skipped. This is identical
        to the sliding-window rule and is stated explicitly here for clarity.
        """
        return max(0, num_computed_tokens - self.sliding_window + 1)


from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

from keys_values.vllm.scores import BlockScoreTracker, select_evictions


class H2OManager(FullAttentionManager):
    """Manager for the keys_values H2O policy (task 4.2).

    Holds a :class:`BlockScoreTracker` and, when a request exceeds its block
    budget, evicts the lowest-scoring blocks outside the protected recent window
    and grace prefix (delegating the choice to ``select_evictions``).

    Scope: this class owns the eviction *decision* and the score/block
    bookkeeping, which are unit-tested against vLLM's real ``BlockPool`` without
    the engine. Wiring the decision into the engine hot path (when to call
    eviction, and reflecting evicted "holes" in the block table + attention
    masking) is task 4.3 -- vLLM's built-in managers only ever free a contiguous
    prefix, so non-prefix eviction needs additional plumbing.
    """

    def __init__(self, kv_cache_spec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        # kv_cache_spec is an H2OSpec; read policy params defensively.
        self.cache_length = int(getattr(kv_cache_spec, "cache_length", 0))
        self.recent_window = int(getattr(kv_cache_spec, "recent_window", 0))
        self.grace_tokens = int(getattr(kv_cache_spec, "grace_tokens", 0))
        self.scores = BlockScoreTracker(block_size=self.block_size)

    def _budget_blocks(self) -> int:
        return cdiv(self.cache_length, self.block_size)

    def _recent_blocks(self) -> int:
        return cdiv(self.recent_window, self.block_size)

    def _grace_blocks(self) -> int:
        return self.grace_tokens // self.block_size

    def _live_block_ids(self, request_id: str):
        blocks = self.req_to_blocks.get(request_id, [])
        return [b for b in blocks if b is not self._null_block]

    def record_block_scores(self, request_id: str, position_scores) -> None:
        """Accumulate this step's per-KV-position attention mass into blocks.

        ``position_scores`` is length = current KV length; positions map to the
        request's live blocks in order.
        """
        live = self._live_block_ids(request_id)
        self.scores.accumulate(request_id, position_scores, [b.block_id for b in live])

    def select_blocks_to_evict(self, request_id: str):
        """Return the block ids to evict to fit the budget (may be empty)."""
        live = self._live_block_ids(request_id)
        return select_evictions(
            [b.block_id for b in live],
            self.scores,
            request_id,
            max_blocks=self._budget_blocks(),
            recent_window_blocks=self._recent_blocks(),
            grace_blocks=self._grace_blocks(),
        )

    def apply_eviction(self, request_id: str):
        """Free the selected blocks and null them in the block list.

        Returns the freed block ids. Mirrors how sliding window nulls freed
        slots; the resulting "holes" are what task 4.3 must reflect in the
        attention path.
        """
        evict_ids = self.select_blocks_to_evict(request_id)
        if not evict_ids:
            return []
        evict_set = set(evict_ids)
        blocks = self.req_to_blocks[request_id]
        to_free = [b for b in blocks if b.block_id in evict_set]
        self.block_pool.free_blocks(to_free)
        self.req_to_blocks[request_id] = [
            self._null_block if b.block_id in evict_set else b for b in blocks
        ]
        self.scores.free_blocks(request_id, evict_ids)
        return evict_ids

    def free(self, request_id: str) -> None:
        req_blocks = self.req_to_blocks.pop(request_id, [])
        non_null = [b for b in req_blocks if b is not self._null_block]
        self.block_pool.free_blocks(list(reversed(non_null)))
        self.num_cached_block.pop(request_id, None)
        self.scores.free_request(request_id)
