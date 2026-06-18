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
