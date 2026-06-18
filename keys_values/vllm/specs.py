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
vLLM V1 ``KVCacheSpec`` subclasses for keys_values policies.

This module imports vLLM and is therefore loaded lazily (only when policies are
registered), so that ``import keys_values.vllm`` stays vLLM-free.

``LastRecSpec`` (task 2): keys_values' ``lastrec`` keeps the most-recently
inserted ``cache_length`` tokens. That is behaviorally a sliding window of size
``cache_length``, so we subclass vLLM's ``SlidingWindowSpec`` to inherit its
page-size math and, crucially, its recycling-aware admission cap
(``get_manager_for_kv_cache_spec`` applies that cap to ``SlidingWindowSpec``
subclasses). It remains a distinct type so it registers its own manager and can
be extended for ``smart-lastrec`` (grace tokens) later.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.v1.kv_cache_interface import SlidingWindowSpec


@dataclass(frozen=True, kw_only=True)
class LastRecSpec(SlidingWindowSpec):
    """keys_values ``lastrec`` cache spec.

    Carries no new fields over ``SlidingWindowSpec``; ``sliding_window`` holds
    the keys_values ``cache_length``. Distinct type so the registry maps it to
    ``LastRecManager``.
    """


def build_lastrec_spec(
    *,
    cache_length: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
) -> LastRecSpec:
    """Build a :class:`LastRecSpec` from a cache length and layer geometry.

    Args:
        cache_length: Number of most-recent token slots to retain (maps to the
            sliding-window size).
        block_size: KV cache block size (tokens per block).
        num_kv_heads: Number of KV heads for the layer.
        head_size: Head dimension.
        dtype: KV cache element dtype.

    Returns:
        A :class:`LastRecSpec` instance.
    """
    if cache_length <= 0:
        raise ValueError(f"cache_length must be positive, got {cache_length}")
    return LastRecSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        sliding_window=cache_length,
    )
