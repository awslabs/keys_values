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
Policy configuration parsing for the vLLM bridge.

A policy is identified by a name (``lastrec``, ``h2o``, ...) plus parameters.
This mirrors keys_values' own ``kv_cache.name`` convention loosely, but is kept
deliberately small for the first integration slice (task 1). Policy *behavior*
(specs / managers) is added in later tasks; this module only parses and
validates configuration and is torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

# Policy names recognized by the bridge. Behavior is registered separately
# (see registration.py) as each policy is implemented.
SUPPORTED_POLICIES = ("lastrec", "h2o")


@dataclass(frozen=True)
class PolicyConfig:
    """Parsed configuration for a keys_values KV cache policy in vLLM.

    Attributes:
        name: Policy name, one of :data:`SUPPORTED_POLICIES`.
        cache_length: Target number of token slots to retain (the cache bound).
        recent_window: Number of most-recent tokens always retained. Only
            meaningful for score-based policies (h2o); 0 for purely positional
            policies (lastrec).
        grace_tokens: Number of initial tokens always retained (e.g. a system
            prompt). 0 means no grace region.
        extra: Any additional policy-specific keyword arguments, kept verbatim
            for forward compatibility.
    """

    name: str
    cache_length: int
    recent_window: int = 0
    grace_tokens: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_POLICIES:
            raise ValueError(
                f"unknown policy {self.name!r}; supported: {SUPPORTED_POLICIES}"
            )
        if self.cache_length <= 0:
            raise ValueError(f"cache_length must be positive, got {self.cache_length}")
        if self.recent_window < 0:
            raise ValueError(
                f"recent_window must be non-negative, got {self.recent_window}"
            )
        if self.grace_tokens < 0:
            raise ValueError(
                f"grace_tokens must be non-negative, got {self.grace_tokens}"
            )
        if self.recent_window + self.grace_tokens > self.cache_length:
            raise ValueError(
                "recent_window + grace_tokens "
                f"({self.recent_window} + {self.grace_tokens}) must not exceed "
                f"cache_length ({self.cache_length})"
            )


def parse_policy(name: str, cache_length: int, **kwargs: Any) -> PolicyConfig:
    """Build a :class:`PolicyConfig` from a policy name and parameters.

    Args:
        name: Policy name (case-insensitive), one of :data:`SUPPORTED_POLICIES`.
        cache_length: Target cache length (number of token slots).
        **kwargs: ``recent_window`` and/or ``grace_tokens`` are consumed; any
            other keyword arguments are preserved in :attr:`PolicyConfig.extra`.

    Returns:
        A validated :class:`PolicyConfig`.

    Raises:
        ValueError: If the name is unknown or parameters are invalid.
    """
    normalized = name.strip().lower()
    recent_window = int(kwargs.pop("recent_window", 0))
    grace_tokens = int(kwargs.pop("grace_tokens", 0))
    return PolicyConfig(
        name=normalized,
        cache_length=int(cache_length),
        recent_window=recent_window,
        grace_tokens=grace_tokens,
        extra=dict(kwargs),
    )
