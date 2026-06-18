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
Registration plumbing for keys_values KV cache policies in vLLM (V1).

vLLM maps a ``KVCacheSpec`` subclass to a ``SingleTypeKVCacheManager`` via
``KVCacheSpecRegistry.register(spec_cls, manager_cls, uniform_type_base_spec=)``,
and exposes a platform hook ``register_custom_kv_cache_specs(vllm_config)`` that
is the supported seam for third-party specs.

Task 1 establishes the entry point only: :func:`register_policies` is importable
and idempotent, imports vLLM lazily, and currently registers nothing because no
policy specs exist yet. Tasks 2 (lastrec) and 4 (H2O) fill in the
``_POLICY_REGISTRATIONS`` table; the call site does not change.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


# Populated by later tasks as (spec_cls, manager_cls, uniform_type_base_spec)
# tuples. Kept empty in task 1 so the plumbing is exercised end to end without
# any policy behavior yet.
#
#   from keys_values.vllm.specs import LastRecSpec
#   from keys_values.vllm.managers import LastRecManager
#   _POLICY_REGISTRATIONS.append((LastRecSpec, LastRecManager, LastRecSpec))
def _policy_registrations() -> List[Tuple[type, type, type]]:
    registrations: List[Tuple[type, type, type]] = []
    # Task 2: lastrec; Task 4: h2o. Imported lazily here once implemented.
    return registrations


# Guard so repeated calls (e.g. multiple workers) do not double-register.
_REGISTERED = False


def register_policies() -> int:
    """Register all implemented keys_values KV cache policies with vLLM.

    Idempotent: safe to call multiple times. Imports vLLM lazily so that
    ``import keys_values.vllm`` does not require vLLM to be installed.

    Returns:
        The number of policies registered on this call (0 if already registered
        or none implemented yet).

    Raises:
        ImportError: If vLLM is not importable when this is called.
    """
    global _REGISTERED
    if _REGISTERED:
        return 0

    try:
        from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("register_policies() requires vLLM to be installed.") from exc

    registrations = _policy_registrations()
    for spec_cls, manager_cls, base_spec in registrations:
        KVCacheSpecRegistry.register(
            spec_cls, manager_cls, uniform_type_base_spec=base_spec
        )
        logger.info("Registered keys_values KV cache policy: %s", spec_cls.__name__)

    _REGISTERED = True
    return len(registrations)


def register_custom_kv_cache_specs(vllm_config) -> None:
    """Adapter matching vLLM's platform hook signature.

    vLLM calls ``current_platform.register_custom_kv_cache_specs(vllm_config)``
    during engine init. Wiring keys_values into that hook (e.g. via a platform
    plugin) lets policies register without patching vLLM. The ``vllm_config``
    argument is currently unused but kept to match the expected signature.
    """
    register_policies()
