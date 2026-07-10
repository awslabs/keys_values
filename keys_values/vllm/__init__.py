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
Bridge between keys_values' selective KV cache policies and vLLM (V1 engine).

This package is torch-only and MUST NOT import litgpt: it is meant to run inside
a vLLM serving environment. vLLM itself is imported lazily (inside functions),
so ``import keys_values.vllm`` succeeds even where vLLM is not installed; the
registration entry points require vLLM at call time.

See ``.kiro/specs/vllm-keys-values-integration/`` for the design.
"""

from keys_values.vllm.config import PolicyConfig, parse_policy

__all__ = ["PolicyConfig", "parse_policy"]
