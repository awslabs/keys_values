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
"""GRPO (Group Relative Policy Optimization) for KeysAndValues models.

Two entry points:

- :class:`keys_values.rl.grpo.trainer.GRPOLongContextTrainer` — a drop-in
  subclass of TRL's ``GRPOTrainer`` that routes long-sequence log-prob
  computation through the KeysAndValues KV cache.
- :func:`keys_values.rl.grpo.loop.grpo_step` — a standalone GRPO training
  step built entirely on KeysAndValues components (no TRL dependency).
"""
