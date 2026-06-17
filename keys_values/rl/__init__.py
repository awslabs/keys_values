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
"""Reinforcement-learning fine-tuning for KeysAndValues models.

Currently provides GRPO (Group Relative Policy Optimization) under
:mod:`keys_values.rl.grpo`, plus the shared memory-efficient log-probability
utility :func:`keys_values.rl.logprobs.compute_logprobs`.
"""
