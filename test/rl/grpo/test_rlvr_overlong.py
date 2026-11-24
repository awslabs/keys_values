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
Overlong-rollout detection and penalty of the RLVR driver
(`examples/rl/grpo_rlvr.py --overlong-penalty`).
"""

import importlib.util
from pathlib import Path

import torch


def _load_driver():
    # `examples/` is not a package (and the name clashes with an installed
    # module), so load the driver by file path.
    path = Path(__file__).resolve().parents[3] / "examples" / "rl" / "grpo_rlvr.py"
    spec = importlib.util.spec_from_file_location("grpo_rlvr_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_driver = _load_driver()
is_overlong = _driver.is_overlong
overlong_reward = _driver.overlong_reward

PAD, EOS, MAX_NEW = 0, 2, 8


def test_overlong_row_uses_full_budget_without_eos():
    row = torch.tensor([5, 6, 7, 5, 6, 7, 5, 6])  # 8 tokens, no pad, no EOS
    assert is_overlong(row, PAD, MAX_NEW, EOS)
    assert is_overlong(row, PAD, MAX_NEW)  # eos_id unknown: same verdict


def test_row_that_stopped_early_is_not_overlong():
    row = torch.tensor([5, 6, 7, EOS, PAD, PAD, PAD, PAD])
    assert not is_overlong(row, PAD, MAX_NEW, EOS)
    # Shorter completions tensor (every row in the batch stopped early)
    assert not is_overlong(torch.tensor([5, 6, EOS]), PAD, MAX_NEW, EOS)


def test_row_ending_in_eos_at_the_cutoff_is_not_overlong():
    """The stop token is kept in the row, so a sequence that finishes exactly
    at the budget has `max_new_tokens` non-pad tokens but did stop."""
    row = torch.tensor([5, 6, 7, 5, 6, 7, 5, EOS])
    assert not is_overlong(row, PAD, MAX_NEW, EOS)


def test_penalty_subtracted_only_when_flag_positive():
    # Flag off (default 0.0): reward untouched, overlong or not
    assert overlong_reward(0.0, True, 0.0) == 0.0
    assert overlong_reward(1.0, True, 0.0) == 1.0
    # Flag on: truncated rollout scores -penalty, truncated-but-correct
    # 1 - penalty, finished rollouts are untouched
    assert overlong_reward(0.0, True, 0.5) == -0.5
    assert overlong_reward(1.0, True, 0.5) == 0.5
    assert overlong_reward(0.0, False, 0.5) == 0.0
    assert overlong_reward(1.0, False, 0.5) == 1.0
