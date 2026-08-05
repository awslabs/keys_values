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
Smoke tests for the vLLM bridge scaffold (task 1).

The torch-free parts (import, config parsing) run anywhere. The registration
part requires vLLM and is skipped where vLLM is not installed.
"""

import importlib
import subprocess
import sys

import pytest


def test_import_bridge_without_vllm():
    """`import keys_values.vllm` must succeed even without vLLM installed."""
    module = importlib.import_module("keys_values.vllm")
    assert hasattr(module, "parse_policy")
    assert hasattr(module, "PolicyConfig")


def test_bridge_does_not_import_litgpt():
    """The bridge must stay litgpt-free so it can live in a vLLM serving env.

    This is checked in a fresh interpreter via a subprocess: asserting against
    the in-process ``sys.modules`` would be unreliable, since other tests in the
    same session legitimately import litgpt and leave it cached there.
    """
    script = (
        "import sys, importlib;"
        "importlib.import_module('keys_values.vllm.config');"
        "importlib.import_module('keys_values.vllm.registration');"
        "sys.exit('litgpt' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "importing the vLLM bridge pulled in litgpt:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_parse_policy_lastrec():
    from keys_values.vllm import parse_policy

    cfg = parse_policy("LastRec", cache_length=1024)
    assert cfg.name == "lastrec"
    assert cfg.cache_length == 1024
    assert cfg.recent_window == 0
    assert cfg.grace_tokens == 0


def test_parse_policy_h2o_with_params():
    from keys_values.vllm import parse_policy

    cfg = parse_policy(
        "h2o", cache_length=2048, recent_window=256, grace_tokens=64, foo="bar"
    )
    assert cfg.name == "h2o"
    assert cfg.recent_window == 256
    assert cfg.grace_tokens == 64
    assert cfg.extra == {"foo": "bar"}


def test_parse_policy_rejects_unknown_name():
    from keys_values.vllm import parse_policy

    with pytest.raises(ValueError):
        parse_policy("nonexistent", cache_length=128)


def test_parse_policy_rejects_window_exceeding_cache():
    from keys_values.vllm import parse_policy

    with pytest.raises(ValueError):
        parse_policy("h2o", cache_length=100, recent_window=80, grace_tokens=40)


def test_parse_policy_rejects_nonpositive_cache_length():
    from keys_values.vllm import parse_policy

    with pytest.raises(ValueError):
        parse_policy("lastrec", cache_length=0)


def test_register_policies_runs_when_vllm_available():
    """With vLLM installed, registration must run without error and be
    idempotent. Skipped where vLLM is absent (e.g. local dev machines)."""
    pytest.importorskip("vllm")
    from keys_values.vllm import registration

    # First call may register 0 policies (none implemented yet in task 1);
    # the key assertion is that it does not raise and is idempotent.
    first = registration.register_policies()
    second = registration.register_policies()
    assert isinstance(first, int)
    assert second == 0  # idempotent: nothing new on the second call
