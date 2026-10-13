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
"""Round-trip regression tests for KV-cache state and buffer switching.

This exercises the state classes reintroduced from #89, including eviction
scores and grace-period metadata that tentative decoding itself does not use.
"""

import copy
from itertools import product

import pytest
import torch

from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.test_utils import (
    available_backends,
    create_kv_cache,
    random_args_cache_forward,
    range_from_args,
)

CACHE_CASES = [
    ("dense-default", 0),
    ("lastrec-default", 0),
    ("h2o-default", 0),
    ("h2o-vlen-default", 0),
    ("h2o-orig-default", 0),
    ("qh2o-vlen-torch-quantized8", 0),
    ("dense-torch-quantized8", 0),
    ("lastrec-torch-quantized8", 0),
    ("h2o-torch-quantized8", 0),
    ("qh2o-torch-quantized8", 0),
    ("h2o-orig-torch-quantized8", 0),
    ("h2o-default", 4),
    ("h2o-vlen-default", 4),
    ("qh2o-torch-quantized8", 4),
]

STATE_ATTRS = (
    "input_pos",
    "current_length",
    "next_position",
    "next_grace_pos",
    "prefill_length",
    "token_pos",
    "_next_positions",
    "scores",
)


def _snapshot(cache):
    result = {}
    for attr in STATE_ATTRS:
        if hasattr(cache, attr):
            value = getattr(cache, attr)
            result[attr] = value.clone() if isinstance(value, torch.Tensor) else value
    kv = cache.get_keys_values()
    n = cache.current_length
    result["keys"] = kv.keys()[:, :, :n].clone()
    result["values"] = kv.values()[:, :, :n].clone()
    return result


def _assert_same(actual, expected):
    assert actual.keys() == expected.keys()
    for attr in actual:
        a, b = actual[attr], expected[attr]
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            assert isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor), attr
            torch.testing.assert_close(a, b, atol=0, rtol=0, msg=attr)
        else:
            assert a == b, attr


def _forward(cache, data, sizes):
    outputs = []
    offset = 0
    for size in sizes:
        outputs.append(cache(**range_from_args(data, offset, offset + size)).clone())
        offset += size
    return outputs


@pytest.mark.parametrize(
    "device, case, tentative_sizes",
    list(
        product(
            available_backends(),
            CACHE_CASES,
            ((1, 1, 3, 2), (2, 1, 2, 2)),
        )
    ),
)
def test_switch_buffers_restores_state_and_outputs(device, case, tentative_sizes):
    """Restoring state is bit-exact, even after eviction and repeated restore.

    This catches omissions from subclasses' state objects (e.g. H2O scores or
    token positions), not just incorrectly copied K/V buffer contents.
    """
    name, grace_period = case
    cache_length = 64 if name.startswith("dense") else 16
    params = KVCacheParams(
        max_batch_size=2,
        n_query_groups=2,
        cache_length=cache_length,
        head_size=8,
        n_head=4,
        dtype=torch.float32,
    )
    torch.manual_seed(0)
    pre = random_args_cache_forward(
        params, cache_length + 24, vocab_size=64, device=device
    )
    tentative = random_args_cache_forward(params, 12, vocab_size=64, device=device)
    further = random_args_cache_forward(params, 12, vocab_size=64, device=device)
    kwargs = {"grace_period": grace_period} if grace_period else {}
    cache = create_kv_cache(name, params, **kwargs)
    n_pre = 20 if name.startswith("dense") else cache_length
    cache(**range_from_args(pre, 0, n_pre))
    pos = n_pre
    for size in (1, 2, 1, 3):
        cache(**range_from_args(pre, pos, pos + size))
        pos += size

    reference = copy.deepcopy(cache)
    original = _snapshot(reference)
    saved_state = cache.get_state()
    saved_buffers = copy.deepcopy(cache.kv_buffers)
    second_buffers = copy.deepcopy(cache.kv_buffers)

    _forward(cache, tentative, tentative_sizes)
    assert _snapshot(cache)["input_pos"] > original["input_pos"]
    cache.switch_buffers(saved_buffers, cache_state=saved_state)
    _assert_same(_snapshot(cache), original)

    # Reusing the same state object must work; restoring must not mutate it.
    second_cache = copy.deepcopy(cache)
    _forward(second_cache, tentative, (2, 2))
    second_cache.switch_buffers(second_buffers, cache_state=saved_state)
    _assert_same(_snapshot(second_cache), original)

    expected_outputs = _forward(reference, further, (1, 2, 1, 4))
    actual_outputs = _forward(cache, further, (1, 2, 1, 4))
    for actual, expected in zip(actual_outputs, expected_outputs):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    _assert_same(_snapshot(cache), _snapshot(reference))
