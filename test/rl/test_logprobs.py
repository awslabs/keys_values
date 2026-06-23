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
Tests for :mod:`keys_values.rl.logprobs`.

Verifies that ``compute_logprobs`` produces correct per-token
log-probabilities using the LongContextInferenceModel infrastructure.
"""

import pytest
import torch

from keys_values.config import Config
from keys_values.rl.logprobs import compute_logprobs


def _small_config() -> Config:
    """Minimal GPT config for fast CPU tests."""
    return Config(
        n_layer=2,
        n_head=4,
        n_embd=64,
        n_query_groups=2,
        block_size=512,
        vocab_size=256,
        padded_vocab_size=256,
        intermediate_size=128,
    )


def _make_model():
    from keys_values.model import GPT

    config = _small_config()
    with torch.device("cpu"):
        model = GPT(config)
    model.eval()
    return model, config


@pytest.mark.parametrize(
    "seq_length, completion_length, chunk_size",
    [
        (32, 8, 16),
        (64, 16, 8),
        (48, 12, 32),
        (100, 30, 20),
    ],
)
def test_chunked_matches_single_chunk(seq_length, completion_length, chunk_size):
    """Chunked computation should match single-chunk (full forward) result."""
    torch.manual_seed(42)
    model, config = _make_model()
    batch_size = 2

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    targets = input_ids[:, -completion_length:]

    # Reference: dense cache, single large chunk (effectively full forward)
    ref_logps, ref_ent = compute_logprobs(
        gpt_model=model,
        input_ids=input_ids,
        targets=targets,
        cache_name="dense-default",
        cache_length=seq_length,
        chunk_size=seq_length,
        compute_entropy=True,
    )

    # Under test: same dense cache but split into multiple chunks
    test_logps, test_ent = compute_logprobs(
        gpt_model=model,
        input_ids=input_ids,
        targets=targets,
        cache_name="dense-default",
        cache_length=seq_length,
        chunk_size=chunk_size,
        compute_entropy=True,
    )

    torch.testing.assert_close(test_logps, ref_logps, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(test_ent, ref_ent, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_temperature_scaling(temperature):
    """Temperature should consistently scale logits."""
    torch.manual_seed(123)
    model, config = _make_model()

    input_ids = torch.randint(0, config.vocab_size, (1, 40))
    targets = input_ids[:, -10:]

    ref_logps, _ = compute_logprobs(
        gpt_model=model,
        input_ids=input_ids,
        targets=targets,
        cache_name="dense-default",
        cache_length=40,
        chunk_size=40,
        temperature=temperature,
    )
    test_logps, _ = compute_logprobs(
        gpt_model=model,
        input_ids=input_ids,
        targets=targets,
        cache_name="dense-default",
        cache_length=40,
        chunk_size=16,
        temperature=temperature,
    )

    torch.testing.assert_close(test_logps, ref_logps, atol=1e-4, rtol=1e-4)


def test_no_entropy_returns_none():
    """When compute_entropy=False, entropies should be None."""
    torch.manual_seed(7)
    model, config = _make_model()

    input_ids = torch.randint(0, config.vocab_size, (1, 32))
    targets = input_ids[:, -8:]

    _, ent = compute_logprobs(
        gpt_model=model,
        input_ids=input_ids,
        targets=targets,
        cache_name="dense-default",
        cache_length=32,
        chunk_size=16,
        compute_entropy=False,
    )
    assert ent is None


def test_caches_cleaned_up():
    """After compute_logprobs, KV caches should be removed."""
    torch.manual_seed(99)
    model, config = _make_model()

    assert model.get_kv_caches()[0] is None

    input_ids = torch.randint(0, config.vocab_size, (1, 32))
    targets = input_ids[:, -8:]

    compute_logprobs(
        gpt_model=model,
        input_ids=input_ids,
        targets=targets,
        cache_name="dense-default",
        cache_length=32,
        chunk_size=16,
    )

    assert model.get_kv_caches()[0] is None


def test_output_shapes():
    """Output shapes should match (batch_size, completion_length)."""
    torch.manual_seed(55)
    model, config = _make_model()
    batch_size, seq_length, completion_length = 3, 50, 15

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    targets = input_ids[:, -completion_length:]

    logps, ent = compute_logprobs(
        gpt_model=model,
        input_ids=input_ids,
        targets=targets,
        cache_name="dense-default",
        cache_length=seq_length,
        chunk_size=20,
        compute_entropy=True,
    )

    assert logps.shape == (batch_size, completion_length)
    assert ent.shape == (batch_size, completion_length)


def test_logps_are_negative():
    """Log-probabilities should always be <= 0."""
    torch.manual_seed(77)
    model, config = _make_model()

    input_ids = torch.randint(0, config.vocab_size, (2, 40))
    targets = input_ids[:, -10:]

    logps, _ = compute_logprobs(
        gpt_model=model,
        input_ids=input_ids,
        targets=targets,
        cache_name="dense-default",
        cache_length=40,
        chunk_size=16,
    )

    assert (logps <= 0).all(), f"Found positive log-probs: {logps.max()}"
