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
Tests for :mod:`keys_values.logprobs`.

Verifies that ``chunked_per_token_logps`` integrates correctly with the
KeysAndValues model infrastructure (GPT + KV caches + MHA) and produces
correct per-token log-probabilities.
"""
import pytest
import torch

from keys_values.config import Config
from keys_values.logprobs import chunked_per_token_logps
from keys_values.model import GPT


def _small_config(**overrides) -> Config:
    """Minimal GPT config for fast CPU tests."""
    defaults = dict(
        n_layer=2,
        n_head=4,
        n_embd=64,
        n_query_groups=2,
        block_size=512,
        vocab_size=256,
        padded_vocab_size=256,
        intermediate_size=128,
    )
    defaults.update(overrides)
    return Config(**defaults)


def _reference_logps(
    gpt_model: GPT,
    input_ids: torch.Tensor,
    logits_to_keep: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ground truth: use chunked_per_token_logps with dense cache == seq_length.

    With a dense cache large enough for the full sequence, there's no
    eviction — this gives exact log-probs as if we did a single forward pass.
    """
    return chunked_per_token_logps(
        gpt_model=gpt_model,
        input_ids=input_ids,
        logits_to_keep=logits_to_keep,
        cache_name="dense-default",
        cache_length=input_ids.shape[1],
        chunk_size=input_ids.shape[1],  # single chunk = full forward
        temperature=temperature,
        compute_entropy=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq_length, logits_to_keep, chunk_size",
    [
        (32, 8, 16),
        (64, 16, 8),
        (48, 12, 32),
        (100, 30, 20),
    ],
)
def test_chunked_matches_dense_full(seq_length, logits_to_keep, chunk_size):
    """Chunked processing with dense cache should match single-chunk reference.

    Both use a dense cache (no eviction), so the only difference is how many
    chunks the sequence is split into. Results must be identical.
    """
    torch.manual_seed(42)
    config = _small_config()
    batch_size = 2

    with torch.device("cpu"):
        model = GPT(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

    # Reference: dense cache, single chunk (full forward)
    ref_logps, ref_ent = _reference_logps(model, input_ids, logits_to_keep)

    # Under test: dense cache but split into multiple chunks
    test_logps, test_ent = chunked_per_token_logps(
        gpt_model=model,
        input_ids=input_ids,
        logits_to_keep=logits_to_keep,
        cache_name="dense-default",
        cache_length=seq_length,
        chunk_size=chunk_size,
        compute_entropy=True,
    )

    torch.testing.assert_close(test_logps, ref_logps, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(test_ent, ref_ent, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_temperature_scaling(temperature):
    """Temperature should consistently scale logits before softmax."""
    torch.manual_seed(123)
    config = _small_config()
    seq_length, logits_to_keep = 40, 10

    with torch.device("cpu"):
        model = GPT(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, seq_length))

    ref_logps, _ = _reference_logps(
        model, input_ids, logits_to_keep, temperature
    )
    test_logps, _ = chunked_per_token_logps(
        gpt_model=model,
        input_ids=input_ids,
        logits_to_keep=logits_to_keep,
        cache_name="dense-default",
        cache_length=seq_length,
        chunk_size=16,
        temperature=temperature,
    )

    torch.testing.assert_close(test_logps, ref_logps, atol=1e-4, rtol=1e-4)


def test_no_entropy_returns_none():
    """When compute_entropy=False, entropies should be None."""
    torch.manual_seed(7)
    config = _small_config()

    with torch.device("cpu"):
        model = GPT(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 32))

    _, ent = chunked_per_token_logps(
        gpt_model=model,
        input_ids=input_ids,
        logits_to_keep=8,
        cache_name="dense-default",
        cache_length=32,
        chunk_size=16,
        compute_entropy=False,
    )
    assert ent is None


def test_caches_cleaned_up():
    """After chunked_per_token_logps, KV caches should be removed."""
    torch.manual_seed(99)
    config = _small_config()

    with torch.device("cpu"):
        model = GPT(config)
    model.eval()

    assert model.get_kv_caches()[0] is None

    input_ids = torch.randint(0, config.vocab_size, (1, 32))
    chunked_per_token_logps(
        gpt_model=model,
        input_ids=input_ids,
        logits_to_keep=8,
        cache_name="dense-default",
        cache_length=32,
        chunk_size=16,
    )

    assert model.get_kv_caches()[0] is None


def test_logps_shape():
    """Output shapes should match (batch_size, logits_to_keep)."""
    torch.manual_seed(55)
    config = _small_config()
    batch_size, seq_length, logits_to_keep = 3, 50, 15

    with torch.device("cpu"):
        model = GPT(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

    logps, ent = chunked_per_token_logps(
        gpt_model=model,
        input_ids=input_ids,
        logits_to_keep=logits_to_keep,
        cache_name="dense-default",
        cache_length=seq_length,
        chunk_size=20,
        compute_entropy=True,
    )

    assert logps.shape == (batch_size, logits_to_keep)
    assert ent.shape == (batch_size, logits_to_keep)


def test_logps_are_negative():
    """Log-probabilities should always be <= 0."""
    torch.manual_seed(77)
    config = _small_config()

    with torch.device("cpu"):
        model = GPT(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 40))

    logps, _ = chunked_per_token_logps(
        gpt_model=model,
        input_ids=input_ids,
        logits_to_keep=10,
        cache_name="dense-default",
        cache_length=40,
        chunk_size=16,
    )

    assert (logps <= 0).all(), f"Found positive log-probs: {logps.max()}"
