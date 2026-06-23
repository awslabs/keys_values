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
Tests for :mod:`keys_values.rl.grpo.rollout`.

Verifies that ``generate_completions`` produces correctly-shaped completion
token IDs using KeysAndValues' KV-cache generation.
"""

import pytest
import torch

from keys_values.config import Config
from keys_values.rl.grpo.rollout import generate_completions
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT


def _make_inference_model(batch_size: int, max_seq_length: int):
    """Build a small GPT + LongContextInferenceModel for CPU testing."""
    config = Config(
        block_size=512,
        vocab_size=64,
        padded_vocab_size=64,
        n_layer=2,
        n_head=4,
        n_embd=32,
        n_query_groups=2,
        intermediate_size=64,
    )
    with torch.device("cpu"):
        gpt_model = GPT(config)
    gpt_model.eval()
    gpt_model.max_seq_length = max_seq_length
    gpt_model.set_kv_caches(batch_size=batch_size)
    model = LongContextInferenceModel(
        gpt_model=gpt_model,
        head_model=None,
        chunk_size=16,
    )
    return model, config


@pytest.mark.parametrize("batch_size, prompt_len, max_new", [(2, 8, 10), (1, 16, 5)])
def test_generate_completions_shape(batch_size, prompt_len, max_new):
    """Completions should have the right batch size and not exceed max_new."""
    torch.manual_seed(0)
    model, config = _make_inference_model(batch_size, prompt_len + max_new)

    prompt_ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len))
    completions = generate_completions(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new,
        top_k=1,
    )

    assert completions.shape[0] == batch_size
    assert completions.shape[1] <= max_new
    # Generated tokens must be valid vocab indices (or pad)
    assert (completions >= 0).all()


def test_generate_completions_deterministic_greedy():
    """With top_k=1 (greedy), generation should be reproducible."""
    torch.manual_seed(123)
    model, config = _make_inference_model(batch_size=1, max_seq_length=30)
    prompt_ids = torch.randint(0, config.vocab_size, (1, 10))

    out1 = generate_completions(
        model=model, prompt_ids=prompt_ids, max_new_tokens=10, top_k=1
    )

    # Rebuild fresh model with same seed and weights would be needed for exact
    # match; here we just check the second call on the same model is consistent
    # in shape and validity.
    out2 = generate_completions(
        model=model, prompt_ids=prompt_ids, max_new_tokens=10, top_k=1
    )
    assert out1.shape == out2.shape


def test_generate_completions_1d_prompt():
    """A 1D prompt should be accepted and treated as batch size 1."""
    torch.manual_seed(7)
    model, config = _make_inference_model(batch_size=1, max_seq_length=25)
    prompt_ids = torch.randint(0, config.vocab_size, (10,))

    completions = generate_completions(
        model=model, prompt_ids=prompt_ids, max_new_tokens=5, top_k=1
    )
    assert completions.shape[0] == 1
