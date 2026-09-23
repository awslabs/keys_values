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
Tests for :class:`keys_values.long_context.LongContextInferenceModel`.

On a dense (lossless) KV cache, the chunked forward must be exactly
equivalent to a plain causal `GPT.forward`. In particular, the final
layer norm `transformer.ln_f` and logit softcapping must be applied
before the head, see
https://github.com/awslabs/keys_values/issues/139
"""

from typing import Optional

import pytest
import torch

from keys_values.config import Config
from keys_values.head_model import CrossEntropyOnLogits, HeadModel
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT

SEQ_LENGTH = 48
NUM_TARGETS = 16


def _small_config(final_logit_softcapping: Optional[float] = None) -> Config:
    return Config(
        n_layer=2,
        n_head=4,
        n_embd=64,
        n_query_groups=2,
        block_size=512,
        vocab_size=256,
        padded_vocab_size=256,
        intermediate_size=128,
        final_logit_softcapping=final_logit_softcapping,
    )


def _make_model_and_data(config: Config):
    torch.random.manual_seed(31415927)
    with torch.device("cpu"):
        gpt_model = GPT(config)
        gpt_model.apply(gpt_model._init_weights)
    gpt_model.eval()
    token_ids = torch.randint(0, config.vocab_size, (2, SEQ_LENGTH))
    return gpt_model, token_ids


def _plain_forward(gpt_model: GPT, input_ids: torch.Tensor) -> torch.Tensor:
    """Reference: plain causal forward without KV caches."""
    gpt_model.assign_kv_caches([None] * gpt_model.config.n_layer)
    gpt_model.max_seq_length = input_ids.shape[1]
    with torch.no_grad():
        return gpt_model(input_ids)


def _assign_dense_caches(gpt_model: GPT, batch_size: int, cache_length: int):
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            gpt_model=gpt_model,
            name="dense-default",
            max_batch_size=batch_size,
            cache_length=cache_length,
            dtype=torch.float32,
            cache_kwargs={},
        )
    )


@pytest.mark.parametrize("final_logit_softcapping", [None, 5.0])
@pytest.mark.parametrize("chunk_size", [8, 64])
def test_loss_matches_plain_forward(final_logit_softcapping, chunk_size):
    """
    Cross-entropy loss from the chunked forward (logits head path) must
    equal the loss computed from plain `GPT.forward` logits.
    """
    config = _small_config(final_logit_softcapping)
    gpt_model, token_ids = _make_model_and_data(config)
    input_ids = token_ids[:, :-1]
    targets = token_ids[:, -NUM_TARGETS:]

    logits = _plain_forward(gpt_model, input_ids)
    losses_ref = (
        torch.nn.functional.cross_entropy(
            logits[:, -NUM_TARGETS:, :].reshape(-1, config.padded_vocab_size),
            targets.reshape(-1),
            reduction="none",
        )
        .view(*targets.shape)
        .sum(dim=-1)
        / NUM_TARGETS
    )

    _assign_dense_caches(gpt_model, input_ids.shape[0], input_ids.shape[1])
    model = LongContextInferenceModel(
        gpt_model=gpt_model,
        head_model=CrossEntropyOnLogits(config),
        chunk_size=chunk_size,
    )
    with torch.no_grad():
        losses = model(input_ids, targets)

    torch.testing.assert_close(losses, losses_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("final_logit_softcapping", [None, 5.0])
@pytest.mark.parametrize("chunk_size", [8, 64])
def test_prefill_logits_match_plain_forward(final_logit_softcapping, chunk_size):
    """
    Generation prefill (`targets=None`) must return the same final-position
    logits as plain `GPT.forward`. This is where the first generated token
    of each sequence is sampled from.
    """
    config = _small_config(final_logit_softcapping)
    gpt_model, token_ids = _make_model_and_data(config)

    logits_ref = _plain_forward(gpt_model, token_ids)[:, -1:, :]

    _assign_dense_caches(gpt_model, token_ids.shape[0], token_ids.shape[1])
    model = LongContextInferenceModel(
        gpt_model=gpt_model,
        head_model=None,
        chunk_size=chunk_size,
    )
    with torch.no_grad():
        logits = model(token_ids, targets=None)

    torch.testing.assert_close(logits, logits_ref, atol=1e-5, rtol=1e-5)


class _RecordingHead(HeadModel):
    """
    Head model with `needs_logits() == False` which records the model
    outputs passed to it.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.n_embd = config.n_embd
        self.output_chunks = []

    def needs_logits(self) -> bool:
        return False

    def forward(
        self,
        model_outputs: torch.Tensor,
        targets: Optional[torch.Tensor],
        input_pos: int,
    ) -> torch.Tensor:
        assert model_outputs.shape[-1] == self.n_embd
        self.output_chunks.append(model_outputs)
        return torch.zeros(model_outputs.shape[0])

    def num_target_entries(self, targets: torch.Tensor) -> Optional[torch.Tensor]:
        return None


@pytest.mark.parametrize("chunk_size", [8, 64])
def test_non_logits_head_gets_normed_outputs(chunk_size):
    """
    Head models with `needs_logits() == False` must receive final layer
    outputs with `transformer.ln_f` applied, as returned by
    `GPT.forward(skip_lm_head=True)`.
    """
    config = _small_config()
    gpt_model, token_ids = _make_model_and_data(config)
    targets = token_ids[:, -NUM_TARGETS:]

    gpt_model.assign_kv_caches([None] * config.n_layer)
    gpt_model.max_seq_length = token_ids.shape[1]
    with torch.no_grad():
        outputs_ref = gpt_model(token_ids, skip_lm_head=True)

    head_model = _RecordingHead(config)
    _assign_dense_caches(gpt_model, token_ids.shape[0], token_ids.shape[1])
    model = LongContextInferenceModel(
        gpt_model=gpt_model,
        head_model=head_model,
        chunk_size=chunk_size,
    )
    with torch.no_grad():
        model(token_ids, targets)

    outputs = torch.cat(head_model.output_chunks, dim=1)
    torch.testing.assert_close(outputs, outputs_ref, atol=1e-5, rtol=1e-5)
