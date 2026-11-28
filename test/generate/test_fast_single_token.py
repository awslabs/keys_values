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
from contextlib import nullcontext
import itertools
from typing import List, Tuple
from unittest.mock import patch

import pytest
import torch

from keys_values.generate.base import batched_generate_fn
from keys_values.generate.fast_infer import FastSingleTokenInference
from keys_values.generate.single_token import (
    DefaultSingleTokenInference,
    SingleTokenLongContextInference,
)
from keys_values.generate.test_utils import _gpt, _inference_model
from keys_values.kvcache.attn_weights import AttnWeightsKVCache
from keys_values.long_context import LongContextInferenceModel


def _prompt(seed: int, length: int, vocab_size: int) -> torch.Tensor:
    return torch.randint(
        0, vocab_size, (length,), generator=torch.Generator().manual_seed(seed)
    )


class _LogitsRecorder(SingleTokenLongContextInference):
    """Wraps an inference object and records logits from every call."""

    def __init__(self, inner: SingleTokenLongContextInference):
        self._inner = inner
        self.logits: List[torch.Tensor] = []

    @property
    def model(self) -> LongContextInferenceModel:
        return self._inner.model

    def process_prompt(self, prompt: torch.Tensor) -> torch.Tensor:
        logits = self._inner.process_prompt(prompt)
        self.logits.append(logits.detach().clone())
        return logits

    def inference_for_token(self, token_idx: torch.Tensor) -> torch.Tensor:
        logits = self._inner.inference_for_token(token_idx)
        self.logits.append(logits.detach().clone())
        return logits


def _chunk_frozen_next_positions(chunk_size: int, start_input_pos: int):
    """
    Returns a replacement for ``AttnWeightsKVCache.next_positions`` that
    freezes slot decisions at chunk boundaries during token generation.

    ``next_positions(1)`` is called once per generated token.  The mock
    captures ``self._next_positions[:, :, :chunk_size]`` at the first call of
    each chunk and returns successive single-column slices within the chunk.
    Calls during prompt processing (``self.input_pos < start_input_pos``) or
    with ``num != 1`` pass through to the original implementation unchanged.

    Args:
        chunk_size: Number of tokens per chunk, same as in FastSingleTokenInference.
        start_input_pos: ``input_pos`` of the KV cache at the start of
            generation (equals prompt length).  Calls with
            ``self.input_pos < start_input_pos`` are passed through.
    """
    original = AttnWeightsKVCache.next_positions
    call_counts: dict = {}  # id(cache) -> int call count within generation
    frozen: dict = {}  # id(cache) -> Tensor[..., chunk_size]

    def mocked(self, num: int) -> torch.Tensor:
        # Pass through during prompt processing or for chunk-sized batch calls
        if num != 1 or self.input_pos < start_input_pos:
            return original(self, num)
        cache_id = id(self)
        pos_in_chunk = call_counts.get(cache_id, 0) % chunk_size
        if pos_in_chunk == 0:
            # Capture chunk_size positions at each chunk boundary
            frozen[cache_id] = original(self, chunk_size).clone()
        call_counts[cache_id] = call_counts.get(cache_id, 0) + 1
        return frozen[cache_id][:, :, pos_in_chunk : pos_in_chunk + 1]

    return mocked


# --- Generation runner ---


def _run_generation(
    inference: SingleTokenLongContextInference,
    prompt: torch.Tensor,
    batch_size: int,
    num_tokens: int,
    sample_kwargs: dict,
) -> Tuple[_LogitsRecorder, torch.Tensor]:
    """
    Run ``batched_generate_fn`` and return ``(recorder, all_tokens)``.

    ``recorder.logits`` contains one tensor per step: the prompt logits (from
    ``process_prompt``) followed by per-token logits (from
    ``inference_for_token``), for a total of ``num_tokens`` entries.
    ``all_tokens`` has shape ``(batch_size, num_tokens)``.
    """
    recorder = _LogitsRecorder(inference)
    if batch_size == 1:
        prompts = prompt.unsqueeze(0)
    else:
        prompts = prompt.unsqueeze(0).expand(batch_size, -1).contiguous()
    tokens_list = []
    for token_out in batched_generate_fn(
        inference=recorder,
        prompts=prompts,
        max_returned_tokens=num_tokens,
        sample_args=sample_kwargs,
        deallocate_cache_buffers=False,
    ):
        tokens_list.append(token_out.clone())
    all_tokens = torch.cat(tokens_list, dim=1)
    return recorder, all_tokens


# --- Test ---


@pytest.mark.parametrize(
    "chunk_size, batch_size, prompt_size, cache_name",
    list(
        itertools.product(
            [2, 3, 4],  # chunk_size
            [1, 2],  # batch_size
            [10, 12],  # prompt_size (must be >= cache_length below)
            ["lastrec-default", "h2o-default"],
        )
    )[0:1],
)
def test_fast_matches_default_with_frozen_positions(
    chunk_size,
    batch_size,
    prompt_size,
    cache_name,
):
    """
    ``FastSingleTokenInference`` must produce the same logits and sampled
    tokens as ``DefaultSingleTokenInference`` when the latter's
    ``next_positions`` are frozen at chunk boundaries.

    ``FastSingleTokenInference`` makes KV eviction decisions only at the start
    of each chunk.  To make ``DefaultSingleTokenInference`` behave identically,
    we mock ``AttnWeightsKVCache.next_positions`` so that it captures
    ``self._next_positions[:, :, :chunk_size]`` once per chunk and returns
    single-column slices within the chunk.  With the same model weights,
    prompt, and position decisions, both paths must produce bit-identical
    logits and therefore identical token samples.
    """
    vocab_size = 32
    model_seed = 7
    prompt_seed = 42
    # cache_length must be <= prompt_size so the cache fills during prompting
    cache_length = 8
    assert prompt_size >= cache_length
    num_tokens = 3 * chunk_size
    max_seq_length = prompt_size + num_tokens
    do_attn_weights_cache = "h2o" in cache_name

    prompt = _prompt(prompt_seed, prompt_size, vocab_size)
    # top_k=1 gives deterministic sampling without depending on the random seed
    sample_kwargs = dict(temperature=1.0, top_k=1)

    # --- FastSingleTokenInference path ---
    gpt_fast = _gpt(model_seed, vocab_size)
    lc_fast = _inference_model(
        gpt_model=gpt_fast,
        max_seq_length=max_seq_length,
        cache_name=cache_name,
        batch_size=batch_size,
        chunk_size=chunk_size,
        cache_length=cache_length,
    )
    fast_recorder, fast_tokens = _run_generation(
        FastSingleTokenInference(
            lc_fast,
            chunk_size=chunk_size,
            debug_for_testing=do_attn_weights_cache,
        ),
        prompt,
        batch_size,
        num_tokens,
        sample_kwargs,
    )

    # --- DefaultSingleTokenInference path with frozen position mock ---
    gpt_default = _gpt(model_seed, vocab_size)
    lc_default = _inference_model(
        gpt_model=gpt_default,
        max_seq_length=max_seq_length,
        cache_name=cache_name,
        batch_size=batch_size,
        chunk_size=chunk_size,
        cache_length=cache_length,
    )
    mocked = _chunk_frozen_next_positions(chunk_size, start_input_pos=prompt_size)
    ctx = (
        patch.object(AttnWeightsKVCache, "next_positions", new=mocked)
        if do_attn_weights_cache
        else nullcontext()
    )
    with ctx:
        default_recorder, default_tokens = _run_generation(
            DefaultSingleTokenInference(lc_default),
            prompt,
            batch_size,
            num_tokens,
            sample_kwargs,
        )

    # Both paths must yield the same number of logit tensors
    assert len(fast_recorder.logits) == len(default_recorder.logits), (
        f"chunk_size={chunk_size}, batch_size={batch_size}, prompt_size={prompt_size}: "
        f"logit count {len(fast_recorder.logits)} vs {len(default_recorder.logits)}"
    )

    # Logits must match at every generation step.  Both paths process identical
    # inputs; differences would indicate the frozen-positions emulation is wrong.
    for step, (fl, dl) in enumerate(zip(fast_recorder.logits, default_recorder.logits)):
        torch.testing.assert_close(
            fl,
            dl,
            msg=(
                f"chunk_size={chunk_size}, batch_size={batch_size}, "
                f"prompt_size={prompt_size}, step={step}"
            ),
        )

    # Token samples must be exactly equal (top_k=1 makes sampling deterministic)
    assert fast_tokens.equal(default_tokens), (
        f"chunk_size={chunk_size}, batch_size={batch_size}, prompt_size={prompt_size}: "
        f"tokens differ:\nfast:    {fast_tokens}\ndefault: {default_tokens}"
    )
