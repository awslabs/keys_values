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
import itertools
from typing import List, Optional, Tuple

import pytest
import torch

from keys_values.generate.base import batched_generate_fn
from keys_values.generate.fast_infer import FastSingleTokenInference
from keys_values.generate.single_token import (
    DefaultSingleTokenInference,
    SingleTokenLongContextInference,
)
from keys_values.generate.test_utils import _gpt, _inference_model
from keys_values.kvcache.basics import LastRecentlyInsertedKVCache
from keys_values.kvcache.h2o import H2OKVCache
from keys_values.kvcache.test_utils import product_with_devices
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT


def _prompt(
    seed: int,
    length: int,
    vocab_size: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    return torch.randint(
        0, vocab_size, (length,), generator=torch.Generator().manual_seed(seed), device=device,
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


# --- Helpers for _ChunkJointReference: replicate joint-pass semantics ---


def _save_cache_state(gpt: GPT) -> List[dict]:
    """
    Deep-copy the full KV state of every cache layer.
    Supports ``LastRecentlyInsertedKVCache`` and ``H2OKVCache``.
    """
    state = []
    for cache in gpt.get_kv_caches():
        kv = cache.get_keys_values()
        entry: dict = {
            "keys": kv.keys().clone(),
            "values": kv.values().clone(),
            "input_pos": cache._input_pos,
        }
        if isinstance(cache, LastRecentlyInsertedKVCache):
            entry["token_pos"] = cache.token_pos.clone()
            entry["next_position"] = cache.next_position
        elif isinstance(cache, H2OKVCache):
            entry["token_pos"] = cache.token_pos.clone()
            entry["scores"] = cache.scores.clone()
            entry["_next_positions"] = (
                cache._next_positions.clone()
                if cache._next_positions is not None
                else None
            )
            entry["_use_initial_rule"] = cache._use_initial_rule
            entry["next_grace_pos"] = cache.next_grace_pos
        else:
            raise ValueError(f"Unsupported cache type: {type(cache)}")
        state.append(entry)
    return state


def _restore_cache_state(gpt: GPT, state: List[dict]) -> None:
    """Restore KV state saved by ``_save_cache_state``."""
    for cache, s in zip(gpt.get_kv_caches(), state):
        kv = cache.get_keys_values()
        kv.keys().copy_(s["keys"])
        kv.values().copy_(s["values"])
        cache._input_pos = s["input_pos"]
        if isinstance(cache, LastRecentlyInsertedKVCache):
            cache.token_pos.copy_(s["token_pos"])
            cache.next_position = s["next_position"]
        elif isinstance(cache, H2OKVCache):
            cache.token_pos.copy_(s["token_pos"])
            cache.scores.copy_(s["scores"])
            if s["_next_positions"] is None:
                cache._next_positions = None
            else:
                cache._next_positions = s["_next_positions"].clone()
            cache._use_initial_rule = s["_use_initial_rule"]
            cache.next_grace_pos = s["next_grace_pos"]
        else:
            raise ValueError(f"Unsupported cache type: {type(cache)}")


class _ChunkJointReference(SingleTokenLongContextInference):
    """
    Reference that replicates the chunk-boundary behaviour of
    ``SingleTokenFastInferenceDriver.update()`` for both lastrec and h2o caches.

    Within each chunk it delegates to ``DefaultSingleTokenInference`` (sequential
    single-token calls), so the within-chunk logits match the compiled graph.
    At every chunk boundary it:
      1. Restores the KV state to what it was at the *start* of the completed chunk,
      2. Runs ``gpt_model(all_chunk_tokens)`` as a single joint forward pass —
         exactly what ``update()`` does,
      3. Saves the resulting state as the start of the next chunk.

    For ``H2OKVCache`` layers, ``_next_positions`` is additionally frozen within
    each chunk to the values captured from the most recent joint pass, matching
    the slot decisions the compiled graph uses.
    """

    def __init__(
        self,
        inner: SingleTokenLongContextInference,
        chunk_size: int,
    ):
        self._inner = inner
        self._chunk_size = chunk_size
        self._pending: List[torch.Tensor] = []
        self._pos_in_chunk: int = 0
        self._chunk_completed: bool = False
        self._chunk_start_state: Optional[List[dict]] = None
        self._gpt: GPT = inner.model.gpt_model
        self._frozen_positions: dict = {}  # id(cache) -> Tensor[B, G, chunk_size]

    @property
    def model(self) -> LongContextInferenceModel:
        return self._inner.model

    def _capture_frozen_h2o_positions(self) -> None:
        """Capture chunk-sized _next_positions from H2O caches after a joint pass."""
        self._frozen_positions = {}
        for cache in self._gpt.get_kv_caches():
            if isinstance(cache, H2OKVCache):
                self._frozen_positions[id(cache)] = cache.next_positions_ext(
                    self._chunk_size
                ).clone()

    def _freeze_h2o_positions(self, pos_in_chunk: int) -> None:
        """Before sequential token pos_in_chunk, pin _next_positions for H2O caches."""
        for cache in self._gpt.get_kv_caches():
            cid = id(cache)
            if cid in self._frozen_positions:
                fp = self._frozen_positions[cid]
                cache._next_positions = fp[
                    :, :, pos_in_chunk : pos_in_chunk + 1
                ].clone()

    def process_prompt(self, prompt: torch.Tensor) -> torch.Tensor:
        logits = self._inner.process_prompt(prompt)
        self._chunk_start_state = _save_cache_state(self._gpt)
        self._capture_frozen_h2o_positions()
        return logits

    def inference_for_token(self, token_idx: torch.Tensor) -> torch.Tensor:
        if self._pos_in_chunk == 0 and self._chunk_completed:
            # Restore KV to chunk-start state, then run joint pass (= update())
            _restore_cache_state(self._gpt, self._chunk_start_state)
            token_batch = torch.cat([t.view(-1, 1) for t in self._pending], dim=-1)
            self._gpt(token_batch)
            self._chunk_start_state = _save_cache_state(self._gpt)
            self._capture_frozen_h2o_positions()
            self._pending = []
            self._chunk_completed = False

        self._freeze_h2o_positions(self._pos_in_chunk)

        logits = self._inner.inference_for_token(token_idx)
        self._pending.append(token_idx)
        self._pos_in_chunk += 1
        if self._pos_in_chunk == self._chunk_size:
            self._pos_in_chunk = 0
            self._chunk_completed = True
        return logits


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


def args_fast_matches_default_with_frozen_positions() -> Tuple[str, List[tuple]]:
    list_no_devices = list(
        itertools.product(
            [2, 3, 4],  # chunk_size
            [1, 2],  # batch_size
            [10, 12],  # prompt_size (must be >= cache_length below)
            ["lastrec-default", "h2o-default"],
        )
    )
    return product_with_devices(list_no_devices, "chunk_size, batch_size, prompt_size, cache_name")


@pytest.mark.parametrize(*args_fast_matches_default_with_frozen_positions())
def test_fast_matches_default_with_frozen_positions(
    device,
    chunk_size,
    batch_size,
    prompt_size,
    cache_name,
):
    """
    ``FastSingleTokenInference`` must produce the same logits and sampled
    tokens as ``_ChunkJointReference`` wrapping ``DefaultSingleTokenInference``.

    ``SingleTokenFastInferenceDriver.update()`` processes each chunk as a single
    joint forward pass.  ``_ChunkJointReference`` replicates this: within each
    chunk it uses sequential single-token inference; at chunk boundaries it
    restores the KV state and runs the same joint pass, so both paths share
    identical KV buffers at the start of each new chunk.

    For h2o caches, ``_next_positions`` are frozen within each chunk (to the
    values captured by the joint pass) so the reference uses the same write
    positions as the compiled graph.
    """
    vocab_size = 32
    model_seed = 7
    prompt_seed = 42
    # cache_length must be <= prompt_size so the cache fills during prompting
    cache_length = 8
    assert prompt_size >= cache_length
    num_tokens = 3 * chunk_size
    max_seq_length = prompt_size + num_tokens

    prompt = _prompt(prompt_seed, prompt_size, vocab_size)
    # top_k=1 gives deterministic sampling without depending on the random seed
    sample_kwargs = dict(temperature=1.0, top_k=1)

    # --- FastSingleTokenInference path ---
    gpt_fast = _gpt(model_seed, vocab_size, device=device)
    lc_fast = _inference_model(
        gpt_model=gpt_fast,
        max_seq_length=max_seq_length,
        cache_name=cache_name,
        batch_size=batch_size,
        chunk_size=chunk_size,
        cache_length=cache_length,
        device=device,
    )
    fast_recorder, fast_tokens = _run_generation(
        FastSingleTokenInference(lc_fast, chunk_size=chunk_size),
        prompt,
        batch_size,
        num_tokens,
        sample_kwargs,
    )

    # --- Reference path: _ChunkJointReference replicates update()'s joint pass ---
    # update() processes each chunk as a single joint forward pass, which produces
    # different KV state than sequential single-token calls.  _ChunkJointReference
    # restores to chunk-start state at each boundary and runs the same joint pass.
    # For h2o caches it also freezes _next_positions within each chunk to match
    # the slot decisions captured by _set_next_positions() in the driver.
    gpt_default = _gpt(model_seed, vocab_size, device=device)
    lc_default = _inference_model(
        gpt_model=gpt_default,
        max_seq_length=max_seq_length,
        cache_name=cache_name,
        batch_size=batch_size,
        chunk_size=chunk_size,
        cache_length=cache_length,
        device=device,
    )
    reference = _ChunkJointReference(
        DefaultSingleTokenInference(lc_default), chunk_size
    )
    default_recorder, default_tokens = _run_generation(
        reference,
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
