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
Completion generation for GRPO rollouts, built on KeysAndValues.

- :func:`generate_completions`: Generate completion token IDs from prompts
  using KeysAndValues' KV-cache long-context generation. Wraps the existing
  ``batched_generate_fn``, exposing a simple tensor-in / tensor-out interface.

Together with :func:`keys_values.logprobs.compute_logprobs` (log-probs), this
means both the generation and scoring steps of GRPO run through the KV cache,
keeping GPU memory bounded for long prompts.
"""

from __future__ import annotations

from typing import Optional

import torch

from keys_values.generate.base import batched_generate_fn
from keys_values.long_context import LongContextInferenceModel


def generate_completions(
    model: LongContextInferenceModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Generate completions for a batch of (left-padded) prompts.

    The prompt is processed through the KV cache in chunks (so even very long
    prompts use bounded memory), then tokens are generated one at a time.
    Internally delegates to :func:`batched_generate_fn`.

    Parameters
    ----------
    model : LongContextInferenceModel
        Model with KV caches assigned, providing chunked prefill + decoding.
    prompt_ids : torch.Tensor
        Prompt token IDs, shape ``(batch_size, prompt_len)``. All prompts
        must share the same length (use left padding).
    max_new_tokens : int
        Maximum number of new tokens to generate per sequence.
    temperature : float
        Sampling temperature. Use a very small value for near-greedy.
    top_k : int | None
        Top-k filtering parameter. ``None`` disables it.
    top_p : float
        Nucleus sampling threshold. ``1.0`` disables it.
    eos_token_id : int | None
        Token ID that ends generation early. ``None`` disables early stop.
    pad_token_id : int
        Token used to pad completions of sequences that stopped early.

    Returns
    -------
    torch.Tensor
        Completion token IDs (without the prompt), shape
        ``(batch_size, num_generated)``. Sequences that stopped early are
        padded with ``pad_token_id``.
    """
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    batch_size = prompt_ids.shape[0]
    sample_args = dict(temperature=temperature, top_k=top_k, top_p=top_p)
    stop_tokens = ([eos_token_id],) if eos_token_id is not None else ()

    chunks = []
    for token_batch in batched_generate_fn(
        model=model,
        prompts=prompt_ids,
        max_returned_tokens=max_new_tokens,
        sample_args=sample_args,
        stop_tokens=stop_tokens,
        ignore_index=pad_token_id,
        deallocate_cache_buffers=True,
    ):
        chunks.append(token_batch)

    if not chunks:
        return torch.full(
            (batch_size, 1),
            pad_token_id,
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )
    return torch.cat(chunks, dim=1)


def generate_completions_with_logprobs(
    model: LongContextInferenceModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate completions **and** capture old-policy log-probs in one pass.

    Generation already runs a forward pass that produces exactly the logits
    used to sample each token, so we capture the per-token log-prob at the same
    time. This removes the need for a separate scoring forward pass over the
    prompt+completion (as :func:`keys_values.logprobs.compute_logprobs` would
    do) -- roughly a third of the per-step compute in GRPO.

    For a sparse KV cache this is also *more* correct than re-scoring: the
    captured value is the true sampling distribution the token was drawn from,
    whereas re-scoring rebuilds the (lossy) cache with a different eviction
    pattern.

    Parameters
    ----------
    model : LongContextInferenceModel
        Model with KV caches assigned.
    prompt_ids : torch.Tensor
        Left-padded prompts, shape ``(batch_size, prompt_len)``.
    max_new_tokens : int
        Maximum number of new tokens to generate per sequence.
    temperature, top_k, top_p, eos_token_id, pad_token_id
        Sampling / padding controls, see :func:`generate_completions`.

    Returns
    -------
    completions : torch.Tensor
        Completion token IDs, shape ``(batch_size, num_generated)``. Stopped
        rows are padded with ``pad_token_id``.
    old_logps : torch.Tensor
        Per-token log-probs of the sampled tokens under the temperature-scaled
        policy, shape ``(batch_size, num_generated)``. Zero at padded positions.
    mask : torch.Tensor
        Float mask, shape ``(batch_size, num_generated)``, ``1.0`` for real
        generated tokens and ``0.0`` for padding after a stop.
    """
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    batch_size = prompt_ids.shape[0]
    sample_args = dict(temperature=temperature, top_k=top_k, top_p=top_p)
    stop_tokens = ([eos_token_id],) if eos_token_id is not None else ()

    tok_chunks = []
    logp_chunks = []
    mask_chunks = []
    for tokens, logps, active in batched_generate_fn(
        model=model,
        prompts=prompt_ids,
        max_returned_tokens=max_new_tokens,
        sample_args=sample_args,
        stop_tokens=stop_tokens,
        ignore_index=pad_token_id,
        deallocate_cache_buffers=True,
        return_logprobs=True,
        no_inference_mode=True,  # buffers are updated in place by the grad pass
    ):
        tok_chunks.append(tokens)
        logp_chunks.append(logps)
        mask_chunks.append(active)

    if not tok_chunks:
        completions = torch.full(
            (batch_size, 1),
            pad_token_id,
            dtype=prompt_ids.dtype,
            device=prompt_ids.device,
        )
        zeros = torch.zeros((batch_size, 1), device=prompt_ids.device)
        return completions, zeros, zeros

    completions = torch.cat(tok_chunks, dim=1)
    old_logps = torch.cat(logp_chunks, dim=1)
    mask = torch.cat(mask_chunks, dim=1).to(old_logps.dtype)
    return completions, old_logps, mask
