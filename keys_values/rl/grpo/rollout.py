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
Generation adapter bridging KeysAndValues with TRL's GRPO rollout interface.

TRL's GRPOTrainer accepts a ``rollout_func`` callable as the official hook for
custom completion generation. This module provides the building blocks:

- :func:`generate_completions`: Generate completion token IDs from prompts
  using KeysAndValues' KV-cache long-context generation. Wraps the existing
  ``batched_generate_fn``, exposing a simple tensor-in / tensor-out interface.

This is phase 2 of the TRL integration (generation). It complements
``keys_values.rl.logprobs.compute_logprobs`` (phase 1, log-probs) so that both
the generation and scoring steps of GRPO run through the KV cache, keeping
GPU memory bounded for long prompts.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from keys_values.generate.base import batched_generate_fn
from keys_values.long_context import LongContextInferenceModel


def generate_completions(
    model: LongContextInferenceModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    *,
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

    chunks: List[torch.Tensor] = []
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
