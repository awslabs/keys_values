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
Memory-efficient per-token log-probability computation using KV cache.

Implements a :class:`HeadModel` that accumulates per-token log-probs
(and optionally entropies) instead of a scalar loss. Plug it into
:class:`LongContextInferenceModel` and the existing chunked forward
infrastructure handles everything — no new forward loop needed.

Usage::

    from keys_values.rl.logprobs import compute_logprobs

    logps, entropies = compute_logprobs(
        gpt_model=model,
        input_ids=input_ids,
        targets=completion_ids,
        cache_name="h2o-torch-quantized8",
        cache_length=16384,
        chunk_size=1024,
    )
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from keys_values.config import Config
from keys_values.head_model import HeadModel
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT
from keys_values.utils import VerbosityLevels


class LogProbsHeadModel(HeadModel):
    """HeadModel that accumulates per-token log-probs instead of a loss.

    Wraps the same logic as :class:`CrossEntropyOnLogits` but instead of
    reducing to a scalar loss, it gathers the log-probability of each target
    token and stores it. After the full chunked forward pass completes,
    call :meth:`get_results` to retrieve the accumulated tensors.

    This is meant to be used with :class:`LongContextInferenceModel` — the
    existing chunk/cell/layer loop calls ``forward()`` chunk by chunk, and
    this class collects log-probs as they come.
    """

    NAME = "log_probs"

    def __init__(
        self, config: Config, temperature: float = 1.0, compute_entropy: bool = False
    ):
        super().__init__()
        self._vocab_size = config.padded_vocab_size
        self._temperature = temperature
        self._compute_entropy = compute_entropy
        self._logps_chunks: list[torch.Tensor] = []
        self._entropy_chunks: list[torch.Tensor] = []

    def needs_logits(self) -> bool:
        return True

    def forward(
        self,
        model_outputs: torch.Tensor,
        targets: Optional[torch.Tensor],
        input_pos: int,
    ) -> torch.Tensor:
        """Accumulate log-probs for target tokens in this chunk.

        Called by LongContextInferenceModel for each chunk. When targets
        is None (prompt-only chunk), we skip. When targets are present,
        we gather log-probs and optionally entropy.
        """
        if input_pos == 0:
            self._logps_chunks.clear()
            self._entropy_chunks.clear()

        diff = self._check_model_outputs_targets(
            model_outputs, targets, final_dim=self._vocab_size
        )

        if diff is not None:
            logits = model_outputs[:, diff:, :]
            if self._temperature != 1.0:
                logits = logits / self._temperature

            # Per-token log-probs
            log_probs = F.log_softmax(logits, dim=-1)
            token_logps = torch.gather(
                log_probs, dim=-1, index=targets.unsqueeze(-1)
            ).squeeze(-1)
            self._logps_chunks.append(token_logps)

            if self._compute_entropy:
                ent = -(log_probs.exp() * log_probs).sum(dim=-1)
                self._entropy_chunks.append(ent)

        # Return zeros
        return torch.zeros(
            model_outputs.shape[0],
            device=model_outputs.device,
            dtype=model_outputs.dtype,
        )

    def num_target_entries(self, targets: torch.Tensor) -> Optional[torch.Tensor]:
        return None

    def get_results(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Retrieve accumulated log-probs and entropies after forward pass.

        Returns
        -------
        logps : torch.Tensor
            Shape ``(batch_size, num_target_tokens)``.
        entropies : torch.Tensor | None
            Shape ``(batch_size, num_target_tokens)`` or None.
        """
        logps = torch.cat(self._logps_chunks, dim=1)
        entropies = (
            torch.cat(self._entropy_chunks, dim=1) if self._entropy_chunks else None
        )
        return logps, entropies

    def _empty_clone(self, device: Optional[torch.device] = None) -> "HeadModel":
        config = Config()
        config.padded_vocab_size = self._vocab_size
        return LogProbsHeadModel(
            config,
            temperature=self._temperature,
            compute_entropy=self._compute_entropy,
        )


def compute_logprobs(
    gpt_model: GPT,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    cache_name: str = "h2o-torch-quantized8",
    cache_length: int = 16384,
    chunk_size: int = 1024,
    cache_kwargs: Optional[dict] = None,
    temperature: float = 1.0,
    compute_entropy: bool = False,
    verbose: VerbosityLevels = VerbosityLevels.SOME,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute per-token log-probs via LongContextInferenceModel.

    This is the primary entry point. It creates a :class:`LogProbsHeadModel`,
    plugs it into :class:`LongContextInferenceModel`, and runs the existing
    chunked forward pass. All KV cache management, chunk/cell grouping, and
    layer processing is handled by the existing infrastructure.

    Args:
        gpt_model: KeysAndValues GPT model.
        input_ids: Full sequence (prompt + completion), shape
            ``(batch_size, seq_length)``.
        targets: Target tokens (right-aligned with input_ids), shape
            ``(batch_size, num_completion_tokens)``.
        cache_name: KV cache policy name.
        cache_length: Number of slots in the KV cache.
        chunk_size: Chunk size for post-prefill processing.
        cache_kwargs: Extra args for KV cache construction.
        temperature: Scales logits before softmax.
        compute_entropy: Whether to also return per-token entropy.

    Returns:
        Tuple of (log_probs, entropies).
    """
    batch_size = input_ids.shape[0]
    config = gpt_model.config
    dtype = next(gpt_model.parameters()).dtype

    head = LogProbsHeadModel(
        config, temperature=temperature, compute_entropy=compute_entropy
    )

    caches_created = False
    if gpt_model.get_kv_caches()[0] is None:
        gpt_model.assign_kv_caches(
            KVCacheFactory.create(
                gpt_model=gpt_model,
                name=cache_name,
                max_batch_size=batch_size,
                cache_length=cache_length,
                dtype=dtype,
                cache_kwargs=cache_kwargs or {},
            )
        )
        caches_created = True

    inference_model = LongContextInferenceModel(
        gpt_model=gpt_model,
        head_model=head,
        chunk_size=chunk_size,
        verbose=verbose,
    )

    # Run the forward pass
    inference_model(input_ids=input_ids, targets=targets)

    logps, entropies = head.get_results()

    if caches_created:
        from keys_values.kvcache.factory import deallocate_kv_cache_buffers_of_model

        deallocate_kv_cache_buffers_of_model(gpt_model)
        gpt_model.assign_kv_caches([None] * config.n_layer)

    return logps, entropies
