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
GRPO loss as a KeysAndValues HeadModel.

This is the piece that lets the *policy gradient* flow through
KeysAndValues' memory-efficient gradient computation
(:class:`LongContextGradientModel`). Instead of materializing per-token
log-probs for the whole sequence and computing the GRPO loss externally
(which would require retaining activations for the full sequence), we
express the GRPO loss as a :class:`HeadModel`.

The chunked forward/backward infrastructure then computes the policy
gradient with bounded GPU memory, regardless of completion length.

The GRPO per-token loss (with ``beta=0``, the TRL default, i.e. no KL term)
is::

    ratio_t       = exp(policy_logp_t - old_logp_t)
    unclipped_t   = ratio_t * A
    clipped_t     = clip(ratio_t, 1 - eps_low, 1 + eps_high) * A
    loss_t        = -min(unclipped_t, clipped_t)

where ``A`` is the per-sequence (group-relative) advantage. The head model
accumulates the summed loss over completion tokens; normalization is handled
by :meth:`num_target_entries`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from keys_values.config import Config
from keys_values.head_model import HeadModel


class GRPOLossHeadModel(HeadModel):
    """HeadModel computing the (clipped) GRPO policy-gradient loss.

    Before each forward pass over a batch, call :meth:`set_batch` to provide
    the per-token ``old_logps`` and per-sequence ``advantages`` for that batch.
    Then drive the chunked forward/backward via
    :class:`LongContextGradientModel`, exactly as you would with
    :class:`CrossEntropyOnLogits`.

    Parameters
    ----------
    config : Config
        Model config (used for vocab size).
    epsilon_low, epsilon_high : float
        Lower/upper clipping bounds for the importance ratio.
    """

    NAME = "grpo_loss"

    def __init__(
        self,
        config: Config,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.2,
    ):
        super().__init__()
        self._vocab_size = config.padded_vocab_size
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self._old_logps: Optional[torch.Tensor] = None
        self._advantages: Optional[torch.Tensor] = None
        self._offset = 0

    def set_batch(self, old_logps: torch.Tensor, advantages: torch.Tensor):
        """Provide per-token old log-probs and per-sequence advantages.

        Parameters
        ----------
        old_logps : torch.Tensor
            Shape ``(batch_size, completion_length)``. Detached log-probs of
            the completion under the sampling (old) policy.
        advantages : torch.Tensor
            Shape ``(batch_size,)``. Group-relative advantage per sequence.
        """
        self._old_logps = old_logps
        self._advantages = advantages

    def needs_logits(self) -> bool:
        return True

    def forward(
        self,
        model_outputs: torch.Tensor,
        targets: Optional[torch.Tensor],
        input_pos: int,
    ) -> torch.Tensor:
        if input_pos == 0:
            self._offset = 0

        diff = self._check_model_outputs_targets(
            model_outputs, targets, final_dim=self._vocab_size
        )
        if diff is None:
            return torch.zeros(
                model_outputs.shape[0],
                device=model_outputs.device,
                dtype=model_outputs.dtype,
            )

        if self._old_logps is None or self._advantages is None:
            raise RuntimeError(
                "Call set_batch(old_logps, advantages) before the forward pass."
            )

        logits = model_outputs[:, diff:, :]
        num = targets.shape[1]

        # Policy log-probs for the target tokens in this chunk
        log_probs = F.log_softmax(logits, dim=-1)
        policy_logp = torch.gather(
            log_probs, dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)

        # Align old_logps to this chunk via a running offset
        old_logp = self._old_logps[:, self._offset : self._offset + num].to(
            policy_logp.device
        )
        self._offset += num

        advantages = self._advantages.to(policy_logp.device).unsqueeze(1)

        ratio = torch.exp(policy_logp - old_logp)
        unclipped = ratio * advantages
        clipped = (
            torch.clamp(ratio, 1.0 - self.epsilon_low, 1.0 + self.epsilon_high)
            * advantages
        )
        per_token_loss = -torch.min(unclipped, clipped)

        # Sum over chunk tokens; LongContextGradientModel sums across chunks
        return per_token_loss.sum(dim=-1)

    def num_target_entries(self, targets: torch.Tensor) -> Optional[torch.Tensor]:
        # Normalize the summed loss by the number of completion tokens,
        # giving the per-sequence mean (GRPO "grpo" loss type).
        assert 1 <= targets.ndim <= 2
        return torch.full(
            (targets.shape[0],),
            float(targets.shape[-1]),
            dtype=torch.float32,
        )

    def _empty_clone(self, device: Optional[torch.device] = None) -> "HeadModel":
        config = Config()
        config.padded_vocab_size = self._vocab_size
        clone = GRPOLossHeadModel(
            config,
            epsilon_low=self.epsilon_low,
            epsilon_high=self.epsilon_high,
        )
        return clone
