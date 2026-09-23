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
Standalone GRPO training loop built entirely on KeysAndValues components.

This demonstrates the full GRPO pipeline end-to-end, with every memory-heavy
step routed through KeysAndValues' KV-cache infrastructure:

1. **Generation** via :func:`generate_completions` (chunked KV-cache decode).
2. **Reward** via a user-supplied reward function.
3. **Group-relative advantages** via :func:`compute_group_advantages`.
4. **Old (sampling) log-probs** via :func:`compute_logprobs` (no grad).
5. **Policy gradient** via :class:`GRPOLossHeadModel` +
   :class:`LongContextGradientModel` (memory-bounded backward).
6. **Optimizer step**.

Unlike :class:`keys_values.rl.grpo.trainer.GRPOLongContextTrainer` (which plugs
into TRL's ``GRPOTrainer`` and relies on its HuggingFace-model machinery), this
loop uses only a ``keys_values.model.GPT``, so it runs anywhere the rest of
the library does (including CPU).
"""

from __future__ import annotations

from typing import Callable, Dict

import torch

from keys_values.rl.grpo.loss import GRPOLossHeadModel
from keys_values.rl.grpo.rollout import generate_completions
from keys_values.kvcache.gradient.main import LongContextGradientModel
from keys_values.rl.logprobs import compute_logprobs
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT
from keys_values.utils import VerbosityLevels


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute group-relative advantages, the core of GRPO.

    Rewards are assumed laid out as ``num_groups`` contiguous groups of
    ``group_size`` completions each (the standard GRPO layout: ``G``
    completions per prompt). Within each group, advantages are the rewards
    normalized to zero mean and unit standard deviation.

    Parameters
    ----------
    rewards : torch.Tensor
        Shape ``(num_groups * group_size,)``.
    group_size : int
        Number of completions per prompt (``G`` in the GRPO paper).
    eps : float
        Numerical stabilizer for the per-group std.

    Returns
    -------
    torch.Tensor
        Advantages, same shape as ``rewards``.
    """
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1D, got shape {tuple(rewards.shape)}")
    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"rewards length {rewards.numel()} not divisible by group_size {group_size}"
        )

    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    advantages = (grouped - mean) / (std + eps)
    return advantages.reshape(-1)


def grpo_step(
    gpt_model: GPT,
    prompt_ids: torch.Tensor,
    reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    group_size: int,
    max_new_tokens: int,
    chunk_size: int = 16,
    layers_per_cell: int = 1,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    pad_token_id: int = 0,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    verbose: VerbosityLevels = VerbosityLevels.NONE,
) -> Dict[str, float]:
    """Run one GRPO optimization step end-to-end on a KeysAndValues model.

    The ``gpt_model`` must have (non-dense) KV caches assigned, e.g. via
    ``gpt_model.set_kv_caches(...)`` or ``KVCacheFactory.create``. These caches
    are used for generation, old-log-prob scoring, and the gradient pass.

    Parameters
    ----------
    gpt_model : GPT
        Policy model with KV caches assigned.
    prompt_ids : torch.Tensor
        Left-padded prompts, shape ``(num_prompts, prompt_len)``. Each prompt
        is expanded into ``group_size`` completions.
    reward_fn : callable
        Maps ``(prompt_ids, completion_ids)`` to a reward tensor of shape
        ``(num_prompts * group_size,)``.
    optimizer : torch.optim.Optimizer
        Optimizer over ``gpt_model`` parameters.
    group_size : int
        Completions sampled per prompt.
    max_new_tokens : int
        Completion length cap.
    chunk_size, layers_per_cell : int
        Control the chunked gradient computation memory/speed tradeoff.

    Returns
    -------
    dict
        Metrics: ``loss``, ``mean_reward``, ``mean_advantage``.
    """
    device = next(gpt_model.parameters()).device
    num_prompts, prompt_len = prompt_ids.shape

    # Expand each prompt into `group_size` completions
    expanded_prompts = prompt_ids.repeat_interleave(group_size, dim=0).to(device)
    total = expanded_prompts.shape[0]

    # --- 1. Generation (KV-cache chunked decode) ---
    gpt_model.eval()
    gpt_model.max_seq_length = prompt_len + max_new_tokens
    inference_model = LongContextInferenceModel(
        gpt_model=gpt_model,
        head_model=None,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    completions = generate_completions(
        model=inference_model,
        prompt_ids=expanded_prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    completion_len = completions.shape[1]

    # --- 2. Reward ---
    rewards = reward_fn(expanded_prompts, completions).to(device)

    # --- 3. Group-relative advantages ---
    advantages = compute_group_advantages(rewards, group_size)

    # --- 4. Old (sampling) log-probs (no grad) ---
    # Caches are already assigned on gpt_model, so compute_logprobs reuses
    # them; the cache_name/cache_length args below are only used when caches
    # need to be created, so their values are immaterial here.
    full_ids = torch.cat([expanded_prompts, completions], dim=1)
    with torch.no_grad():
        old_logps, _ = compute_logprobs(
            gpt_model=gpt_model,
            input_ids=full_ids,
            targets=completions,
            chunk_size=chunk_size,
            temperature=temperature,
            verbose=verbose,
        )

    # --- 5. Policy gradient via memory-bounded backward ---
    head = GRPOLossHeadModel(
        gpt_model.config,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
    )
    head.set_batch(old_logps=old_logps, advantages=advantages)

    grad_model = LongContextGradientModel(
        gpt_model=gpt_model,
        head_model=head,
        layers_per_cell=layers_per_cell,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    grad_model.train()
    optimizer.zero_grad(set_to_none=True)

    loss = grad_model(full_ids, completions)
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().mean().item()),
        "mean_reward": float(rewards.mean().item()),
        "mean_advantage": float(advantages.mean().item()),
        "advantage_std": float(advantages.std().item()),
        "completion_len": completion_len,
        "total_completions": total,
    }
