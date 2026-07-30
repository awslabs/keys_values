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

This runs the full GRPO pipeline end-to-end, with every memory-heavy step
routed through KeysAndValues' KV-cache infrastructure:

1. **Generation + old log-probs** via
   :func:`generate_completions_with_logprobs` (chunked KV-cache decode; the
   sampling log-probs are captured in the same forward pass, so no separate
   scoring pass is needed).
2. **Reward** via a user-supplied reward function.
3. **Group-relative advantages** via :func:`compute_group_advantages`.
4. **Policy gradient** via :class:`GRPOLossHeadModel` +
   :class:`LongContextGradientModel` (memory-bounded backward).
5. **Optimizer step**.

The loop uses only a ``keys_values.model.GPT``, so it runs anywhere the rest
of the library does (including CPU). This is the path that actually exercises
sparse KV caches through generation, scoring, and the policy gradient -- which
is the reason to use KeysAndValues for RL in the first place.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch

import time
from contextlib import contextmanager

from keys_values.rl.grpo.loss import GRPOLossHeadModel
from keys_values.rl.grpo.rollout import generate_completions_with_logprobs
from keys_values.kvcache.gradient.main import LongContextGradientModel
from keys_values.rl.logprobs import compute_logprobs
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT
from keys_values.utils import VerbosityLevels


@contextmanager
def _phase_timer(store: Dict[str, float], key: str, device: torch.device):
    """Record wall-clock time (ms) for a phase, synchronizing CUDA if needed."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        store[key] = (time.perf_counter() - start) * 1000.0


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
    mean = grouped.mean(dim=-1, keepdim=True)
    std = grouped.std(dim=-1, keepdim=True)
    advantages = (grouped - mean) / (std + eps)
    return advantages.reshape(-1)


def grpo_step(
    gpt_model: GPT,
    prompt_ids: torch.Tensor,
    reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    group_size: int = 16,
    max_new_tokens: int = 256,
    chunk_size: int = 64,
    layers_per_cell: int = 1,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    pad_token_id: int = 0,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    rescore_old_logps: bool = False,
    profile: bool = False,
    verbose: VerbosityLevels = VerbosityLevels.NONE,
) -> Dict[str, float]:
    """Run one GRPO optimization step end-to-end on a KeysAndValues model.

    The ``gpt_model`` must have (non-dense) KV caches assigned, e.g. via
    ``KVCacheFactory.create`` + ``gpt_model.assign_kv_caches(...)``. These
    caches are used for generation, old-log-prob scoring, and the gradient
    pass.

    Old log-probs are captured *during* generation by default (the rollout
    forward pass already produces the exact sampling logits), so the separate
    scoring forward pass is skipped. Set ``rescore_old_logps=True`` to instead
    recompute them with :func:`compute_logprobs` -- useful for A/B comparison
    or to force old==new-policy semantics on the first inner step with a dense
    cache.

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
    rescore_old_logps : bool
        If ``True``, recompute old log-probs with a separate scoring pass
        instead of using the values captured during generation.
    profile : bool
        If ``True``, include per-phase wall-clock timings (ms) in the returned
        metrics: ``gen_time_ms``, ``score_time_ms``, ``grad_time_ms``.

    Returns
    -------
    dict
        Metrics: ``loss``, ``mean_reward``, ``mean_advantage``,
        ``advantage_std``, ``completion_len``, ``total_completions`` (plus
        timing keys if ``profile=True``).
    """
    device = next(gpt_model.parameters()).device
    num_prompts, prompt_len = prompt_ids.shape
    times: Dict[str, float] = {}

    # A processing chunk cannot exceed the caches' forward capacity. Grace
    # slots reserved by the factory's eviction defaults reduce it; compute the
    # static bound cache_length - grace (kv_cache_max_forward_length() cannot
    # be used here, as it requires an initialized/prefilled cache).
    caps = [
        kvc.cache_length
        - (getattr(kvc, "grace_period", 0) or getattr(kvc, "init_grace_tokens", 0) or 0)
        for kvc in gpt_model.get_kv_caches()
        if kvc is not None
    ]
    if caps:
        chunk_size = max(min(chunk_size, min(caps)), 1)

    # Expand each prompt into `group_size` completions (GRPO group layout).
    expanded_prompts = prompt_ids.repeat_interleave(group_size, dim=0).to(device)
    total = expanded_prompts.shape[0]

    # 1. Generation (chunked KV-cache decode) + old log-prob capture.
    gpt_model.eval()
    gpt_model.max_seq_length = prompt_len + max_new_tokens
    inference_model = LongContextInferenceModel(
        gpt_model=gpt_model,
        head_model=None,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    with _phase_timer(times, "gen_time_ms", device):
        completions, gen_logps, mask = generate_completions_with_logprobs(
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

    # 2. Reward + 3. group-relative advantages.
    rewards = reward_fn(expanded_prompts, completions).to(device)
    advantages = compute_group_advantages(rewards, group_size)

    # Scoring/gradient passes are fed the sequence without its last token, so
    # next-token prediction aligns each completion token `p` with the logits at
    # position `p-1` -- the same distribution it was sampled from. This matches
    # the log-probs captured during generation.
    full_ids = torch.cat([expanded_prompts, completions], dim=1)
    model_input_ids = full_ids[:, :-1]

    # 4. Old (sampling) log-probs. Default (single-epoch GRPO): none -- the
    #    loss head reuses the gradient pass's own forward (policy_logp.detach()),
    #    so the ratio is exactly 1 and no separate scoring pass runs. Set
    #    rescore_old_logps=True for multi-epoch updates / comparison.
    times["score_time_ms"] = 0.0
    old_logps = None
    if rescore_old_logps:
        with _phase_timer(times, "score_time_ms", device):
            with torch.no_grad():
                old_logps, _ = compute_logprobs(
                    gpt_model=gpt_model,
                    input_ids=model_input_ids,
                    targets=completions,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    verbose=verbose,
                )

    # 5. Policy gradient via memory-bounded chunked backward.
    head = GRPOLossHeadModel(
        gpt_model.config,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
    )
    head.set_batch(advantages=advantages, old_logps=old_logps, mask=mask)
    grad_model = LongContextGradientModel(
        gpt_model=gpt_model,
        head_model=head,
        layers_per_cell=layers_per_cell,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    grad_model.train()
    optimizer.zero_grad(set_to_none=True)

    # 6. Optimizer step.
    with _phase_timer(times, "grad_time_ms", device):
        loss = grad_model(model_input_ids, completions)
        loss.backward()
        optimizer.step()

    metrics = {
        "loss": float(loss.detach().mean().item()),
        "mean_reward": float(rewards.mean().item()),
        "mean_advantage": float(advantages.mean().item()),
        "advantage_std": float(advantages.std().item()),
        "completion_len": completion_len,
        "total_completions": total,
        "mean_completion_tokens": float(mask.sum(dim=-1).mean().item()),
    }
    if rescore_old_logps:
        # Quantify the rollout (decode) vs. training-forward log-prob skew over
        # real completion tokens -- a measure of the train/inference gap.
        with torch.no_grad():
            skew = ((gen_logps - old_logps).abs() * mask).sum() / mask.sum().clamp_min(1.0)
        metrics["logp_skew_decode_vs_forward"] = float(skew.item())
    if profile:
        metrics.update(times)
    return metrics
