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

from typing import Any, Callable, Dict, Tuple

import functools
import os

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

import time
from contextlib import contextmanager

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.attention.base import do_softcapping
from keys_values.finetune.utils import may_match_twice_flex_attention_sdpa
from keys_values.kvcache.factory import deallocate_kv_cache_buffers_of_model
from keys_values.rl.grpo.attention import gradient_sdpa_mode_name
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
    mode: str = "grpo",
) -> torch.Tensor:
    """Compute group-relative advantages.

    Rewards are assumed laid out as ``num_groups`` contiguous groups of
    ``group_size`` completions each (the standard GRPO layout: ``G``
    completions per prompt).

    Parameters
    ----------
    rewards : torch.Tensor
        Shape ``(num_groups * group_size,)``.
    group_size : int
        Number of completions per prompt (``G`` in the GRPO paper).
    eps : float
        Numerical stabilizer for the per-group std (``grpo`` mode).
    mode : str
        ``"grpo"``: rewards normalized to zero mean / unit std within the
        group (Shao et al., DeepSeekMath).
        ``"rloo"``: leave-one-out baseline, ``a_i = r_i - mean(r_j, j != i)``
        (Ahmadian et al., RLOO) -- unbiased, no std division, useful when
        reward scales are meaningful.

    Returns
    -------
    torch.Tensor
        Advantages, same shape as ``rewards``.
    """
    if mode not in ("grpo", "rloo"):
        raise ValueError(f"mode = {mode}, must be 'grpo' or 'rloo'")
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1D, got shape {tuple(rewards.shape)}")
    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"rewards length {rewards.numel()} not divisible by group_size {group_size}"
        )
    grouped = rewards.view(-1, group_size)
    if mode == "rloo":
        if group_size < 2:
            raise ValueError("rloo mode requires group_size >= 2")
        total = grouped.sum(dim=-1, keepdim=True)
        loo_mean = (total - grouped) / (group_size - 1)
        advantages = grouped - loo_mean
    else:
        mean = grouped.mean(dim=-1, keepdim=True)
        std = grouped.std(dim=-1, keepdim=True)
        advantages = (grouped - mean) / (std + eps)
    return advantages.reshape(-1)


def _dense_baseline_backward(
    gpt_model: GPT,
    head: GRPOLossHeadModel,
    model_input_ids: torch.Tensor,
    completions: torch.Tensor,
    grad_scale: float,
    activation_checkpointing: bool = True,
    logits_chunk: int = 512,
) -> torch.Tensor:
    """
    Standard (non-chunked) dense backward, for the dense-RL baseline.

    This is the comparison arm for the memory-bounded chunked path: same
    model, same rollout, same advantages, same loss head, same
    normalization. The *only* difference is how the gradient is computed --
    one full-sequence forward with all activations retained, and a plain
    ``backward()``, which is what standard RL implementations do.

    Equivalence with :class:`LongContextGradientModel` (verified in
    `test/rl/grpo/test_loop.py::test_dense_baseline_matches_chunked_gradient`):

    * The KV caches are detached for the duration of the forward, so
      attention runs as default causal self-attention over the full
      sequence (see :meth:`GPT.forward`: training mode = no caches).
    * The head returns the per-sequence *sum* of per-token losses; we then
      apply ``scale_factor / num_target_entries`` exactly as
      :meth:`LongContextInferenceModel._forward_with_targets` does, with
      ``average_loss_per_batch=True`` semantics (mean over the batch), which
      is the default of :class:`LongContextGradientModel`.
    * The chunked path's ``loss.backward()`` on the per-sequence loss vector
      accumulates gradients of the *mean* over sequences, so we call
      ``.mean().backward()`` here. (Verified empirically: using ``.sum()``
      made every gradient exactly ``batch_size`` times too large.)

    Strong-baseline memory optimizations (all exact; none changes the loss
    or gradient, only what is resident at once):

    * **Activation checkpointing** of every transformer block
      (`torch.utils.checkpoint`, non-reentrant), the standard dense trick.
    * **One sequence per backward**: the group is processed sequence by
      sequence with gradient accumulation. GRPO does not need the whole
      group in one graph.
    * **No full-vocabulary logits**: the LM head is applied only to the
      completion positions, in chunks of `logits_chunk` tokens, and each
      chunk is fed to the head immediately. A full `B x S x V` logits
      tensor at 8k tokens is ~2.3 GiB per sequence in bf16 and was the
      actual cause of the first dense OOM we observed (not attention).

    A dense frontier measured without these would be a strawman.

    Returns the per-sequence loss vector (detached-comparable to the chunked
    path's return value).
    """
    kv_caches = gpt_model.get_kv_caches()
    gpt_model.clear_kv_caches()
    blocks = gpt_model.transformer.h
    orig_forwards = [b.forward for b in blocks]
    # Per-batch head state, sliced per sequence below and restored after
    advantages_all = head._advantages
    old_logps_all = head._old_logps
    mask_all = head._mask
    try:
        # Free the (now detached) sparse cache buffers for the duration of
        # the dense pass. They are reallocated lazily on the next forward.
        deallocate_kv_cache_buffers_of_model(gpt_model)
        gpt_model.max_seq_length = int(model_input_ids.shape[1])
        if activation_checkpointing:
            for block in blocks:
                block.forward = functools.partial(
                    torch_checkpoint,
                    block.forward,
                    use_reentrant=False,
                )
        batch_size, num = completions.shape
        # `average_loss_per_batch=True` semantics, identical to the chunked
        # path: every per-token loss is divided by the batch mean of valid
        # token counts, then per-sequence sums are averaged over the batch.
        # With `head.per_sequence_norm`, the head divides inside `forward`
        # and `num_target_entries` returns ones, so `denom == 1` here too.
        # Must be evaluated on the full-batch mask, before the per-sequence
        # `set_batch` calls below.
        num_entries = head.num_target_entries(completions)
        if num_entries is not None:
            denom = num_entries.to(dtype=torch.float32).mean().item()
        else:
            denom = 1.0
        per_seq_scale = grad_scale / (denom * batch_size)
        losses = []
        for b in range(batch_size):
            head.set_batch(
                advantages=advantages_all[b : b + 1],
                old_logps=None if old_logps_all is None else old_logps_all[b : b + 1],
                mask=None if mask_all is None else mask_all[b : b + 1],
            )
            hidden = gpt_model(model_input_ids[b : b + 1], skip_lm_head=True)
            hidden = hidden[:, -num:, :]  # positions that predict completions
            seq_sum = None
            for start in range(0, num, logits_chunk):
                end = min(start + logits_chunk, num)
                logits = do_softcapping(
                    gpt_model.lm_head(hidden[:, start:end, :]),
                    thresh=gpt_model.config.final_logit_softcapping,
                )
                # `input_pos=0` resets the head's offset; later chunks
                # advance it, exactly as consecutive cells do in the
                # chunked path
                part = head(logits, completions[b : b + 1, start:end], input_pos=start)
                seq_sum = part if seq_sum is None else seq_sum + part
            loss_b = seq_sum * per_seq_scale
            loss_b.sum().backward()
            losses.append(loss_b.detach() * batch_size)  # per-seq value as chunked path
        loss = torch.cat(losses)
    finally:
        for block, fwd in zip(blocks, orig_forwards):
            block.forward = fwd
        head.set_batch(
            advantages=advantages_all, old_logps=old_logps_all, mask=mask_all
        )
        gpt_model.assign_kv_caches(kv_caches)
    return loss


def _chunked_micro_batched_backward(
    grad_model: LongContextGradientModel,
    head: GRPOLossHeadModel,
    model_input_ids: torch.Tensor,
    completions: torch.Tensor,
    grad_scale: float,
    grad_micro_batch: int,
) -> Tuple[torch.Tensor, int, int]:
    """
    Chunked (sparse-KV) backward over the group in micro-batches of
    ``grad_micro_batch`` sequences, accumulating gradients.

    Every memory term of the chunked backward carries the batch size as a
    factor (per-chunk K-long attention tensors, per-token block activations,
    replay/checkpoint buffers), so processing ``mb`` of the ``B`` sequences
    at a time divides the gradient-phase peak by ``B / mb``. This is what
    lets H2O caches with K=16384 / K=32768 fit a 44 GiB GPU for Qwen3-1.7B.

    Exactness. The full-batch pass computes (``accumulate.py:run_head_model``
    with ``average_loss_per_batch=True``)::

        loss = grad_scale / (B * denom_full) * sum_b sum_t loss_bt

    where ``denom_full`` is the batch mean of the per-sequence valid token
    counts (``per_sequence_norm=False``) or ``1`` (``per_sequence_norm=True``,
    the head divides inside ``forward`` and ``num_target_entries`` returns
    ones). For a micro-batch ``S`` of size ``mb`` the model instead divides by
    ``mb * denom_S`` (``denom_S`` = mean count over ``S``), so passing
    ``scale_factor = grad_scale * mb * denom_S / (B * denom_full)`` makes
    its loss equal to ``grad_scale / (B * denom_full) * sum_{b in S} sum_t
    loss_bt``; the micro-batch losses (and gradients) then sum to the
    full-batch ones. This mirrors ``per_seq_scale`` in
    :func:`_dense_baseline_backward`.

    Micro-batches are contiguous slices of the group. Advantages are computed
    once on the full group before the backward (:func:`compute_group_advantages`)
    and are per-sequence constants from then on, as are the completion mask
    and the old log-probs, so slicing them needs no further care; no
    normalization in the loss depends on which sequences share a pass once
    ``denom_full`` is fixed above.

    Returns ``(loss, parked_peak_bytes, parked_peak_count)``: the summed loss
    (equal to the full-batch loss value) and the max over passes of
    :attr:`LongContextGradientModel.last_parked_peak`.
    """
    batch_size = completions.shape[0]
    # Per-batch head state, sliced per micro-batch below and restored after
    advantages_all = head._advantages
    old_logps_all = head._old_logps
    mask_all = head._mask

    def _denom(targets: torch.Tensor) -> float:
        num_entries = head.num_target_entries(targets)
        if num_entries is None:
            return 1.0
        return num_entries.to(dtype=torch.float32).mean().item()

    # Must be evaluated on the full-batch mask, before the per-micro-batch
    # `set_batch` calls below.
    denom_full = _denom(completions)
    loss = None
    parked_bytes, parked_count = 0, 0
    try:
        for start in range(0, batch_size, grad_micro_batch):
            end = min(start + grad_micro_batch, batch_size)
            mb = end - start
            head.set_batch(
                advantages=advantages_all[start:end],
                old_logps=None if old_logps_all is None else old_logps_all[start:end],
                mask=None if mask_all is None else mask_all[start:end],
            )
            denom_mb = _denom(completions[start:end])
            scale_factor = grad_scale * mb * denom_mb / (batch_size * denom_full)
            loss_mb = grad_model(
                model_input_ids[start:end],
                completions[start:end],
                scale_factor=scale_factor,
            )
            loss_mb.backward()
            loss_mb = loss_mb.detach()
            loss = loss_mb if loss is None else loss + loss_mb
            pb, pc = grad_model.last_parked_peak
            parked_bytes = max(parked_bytes, pb)
            parked_count = max(parked_count, pc)
    finally:
        head.set_batch(
            advantages=advantages_all, old_logps=old_logps_all, mask=mask_all
        )
    return loss, parked_bytes, parked_count


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
    zero_grad: bool = True,
    optimizer_step: bool = True,
    grad_scale: float = 1.0,
    advantage_mode: str = "grpo",
    backward_tmp_gb: float = 0.0,
    dense_baseline: bool = False,
    per_sequence_norm: bool = False,
    grad_micro_batch: int | None = None,
    train_cache_kwargs: Dict[str, Any] | None = None,
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
    zero_grad, optimizer_step, grad_scale
        Gradient-accumulation controls. To accumulate K prompts into one
        optimizer update, call ``grpo_step`` K times with
        ``zero_grad=(k == 0)``, ``optimizer_step=(k == K-1)`` and
        ``grad_scale=1/K``; gradients sum across the calls.
    per_sequence_norm : bool
        Normalize each sequence's loss by its own token count instead of the
        group-mean count, so long and short rollouts contribute equal
        gradient mass. See :class:`GRPOLossHeadModel`. Default ``False``.
    grad_micro_batch : int | None
        Number of sequences per chunked backward pass. ``None`` (default):
        the whole group ``(num_prompts * group_size, T)`` goes through one
        pass, today's behavior and memory. ``1..B``: the group is processed
        in ``ceil(B / grad_micro_batch)`` passes with gradient accumulation
        and a per-pass ``scale_factor`` that reproduces the full-batch
        normalization exactly (both ``per_sequence_norm`` modes), dividing
        the gradient-phase device peak by about ``B / grad_micro_batch``.
        See :func:`_chunked_micro_batched_backward`. Ignored when
        ``dense_baseline=True`` (that arm is already per sequence).
    train_cache_kwargs : dict | None
        Extra keyword arguments for the training replay caches of the
        gradient cells (``LongContextGradientModel(train_cache_kwargs=...)``).
        ``{"flexatt_args": FlexAttentionArgs(...)}`` makes the cells run their
        chunk SDPA with FlexAttention (C-long queries, no K-long SDPA output
        per chunk) while the model's own MHA objects, and so rollout decode,
        stay on FlashInfer. Use :func:`gradient_train_cache_kwargs`. Requires
        CUDA. ``None`` (default): the cells follow the MHA's configuration,
        i.e. query-padded PyTorch SDPA unless the MHA has ``flexatt_args``.

    Returns
    -------
    dict
        Metrics: ``loss``, ``mean_reward``, ``mean_advantage``,
        ``advantage_std``, ``completion_len``, ``total_completions``,
        ``grad_sdpa_mode`` (SDPA mode of the gradient cells; not with
        ``dense_baseline``) (plus timing keys if ``profile=True``).
    """
    if grad_micro_batch is not None and grad_micro_batch < 1:
        raise ValueError(f"grad_micro_batch = {grad_micro_batch}, must be >= 1")
    device = next(gpt_model.parameters()).device
    train_cache_kwargs = dict(train_cache_kwargs or {})
    if train_cache_kwargs.get("flexatt_args") is not None and device.type != "cuda":
        # The replay cache would select FlexAttention and fail in torch.compile
        raise ValueError(
            "train_cache_kwargs['flexatt_args'] (FlexAttention in the gradient "
            f"cells) requires a CUDA device, but the model is on {device}"
        )
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
    advantages = compute_group_advantages(rewards, group_size, mode=advantage_mode)

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
        temperature=temperature,
        per_sequence_norm=per_sequence_norm,
    )
    head.set_batch(advantages=advantages, old_logps=old_logps, mask=mask)
    # The reordered / GQA-expanded K/V an "ext-*" annotation describes can be
    # saved twice in the autograd graph, under both FlexAttention and
    # query-padded SDPA in the cells. `may_match_twice` keeps the annotation
    # alive after its first match (a permissive superset: unmatched leftovers
    # are harmless); without it, the second save is left as an unmatched pack
    # argument, which can stall the annotation chain in the chunked backward
    # (issue #148). This mirrors the finetune path (`may_match_twice_factory`).
    autograd_hooks_kwargs: Dict[str, Any] = dict(
        may_match_twice=may_match_twice_flex_attention_sdpa,
    )
    # Env-gated annotation tracing for debugging the chunked backward
    # (issue #148); prints every annotation created/matched/unpacked.
    if os.environ.get("KV_DEBUG_ANNOTATIONS") == "1":
        autograd_hooks_kwargs["debug_print_annotations"] = True
    # Bound the size of temporary device arrays in the backward. Without
    # this, single attention temporaries at 32k contexts exceed 2 GiB and
    # OOM an otherwise-fitting configuration. Mirrors the finetune path.
    _grad_limit_kwargs: Dict[str, Any] = {}
    if backward_tmp_gb > 0:
        _grad_limit_kwargs["backward_tmp_array_limit_gb"] = TemporaryArrayLimit(
            init_val=backward_tmp_gb,
            name="backward_tmp_array_limit_gb",
        )
    grad_model = LongContextGradientModel(
        gpt_model=gpt_model,
        head_model=head,
        layers_per_cell=layers_per_cell,
        chunk_size=chunk_size,
        verbose=verbose,
        train_cache_kwargs=train_cache_kwargs,
        autograd_hooks_kwargs=autograd_hooks_kwargs,
        **_grad_limit_kwargs,
    )
    grad_model.train()
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)

    # 6. Backward (+ optimizer step unless accumulating).
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with _phase_timer(times, "grad_time_ms", device):
        if dense_baseline:
            loss = _dense_baseline_backward(
                gpt_model=gpt_model,
                head=head,
                model_input_ids=model_input_ids,
                completions=completions,
                grad_scale=grad_scale,
            )
        elif grad_micro_batch is None or grad_micro_batch >= total:
            loss = grad_model(model_input_ids, completions, scale_factor=grad_scale)
            loss.backward()
            parked_bytes, parked_count = grad_model.last_parked_peak
        else:
            # Micro-batched: same gradient, device peak divided by ~G/mb
            loss, parked_bytes, parked_count = _chunked_micro_batched_backward(
                grad_model=grad_model,
                head=head,
                model_input_ids=model_input_ids,
                completions=completions,
                grad_scale=grad_scale,
                grad_micro_batch=grad_micro_batch,
            )
        if optimizer_step:
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
    # Memory accounting for the gradient pass. `grad_peak_device_mib` is the
    # device peak during backward (comparable across the chunked and dense
    # arms). `parked_peak_*` is the CPU memory retained by the issue-#148
    # chain-walk parking, the measured counterpart of the bound in
    # test_parked_memory_is_bounded_and_released.
    if device.type == "cuda":
        metrics["grad_peak_device_mib"] = torch.cuda.max_memory_allocated(device) / (
            1 << 20
        )
    if not dense_baseline:
        # Max over the micro-batch passes (one pass if grad_micro_batch=None)
        metrics["parked_peak_mib"] = parked_bytes / (1 << 20)
        metrics["parked_peak_count"] = float(parked_count)
        # SDPA mode of the gradient cells (flexattention / qpadded-pytorch)
        metrics["grad_sdpa_mode"] = gradient_sdpa_mode_name(
            gpt_model, train_cache_kwargs
        )
    if rescore_old_logps:
        # Quantify the rollout (decode) vs. training-forward log-prob skew over
        # real completion tokens -- a measure of the train/inference gap.
        with torch.no_grad():
            skew = ((gen_logps - old_logps).abs() * mask).sum() / mask.sum().clamp_min(
                1.0
            )
        metrics["logp_skew_decode_vs_forward"] = float(skew.item())
    if profile:
        metrics.update(times)
    return metrics
