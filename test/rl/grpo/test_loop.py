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
End-to-end tests for the standalone GRPO loop on KeysAndValues components.

These run on CPU with a tiny model, exercising the full pipeline:
generation -> reward -> advantages -> old log-probs -> policy gradient ->
optimizer step.
"""

import pytest
import torch

from keys_values.config import Config
from keys_values.rl.grpo.loop import compute_group_advantages, grpo_step
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.model import GPT


def test_compute_group_advantages_basic():
    """Advantages should be zero-mean within each group."""
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    adv = compute_group_advantages(rewards, group_size=3)

    # Each group normalized to ~zero mean
    g0 = adv[:3]
    g1 = adv[3:]
    assert abs(g0.mean().item()) < 1e-5
    assert abs(g1.mean().item()) < 1e-5
    # Higher reward in a group -> higher advantage
    assert g0[2] > g0[0]
    assert g1[2] > g1[0]


def test_compute_group_advantages_constant_group():
    """A group with identical rewards yields ~zero advantages (no blow-up)."""
    rewards = torch.tensor([5.0, 5.0, 5.0, 5.0])
    adv = compute_group_advantages(rewards, group_size=4)
    assert torch.isfinite(adv).all()
    assert adv.abs().max().item() < 1e-3


def _make_model_with_caches(batch_size, cache_length, dtype=torch.float32):
    config = Config(
        block_size=512,
        vocab_size=64,
        padded_vocab_size=64,
        n_layer=2,
        n_head=4,
        n_embd=32,
        n_query_groups=2,
        intermediate_size=64,
        rotary_percentage=1,
    )
    torch.set_default_dtype(dtype)
    with torch.device("cpu"):
        gpt_model = GPT(config)
        gpt_model.apply(gpt_model._init_weights)
    # Non-dense caches (required by LongContextGradientModel)
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            gpt_model=gpt_model,
            name="lastrec-default",
            max_batch_size=batch_size,
            cache_length=cache_length,
            dtype=dtype,
        )
    )
    return gpt_model, config


def test_grpo_step_runs_and_updates_params():
    """A full GRPO step should produce a finite loss and update parameters."""
    torch.manual_seed(0)
    num_prompts, prompt_len = 2, 8
    group_size = 2
    max_new_tokens = 6
    cache_length = 32

    batch_size = num_prompts * group_size
    gpt_model, config = _make_model_with_caches(batch_size, cache_length)

    prompt_ids = torch.randint(0, config.vocab_size, (num_prompts, prompt_len))

    # Reward: prefer completions with higher mean token id (arbitrary but
    # deterministic signal)
    def reward_fn(prompts, completions):
        return completions.float().mean(dim=1)

    optimizer = torch.optim.SGD(gpt_model.parameters(), lr=0.01)

    # Snapshot all parameters to verify at least one changes
    before = [p.detach().clone() for p in gpt_model.parameters()]

    metrics = grpo_step(
        gpt_model=gpt_model,
        prompt_ids=prompt_ids,
        reward_fn=reward_fn,
        optimizer=optimizer,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
        chunk_size=16,
        temperature=1.0,  # sample (not greedy) so completions differ in a group
    )

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["total_completions"] == batch_size
    # At least one parameter should have moved (non-zero advantages -> grad)
    after = list(gpt_model.parameters())
    changed = any(not torch.equal(b, a.detach()) for b, a in zip(before, after))
    assert changed, "No parameter updated; expected non-zero GRPO gradient"


def test_grpo_step_multiple_iterations():
    """Several GRPO steps should run without error and keep loss finite."""
    torch.manual_seed(1)
    num_prompts, prompt_len = 2, 6
    group_size = 2
    cache_length = 24

    batch_size = num_prompts * group_size
    gpt_model, config = _make_model_with_caches(batch_size, cache_length)
    prompt_ids = torch.randint(0, config.vocab_size, (num_prompts, prompt_len))

    def reward_fn(prompts, completions):
        return completions.float().mean(dim=1)

    optimizer = torch.optim.SGD(gpt_model.parameters(), lr=0.01)

    for _ in range(3):
        metrics = grpo_step(
            gpt_model=gpt_model,
            prompt_ids=prompt_ids,
            reward_fn=reward_fn,
            optimizer=optimizer,
            group_size=group_size,
            max_new_tokens=5,
            chunk_size=16,
            top_k=1,
        )
        assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_grpo_step_with_backward_tmp_limit():
    """
    `backward_tmp_gb > 0` must actually work end-to-end.

    Regression guard: the limit is constructed inside `grpo_step`, so a
    missing import is invisible to `import keys_values.rl.grpo.loop` and only
    fails at call time -- which is how a GPU job died with
    `NameError: name 'TemporaryArrayLimit' is not defined` after the module
    imported cleanly.
    """
    torch.manual_seed(0)
    num_prompts, prompt_len = 2, 8
    group_size = 2
    max_new_tokens = 6
    cache_length = 32

    batch_size = num_prompts * group_size
    gpt_model, config = _make_model_with_caches(batch_size, cache_length)
    prompt_ids = torch.randint(0, config.vocab_size, (num_prompts, prompt_len))

    def reward_fn(prompts, completions):
        return completions.float().mean(dim=1)

    optimizer = torch.optim.SGD(gpt_model.parameters(), lr=0.01)
    metrics = grpo_step(
        gpt_model=gpt_model,
        prompt_ids=prompt_ids,
        reward_fn=reward_fn,
        optimizer=optimizer,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
        chunk_size=16,
        temperature=1.0,
        backward_tmp_gb=2.0,
    )
    assert torch.isfinite(torch.tensor(metrics["loss"]))


@pytest.mark.parametrize("per_sequence_norm", [False, True])
def test_dense_baseline_matches_chunked_gradient(per_sequence_norm):
    """
    The dense baseline must compute the SAME loss and the SAME gradients as
    the chunked path, so that paper comparisons isolate memory/feasibility
    rather than a different objective.

    Same seed -> same rollout, advantages and mask; the only difference is
    the gradient computation (full-sequence backward vs chunked cells).

    Run for both normalizations: `per_sequence_norm=True` moves the division
    into the head's forward and makes `num_target_entries` a no-op, and both
    arms must agree on that too.
    """
    num_prompts, prompt_len = 2, 8
    group_size = 2
    max_new_tokens = 6
    cache_length = 32

    def run(dense: bool):
        torch.manual_seed(7)
        batch_size = num_prompts * group_size
        gpt_model, config = _make_model_with_caches(batch_size, cache_length)
        prompt_ids = torch.randint(0, config.vocab_size, (num_prompts, prompt_len))

        def reward_fn(prompts, completions):
            return completions.float().mean(dim=1)

        optimizer = torch.optim.SGD(gpt_model.parameters(), lr=0.0)  # no update
        torch.manual_seed(11)  # identical sampling in both arms
        metrics = grpo_step(
            gpt_model=gpt_model,
            prompt_ids=prompt_ids,
            reward_fn=reward_fn,
            optimizer=optimizer,
            group_size=group_size,
            max_new_tokens=max_new_tokens,
            chunk_size=16,
            temperature=1.0,
            dense_baseline=dense,
            eos_token_id=5,  # on this seed: early stops -> rows of 2, 4, 6, 6
            per_sequence_norm=per_sequence_norm,
        )
        # Rows of different length, otherwise the two normalizations coincide
        # and the per_sequence_norm=True arm would test nothing new.
        assert metrics["mean_completion_tokens"] < max_new_tokens
        grads = {
            name: p.grad.detach().clone()
            for name, p in gpt_model.named_parameters()
            if p.grad is not None
        }
        return metrics, grads

    m_chunked, g_chunked = run(dense=False)
    m_dense, g_dense = run(dense=True)

    # Same rollout (sanity: the arms really are comparable)
    assert m_chunked["total_completions"] == m_dense["total_completions"]
    torch.testing.assert_close(
        torch.tensor(m_chunked["mean_reward"]),
        torch.tensor(m_dense["mean_reward"]),
    )
    # Same loss
    torch.testing.assert_close(
        torch.tensor(m_chunked["loss"]),
        torch.tensor(m_dense["loss"]),
        rtol=1e-4,
        atol=1e-5,
    )
    # Same gradients
    assert g_chunked and g_dense, "expected gradients in both arms"
    assert set(g_chunked) == set(g_dense)
    for name in g_chunked:
        torch.testing.assert_close(
            g_chunked[name], g_dense[name], rtol=2e-3, atol=2e-4, msg=name
        )


@pytest.mark.parametrize("per_sequence_norm", [False, True])
@pytest.mark.parametrize("grad_micro_batch", [1, 3])
def test_grad_micro_batch_matches_full_batch_chunked_gradient(
    grad_micro_batch, per_sequence_norm
):
    """
    `grad_micro_batch` exists only to cut the gradient-phase memory peak
    (every term of the chunked backward carries the batch size); it must not
    change the objective. Gradients and loss over the 4-sequence batch,
    accumulated over micro-batches of 1 (4 passes) and of 3 (a non-divisor:
    passes of 3 and 1 sequences, so the per-pass `scale_factor` must undo
    the model's own mean over a *different* micro-batch size and token-count
    mean each time), must equal the single full-batch pass, under both
    normalizations. Rollout, reward and advantages are computed before the
    backward and must be identical.
    """
    num_prompts, prompt_len = 2, 8
    group_size = 2
    max_new_tokens = 6
    cache_length = 32

    def run(mb):
        torch.manual_seed(7)
        batch_size = num_prompts * group_size
        gpt_model, config = _make_model_with_caches(batch_size, cache_length)
        prompt_ids = torch.randint(0, config.vocab_size, (num_prompts, prompt_len))

        def reward_fn(prompts, completions):
            return completions.float().mean(dim=1)

        optimizer = torch.optim.SGD(gpt_model.parameters(), lr=0.0)  # no update
        torch.manual_seed(11)  # identical sampling in both arms
        metrics = grpo_step(
            gpt_model=gpt_model,
            prompt_ids=prompt_ids,
            reward_fn=reward_fn,
            optimizer=optimizer,
            group_size=group_size,
            max_new_tokens=max_new_tokens,
            chunk_size=16,
            temperature=1.0,
            eos_token_id=5,  # on this seed: early stops -> rows of 2, 4, 6, 6
            per_sequence_norm=per_sequence_norm,
            grad_micro_batch=mb,
        )
        # Rows of different length, otherwise group-mean and per-sequence
        # normalization coincide and the scale_factor bookkeeping is untested.
        assert metrics["mean_completion_tokens"] < max_new_tokens
        grads = {
            name: p.grad.detach().clone()
            for name, p in gpt_model.named_parameters()
            if p.grad is not None
        }
        return metrics, grads

    m_full, g_full = run(None)
    m_micro, g_micro = run(grad_micro_batch)

    # Same rollout, reward and advantages (computed before the backward)
    for key in (
        "total_completions",
        "completion_len",
        "mean_reward",
        "mean_advantage",
        "advantage_std",
        "mean_completion_tokens",
    ):
        assert m_full[key] == m_micro[key], key
    # Same loss value (sum of the micro-batch losses)
    torch.testing.assert_close(
        torch.tensor(m_micro["loss"]),
        torch.tensor(m_full["loss"]),
        rtol=1e-5,
        atol=1e-6,
    )
    # Same gradients (fp32 CPU; only the accumulation order differs)
    assert g_full and g_micro, "expected gradients in both arms"
    assert set(g_full) == set(g_micro)
    for name in g_full:
        torch.testing.assert_close(
            g_micro[name], g_full[name], rtol=1e-4, atol=1e-6, msg=name
        )
    # Memory metrics are still reported for the whole step
    assert "parked_peak_mib" in m_micro and "parked_peak_count" in m_micro


def test_grad_micro_batch_rejects_zero():
    gpt_model, config = _make_model_with_caches(2, 32)
    prompt_ids = torch.randint(0, config.vocab_size, (1, 8))
    optimizer = torch.optim.SGD(gpt_model.parameters(), lr=0.0)
    with pytest.raises(ValueError, match="grad_micro_batch"):
        grpo_step(
            gpt_model=gpt_model,
            prompt_ids=prompt_ids,
            reward_fn=lambda p, c: c.float().mean(dim=1),
            optimizer=optimizer,
            group_size=2,
            max_new_tokens=4,
            chunk_size=16,
            grad_micro_batch=0,
        )


def test_grpo_step_trains_lora_adapters():
    """
    LoRA adapters must receive gradients through the chunked backward and
    change the policy's outputs. Added after two LoRA arms sat at exactly
    the base-model eval for 90 steps: this test rules out the pipeline
    (gradient routing, optimizer, eval path) so that a flat curve can be
    attributed to the training recipe, not to a silent no-op.

    `lora_B` is zero-initialised, so on the first step only `lora_B` gets a
    non-zero gradient; `lora_A` follows once `lora_B` has moved. Hence the
    check is over several steps, and on the greedy completion rather than
    on a single gradient.
    """
    from keys_values.lora import (
        GPT as GPTLoRA,
        Config as ConfigLoRA,
        mark_only_lora_as_trainable,
    )
    from keys_values.long_context import LongContextInferenceModel
    from keys_values.rl.grpo.rollout import generate_completions
    from keys_values.utils import VerbosityLevels

    torch.manual_seed(0)
    num_prompts, prompt_len, group_size, max_new_tokens = 2, 8, 2, 6
    cache_length = 32
    batch_size = num_prompts * group_size
    config = ConfigLoRA(
        block_size=512,
        vocab_size=64,
        padded_vocab_size=64,
        n_layer=2,
        n_head=4,
        n_embd=32,
        n_query_groups=2,
        intermediate_size=64,
        rotary_percentage=1,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_query=True,
        lora_key=True,
        lora_value=True,
        lora_projection=True,
        lora_mlp=True,
        lora_head=False,
    )
    torch.set_default_dtype(torch.float32)
    with torch.device("cpu"):
        gpt_model = GPTLoRA(config)
        gpt_model.apply(gpt_model._init_weights)
    mark_only_lora_as_trainable(gpt_model)
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            gpt_model=gpt_model,
            name="lastrec-default",
            max_batch_size=batch_size,
            cache_length=cache_length,
            dtype=torch.float32,
        )
    )
    trainable = [p for p in gpt_model.parameters() if p.requires_grad]
    frozen = [p for p in gpt_model.parameters() if not p.requires_grad]
    assert trainable and frozen
    optimizer = torch.optim.AdamW(trainable, lr=1e-2)
    prompt_ids = torch.randint(0, config.vocab_size, (num_prompts, prompt_len))

    def reward_fn(prompts, completions):
        return completions.float().mean(dim=1)

    def greedy_completion() -> torch.Tensor:
        # Same path as the drivers' eval: inference model + greedy decode
        gpt_model.eval()
        ids = prompt_ids[:1]
        gpt_model.max_seq_length = ids.shape[1] + max_new_tokens
        inf = LongContextInferenceModel(
            gpt_model, head_model=None, chunk_size=16, verbose=VerbosityLevels.NONE
        )
        with torch.no_grad():
            comp = generate_completions(
                model=inf,
                prompt_ids=ids,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_k=1,
                top_p=1.0,
                eos_token_id=None,
                pad_token_id=0,
                no_inference_mode=True,
            )
        return comp[0].clone()

    before_comp = greedy_completion()
    before_lora = [p.detach().clone() for p in trainable]
    before_frozen = [p.detach().clone() for p in frozen]
    # First step: every LoRA module must receive a gradient on lora_B
    grpo_step(
        gpt_model=gpt_model,
        prompt_ids=prompt_ids,
        reward_fn=reward_fn,
        optimizer=optimizer,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
        chunk_size=16,
        temperature=1.0,
        optimizer_step=False,
    )
    lora_b = [
        p for n, p in gpt_model.named_parameters() if p.requires_grad and "lora_B" in n
    ]
    assert lora_b
    for p in lora_b:
        assert p.grad is not None and p.grad.abs().max() > 0
    optimizer.step()
    for _ in range(5):
        grpo_step(
            gpt_model=gpt_model,
            prompt_ids=prompt_ids,
            reward_fn=reward_fn,
            optimizer=optimizer,
            group_size=group_size,
            max_new_tokens=max_new_tokens,
            chunk_size=16,
            temperature=1.0,
        )
    # All adapter tensors moved, base weights did not, and the policy changed
    assert all(not torch.equal(b, a.detach()) for b, a in zip(before_lora, trainable))
    assert all(torch.equal(b, a.detach()) for b, a in zip(before_frozen, frozen))
    assert not torch.equal(before_comp, greedy_completion())
