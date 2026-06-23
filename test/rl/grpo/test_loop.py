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
