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
import torch

from keys_values.config import Config
from keys_values.rl.grpo.loss import GRPOLossHeadModel


def test_loss_head_temperature_matches_rollout_logprobs():
    """The rollout records log-probs under softmax(logits / T). Given explicit
    old log-probs from that distribution and the same parameters, the loss
    head must reproduce them, so the ratio is exactly 1 for any T. Before the
    fix the head ignored T and the ratio was off whenever T != 1."""
    torch.manual_seed(0)
    bs, num, vocab, temp = 2, 5, 16, 0.7
    config = Config()
    config.padded_vocab_size = vocab
    logits = torch.randn(bs, num, vocab)
    targets = torch.randint(0, vocab, (bs, num))
    # What the rollout records (rollout.sampled_token_logprobs semantics)
    rollout_logps = (
        torch.log_softmax(logits / temp, dim=-1)
        .gather(-1, targets.unsqueeze(-1))
        .squeeze(-1)
    )
    advantages = torch.tensor([1.0, -1.0])

    head = GRPOLossHeadModel(config, temperature=temp)
    head.set_batch(advantages=advantages, old_logps=rollout_logps)
    loss = head(logits, targets, input_pos=0)
    # ratio == 1 -> per-token loss == -advantage, summed over `num` tokens
    torch.testing.assert_close(loss, -advantages * num, rtol=1e-5, atol=1e-5)

    head_wrong = GRPOLossHeadModel(config, temperature=1.0)
    head_wrong.set_batch(advantages=advantages, old_logps=rollout_logps)
    loss_wrong = head_wrong(logits, targets, input_pos=0)
    assert not torch.allclose(loss_wrong, -advantages * num, atol=1e-3)


def _cfg(vocab):
    config = Config()
    config.padded_vocab_size = vocab
    return config


def _two_length_batch(vocab=8, num=16, lengths=(4, 16)):
    """Two sequences with 4 and 16 valid tokens; uniform (all-zero) logits so
    every token has the same gradient magnitude and only the normalization
    differs between rows."""
    torch.manual_seed(3)
    bs = len(lengths)
    logits = torch.zeros(bs, num, vocab, requires_grad=True)
    targets = torch.randint(0, vocab, (bs, num))
    mask = torch.zeros(bs, num)
    for b, n in enumerate(lengths):
        mask[b, :n] = 1.0
    advantages = torch.full((bs,), 0.5)
    return logits, targets, mask, advantages


def _normalized_loss(head, logits, targets):
    """Apply the chunked path's normalization (accumulate.py: `scale_factor /
    num_target_entries.mean()` with average_loss_per_batch=True, then the
    batch mean)."""
    per_seq = head(logits, targets, input_pos=0)
    return (per_seq / head.num_target_entries(targets).mean()).mean()


def test_per_sequence_norm_default_is_bit_identical_to_summed_loss():
    """No-regression guarantee: with the default `per_sequence_norm=False`
    the head returns exactly the masked per-token sum it returned before the
    option existed, and `num_target_entries` is exactly the per-row mask
    count. Reference formula written out inline."""
    logits, targets, mask, advantages = _two_length_batch()
    logits = torch.randn(*logits.shape)  # non-trivial logits for this check
    head = GRPOLossHeadModel(_cfg(logits.shape[-1]))
    assert head.per_sequence_norm is False
    head.set_batch(advantages=advantages, mask=mask)
    loss = head(logits, targets, input_pos=0)

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    ratio = torch.exp(logp - logp.detach())
    adv = advantages.unsqueeze(1)
    ref = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv) * mask
    assert torch.equal(loss, ref.sum(dim=-1))
    assert torch.equal(
        head.num_target_entries(targets), mask.sum(dim=-1).clamp_min(1.0)
    )


def test_per_sequence_norm_equalizes_gradient_mass_across_lengths():
    """Lengths 4 and 16, equal advantages. Default normalization divides every
    token by the group-mean count (10), so the long sequence's total
    gradient is 4x the short one's. With `per_sequence_norm=True` each row is
    divided by its own count and the two totals are equal."""
    vocab = 8

    def row_grad_mass(per_sequence_norm):
        logits, targets, mask, advantages = _two_length_batch(vocab=vocab)
        head = GRPOLossHeadModel(_cfg(vocab), per_sequence_norm=per_sequence_norm)
        head.set_batch(advantages=advantages, mask=mask)
        _normalized_loss(head, logits, targets).backward()
        # d loss / d logit[target] is identical for every valid token (uniform
        # logits), so the per-row sum measures that row's gradient mass.
        g = logits.grad.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return g.sum(dim=-1)

    g_default = row_grad_mass(False)
    assert g_default[0] != 0
    torch.testing.assert_close(g_default[1] / g_default[0], torch.tensor(4.0))

    g_per_seq = row_grad_mass(True)
    assert g_per_seq[0] != 0
    torch.testing.assert_close(g_per_seq[0], g_per_seq[1])
    # The per-sequence mode is the per-row mean token loss: num_target_entries
    # is all ones so the downstream normalization is a no-op.
    head = GRPOLossHeadModel(_cfg(vocab), per_sequence_norm=True)
    _, targets, mask, advantages = _two_length_batch(vocab=vocab)
    head.set_batch(advantages=advantages, mask=mask)
    assert torch.equal(head.num_target_entries(targets), torch.ones(2))


def test_per_sequence_norm_chunked_forward_matches_single_call():
    """The per-row division uses the full-completion count, so feeding the
    completion in two chunks (as the chunked gradient path does) sums to the
    same per-sequence loss as one call."""
    logits, targets, mask, advantages = _two_length_batch(vocab=8)
    logits = torch.randn(*logits.shape)
    head = GRPOLossHeadModel(_cfg(8), per_sequence_norm=True)
    head.set_batch(advantages=advantages, mask=mask)
    whole = head(logits, targets, input_pos=0)
    part = head(logits[:, :6], targets[:, :6], input_pos=0) + head(
        logits[:, 6:], targets[:, 6:], input_pos=6
    )
    torch.testing.assert_close(whole, part)
    # Row 0 has 4 valid tokens, row 1 has 16: both are the mean token loss
    # (ratio == 1, so each valid token contributes -advantage / count).
    torch.testing.assert_close(whole, -advantages, rtol=1e-5, atol=1e-5)


def test_empty_clone_carries_per_sequence_norm():
    head = GRPOLossHeadModel(
        _cfg(8), epsilon_low=0.1, epsilon_high=0.3, temperature=0.7
    )
    clone = head._empty_clone()
    assert clone.per_sequence_norm is False
    head = GRPOLossHeadModel(_cfg(8), temperature=0.7, per_sequence_norm=True)
    clone = head._empty_clone()
    assert clone.per_sequence_norm is True
    assert clone.temperature == 0.7


def test_loss_head_bf16_logits_give_fp32_loss():
    """Loss-side softmax runs in fp32 even for bf16 logits."""
    config = Config()
    config.padded_vocab_size = 8
    head = GRPOLossHeadModel(config)
    head.set_batch(advantages=torch.tensor([0.5]))
    logits = torch.randn(1, 3, 8).to(torch.bfloat16)
    targets = torch.randint(0, 8, (1, 3))
    loss = head(logits, targets, input_pos=0)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, torch.tensor([-1.5]))
