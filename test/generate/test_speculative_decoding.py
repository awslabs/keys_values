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
import itertools
import os
from typing import Dict, List, Optional

import pytest
import torch
from scipy.stats import chi2

from litgpt.generate.base import sample_top_p

from keys_values.config import Config
from keys_values.generate.base import generate, generate_fn
import keys_values.generate.speculative_decoding as spec
from keys_values.generate.speculative_decoding import (
    SpeculativeDecodingStats,
    accept_or_reject,
    sampling_probs,
    speculative_generate,
    speculative_generate_fn,
)
from keys_values.kvcache.buffers import DefaultKVCacheBuffers
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.kvcache.quant_buffers import QuantizedKVCacheBuffers
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT
from keys_values.utils import VerbosityLevels


def _config(vocab_size: int) -> Config:
    return Config(
        block_size=128,
        vocab_size=vocab_size,
        padded_vocab_size=vocab_size,
        n_layer=2,
        n_head=4,
        n_embd=32,
        rotary_percentage=1.0,
    )


def _gpt(seed: int, vocab_size: int, noise: float = 0.0) -> GPT:
    """
    Random model in double precision, so that greedy decisions do not depend
    on how tokens are chunked. If `noise > 0`, parameters are perturbed,
    which is used to create draft models agreeing partially with the target.

    """
    torch.random.manual_seed(seed)
    gpt_model = GPT(_config(vocab_size)).to(dtype=torch.float64).eval()
    if noise > 0:
        gen = torch.Generator().manual_seed(seed + 9999)
        with torch.no_grad():
            for param in gpt_model.parameters():
                scale = noise * param.std().clamp_min(1e-3)
                param.add_(scale * torch.randn(param.shape, generator=gen).to(param))
    return gpt_model


def _inference_model(
    gpt_model: GPT,
    max_seq_length: int,
    cache_name: str = "dense",
    cache_length: Optional[int] = None,
) -> LongContextInferenceModel:
    gpt_model.max_seq_length = max_seq_length
    if cache_name == "dense":
        gpt_model.set_kv_caches(batch_size=1, dtype=torch.float64)
    else:
        gpt_model.assign_kv_caches(
            KVCacheFactory.create(
                gpt_model=gpt_model,
                name=cache_name,
                max_batch_size=1,
                cache_length=max_seq_length if cache_length is None else cache_length,
                dtype=torch.float64,
                cache_kwargs=(
                    dict(init_grace_tokens=2)
                    if cache_name.startswith("lastrec")
                    else None
                ),
            )
        )
    return LongContextInferenceModel(
        gpt_model=gpt_model,
        head_model=None,
        chunk_size=4,
        verbose=VerbosityLevels.NONE,
    )


# Draft models: identical weights, partially agreeing, unrelated
DRAFT_KINDS = {
    "identical": dict(seed=1, noise=0.0),
    "close": dict(seed=1, noise=0.2),
    "far": dict(seed=1, noise=0.4),
    "unrelated": dict(seed=2, noise=0.0),
}


# Greedy decoding with plain generation, for all LitGPT versions. Up to LitGPT
# 0.5.12, `sample` with `temperature=0` and `top_p > 0` does not pick the most
# probable token, but samples from `softmax(logits)`, see `is_greedy`.
GREEDY_REFERENCE = dict(temperature=1.0, top_k=1)


def _prompt(seed: int, length: int, vocab_size: int) -> torch.Tensor:
    return torch.randint(
        0, vocab_size, (length,), generator=torch.Generator().manual_seed(seed)
    )


@pytest.mark.parametrize(
    "temperature, top_k, top_p",
    [
        (1.0, None, 1.0),
        (0.7, None, 1.0),
        (1.3, 5, 1.0),
        (0.8, None, 0.9),
        (0.5, 10, 0.75),
        (1.0, 1, 1.0),
        (0.0, None, 1.0),
        (1.0, None, 0.0),
        (0.0, 3, 0.5),
    ],
)
def test_sampling_probs_matches_litgpt(temperature, top_k, top_p):
    torch.random.manual_seed(31415)
    logits = torch.randn(7, 20) * 3
    probs = sampling_probs(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    assert probs.shape == logits.shape
    for row, prob_row in zip(logits, probs):
        # Same steps as in `litgpt.generate.base.sample` (LitGPT 0.5.13 and
        # later, see `is_greedy`)
        x = row.clone()
        if top_k is not None:
            v, i = torch.topk(x, min(top_k, x.size(-1)))
            x = torch.full_like(x, float("-inf")).scatter_(-1, i, v)
        if temperature > 0.0 and top_p > 0.0:
            x = x / temperature
            if top_p < 1.0:
                x = sample_top_p(x, top_p)
            expected = torch.softmax(x, dim=-1)
        else:
            expected = torch.zeros_like(x)
            expected[torch.argmax(x)] = 1.0
        torch.testing.assert_close(prob_row, expected)


def _outside_support(counts: torch.Tensor, probs: torch.Tensor) -> int:
    return int(counts.flatten()[probs.flatten() == 0].sum().item())


def _chi2_pvalue(counts: torch.Tensor, probs: torch.Tensor) -> float:
    """
    Pearson chi-square test of `counts` against distribution `probs`. Cells
    with small expected counts are merged into one cell. Samples outside the
    support of `probs` rule it out, and 0 is returned.

    """
    if _outside_support(counts, probs) > 0:
        return 0.0
    counts = counts.flatten().to(torch.float64)
    expected = probs.flatten().to(torch.float64) * counts.sum()
    small = expected < 5
    obs = torch.cat([counts[~small], counts[small].sum().view(1)])
    exp = torch.cat([expected[~small], expected[small].sum().view(1)])
    keep = exp > 0
    obs, exp = obs[keep], exp[keep]
    stat = ((obs - exp) ** 2 / exp).sum().item()
    return float(chi2.sf(stat, df=obs.numel() - 1))


def test_accept_or_reject_is_lossless():
    """
    Monte Carlo test of the rejection sampling step, with fixed target and
    draft distributions per position. Emitted tokens must follow the target
    distributions, not the draft ones. This includes the extra token drawn
    if all draft tokens are accepted.

    """
    torch.random.manual_seed(2718)
    vocab_size, num_draft, num_samples = 6, 3, 10000
    target_probs = torch.softmax(torch.randn(num_draft + 1, vocab_size) * 1.5, -1)
    draft_probs = torch.softmax(torch.randn(num_draft, vocab_size) * 1.5, -1)
    # Support of draft larger than support of target in one row
    target_probs[1, 0] = 0
    target_probs[1] /= target_probs[1].sum()
    counts_first = torch.zeros(vocab_size)
    counts_second = torch.zeros(vocab_size)
    counts_extra = torch.zeros(vocab_size)
    counts_draft = torch.zeros(vocab_size)
    num_accepted_hist = torch.zeros(num_draft + 1)
    for _ in range(num_samples):
        draft_tokens = torch.multinomial(draft_probs, 1).flatten()
        num_accepted, next_token = accept_or_reject(
            draft_tokens, draft_probs, target_probs, greedy=False
        )
        assert num_accepted.shape == () and next_token.shape == ()
        num_accepted = num_accepted.item()
        emitted = draft_tokens[:num_accepted].tolist() + [next_token.item()]
        counts_first[emitted[0]] += 1
        counts_draft[draft_tokens[0]] += 1
        if len(emitted) > 1:
            counts_second[emitted[1]] += 1
        if num_accepted == num_draft:
            counts_extra[emitted[-1]] += 1
        num_accepted_hist[num_accepted] += 1
    # First emitted token follows `target_probs[0]`. Given that there is a
    # second one in this step, it follows `target_probs[1]`. The extra token
    # follows the final row.
    assert _chi2_pvalue(counts_first, target_probs[0]) > 1e-4
    assert _chi2_pvalue(counts_second, target_probs[1]) > 1e-4
    assert _chi2_pvalue(counts_extra, target_probs[num_draft]) > 1e-4
    # Power check: Draft tokens themselves are clearly distinguished
    assert _chi2_pvalue(counts_draft, target_probs[0]) < 1e-10
    # All acceptance counts occur
    assert torch.all(num_accepted_hist > 0)


def test_accept_or_reject_no_draft_tokens():
    torch.random.manual_seed(1)
    target_probs = torch.softmax(torch.randn(1, 5), -1)
    counts = torch.zeros(5)
    for _ in range(4000):
        num_accepted, next_token = accept_or_reject(
            torch.zeros(0, dtype=torch.int64), None, target_probs, greedy=False
        )
        assert num_accepted.item() == 0
        counts[next_token.item()] += 1
    assert _chi2_pvalue(counts, target_probs[0]) > 1e-4


def test_accept_or_reject_greedy():
    vocab_size = 5
    argmaxes = [3, 1, 4, 0]
    target_probs = torch.zeros(4, vocab_size)
    target_probs[torch.arange(4), torch.tensor(argmaxes)] = 1.0
    for draft_tokens, expected in [
        ([3, 1, 4], (3, 0)),
        ([3, 1, 2], (2, 4)),
        ([2, 1, 4], (0, 3)),
        ([3, 0, 4], (1, 1)),
    ]:
        draft_probs = torch.zeros(3, vocab_size)
        draft_probs[torch.arange(3), torch.tensor(draft_tokens)] = 1.0
        num_accepted, next_token = accept_or_reject(
            torch.tensor(draft_tokens), draft_probs, target_probs, greedy=True
        )
        assert (num_accepted.item(), next_token.item()) == expected


@pytest.mark.parametrize(
    "cache_name, draft_kind, speculative_k, sample_kwargs",
    [
        (cache_name, draft_kind, speculative_k, sample_kwargs)
        for cache_name, draft_kind, speculative_k, sample_kwargs in itertools.product(
            ["dense", "lastrec-default"],
            list(DRAFT_KINDS.keys()),
            [1, 4],
            [dict(temperature=0.0), dict(temperature=1.0, top_k=1)],
        )
    ],
)
def test_greedy_matches_generate(cache_name, draft_kind, speculative_k, sample_kwargs):
    """
    Greedy speculative decoding must produce the greedy output of the target
    model, whatever the draft model. For `lastrec`, the cache is large enough
    so that nothing is evicted.

    """
    vocab_size, prompt_len, max_returned_tokens = 32, 6, 32
    for prompt_seed in range(2):
        prompt = _prompt(100 + prompt_seed, prompt_len, vocab_size)
        expected = generate(
            _inference_model(_gpt(1, vocab_size), max_returned_tokens),
            prompt,
            max_returned_tokens,
            **GREEDY_REFERENCE,
        )
        stats = SpeculativeDecodingStats()
        output = speculative_generate(
            target_model=_inference_model(
                _gpt(1, vocab_size), max_returned_tokens, cache_name
            ),
            draft_model=_inference_model(
                _gpt(vocab_size=vocab_size, **DRAFT_KINDS[draft_kind]),
                max_returned_tokens,
                cache_name,
            ),
            prompt=prompt,
            max_returned_tokens=max_returned_tokens,
            speculative_k=speculative_k,
            stats=stats,
            **sample_kwargs,
        )
        assert output.tolist() == expected.tolist(), (prompt_seed, stats)
        assert stats.num_generated_tokens + 1 == max_returned_tokens - prompt_len
        if draft_kind == "identical":
            assert stats.acceptance_rate == 1.0


@pytest.mark.parametrize("top_p", [1.0, 0.5])
def test_temperature_zero_is_greedy(top_p):
    """
    With `temperature=0`, speculative decoding is greedy, whatever `top_p`
    and the LitGPT version. The output does not depend on the random seed,
    and is the greedy output of the target model, also if draft tokens are
    rejected.

    """
    assert spec.is_greedy(0.0, top_p)
    assert spec.is_greedy(0.7, 0.0)
    assert not spec.is_greedy(0.7, top_p)
    logits = torch.randn(5, 16)
    probs = sampling_probs(logits, temperature=0.0, top_p=top_p)
    expected = torch.zeros_like(probs).scatter_(-1, logits.argmax(-1, True), 1.0)
    assert probs.equal(expected)

    vocab_size, prompt_len, max_returned_tokens = 32, 6, 32
    prompt = _prompt(102, prompt_len, vocab_size)
    expected = generate(
        _inference_model(_gpt(1, vocab_size), max_returned_tokens),
        prompt,
        max_returned_tokens,
        **GREEDY_REFERENCE,
    ).tolist()
    for seed in range(3):
        torch.random.manual_seed(seed)
        stats = SpeculativeDecodingStats()
        output = speculative_generate(
            target_model=_inference_model(_gpt(1, vocab_size), max_returned_tokens),
            draft_model=_inference_model(
                _gpt(vocab_size=vocab_size, **DRAFT_KINDS["far"]),
                max_returned_tokens,
            ),
            prompt=prompt,
            max_returned_tokens=max_returned_tokens,
            speculative_k=4,
            temperature=0.0,
            top_p=top_p,
            stats=stats,
        )
        assert output.tolist() == expected, seed
        assert stats.acceptance_rate < 1.0


def _exact_joint_distribution(
    gpt_model: GPT,
    prompt: torch.Tensor,
    num_new: int,
    sample_kwargs: Dict,
) -> torch.Tensor:
    """
    Computes the joint distribution of the next `num_new` tokens by evaluating
    the model on all continuations, without KV caches.

    """
    vocab_size = gpt_model.config.padded_vocab_size
    continuations = torch.tensor(
        list(itertools.product(range(vocab_size), repeat=num_new))
    )
    inputs = torch.cat(
        [prompt.unsqueeze(0).expand(continuations.shape[0], -1), continuations],
        dim=1,
    )
    with torch.no_grad():
        logits = gpt_model(inputs)
    prompt_len = prompt.numel()
    probs = sampling_probs(
        logits[:, prompt_len - 1 : prompt_len - 1 + num_new, :], **sample_kwargs
    )
    token_probs = probs.gather(-1, continuations.unsqueeze(-1)).squeeze(-1)
    return token_probs.prod(dim=-1).view(*([vocab_size] * num_new))


@pytest.mark.parametrize(
    "cache_name, sample_kwargs, draft_noise",
    [
        ("dense", dict(temperature=0.7, top_k=2, top_p=0.95), 1.5),
        ("lastrec-default", dict(temperature=1.2), 0.5),
    ],
)
def test_sampling_is_lossless(cache_name, sample_kwargs, draft_noise):
    """
    End-to-end Monte Carlo test: The joint distribution of the first four
    generated tokens must be the one of the target model. With
    `speculative_k = 2`, the second step proposes two draft tokens, so all of
    acceptance, rejection with partially kept tentative updates, and the extra token if all
    are accepted are exercised.

    """
    vocab_size, num_new, num_samples = 3, 4, 1500
    prompt = _prompt(7, 5, vocab_size)
    max_returned_tokens = prompt.numel() + num_new
    target_gpt = _gpt(1, vocab_size)
    draft_gpt = _gpt(1, vocab_size, noise=draft_noise)
    exact = _exact_joint_distribution(target_gpt, prompt, num_new, sample_kwargs)
    exact_draft = _exact_joint_distribution(draft_gpt, prompt, num_new, sample_kwargs)
    target_model = _inference_model(target_gpt, max_returned_tokens, cache_name)
    draft_model = _inference_model(draft_gpt, max_returned_tokens, cache_name)
    torch.random.manual_seed(12345)
    counts = torch.zeros_like(exact)
    stats = SpeculativeDecodingStats()
    for _ in range(num_samples):
        output = speculative_generate(
            target_model=target_model,
            draft_model=draft_model,
            prompt=prompt,
            max_returned_tokens=max_returned_tokens,
            speculative_k=2,
            include_prompt=False,
            stats=stats,
            **sample_kwargs,
        )
        counts[tuple(output.tolist())] += 1
    # Draft model must be rejected sometimes and accepted sometimes
    assert 0.2 < stats.acceptance_rate < 0.95, stats
    assert _outside_support(counts, exact) == 0
    assert _chi2_pvalue(counts, exact) > 1e-4
    # Power check: With this sample size, the draft distribution is clearly
    # distinguished from the target one
    draft_counts = torch.bincount(
        torch.multinomial(exact_draft.flatten(), num_samples, replacement=True),
        minlength=exact.numel(),
    )
    assert _chi2_pvalue(draft_counts, exact) < 1e-6


@pytest.mark.parametrize(
    "draft_kind, sample_kwargs",
    list(
        itertools.product(
            ["identical", "close", "far"],
            [dict(temperature=0.0), dict(temperature=1.0, top_k=8)],
        )
    ),
)
def test_lastrec_with_eviction(draft_kind, sample_kwargs):
    """
    With small `lastrec` caches, slots are overwritten and restored many
    times when tentative updates are discarded. Afterwards, both caches must
    contain exactly the initial and most recent tokens of the sequence, and
    use their original buffers again.

    """
    vocab_size, prompt_len, max_returned_tokens = 32, 6, 80
    cache_length, init_grace_tokens = 12, 2
    torch.random.manual_seed(5)
    prompt = _prompt(3, prompt_len, vocab_size)
    target_model = _inference_model(
        _gpt(1, vocab_size), max_returned_tokens, "lastrec-default", cache_length
    )
    draft_model = _inference_model(
        _gpt(vocab_size=vocab_size, **DRAFT_KINDS[draft_kind]),
        max_returned_tokens,
        "lastrec-default",
        cache_length,
    )
    stats = SpeculativeDecodingStats()
    output = speculative_generate(
        target_model=target_model,
        draft_model=draft_model,
        prompt=prompt,
        max_returned_tokens=max_returned_tokens,
        speculative_k=4,
        deallocate_cache_buffers=False,
        stats=stats,
        **sample_kwargs,
    )
    assert output.numel() == max_returned_tokens
    if draft_kind != "identical":
        assert stats.acceptance_rate < 1.0
    # The target caches contain all tokens except for the final one. The
    # draft caches lack one or two tokens, or three if the final step does
    # not use the draft model.
    for model, num_missing in ((target_model, (1,)), (draft_model, (1, 2, 3))):
        for kv_cache in model.gpt_model.get_kv_caches():
            num_inserted = kv_cache.input_pos
            assert max_returned_tokens - num_inserted in num_missing
            positions = sorted(kv_cache.token_positions()[0, 0, :].tolist())
            expected = list(range(init_grace_tokens)) + list(
                range(num_inserted - cache_length + init_grace_tokens, num_inserted)
            )
            assert positions == expected
            assert type(kv_cache.kv_buffers) is DefaultKVCacheBuffers


@pytest.mark.parametrize(
    "cache_name, cache_length",
    [("lastrec-torch-quantized8", 12), ("dense-torch-quantized8", 64)],
)
def test_quantized_caches(cache_name, cache_length):
    """
    Speculative decoding with quantized KV caches, whose tentative updates
    are kept in replay buffers. With quantization, outputs are not the same as
    for plain generation, so we check the cache states at the end, and that a
    draft model with the same weights is accepted most of the time.

    """
    vocab_size, prompt_len, max_returned_tokens = 32, 6, 60
    init_grace_tokens = 2
    torch.random.manual_seed(6)
    prompt = _prompt(4, prompt_len, vocab_size)

    def inference_model(seed: int) -> LongContextInferenceModel:
        gpt_model = _gpt(seed, vocab_size).to(dtype=torch.float32)
        gpt_model.max_seq_length = max_returned_tokens
        gpt_model.assign_kv_caches(
            KVCacheFactory.create(
                gpt_model=gpt_model,
                name=cache_name,
                max_batch_size=1,
                cache_length=cache_length,
                dtype=torch.float32,
                cache_kwargs=(
                    dict(init_grace_tokens=init_grace_tokens)
                    if cache_name.startswith("lastrec")
                    else None
                ),
            )
        )
        return LongContextInferenceModel(
            gpt_model=gpt_model,
            head_model=None,
            chunk_size=4,
            verbose=VerbosityLevels.NONE,
        )

    target_model, draft_model = inference_model(1), inference_model(1)
    stats = SpeculativeDecodingStats()
    output = speculative_generate(
        target_model=target_model,
        draft_model=draft_model,
        prompt=prompt,
        max_returned_tokens=max_returned_tokens,
        speculative_k=3,
        temperature=0.0,
        deallocate_cache_buffers=False,
        stats=stats,
    )
    assert output.numel() == max_returned_tokens
    assert stats.acceptance_rate > 0.8
    for model, num_missing in ((target_model, (1,)), (draft_model, (1, 2, 3))):
        for kv_cache in model.gpt_model.get_kv_caches():
            assert type(kv_cache.kv_buffers) is QuantizedKVCacheBuffers
            num_inserted = kv_cache.input_pos
            assert max_returned_tokens - num_inserted in num_missing
            positions = sorted(kv_cache.token_positions()[0, 0, :].tolist())
            if cache_name.startswith("lastrec"):
                expected = list(range(init_grace_tokens)) + list(
                    range(num_inserted - cache_length + init_grace_tokens, num_inserted)
                )
            else:
                expected = list(range(num_inserted))
            assert positions == expected


def _expected_with_stop_tokens(
    tokens: List[int],
    stop_tokens,
    include_eos: bool,
) -> List[int]:
    """
    Brute-force reference for stop sequences: Generation stops at the first
    position where some stop sequence ends. Tokens up to there are returned,
    without the stop sequence (the longest one ending there) if
    `include_eos == False`.

    """
    for pos in range(len(tokens)):
        completed = [
            len(seq)
            for seq in stop_tokens
            if len(seq) <= pos + 1 and tokens[pos + 1 - len(seq) : pos + 1] == seq
        ]
        if completed:
            end = pos + 1 if include_eos else pos + 1 - max(completed)
            return tokens[:end]
    return tokens


def test_stop_sequence_matcher_overlapping():
    from keys_values.generate.speculative_decoding import _StopSequenceMatcher

    matcher = _StopSequenceMatcher([1, 2, 1, 2, 3])
    assert [matcher.append(t) for t in [1, 2, 1, 2, 1, 2, 3]] == [False] * 6 + [True]
    matcher = _StopSequenceMatcher([4, 4, 5])
    assert [matcher.append(t) for t in [4, 4, 4, 5]] == [False] * 3 + [True]


@pytest.mark.parametrize("include_eos", [True, False])
def test_token_emitter_matches_reference(include_eos):
    """
    Random token streams over a small alphabet, so that stop sequences
    overlap often, fed to the emitter in random step sizes. The concatenated
    output must equal the brute-force reference, and nothing yielded may be
    retracted (which holds by construction, since outputs are only appended).

    """
    from keys_values.generate.speculative_decoding import _TokenEmitter

    gen = torch.Generator().manual_seed(13)

    def randint(low, high):
        return int(torch.randint(low, high, (1,), generator=gen))

    for _ in range(3000):
        stop_tokens = tuple(
            [randint(0, 3) for _ in range(randint(1, 5))] for _ in range(randint(1, 4))
        )
        tokens = [randint(0, 3) for _ in range(randint(1, 30))]
        emitter = _TokenEmitter(stop_tokens, include_eos)
        output = []
        pos = 0
        while pos < len(tokens) and not emitter.stopped:
            num = randint(1, 6)
            output.extend(emitter.add(tokens[pos : pos + num]))
            pos += num
        if not emitter.stopped:
            output.extend(emitter.flush())
        expected = _expected_with_stop_tokens(tokens, stop_tokens, include_eos)
        assert output == expected, (stop_tokens, tokens, output, expected)


@pytest.mark.parametrize("include_eos", [True, False])
def test_stop_tokens(include_eos):
    vocab_size, prompt_len, max_returned_tokens = 32, 7, 45
    prompt = _prompt(101, prompt_len, vocab_size)
    reference = generate(
        _inference_model(_gpt(1, vocab_size), max_returned_tokens),
        prompt,
        max_returned_tokens,
        include_prompt=False,
        **GREEDY_REFERENCE,
    ).tolist()
    stop_token_cases = [
        ([reference[9]],),
        ([reference[12], reference[13]],),
        ([vocab_size + 5],),  # Never generated
        ([reference[4]], [reference[20], reference[21], reference[22]]),
        ([reference[17], reference[18], reference[19]],),
        # Overlapping: a prefix of the stop sequence occurs right before it
        ([reference[15], reference[16], reference[15], reference[16]],),
    ]
    for stop_tokens in stop_token_cases:
        expected = _expected_with_stop_tokens(reference, stop_tokens, include_eos)
        if all(len(seq) == 1 for seq in stop_tokens):
            # Same as without speculative decoding
            expected_base = generate_fn(
                _inference_model(_gpt(1, vocab_size), max_returned_tokens),
                prompt,
                max_returned_tokens,
                stop_tokens=stop_tokens,
                include_prompt=False,
                include_eos=include_eos,
                **GREEDY_REFERENCE,
            )
            assert _token_list(list(expected_base)) == expected
        output = list(
            speculative_generate_fn(
                target_model=_inference_model(_gpt(1, vocab_size), max_returned_tokens),
                draft_model=_inference_model(
                    _gpt(vocab_size=vocab_size, **DRAFT_KINDS["close"]),
                    max_returned_tokens,
                ),
                prompt=prompt,
                max_returned_tokens=max_returned_tokens,
                speculative_k=4,
                temperature=0.0,
                stop_tokens=stop_tokens,
                include_prompt=False,
                include_eos=include_eos,
            )
        )
        assert _token_list(output) == expected, stop_tokens
        assert all(t.shape == (1,) for t in output)


def test_include_prompt_and_short_budget():
    vocab_size, prompt_len = 32, 7
    prompt = _prompt(5, prompt_len, vocab_size)
    for max_returned_tokens in (prompt_len + 1, prompt_len + 2, prompt_len + 3):
        target = _inference_model(_gpt(1, vocab_size), max_returned_tokens)
        draft = _inference_model(
            _gpt(vocab_size=vocab_size, **DRAFT_KINDS["identical"]),
            max_returned_tokens,
        )
        output = speculative_generate(
            target, draft, prompt, max_returned_tokens, speculative_k=5, temperature=0
        )
        assert output.numel() == max_returned_tokens
        assert output[:prompt_len].tolist() == prompt.tolist()


def test_errors():
    vocab_size, max_returned_tokens = 32, 20
    prompt = _prompt(5, 5, vocab_size)

    def models(target_cache="dense", draft_cache="dense", draft_vocab=vocab_size):
        return (
            _inference_model(_gpt(1, vocab_size), max_returned_tokens, target_cache),
            _inference_model(_gpt(2, draft_vocab), max_returned_tokens, draft_cache),
        )

    kwargs = dict(prompt=prompt, max_returned_tokens=max_returned_tokens)
    target, draft = models()
    with pytest.raises(ValueError):
        speculative_generate(target, draft, speculative_k=0, **kwargs)
    with pytest.raises(ValueError):
        speculative_generate(target, target, speculative_k=2, **kwargs)
    with pytest.raises(NotImplementedError):
        speculative_generate(
            target,
            draft,
            speculative_k=2,
            prompt=prompt.view(1, -1).expand(2, -1),
            max_returned_tokens=max_returned_tokens,
        )
    with pytest.raises(ValueError):
        speculative_generate(
            target, draft, speculative_k=2, prompt=prompt, max_returned_tokens=5
        )
    target, draft = models(target_cache="h2o-default")
    with pytest.raises(ValueError, match="tentative updates"):
        speculative_generate(target, draft, speculative_k=2, **kwargs)
    target, draft = models(draft_vocab=vocab_size + 4)
    with pytest.raises(ValueError, match="vocabulary"):
        speculative_generate(target, draft, speculative_k=2, **kwargs)


def _token_list(tokens: List[torch.Tensor]) -> List[int]:
    return [t.item() for t in tokens]


def _num_cuda_devices() -> int:
    from keys_values.kvcache.test_utils import ENV_VAR_SKIP_CUDA_TESTS

    if os.environ.get(ENV_VAR_SKIP_CUDA_TESTS) is not None:
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


@pytest.mark.skipif(_num_cuda_devices() < 1, reason="Needs a CUDA device")
@pytest.mark.parametrize("cache_name", ["dense", "lastrec-default"])
def test_prompt_on_cpu_models_on_cuda(cache_name):
    vocab_size, prompt_len, max_returned_tokens = 32, 6, 32
    device = torch.device("cuda:0")
    prompt = _prompt(11, prompt_len, vocab_size)

    def models():
        return (
            _inference_model(
                _gpt(1, vocab_size).to(device), max_returned_tokens, cache_name
            ),
            _inference_model(
                _gpt(vocab_size=vocab_size, **DRAFT_KINDS["close"]).to(device),
                max_returned_tokens,
                cache_name,
            ),
        )

    expected = generate(
        _inference_model(_gpt(1, vocab_size).to(device), max_returned_tokens),
        prompt.to(device),
        max_returned_tokens,
        temperature=0.0,
    )
    target, draft = models()
    output = speculative_generate(
        target, draft, prompt, max_returned_tokens, speculative_k=4, temperature=0.0
    )
    assert output.device == prompt.device
    assert output.tolist() == expected.tolist()


@pytest.mark.skipif(_num_cuda_devices() < 2, reason="Needs two CUDA devices")
def test_draft_on_other_cuda_device():
    vocab_size, prompt_len, max_returned_tokens = 32, 6, 32
    prompt = _prompt(12, prompt_len, vocab_size)
    target_device, draft_device = torch.device("cuda:0"), torch.device("cuda:1")
    expected = generate(
        _inference_model(_gpt(1, vocab_size).to(target_device), max_returned_tokens),
        prompt.to(target_device),
        max_returned_tokens,
        temperature=0.0,
    )
    output = speculative_generate(
        _inference_model(_gpt(1, vocab_size).to(target_device), max_returned_tokens),
        _inference_model(
            _gpt(vocab_size=vocab_size, **DRAFT_KINDS["close"]).to(draft_device),
            max_returned_tokens,
        ),
        prompt,
        max_returned_tokens,
        speculative_k=4,
        temperature=0.0,
    )
    assert output.tolist() == expected.tolist()
