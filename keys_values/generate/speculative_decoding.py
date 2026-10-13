# Original Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# Modification Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
Speculative decoding with KV caches.

A small draft model proposes `speculative_k` tokens, which a large target
model verifies in a single forward pass. Tokens are accepted or rejected by
the rejection sampling scheme of

    Leviathan, Kalman, Matias
    Fast Inference from Transformers via Speculative Decoding
    https://arxiv.org/abs/2211.17192

    Chen, Borgeaud, Irving, Lespiau, Sifre, Jumper
    Accelerating Large Language Model Decoding with Speculative Sampling
    https://arxiv.org/abs/2302.01318

so that every emitted token is distributed according to the target model,
using the same sampling parameters (`temperature`, `top_k`, `top_p`) as
:func:`keys_values.generate.base.generate`. With `temperature=0`, it
implements greedy target-model decoding in exact arithmetic; finite precision
can produce differences from token-by-token decoding (see below).

Draft tokens are inserted into the KV caches of both models as tentative
updates, and only the accepted ones are kept, see
:class:`keys_values.kvcache.tentative.TentativeKVCacheUpdates`. This supports
`dense` and `lastrec` caches, with default or quantized buffers.

Note on KV caches with eviction (for example, `lastrec`): The target model
verifies `speculative_k + 1` tokens in one :meth:`forward` call. New tokens
are inserted before attention is computed, so each of them can only attend
to cache content which survives the insertion of all of them. Once the cache
is full, outputs are therefore not distributed exactly as for token-by-token
decoding with the same cache, also when all draft tokens are rejected. This
is the same approximation as processing a prompt in chunks. For caches
without eviction (`dense`), the algorithm has the same distribution as
token-by-token decoding in exact arithmetic. In finite precision, verifying
multiple tokens in one call can produce different logits from single-token
calls, particularly with quantized buffers: within one call new tokens see
each other unquantized. Tentative updates themselves don't change the
verification: the target's call gives the same logits as the same call in
normal mode.

"""

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import torch
from torch.profiler import record_function

from keys_values.kvcache.factory import deallocate_kv_cache_buffers_of_model
from keys_values.kvcache.tentative import TentativeKVCacheUpdates
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT


@dataclass
class SpeculativeDecodingStats:
    """
    Statistics collected by :func:`speculative_generate_fn`. They describe
    the work done by the verification steps. If generation ends with a stop
    sequence, the final step is counted completely, including tokens after
    the stop sequence which are not returned. For throughput, count returned
    tokens instead.

    Args:
        num_steps: Number of target model forward calls after prompt
            processing (one per verification step)
        num_draft_tokens: Number of tokens proposed by the draft model
        num_accepted_tokens: Number of draft tokens accepted by the target
            model
        num_generated_tokens: Number of tokens produced by the verification
            steps (accepted draft tokens, plus one token sampled from the
            target model per step). This does not include the first token,
            which is sampled right after prompt processing.

    """

    num_steps: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_generated_tokens: int = 0

    @property
    def acceptance_rate(self) -> float:
        """
        Returns:
            Fraction of draft tokens which have been accepted

        """
        if self.num_draft_tokens == 0:
            return 0.0
        return self.num_accepted_tokens / self.num_draft_tokens

    @property
    def tokens_per_step(self) -> float:
        """
        Returns:
            Average number of tokens generated per target model call

        """
        if self.num_steps == 0:
            return 0.0
        return self.num_generated_tokens / self.num_steps


def _filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Batched version of :func:`litgpt.generate.base.sample_top_p`, operating
    on the final dimension.

    """
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least the most probable token
    sorted_to_remove[..., -1:] = False
    to_remove = sorted_to_remove.scatter(-1, sorted_indices, sorted_to_remove)
    return logits.masked_fill(to_remove, float("-inf"))


def is_greedy(temperature: float, top_p: float) -> bool:
    """
    Sampling is greedy if `temperature == 0` or `top_p == 0`, as in
    :func:`litgpt.generate.base.sample` from LitGPT 0.5.13 on. Up to LitGPT
    0.5.12, `sample` with `temperature == 0` and `top_p > 0` does not pick
    the most probable token, but samples from `softmax(logits)`. With these
    versions, :func:`keys_values.generate.base.generate` is not greedy for
    `temperature=0` (unless `top_p=0`), while speculative decoding is.

    Returns:
        Is sampling greedy for these arguments?

    """
    return not (temperature > 0.0 and top_p > 0.0)


def sampling_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Computes the distribution :func:`litgpt.generate.base.sample` draws from:
    `top_k` filtering, then `temperature` scaling, then `top_p` filtering.
    If `temperature == 0` or `top_p == 0`, sampling is greedy, and the
    distribution puts all mass on the most probable token.

    Rejection sampling requires these distributions, for both target and
    draft model. They must be the distributions tokens are actually drawn
    from, after all filtering.

    Args:
        logits: Logits, shape `(..., vocab_size)`
        temperature: See :func:`litgpt.generate.base.sample`
        top_k: See :func:`litgpt.generate.base.sample`
        top_p: See :func:`litgpt.generate.base.sample`

    Returns:
        Probabilities, shape `(..., vocab_size)`, `dtype=float32`

    """
    if not (0.0 <= top_p <= 1.0):
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    logits = logits.to(dtype=torch.float32)
    if top_k is not None:
        v, i = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, i, v)
    if not is_greedy(temperature, top_p):
        logits = logits / temperature
        if top_p < 1.0:
            logits = _filter_top_p(logits, top_p)
        return torch.softmax(logits, dim=-1)
    index = torch.argmax(logits, dim=-1, keepdim=True)
    return torch.zeros_like(logits).scatter_(-1, index, 1.0)


def _draw(probs: torch.Tensor, greedy: bool) -> torch.Tensor:
    """
    Draws a token from `probs`, shape `(vocab_size,)`, without synchronizing
    with the host.

    Returns:
        Token index, shape `(1,)`, on the device of `probs`

    """
    if greedy:
        return torch.argmax(probs, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def accept_or_reject(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    greedy: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rejection sampling step of speculative decoding. Draft token `i` is
    accepted with probability `min(1, p_i(x_i) / q_i(x_i))`, where `p_i` and
    `q_i` are the target and draft distributions for its position. At the
    first rejection, a replacement token is drawn from
    `normalize(max(p_i - q_i, 0))`. If all `k` draft tokens are accepted, an
    additional token is drawn from `p_k`, which is the same formula with
    `q_k = 0`. Every token returned here is distributed according to the
    target model.

    If `greedy == True`, the distributions are one-hot (see
    :func:`sampling_probs`), and this amounts to accepting draft tokens as
    long as they agree with the greedy choice of the target model.

    All computations stay on the device, so that the caller needs a single
    transfer to the host per step.

    Args:
        draft_tokens: Draft tokens `x_1, ..., x_k`, sampled from `q_i`, shape
            `(k,)`, where `k >= 0`
        draft_probs: Draft distributions `q_i`, shape `(k, vocab_size)`
        target_probs: Target distributions `p_i`, shape `(k + 1, vocab_size)`.
            The final row is used if all draft tokens are accepted.
        greedy: Is sampling greedy?

    Returns:
        `(num_accepted, next_token)`, both scalar `int64` tensors on the
        device of `target_probs`. The first `num_accepted` draft tokens are
        accepted, and `next_token` is to be appended to them.

    """
    device = target_probs.device
    num_draft = draft_tokens.numel()
    vocab_size = target_probs.shape[-1]
    zero_row = torch.zeros(
        (1, vocab_size), dtype=target_probs.dtype, device=target_probs.device
    )
    if num_draft > 0:
        index = draft_tokens.view(-1, 1).to(device=device, dtype=torch.int64)
        p = target_probs[:num_draft].gather(-1, index).squeeze(-1)
        q = draft_probs.gather(-1, index).squeeze(-1)
        if greedy:
            accepted = p > 0
        else:
            # Accept iff u < p / q, where u is uniform on [0, 1). Note that
            # q > 0, since draft tokens are sampled from q
            u = torch.rand(num_draft, device=device, dtype=p.dtype)
            accepted = u * q < p
        # Number of leading accepted tokens
        num_accepted = accepted.to(torch.int64).cumprod(dim=0).sum()
        q_ext = torch.cat((draft_probs.to(target_probs), zero_row), dim=0)
    else:
        num_accepted = torch.zeros((), dtype=torch.int64, device=device)
        q_ext = zero_row
    row = num_accepted.view(1)
    p_row = target_probs.index_select(0, row)
    residual = (p_row - q_ext.index_select(0, row)).clamp_min(0)
    total = residual.sum(dim=-1, keepdim=True)
    # `total > 0` holds in exact arithmetic: a rejection at position `i`
    # requires `p_i(x) < q_i(x)` for some `x`, hence `p_i(y) > q_i(y)` for
    # some `y`
    residual = torch.where(total > 0, residual / total.clamp_min(1e-30), p_row)
    next_token = _draw(residual[0], greedy)
    return num_accepted, next_token.view(())


class _StopSequenceMatcher:
    """
    Tracks the longest prefix of a stop sequence which is a suffix of the
    tokens seen so far, using the Knuth-Morris-Pratt failure function, so
    that overlapping occurrences are detected. For example, the stop sequence
    `[1, 2, 1, 2, 3]` is found in `[1, 2, 1, 2, 1, 2, 3]`.

    """

    def __init__(self, seq: List[int]):
        if len(seq) == 0:
            raise ValueError("Stop sequences must not be empty")
        self.seq = list(seq)
        # `failure[i]`: Length of longest proper prefix of `seq[: i + 1]`
        # which is also a suffix of it
        failure = [0] * len(seq)
        length = 0
        for i in range(1, len(seq)):
            while length > 0 and seq[i] != seq[length]:
                length = failure[length - 1]
            if seq[i] == seq[length]:
                length += 1
            failure[i] = length
        self._failure = failure
        self.num_matched = 0

    def append(self, token: int) -> bool:
        """
        Returns:
            Has the stop sequence been completed with `token`?

        """
        num = self.num_matched
        if num == len(self.seq):
            num = self._failure[num - 1]
        while num > 0 and token != self.seq[num]:
            num = self._failure[num - 1]
        if token == self.seq[num]:
            num += 1
        self.num_matched = num
        return num == len(self.seq)


class _TokenEmitter:
    """
    Decides which generated tokens can be yielded, given stop sequences.
    Tokens which may be the start of a stop sequence are held back. Once a
    stop sequence is complete, the remaining tokens are released, without
    the stop sequence if `include_eos == False`.

    """

    def __init__(self, stop_tokens: Tuple[List[int], ...], include_eos: bool):
        self._matchers = [_StopSequenceMatcher(seq) for seq in stop_tokens]
        self._include_eos = include_eos
        self._generated = []
        self._num_released = 0
        self.stopped = False

    def add(self, tokens: List[int]) -> List[int]:
        """
        Args:
            tokens: New tokens, in order

        Returns:
            Tokens which can be yielded now. If a stop sequence is completed,
            `stopped` is set, and tokens after it are dropped.

        """
        assert not self.stopped
        released = []
        for token in tokens:
            self._generated.append(token)
            completed = [len(m.seq) for m in self._matchers if m.append(token)]
            if completed:
                end = len(self._generated)
                if not self._include_eos:
                    end -= max(completed)
                released.extend(self._generated[self._num_released : end])
                self._num_released = end
                self.stopped = True
                break
            num_held_back = max((m.num_matched for m in self._matchers), default=0)
            safe_idx = len(self._generated) - num_held_back
            if safe_idx > self._num_released:
                released.extend(self._generated[self._num_released : safe_idx])
                self._num_released = safe_idx
        return released

    def flush(self) -> List[int]:
        """
        Returns:
            All tokens not yielded so far (end of generation)

        """
        released = self._generated[self._num_released :]
        self._num_released = len(self._generated)
        return released


def _check_models(
    target_model: LongContextInferenceModel,
    draft_model: LongContextInferenceModel,
) -> int:
    """
    Checks that models can be used for speculative decoding.

    Returns:
        Size of vocabulary to be used for both models

    """
    target_gpt = target_model.gpt_model
    draft_gpt = draft_model.gpt_model
    if target_gpt is draft_gpt:
        raise ValueError(
            "target_model and draft_model must be different models, each with "
            "its own KV caches"
        )
    target_caches = target_gpt.get_kv_caches()
    draft_caches = draft_gpt.get_kv_caches()
    if not set(map(id, target_caches)).isdisjoint(map(id, draft_caches)):
        raise ValueError("target_model and draft_model must not share KV caches")
    for kind, caches in (
        ("target_model", target_caches),
        ("draft_model", draft_caches),
    ):
        if any(cache is None for cache in caches):
            raise ValueError(f"{kind}: KV caches must be assigned to all layers")
        try:
            TentativeKVCacheUpdates(caches)
        except NotImplementedError as ex:
            raise ValueError(
                f"{kind}: Speculative decoding requires KV caches which support "
                f"tentative updates. {ex}"
            ) from ex
    target_config = target_gpt.config
    draft_config = draft_gpt.config
    if target_config.vocab_size != draft_config.vocab_size:
        raise ValueError(
            f"target_model and draft_model must have the same vocabulary, but "
            f"vocab_size = {target_config.vocab_size} and "
            f"{draft_config.vocab_size}"
        )
    return min(target_config.padded_vocab_size, draft_config.padded_vocab_size)


def _model_device(gpt_model: GPT) -> torch.device:
    return gpt_model.transformer.wte.weight.device


@torch.inference_mode()
def speculative_generate_fn(
    target_model: LongContextInferenceModel,
    draft_model: LongContextInferenceModel,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    *,
    speculative_k: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    stop_tokens: Tuple[List[int], ...] = (),
    include_prompt: bool,
    include_eos: bool,
    deallocate_cache_buffers: bool = True,
    stats: Optional[SpeculativeDecodingStats] = None,
) -> Iterator[torch.Tensor]:
    """
    Generates tokens for a single prompt by speculative decoding. Same
    semantics as :func:`keys_values.generate.base.generate_fn`, but there is
    a target and a draft model.

    In each step, the draft model generates up to `speculative_k` tokens,
    which are verified by the target model in a single forward call. Between
    1 and `speculative_k + 1` tokens are generated per step. Draft tokens are
    inserted into the KV caches of both models as tentative updates (see
    :class:`TentativeKVCacheUpdates`), and only the accepted ones are kept.

    Stop sequences are matched with the Knuth-Morris-Pratt algorithm, so
    overlapping occurrences are found, different from
    :func:`keys_values.generate.base.generate_fn`. If `include_eos == False`,
    all tokens before the completed stop sequence are returned.

    Phases of each step are annotated with `torch.profiler.record_function`
    (names starting with `speculative_decoding::`), so that their cost can
    be analyzed with `torch.profiler`.

    Args:
        target_model: Model to sample from. Must be
            :class:`LongContextInferenceModel`, defining the chunking used
            for prompt processing.
        draft_model: Model to propose tokens. Must be
            :class:`LongContextInferenceModel`, with its own KV caches. Must
            have the same vocabulary as `target_model`. It may be on a
            different device.
        prompt: The tokenized prompt to generate from, shape `(T,)`.
        max_returned_tokens: Maximum number of tokens to return, including
            the prompt tokens.
        speculative_k: Maximum number of draft tokens per step, must be
            positive.
        temperature: The temp to pass to sample().
        top_k: The top_k to pass to sample().
        top_p: The top_p to pass to sample().
        stop_tokens: A tuple of stop sequences. If any of the sequences are
            generated, the generation stops early before max_returned_tokens.
        include_prompt: Whether to output the prompt tokens.
        include_eos: Whether to output the stop tokens if generation stops
            early.
        deallocate_cache_buffers: Whether to deallocate KV cache buffers of
            both models at the end.
        stats: If given, statistics are written into this object.

    Yields:
        The prompt (if `include_prompt`), then generated tokens, each of
        shape `(1,)`, on the device of `prompt`

    """
    if speculative_k < 1:
        raise ValueError(f"speculative_k = {speculative_k}, must be >= 1")
    if prompt.ndim == 2 and prompt.shape[0] > 1:
        raise NotImplementedError("Speculative decoding supports batch size 1 only")
    prompt = prompt.flatten()
    prompt_size = prompt.numel()
    if prompt_size == 0:
        raise ValueError("prompt must not be empty")
    if max_returned_tokens <= prompt_size:
        raise ValueError(
            f"max_returned_tokens = {max_returned_tokens}, must be larger than "
            f"the prompt size {prompt_size}"
        )
    vocab_size = _check_models(target_model, draft_model)
    emitter = _TokenEmitter(stop_tokens, include_eos)
    greedy = is_greedy(temperature, top_p)
    sample_kwargs = dict(temperature=temperature, top_k=top_k, top_p=top_p)
    if stats is None:
        stats = SpeculativeDecodingStats()
    target_gpt = target_model.gpt_model
    draft_gpt = draft_model.gpt_model
    target_gpt.max_seq_length = max_returned_tokens
    draft_gpt.max_seq_length = max_returned_tokens
    target_device = _model_device(target_gpt)
    draft_device = _model_device(draft_gpt)
    out_device = prompt.device
    target_updates = TentativeKVCacheUpdates(target_gpt.get_kv_caches())
    draft_updates = TentativeKVCacheUpdates(draft_gpt.get_kv_caches())

    def probs(logits: torch.Tensor) -> torch.Tensor:
        return sampling_probs(logits[..., :vocab_size], **sample_kwargs)

    def to_output(tokens: List[int]) -> Iterator[torch.Tensor]:
        # A single copy to `out_device` per step: host-to-device copies
        # synchronize the device with the host
        if tokens:
            tensor = torch.tensor(tokens, device=out_device, dtype=torch.int64)
            for pos in range(len(tokens)):
                yield tensor[pos : pos + 1]

    try:
        if include_prompt:
            yield prompt

        # Prompt processing for both models. This is done in chunks by the
        # long context inference models. The final logits of the draft model
        # are not needed, since the first token is sampled from the target.
        with record_function("speculative_decoding::prefill"):
            logits = target_model(prompt.unsqueeze(0).to(target_device), targets=None)
            draft_model(prompt.unsqueeze(0).to(draft_device), targets=None)
            token = _draw(probs(logits[0, -1]), greedy)  # (1,)
        num_tokens = prompt_size + 1  # Prompt and generated tokens
        step_tokens = [token.item()]

        # Invariants at the start of each step:
        # - `token` is the most recent token, shape `(1,)` on the target
        #   device. It has not been inserted into the target KV caches yet
        # - `draft_pending` are the most recent tokens which have not been
        #   inserted into the draft KV caches yet. Its last entry is `token`
        draft_pending = [token]
        while True:
            yield from to_output(emitter.add(step_tokens))
            if emitter.stopped:
                return
            if num_tokens >= max_returned_tokens:
                yield from to_output(emitter.flush())
                return

            # Number of draft tokens for this step. At most `num_draft + 1`
            # tokens are generated, which must fit into `max_returned_tokens`.
            # Also, KV caches limit the length of `forward` calls: the target
            # model is called with `num_draft + 1` tokens, the draft model
            # is called with `draft_pending`, then `num_draft - 1` more tokens
            # are inserted as tentative updates.
            target_max_length = target_gpt.kv_cache_max_forward_length()
            if target_max_length < 1:
                raise IndexError("Target model: KV caches cannot take more tokens")
            num_draft = min(
                speculative_k,
                max_returned_tokens - num_tokens - 1,
                target_max_length - 1,
                draft_gpt.kv_cache_max_forward_length() - len(draft_pending) + 1,
            )
            if num_draft > 0:
                # Draft phase: Generate `num_draft` tokens. The pending
                # tokens are inserted first; they are always kept.
                with record_function("speculative_decoding::draft"):
                    logits = draft_gpt(
                        torch.cat(draft_pending).to(draft_device).view(1, -1)
                    )
                    draft_pending = []
                    if num_draft > 1:
                        draft_updates.start()
                    draft_tokens = []
                    draft_probs = []
                    for pos in range(num_draft):
                        if pos > 0:
                            logits = draft_gpt(draft_tokens[-1].view(1, 1))
                        q = probs(logits[0, -1])
                        draft_tokens.append(_draw(q, greedy))
                        draft_probs.append(q)
                    draft_tokens = torch.cat(draft_tokens).to(target_device)
                    draft_probs = torch.stack(draft_probs).to(target_device)
            else:
                draft_tokens = torch.zeros(0, dtype=torch.int64, device=target_device)
                draft_probs = None

            # Verification: The target model processes `token` and the draft
            # tokens in a single call
            with record_function("speculative_decoding::verify"):
                if num_draft > 0:
                    target_updates.start()
                logits = target_gpt(torch.cat((token, draft_tokens)).view(1, -1))
            with record_function("speculative_decoding::accept"):
                num_accepted, next_token = accept_or_reject(
                    draft_tokens=draft_tokens,
                    draft_probs=draft_probs,
                    target_probs=probs(logits[0]),
                    greedy=greedy,
                )
                # Single transfer to the host per step
                host = torch.cat(
                    (num_accepted.view(1), next_token.view(1), draft_tokens)
                ).tolist()
            num_accepted = host[0]

            # Keep `token` and the accepted draft tokens in the KV caches. The
            # tentative updates of the target caches are `token` and all draft
            # tokens, those of the draft caches all draft tokens except for
            # the first and the last one.
            with record_function("speculative_decoding::keep"):
                if num_draft > 0:
                    target_updates.keep(num_accepted + 1)
                if num_draft > 1:
                    draft_updates.keep(min(num_accepted, num_draft - 1))
            token = next_token.view(1)
            if num_draft > 0 and num_accepted == num_draft:
                draft_pending.append(draft_tokens[-1:])
            draft_pending.append(token)

            stats.num_steps += 1
            stats.num_draft_tokens += num_draft
            stats.num_accepted_tokens += num_accepted
            stats.num_generated_tokens += num_accepted + 1
            step_tokens = host[2 : 2 + num_accepted] + [host[1]]
            num_tokens += num_accepted + 1
    finally:
        # If generation stopped in the middle of a step (for example, by an
        # exception), tentative updates are discarded
        for updates in (target_updates, draft_updates):
            if updates.is_active:
                updates.discard()
        if deallocate_cache_buffers:
            deallocate_kv_cache_buffers_of_model(target_gpt)
            deallocate_kv_cache_buffers_of_model(draft_gpt)


def speculative_generate(
    target_model: LongContextInferenceModel,
    draft_model: LongContextInferenceModel,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    *,
    speculative_k: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    eos_id: Optional[int] = None,
    include_prompt: bool = True,
    deallocate_cache_buffers: bool = True,
    stats: Optional[SpeculativeDecodingStats] = None,
) -> torch.Tensor:
    """
    Same as :func:`keys_values.generate.base.generate`, but uses speculative
    decoding with a draft model. See :func:`speculative_generate_fn` for
    details.

    Args:
        target_model: Model to sample from
        draft_model: Model to propose tokens, with its own KV caches
        prompt: Tensor of shape `(T,)` with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given
            plus generated).
        speculative_k: Maximum number of draft tokens per step
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest
            probabilities.
        top_p: See :func:`keys_values.generate.base.generate`.
        eos_id: If specified, stop generating any more token once the <eos>
            token is triggered.
        include_prompt: If true (default) prepends the prompt to the output.
        deallocate_cache_buffers: Whether to deallocate KV cache buffers at
            the end.
        stats: If given, statistics are written into this object.

    Returns:
        Prompt (if `include_prompt`) and generated tokens

    """
    token_list = list(
        speculative_generate_fn(
            target_model=target_model,
            draft_model=draft_model,
            prompt=prompt,
            max_returned_tokens=max_returned_tokens,
            speculative_k=speculative_k,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_tokens=(([eos_id],) if eos_id is not None else ()),
            include_prompt=include_prompt,
            include_eos=True,
            deallocate_cache_buffers=deallocate_cache_buffers,
            stats=stats,
        )
    )
    return torch.cat(token_list) if len(token_list) > 0 else torch.Tensor()
