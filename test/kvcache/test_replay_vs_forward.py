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
Replay buffers must give the same logits as processing the same tokens with
ordinary KV cache buffers.

Setup: Process a prompt, then a small random number of further tokens, and
take the logits for these tokens.

* A: Prompt and tokens are processed by calling the model, with
  non-quantized default buffers being updated explicitly.
* B: Prompt is processed in the same way, but with buffers which can be
  switched to replay buffers. The tokens are then processed with replay
  buffers switched in.

A and B must give the same logits, with default `assert_close` tolerances,
for every attention implementation, FlexAttention included.

In the extended version, the tokens are processed several times, with
different numbers of tokens (drawn at random) and different token values.
For B, the prompt is processed only once, and the replay buffers are rolled
back after each trial.

Why this matters: `test_generate_old_new` in `test_replay_buffers.py` used to
fail with FlexAttention (logits off by up to 1.2e-3 in float16), which looked
like a replay bug but was not. All FlexAttention graphs shared one
`torch.compile` cache. Once several shapes had been seen, dynamo recompiled
with dynamic shapes, so the kernel used for a call depended on which calls
came before it. Two runs doing the same math in a different order then gave
slightly different logits. With `separate_compile_caches=True` in
`FlexAttentionArgs`, each graph is compiled with its own cache and static
shapes (see `compile_with_own_cache`), so results do not depend on the order
of calls. This test and `test_replay_buffers.py` use this option.

For other tests: if two code paths using FlexAttention differ by small,
unexplained amounts, check whether they call compiled graphs in a different
order before loosening tolerances, and use `separate_compile_caches=True`.

"""

from itertools import product
from typing import Any, Dict, List, Optional

import pytest
import torch

from keys_values.attention.flex_attention import FlexAttentionArgs
from keys_values.config import Config
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.kvcache.replay_buffers import (
    ModelForTokenGeneration,
    ReplayKVCacheBuffers,
)
from keys_values.kvcache.test_utils import available_backends
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT

VOCAB_SIZE = 128
CHUNK_SIZE = 16
# A prompt shorter than all caches (caches are not full when the tokens are
# processed), and one longer than all caches (eviction, ring buffer wraps
# around)
PROMPT_LENGTHS = (40, 300)
# Trials of the extended test. The number of tokens processed in each trial
# is drawn at random from `1, ..., MAX_NUM_TOKENS`, different for each trial
NUM_TRIALS = 4
MAX_NUM_TOKENS = 5

# FlexAttention without padding, and with padding of `kv_len` and `q_len`,
# which is how it is used in practice to limit the number of compiled
# graphs. With the latter, `kv_len` is padded for the short prompt, and
# `q_len` is padded for 2, 3 and 5 tokens.
FLEX_CONFIGS = {
    "flex": dict(),
    "flex-padded": dict(
        kv_lens=[48, 64, 96, 128, 160, 256, 320, 384, 448],
        q_lens=[1, 4, 8, 16],
    ),
}


def _flexatt_args(attention: str) -> Optional[FlexAttentionArgs]:
    kwargs = FLEX_CONFIGS.get(attention)
    if kwargs is None:
        return None
    # See "Why this matters" above
    return FlexAttentionArgs(**kwargs, separate_compile_caches=True)


class _Contender:
    """
    Model with KV caches of a certain buffer type. Both contenders share the
    same weights, but use different cache objects.

    """

    def __init__(
        self,
        gpt_model: GPT,
        cache_name: str,
        cache_kwargs: Dict[str, Any],
        cache_lengths: List[int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        for_replay: bool,
    ):
        self.gpt_model = gpt_model
        # B: Buffers of type `TestingKVCacheBuffers`, which store keys and
        # values without quantization, but can be switched to replay buffers.
        # A: Default buffers, no quantization
        name = cache_name + ("-torch-quantized8" if for_replay else "-default")
        self.caches = KVCacheFactory.create(
            gpt_model=gpt_model,
            name=name,
            max_batch_size=batch_size,
            cache_length=cache_lengths,
            device=device,
            dtype=dtype,
            cache_kwargs=cache_kwargs,
            for_testing=for_replay,
        )
        self.for_replay = for_replay

    def activate(self):
        self.gpt_model.assign_kv_caches(self.caches)

    def process_prompt(self, prompt: torch.Tensor, max_seq_length: int):
        self.activate()
        self.gpt_model.reset()
        self.gpt_model.max_seq_length = max_seq_length
        model = LongContextInferenceModel(
            gpt_model=self.gpt_model,
            head_model=None,
            chunk_size=CHUNK_SIZE,
        ).eval()
        return model(prompt, targets=None)

    def process_tokens(self, tokens: torch.Tensor, one_by_one: bool) -> torch.Tensor:
        """
        Returns:
            Logits for all positions of `tokens`, `(batch_size, num, vocab_size)`

        """
        if one_by_one:
            return torch.cat(
                [self.gpt_model(token.unsqueeze(1)) for token in tokens.unbind(dim=1)],
                dim=1,
            )
        else:
            return self.gpt_model(tokens)


def _setup(
    cache_name: str,
    cache_kwargs: Dict[str, Any],
    cache_lengths: List[int],
    attention: str,
    device: torch.device,
    batch_size: int = 3,
    dtype: torch.dtype = torch.float16,
):
    torch.set_default_dtype(dtype)
    flexatt_args = _flexatt_args(attention)
    mha_kwargs = dict() if flexatt_args is None else dict(flexatt_args=flexatt_args)
    n_head = 8
    head_size = 64
    config = Config(
        n_layer=len(cache_lengths),
        n_head=n_head,
        n_query_groups=4,
        n_embd=n_head * head_size,
        block_size=1024,
        vocab_size=VOCAB_SIZE,
        rotary_percentage=1,
    )
    with torch.device(device):
        gpt_model = GPT(config, **mha_kwargs)
        gpt_model.apply(gpt_model._init_weights)
    gpt_model.eval()
    _cache_kwargs = {**cache_kwargs, **mha_kwargs}
    contenders = [
        _Contender(
            gpt_model,
            cache_name,
            _cache_kwargs,
            cache_lengths,
            batch_size,
            device,
            dtype,
            for_replay,
        )
        for for_replay in (False, True)
    ]
    return gpt_model, contenders[0], contenders[1]


def _assert_same_logits(logits_a: torch.Tensor, logits_b: torch.Tensor, msg: str):
    torch.testing.assert_close(logits_b, logits_a, msg=lambda m: f"{msg}\n{m}")


def _attention_choices(device: torch.device) -> List[str]:
    # FlexAttention is only used on CUDA
    if device.type == "cuda":
        return ["default"] + list(FLEX_CONFIGS.keys())
    return ["default"]


CACHE_SETUPS = [
    ("lastrec", dict(), [96, 128]),
    ("dense", dict(), [384, 384]),
    ("h2o", dict(replay_log_blocksize=64), [96, 128]),
]


def args_replay_vs_forward():
    return [
        (
            cache_name,
            cache_kwargs,
            cache_lengths,
            attention,
            device,
            prompt_length,
            one_by_one,
        )
        for device in available_backends()
        for (
            cache_name,
            cache_kwargs,
            cache_lengths,
        ), attention, prompt_length, one_by_one in product(
            CACHE_SETUPS,
            _attention_choices(device),
            PROMPT_LENGTHS,
            (False, True),
        )
    ]


@pytest.mark.parametrize(
    "cache_name, cache_kwargs, cache_lengths, attention, device, prompt_length, one_by_one",
    args_replay_vs_forward(),
)
@torch.inference_mode()
def test_replay_vs_forward(
    cache_name,
    cache_kwargs,
    cache_lengths,
    attention,
    device,
    prompt_length,
    one_by_one,
):
    seed = 31415927 + prompt_length + 2 * int(one_by_one)
    torch.random.manual_seed(seed)
    gpt_model, model_a, model_b = _setup(
        cache_name, cache_kwargs, cache_lengths, attention, device
    )
    batch_size = 3
    prompt = torch.randint(0, VOCAB_SIZE, (batch_size, prompt_length), device=device)
    max_seq_length = prompt_length + MAX_NUM_TOKENS
    num_tokens_per_trial = (torch.randperm(MAX_NUM_TOKENS)[:NUM_TRIALS] + 1).tolist()

    # B: The prompt is processed only once
    prompt_logits_b = model_b.process_prompt(prompt, max_seq_length)
    wrapper = ModelForTokenGeneration(gpt_model)

    for trial, num_tokens in enumerate(num_tokens_per_trial):
        tokens = torch.randint(0, VOCAB_SIZE, (batch_size, num_tokens), device=device)
        msg = (
            f"trial {trial}, num_tokens = {num_tokens}, prompt_length = "
            f"{prompt_length}, one_by_one = {one_by_one}, attention = {attention}"
        )
        # A: Process prompt and tokens from scratch, default buffers
        prompt_logits_a = model_a.process_prompt(prompt, max_seq_length)
        logits_a = model_a.process_tokens(tokens, one_by_one)
        if trial == 0:
            # Same computation on both sides, sanity check
            _assert_same_logits(prompt_logits_a, prompt_logits_b, "prompt: " + msg)
        # B: Replay buffers on top of the processed prompt, rolled back
        # afterwards
        model_b.activate()
        wrapper.switch_status(True)
        assert all(
            isinstance(cache.kv_buffers, ReplayKVCacheBuffers)
            for cache in gpt_model.get_kv_caches()
        )
        logits_b = model_b.process_tokens(tokens, one_by_one)
        wrapper.switch_status(False)
        assert logits_a.shape == (
            batch_size,
            num_tokens,
            gpt_model.config.padded_vocab_size,
        )
        _assert_same_logits(logits_a, logits_b, msg)
