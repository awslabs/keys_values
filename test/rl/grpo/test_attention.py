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
Routing tests for the per-phase attention backend of the GRPO loop: decode
MHAs vs. the gradient cells' MHA copy. CPU only; FlexAttention numerics need
CUDA and are covered by the GPU tests listed in the commit message
(test_gradient_main.py::test_complete_gradient_computation, test_gradient.py,
test/attention/test_flex_attention.py).
"""

import pytest
import torch

from keys_values.attention.base import (
    MultiHeadSelfAttention,
    SDPA_IMPL_EAGER_BLOCKS,
    SDPA_IMPL_FLEXATTENTION,
    SDPA_IMPL_QPADDED_PYTORCH,
)
from keys_values.attention import flashinfer_ops
from keys_values.attention.flex_attention import FlexAttentionArgs, choose_q_lens
from keys_values.config import Config
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.kvcache.gradient.train_attn_weights_replay import (
    gradient_mha,
    select_sdpa_mode,
)
from keys_values.model import GPT
from keys_values.rl.grpo.attention import (
    ATTN_BACKENDS,
    attention_mha_kwargs,
    gradient_sdpa_mode_name,
    gradient_train_cache_kwargs,
)
from keys_values.rl.grpo.loop import grpo_step

CPU = torch.device("cpu")
CUDA = torch.device("cuda")


def _config() -> Config:
    return Config(
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


def _make_model_with_caches(batch_size, cache_length, dtype=torch.float32):
    config = _config()
    torch.set_default_dtype(dtype)
    with torch.device("cpu"):
        gpt_model = GPT(config)
        gpt_model.apply(gpt_model._init_weights)
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


@pytest.mark.parametrize("backend", ATTN_BACKENDS)
@pytest.mark.parametrize("available", [False, True])
def test_attention_mha_kwargs_cpu_never_selects_flex(backend, available, monkeypatch):
    monkeypatch.setattr(flashinfer_ops, "_available", available)
    kwargs = attention_mha_kwargs(
        backend, "h2o-default", 16, CPU, cache_length=64, verbose=False
    )
    assert "flexatt_args" not in kwargs
    if backend == "eager":
        assert kwargs["use_flashinfer"] is False
    else:
        assert "use_flashinfer" not in kwargs


# The CUDA cases below build kwargs only; FlexAttentionArgs is pure Python
# until a graph is compiled, so no GPU is needed.
@pytest.mark.parametrize("kv_cache_name", ["h2o-default", "dense-default"])
def test_attention_mha_kwargs_cuda_auto_without_flashinfer_selects_padded_flex(
    kv_cache_name, monkeypatch
):
    monkeypatch.setattr(flashinfer_ops, "_available", False)
    kwargs = attention_mha_kwargs(
        "auto", kv_cache_name, 1024, CUDA, cache_length=16384, verbose=False
    )
    # No use_flashinfer=False: _sdpa_mode ignores FlashInfer when it is not
    # built, and the kwargs stay valid if it is built later.
    assert "use_flashinfer" not in kwargs
    fa = kwargs["flexatt_args"]
    assert isinstance(fa, FlexAttentionArgs)
    assert fa.kv_lens == [4096, 8192, 12288, 16384]
    assert fa.q_lens == choose_q_lens(chunk_size=1024, num_q_lens=4)
    assert fa.extend_kv is False
    assert fa.forward_return_lse is (kv_cache_name == "h2o-default")


def test_attention_mha_kwargs_cuda_auto_with_flashinfer_has_no_flex(monkeypatch):
    monkeypatch.setattr(flashinfer_ops, "_available", True)
    kwargs = attention_mha_kwargs(
        "auto", "h2o-default", 1024, CUDA, cache_length=16384, verbose=False
    )
    assert "flexatt_args" not in kwargs
    assert "use_flashinfer" not in kwargs


@pytest.mark.parametrize("available", [False, True])
def test_attention_mha_kwargs_cuda_flex_is_padded(available, monkeypatch):
    monkeypatch.setattr(flashinfer_ops, "_available", available)
    kwargs = attention_mha_kwargs(
        "flex", "h2o-default", 1024, CUDA, cache_length=16384, verbose=False
    )
    assert kwargs["use_flashinfer"] is False
    fa = kwargs["flexatt_args"]
    assert fa.kv_lens == [4096, 8192, 12288, 16384]
    assert fa.forward_return_lse is True


@pytest.mark.parametrize("available", [False, True])
def test_attention_mha_kwargs_cuda_eager_never_selects_flex(available, monkeypatch):
    monkeypatch.setattr(flashinfer_ops, "_available", available)
    kwargs = attention_mha_kwargs(
        "eager", "h2o-default", 1024, CUDA, cache_length=16384, verbose=False
    )
    assert "flexatt_args" not in kwargs
    assert kwargs["use_flashinfer"] is False


@pytest.mark.parametrize("backend", ATTN_BACKENDS)
def test_gradient_train_cache_kwargs_cpu_is_empty(backend):
    assert gradient_train_cache_kwargs(backend, 16, CPU, verbose=False) == {}


def test_gradient_train_cache_kwargs_cuda_eager_is_empty():
    # No GPU needed: only the kwargs are built
    assert gradient_train_cache_kwargs("eager", 16, CUDA, verbose=False) == {}


@pytest.mark.parametrize("backend", ["auto", "flex"])
def test_gradient_train_cache_kwargs_cuda_auto_flex_selects_unpadded_flex(backend):
    # No GPU needed: FlexAttentionArgs is pure Python until a graph is compiled.
    # The cells get kv_lens=None under both auto and flex, even though the
    # decode MHAs may carry kv_lens-padded args.
    kwargs = gradient_train_cache_kwargs(backend, 16, CUDA, verbose=False)
    assert list(kwargs.keys()) == ["flexatt_args"]
    fa = kwargs["flexatt_args"]
    assert isinstance(fa, FlexAttentionArgs)
    assert fa.kv_lens is None
    assert fa.q_lens == choose_q_lens(chunk_size=16, num_q_lens=4)
    assert fa.extend_kv is False
    assert fa.forward_return_lse is False


def test_gradient_train_cache_kwargs_rejects_unknown_backend():
    with pytest.raises(ValueError):
        gradient_train_cache_kwargs("sdpa", 16, CPU, verbose=False)


def _decode_mode(mha: MultiHeadSelfAttention) -> int:
    # An attention-weight (H2O) decode call on CPU
    return mha._sdpa_mode(
        return_attn_weights=True,
        is_causal=False,
        q_len=1,
        kv_len=32,
        sliding_window_size=None,
        dtype=torch.float32,
        device=CPU,
    )


def test_gradient_mha_leaves_decode_mha_untouched():
    config = _config()
    mha = MultiHeadSelfAttention(config)
    assert gradient_mha(mha, None) is mha
    assert _decode_mode(mha) == SDPA_IMPL_EAGER_BLOCKS

    fa = FlexAttentionArgs(q_lens=[1, 8, 16])
    grad = gradient_mha(mha, fa)
    assert grad is not mha
    assert grad.flexatt_args is fa
    assert grad.pos_encoding is mha.pos_encoding
    # The shared decode MHA has not been modified
    assert mha.flexatt_args is None
    assert not mha.has_flex_attention
    assert _decode_mode(mha) == SDPA_IMPL_EAGER_BLOCKS

    # An MHA that already has (kv_lens-padded) flexatt_args, as the decode
    # MHAs do under --attn flex or the auto fallback without FlashInfer: the
    # cells still get their own args, and the decode MHA keeps the padded ones
    fa_padded = FlexAttentionArgs(q_lens=[1, 8, 16], kv_lens=[8, 16, 24, 32])
    mha_flex = MultiHeadSelfAttention(config, flexatt_args=fa_padded)
    grad_flex = gradient_mha(mha_flex, fa)
    assert grad_flex is not mha_flex
    assert grad_flex.flexatt_args is fa
    assert grad_flex.flexatt_args.kv_lens is None
    assert mha_flex.flexatt_args is fa_padded
    assert gradient_mha(mha_flex, None) is mha_flex


def test_select_sdpa_mode():
    config = _config()
    mha = MultiHeadSelfAttention(config)
    fa = FlexAttentionArgs(q_lens=[1, 8, 16])
    assert select_sdpa_mode(mha, None) == SDPA_IMPL_QPADDED_PYTORCH
    assert select_sdpa_mode(gradient_mha(mha, fa), None) == SDPA_IMPL_FLEXATTENTION
    assert select_sdpa_mode(mha, 8) == SDPA_IMPL_EAGER_BLOCKS


def test_gradient_sdpa_mode_name():
    gpt_model, _ = _make_model_with_caches(batch_size=2, cache_length=16)
    assert gradient_sdpa_mode_name(gpt_model, {}) == "qpadded-pytorch"
    fa = FlexAttentionArgs(q_lens=[1, 8, 16])
    assert gradient_sdpa_mode_name(gpt_model, {"flexatt_args": fa}) == "flexattention"
    # Query only; the model's caches are not modified
    assert gradient_sdpa_mode_name(gpt_model, {}) == "qpadded-pytorch"
    for kvc in gpt_model.get_kv_caches():
        assert kvc.mha.flexatt_args is None


def _reward_fn(prompts, completions):
    return completions.float().mean(dim=1)


def test_grpo_step_train_cache_kwargs_plumbing_cpu():
    """`train_cache_kwargs` reaches the replay caches; mode reported."""
    torch.manual_seed(0)
    num_prompts, prompt_len, group_size = 2, 8, 2
    gpt_model, config = _make_model_with_caches(
        batch_size=num_prompts * group_size, cache_length=32
    )
    prompt_ids = torch.randint(0, config.vocab_size, (num_prompts, prompt_len))
    optimizer = torch.optim.SGD(gpt_model.parameters(), lr=0.01)
    metrics = grpo_step(
        gpt_model=gpt_model,
        prompt_ids=prompt_ids,
        reward_fn=_reward_fn,
        optimizer=optimizer,
        group_size=group_size,
        max_new_tokens=6,
        chunk_size=16,
        train_cache_kwargs={"flexatt_args": None},
    )
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["grad_sdpa_mode"] == "qpadded-pytorch"


def test_grpo_step_rejects_flex_on_cpu():
    """FlexAttention in the cells needs CUDA; fail before any forward."""
    torch.manual_seed(0)
    num_prompts, prompt_len, group_size = 2, 8, 2
    gpt_model, config = _make_model_with_caches(
        batch_size=num_prompts * group_size, cache_length=32
    )
    prompt_ids = torch.randint(0, config.vocab_size, (num_prompts, prompt_len))
    optimizer = torch.optim.SGD(gpt_model.parameters(), lr=0.01)
    before = [p.detach().clone() for p in gpt_model.parameters()]
    with pytest.raises(ValueError, match="CUDA"):
        grpo_step(
            gpt_model=gpt_model,
            prompt_ids=prompt_ids,
            reward_fn=_reward_fn,
            optimizer=optimizer,
            group_size=group_size,
            max_new_tokens=6,
            chunk_size=16,
            train_cache_kwargs={"flexatt_args": FlexAttentionArgs(q_lens=[1, 8, 16])},
        )
    for b, p in zip(before, gpt_model.parameters()):
        assert torch.equal(b, p.detach())
