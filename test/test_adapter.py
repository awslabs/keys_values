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
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch
from lightning import Fabric
from torch._dynamo.backends import debugging
from transformers.models.gemma import GemmaConfig, GemmaForCausalLM
from transformers.models.gemma2 import Gemma2Config, Gemma2ForCausalLM
from transformers.models.gemma3 import Gemma3ForCausalLM, Gemma3TextConfig

from litgpt.adapter import adapter_filter, Config
from litgpt.scripts.convert_hf_checkpoint import copy_weights_gemma_2, copy_weights_gemma_3, copy_weights_hf_llama
from litgpt.scripts.convert_lit_checkpoint import qkv_reassemble as make_qkv_interleaved
from litgpt.utils import _RunIf

import keys_values.adapter as gpt_adapter
import keys_values.model as gpt
from keys_values.adapter import GPT, CausalSelfAttention


def test_config_identical():
    name = "pythia-14m"
    base_config = asdict(gpt.Config.from_name(name))
    adapter_config = asdict(gpt_adapter.Config.from_name(name))
    del adapter_config["adapter_prompt_length"]
    del adapter_config["adapter_start_layer"]
    assert adapter_config == base_config

    with Fabric(accelerator="cpu").init_module(empty_init=True):
        base_model = gpt.GPT.from_name(name)
        adapter_model = gpt_adapter.GPT.from_name(name)
    assert adapter_model.lm_head.weight.shape == base_model.lm_head.weight.shape


def test_adapter_filter(tmp_path):
    fabric = Fabric(devices=1)
    model = GPT.from_name("pythia-14m", n_layer=4)
    save_path = tmp_path / "model.pth"
    fabric.save(save_path, {"model": model}, filter={"model": adapter_filter})
    saved = torch.load(save_path)["model"]

    expected = {
        "transformer.h.2.attn.adapter_wte.weight",
        "transformer.h.2.attn.gating_factor",
        "transformer.h.3.attn.adapter_wte.weight",
        "transformer.h.3.attn.gating_factor",
    }
    assert set(saved) == expected


def test_adapter_gpt_init_weights():
    config = Config(n_layer=1, n_head=6, n_embd=12, block_size=1, vocab_size=1, adapter_start_layer=0)
    model = GPT(config)
    param = model.transformer.h[0].attn.gating_factor

    assert (param == 0).all()
    torch.nn.init.constant_(param, 1.23)
    assert (param != 0).any()
    model.apply(model._init_weights)
    assert (param == 0).all()


@_RunIf(dynamo=True)
@torch.inference_mode()
def test_adapter_compile():
    model = GPT.from_name("pythia-14m", n_layer=3)
    x = torch.randint(model.config.vocab_size, size=(2, model.config.block_size), dtype=torch.int64)

    explanation = torch._dynamo.explain(model)(x)
    assert isinstance(explanation, debugging.ExplainOutput)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0

    model = GPT(model.config)
    model.set_kv_caches(2)
    explanation = torch._dynamo.explain(model)(x)
    assert isinstance(explanation, debugging.ExplainOutput)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ["gemma-2b", "gemma-7b"])
def test_against_hf_gemma(model_name):
    device = torch.device("cpu")
    dtype = torch.float32
    T = 5
    ours_config = Config.from_name(model_name, n_layer=2, n_head=16, n_embd=32, intermediate_size=86)
    theirs_config = GemmaConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        head_dim=ours_config.head_size,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=T,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        attention_bias=ours_config.bias,
        tie_word_embeddings=True,
        hidden_act="gelu_pytorch_tanh",
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = GemmaForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    # Gemma weights are shipped without `lm_head.weight`
    theirs_state_dict.pop("lm_head.weight")
    state_dict = {}
    copy_weights_hf_llama(ours_config, {}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.tensor([[9856, 23, 491, 1536, 304]], dtype=torch.int32, device=device)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("gemma-2-9b", "gemma-2-27b"))
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                _RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_original_gemma_2(model_name, device, dtype):
    torch.set_default_dtype(dtype)

    T = 20
    ours_config = Config.from_name(
        model_name,
        block_size=T,
        sliding_window_size=T // 2,
        n_layer=2,
        n_head=16,
        n_embd=32,
        intermediate_size=86,
    )
    theirs_config = Gemma2Config(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        head_dim=ours_config.head_size,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=ours_config.block_size,
        sliding_window=ours_config.sliding_window_size,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        attention_bias=ours_config.bias,
        tie_word_embeddings=True,
        hidden_act="gelu_pytorch_tanh",
        attn_logit_softcapping=ours_config.attention_logit_softcapping,
        final_logit_softcapping=ours_config.final_logit_softcapping,
        initializer_range=1.0,  # to make the affect of attention_logit_softcapping more prominent
        attn_implementation="eager",
        query_pre_attn_scalar=ours_config.attention_scores_scalar,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = Gemma2ForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    # Gemma weights are shipped without `lm_head.weight`
    theirs_state_dict.pop("lm_head.weight")
    state_dict = {}
    copy_weights_gemma_2({}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.randint(low=0, high=ours_config.padded_vocab_size, size=(T,), device=device).unsqueeze(0)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y, atol=1e-4, rtol=1e-5)


@torch.inference_mode()
@pytest.mark.parametrize("model_name", ("gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"))
@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float32),
        pytest.param(
            torch.device("cuda"),
            torch.float16,
            marks=[
                # the reference does softmax upscaled to fp32 during attention. additionally, the final layernorm input
                # is slightly different
                pytest.mark.xfail(raises=AssertionError, strict=False),
                _RunIf(min_cuda_gpus=1),
            ],
        ),
    ],
)
def test_against_original_gemma_3(model_name, device, dtype):
    torch.set_default_dtype(dtype)

    T = 20
    ours_config = Config.from_name(
        model_name,
        block_size=T,
        sliding_window_size=T // 2,
        n_layer=2,
        n_head=16,
        n_embd=32,
        intermediate_size=86,
    )
    theirs_config = Gemma3TextConfig(
        vocab_size=ours_config.padded_vocab_size,
        hidden_size=ours_config.n_embd,
        head_dim=ours_config.head_size,
        num_attention_heads=ours_config.n_head,
        num_hidden_layers=ours_config.n_layer,
        intermediate_size=ours_config.intermediate_size,
        max_position_embeddings=ours_config.block_size,
        sliding_window=ours_config.sliding_window_size,
        rms_norm_eps=ours_config.norm_eps,
        num_key_value_heads=ours_config.n_query_groups,
        rope_theta=ours_config.rope_base,
        attention_bias=ours_config.bias,
        tie_word_embeddings=True,
        hidden_act="gelu_pytorch_tanh",
        attn_logit_softcapping=ours_config.attention_logit_softcapping,
        final_logit_softcapping=ours_config.final_logit_softcapping,
        initializer_range=1.0,  # to make the affect of attention_logit_softcapping more prominent
        attn_implementation="eager",
        query_pre_attn_scalar=ours_config.attention_scores_scalar,
    )
    assert ours_config.intermediate_size == theirs_config.intermediate_size

    theirs_model = Gemma3ForCausalLM(theirs_config).to(device)
    theirs_state_dict = theirs_model.state_dict()
    # Gemma weights are shipped without `lm_head.weight`
    theirs_state_dict.pop("lm_head.weight")
    state_dict = {}
    copy_weights_gemma_3({}, state_dict, theirs_state_dict)
    ours_model = GPT(ours_config).to(device)
    ours_model.load_state_dict(state_dict)

    # test end to end
    x = torch.randint(low=0, high=ours_config.padded_vocab_size, size=(T,), device=device).unsqueeze(0)
    assert x.size(1) == T
    ours_y = ours_model(x)
    theirs_y = theirs_model(x)["logits"].to(dtype)  # HF converts logits to float
    torch.testing.assert_close(ours_y, theirs_y, atol=1e-4, rtol=1e-5)


def test_load_legacy_state_dict():
    """Check that a legacy state dict (with an interleaved placement in QKV matrix) can be loaded into a model with CausalSelfAttention layers."""
    config = Config(
        n_embd=32,
        n_head=4,
        head_size=8,
        n_query_groups=4,
        bias=True,
    )

    attention_1 = CausalSelfAttention(config=config, block_idx=0)

    # make weights to be as-like in a legacy checkpoint, with `attn.attn.weight` instead of `attn.qkv.weight`
    # and make them interleaved
    state_dict = deepcopy(attention_1.state_dict())
    state_dict["attn.weight"] = make_qkv_interleaved(state_dict.pop("qkv.weight"), config)
    state_dict["attn.bias"] = make_qkv_interleaved(state_dict.pop("qkv.bias"), config)

    attention_2 = CausalSelfAttention(config=config, block_idx=0)
    attention_2.load_state_dict(state_dict)
