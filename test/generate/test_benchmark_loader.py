# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import patch

import pytest
import torch

from litgpt.utils import save_config

from keys_values.config import Config
from keys_values.data.constants import LIT_MODEL_FNAME
from keys_values.kvcache.quant_buffers import QuantizedKVCacheBuffers
from keys_values.model import GPT
from keys_values.scripts import benchmark_speculative_decoding as benchmark


@pytest.fixture
def tiny_checkpoint(tmp_path):
    config = Config(
        block_size=128,
        vocab_size=64,
        padded_vocab_size=64,
        n_layer=2,
        n_head=4,
        n_query_groups=2,
        n_embd=32,
        rotary_percentage=1.0,
    )
    torch.manual_seed(1234)
    gpt = GPT(config)
    weights = {key: value.clone() for key, value in gpt.state_dict().items()}
    save_config(config, tmp_path)
    torch.save(weights, tmp_path / LIT_MODEL_FNAME)
    return tmp_path, weights


def load_tiny(checkpoint_dir, cache_factory_kwargs, precision="32-true"):
    return benchmark._load_inference_model(
        checkpoint_dir=checkpoint_dir,
        cache_factory_kwargs=cache_factory_kwargs,
        max_returned_tokens=96,
        prompt_chunksize=16,
        precision=precision,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize(
    "precision, dtype",
    [("32-true", torch.float32), ("bf16-true", torch.bfloat16)],
)
def test_quantized_loader_passes_dtype_and_loads_weights(
    tiny_checkpoint, precision, dtype
):
    checkpoint_dir, expected_weights = tiny_checkpoint
    kwargs = dict(name="dense-torch-quantized8", cache_length=96)
    original_kwargs = dict(kwargs)
    with patch.object(
        benchmark.KVCacheFactory,
        "create",
        wraps=benchmark.KVCacheFactory.create,
    ) as create:
        model = load_tiny(checkpoint_dir, kwargs, precision=precision)

    assert kwargs == original_kwargs
    assert create.call_args.kwargs["dtype"] == dtype
    assert model.gpt_model.max_seq_length == 96
    assert model.gpt_model.mha.pos_encoding.context_width == 96
    assert model.gpt_model.mha.pos_encoding._cos.device.type == "cpu"
    assert model.gpt_model.mha.pos_encoding._sin.device.type == "cpu"
    assert model.gpt_model.mha.pos_encoding._cos.dtype == torch.float32
    assert model.gpt_model.transformer.wte.weight.dtype == dtype
    assert not model.gpt_model.transformer.wte.weight.is_inference()
    for cache in model.gpt_model.get_kv_caches():
        assert isinstance(cache.kv_buffers, QuantizedKVCacheBuffers)
        assert cache.dtype == dtype
    state = model.gpt_model.state_dict()
    for name, expected in expected_weights.items():
        actual = state[name]
        assert torch.equal(actual, expected.to(actual.dtype)), name


def test_default_loader_accepts_32_alias_without_dtype(tiny_checkpoint):
    checkpoint_dir, _ = tiny_checkpoint
    kwargs = dict(name="dense-default", cache_length=96)
    with patch.object(
        benchmark.KVCacheFactory,
        "create",
        wraps=benchmark.KVCacheFactory.create,
    ) as create:
        model = load_tiny(checkpoint_dir, kwargs, precision="32")

    assert "dtype" not in create.call_args.kwargs
    assert model.gpt_model.transformer.wte.weight.dtype == torch.float32
    assert all(cache.dtype is None for cache in model.gpt_model.get_kv_caches())


def test_explicit_cache_dtype_is_not_overridden(tiny_checkpoint):
    checkpoint_dir, _ = tiny_checkpoint
    kwargs = dict(name="dense-torch-quantized8", cache_length=96, dtype=torch.bfloat16)
    with patch.object(
        benchmark.KVCacheFactory,
        "create",
        wraps=benchmark.KVCacheFactory.create,
    ) as create:
        model = load_tiny(checkpoint_dir, kwargs)

    assert create.call_args.kwargs["dtype"] == torch.bfloat16
    assert kwargs["dtype"] == torch.bfloat16
    assert all(
        cache.dtype == torch.bfloat16 for cache in model.gpt_model.get_kv_caches()
    )


def test_nested_factory_kwargs_are_not_mutated(tiny_checkpoint):
    checkpoint_dir, _ = tiny_checkpoint
    kwargs = dict(
        name="dense-torch-quantized8",
        cache_length=96,
        cache_kwargs=dict(allocate_buffers=False),
    )
    load_tiny(checkpoint_dir, kwargs)
    assert kwargs == dict(
        name="dense-torch-quantized8",
        cache_length=96,
        cache_kwargs=dict(allocate_buffers=False),
    )


@pytest.mark.parametrize("precision", ["bf16", "16", "bf16-mixed", "16-mixed"])
def test_loader_rejects_mixed_precision(tiny_checkpoint, precision):
    checkpoint_dir, _ = tiny_checkpoint
    with pytest.raises(ValueError, match="Mixed precision.*autocast"):
        load_tiny(
            checkpoint_dir,
            dict(name="dense-default", cache_length=96),
            precision=precision,
        )


@pytest.mark.parametrize("precision", ["32", "32-true", "bf16-true", "16-true"])
def test_normalized_true_precision(precision):
    assert benchmark._normalize_inference_precision(precision) == (
        "32-true" if precision == "32" else precision
    )


def test_unsupported_precision_rejected():
    with pytest.raises(ValueError, match="not yet supported"):
        benchmark._normalize_inference_precision("64-true")
