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
import random
from typing import Optional, Dict, Any
from functools import partial

import torch
import pytest

from litgpt.config import Config

from keys_values.kvcache.attn_weights import AttnWeightsKVCache
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.factory import KVCacheFactory, split_name
from keys_values.kvcache.gradient.inference_replay import (
    inference_replay_cache_factory,
    get_replay_logs,
    InferenceAttnWeightsReplayCache,
)
from keys_values.kvcache.gradient.train_attn_weights_replay import (
    TrainingAttnWeightsReplayCache,
)
from keys_values.kvcache.test_utils import (
    create_kv_cache,
    device_for_cache_name,
    filter_cache_names,
    cache_name_gpu_only,
)
from keys_values.model import GPT


class ForTestInferenceAttnWeightsReplayCache(InferenceAttnWeightsReplayCache):
    """
    For testing, we force the same :meth:`update_requires_attn_weights` response
    than :class:`AttnWeightsReplayCache`, so that the same MHA code is run.

    If we use :class:`InferenceAttnWeightsReplayCache` and `dtype = bfloat16` in
    the :func:`test_inference_replay`, tests for `h2o-default`, `h2o-vlen-default`
    fail with small abs diffs, and some very small entries rounded to 0 (large
    rel diff). This is because different MHA code is run, since for replaying
    the attention weights are not needed.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_requires_attn_weights(self) -> bool:
        return True


def _cache_kwargs(name: str) -> Dict[str, Any]:
    if name.startswith("lastrec"):
        return dict()
    else:
        return dict(replay_log_blocksize=16)


def args_inference_replay():
    names = filter_cache_names(
        [
            name for name in KVCacheFactory.supported_names()
            if not name.startswith("dense") and not name.startswith("h2o-orig")
        ]
    )
    result = [(name, _cache_kwargs(name)) for name in names]
    result += [
        (name, dict(_cache_kwargs(name), grace_period=10))
        for name in names
        if split_name(name)[0] in ("h2o", "h2o-vlen", "qh2o")
    ]
    if not torch.cuda.is_available():
        result = [x for x in result if not cache_name_gpu_only(x[0])]
    return result


@pytest.mark.parametrize(
    "cache_name, cache_kwargs", args_inference_replay(),
)
def test_inference_replay(cache_name, cache_kwargs):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)

    device = device_for_cache_name(cache_name)
    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)  # Set default dtype

    batch_size = 5
    n_head = 8
    n_query_groups = 4
    cache_length = 512
    head_size = 64
    vocab_size = 48
    tokens_per_chunk = [cache_length - 1, 1, 8, 4, 8, 2, 8, 2, 8, 8]
    seq_length = sum(tokens_per_chunk)

    layer_outputs = dict()

    def start_of_layer_hook(
        x: torch.Tensor, l_ix: int, input_pos: Optional[int], tag: str,
    ):
        if l_ix == 1:
            assert input_pos is not None
            current = layer_outputs.get(tag)
            if current is None:
                assert input_pos == 0
                layer_outputs[tag] = x
            else:
                assert input_pos == current.shape[1]
                layer_outputs[tag] = torch.cat([current, x], dim=1)

    config = Config(
        n_layer=1,
        n_head=n_head,
        n_query_groups=n_query_groups,
        n_embd=n_head * head_size,
        block_size=seq_length,
        vocab_size=vocab_size,
        rotary_percentage=1,
    )
    params = KVCacheParams.from_config(
        config=config,
        max_batch_size=batch_size,
        cache_length=cache_length,
        device=device,
        dtype=dtype,
    )
    model = GPT(config).to(device=device)
    kv_cache = create_kv_cache(
        name=cache_name,
        params=params,
        block_idx=0,
        **cache_kwargs,
    )
    kv_cache.switch_replay_logging(True)
    token_idxs = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_length),
        device=device,
    )
    print(f"{cache_name} ({cache_kwargs})")

    # Two inference runs: First with `kv_cache`, second with inference replay
    # cache based on the replay log of the first. Layer outputs, as recorded
    # by the hook, must be the same.
    for tag in ("original", "replayed"):
        model.set_start_of_layer_hook(partial(start_of_layer_hook, tag=tag))
        if tag == "original":
            model.assign_kv_caches([kv_cache])
            print("* Original inference pass")
        else:
            # Create inference replay cache. We use the buffer of `kv_cache`
            # here.
            replay_log = get_replay_logs(model)[0]
            kv_cache.kv_buffers.current_length = None  # Not really needed
            kwargs = dict(
                config=config,
                buffers=kv_cache.kv_buffers,
                block_idx=0,
                replay_log=replay_log,
            )
            if isinstance(kv_cache, AttnWeightsKVCache):
                ir_cache = ForTestInferenceAttnWeightsReplayCache(**kwargs)
            else:
                ir_cache = inference_replay_cache_factory(
                    kv_cache=kv_cache, **kwargs,
                )
            model.transformer.h[0].attn.kv_cache = ir_cache
            print("* Replay inference pass")

        with torch.no_grad():
            input_pos = 0
            y_parts = []
            for num in tokens_per_chunk:
                y_parts.append(
                    model(
                        token_idxs[:, input_pos:(input_pos + num)],
                        input_pos=input_pos,
                    )
                )
                input_pos += num
            y = torch.cat(y_parts, dim=1)

    # Recorded outputs of first (and only) layer must be the same
    outputs_orig = layer_outputs.get("original")
    outputs_replayed = layer_outputs.get("replayed")
    assert outputs_orig is not None
    assert outputs_replayed is not None
    assert outputs_orig.shape == outputs_replayed.shape
    input_pos = 0
    for num in tokens_per_chunk:
        end = input_pos + num
        print(f"Comparing {input_pos}:{end}")
        torch.testing.assert_close(
            outputs_orig[:, input_pos:end, :],
            outputs_replayed[:, input_pos:end, :],
        )
        input_pos += num


def args_training_replay():
    names = [
        name for name in KVCacheFactory.supported_names()
        if split_name(name)[1] == "default" and split_name(name)[0] not in ("dense", "h2o-orig")
    ]
    result1 = [(name, _cache_kwargs(name)) for name in names]
    result2 = [
        (name, dict(_cache_kwargs(name), grace_period=10))
        for name in names
        if split_name(name)[0] in ("h2o", "h2o-vlen")
    ]
    return result1 + result2


@pytest.mark.parametrize(
    "cache_name, cache_kwargs", args_training_replay(),
)
def test_training_replay(cache_name, cache_kwargs):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)

    device = torch.device("cpu")
    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)  # Set default dtype
    if cache_name.startswith("lastrec-alt"):
        tol_kwargs = dict(atol=0.005, rtol=2)
    else:
        tol_kwargs = dict()

    batch_size = 5
    n_head = 8
    n_query_groups = 4
    cache_length = 512
    head_size = 64
    vocab_size = 48
    tokens_per_chunk = [cache_length - 1, 1, 8, 4, 8, 2, 8, 2, 8, 8]
    seq_length = sum(tokens_per_chunk)

    layer_outputs = dict()

    def start_of_layer_hook(
        x: torch.Tensor, l_ix: int, input_pos: Optional[int], tag: str,
    ):
        if l_ix == 1:
            assert input_pos is not None
            current = layer_outputs.get(tag)
            if current is None:
                assert input_pos == 0
                layer_outputs[tag] = x
            else:
                assert input_pos == current.shape[1]
                layer_outputs[tag] = torch.cat([current, x], dim=1)

    config = Config(
        n_layer=1,
        n_head=n_head,
        n_query_groups=n_query_groups,
        n_embd=n_head * head_size,
        block_size=seq_length,
        vocab_size=vocab_size,
        rotary_percentage=1,
    )
    params = KVCacheParams.from_config(
        config=config,
        max_batch_size=batch_size,
        cache_length=cache_length,
        device=device,
        dtype=dtype,
    )
    model = GPT(config)
    kv_cache = create_kv_cache(
        name=cache_name,
        params=params,
        block_idx=0,
        **cache_kwargs,
    )
    kv_cache.switch_replay_logging(True)
    token_idxs = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_length),
        device=device,
    )
    print(f"{cache_name} ({cache_kwargs})")

    # Two inference runs: First with `kv_cache`, second with training replay
    # cache based on the replay log of the first. Layer outputs, as recorded
    # by the hook, must be the same.
    for tag in ("original", "replayed"):
        model.set_start_of_layer_hook(partial(start_of_layer_hook, tag=tag))
        if tag == "original":
            model.assign_kv_caches([kv_cache])
            print("* Original inference pass")
        else:
            # Create training replay cache. We use the buffer of `kv_cache`
            # here.
            replay_log = get_replay_logs(model)[0]
            tr_cache = TrainingAttnWeightsReplayCache(
                config=config,
                batch_size=batch_size,
                cache_length=cache_length,
                replay_log=replay_log,
                start_token_pos=0,
                layer_idx=0,
                num_chunks=len(tokens_per_chunk),
                device=device,
                node_annotations=None,
            )
            model.transformer.h[0].attn.kv_cache = tr_cache
            print("* Replay inference pass")

        with torch.no_grad():
            input_pos = 0
            y_parts = []
            for num in tokens_per_chunk:
                y_parts.append(
                    model(
                        token_idxs[:, input_pos:(input_pos + num)],
                        input_pos=input_pos,
                    )
                )
                input_pos += num
            y = torch.cat(y_parts, dim=1)

    # Recorded outputs of first (and only) layer must be the same
    outputs_orig = layer_outputs.get("original")
    outputs_replayed = layer_outputs.get("replayed")
    assert outputs_orig is not None
    assert outputs_replayed is not None
    assert outputs_orig.shape == outputs_replayed.shape
    input_pos = 0
    for num in tokens_per_chunk:
        end = input_pos + num
        print(f"Comparing {input_pos}:{end}")
        torch.testing.assert_close(
            outputs_orig[:, input_pos:end, :],
            outputs_replayed[:, input_pos:end, :],
            **tol_kwargs,
        )
        input_pos += num
