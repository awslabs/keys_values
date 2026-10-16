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
"""Generation/replay regression tests ported from upstream switch_buffers4 (ea31f1a)."""

from dataclasses import replace
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest import mock

import pytest
import torch

from keys_values.config import Config
from keys_values.attention.flex_attention import FlexAttentionArgs
from keys_values.generate.base import batched_generate_fn
from keys_values.head_model import CrossEntropyOnLogits
from keys_values.head_model_factory import HeadModelFactory
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.kvcache.replay_buffers import ModelForTokenGeneration
from keys_values.kvcache.test_utils import available_backends, create_kv_cache
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT
from keys_values.utils import randint_torch


# Compiled FlexAttention kernels can round differently between the two cache
# layouts in float16. Non-FlexAttention configurations keep the default strict
# comparison; the CUDA FlexAttention path uses a bounded fp16 tolerance.
def _assert_logits_close(a: torch.Tensor, b: torch.Tensor, use_flex_attention: bool):
    if use_flex_attention:
        torch.testing.assert_close(a, b, atol=4e-3, rtol=1e-2)
    else:
        torch.testing.assert_close(a, b)


def create_model_and_data(
    cache_name: str,
    cache_kwargs: Dict[str, Any],
    cache_lengths: List[int],
    device: torch.device,
    max_num_targets: Optional[int] = None,
    non_quantized_buffers: bool = False,
    use_flex_attention: bool = False,
) -> Tuple[
    LongContextInferenceModel,
    ModelForTokenGeneration,
    List[torch.Tensor],
    List[torch.Tensor],
    KVCacheParams,
    Dict[str, Any],
]:
    dtype = torch.float16
    torch.set_default_dtype(dtype)
    qname = "torch-quantized8"
    batch_sizes = [5] + [4] * (len(cache_lengths) - 1)
    n_head = 8
    n_query_groups = 4
    head_size = 64
    vocab_size = 128
    max_sequence_length = max(cache_lengths) * 8
    min_sequence_length = max(cache_lengths) * 2
    config = Config(
        n_layer=len(cache_lengths),
        n_head=n_head,
        n_query_groups=n_query_groups,
        n_embd=n_head * head_size,
        block_size=max_sequence_length,
        vocab_size=vocab_size,
        rotary_percentage=1,
    )
    params = KVCacheParams.from_config(
        config=config,
        max_batch_size=max(batch_sizes),
        cache_length=cache_lengths[0],
        dtype=dtype,
    )
    mha_kwargs = {"flexatt_args": FlexAttentionArgs()} if use_flex_attention else {}
    _cache_kwargs = {**cache_kwargs, **mha_kwargs}
    with torch.device(device):
        gpt_model = GPT(config, **mha_kwargs)
        gpt_model.apply(gpt_model._init_weights)
    caches = KVCacheFactory.create(
        gpt_model=gpt_model,
        name=cache_name + "-" + qname,
        max_batch_size=max(batch_sizes),
        cache_length=cache_lengths,
        device=device,
        dtype=dtype,
        cache_kwargs=_cache_kwargs,
        for_testing=non_quantized_buffers,
    )
    gpt_model.assign_kv_caches(caches)
    all_input_ids, all_targets = [], []
    for batch_size in batch_sizes:
        seq_length = randint_torch(min_sequence_length, max_sequence_length)
        token_ids = torch.randint(
            0, vocab_size, (batch_size, seq_length), device=device
        )
        num_output_tokens = randint_torch(4, int(seq_length * 0.5))
        if max_num_targets is not None:
            num_output_tokens = min(max_num_targets, num_output_tokens)
        all_input_ids.append(token_ids[:, :-1])
        all_targets.append(token_ids[:, -num_output_tokens:])
    head_model = HeadModelFactory.create(name=CrossEntropyOnLogits.NAME, config=config)
    model = LongContextInferenceModel(
        gpt_model=gpt_model,
        head_model=head_model,
        chunk_size=16,
    )
    model.eval()
    return (
        model,
        ModelForTokenGeneration(gpt_model),
        all_input_ids,
        all_targets,
        params,
        mha_kwargs,
    )


def args_loss_after_replay():
    return [
        b + c + (a, use_flex)
        for (a, use_flex), b, c in product(
            [
                (device, use_flex)
                for device in available_backends()
                for use_flex in ((False, True) if device.type == "cuda" else (False,))
            ],
            [
                ("lastrec", {}),
                ("h2o", {"replay_log_blocksize": 64}),
                ("h2o", {"grace_period": 10, "replay_log_blocksize": 64}),
            ],
            [([128, 128],), ([96, 128],)],
        )
    ]


@pytest.mark.parametrize(
    "cache_name, cache_kwargs, cache_lengths, device, use_flex_attention",
    args_loss_after_replay(),
)
def test_loss_after_replay(
    cache_name, cache_kwargs, cache_lengths, device, use_flex_attention
):
    torch.random.manual_seed(31415927)
    model, wrapper, batches, targets_batches, _, _ = create_model_and_data(
        cache_name,
        cache_kwargs,
        cache_lengths,
        device,
        use_flex_attention=use_flex_attention,
    )
    direct = [
        model(inputs, targets).mean().detach()
        for inputs, targets in zip(batches, targets_batches)
    ]
    # Computing the loss clears the chunk state
    assert model.chunk_sizes is None and model.chunks_per_cell is None
    indirect = []
    for inputs, targets in zip(batches, targets_batches):
        logits = model(inputs, targets, mode="inputs")
        wrapper.switch_status(True)
        _ = list(
            batched_generate_fn(
                model=model,
                prompts=logits,
                max_returned_tokens=8,
                sample_args={},
                deallocate_cache_buffers=False,
            )
        )
        wrapper.switch_status(False)
        loss = model(inputs, targets, mode="targets")
        indirect.append(loss.mean().detach())
    for a, b in zip(direct, indirect):
        _assert_logits_close(a, b, use_flex_attention)


@pytest.mark.parametrize(
    "cache_name, cache_kwargs, cache_lengths, device, use_flex_attention",
    args_loss_after_replay(),
)
def test_generate_several_times(
    cache_name, cache_kwargs, cache_lengths, device, use_flex_attention
):
    torch.random.manual_seed(31415927)
    model, wrapper, batches, targets_batches, _, _ = create_model_and_data(
        cache_name,
        cache_kwargs,
        cache_lengths,
        device,
        max_num_targets=20,
        use_flex_attention=use_flex_attention,
    )
    generated_logits = []
    data_idx = [0]

    def forced_sample(
        logits: torch.Tensor, kwargs: Union[dict, List[dict]]
    ) -> torch.Tensor:
        pos = len(generated_logits)
        generated_logits.append(logits.detach().clone())
        return targets_batches[data_idx[0]][:, pos].unsqueeze(-1)

    for i, inputs in enumerate(batches):
        data_idx[0] = i
        num_targets = targets_batches[i].shape[-1]
        model.gpt_model.max_seq_length = inputs.shape[-1]
        init_logits = model(inputs[:, :-num_targets], targets=None)
        all_results = []
        for _ in range(4):
            wrapper.switch_status(True)
            generated_logits.clear()
            with mock.patch("keys_values.generate.base.batched_sample", forced_sample):
                list(
                    batched_generate_fn(
                        model=model,
                        prompts=init_logits,
                        max_returned_tokens=num_targets,
                        sample_args={},
                        deallocate_cache_buffers=False,
                    )
                )
            assert len(generated_logits) == num_targets
            all_results.append(generated_logits.copy())
            wrapper.switch_status(False)
        for logits_for_position in zip(*all_results):
            for other in logits_for_position[1:]:
                _assert_logits_close(logits_for_position[0], other, use_flex_attention)


@pytest.mark.parametrize(
    "cache_name, cache_kwargs, cache_lengths, device, use_flex_attention",
    args_loss_after_replay(),
)
def test_generate_old_new(
    cache_name, cache_kwargs, cache_lengths, device, use_flex_attention
):
    torch.random.manual_seed(31415927)
    (
        model,
        wrapper,
        batches,
        targets_batches,
        params,
        mha_kwargs,
    ) = create_model_and_data(
        cache_name,
        cache_kwargs,
        cache_lengths,
        device,
        max_num_targets=20,
        non_quantized_buffers=True,
        use_flex_attention=use_flex_attention,
    )
    generated_logits = []
    data_idx = [0]

    def forced_sample(
        logits: torch.Tensor, kwargs: Union[dict, List[dict]]
    ) -> torch.Tensor:
        pos = len(generated_logits)
        generated_logits.append(logits.detach().clone())
        return targets_batches[data_idx[0]][:, pos].unsqueeze(-1)

    replay_results = []
    for i, inputs in enumerate(batches):
        data_idx[0] = i
        num_targets = targets_batches[i].shape[-1]
        model.gpt_model.max_seq_length = inputs.shape[-1]
        init_logits = model(inputs[:, :-num_targets], targets=None)
        wrapper.switch_status(True)
        generated_logits.clear()
        with mock.patch("keys_values.generate.base.batched_sample", forced_sample):
            list(
                batched_generate_fn(
                    model=model,
                    prompts=init_logits,
                    max_returned_tokens=num_targets,
                    sample_args={},
                    deallocate_cache_buffers=False,
                )
            )
        assert len(generated_logits) == num_targets
        replay_results.append(generated_logits.copy())
        wrapper.switch_status(False)

    gpt_model = model.gpt_model
    gpt_model.assign_kv_caches(
        [
            create_kv_cache(
                name=cache_name + "-default",
                params=replace(params, cache_length=cache_length),
                block_idx=block_idx,
                **cache_kwargs,
                **mha_kwargs,
            )
            for block_idx, cache_length in enumerate(cache_lengths)
        ]
    )
    model2 = LongContextInferenceModel(
        gpt_model=gpt_model,
        head_model=model.head_model,
        chunk_size=16,
    ).eval()
    normal_results = []
    for i, inputs in enumerate(batches):
        data_idx[0] = i
        gpt_model.reset()
        num_targets = targets_batches[i].shape[-1]
        gpt_model.max_seq_length = inputs.shape[-1]
        generated_logits.clear()
        with mock.patch("keys_values.generate.base.batched_sample", forced_sample):
            list(
                batched_generate_fn(
                    model=model2,
                    prompts=inputs[:, :-num_targets],
                    max_returned_tokens=num_targets,
                    sample_args={},
                    deallocate_cache_buffers=False,
                )
            )
        assert len(generated_logits) == num_targets
        normal_results.append(generated_logits.copy())
    for replay_case, normal_case in zip(replay_results, normal_results):
        for replay_logits, normal_logits in zip(replay_case, normal_case):
            _assert_logits_close(replay_logits, normal_logits, use_flex_attention)
