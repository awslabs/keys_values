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
import math
from typing import Optional
from itertools import product
from dataclasses import replace

import torch
import pytest

from litgpt.config import Config

from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.gradient.accumulate import GradientAccumulator
from keys_values.kvcache.test_utils import (
    create_kv_cache,
    copy_gradients,
    exchange_kv_cache_checkpoints,
    available_backends,
)
from keys_values.kvcache.gradient.autograd_hooks import CellComputationAutogradHooks
from keys_values.kvcache.gradient.cell import GetInputSlice, WriteOutputsSlice
from keys_values.kvcache.gradient.inference_replay import get_replay_logs
from keys_values.kvcache.gradient.monitor_autograd_hooks import MonitorCellComputationAutogradHooks
from keys_values.kvcache.stack_layers import DefaultCellBlocks
from keys_values.kvcache.utils import VerbosityLevels
from keys_values.model import GPT


def make_get_inputs_slice(x: torch.Tensor) -> GetInputSlice:
    return lambda start, end: x[:, start:end, :]


def make_write_outputs_slice(x: torch.Tensor) -> WriteOutputsSlice:
    def result(start: int, value: torch.Tensor):
        x[:, start:(start + value.shape[1]), :].copy_(value)

    return result


def args_gradient_row_of_cells():
    return [
        a + b + (c,)
        for a, b, c in product(
            [
                ("lastrec", dict()),
                ("h2o", {"replay_log_blocksize": 64}),
                ("qh2o", {"replay_log_blocksize": 64}),
                ("h2o", {"grace_period": 10, "replay_log_blocksize": 64}),
                ("qh2o", {"grace_period": 12, "replay_log_blocksize": 64}),
            ],
            [
                ([512, 512], [511, 1, 8, 4, 8, 2, 8, 2, 8, 8], [2, 3, 3, 2]),
                ([512, 504], [503, 1, 4, 4, 8, 4, 8, 2, 8, 2, 8, 8], [2, 2, 3, 3, 2]),
            ],
            available_backends(),
        )
    ]


@pytest.mark.parametrize(
    "cache_name, cache_kwargs, cache_lengths, tokens_per_chunk, chunks_per_cell, device",
    [args_gradient_row_of_cells()[0]],  # DEBUG!
)
def test_gradient_row_of_cells(
    cache_name,
    cache_kwargs,
    cache_lengths,
    tokens_per_chunk,
    chunks_per_cell,
    device,
):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    print(f"cache_name={cache_name}, cache_kwargs={cache_kwargs}")
    print(f"cache_length={cache_lengths}\ntokens_per_chunk={tokens_per_chunk}\nchunks_per_cell={chunks_per_cell}")

    use_autograd_hooks = True
    hooks_for_comp = False
    do_gradient_testing = True
    # Additional comparison of all autograd hook pack arguments
    debug_test_args = True
    do_compare_cache_tensors = False
    use_monitoring_autograd_hooks = False
    assert not (
        use_autograd_hooks and use_monitoring_autograd_hooks
    ), "Can only set one of use_autograd_hooks or use_monitoring_autograd_hooks"
    assert use_autograd_hooks or (
        not debug_test_args
    ), "If debug_test_args is set, so must be use_autograd_hooks"
    if do_gradient_testing:
        # Gradient testing fails with too low precision
        dtype = torch.float32
    else:
        dtype = torch.bfloat16
    torch.set_default_dtype(dtype)  # Set default dtype

    qname = "torch-quantized8"
    batch_size = 5
    n_layer = len(cache_lengths)
    n_head = 8
    n_query_groups = 4
    head_size = 64
    vocab_size = 48
    num_chunks = len(tokens_per_chunk)
    block_size = sum(tokens_per_chunk) + 16
    assert sum(chunks_per_cell) == num_chunks

    layer_inputs = dict()

    def start_of_layer_hook(x: torch.Tensor, l_ix: int, input_pos: Optional[int]):
        if l_ix in (0, n_layer):
            assert input_pos is not None
            current = layer_inputs.get(l_ix)
            if current is None:
                assert input_pos == 0
                layer_inputs[l_ix] = x
            else:
                assert input_pos == current.shape[1]
                layer_inputs[l_ix] = torch.cat([current, x], dim=1)

    # Create model and data
    config = Config(
        n_layer=n_layer,
        n_head=n_head,
        n_query_groups=n_query_groups,
        n_embd=n_head * head_size,
        block_size=block_size,
        vocab_size=vocab_size,
        rotary_percentage=1,
    )
    params = KVCacheParams.from_config(
        config=config,
        max_batch_size=batch_size,
        cache_length=cache_lengths[0],
        device=device,
        dtype=dtype,
    )
    gpt_model = GPT(config).to(device=device)
    gpt_model.set_start_of_layer_hook(start_of_layer_hook)
    token_idxs = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, config.block_size),
        device=device,
    )
    kv_caches = []
    for block_idx, cache_length in enumerate(cache_lengths):
        kv_cache = create_kv_cache(
            name=cache_name + "-" + qname,
            params=replace(params, cache_length=cache_length),
            block_idx=block_idx,
            **cache_kwargs,
        )
        kv_cache.switch_replay_logging(True)
        kv_caches.append(kv_cache)
    gpt_model.assign_kv_caches(kv_caches)

    # Forward pass in inference mode. This is assembling the replay log and
    # also populates `layer_inputs`
    print("\nForward inference pass, recording replay logs and layer inputs")
    with torch.no_grad():
        input_pos = 0
        y_parts = []
        for num in tokens_per_chunk:
            y_parts.append(
                gpt_model(
                    token_idxs[:, input_pos:(input_pos + num)],
                    input_pos=input_pos,
                )
            )
            input_pos += num
        y = torch.cat(y_parts, dim=1)

    gpt_model.set_start_of_layer_hook(None)   # Do not record layer inputs from now on
    replay_logs = get_replay_logs(gpt_model)
    assert len(replay_logs) == n_layer
    # Checks on replay logs
    seq_len = sum(tokens_per_chunk)
    for replay_log in replay_logs:
        assert len(replay_log) == seq_len
        assert len(replay_log.token_chunks) == num_chunks
    # Check on layer inputs
    assert set(layer_inputs.keys()) == {0, n_layer}
    shape = (batch_size, seq_len, config.n_embd)
    for x in layer_inputs.values():
        assert x.shape == shape

    # Setup gradient accumulator
    if use_autograd_hooks:
        autograd_hooks = CellComputationAutogradHooks(
            config=config,
            batch_size=batch_size,
            debug_test_args=debug_test_args,
        )
    elif use_monitoring_autograd_hooks:
        autograd_hooks = MonitorCellComputationAutogradHooks(
            config=config,
            batch_size=batch_size,
            cache_length=cache_lengths[0],
            num_layers=config.n_layer,
            device=params.device,
            dtype=params.dtype,
        )
    else:
        autograd_hooks = None
    if do_compare_cache_tensors:
        debug_cache_tensors = dict()
    else:
        debug_cache_tensors = None
    accumulator = GradientAccumulator(
        config=config,
        autograd_hooks=None if hooks_for_comp else autograd_hooks,
        qname=qname,
        debug_tensors=debug_cache_tensors,
        verbose=VerbosityLevels.SOME,
        train_cache_kwargs=dict(use_new_cache=True),  # DEBUG!
    )
    accumulator._batch_size = batch_size
    accumulator._initialize_internal(
        replay_logs, chunks_per_cell, weights_dtype=params.dtype,
    )
    if do_gradient_testing:
        # Replace KV cache checkpoint objects by such which do not quantize
        # the checkpoints. This allows for simple gradient testing
        exchange_kv_cache_checkpoints(accumulator)

    # Run gradient accumulation
    gpt_model.zero_grad()  # Reset gradients to 0
    inputs = layer_inputs[0]
    # We could compute real head gradients from the outputs
    head_gradients = torch.randn(
        *inputs.shape, device=inputs.device, dtype=inputs.dtype
    )
    below_gradients = torch.zeros_like(head_gradients)
    print(f"\nGradient accumulation with activation checkpointing: {chunks_per_cell}")
    model_part = DefaultCellBlocks(
        model=gpt_model,
        first_layer_idx=0,
        num_layers=config.n_layer,
    )
    accumulator.run(
        model_part=model_part,
        get_inputs_slice=make_get_inputs_slice(inputs),
        get_head_gradients_slice=make_get_inputs_slice(head_gradients),
        write_head_gradients_slice=make_write_outputs_slice(below_gradients),
    )
    param_gradients = copy_gradients(gpt_model, device=torch.device("cpu"))
    print(f"Number of gradients: {len(param_gradients)}")
    # print("\n".join(param_gradients.keys()))
    below_gradients = below_gradients.to(torch.device("cpu"))

    # Compare against gradients computed in a single pass, not using autograd
    # hooks
    if do_compare_cache_tensors:
        debug_cache_tensors_comp = dict()
    else:
        debug_cache_tensors_comp = None
    accumulator_comp = GradientAccumulator(
        config=config,
        autograd_hooks=autograd_hooks if hooks_for_comp else None,
        qname="torch-quantized8",  # will not be used
        debug_tensors=debug_cache_tensors_comp,
        verbose=VerbosityLevels.SOME,
    )
    accumulator_comp._batch_size = batch_size
    accumulator_comp._initialize_internal(
        replay_logs, chunks_per_cell=[num_chunks], weights_dtype=params.dtype,
    )
    gpt_model.zero_grad()
    below_gradients_comp = torch.zeros_like(head_gradients)
    print("\nGradient accumulation without activation checkpointing")
    accumulator_comp.run(
        model_part=model_part,
        get_inputs_slice=make_get_inputs_slice(inputs),
        get_head_gradients_slice=make_get_inputs_slice(head_gradients),
        write_head_gradients_slice=make_write_outputs_slice(below_gradients_comp),
    )
    param_gradients_comp = copy_gradients(gpt_model, device=torch.device("cpu"))
    print(f"Number of gradients: {len(param_gradients_comp)}")
    # print("\n".join(param_gradients_comp.keys()))
    below_gradients_comp = below_gradients_comp.to(torch.device("cpu"))

    # Test all pack arguments
    if debug_test_args:
        print("\nComparing pack arguments with their reconstructions:")
        for pack_arg, annotation in autograd_hooks.debug_log_args():
            print(f"kind={annotation.kind}, layer_idx={annotation.layer_idx}, chunk_idx={annotation.chunk_idx}, shape={annotation.shape}")
            torch.testing.assert_close(pack_arg, annotation.debug_full_arg)

    # Compare cache tensors
    if do_compare_cache_tensors:
        print("\nComparing cache tensors stored along the way:")
        for name in sorted(debug_cache_tensors.keys()):
            value = debug_cache_tensors[name]
            value_comp = debug_cache_tensors_comp.get(name)
            if value_comp is None:
                print(f"{name} is in debug_cache_tensors, but not in debug_cache_tensors_comp")
            else:
                try:
                    torch.testing.assert_close(value, value_comp)
                    print(f"{name}: Tensors are close")
                except AssertionError as ex:
                    print(f"{name}: {ex}")

    if use_autograd_hooks:
        logs = accumulator_comp.annotation_usage_logs() if hooks_for_comp else accumulator.annotation_usage_logs()
        print("\nAnnotation usage logs (per cell):")
        all_args_matched = []
        for first_chunk_idx, annotation_usage in sorted(
            list(logs.items()), reverse=True,
        ):
            print(f"\nCell(first_chunk_idx {first_chunk_idx}):")
            print(annotation_usage.report())
            all_args_matched.append(
                len(annotation_usage.unmatched_pack_args) == 0
            )
        assert all(all_args_matched)

    print("\nComparing gradients")
    for name, value in param_gradients.items():
        value_comp = param_gradients_comp.get(name)
        if value_comp is None:
            raise IndexError(f"name = {name} is in param_gradients, but not in param_gradients_comp")
        print(f"Comparing gradient for {name}")
        torch.testing.assert_close(value, value_comp)
    print("Comparing below_gradients:")
    torch.testing.assert_close(below_gradients, below_gradients_comp)

    if use_autograd_hooks and autograd_hooks.log_all_shapes:
        print("\nAutograd hooks logged these shapes:")
        for shape, numel, count in sorted(
            [
                (shape, math.prod(shape[:-1]), count)
                for shape, count in autograd_hooks.shapes_counter().items()
            ],
            key=lambda x: x[1], reverse=True,
        ):
            print(f"{shape} [{numel}]: {count}")
