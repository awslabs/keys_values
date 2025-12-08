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
from itertools import product
from typing import Optional
import math
import random

import torch
from torch.nn import functional as F
import pytest
from torch.nn.attention import SDPBackend

from litgpt.config import Config

from keys_values.attention_utils import (
    build_mask_slice,
    sample_token_positions,
    ENTRIES_PER_GB,
)
from keys_values.kvcache.attn_weights import (
    update_token_positions,
    AttnWeightsKVCache,
)
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.buffers import KVCacheBuffers
from keys_values.kvcache.gradient.accumulate import copy_requires_grad
from keys_values.kvcache.gradient.inference_replay import get_replay_logs
from keys_values.kvcache.gradient.sdpa_op import (
    SDPAFunction,
    sdpa_backward,
    KVCacheCatUpdateAndSDPAFunction,
    KVCacheScatterUpdateAndSDPAFunction,
    cat_on_buffers,
    scatter_on_buffers,
)
from keys_values.kvcache.h2o import H2OKVCache
from keys_values.kvcache.qh2o import QuantizedH2OKVCache
from keys_values.kvcache.test_before_after import TestLogInputsKVCacheMixin
from keys_values.kvcache.test_utils import (
    product_with_devices,
    available_backends,
    create_kv_cache,
)
from keys_values.model import GPT
from keys_values.sdpa_wrapper import scaled_dot_product_attention
from keys_values.utils import repeat_interleave


@pytest.mark.parametrize(
    *product_with_devices(
        [
            (4, 2, 128, 512, torch.float32, None),
            (4, 4, 1, 256, torch.float32, None),
            (8, 4, 128, 128, torch.float32, None),
            (12, 4, 16, 512, torch.float32, None),
            (24, 8, 2, 512, torch.float16, None),
            (9, 3, 128, 512, torch.bfloat16, None),
            (16, 16, 128, 512, torch.float16, None),
            (12, 4, 16, 512, torch.float32, 12),
            (24, 8, 2, 512, torch.float16, 64),
            (9, 3, 128, 512, torch.bfloat16, 96),
        ],
        "n_head, n_query_groups, q_len, kv_len, dtype, sliding_window_size",
    ),
)
def test_sdpa_op_gradients(device, n_head, n_query_groups, q_len, kv_len, dtype, sliding_window_size):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    num_repeats = 32
    seq_len = 2 * kv_len
    is_causal = q_len == kv_len
    input_pos = seq_len - q_len if not is_causal else 0
    enable_gqa = n_query_groups < n_head

    print(f"n_head={n_head}, n_query_groups={n_query_groups}, q_len={q_len}, kv_len={kv_len}, is_causal={is_causal}, dtype={dtype}, device={device}")
    kwargs = dict(device=device, dtype=dtype)
    for repeat in range(num_repeats):
        head_size = 2 ** random.randint(3, 6)
        batch_size = random.randint(1, 5)
        print(f"repeat={repeat}, head_size={head_size}, batch_size={batch_size}")
        if is_causal:
            token_positions = None
        else:
            token_positions = sample_token_positions(
                batch_size,
                n_query_groups,
                q_len,
                kv_len,
                input_pos,
                device=device,
            )
        shape = (batch_size, n_head, q_len, head_size)
        _query = torch.randn(shape, **kwargs)
        shape = (batch_size, n_query_groups, kv_len, head_size)
        _key = torch.randn(shape, **kwargs)
        _value = torch.randn(shape, **kwargs)
        print(f"query {_query.shape}, key {_key.shape}, value {_value.shape}")
        if token_positions is not None:
            print(f"token_positions {token_positions.shape}")
        scale = 1.0 / math.sqrt(head_size)
        gradients = dict()
        for kind in ("op", "noop"):
            query = copy_requires_grad(_query)
            key = copy_requires_grad(_key)
            value = copy_requires_grad(_value)
            if kind == "op":
                y = SDPAFunction.apply(
                    query,
                    key,
                    value,
                    token_positions,
                    input_pos,
                    scale,
                    sliding_window_size,
                )
            else:
                if enable_gqa:
                    # Some efficient kernels have not reliably implemented
                    # `enabla_gqa=True`. It is better to extend keys, values in
                    # this case.
                    _dtype = torch.float32
                    key2 = repeat_interleave(key.to(_dtype), n_head)
                    value2 = repeat_interleave(value.to(_dtype), n_head)
                    query2 = query.to(_dtype)
                    _enable_gqa = key2.shape[1] == n_query_groups
                else:
                    query2 = query
                    key2 = key
                    value2 = value
                    _enable_gqa = enable_gqa
                    _dtype = dtype
                if is_causal:
                    mask = None
                else:
                    mask = build_mask_slice(
                        input_pos=input_pos,
                        num=q_len,
                        token_positions=token_positions,
                        n_head=n_head,
                        dtype=_dtype,
                        sliding_window_size=sliding_window_size,
                    ).detach()
                    print(f"mask {mask.shape}")
                y = F.scaled_dot_product_attention(
                    query=query2,
                    key=key2,
                    value=value2,
                    attn_mask=mask,
                    dropout_p=0.0,
                    scale=scale,
                    is_causal=is_causal,
                    enable_gqa=_enable_gqa,
                )
            loss = y.sum()
            loss.backward()
            gradients[kind] = (query.grad, key.grad, value.grad)
        # Compare
        for name, grad_op, grad_noop in zip(
            ("query", "key", "value"), gradients["op"], gradients["noop"],
        ):
            print(f"Compare gradients for {name}")
            torch.testing.assert_close(
                grad_op, grad_noop, atol=0.0005, rtol=0.05,
            )


@pytest.mark.parametrize(
    *product_with_devices(
        [
            (4, 2, 128, 512, torch.float16, None),
            (4, 4, 8, 256, torch.bfloat16, None),
            (8, 4, 128, 128, torch.float16, None),
            (12, 4, 16, 512, torch.bfloat16, None),
            (24, 8, 2, 512, torch.float16, None),
            (9, 3, 128, 512, torch.bfloat16, None),
            (16, 16, 128, 512, torch.bfloat16, None),
            (16, 16, 128, 512, torch.float16, None),
            (12, 4, 16, 512, torch.float16, 12),
            (24, 8, 2, 512, torch.bfloat16, 64),
            (9, 3, 128, 512, torch.float16, 96),
        ],
        "n_head, n_query_groups, q_len, kv_len, dtype, sliding_window_size",
    ),
)
def test_sdpa_backward(device, n_head, n_query_groups, q_len, kv_len, dtype, sliding_window_size):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    num_repeats = 32
    seq_len = 2 * kv_len
    is_causal = q_len == kv_len
    input_pos = seq_len - q_len if not is_causal else 0

    print(f"n_head={n_head}, n_query_groups={n_query_groups}, q_len={q_len}, kv_len={kv_len}, is_causal={is_causal}, dtype={dtype}, device={device}")
    kwargs = dict(device=device, dtype=dtype)
    for repeat in range(num_repeats):
        head_size = 2 ** random.randint(3, 6)
        batch_size = random.randint(1, 5)
        if q_len % 2 != 0 and batch_size % 2 != 0:
            batch_size += 1
        print(f"repeat={repeat}, head_size={head_size}, batch_size={batch_size}")
        if is_causal:
            token_positions = None
        else:
            token_positions = sample_token_positions(
                batch_size,
                n_query_groups,
                q_len,
                kv_len,
                input_pos,
                device=device,
            )
        shape = (batch_size, n_head, q_len, head_size)
        query = torch.randn(shape, **kwargs)
        grad_attn_output = torch.randn(shape, **kwargs)
        shape = (batch_size, n_query_groups, kv_len, head_size)
        key = torch.randn(shape, **kwargs)
        value = torch.randn(shape, **kwargs)
        print(f"query {query.shape}, key {key.shape}, value {value.shape}")
        if token_positions is not None:
            print(f"token_positions {token_positions.shape}")
        scale_factor = 1.0 / math.sqrt(head_size)
        gradients = {"query": [], "key": [], "value": []}
        numel_tmp = batch_size * n_head * q_len * kv_len
        for kind in ("no", "yes"):
            num_temp_entry_limit = 2 * numel_tmp if kind == "no" else numel_tmp // 2
            grad_query, grad_key, grad_value = sdpa_backward(
                grad_attn_output=grad_attn_output,
                query=query,
                key=key,
                value=value,
                token_positions=token_positions,
                input_pos=input_pos,
                scale_factor=scale_factor,
                sliding_window_size=sliding_window_size,
                need_query=True,
                need_key=True,
                need_value=True,
                tmp_array_limit_gb=num_temp_entry_limit / ENTRIES_PER_GB,
            )
            gradients["query"].append(grad_query)
            gradients["key"].append(grad_key)
            gradients["value"].append(grad_value)
        # Compare
        for name, grads in gradients.items():
            print(f"Compare gradients for {name}")
            torch.testing.assert_close(
                grads[0], grads[1], atol=0.0005, rtol=0.05,
            )


class H2OLogInputsKVCache(H2OKVCache, TestLogInputsKVCacheMixin):
    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        grace_period: int = 0,
        replay_log_blocksize: Optional[int] = None,
        detach_attn_weights: bool = False,
        normalize_scores: bool = False,
        **base_kwargs,
    ):
        super().__init__(
            config=config,
            buffers=buffers,
            block_idx=block_idx,
            grace_period=grace_period,
            replay_log_blocksize=replay_log_blocksize,
            detach_attn_weights=detach_attn_weights,
            normalize_scores=normalize_scores,
            **base_kwargs,
        )

    def _get_self(self) -> AttnWeightsKVCache:
        return self

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        input_pos: int,
    ) -> torch.Tensor:
        _kwargs = dict(
            query=query,
            key=key,
            value=value,
            token_idx=token_idx,
            input_pos=input_pos,
        )
        forward_kwargs = self.call_before_forward(**_kwargs)
        attn_outputs = super().forward(**forward_kwargs)
        self.call_after_forward(attn_outputs)
        return attn_outputs


class QuantizedH2OLogInputsKVCache(QuantizedH2OKVCache, TestLogInputsKVCacheMixin):
    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        grace_period: int = 0,
        replay_log_blocksize: Optional[int] = None,
        detach_attn_weights: bool = False,
        normalize_scores: bool = False,
        **base_kwargs,
    ):
        super().__init__(
            config=config,
            buffers=buffers,
            block_idx=block_idx,
            grace_period=grace_period,
            replay_log_blocksize=replay_log_blocksize,
            detach_attn_weights=detach_attn_weights,
            normalize_scores=normalize_scores,
            **base_kwargs,
        )

    def _get_self(self) -> AttnWeightsKVCache:
        return self

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        input_pos: int,
    ) -> torch.Tensor:
        _kwargs = dict(
            query=query,
            key=key,
            value=value,
            token_idx=token_idx,
            input_pos=input_pos,
        )
        forward_kwargs = self.call_before_forward(**_kwargs)
        attn_outputs = super().forward(**forward_kwargs)
        self.call_after_forward(attn_outputs)
        return attn_outputs


def args_gradient_new_and_old_spda():
    return [
        a + b + (c,)
        for c, a, b in product(
            available_backends(),
            [
                ("h2o", {"replay_log_blocksize": 64}),
                ("qh2o", {"replay_log_blocksize": 64}),
                ("h2o", {"grace_period": 10, "replay_log_blocksize": 64}),
                ("qh2o", {"grace_period": 12, "replay_log_blocksize": 64}),
            ],
            [
                (512, [511, 1, 8, 4, 8, 2, 8, 2, 8, 8], [2, 3, 3, 2]),
                (504, [503, 1, 4, 4, 8, 4, 8, 2, 8, 2, 8, 8], [2, 2, 3, 3, 2]),
                (512, [503, 1, 4, 4, 8, 4, 8, 2, 8, 2, 8, 8], [4, 3, 3, 2]),
                (512, [384, 128, 8, 16, 4, 24, 4, 4, 16, 8], [2, 3, 2, 3]),
            ],
        )
    ]


@pytest.mark.parametrize(
    "cache_name, cache_kwargs, cache_length, tokens_per_chunk, chunks_per_cell, device",
    args_gradient_new_and_old_spda(),
)
def test_gradient_new_and_old_spda(
    cache_name,
    cache_kwargs,
    cache_length,
    tokens_per_chunk,
    chunks_per_cell,
    device,
):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    print(f"cache_name={cache_name}, cache_kwargs={cache_kwargs}")
    print(f"cache_length={cache_length}\ntokens_per_chunk={tokens_per_chunk}\nchunks_per_cell={chunks_per_cell}")
    dtype = torch.float32
    torch.set_default_dtype(dtype)  # Set default dtype

    qname = "torch-quantized8"
    batch_size = 5
    n_layer = 1
    n_head = 8
    n_query_groups = 4
    head_size = 64
    scale_factor = 1.0 / math.sqrt(head_size)
    vocab_size = 48
    grace_period = cache_kwargs.get("grace_period", 0)
    sdpa_kernels = [
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.MATH,
    ]
    num_chunks = len(tokens_per_chunk)
    block_size = sum(tokens_per_chunk) + 16
    assert sum(chunks_per_cell) == num_chunks

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
        cache_length=cache_length,
        device=device,
        dtype=dtype,
    )
    gpt_model = GPT(config).to(device=device)
    token_idxs = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, config.block_size),
        device=device,
    )
    # HIER: Create special cache!
    _kv_cache = create_kv_cache(
        name=cache_name + "-" + qname,
        params=params,
        block_idx=0,
        **cache_kwargs,
    )
    if cache_name == "h2o":
        kv_cache = H2OLogInputsKVCache(
            config=config,
            buffers=_kv_cache.kv_buffers,
            block_idx=0,
            **cache_kwargs,
        )
    else:
        kv_cache = QuantizedH2OLogInputsKVCache(
            config=config,
            buffers=_kv_cache.kv_buffers,
            block_idx=0,
            **cache_kwargs,
        )
    kv_cache.switch_replay_logging(True)
    # Collect inputs to forward calls
    list_inputs = []
    kv_cache.start_logging(list_inputs)
    gpt_model.assign_kv_caches([kv_cache])

    # Forward pass in inference mode. We record the inputs to
    # `kv_cache.forward`, based on which we do the gradient comparisons
    # below. We also record the replay logs
    print("\nForward inference pass, recording KV cache calls")
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

    assert len(list_inputs) == num_chunks
    # Sanity check:
    for num, entry in zip(tokens_per_chunk, list_inputs):
        assert entry.token_idx.shape == (batch_size, num)
    # Checks on replay logs
    replay_log = get_replay_logs(gpt_model)
    assert len(replay_log) == n_layer
    replay_log = replay_log[0]
    assert len(replay_log) == sum(tokens_per_chunk)
    assert len(replay_log.token_chunks) == num_chunks

    # Loop over chunks. For each chunk, we compute gradients w.r.t.
    # inputs in two different ways. They need to be the same.
    token_chunk_pos = 0
    next_token_pos = 0
    next_grace_pos = None
    index_kwargs = {"dtype": torch.int64, "device": device}
    head_kwargs = {"dtype": dtype, "device": device}
    for entry in list_inputs:
        other = replay_log.token_chunks[token_chunk_pos]
        if not entry.token_idx.to(device=other.device).equal(other):
            raise ValueError(f"token_idx:\n{entry.token_idx}\nreplay_log.token_chunks[{token_chunk_pos}]:\n{other}\nShould be the same!")
        num = other.shape[-1]
        is_prefill = entry.input_pos == 0
        is_cat = entry.input_pos < cache_length
        print(f"input_pos={entry.input_pos}, num={num}, is_prefill={is_prefill}, is_cat={is_cat}")
        # Sampling head gradients at random
        head_attn_outputs = torch.randn(
            *entry.attn_outputs_shape, **head_kwargs,
        )
        head_keys_after = torch.randn(
            *entry.cache_keys_after_shape, **head_kwargs,
        )
        head_values_after = torch.randn(
            *entry.cache_values_after_shape, **head_kwargs,
        )
        gradients = dict()
        # Way 1: Special SDPA operator
        query1 = copy_requires_grad(entry.query)
        key1 = copy_requires_grad(entry.key)
        value1 = copy_requires_grad(entry.value)
        key_buffer1 = None if is_prefill else copy_requires_grad(entry.cache_keys)
        value_buffer1 = None if is_prefill else copy_requires_grad(entry.cache_values)
        index = None
        positions = None
        update_result = None
        token_positions = None
        if is_prefill:
            cache_keys_after1 = key1
            cache_values_after1 = value1
            attn_outputs1 = SDPAFunction.apply(
                query1,
                key1,
                value1,
                None,
                entry.input_pos,
                scale_factor,
                None,
                sdpa_kernels,
                None,
            )
        elif is_cat:
            assert entry.input_pos == key_buffer1.shape[2]
            attn_outputs1, cache_keys_after1, cache_values_after1 = KVCacheCatUpdateAndSDPAFunction.apply(
                query1,
                key1,
                value1,
                key_buffer1,
                value_buffer1,
                scale_factor,
                None,
                sdpa_kernels,
                None,
            )
        else:
            # Scatter case
            index = replay_log.extract_index(
                next_token_pos, num, **index_kwargs
            )
            token_positions = entry.cache_token_pos.clone()
            # Note: `token_positions` is updated here. This is also used in
            # way 2 below
            update_result = update_token_positions(
                token_positions=token_positions,
                next_token_pos=next_token_pos,
                num=num,
                batch_size=batch_size,
                index=index,
                grace_period=grace_period,
                next_grace_pos=next_grace_pos,
            )
            if update_result is not None:
                positions = update_result.positions[0, 0, :]
            else:
                positions = None
            attn_outputs1, cache_keys_after1, cache_values_after1 = KVCacheScatterUpdateAndSDPAFunction.apply(
                query1,
                key1,
                value1,
                key_buffer1,
                value_buffer1,
                index,
                token_positions,  # after update
                entry.input_pos,
                scale_factor,
                positions,
                None,
                sdpa_kernels,
                None,
            )
        attn_outputs1 = attn_outputs1.transpose(1, 2).reshape(batch_size, num, -1)
        loss = (attn_outputs1 * head_attn_outputs).sum() + (
            cache_keys_after1 * head_keys_after
        ).sum() + (cache_values_after1 * head_values_after).sum()
        loss.backward()
        grads1 = dict(
            query=query1.grad,
            key=key1.grad,
            value=value1.grad,
        )
        if not is_prefill:
            grads1["key_buffer"] = key_buffer1.grad
            grads1["value_buffer"] = value_buffer1.grad
        gradients["sdpa_op"] = grads1
        # Way 2: Using PyTorch SDPA with padded query
        query2 = copy_requires_grad(entry.query)
        key2 = copy_requires_grad(entry.key)
        value2 = copy_requires_grad(entry.value)
        key_buffer2 = None if is_prefill else copy_requires_grad(entry.cache_keys)
        value_buffer2 = None if is_prefill else copy_requires_grad(entry.cache_values)
        if is_cat:
            # Cat case
            cache_keys_after2, cache_values_after2, _, _ = cat_on_buffers(
                key2, value2, key_buffer2, value_buffer2,
            )
            token_positions = None
        else:
            # Scatter case
            cache_keys_after2, cache_values_after2 = scatter_on_buffers(
                key2,
                value2,
                key_buffer2,
                value_buffer2,
                index,
                positions,
            )
        attn_outputs2, _ = scaled_dot_product_attention(
            query=query2,
            key=cache_keys_after2,
            value=cache_values_after2,
            scale_factor=scale_factor,
            input_pos=entry.input_pos,
            token_positions=token_positions,
            sdpa_kernels=sdpa_kernels,
        )
        attn_outputs2 = attn_outputs2.transpose(1, 2).reshape(batch_size, num, -1)
        loss = (attn_outputs2 * head_attn_outputs).sum() + (
            cache_keys_after2 * head_keys_after
        ).sum() + (cache_values_after2 * head_values_after).sum()
        loss.backward()
        grads2 = dict(
            query=query2.grad,
            key=key2.grad,
            value=value2.grad,
        )
        if not is_prefill:
            grads2["key_buffer"] = key_buffer2.grad
            grads2["value_buffer"] = value_buffer2.grad
        gradients["pytorch"] = grads2
        # Compare gradients
        for name in gradients["sdpa_op"].keys():
            print(f"Compare grad[{name}]")
            torch.testing.assert_close(
                gradients["sdpa_op"][name],
                gradients["pytorch"][name],
            )
        # Update for next chunk
        token_chunk_pos += 1
        next_token_pos += num
        if grace_period > 0:
            if is_prefill:
                # First slot to move out once the cache is full
                next_grace_pos = cache_length - grace_period
            elif update_result is not None and update_result.num1 == 0:
                # Increment in round-robin fashion (only if num <= grace_period)
                prefix = update_result.prefix
                next_grace_pos = (
                    next_grace_pos - prefix + num
                ) % grace_period + prefix
