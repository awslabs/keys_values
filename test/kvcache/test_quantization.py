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
from functools import partial
from itertools import product
from typing import List

import torch
from torch.linalg import vector_norm
import pytest

from litgpt.config import Config
from litgpt.utils import _RunIf

from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.basics import KVCacheWithBuffers
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.kvcache.quant_buffers import QuantizedKVCacheBuffers
from keys_values.kvcache.quantize.pytorch import TorchBasicQuantizer
from keys_values.kvcache.quantize.bitsandbytes import (
    ALLOWED_BLOCK_SIZE,
    ALLOWED_SOURCE_DTYPES,
    BitsAndBytesQuantizer,
)
from keys_values.kvcache.test_utils import (
    create_kv_cache,
    random_tensor,
    random_keys_values,
    cache_names_and_devices,
    random_args_cache_forward,
    random_index,
)
from keys_values.model import GPT


def args_for_one_cache(cname: str) -> List[tuple]:
    return [
        (a, b) + c for a, b, c in product(
            [torch.float32, torch.float16, torch.bfloat16],
            [False, True],
            cache_names_and_devices(
                filter_name=lambda name: name.startswith(cname) and not name.endswith("default"),
            ),
        )
    ]


# TODO:
# We currently skip blocks_over_heads = True, name = 'dense-bnb-quantized*'.
# Need to understand what is going on there
@pytest.mark.parametrize(
    "dtype, blocks_over_heads, name, device",
    args_for_one_cache("dense"),
)
def test_quantization_error(dtype, blocks_over_heads, name, device):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    if not ("bnb" in name and blocks_over_heads):
        print(f"dtype={dtype}, blocks_over_heads={blocks_over_heads}, name={name}, device={device}")

        if "bnb" in name and not blocks_over_heads:
            # Minimum blocksize for bitsandbytes is 64
            head_sizes = (64, 128, 256)
        else:
            head_sizes = (16, 32, 64)
        max_i = len(head_sizes) - 1
        batch_size = 3
        n_query_groups = 4
        params = [
            KVCacheParams(
                max_batch_size=batch_size * 2 ** (max_i - i),
                n_query_groups=4,
                cache_length=32,
                head_size=head_size,
                n_head=4,
                device=device,
                dtype=dtype,
            )
            for i, head_size in enumerate(head_sizes)
        ]
        cache_length = params[0].cache_length

        kv_caches = [
            create_kv_cache(name, p, blocks_over_heads=blocks_over_heads)
            for p in params
        ]
        keys = random_tensor(params[-1], num=cache_length)
        assert keys.shape == (batch_size, n_query_groups, cache_length, head_sizes[-1])
        # Errors with larger blocksize
        q_errors = []
        for i, kv_cache in enumerate(kv_caches[:-1]):
            # Split blocks into parts
            n_parts = 2 ** (max_i - i)
            head_size = head_sizes[i]
            assert n_parts * head_size == head_sizes[-1]
            assert n_parts * batch_size == params[i].max_batch_size
            _keys = keys.view(*keys.shape[:-1], n_parts, head_size).permute(
                3, 0, 1, 2, 4,
            ).reshape(n_parts * batch_size, n_query_groups, -1, head_size)
            # Errors with smaller blocksize (should be smaller)
            # Only error for keys, ignore for values
            errors = kv_cache.kv_buffers.quantization_error(_keys, _keys)[0].view(
                n_parts, batch_size, n_query_groups, -1,
            )
            assert errors.shape[-1] == cache_length
            errors = vector_norm(errors, dim=0)
            q_errors.append(errors)
        q_errors.append(
            kv_caches[-1].kv_buffers.quantization_error(keys, keys)[0]
        )
        assert q_errors[0].shape == q_errors[1].shape
        assert q_errors[0].shape == q_errors[2].shape
        # Weak test: The smaller the blocksize, the smaller the error should be,
        # but this holds only "on average", since `round` is used in quantization,
        # which is strongly nonlinear
        total_sz = q_errors[0].numel()
        for i in range(2):
            index_lt = torch.lt(q_errors[i + 1], q_errors[i])
            num_lt = int(index_lt.sum().item())
            if num_lt > 0:
                hs_gt = params[i + 1].head_size
                hs_lt = params[i].head_size
                index_lt = index_lt.nonzero()
                print(f"{num_lt} violations of total {total_sz}")
                for row in index_lt:
                    print(f"{row.tolist()}: err{hs_gt} = {q_errors[i + 1][*row]:.7f} < {q_errors[i][*row]:.7f} = err{hs_lt}")
            # Only a fraction of the comparisons should violate the relation
            # which holds "on average"
            assert num_lt < total_sz / 4


@pytest.mark.parametrize(
    "dtype, blocks_over_heads, name, device",
    args_for_one_cache("lastrec"),
)
def test_concatenation(dtype, blocks_over_heads, name, device):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    print(f"name={name}, dtype={dtype}, blocks_over_heads={blocks_over_heads}, device={device}")

    vocab_size = 128
    params = KVCacheParams(
        max_batch_size=3,
        n_query_groups=4,
        cache_length=32,
        head_size=64,
        n_head=4,
        device=device,
        dtype=dtype,
    )
    cache_length = params.cache_length
    kv_cache = create_kv_cache(name, params, blocks_over_heads=blocks_over_heads)
    data = random_args_cache_forward(
        params, num=cache_length, vocab_size=vocab_size,
    )
    kv_cache._prefill(data["key"], data["value"], data["token_idx"])
    positions = random_index(params, 0, cache_length, num=7)
    keys_1, values_1 = kv_cache.kv_buffers.get_slots(positions)
    kv_cache.kv_buffers.set_slots(positions, keys_1, values_1)
    keys_2, values_2 = kv_cache.kv_buffers.get_slots(positions)
    acc_kwargs = dict(rtol=0.01, atol=0.05)
    torch.testing.assert_close(keys_1, keys_2, **acc_kwargs)
    torch.testing.assert_close(values_1, values_2, **acc_kwargs)


@pytest.mark.parametrize(
    "dtype, blocks_over_heads, name, device",
    args_for_one_cache("lastrec"),
)
def test_quantizer_states(dtype, blocks_over_heads, name, device):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)

    vocab_size = 128
    params = KVCacheParams(
        max_batch_size=3,
        n_query_groups=4,
        cache_length=32,
        head_size=64,
        n_head=4,
        device=device,
        dtype=dtype,
    )
    cache_length = params.cache_length
    kv_cache = create_kv_cache(name, params, blocks_over_heads=blocks_over_heads)
    data = random_args_cache_forward(
        params, num=cache_length, vocab_size=vocab_size,
    )
    kv_cache._prefill(data["key"], data["value"], data["token_idx"])
    kv_buffers = kv_cache.kv_buffers
    quantizer_k = kv_buffers.quantizer_k
    quantizer_v = kv_buffers.quantizer_v
    checkpoint = (
        quantizer_k.create_quantizer_state(device=params.device),
        quantizer_v.create_quantizer_state(device=params.device),
    )
    for _ in range(50):
        k_and_v = kv_buffers.get_keys_values()
        before_k = k_and_v.keys().clone()
        before_v = k_and_v.values().clone()
        # Copy certain range into checkpoint
        start = random.randint(0, 3 * cache_length // 4)
        end = random.randint(start + 1, cache_length)
        checkpoint[0].copy_()
        checkpoint[1].copy_()
        # Overwrite this range with new values
        new_keys, new_values = random_keys_values(params, end - start)
        quantizer_k.quantize(start, end, new_keys)
        quantizer_v.quantize(start, end, new_values)
        # Restore from checkpoint
        checkpoint[0].restore()
        checkpoint[1].restore()
        k_and_v = kv_buffers.get_keys_values()
        after_k = k_and_v.keys().clone()
        after_v = k_and_v.values().clone()
        # Content must be restored
        torch.testing.assert_close(before_k, after_k)
        torch.testing.assert_close(before_v, after_v)


_MAX_TEMP_SIZE_IN_BYTES = 2 ** 16

class _TorchBasicQuantizer(TorchBasicQuantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _chunk_size(self, num_slots: int) -> int:
        return max(
            min(num_slots, int(_MAX_TEMP_SIZE_IN_BYTES / self._bytes_per_entry)),
            1,
        )


@_RunIf(min_cuda_gpus=1)
def test_explore_bitsandbytes():
    from bitsandbytes.functional import quantize_4bit, quantize_blockwise

    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)

    device = torch.device("cuda:0")
    num_dtypes = len(ALLOWED_SOURCE_DTYPES)
    num_repeats = 64 * num_dtypes

    code_4 = None
    code_8 = None
    for rep in range(num_repeats):
        blocksize = ALLOWED_BLOCK_SIZE[random.randint(0, len(ALLOWED_BLOCK_SIZE) - 1)]
        if rep % 2 == 0:
            quant_func = partial(quantize_4bit, blocksize=blocksize)
            num_bits = 4
        else:
            quant_func = partial(quantize_blockwise, blocksize=blocksize)
            num_bits = 8
        dtype = ALLOWED_SOURCE_DTYPES[rep % num_dtypes]
        n_query_groups = random.randint(1, 4)
        params = KVCacheParams(
            max_batch_size=random.randint(1, 4),
            n_query_groups=n_query_groups,
            cache_length=random.randint(16, 64),
            head_size=blocksize,
            n_head=n_query_groups,
            device=device,
            dtype=dtype,
        )
        num_channels = params.max_batch_size * n_query_groups * params.cache_length

        x = random_tensor(params, num=params.cache_length)
        shape = (params.max_batch_size, n_query_groups, params.cache_length)
        assert x.shape == shape + (blocksize,)
        q_x, state = quant_func(x)
        lines = [
            f"\nx.shape = {tuple(x.shape)}",
            f"x.dtype = {x.dtype}",
            f"num_bits = {num_bits}",
            f"blocksize = {blocksize}",
            f"num_channels = {num_channels}",
            f"q_x.shape = {tuple(q_x.shape)}",
            f"q_x.dtype = {q_x.dtype}",
            f"absmax.shape = {tuple(state.absmax.shape)}",
            f"absmax.dtype = {state.absmax.dtype}",
            f"state.shape = {state.shape}",
            f"state.dtype = {state.dtype}",
            f"state.offset = {state.offset}",
            f"state.quant_type = {state.quant_type}",
        ]
        print("\n".join(lines))
        assert state.dtype == x.dtype
        assert tuple(state.absmax.shape) == (num_channels,)
        assert state.absmax.dtype == torch.float32
        absmax_cmp = x.to(torch.float32).view(-1, blocksize).abs().max(dim=-1)[0]
        torch.testing.assert_close(state.absmax, absmax_cmp)
        assert state.offset is None
        assert q_x.dtype == torch.uint8
        if num_bits == 4:
            if code_4 is None:
                code_4 = state.code
            else:
                torch.testing.assert_close(state.code, code_4)
            assert tuple(q_x.shape) == (num_channels * blocksize // 2, 1)
            assert state.shape == x.shape
            assert state.quant_type == "fp4"
        else:
            if code_8 is None:
                code_8 = state.code
            else:
                torch.testing.assert_close(state.code, code_8)
            assert q_x.shape == x.shape
            assert state.shape is None
            assert state.quant_type is None
        # Test memory layout for `q_x`
        start = random.randint(0, params.cache_length // 2)
        end = random.randint(start + 1, params.cache_length)
        # Note: Without `contiguous`, this fails! Apparently, `quant_func`
        # needs contiguous memory, fails otherwise
        xpart = x[:, :, start:end, :].contiguous()
        q_xpart, state_part = quant_func(xpart)
        absmax_cmp = xpart.to(torch.float32).view(-1, blocksize).abs().max(dim=-1)[0]
        torch.testing.assert_close(state_part.absmax, absmax_cmp)
        num = end - start
        full = state.absmax.view(*shape)
        part = state_part.absmax.view(*shape[:-1], num)
        torch.testing.assert_close(full[:, :, start:end], part)
        if num_bits == 4:
            size_parts = params.max_batch_size * n_query_groups * num
            assert tuple(state_part.absmax.shape) == (size_parts,)
            assert tuple(q_xpart.shape) == (size_parts * blocksize // 2, 1)
            full = q_x.view(*shape, -1)
            part = q_xpart.view(*shape[:-1], num, -1)
            assert full.shape[-1] == blocksize // 2
            assert part.shape[-1] == blocksize // 2
            torch.testing.assert_close(full[:, :, start:end, :], part)
        else:
            torch.testing.assert_close(q_x[:, :, start:end, :], q_xpart)


def args_bitsandbytes_with_blocks_over_heads() -> List[tuple]:
    qnames = ["bnb-quantized8", "bnb-quantized4"]
    # (batch_size, n_query_groups, head_size, blocksize, reminder, is_valid)
    args = [
        (1, 16, 5 * 16, 256, 5, True),
        (3 * 4, 8, 7 * 16, 512, 3 * 7, True),
        (1, 1, 1024, 1024, 1, True),
        (16, 32, 17 * 8, 4096, 17, True),
        (4, 3 * 4, 19 * 2, 32, 3 * 19, False),
        (3 * 16, 5 * 32, 7 * 16, 4096, 2 * 3 * 5 * 7, True),
        (1, 2 * 13, 4 * 5, 8, 13 * 5, False),
    ]
    return [
        (qname,) + tup[:3] + ((tup[4], tup[3] // (i + 1)), tup[-1])
        for i, qname in enumerate(qnames)
        for tup in args
    ]


@_RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    "qname, batch_size, n_query_groups, head_size, shape, is_valid",
    args_bitsandbytes_with_blocks_over_heads(),
)
def test_bitsandbytes_with_blocks_over_heads(
    qname, batch_size, n_query_groups, head_size, shape, is_valid,
):
    name = "lastrec-" + qname
    device = torch.device("cuda:0")
    print(f"qname={qname}, batch_size={batch_size}, n_query_groups={n_query_groups}, head_size={head_size}, shape={shape}, is_valid={is_valid}")
    params = KVCacheParams(
        max_batch_size=batch_size,
        n_query_groups=n_query_groups,
        cache_length=32,
        head_size=head_size,
        n_head=n_query_groups * 2,
        device=device,
        dtype=torch.float32,
    )
    if is_valid:
        kv_cache = create_kv_cache(name, params, blocks_over_heads=True)
        quantizer_k = kv_cache.kv_buffers.quantizer_k
        assert isinstance(quantizer_k, BitsAndBytesQuantizer)
        quant_shape = quantizer_k._quant_shape
        required_shape = (params.cache_length,) + shape
        assert quant_shape == required_shape
    else:
        with pytest.raises(ValueError):
            kv_cache = create_kv_cache(name, params, blocks_over_heads=True)


def write_back_all(caches: List[KVCacheWithBuffers]):
    for cache in caches:
        cache.kv_buffers.write_back()


def compare_buffers(
    caches1: List[KVCacheWithBuffers], caches2: List[KVCacheWithBuffers],
):
    assert len(caches1) == len(caches2)
    for block_idx, (cache1, cache2) in enumerate(zip(caches1, caches2)):
        buffer1 = cache1.kv_buffers
        buffer2 = cache2.kv_buffers
        assert isinstance(buffer1, QuantizedKVCacheBuffers)
        assert isinstance(buffer2, QuantizedKVCacheBuffers)
        if isinstance(buffer1.quantizer_k, TorchBasicQuantizer):
            compare_these = [
                (buffer1.quantizer_k.quant_scales, buffer2.quantizer_k.quant_scales, "k_quant_scales"),
                (buffer1.quantizer_k.quant_zero_points, buffer2.quantizer_k.quant_zero_points, "k_quant_zero_points"),
                (buffer1.quantizer_v.quant_scales, buffer2.quantizer_v.quant_scales, "v_quant_scales"),
                (buffer1.quantizer_v.quant_zero_points, buffer2.quantizer_v.quant_zero_points, "v_quant_zero_points"),
                (buffer1.quantizer_k.quant_buffer, buffer2.quantizer_k.quant_buffer, "k_quant_buffer"),
                (buffer1.quantizer_v.quant_buffer, buffer2.quantizer_v.quant_buffer, "v_quant_buffer"),
            ]
        else:
            compare_these = [
                (buffer1.quantizer_k.quant_absmax, buffer2.quantizer_k.quant_absmax, "k_quant_absmax"),
                (buffer1.quantizer_v.quant_absmax, buffer2.quantizer_v.quant_absmax, "v_quant_absmax"),
                (buffer1.quantizer_k.quant_buffer, buffer2.quantizer_k.quant_buffer, "k_quant_buffer"),
                (buffer1.quantizer_v.quant_buffer, buffer2.quantizer_v.quant_buffer, "v_quant_buffer"),
            ]
        for x1, x2, name in compare_these:
            print(f"Comparing {block_idx}: {name}")
            torch.testing.assert_close(x1, x2)


def check_same_events(
    cache1: KVCacheWithBuffers, caches2: List[KVCacheWithBuffers],
):
    events_all = [str(x) for x in cache1.kv_buffers.dequant_buffers.debug_events]
    _events_all = set(events_all)
    events_sep = [
        [str(x) for x in cache.kv_buffers.dequant_buffers.debug_events]
        for cache in caches2
    ]
    _events_sep = set()
    for lst in events_sep:
        _events_sep.update(lst)
    if _events_all != _events_sep:
        lines = ["Event log for common:"] + events_all
        for idx, events in enumerate(events_sep):
            lines.append(f"Event log for cache in layer {idx}")
            lines.extend(events)
        print("\n".join(lines))
        assert 1 == 0


def args_quantized_buffers_write_back() -> List[tuple]:
    args = []
    for cname in ("lastrec", "h2o"):
        args.extend([(a, c, d) for a, b, c, d in args_for_one_cache(cname) if b])
    return args


@pytest.mark.parametrize(
    "dtype, name, device", args_quantized_buffers_write_back(),
)
def test_quantized_buffers_write_back(dtype, name, device):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    batch_size = 4
    cache_length = 64

    config = Config(
        n_layer=8,
        n_head=8,
        n_query_groups=4,
        n_embd=8 * 64,
        block_size=128,
        vocab_size=48,
        rotary_percentage=1,
    )
    params = KVCacheParams(
        max_batch_size=batch_size,
        n_query_groups=config.n_query_groups,
        cache_length=cache_length,
        head_size=config.head_size,
        n_head=config.n_head,
        device=device,
        dtype=dtype,
    )
    with torch.device(device):
        gpt_model = GPT(config)
    gpt_model.apply(gpt_model._init_weights)  # Initialize
    # Create caches
    # Share the same dequantization buffers
    caches_common = KVCacheFactory.create(
        gpt_model=gpt_model,
        name=name,
        max_batch_size=batch_size,
        dtype=dtype,
        cache_length=cache_length,
    )
    caches_common[0].kv_buffers.dequant_buffers.start_debug_event_protocol()
    # Separate dequantization buffers
    caches_separate = [
        KVCacheFactory.create_single(
            name=name,
            config=config,
            max_batch_size=batch_size,
            cache_length=cache_length,
            block_idx=block_idx,
            device=device,
            dtype=dtype,
        )
        for block_idx in range(config.n_layer)
    ]
    # Do the same with different caches
    # Prefill
    print(f"name={name}, dtype={dtype}, device={device}")
    print(f"Prefill: {cache_length}")
    input_pos = 0
    num_prefill = caches_common[0].max_prefill_length
    if num_prefill is None:
        num_prefill = cache_length
    for c_comm, c_sep in zip(caches_common, caches_separate):
        c_sep.kv_buffers.dequant_buffers.start_debug_event_protocol()
        data = random_args_cache_forward(
            params, num=num_prefill, vocab_size=config.vocab_size,
        )
        c_comm(**data, input_pos=input_pos)
        c_sep(**data, input_pos=input_pos)
    write_back_all(caches_common)
    write_back_all(caches_separate)
    check_same_events(caches_common[0], caches_separate)
    compare_buffers(caches_common, caches_separate)
    input_pos += num_prefill
    # Several updates
    for n_upd in range(5):
        q_len = min(
            random.randint(1, cache_length // 2),
            caches_common[0].max_tokens_forward,
        )
        print(f"Update {n_upd}: {q_len}")
        for c_comm, c_sep in zip(caches_common, caches_separate):
            # If this is not done, the dequant buffers content is used without
            # reading from quantized, which gives differences
            c_sep.kv_buffers.drop_association()
            data = random_args_cache_forward(
                params, num=q_len, vocab_size=config.vocab_size,
            )
            c_comm(**data, input_pos=input_pos)
            c_sep(**data, input_pos=input_pos)
        write_back_all(caches_common)
        write_back_all(caches_separate)
        check_same_events(caches_common[0], caches_separate)
        compare_buffers(caches_common, caches_separate)
        input_pos += q_len
