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
import copy
from itertools import product
from typing import Any, Dict, List, Optional

import pytest
import torch

from keys_values.config import Config
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.buffers import DefaultKVCacheBuffers
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.kvcache.quant_buffers import QuantizedKVCacheBuffers
from keys_values.kvcache.tentative import (
    TentativeKVCacheUpdates,
    UndoKVCacheBuffers,
)
from keys_values.kvcache.test_utils import (
    available_backends,
    cache_name_gpu_only,
    create_kv_cache,
    filter_cache_names,
    random_args_cache_forward,
    range_from_args,
)
from keys_values.model import GPT

# Windows of tentative tokens: one call (like the target model in speculative
# decoding), single-token calls (like the draft model), and mixed. Prefill and
# warm-up sizes are chosen so that the ring buffer of `lastrec` wraps around
# inside the window in some cases, and that windows cross the point where the
# cache becomes full.
WINDOWS = ([2], [5], [1, 1], [1, 1, 1, 1], [3, 1, 2], [2, 4])


def _scenarios(cache_length: int, warmups=(0, 3, 9, 20)):
    return list(product((6, cache_length), warmups, WINDOWS))


def _params(cache_length: int, dtype: torch.dtype) -> KVCacheParams:
    return KVCacheParams(
        max_batch_size=2,
        n_query_groups=2,
        cache_length=cache_length,
        head_size=8,
        n_head=4,
        dtype=dtype,
    )


def _create_caches(name: str, params: KVCacheParams, n_layer: int):
    """
    For quantized buffers, the caches of all layers share a de-quantized
    buffer, as in a model.

    """
    config = Config(
        n_embd=params.n_head * params.head_size,
        n_head=params.n_head,
        n_query_groups=params.n_query_groups,
        n_layer=n_layer,
        vocab_size=64,
    )
    torch.random.manual_seed(0)
    return KVCacheFactory.create(
        gpt_model=GPT(config),
        name=name,
        max_batch_size=params.max_batch_size,
        cache_length=params.cache_length,
        dtype=params.dtype,
    )


def _cache_state(kv_cache) -> Dict[str, Any]:
    current_length = kv_cache.current_length
    k_and_v = kv_cache.get_keys_values()
    result = dict(
        input_pos=kv_cache.input_pos,
        current_length=current_length,
        token_positions=kv_cache.token_positions()[0, 0, :].tolist(),
        next_position=getattr(kv_cache, "next_position", None),
        buffers_type=type(kv_cache.kv_buffers),
        keys=k_and_v.keys()[:, :, :current_length, :].clone(),
        values=k_and_v.values()[:, :, :current_length, :].clone(),
    )
    if isinstance(kv_cache.kv_buffers, QuantizedKVCacheBuffers):
        buffers = kv_cache.kv_buffers
        buffers.write_back()
        for kind, quantizer in (("k", buffers.quantizer_k), ("v", buffers.quantizer_v)):
            result[f"quant_{kind}"] = _quantizer_storage(quantizer)
    return result


# Quantized content and quantization states, depending on the quantizer:
# `TorchBasicQuantizer` uses `quant_scales` and `quant_zero_points`,
# `BitsAndBytesQuantizer` uses `quant_absmax`
QUANTIZER_STORAGE_NAMES = (
    "quant_buffer",
    "quant_scales",
    "quant_zero_points",
    "quant_absmax",
)


def _quantizer_storage(quantizer) -> Dict[str, torch.Tensor]:
    result = {
        name: getattr(quantizer, name).clone()
        for name in QUANTIZER_STORAGE_NAMES
        if getattr(quantizer, name, None) is not None
    }
    assert "quant_buffer" in result and len(result) >= 2, (
        type(quantizer),
        list(result),
    )
    return result


def _run(
    name: str,
    params: KVCacheParams,
    n_layer: int,
    prefill: int,
    warmup: int,
    window: List[int],
    num_keep: int,
    use_tentative: bool,
    seed: int,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    If `use_tentative`, inserts all tokens of `window` as tentative updates
    and keeps the first `num_keep` of them. Otherwise, the same chunks are
    inserted directly, but truncated to their first `num_keep` tokens.
    Finally, one more token is inserted. Returns cache states and this
    attention output for every layer.

    """
    torch.random.manual_seed(seed)
    total = prefill + warmup + sum(window) + 1
    kv_caches = _create_caches(name, params, n_layer)
    data = [
        random_args_cache_forward(params, total, vocab_size=64, device=device)
        for _ in kv_caches
    ]

    def forward_all(start: int, end: int) -> List[torch.Tensor]:
        return [
            cache(**range_from_args(d, start, end)) for cache, d in zip(kv_caches, data)
        ]

    forward_all(0, prefill)
    pos = prefill
    for _ in range(warmup):
        forward_all(pos, pos + 1)
        pos += 1
    if use_tentative:
        updates = TentativeKVCacheUpdates(kv_caches)
        updates.start()
        for num in window:
            forward_all(pos, pos + num)
            pos += num
        assert updates.num_tentative == sum(window)
        updates.keep(num_keep)
        assert not updates.is_active
    else:
        num_left = num_keep
        for num in window:
            num_ins = min(num, num_left)
            if num_ins > 0:
                forward_all(pos, pos + num_ins)
            num_left -= num_ins
            pos += num
    attn_outputs = forward_all(total - 1, total)
    results = []
    for cache, attn_output in zip(kv_caches, attn_outputs):
        state = _cache_state(cache)
        state["attn_output"] = attn_output
        results.append(state)
    return results


def _assert_same(states_a, states_b, exact: bool = True, atol: float = 0.0):
    for state_a, state_b in zip(states_a, states_b):
        for key in (
            "input_pos",
            "current_length",
            "token_positions",
            "next_position",
            "buffers_type",
        ):
            assert state_a[key] == state_b[key], (key, state_a[key], state_b[key])
        for key in ("keys", "values", "attn_output"):
            if exact:
                torch.testing.assert_close(state_a[key], state_b[key], atol=0, rtol=0)
            else:
                torch.testing.assert_close(
                    state_a[key].float(), state_b[key].float(), atol=atol, rtol=0
                )
        if exact:
            for key in ("quant_k", "quant_v"):
                if key in state_a:
                    storage_a, storage_b = state_a[key], state_b[key]
                    assert storage_a.keys() == storage_b.keys(), key
                    for name, x in storage_a.items():
                        assert x.equal(storage_b[name]), (key, name)


def _default_cases():
    cache_length = 16
    result = []
    for device, name, (prefill, warmup, window) in product(
        available_backends(),
        ["dense-default", "lastrec-default"],
        _scenarios(cache_length),
    ):
        if name.startswith("dense") and prefill + warmup + sum(window) >= cache_length:
            continue
        for num_keep in range(sum(window) + 1):
            result.append((device, name, prefill, warmup, window, num_keep))
    return result


@pytest.mark.parametrize(
    "device, name, prefill, warmup, window, num_keep", _default_cases()
)
def test_keep_default_buffers(device, name, prefill, warmup, window, num_keep):
    """
    Keeping a prefix of tentative updates gives the same cache as inserting
    only these tokens: positions, buffer content and the attention output of
    a further token, bit-exact.

    """
    params = _params(16, torch.float32)
    kwargs = dict(
        name=name,
        params=params,
        n_layer=1,
        prefill=prefill,
        warmup=warmup,
        window=window,
        num_keep=num_keep,
        seed=100 + num_keep,
        device=device,
    )
    states = _run(use_tentative=True, **kwargs)
    states_ref = _run(use_tentative=False, **kwargs)
    _assert_same(states, states_ref)
    assert states[0]["buffers_type"] is DefaultKVCacheBuffers


QUANTIZED_NAMES = [
    "lastrec-torch-quantized8",
    "lastrec-torch-quantized4",
    "dense-torch-quantized8",
    # Only run if CUDA is available:
    "lastrec-bnb-quantized8",
    "lastrec-bnb-quantized4",
]


def _quantized_cases(cache_length: int):
    result = []
    for device, name, (prefill, warmup, window) in product(
        available_backends(),
        filter_cache_names(QUANTIZED_NAMES),
        _scenarios(cache_length, warmups=(0, 3, 9)),
    ):
        if cache_name_gpu_only(name) and device.type != "cuda":
            continue
        if name.startswith("dense") and prefill + warmup + sum(window) >= cache_length:
            continue
        for num_keep in range(sum(window) + 1):
            result.append((device, name, prefill, warmup, window, num_keep))
    return result


@pytest.mark.parametrize(
    "device, name, prefill, warmup, window, num_keep",
    _quantized_cases(cache_length=32),
)
def test_keep_quantized_buffers(device, name, prefill, warmup, window, num_keep):
    """
    Same for quantized buffers, with the caches of three layers sharing a
    de-quantized buffer. The kept tokens are quantized directly into their
    slots, so the quantized content (and everything else) is bit-exact the
    same as when inserting only these tokens.

    Note: Here, at most 20% of the slots are written by any call. Otherwise,
    the reference writes back the whole cache, see
    :func:`test_keep_quantized_buffers_large_calls`.

    """
    params = _params(32, torch.bfloat16)
    kwargs = dict(
        name=name,
        params=params,
        n_layer=3,
        prefill=prefill,
        warmup=warmup,
        window=window,
        num_keep=num_keep,
        seed=200 + num_keep,
        device=device,
    )
    states = _run(use_tentative=True, **kwargs)
    states_ref = _run(use_tentative=False, **kwargs)
    _assert_same(states, states_ref)
    assert states[0]["buffers_type"] is QuantizedKVCacheBuffers


@pytest.mark.parametrize(
    "name, atol",
    [("lastrec-torch-quantized8", 0.0625), ("lastrec-torch-quantized4", 0.3)],
)
def test_keep_quantized_buffers_large_calls(name, atol):
    """
    If a call writes more than 20% of the slots, de-quantized buffers write
    back (re-quantize) the whole cache. This happens for the reference, but
    not when keeping tentative updates, which only quantizes the kept slots.
    Bookkeeping must still be exact, and contents agree up to the error of
    re-quantization.

    """
    params = _params(16, torch.bfloat16)
    for prefill, warmup, window in _scenarios(16, warmups=(0, 9)):
        for num_keep in range(sum(window) + 1):
            kwargs = dict(
                name=name,
                params=params,
                n_layer=3,
                prefill=prefill,
                warmup=warmup,
                window=window,
                num_keep=num_keep,
                seed=300 + num_keep,
            )
            states = _run(use_tentative=True, **kwargs)
            states_ref = _run(use_tentative=False, **kwargs)
            _assert_same(states, states_ref, exact=False, atol=atol)


# Windows of tentative tokens as in speculative decoding, `(calls, num_keep)`:
# the target model inserts all tokens of a window in one call, the draft model
# inserts one token per call
TARGET_WINDOWS = [([4], 1), ([4], 4), ([4], 0), ([2], 2), ([4], 3), ([1], 1)]
DRAFT_WINDOWS = [([1, 1, 1], 1), ([1, 1, 1], 3), ([1, 1], 0), ([1, 1, 1], 2)]

# Cache lengths: `lastrec` caches evict tokens
MODEL_CACHE_LENGTHS = {
    "dense-default": 128,
    "lastrec-default": 12,
    "dense-torch-quantized8": 128,
    "lastrec-torch-quantized8": 12,
    "lastrec-torch-quantized4": 32,
}

# Tolerance for logits of calls after the first one in a window, with
# quantized buffers. About 10 times the largest difference we observed (5e-4
# for 8 bit, 1e-2 for 4 bit). For 8 bit, this detects replay buffers which
# replay only some of the updates (differences about 5e-2). For 4 bit, the
# tolerance is too large for that, so only the first call of a window is
# checked tightly.
REPLAY_LOGITS_ATOL = {"quantized8": 5e-3, "quantized4": 1e-1}


@pytest.mark.parametrize(
    "device, name, pattern",
    list(
        product(
            available_backends(),
            list(MODEL_CACHE_LENGTHS.keys()),
            ["target", "draft"],
        )
    ),
)
def test_logits_same_as_normal_mode(device, name, pattern):
    """
    Logits computed while updates are tentative, compared to the same calls in
    normal mode: before each window, the model is copied, and the copy makes
    the same calls. Logits of the first call of a window must be bit-exact the
    same, which covers the target model in speculative decoding. With default
    buffers, this holds for all calls. With quantized buffers, entries
    inserted before in the same window are not quantized in replay mode, so
    later calls differ by about the quantization error.

    """
    vocab_size, prompt_len, num_repeats = 32, 6, 5
    config = Config(
        block_size=128,
        vocab_size=vocab_size,
        padded_vocab_size=vocab_size,
        n_layer=2,
        n_head=4,
        n_query_groups=2,
        n_embd=32,
        rotary_percentage=1.0,
    )
    torch.random.manual_seed(1)
    with torch.device(device):
        gpt_model = GPT(config).eval()
    gpt_model.max_seq_length = config.block_size
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            gpt_model=gpt_model,
            name=name,
            max_batch_size=1,
            cache_length=MODEL_CACHE_LENGTHS[name],
            device=device,
            dtype=torch.float32,
            cache_kwargs=(
                dict(init_grace_tokens=2) if name.startswith("lastrec") else None
            ),
        )
    )
    atol = next(
        (v for k, v in REPLAY_LOGITS_ATOL.items() if name.endswith(k)),
        0.0,
    )
    updates = TentativeKVCacheUpdates(gpt_model.get_kv_caches())

    def tokens(num: int) -> torch.Tensor:
        return torch.randint(0, vocab_size, (1, num), device=device)

    with torch.inference_mode():
        gpt_model(tokens(prompt_len))
        windows = TARGET_WINDOWS if pattern == "target" else DRAFT_WINDOWS
        for calls, num_keep in windows * num_repeats:
            model_copy = copy.deepcopy(gpt_model)
            updates.start()
            for pos, num in enumerate(calls):
                inputs = tokens(num)
                logits = gpt_model(inputs)
                logits_normal = model_copy(inputs)
                torch.testing.assert_close(
                    logits,
                    logits_normal,
                    atol=0.0 if pos == 0 else atol,
                    rtol=0.0,
                )
            updates.keep(num_keep)


@pytest.mark.parametrize(
    "name",
    ["dense-default", "lastrec-default", "lastrec-torch-quantized8"],
)
def test_keep_repeated(name):
    """
    Speculative decoding starts tentative updates, inserts tokens over several
    calls, keeps a prefix, and repeats. The caches must track the sequence
    exactly.

    """
    dtype = torch.bfloat16 if "quantized" in name else torch.float32
    cache_length = 64 if "quantized" in name else 16
    params = _params(cache_length, dtype)
    total = 200 if name.startswith("lastrec") else cache_length
    n_layer = 2
    caches = _create_caches(name, params, n_layer)
    reference = _create_caches(name, params, n_layer)
    torch.random.manual_seed(1234)
    data = [random_args_cache_forward(params, total, vocab_size=64) for _ in caches]

    def forward(kv_caches, start, end):
        return [c(**range_from_args(d, start, end)) for c, d in zip(kv_caches, data)]

    # Prefill must not be shorter than `init_grace_tokens` of lastrec
    prefill = 10
    forward(caches, 0, prefill)
    forward(reference, 0, prefill)
    updates = TentativeKVCacheUpdates(caches)
    pos = prefill
    gen = torch.Generator().manual_seed(5)
    while True:
        num_draft = int(torch.randint(1, 4, (1,), generator=gen))
        # Leave room for a final token
        if pos + num_draft + 2 > total:
            break
        updates.start()
        forward(caches, pos, pos + 1)
        for i in range(num_draft):
            p = pos + 1 + i
            forward(caches, p, p + 1)
        num_keep = int(torch.randint(1, num_draft + 2, (1,), generator=gen))
        updates.keep(num_keep)
        forward(reference, pos, pos + num_keep)
        pos += num_keep
        for cache, ref in zip(caches, reference):
            assert cache.input_pos == ref.input_pos == pos
    assert pos < total
    outputs = forward(caches, pos, pos + 1)
    outputs_ref = forward(reference, pos, pos + 1)
    for cache, ref, output, output_ref in zip(caches, reference, outputs, outputs_ref):
        state, state_ref = _cache_state(cache), _cache_state(ref)
        state["attn_output"] = output
        state_ref["attn_output"] = output_ref
        _assert_same([state], [state_ref])


@pytest.mark.parametrize(
    "name",
    ["dense-default", "lastrec-default", "lastrec-torch-quantized8"],
)
def test_discard_restores(name):
    """
    Discarding restores everything, also if the caches received different
    numbers of tokens (as after a failed forward pass).

    """
    dtype = torch.bfloat16 if "quantized" in name else torch.float32
    params = _params(16, dtype)
    for prefill, warmup in product((6, 16), (0, 9)):
        if name.startswith("dense") and prefill + warmup + 5 >= 16:
            continue
        states = []
        for use_tentative in (True, False):
            torch.random.manual_seed(7)
            caches = _create_caches(name, params, n_layer=3)
            total = prefill + warmup + 5
            data = [
                random_args_cache_forward(params, total, vocab_size=64) for _ in caches
            ]
            for c, d in zip(caches, data):
                c(**range_from_args(d, 0, prefill))
            for p in range(prefill, prefill + warmup):
                for c, d in zip(caches, data):
                    c(**range_from_args(d, p, p + 1))
            pos = prefill + warmup
            if use_tentative:
                updates = TentativeKVCacheUpdates(caches)
                updates.start()
                for c, d in zip(caches, data):
                    c(**range_from_args(d, pos, pos + 2))
                # Only the first layer receives more tokens
                caches[0](**range_from_args(data[0], pos + 2, pos + 4))
                with pytest.raises(IndexError):
                    updates.keep(1)
                updates.discard()
            outputs = [
                c(**range_from_args(d, total - 1, total)) for c, d in zip(caches, data)
            ]
            result = [_cache_state(c) for c in caches]
            for state, output in zip(result, outputs):
                state["attn_output"] = output
            states.append(result)
        _assert_same(states[0], states[1])


def test_errors():
    params = _params(16, torch.float32)
    torch.random.manual_seed(3)
    data = random_args_cache_forward(params, 10, vocab_size=64)
    with pytest.raises(NotImplementedError, match="H2OKVCache"):
        TentativeKVCacheUpdates([create_kv_cache("h2o-default", params)])
    kv_cache = create_kv_cache("lastrec-default", params)
    updates = TentativeKVCacheUpdates([kv_cache])
    with pytest.raises(IndexError, match="prefilled"):
        updates.start()
    kv_cache(**range_from_args(data, 0, 4))
    with pytest.raises(IndexError, match="not been started"):
        updates.keep(0)
    kv_cache.switch_replay_logging(True)
    with pytest.raises(IndexError, match="replay logging"):
        updates.start()
    kv_cache.switch_replay_logging(False)
    kv_cache.set_checkpoint_hook(lambda buffers, chunk_idx: None)
    with pytest.raises(IndexError, match="checkpoint hook"):
        updates.start()
    kv_cache.set_checkpoint_hook(None)
    updates.start()
    assert isinstance(kv_cache.kv_buffers, UndoKVCacheBuffers)
    with pytest.raises(IndexError, match="already been started"):
        updates.start()
    kv_cache(**range_from_args(data, 4, 6))
    with pytest.raises(ValueError):
        updates.keep(3)
    with pytest.raises(NotImplementedError, match="tentative"):
        kv_cache.deallocate_buffers()
    updates.keep(2)
    assert type(kv_cache.kv_buffers) is DefaultKVCacheBuffers
    assert kv_cache.input_pos == 6


def test_failed_keep_resets_caches():
    """
    If :meth:`TentativeKVCacheUpdates.keep` fails for one of the caches, the
    caches done before it would hold more tokens than the others. Instead, all
    caches are switched back to their original buffers and reset, and can be
    used again after prefill.

    """
    params = _params(16, torch.float32)
    caches = _create_caches("lastrec-default", params, n_layer=3)
    torch.random.manual_seed(9)
    data = [random_args_cache_forward(params, 20, vocab_size=64) for _ in caches]
    for c, d in zip(caches, data):
        c(**range_from_args(d, 0, 10))
    updates = TentativeKVCacheUpdates(caches)
    updates.start()
    for c, d in zip(caches, data):
        c(**range_from_args(d, 10, 13))

    def failing_commit_prefix(num: int):
        raise RuntimeError("commit failed")

    caches[1].kv_buffers.commit_prefix = failing_commit_prefix
    with pytest.raises(RuntimeError, match="commit failed"):
        updates.keep(2)
    assert not updates.is_active
    for cache in caches:
        assert type(cache.kv_buffers) is DefaultKVCacheBuffers
        assert cache.input_pos == 0
    # Caches can be used again
    for c, d in zip(caches, data):
        c(**range_from_args(d, 0, 10))
        c(**range_from_args(d, 10, 11))
        assert c.input_pos == 11
