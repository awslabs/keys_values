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
from typing import Tuple, Optional, List, Dict, Callable
import math
from functools import partial
from itertools import product

import torch

from litgpt.config import Config

from keys_values.attention import (
    KeysAndValues,
    eager_scaled_dot_product_attention,
    DefaultKeysAndValues,
)
from keys_values.attention_utils import build_mask_cache
from keys_values.kvcache.base import KVCacheParams, KVCache
from keys_values.kvcache.buffers import DefaultKVCacheBuffers
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.kvcache.gradient.accumulate import GradientAccumulator
from keys_values.kvcache.gradient.checkpoints import KVCacheBufferCheckpoints


# Tests run quite slowly for "mps". If this changes, switch this to True
RUN_TESTS_FOR_MPS = False


def create_kv_cache(
    name: str,
    params: KVCacheParams,
    block_idx: int = 0,
    **kwargs,
) -> KVCache:
    config = Config(
        n_embd=params.n_head * params.head_size,
        n_head=params.n_head,
        n_query_groups=params.n_query_groups,
        n_layer=1,
    )
    return KVCacheFactory.create_single(
        name=name,
        config=config,
        max_batch_size=params.max_batch_size,
        cache_length=params.cache_length,
        block_idx=block_idx,
        dtype=params.dtype,
        cache_kwargs=kwargs,
    )


def tensor_is_simple(x: torch.Tensor) -> bool:
    assert x.ndim > 1
    x = x.view(-1, x.shape[-1])
    other = x[0].unsqueeze(0).expand(*x.shape)
    return x.equal(other)


def random_tensor(
    params: KVCacheParams,
    num: Optional[int] = None,
    is_query: bool = False,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if batch_size is None:
        batch_size = params.max_batch_size
    if num is None:
        num = params.cache_length
    dim1 = params.n_head if is_query else params.n_query_groups
    shape = (batch_size, dim1, num, params.head_size)
    return torch.randn(*shape, device=device, dtype=params.dtype)


def random_keys_values(
    params: KVCacheParams,
    num: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    keys = random_tensor(params, num, device=device)
    values= random_tensor(params, num, device=device)
    return keys, values


def random_index(
    params: KVCacheParams,
    start: int,
    end: int,
    num: Optional[int] = None,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    if batch_size is None:
        batch_size = params.max_batch_size
    if num is None:
        num = params.cache_length
    diff = end - start
    if diff < num:
        raise ValueError(f"end - start = {diff}, must be >= num = {num}")
    index_kwargs = dict(dtype=torch.int64, device=device)
    result = torch.empty(
        (batch_size, params.n_query_groups, num), **index_kwargs,
    )
    for b in range(batch_size):
        for h in range(params.n_query_groups):
            result[b, h, :] = (
                torch.randperm(diff, **index_kwargs) + start
            )[:num]
    return result


def compute_attn_weights(
    query: torch.Tensor,
    k_and_v: KeysAndValues,
    **sdpa_kwargs,
) -> torch.Tensor:
    q_len, head_size = query.shape[-2:]
    kv_len = k_and_v.keys().shape[-2]
    assert q_len <= kv_len
    # Need causal mask
    kwargs = dict(dtype=query.dtype, device=query.device)
    mask = build_mask_cache(
        max_seq_length=q_len,
        sliding_window_size=None,
        **kwargs,
    )
    if q_len < kv_len:
        _pad_zeros = torch.zeros((1, 1), **kwargs).expand(q_len, kv_len - q_len)
        mask = torch.cat((mask, _pad_zeros), dim=-1)
    _, attn_weights = eager_scaled_dot_product_attention(
        query=query,
        k_and_v=k_and_v,
        scale_factor=1.0 / math.sqrt(head_size),
        use_blocking=False,
        return_attn_weights=True,
        input_pos=0,
        token_positions=None,
        sliding_window_size=None,
        mask=mask,
        **sdpa_kwargs,
    )
    return attn_weights


def attention_mask_forward(
    q_len: int,
    k_len: int,
    dtype: torch.dtype,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    assert q_len <= k_len
    if q_len <= 1:
        return None
    if device is None:
        device = torch.device("cpu")
    mask = torch.cat(
(
            torch.zeros(q_len, k_len - q_len, dtype=dtype, device=device),
            torch.ones(q_len, q_len, dtype=dtype, device=device).triu(diagonal=1),
        ),
        dim=-1,
    )
    mask.masked_fill_(mask.bool(), torch.finfo(dtype).min)
    return mask.view(1, 1, q_len, k_len)


def test_bitsandbytes() -> bool:
    from bitsandbytes.cextension import COMPILED_WITH_CUDA

    return COMPILED_WITH_CUDA


class KVCacheBufferTestingCheckpoints(KVCacheBufferCheckpoints):
    """
    Checkpointing class used for testing. The checkpoints are not quantized,
    but the buffers are stored as they are. This is not recommended in
    practice, but simplifies gradient testing. Also, we do not reserve
    memory for checkpoints up front, but copy them as they come in.

    """
    def __init__(
        self,
        chunk_numbers: List[int],
    ):
        super().__init__(chunk_numbers)
        self._checkpoints: List[
            Optional[DefaultKeysAndValues]
        ] = [None] * len(chunk_numbers)

    def _set_checkpoint(
        self,
        pos: int,
        buffers: DefaultKVCacheBuffers,
    ) -> int:
        k_and_v = buffers.get_keys_values()
        device = torch.device("cpu")
        self._checkpoints[pos] = DefaultKeysAndValues(
            keys=k_and_v.keys().to(device=device, copy=True),
            values=k_and_v.values().to(device=device, copy=True),
        )
        return pos

    def _get_checkpoint(
        self,
        pos: int,
        out: DefaultKVCacheBuffers,
    ):
        checkpoint = self._checkpoints[pos]
        if checkpoint is None:
            raise ValueError(f"checkpoint at pos={pos} is still empty. Use 'set_checkpoint'")
        out.prefill_from_keys_values(checkpoint)


def copy_gradients(
    model: torch.nn.Module, device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    return {
        name: param.grad.data.to(device=device, copy=True)
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def exchange_kv_cache_checkpoints(accumulator: GradientAccumulator):
    """
    Ensures that `accumulator._kv_cache_checkpoints` are of testing type
    :class:`KVCacheBufferTestingCheckpoints`. These do not quantize checkpoints,
    which simplifies gradient testing a lot.

    """
    def wrapped_create_checkpoints_and_buffers(
        orig_func, model_part,
    ):
        cache_buffers, checkpoints = orig_func(model_part)
        # Need to replace checkpoints
        chunk_numbers = checkpoints[0].chunk_numbers
        checkpoints = [
            KVCacheBufferTestingCheckpoints(chunk_numbers=chunk_numbers)
            for _ in range(len(checkpoints))
        ]
        return cache_buffers, checkpoints

    accumulator._create_checkpoints_and_buffers = partial(
        wrapped_create_checkpoints_and_buffers,
        accumulator._create_checkpoints_and_buffers
    )


def available_backends(do_mps: bool = True) -> List[torch.device]:
    result = [torch.device("cpu")]
    if do_mps and RUN_TESTS_FOR_MPS and torch.backends.mps.is_available():
        result.append(torch.device("mps"))
    if torch.cuda.is_available():
        result.append(torch.device("cuda:0"))
    return result


def cache_name_is_bitsandbytes(name: str) -> bool:
    return name[:-1].endswith("bnb-quantized")


def cache_name_is_ao(name: str) -> bool:
    return name[:-1].endswith("ao-quantized")


def cache_name_gpu_only(name: str) -> bool:
    return cache_name_is_bitsandbytes(name) or cache_name_is_ao(name)


def device_for_cache_name(name: str) -> torch.device:
    if cache_name_gpu_only(name):
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def filter_cache_names(names: List[str]) -> List[str]:
    if torch.cuda.is_available():
        return names
    else:
        return [
            name for name in names if not cache_name_gpu_only(name)
        ]


def cache_names_and_devices(
    only_cpu: bool = False,
    filter_name: Callable[[str], bool] = None,
) -> List[Tuple[str, torch.device]]:
    if filter_name is None:
        filter_name = lambda name: True
    result = []
    for name, device in product(
        KVCacheFactory.supported_names(), available_backends(),
    ):
        if filter_name(name):
            is_cpu = device.type != "cuda"
            if (is_cpu and not cache_name_gpu_only(name)) or (not is_cpu and not only_cpu):
                result.append((name, device))
    return result


def product_with_devices(
    list_tuples: List[tuple],
    arg_names: str,
) -> Tuple[str, List[tuple]]:
    return "device, " + arg_names, [
        (a,) + b for a, b in product(
            available_backends(),
            list_tuples,
        )
    ]


def random_args_cache_forward(
    params: KVCacheParams,
    num: int,
    vocab_size: int,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    query = random_tensor(params, num=num, is_query=True, device=device)
    kv = random_keys_values(params, num=num, device=device)
    idx = torch.randint(
        low=0,
        high=vocab_size,
        size=(params.max_batch_size, num),
    )
    return {
        "query": query,
        "key": kv[0],
        "value": kv[1],
        "token_idx": idx,
    }


def range_from_args(
    data: Dict[str, torch.Tensor], start: int, end: int,
) -> Dict[str, torch.Tensor]:
    return {
        "query": data["query"][:, :, start:end, :],
        "key": data["key"][:, :, start:end, :],
        "value": data["value"][:, :, start:end, :],
        "token_idx": data["token_idx"][:, start:end],
    }


def debug_print_gradients(model: torch.nn.Module):
    rows = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                row = f"{name}: {param.grad.data.shape}"
            else:
                row = f"{name}: None"
            rows.append(row)
    print("\n".join(rows))
