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
from functools import partial
import math
from typing import Optional, Callable, Tuple

import torch
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
)


FlexAttnWithBlockMask = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float, bool],
    torch.Tensor,
]


def causal_mask_for_prefill(
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    return q_idx >= kv_idx


def attn_fn_for_prefill(
    kv_len: int,
    device: torch.device,
    _compile: bool = True,
) -> FlexAttnWithBlockMask:
    if _compile:
        _create_block_mask = torch.compile(create_block_mask)
    else:
        _create_block_mask = create_block_mask
    block_mask = _create_block_mask(
        causal_mask_for_prefill,
        B=None,
        H=None,
        Q_LEN=kv_len,
        KV_LEN=kv_len,
        device=device,
    )
    attn_fn_compiled = torch.compile(flex_attention)
    return partial(attn_fn_compiled, block_mask=block_mask)


def transform_token_positions(
    token_positions: torch.Tensor,
    n_head: int,
) -> torch.Tensor:
    batch_size, n_query_groups, _ = token_positions.shape
    q_per_kv = n_head // n_query_groups
    if q_per_kv > 1:
        token_positions = (
            token_positions.unsqueeze(2)
            .expand(
                -1,
                -1,
                q_per_kv,
                -1,
            )
            .reshape(batch_size, n_head, -1)
            .contiguous()
        )
    return token_positions


def causal_mask_for_chunk(
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    input_pos: torch.Tensor,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    batch = batch.flatten()
    head = head.flatten()
    _q_idx = q_idx.flatten()
    kv_idx = kv_idx.flatten()
    # result = left_arg >= right_arg, where
    # left_arg = A[batch, head, q_idx, kv_idx],
    # right_arg = B[batch, head, q_idx, kv_idx]
    left_arg = _q_idx + input_pos
    right_arg = token_positions[batch, head, kv_idx]
    result = left_arg >= right_arg
    return result.view_as(q_idx)


def causal_mask_for_chunk_1d(
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    input_pos: torch.Tensor,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    _q_idx = q_idx.flatten()
    kv_idx = kv_idx.flatten()
    # result = left_arg >= right_arg, where
    # left_arg = A[batch, head, q_idx, kv_idx],
    # right_arg = B[batch, head, q_idx, kv_idx]
    left_arg = _q_idx + input_pos
    right_arg = token_positions[kv_idx]
    result = left_arg >= right_arg
    return result.view_as(q_idx)


class AttnFunctionForChunk:
    def __init__(
        self,
        q_len: int,
        kv_len: int,
        batch_size: int,
        n_head: int,
        device: torch.device,
        _compile: bool = True,
    ):
        kwargs = dict(device=device, dtype=torch.int32)
        self.input_pos = torch.zeros((1,), **kwargs)
        self.token_positions = torch.zeros((batch_size, n_head, kv_len), **kwargs)
        if _compile:
            _create_block_mask = torch.compile(
                create_block_mask,
            )
        else:
            _create_block_mask = create_block_mask
        self.block_mask = _create_block_mask(
            partial(
                causal_mask_for_chunk,
                input_pos=self.input_pos,
                token_positions=self.token_positions,
            ),
            B=batch_size,
            H=n_head,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )
        self.attn_fn_compiled = torch.compile(flex_attention)

    def __call__(
        self,
        input_pos: int,
        token_positions: torch.Tensor,
    ) -> FlexAttnWithBlockMask:
        n_head = self.token_positions.shape[1]
        token_positions = transform_token_positions(token_positions, n_head)
        assert token_positions.shape == self.token_positions.shape
        assert input_pos > 0
        self.input_pos[0] = input_pos
        self.token_positions[:] = token_positions
        return partial(self.attn_fn_compiled, block_mask=self.block_mask)


class AttnFunctionForChunk1D:
    def __init__(
        self,
        q_len: int,
        kv_len: int,
        device: torch.device,
        _compile: bool = True,
    ):
        kwargs = dict(device=device, dtype=torch.int32)
        self.input_pos = torch.zeros((1,), **kwargs)
        self.token_positions = torch.zeros((kv_len,), **kwargs)
        if _compile:
            _create_block_mask = torch.compile(create_block_mask)
        else:
            _create_block_mask = create_block_mask
        self.block_mask = _create_block_mask(
            partial(
                causal_mask_for_chunk_1d,
                input_pos=self.input_pos,
                token_positions=self.token_positions,
            ),
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )
        self.attn_fn_compiled = torch.compile(flex_attention)

    def __call__(
        self,
        input_pos: int,
        token_positions: torch.Tensor,
    ) -> FlexAttnWithBlockMask:
        assert token_positions.shape == self.token_positions.shape, (
            token_positions.shape,
            self.token_positions.shape,
        )
        assert input_pos > 0
        self.input_pos[0] = input_pos
        self.token_positions[:] = token_positions
        return partial(self.attn_fn_compiled, block_mask=self.block_mask)


def attn_fn_for_chunk_direct(
    q_len: int,
    kv_len: int,
    batch_size: int,
    n_head: int,
    device: torch.device,
    input_pos: int,
    token_positions: torch.Tensor,
    _compile: bool = True,
) -> FlexAttnWithBlockMask:
    token_positions = transform_token_positions(token_positions, n_head)
    kwargs = dict(device=device, dtype=torch.int32)
    input_pos_capt = torch.tensor([input_pos], **kwargs)
    token_positions_capt = token_positions.clone()
    if _compile:
        _create_block_mask = torch.compile(create_block_mask)
    else:
        _create_block_mask = create_block_mask
    block_mask = _create_block_mask(
        partial(
            causal_mask_for_chunk,
            input_pos=input_pos_capt,
            token_positions=token_positions_capt,
        ),
        B=batch_size,
        H=n_head,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=device,
    )
    attn_fn_compiled = torch.compile(flex_attention)
    return partial(attn_fn_compiled, block_mask=block_mask)


class FlexAttentionArgs:
    def __init__(
        self,
        q_len: int,
        kv_len: int,
        batch_size: int,
        n_head: int,
        device: torch.device,
        _compile: bool = True,
        is_1d: bool = False,
    ):
        self.attn_fn_for_prefill = attn_fn_for_prefill(kv_len, device, _compile)
        if not is_1d:
            self.attn_fn_for_chunk = AttnFunctionForChunk(
                q_len,
                kv_len,
                batch_size,
                n_head,
                device,
                _compile,
            )
        else:
            self.attn_fn_for_chunk = AttnFunctionForChunk1D(
                q_len,
                kv_len,
                device,
                _compile,
            )

    def attn_fn(
        self,
        input_pos: int,
        token_positions: Optional[torch.Tensor],
    ) -> FlexAttnWithBlockMask:
        if input_pos == 0:
            return self.attn_fn_for_prefill
        else:
            return self.attn_fn_for_chunk(
                input_pos=input_pos,
                token_positions=token_positions,
            )


def sdpa_check_args(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> Tuple[int, int, int, int, int, int]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, value must be 4D tensors")
    if key.shape != value.shape:
        raise ValueError("key, value must have same shape")
    batch_size, n_head, q_len, head_size = query.shape
    if key.shape[0] != batch_size or key.shape[-1] != head_size:
        raise ValueError(
            f"key.shape = {key.shape}, must be ({batch_size}, _, _, {head_size})"
        )
    _, n_query_groups, kv_len, _ = key.shape
    if not (0 < q_len <= kv_len):
        raise ValueError(
            f"Must have 0 < q_len = {q_len} <= kv_len = {kv_len}. Don't use this for prefill"
        )
    if n_query_groups <= 0 or n_head % n_query_groups != 0 or n_head < n_query_groups:
        raise ValueError(
            f"n_head = {n_head}, n_query_groups = {n_query_groups}: n_head must be positive multiple of n_query_groups"
        )
    return batch_size, n_head, n_query_groups, q_len, kv_len, head_size


def scaled_dot_product_attention_flexatt(
    flexatt_args: FlexAttentionArgs,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: float,
    input_pos: int,
    token_positions: Optional[torch.Tensor],
) -> torch.Tensor:
    n_query_groups = key.shape[1]
    n_head = query.shape[1]
    attn_fn = flexatt_args.attn_fn(
        input_pos=input_pos,
        token_positions=token_positions,
    )
    return attn_fn(
        query=query,
        key=key,
        value=value,
        scale=scale_factor,
        enable_gqa=n_query_groups < n_head,
    )


def scaled_dot_product_attention_flexatt_direct(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: float,
    input_pos: int,
    token_positions: Optional[torch.Tensor],
    _compile: bool = True,
) -> torch.Tensor:
    batch_size, n_head, q_len, _ = query.shape
    _, n_query_groups, kv_len, _ = key.shape
    if input_pos == 0:
        attn_fn = attn_fn_for_prefill(kv_len, device, _compile)
    else:
        attn_fn = attn_fn_for_chunk_direct(
            q_len,
            kv_len,
            batch_size,
            n_head,
            device,
            input_pos,
            token_positions,
            _compile,
        )
    return attn_fn(
        query=query,
        key=key,
        value=value,
        scale=scale_factor,
        enable_gqa=n_query_groups < n_head,
    )


def random_index(
    batch_size: int,
    n_query_groups: int,
    cache_length: int,
    device: torch.device,
):
    index_kwargs = dict(dtype=torch.int64, device=device)
    result = torch.empty(
        (batch_size, n_query_groups, cache_length),
        **index_kwargs,
    )
    for b in range(batch_size):
        for h in range(n_query_groups):
            result[b, h, :] = torch.randperm(cache_length, **index_kwargs)
    return result


def main(
    cache_length: int,
    chunk_size: int,
    batch_size: int,
    n_head: int,
    n_query_groups: int,
    head_size: int,
    device: torch.device,
):
    seed = 31415927
    torch.manual_seed(seed)
    dtype = torch.float16
    scale_factor = 1.0 / math.sqrt(head_size)
    _compile = False
    do_direct = False
    do_1d = True

    if not do_direct:
        flexatt_args = FlexAttentionArgs(
            q_len=chunk_size,
            kv_len=cache_length,
            batch_size=batch_size,
            n_head=n_head,
            device=device,
            _compile=_compile,
            is_1d=do_1d,
        )
    else:
        flexatt_args = None

    # Prefill
    print(f"Computing prefill MHA (cache_length={cache_length})")
    q_shape = (batch_size, n_head, cache_length, head_size)
    query = torch.randn(*q_shape, device=device, dtype=dtype)
    kv_shape = (batch_size, n_query_groups, cache_length, head_size)
    key = torch.randn(*kv_shape, device=device, dtype=dtype)
    value = torch.randn(*kv_shape, device=device, dtype=dtype)
    if not do_direct:
        attn_outputs_prefill = scaled_dot_product_attention_flexatt(
            flexatt_args=flexatt_args,
            query=query,
            key=key,
            value=value,
            scale_factor=scale_factor,
            input_pos=0,
            token_positions=None,
        )
    else:
        attn_outputs_prefill = scaled_dot_product_attention_flexatt_direct(
            query=query,
            key=key,
            value=value,
            scale_factor=scale_factor,
            input_pos=0,
            token_positions=None,
            _compile=_compile,
        )
    print(attn_outputs_prefill[-1].sum().item())

    # Process next chunk
    print(f"Computing chunk MHA (chunk_size={chunk_size})")
    q_shape = (batch_size, n_head, chunk_size, head_size)
    query = torch.randn(*q_shape, device=device, dtype=dtype)
    key = torch.randn(*kv_shape, device=device, dtype=dtype)
    value = torch.randn(*kv_shape, device=device, dtype=dtype)
    if not do_1d:
        token_positions = random_index(
            batch_size=batch_size,
            n_query_groups=n_query_groups,
            cache_length=cache_length,
            device=device,
        )
    else:
        token_positions = random_index(
            batch_size=1,
            n_query_groups=1,
            cache_length=cache_length,
            device=device,
        ).flatten()
    if not do_direct:
        attn_outputs_chunk = scaled_dot_product_attention_flexatt(
            flexatt_args=flexatt_args,
            query=query,
            key=key,
            value=value,
            scale_factor=scale_factor,
            input_pos=cache_length,
            token_positions=token_positions,
        )
    else:
        attn_outputs_chunk = scaled_dot_product_attention_flexatt_direct(
            query=query,
            key=key,
            value=value,
            scale_factor=scale_factor,
            input_pos=cache_length,
            token_positions=token_positions,
            _compile=_compile,
        )
    print(attn_outputs_chunk[-1].sum().item())


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise AssertionError("CUDA not available")
    batch_size = 2
    n_head = 32
    n_query_groups = 8
    cache_length = 4096
    head_dim = 128
    chunk_size = 1024
    device = torch.device("cuda", 0)
    main(
        cache_length=cache_length,
        chunk_size=chunk_size,
        batch_size=batch_size,
        n_head=n_head,
        n_query_groups=n_query_groups,
        head_size=head_dim,
        device=device,
    )
