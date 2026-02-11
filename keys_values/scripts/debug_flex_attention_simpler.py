from functools import partial
import math
from typing import Callable, Tuple

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


def causal_mask_for_chunk(
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    input_pos: torch.Tensor,
) -> torch.Tensor:
    left_arg = q_idx + input_pos
    return left_arg >= kv_idx


class AttnFunctionForChunk:
    def __init__(
        self,
        q_len: int,
        kv_len: int,
        device: torch.device,
        _compile: bool = True,
    ):
        kwargs = dict(device=device, dtype=torch.int32)
        self.input_pos = torch.tensor(0, **kwargs)
        if _compile:
            _create_block_mask = torch.compile(create_block_mask)
        else:
            _create_block_mask = create_block_mask
        self.block_mask = _create_block_mask(
            partial(
                causal_mask_for_chunk,
                input_pos=self.input_pos,
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
    ) -> FlexAttnWithBlockMask:
        self.input_pos.copy_(input_pos)
        return partial(self.attn_fn_compiled, block_mask=self.block_mask)


def attn_fn_for_chunk_direct(
    q_len: int,
    kv_len: int,
    device: torch.device,
    input_pos: int,
    _compile: bool = True,
) -> FlexAttnWithBlockMask:
    kwargs = dict(device=device, dtype=torch.int32)
    input_pos_capt = torch.tensor(input_pos, **kwargs)
    if _compile:
        _create_block_mask = torch.compile(create_block_mask)
    else:
        _create_block_mask = create_block_mask
    block_mask = _create_block_mask(
        partial(
            causal_mask_for_chunk,
            input_pos=input_pos_capt,
        ),
        B=None,
        H=None,
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
        device: torch.device,
        _compile: bool = True,
    ):
        self.attn_fn_for_prefill = attn_fn_for_prefill(kv_len, device, _compile)
        self.attn_fn_for_chunk = AttnFunctionForChunk(
            q_len,
            kv_len,
            device,
            _compile,
        )

    def attn_fn(
        self,
        input_pos: int,
    ) -> FlexAttnWithBlockMask:
        if input_pos == 0:
            return self.attn_fn_for_prefill
        else:
            return self.attn_fn_for_chunk(input_pos=input_pos)


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
) -> torch.Tensor:
    n_query_groups = key.shape[1]
    n_head = query.shape[1]
    attn_fn = flexatt_args.attn_fn(input_pos=input_pos)
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
            device,
            input_pos,
            _compile,
        )
    return attn_fn(
        query=query,
        key=key,
        value=value,
        scale=scale_factor,
        enable_gqa=n_query_groups < n_head,
    )


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

    if not do_direct:
        flexatt_args = FlexAttentionArgs(
            q_len=chunk_size,
            kv_len=cache_length,
            device=device,
            _compile=_compile,
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
        )
    else:
        attn_outputs_prefill = scaled_dot_product_attention_flexatt_direct(
            query=query,
            key=key,
            value=value,
            scale_factor=scale_factor,
            input_pos=0,
            _compile=_compile,
        )
    print(attn_outputs_prefill[-1].sum().item())

    # Process next chunk
    print(f"Computing chunk MHA (chunk_size={chunk_size})")
    q_shape = (batch_size, n_head, chunk_size, head_size)
    query = torch.randn(*q_shape, device=device, dtype=dtype)
    key = torch.randn(*kv_shape, device=device, dtype=dtype)
    value = torch.randn(*kv_shape, device=device, dtype=dtype)
    if not do_direct:
        attn_outputs_chunk = scaled_dot_product_attention_flexatt(
            flexatt_args=flexatt_args,
            query=query,
            key=key,
            value=value,
            scale_factor=scale_factor,
            input_pos=cache_length,
        )
    else:
        attn_outputs_chunk = scaled_dot_product_attention_flexatt_direct(
            query=query,
            key=key,
            value=value,
            scale_factor=scale_factor,
            input_pos=cache_length,
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
