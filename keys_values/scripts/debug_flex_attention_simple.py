from functools import partial
import math
from typing import Optional, Callable

import torch
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
)
import torch.nn.functional as F


def causal_mask_for_chunk_1d(
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    input_pos: torch.Tensor,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    left_arg = q_idx + input_pos
    right_arg = token_positions[kv_idx]
    return left_arg >= right_arg


def causal_mask_for_chunk_notp(
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    offset: torch.Tensor,
) -> torch.Tensor:
    left_arg = q_idx + offset
    return left_arg >= kv_idx


class AttnFunctionForChunk:
    def __init__(
        self,
        q_len: int,
        kv_len: int,
        device: torch.device,
        use_tp: bool,
    ):
        kwargs = dict(device=device, dtype=torch.int32)
        self.input_pos = torch.tensor(kv_len - q_len, **kwargs)
        if use_tp:
            self.input_pos = torch.tensor(0, **kwargs)
            self.token_positions = torch.zeros((kv_len,), **kwargs)
            mask_mod = partial(
                causal_mask_for_chunk_1d,
                input_pos=self.input_pos,
                token_positions=self.token_positions,
            )
        else:
            self.input_pos = None
            self.token_positions = None
            mask_mod = partial(
                causal_mask_for_chunk_notp,
                offset=kv_len - q_len,
            )
        self.block_mask = create_block_mask(
            mask_mod,
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
        token_positions: Optional[torch.Tensor],
    ) -> Callable:
        if self.input_pos is not None:
            self.input_pos.copy_(input_pos)
            self.token_positions[:] = token_positions
        return partial(
            self.attn_fn_compiled,
            block_mask=self.block_mask,
        )


def scaled_dot_product_attention_flexatt(
    flexatt_args: AttnFunctionForChunk,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: float,
    input_pos: int,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    n_head = query.shape[1]
    n_query_groups = key.shape[1]
    attn_fn = flexatt_args(
        input_pos=input_pos,
        token_positions=token_positions,
    )
    if flexatt_args.input_pos is None:
        # Sort to obtain the right attention mask
        sort_index = torch.argsort(token_positions)
        key = key[:, :, sort_index, :]
        value = value[:, :, sort_index, :]
    return attn_fn(
        query=query,
        key=key,
        value=value,
        scale=scale_factor,
        enable_gqa=n_query_groups < n_head,
    )


def attention_compute_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Compute inner product scores (without masking). Here,
    `nh_q = q_per_kv * nh_k` with `q_per_kv >= 1`.

    Args:
        query: Query tensor, `(bs, nh_q, q_len, hs)`
        key: Key tensor, `(bs, nh_k, kv_len, hs)`
        out: Result written here, if given
        scale_factor: Scale factor for inner product scores

    Returns:
        Inner product scores, `(bs, nh_q, q_len, kv_len)`. This is `out` if given

    """
    assert query.ndim == key.ndim == 4
    assert query.shape[0] == key.shape[0] and query.shape[3] == key.shape[3]
    nh_q = query.shape[1]
    nh_k = key.shape[1]
    assert nh_q % nh_k == 0
    # - query, arg1: (bs, nh_q, q_len, hs)
    # - key: (bs, nh_k, kv_len, hs)
    # - key_transposed: (bs, nh_k, hs, kv_len)
    q_per_kv = nh_q // nh_k
    if scale_factor == 1.0:
        key_transposed = key.mT
        arg1 = query
    elif query.numel() <= key.numel():
        key_transposed = key.mT
        arg1 = query * scale_factor
    else:
        key_transposed = key.mT * scale_factor
        arg1 = query
    if q_per_kv == 1:
        out = torch.matmul(arg1, key_transposed, out=out)
    else:
        assert q_per_kv > 1
        q_shape = query.shape[:1] + (nh_k, q_per_kv) + query.shape[2:]
        _query = arg1.view(*q_shape)
        key_transposed = key_transposed.unsqueeze(2)
        # At this point:
        # - _query: (bs, nh_k, q_per_kv, q_len, hs)
        # - key_transposed: (bs, nh_k, 1, hs, kv_len)
        # - scores: (bs, nh_k, q_per_kv, q_len, kv_len)
        if out is not None:
            out = out.view(_query.shape[:-1] + (key.shape[2],))
        out = torch.matmul(_query, key_transposed, out=out)
        s_shape = query.shape[:-1] + (key.shape[2],)
        out = out.view(*s_shape)
    return out


def attention_compute_weighted_values(
    scores: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """
     Args:
        scores: Attention weights, `(bs, nh_q, q_len, kv_len)`
        value: Value tensor, `(bs, nh_k, kv_len, hs)`

    Returns:
        Attention outputs, `(bs, nh_q, q_len, hs)`

    """
    assert scores.ndim == value.ndim == 4
    assert scores.shape[0] == scores.shape[0] and scores.shape[3] == value.shape[2]
    nh_q = scores.shape[1]
    nh_k = value.shape[1]
    assert nh_q % nh_k == 0
    # - scores: (bs, nh_q, q_len, kv_len)
    # - value: (bs, nh_k, kv_len, hs)
    q_per_kv = nh_q // nh_k
    if q_per_kv == 1:
        return torch.matmul(scores, value)
    else:
        s_shape = scores.shape[:1] + (nh_k, q_per_kv) + scores.shape[2:]
        _scores = scores.view(*s_shape)
        _value = value.unsqueeze(2)
        # At this point:
        # - _scores: (bs, nh_k, q_per_kv, q_len, kv_len)
        # - _value: (bs, nh_k, 1, kv_len, hs)
        # - result: (bs, nh_k, q_per_kv, q_len, hs)
        result = torch.matmul(_scores, _value)
        r_shape = scores.shape[:-1] + (value.shape[-1],)
        return result.view(*r_shape)


def minus_infinity(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min


def mask_slice_bool(
    input_pos: int,
    num: int,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    assert token_positions.ndim == 1
    kwargs = dict(device=token_positions.device, dtype=token_positions.dtype)
    return (
        torch.arange(
            input_pos,
            input_pos + num,
            **kwargs,
        ).view(-1, 1)
        < token_positions.view(1, -1)
    )


def build_mask_slice(
    input_pos: int,
    num: int,
    token_positions: torch.Tensor,
    batch_size: int,
    n_head: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    bool_mask = mask_slice_bool(
        input_pos,
        num,
        token_positions,
    )
    mask = torch.zeros(
        bool_mask.shape,
        dtype=dtype,
        device=token_positions.device,
    )
    mask.masked_fill_(bool_mask, minus_infinity(dtype))
    return mask[None, None, :, :].expand(batch_size, n_head, -1, -1)


def eager_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: float,
    input_pos: int,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    assert input_pos > 0
    dtype = torch.float32
    batch_size, n_head, q_len, _ = query.shape
    _, n_query_groups, kv_len, _ = key.shape
    assert token_positions.shape == (kv_len,)
    query32 = query.to(dtype)
    key32 = key.to(dtype)
    attn_weights = attention_compute_scores(query32, key32) * scale_factor
    # Attention masking
    mask = build_mask_slice(
        input_pos, q_len, token_positions, batch_size, n_head, dtype,
    )
    attn_weights = attn_weights + mask
    attn_weights = F.softmax(attn_weights, dim=-1)
    del query32, key32
    value32 = value.to(torch.float32)
    return attention_compute_weighted_values(attn_weights, value32).to(query.dtype)


def main(
    cache_length: int,
    chunk_size: int,
    batch_size: int,
    n_head: int,
    n_query_groups: int,
    head_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    seed = 31415927
    torch.manual_seed(seed)
    scale_factor = 1.0 / math.sqrt(head_size)

    attn_outputs = []
    q_shape = (batch_size, n_head, chunk_size, head_size)
    query = torch.randn(*q_shape, device=device, dtype=dtype)
    kv_shape = (batch_size, n_query_groups, cache_length, head_size)
    key = torch.randn(*kv_shape, device=device, dtype=dtype)
    value = torch.randn(*kv_shape, device=device, dtype=dtype)
    input_pos = cache_length
    start = input_pos + chunk_size - cache_length
    token_positions = torch.randperm(cache_length, device=device, dtype=torch.int64) + start
    # Eager attention
    attn_outputs.append(
        eager_scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            scale_factor=scale_factor,
            input_pos=cache_length,
            token_positions=token_positions,
        )
    )
    # FlexAttention (use_tp=False)
    flexatt_args = AttnFunctionForChunk(
        q_len=chunk_size,
        kv_len=cache_length,
        device=device,
        use_tp=False,
    )
    attn_outputs.append(
        scaled_dot_product_attention_flexatt(
            flexatt_args=flexatt_args,
            query=query,
            key=key,
            value=value,
            scale_factor=scale_factor,
            input_pos=cache_length,
            token_positions=token_positions,
        )
    )
    # FlexAttention (use_tp=True)
    flexatt_args = AttnFunctionForChunk(
        q_len=chunk_size,
        kv_len=cache_length,
        device=device,
        use_tp=True,
    )
    attn_outputs.append(
        scaled_dot_product_attention_flexatt(
            flexatt_args=flexatt_args,
            query=query,
            key=key,
            value=value,
            scale_factor=scale_factor,
            input_pos=cache_length,
            token_positions=token_positions,
        )
    )
    # Comparison
    print("Compare eager vs flex_attention (use_tp=False)")
    torch.testing.assert_close(attn_outputs[0], attn_outputs[1])
    print("Compare eager vs flex_attention (use_tp=True)")
    torch.testing.assert_close(attn_outputs[0], attn_outputs[2])


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise AssertionError("CUDA not available")
    batch_size = 2
    n_head = 32
    n_query_groups = 32
    cache_length = 4096
    head_dim = 128
    chunk_size = 512
    device = torch.device("cuda", 0)
    dtype = torch.float32
    main(
        cache_length=cache_length,
        chunk_size=chunk_size,
        batch_size=batch_size,
        n_head=n_head,
        n_query_groups=n_query_groups,
        head_size=head_dim,
        device=device,
        dtype=dtype,
    )
