from functools import partial
import math
import time
from typing import Optional, Callable, Tuple

import torch
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
)
import torch.nn.functional as F


FlexAttnWithBlockMask = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float, bool],
    torch.Tensor,
]


def is_index_1d(index: torch.Tensor) -> bool:
    ndim = index.ndim
    return index.stride() == (0,) * (ndim - 1) + (1,)


def index_to_3d(index: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    assert index.ndim == 1
    return index.view(1, 1, -1).expand(dim0, dim1, -1)


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
) -> FlexAttnWithBlockMask:
    block_mask = create_block_mask(
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
    """
    Transforms `token_positions` argument to
    :func:`scaled_dot_product_attention_flexatt` into tensor used in the
    kernel. If the argument is extended from 1D, we return the 1D slice.
    Otherwise, if `q_per_kv > 1`, we create an expanded copy of shape
    `(batch_size, n_head, kv_len)`.

    """
    if is_index_1d(token_positions):
        return token_positions[0, 0, :]
    else:
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


def causal_mask_for_chunk_3d(
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    input_pos: torch.Tensor,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    left_arg = q_idx + input_pos
    right_arg = token_positions[batch, head, kv_idx]
    return left_arg >= right_arg


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
    input_pos: torch.Tensor,
) -> torch.Tensor:
    left_arg = q_idx + input_pos
    return left_arg >= kv_idx


class AttnFunctionForChunk:
    def __init__(
        self,
        q_len: int,
        kv_len: int,
        batch_size: int,
        n_head: int,
        device: torch.device,
        tp_ndim: int,
    ):
        assert tp_ndim in (0, 1, 3)
        tp_is_3d = tp_ndim == 3
        tp_is_none = tp_ndim == 0
        kwargs = dict(device=device, dtype=torch.int32)
        self.n_head = n_head
        self.input_pos = torch.tensor(kv_len - q_len, **kwargs)
        if tp_is_none:
            self.token_positions = None
        else:
            self.token_positions = torch.zeros(
                (batch_size, n_head, kv_len) if tp_is_3d else (kv_len,),
                **kwargs,
            )
        if tp_is_none:
            mask_mod = partial(
                causal_mask_for_chunk_notp,
                input_pos=self.input_pos,
            )
        else:
            mask_mod = partial(
                causal_mask_for_chunk_3d if tp_is_3d else causal_mask_for_chunk_1d,
                input_pos=self.input_pos,
                token_positions=self.token_positions,
            )
        self.block_mask = create_block_mask(
            mask_mod,
            B=batch_size if tp_is_3d else None,
            H=n_head if tp_is_3d else None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )
        self.attn_fn_compiled = torch.compile(flex_attention)

    def __call__(
        self,
        input_pos: int,
        token_positions: Optional[torch.Tensor],
    ) -> FlexAttnWithBlockMask:
        if self.token_positions is not None:
            token_positions = transform_token_positions(token_positions, self.n_head)
            self.token_positions[:] = token_positions
            self.input_pos.copy_(input_pos)
        return partial(
            self.attn_fn_compiled,
            block_mask=self.block_mask,
        )


class FlexAttentionArgs:
    def __init__(
        self,
        q_len: int,
        kv_len: int,
        batch_size: int,
        n_head: int,
        device: torch.device,
        tp_ndim: int,
    ):
        self.attn_fn_for_prefill = attn_fn_for_prefill(kv_len, device)
        self.attn_fn_for_chunk = AttnFunctionForChunk(
            q_len,
            kv_len,
            batch_size,
            n_head,
            device,
            tp_ndim,
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


def random_index(
    start: int,
    end: int,
    num: int,
    batch_size: int,
    n_query_groups: int,
    device: torch.device,
):
    diff = end - start
    if diff < num:
        raise ValueError(f"end - start = {diff}, must be >= num = {num}")
    index_kwargs = dict(dtype=torch.int64, device=device)
    result = torch.empty((batch_size, n_query_groups, num), **index_kwargs)
    for b in range(batch_size):
        for h in range(n_query_groups):
            result[b, h, :] = (torch.randperm(diff, **index_kwargs) + start)[:num]
    return result


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


def mask_cache_bool(
    max_seq_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Usual causal mask:
    return torch.ones(
        max_seq_length,
        max_seq_length,
        device=device,
        dtype=dtype,
    ).triu(diagonal=1)


def build_mask_cache(
    max_seq_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = mask_cache_bool(max_seq_length, device, dtype)
    mask.masked_fill_(mask.bool(), minus_infinity(dtype))
    return mask


def mask_slice_bool(
    input_pos: int,
    num: int,
    token_positions: torch.Tensor,
    n_head: int,
) -> torch.Tensor:
    # Build boolean mask, then map False -> 0, True -> -infty
    # If (i, j) indexes the complete (seq_len, seq_len) mask matrix,
    # causality is given by I(i < j). If `sliding_window_size` is given,
    # this translates to I(i >= j + sws) if sws = sliding_window_size.
    assert token_positions.ndim == 3
    batch_size, n_query_groups, _ = token_positions.shape
    q_per_kv = n_head // n_query_groups
    assert n_head == n_query_groups * q_per_kv and q_per_kv >= 1
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
        )
    token_positions = token_positions.unsqueeze(2).expand(
        -1,
        -1,
        num,
        -1,
    )
    kwargs = dict(device=token_positions.device, dtype=token_positions.dtype)
    return (
        torch.arange(
            input_pos,
            input_pos + num,
            **kwargs,
        )
        .view(1, 1, -1, 1)
        .expand_as(token_positions)
        < token_positions
    )


def build_mask_slice(
    input_pos: int,
    num: int,
    token_positions: torch.Tensor,
    n_head: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    bool_mask = mask_slice_bool(
        input_pos,
        num,
        token_positions,
        n_head,
    )
    mask = torch.zeros(
        bool_mask.shape,
        dtype=dtype,
        device=token_positions.device,
    )
    mask.masked_fill_(bool_mask, minus_infinity(dtype))
    return mask


def eager_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: float,
    input_pos: int,
    token_positions: Optional[torch.Tensor],
) -> torch.Tensor:
    device = query.device
    dtype = torch.float32
    _, n_head, q_len, _ = query.shape
    _, n_query_groups, kv_len, _ = key.shape
    query32 = query.to(dtype)
    key32 = key.to(dtype)
    attn_weights = attention_compute_scores(query32, key32) * scale_factor
    # Attention masking
    if input_pos == 0:
        assert q_len == kv_len
        assert token_positions is None
        mask = build_mask_cache(kv_len, device, dtype)
        attn_weights = attn_weights + mask[None, None, :, :]
    elif token_positions is not None:
        mask = build_mask_slice(input_pos, q_len, token_positions, n_head, dtype)
        attn_weights = attn_weights + mask
    else:
        full_mask = build_mask_cache(kv_len, device, dtype)
        mask = full_mask[(-q_len):, :]
        attn_weights = attn_weights + mask[None, None, :, :]
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
):
    seed = 31415927
    torch.manual_seed(seed)
    dtype = torch.float32
    scale_factor = 1.0 / math.sqrt(head_size)
    num_repeats = 10

    for tp_ndim in (0, 1, 3):
        print(f"\n*** tp_ndim={tp_ndim}")
        flexatt_args = FlexAttentionArgs(
            q_len=chunk_size,
            kv_len=cache_length,
            batch_size=batch_size,
            n_head=n_head,
            device=device,
            tp_ndim=tp_ndim,
        )

        for _ in range(num_repeats):
            # Prefill
            attn_outputs = [[], []]
            print(f"Computing prefill MHA (cache_length={cache_length})")
            q_shape = (batch_size, n_head, cache_length, head_size)
            query = torch.randn(*q_shape, device=device, dtype=dtype)
            kv_shape = (batch_size, n_query_groups, cache_length, head_size)
            key = torch.randn(*kv_shape, device=device, dtype=dtype)
            value = torch.randn(*kv_shape, device=device, dtype=dtype)
            torch.cuda.current_stream().synchronize()
            timer_start = time.perf_counter()
            outputs = scaled_dot_product_attention_flexatt(
                flexatt_args=flexatt_args,
                query=query,
                key=key,
                value=value,
                scale_factor=scale_factor,
                input_pos=0,
                token_positions=None,
            )
            torch.cuda.current_stream().synchronize()
            timer_length = time.perf_counter() - timer_start
            print(f"time_flexatt: {timer_length:.5f}")
            timer_start = time.perf_counter()
            attn_outputs[0].append(outputs)
            gt_outputs = eager_scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                scale_factor=scale_factor,
                input_pos=0,
                token_positions=None,
            )
            torch.cuda.current_stream().synchronize()
            timer_length = time.perf_counter() - timer_start
            print(f"time_eager:   {timer_length:.5f}")
            attn_outputs[0].append(gt_outputs)

            # Process next chunk
            print(f"Computing chunk MHA (chunk_size={chunk_size})")
            q_shape = (batch_size, n_head, chunk_size, head_size)
            query = torch.randn(*q_shape, device=device, dtype=dtype)
            key = torch.randn(*kv_shape, device=device, dtype=dtype)
            value = torch.randn(*kv_shape, device=device, dtype=dtype)
            if tp_ndim == 0:
                token_positions = None
            else:
                start = chunk_size
                end = cache_length + chunk_size
                if tp_ndim == 1:
                    _ind = random_index(
                        start=start,
                        end=end,
                        num=cache_length,
                        batch_size=1,
                        n_query_groups=1,
                        device=device,
                    )
                    token_positions = index_to_3d(
                        _ind[0, 0, :],
                        batch_size,
                        n_query_groups,
                    )
                else:
                    token_positions = random_index(
                        start=start,
                        end=end,
                        num=cache_length,
                        batch_size=batch_size,
                        n_query_groups=n_query_groups,
                        device=device,
                    )
            torch.cuda.current_stream().synchronize()
            timer_start = time.perf_counter()
            outputs = scaled_dot_product_attention_flexatt(
                flexatt_args=flexatt_args,
                query=query,
                key=key,
                value=value,
                scale_factor=scale_factor,
                input_pos=cache_length,
                token_positions=token_positions,
            )
            torch.cuda.current_stream().synchronize()
            timer_length = time.perf_counter() - timer_start
            print(f"time_flexatt: {timer_length:.5f}")
            attn_outputs[1].append(outputs)
            timer_start = time.perf_counter()
            gt_outputs = eager_scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                scale_factor=scale_factor,
                input_pos=cache_length,
                token_positions=token_positions,
            )
            torch.cuda.current_stream().synchronize()
            timer_length = time.perf_counter() - timer_start
            print(f"time_eager:   {timer_length:.5f}")
            attn_outputs[1].append(gt_outputs)

            # Comparison
            for i, outputs in enumerate(attn_outputs):
                print(f"Comparing for chunk {i}")
                torch.testing.assert_close(outputs[0], outputs[1])


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
