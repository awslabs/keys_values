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
from typing import List, Optional, Tuple, Union
import math

import torch
from torch.backends.cuda import (
    can_use_cudnn_attention,
    can_use_efficient_attention,
    can_use_flash_attention,
)
from torch.nn import functional as F
from torch.nn.attention import SDPAParams, SDPBackend, sdpa_kernel

from keys_values.utils import repeat_interleave, index_to_3d


def filter_sdpa_kernels(
    sdpa_kernels: List[SDPBackend],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_p: float,
    is_causal: bool,
    enable_gqa: bool,
    **kwargs,
) -> List[SDPBackend]:
    params = SDPAParams(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa)
    new_kernels = []
    for kernel in sdpa_kernels:
        if kernel == SDPBackend.FLASH_ATTENTION and not can_use_flash_attention(params):
            continue
        elif (
            kernel == SDPBackend.EFFICIENT_ATTENTION
            and not can_use_efficient_attention(params)
        ):
            continue
        elif kernel == SDPBackend.CUDNN_ATTENTION and not can_use_cudnn_attention(
            params
        ):
            continue
        new_kernels.append(kernel)
    return new_kernels


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
    sliding_window_size: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Usual causal mask:
    mask = torch.ones(
        max_seq_length,
        max_seq_length,
        device=device,
        dtype=dtype,
    ).triu(diagonal=1)
    if sliding_window_size is not None:
        mask += torch.ones_like(mask).tril(diagonal=-sliding_window_size)
    return mask


def build_mask_cache(
    max_seq_length: int,
    sliding_window_size: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
          Global Window              Sliding window             Sliding window
          attention mask      +            bias          =      attention mask
    ┌────────────────────────┐  ┌───────────────────────┐  ┌─────────────────────────┐
    │ True False False False │  │ True  True  True True │  │ True  False False False │
    │ True True  False False │  │ True  True  True True │  │ True  True  False False │
    │ True True  True  False │  │ False True  True True │  │ False True  True  False │
    │ True True  True  True  │  │ False False True True │  │ False False True  True  │
    └────────────────────────┘  └───────────────────────┘  └─────────────────────────┘
    """
    mask = mask_cache_bool(max_seq_length, sliding_window_size, device, dtype)
    mask.masked_fill_(mask.bool(), minus_infinity(dtype))
    return mask


def mask_slice_bool(
    input_pos: int,
    num: int,
    token_positions: torch.Tensor,
    n_head: int,
    sliding_window_size: Optional[int] = None,
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
    bool_mask = (
        torch.arange(
            input_pos,
            input_pos + num,
            **kwargs,
        )
        .view(1, 1, -1, 1)
        .expand_as(token_positions)
        < token_positions
    )
    if sliding_window_size is not None:
        extra_mask = (
            torch.arange(
                input_pos - sliding_window_size,
                input_pos + num - sliding_window_size,
                **kwargs,
            )
            .view(1, 1, -1, 1)
            .expand_as(token_positions)
            >= token_positions
        )
        bool_mask |= extra_mask
    return bool_mask


def build_mask_slice(
    input_pos: int,
    num: int,
    token_positions: torch.Tensor,
    n_head: int,
    dtype: torch.dtype,
    sliding_window_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Returns mask for case `input_pos > 0` in :class:`MultiHeadSelfAttention`.

    Args:
        input_pos: Position in input sequence, must be positive
        num: Length of query argument `q_len`
        token_positions: Token positions in KV cache, shape
            `(batch_size, n_query_groups, cache_length)`
        n_head: Number of attention heads, must be multiple of
            `n_query_groups`
        dtype: Data type of the output mask
        sliding_window_size: Size of sliding window (if any)

    Returns:
        Mask tensor, shape `(batch_size, n_head, num, cache_length)`

    """
    bool_mask = mask_slice_bool(
        input_pos,
        num,
        token_positions,
        n_head,
        sliding_window_size,
    )
    mask = torch.zeros(
        bool_mask.shape,
        dtype=dtype,
        device=token_positions.device,
    )
    mask.masked_fill_(bool_mask, minus_infinity(dtype))
    return mask


# Maximum number of `float32` entries for `tmp_array` for GB
ENTRIES_PER_GB = 2**28

# Maximum size of `tmp_array` in GB
DEFAULT_TMP_ARRAY_LIMIT_GB = 3


def max_num_entries_from_max_gb(max_gb: float) -> int:
    assert max_gb > 0
    return int(max_gb * ENTRIES_PER_GB)


def create_temp_array(
    batch_size: int,
    n_head: int,
    q_len: int,
    kv_len: int,
    device: torch.device,
    tmp_array_limit_gb: Optional[float] = None,
) -> Tuple[torch.Tensor, int, int]:
    """
    Creates a temporary array of shape `(batch_size, n_head, tmp_len, kv_len)`.
    Here, `tmp_len` is such that the number of entries is
    `<= max_num_entries_from_max_gb(temp_entry_limit_gb)`, and
    `tmp_len * num_splits >= q_len`.

    Args:
        batch_size: Batch size
        n_head: Number of attention heads
        q_len: Length of query sequence
        kv_len: Length of key, value sequence
        device: Device for arrays
        tmp_array_limit_gb: See above, defaults to
            :const:`DEFAULT_TMP_ARRAY_LIMIT_GB`

    Returns:
        `tmp_array, num_splits, tmp_len`

    """
    if tmp_array_limit_gb is None:
        tmp_array_limit_gb = DEFAULT_TMP_ARRAY_LIMIT_GB
    tmp_array_max_num_entries = max_num_entries_from_max_gb(tmp_array_limit_gb)
    factor = batch_size * n_head * kv_len
    if factor * q_len <= tmp_array_max_num_entries:
        tmp_len = q_len
        num_splits = 1
    else:
        tmp_len = tmp_array_max_num_entries // factor
        if tmp_len < 1:
            raise ValueError(
                f"batch_size={batch_size}, n_head={n_head}, kv_len={kv_len} too large. Their product must be <= {tmp_array_max_num_entries}"
            )
        num_splits = int(math.ceil(q_len / tmp_len))
    shape = (batch_size, n_head, tmp_len, kv_len)
    kwargs = dict(device=device, dtype=torch.float32)
    tmp_array = torch.empty(shape, **kwargs)
    return tmp_array, num_splits, tmp_len


def slice_as_flat(tmp_array: torch.Tensor, sz: int) -> torch.Tensor:
    assert tmp_array.ndim == 4
    d0, d1, d2, d3 = tmp_array.shape
    if sz == d2:
        return tmp_array
    assert 0 < sz <= d2
    numel = d0 * d1 * sz * d3
    return tmp_array.view(-1)[:numel].view(d0, d1, sz, d3)


# Note: This is quite a bit slower than `F.softmax`, so we rather spend the
# extra memory
def softmax_in_place(x: torch.Tensor, dim: int):
    max_x = x.max(dim=dim, keepdim=True)[0]
    x.sub_(max_x)
    x.exp_()
    norm_x = x.sum(dim=dim, keepdim=True)
    x.div_(norm_x)


def sdpa_attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    tmp_array: torch.Tensor,
    token_positions: Optional[torch.Tensor],
    input_pos: int,
    scale_factor: float,
    sliding_window_size: Optional[int],
) -> torch.Tensor:
    """
    Attention weights for `query`, `key` are returned, using `tmp_array1` for
    temporary storage. Together with :func:`create_temp_arrays`, supports SDPA
    computation in blocks.

    Args:
        query: Query tensor, `(bs, nh_q, tmp_len, hs)`
        key: Key tensor, `(bs, nh_k, kv_len, hs)`
        tmp_array: Temp array, `(bs, nh_q, tmp_len, kv_len)`
        token_positions: Token positions for cache content,
            `(bs, nh_k, kv_len)`. If not given, this is `range(kv_len)`
        input_pos: New tokens have positions `input_pos:(input_pos + tmp_len)`
        scale_factor: Scale factor for SDPA
        sliding_window_size: Affects attention masking

    Returns:
        Attention weights, `(bs, nh_q, tmp_len, kv_len)`

    """
    # We use a buffer `tmp_array` of shape
    # `(batch_size, n_head, q_len, kv_len)`, which is in general the
    # largest size. We make sure that no extra copies of this size
    # are created.
    batch_size, n_head, q_len, head_size = query.shape
    _, n_query_groups, kv_len, _ = key.shape
    # Compute attention weights f(S)
    attention_compute_scores(
        query=query,
        key=key,
        out=tmp_array,
        scale_factor=scale_factor,
    )
    # Attention masking
    if token_positions is None:
        _token_positions = index_to_3d(
            torch.arange(kv_len, device=query.device, dtype=torch.int64),
            batch_size,
            n_query_groups,
        )
    else:
        _token_positions = token_positions
    bool_mask = mask_slice_bool(
        input_pos=input_pos,
        num=q_len,
        token_positions=_token_positions,
        n_head=n_head,
        sliding_window_size=sliding_window_size,
    )
    assert bool_mask.shape == tmp_array.shape, (bool_mask.shape, tmp_array.shape)
    tmp_array[bool_mask] = minus_infinity(dtype=tmp_array.dtype)  # S
    # This creates an extra copy, but is much faster than `softmax_in_place`
    return tmp_array.softmax(dim=-1)  # f(S)


def sample_token_positions(
    batch_size: int,
    n_query_groups: int,
    q_len: int,
    kv_len: int,
    input_pos: int,
    device: torch.device,
) -> torch.Tensor:
    index_kwargs = dict(dtype=torch.int64, device=device)
    token_positions = torch.zeros(
        (batch_size, n_query_groups, kv_len),
        **index_kwargs,
    )
    for bs in range(batch_size):
        for nq in range(n_query_groups):
            token_positions[bs, nq, :] = torch.randperm(
                input_pos,
                **index_kwargs,
            )[:kv_len]
            # Ensure that `input_pos:(input_pos + q_len)` is present
            index = torch.randperm(kv_len, **index_kwargs)[:q_len]
            token_positions[bs, nq, index] = torch.arange(
                input_pos,
                input_pos + q_len,
                **index_kwargs,
            )
    return token_positions


def pytorch_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: float,
    sdpa_kernels: Union[SDPBackend, List[SDPBackend]],
    do_filter_kernels: bool = False,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[List[SDPBackend]]]:
    """
    If you call this repeatedly and want to filter `sdpa_kernels`, use
    `do_filter_kernels=True` for the first call, which returns the filtered
    list of kernels, and pass that in subsequent calls (with
    `do_filter_kernels=False`). Otherwise, you will get graph breaks when
    using `torch.compile`.

    Args:
        query: Query tensor, `(bs, nh_q, q_len, hs)`
        key: Key tensor, `(bs, nh_k, kv_len, hs)`
        value: Value tensor, `(bs, nh_k, kv_len, hs)`
        scale_factor: Scale factor for SDPA
        sdpa_kernels: Kernels to be considered, in this order
        do_filter_kernels: See above
        mask: Mask tensor, optional

    Returns:
        `(attn_outputs, filtered_sdpa_kernels)`, where `attn_outputs` is
        the SDPA output, and `filtered_sdpa_kernels` is the filtered list of
        SDPA kernels if `do_filter_kernels=True`, and `None` otherwise.

    """
    is_causal = mask is None
    n_head = query.shape[1]
    n_query_groups = key.shape[1]
    enable_gqa = n_query_groups < n_head
    if enable_gqa:
        # Some efficient kernels have not implemented
        # `enabla_gqa=True`. It is better to extend keys, values in
        # this case.
        key = repeat_interleave(key, n_head)
        value = repeat_interleave(value, n_head)
        enable_gqa = key.shape[1] == n_query_groups
    kwargs = dict(
        query=query,
        key=key,
        value=value,
        attn_mask=mask,
        dropout_p=0.0,
        scale=scale_factor,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
    )
    if not isinstance(sdpa_kernels, list):
        sdpa_kernels = [sdpa_kernels]
    if sdpa_kernels:
        if do_filter_kernels:
            sdpa_kernels = filter_sdpa_kernels(sdpa_kernels=sdpa_kernels, **kwargs)
    if sdpa_kernels:
        with sdpa_kernel(sdpa_kernels):
            attn_outputs = F.scaled_dot_product_attention(**kwargs)
    else:
        attn_outputs = F.scaled_dot_product_attention(**kwargs)
    return attn_outputs, sdpa_kernels if do_filter_kernels else None
