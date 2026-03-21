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
from typing import List, Optional, Union, Tuple, Callable, Dict, Any

import torch
from torch.nn.attention import SDPBackend

from keys_values.attention_utils import pytorch_scaled_dot_product_attention
from keys_values.utils import expand_index, is_index_1d, index_to_3d


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


# `callback(key, value, extra_info, extend_kv)`
ReorderAnnotationCallback = Callable[
    [torch.Tensor, torch.Tensor, Dict[str, Any], bool], None
]


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: float,
    input_pos: int,
    token_positions: Optional[torch.Tensor],
    sdpa_kernels: Optional[Union[SDPBackend, List[SDPBackend]]] = None,
    do_filter_kernels: bool = False,
    annotation_callback: Optional[ReorderAnnotationCallback] = None,
    sort_if_3d: bool = False,
) -> Tuple[torch.Tensor, Optional[List[SDPBackend]]]:
    """
    Wraps `F.scaled_dot_product_attention` in a way which supports
    `q_len < kv_len` and reordered `key`, `value` according to `token_positions`.

    This must not be called for prefill (`input_pos == 0`), and after the KV
    cache buffers `key`, `value` have been updated, meaning that
    `range(input_pos, input_pos + q_len)` must be in each
    `token_positions[b, h]`.

    Note: Since efficient SDPA kernels do not support `q_len < kv_len` with
    causal masking, we call them with a padded query tensor of length `kv_len`.
    Once this case is properly supported by kernels other than the C++
    reference kernel, this function here becomes obsolete.

    Note: The reordering of `key` and `value` entries we do here implicitly,
    could also be done in the KV buffers. Then, the `q_len` new entries would
    occupy the right end of `key`, `value`, and `token_positions` would not be
    needed here. But in the long run, a better solution is to create an
    efficient SDPA kernel which does its causal masking based on
    `token_positions`, since reordering buffer entries takes time and memory.

    Args:
        query: Queries, shape `(batch_size, n_head, q_len, head_size)`
        key: Keys, shape `(batch_size, n_query_groups, kv_len, head_size)`
        value: Values, shape `(batch_size, n_query_groups, kv_len, head_size)`
        scale_factor: Scale factor for attention
        input_pos: Position in input sequence
        token_positions: Contains token positions in KV cache, shape
            `(batch_size, n_query_groups, kv_len)`. See above. If not given,
            we must have `input_pos + q_len == kv_len`, and the new KV
            entries are on the right end. This happens when the cache is
            built up.
        sdpa_kernels: Kernels to be used for SDPA can be restricted by
            `sdpa_kernels`.
        annotation_callback: If this is given and `key, value` are reordered,
            the results are passed to this callback.
        sort_if_3d: See :func:`reorder_key_value`.

    Returns:
        Attention outputs, shape `(batch_size, n_heads, q_len, head_size)`

    """
    batch_size, n_head, n_query_groups, q_len, kv_len, head_size = sdpa_check_args(
        query,
        key,
        value,
    )
    if sdpa_kernels is None:
        sdpa_kernels = []
    if token_positions is None:
        if input_pos + q_len != kv_len:
            raise ValueError(
                f"Without token_positions, must have input_pos + q_len = {input_pos + q_len} == {kv_len} = kv_len"
            )
        extra_info = dict()
    else:
        # Reorder entries in `key`, `value`, so that new entries are on the
        # right. New entries are those with `token_positions >= input_pos`.
        # Note: This simple solution just reorders all entries in `key`,
        # `buffer`, using the index which sorts `token_positions`.
        # We implemented an alternative which exchanges smaller parts of
        # `key`, `value`, but this does not end up being faster
        # (see `sdpa_wrapper_old` module).
        if input_pos == 0:
            raise ValueError("For input_pos=0, token_positions must be None")
        if token_positions.shape != key.shape[:-1]:
            raise ValueError(
                f"token_positions.shape = {token_positions.shape}, key.shape = {key.shape}: Not compatible"
            )
        key, value, extra_info = reorder_key_value(
            key,
            value,
            token_positions.detach(),
            input_pos,
            q_len,
            sort_if_3d,
        )

    # At this point, the new entries in `key`, `value`, corresponding to the
    # `query` tokens, are on the right end. Causal masking works if `query`
    # is zero-padded on the left
    if q_len < kv_len:
        fill_left = torch.zeros(
            (1, 1, 1, 1),
            dtype=query.dtype,
            device=query.device,
        ).expand(batch_size, n_head, kv_len - q_len, head_size)
        query = torch.cat((fill_left, query), dim=2)
    if annotation_callback is not None:
        annotation_callback = partial(
            annotation_callback,
            extra_info=extra_info,
        )
    full_y, filtered_kernels = pytorch_scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        scale_factor=scale_factor,
        sdpa_kernels=sdpa_kernels,
        do_filter_kernels=do_filter_kernels,
        annotation_callback=annotation_callback,
    )
    if q_len < kv_len:
        attn_output = full_y[:, :, (-q_len):, :].clone()
    else:
        attn_output = full_y
    return attn_output, filtered_kernels


def _extract_index_gather_scatter(
    token_positions: torch.Tensor,
    input_pos: int,
    q_len: int,
    check_token_pos: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Determines two indexes which are used to permute `key`, `value` so
    that information corresponding to the largest token positions
    `>= input_pos` moves to the right end. This "two step" solution is an
    alternative to sorting `token_positions`, which is more expensive if
    the index is 3D.

    Both `index_gat`, `index_scat` contain the same entries, namely the
    positions of entries `>= input_pos` in `token_positions`. But they
    are ordered differently:
    - `index_gat`: Order in which entries `range(index_pos,
        index_pos + q_len)` appear in `token_positions`.
    - `index_scat`: `index_sorted` is permuted by the inverse of the
        permutation which sorts`token_positions[:, :, (-q_len):]`.

    This ensures that things work out if there are overlaps between
    `index_sorted` and the right end of length `q_len`.
    See technical report for details.

    Args:
        token_positions: Token positions in KV cache
        input_pos: Position in input sequence, must be `> 0`
        q_len: Length of query sequence
        check_token_pos: If `True`, check that `token_positions` is valid.
            Use this for testing. Must be `False` if this is used as part
            of a computation graph

    Returns:
        `(index_gat, index_scat)`, each of shape
        `(batch_size, n_query_groups, q_len)`. See above.

    """
    batch_size, n_query_groups, _ = token_positions.shape
    new_entries_mask = token_positions >= input_pos
    if check_token_pos:
        dummy = new_entries_mask.sum(dim=-1)
        if not (dummy == q_len).all().item():
            raise ValueError(
                f"token_positions must have entries [{input_pos}, {input_pos + q_len}) in every slice. dummy = {dummy}"
            )
    nz0, nz1, nz2 = new_entries_mask.nonzero(as_tuple=True)
    if check_token_pos:
        kwargs = dict(dtype=nz0.dtype, device=nz0.device)
        nz0_should_be = (
            torch.arange(batch_size, **kwargs)[:, None, None]
            .expand(-1, n_query_groups, q_len)
            .flatten()
        )
        if not nz0.equal(nz0_should_be):
            raise ValueError(f"nz0 = {nz0}, must equal to {nz0_should_be}")
        nz1_should_be = (
            torch.arange(n_query_groups, **kwargs)[None, :, None]
            .expand(batch_size, -1, q_len)
            .flatten()
        )
        if not nz1.equal(nz1_should_be):
            raise ValueError(f"nz1 = {nz1}, must equal to {nz1_should_be}")
    elif nz2.numel() != batch_size * n_query_groups * q_len:
        raise ValueError(
            f"Invalid token_positions: Number of entries in [{input_pos}, {input_pos + q_len}) must be {batch_size * n_query_groups * q_len}, but is {nz2.numel()}"
        )
    index_sorted = nz2.view(batch_size, n_query_groups, q_len)
    # `index_gat`: Order in which entries `range(index_pos, index_pos + q_len)`
    # appear in `token_positions`.
    new_positions = (
        token_positions[nz0, nz1, nz2].view(
            batch_size,
            n_query_groups,
            q_len,
        )
        - input_pos
    )
    index_gat = torch.zeros_like(index_sorted).scatter(
        -1,
        index=new_positions,
        src=index_sorted,
    )
    # index_scat`: `index_sorted` is permuted by the inverse of the
    # permutation which sorts`token_positions[:, :, (-q_len):]`
    sort_final = torch.argsort(token_positions[:, :, (-q_len):], dim=-1)
    inv_sort_final = torch.zeros_like(sort_final).scatter(
        -1,
        index=sort_final,
        src=index_to_3d(
            torch.arange(
                q_len,
                dtype=sort_final.dtype,
                device=sort_final.device,
            ),
            batch_size,
            n_query_groups,
        ),
    )
    index_scat = index_sorted.gather(-1, inv_sort_final)
    return index_gat, index_scat


def _reorder(
    x: torch.Tensor,
    index_gat: torch.Tensor,
    index_scat: torch.Tensor,
    do_single_step: bool = False,
) -> torch.Tensor:
    """
    Exchange two parts of size `(batch_size, n_query_groups, q_len, head_size)`
    in `x`. One is `x.gather(2, index_gat)`, the other is
    `x[:, :, (-q_len):, :]`. `index_gat[b, h, :]` and `index_scat[b, h, :]`
    have the same values, but in different orderings. `index_scat` is used
    with `scatter`. This is needed in order to not make mistakes when there
    are overlaps.

    """
    q_len = index_gat.shape[-1]
    _, _, kv_len, head_size = x.shape
    x_new = x.gather(2, expand_index(index_gat, head_size))
    x_right = x[:, :, (-q_len):, :]
    if not do_single_step:
        x = x.scatter(2, expand_index(index_scat, head_size), x_right.clone())
        x = torch.cat((x[:, :, :(-q_len), :], x_new), dim=2)
    else:
        # Note: `index_scat`, `index_right` can overlap, in which case the
        # outcome of `scatter` can be non-deterministic. Does this matter?
        # Does it make a difference time-wise?
        index_right = torch.arange(
            kv_len - q_len,
            kv_len,
            dtype=index_gat.dtype,
            device=index_gat.device,
        )[None, None, :].expand(*x.shape[:2], -1)
        x = x.scatter(
            2,
            index=expand_index(
                torch.cat((index_scat, index_right), dim=-1),
                head_size,
            ),
            src=torch.cat((x_right, x_new), dim=2),
        )
    return x


def reorder_key_value(
    key: torch.Tensor,
    value: torch.Tensor,
    token_positions: torch.Tensor,
    input_pos: int,
    q_len: int,
    sort_if_3d: bool = False,
    check_token_pos: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Reorder `key, value` tensors using permutations (for each b, h) which, if
    applied to `token_positions`, place `input_pos:(input_pos + q_len)` at the
    right end. This is done in different ways:

    * If `token_positions` is essentially 1D, we use the sorting permutation,
      returned as `sort_index` (1D)
    * If `token_positions` is 3D, we use the sorting permutation if
      `sort_if_3d == True`. Otherwise, we use a two-step permutation,
      parameterized by `index_gat`, `index_scat`. These are cheaper to obtain.

    """
    if is_index_1d(token_positions):
        # `token_positions` is essentially 1D
        extra_info = dict(sort_index=torch.argsort(token_positions[0, 0, :]))
    elif sort_if_3d:
        extra_info = dict(sort_index=torch.argsort(token_positions, dim=-1))
    else:
        index_gat, index_scat = _extract_index_gather_scatter(
            token_positions,
            input_pos,
            q_len,
            check_token_pos,
        )
        extra_info = dict(index_gat=index_gat, index_scat=index_scat)
    return (
        reorder_buffer_given_extra_info(key, **extra_info),
        reorder_buffer_given_extra_info(value, **extra_info),
        extra_info,
    )


def reorder_buffer_given_extra_info(
    buffer: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Same as :func:`reorder_key_value`, but the permutation indices are
    given here, not determined.

    """
    sort_index = kwargs.get("sort_index")
    if sort_index is not None:
        if sort_index.ndim == 1:
            buffer = buffer[:, :, sort_index, :]
        else:
            index = expand_index(sort_index, buffer.shape[-1])
            buffer = torch.gather(buffer, -2, index)
    else:
        index_gat = kwargs["index_gat"]
        index_scat = kwargs["index_scat"]
        buffer = _reorder(buffer, index_gat, index_scat)
    return buffer
