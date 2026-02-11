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
from typing import List, Optional, Union, Tuple, Callable

import torch
from torch.nn.attention import SDPBackend

from keys_values.attention_utils import pytorch_scaled_dot_product_attention
from keys_values.utils import expand_index, is_index_1d


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


ReorderAnnotationCallback = Callable[
    [torch.Tensor, torch.Tensor, Optional[torch.Tensor]], None
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
        query: Queries, shape `(batch_size, n_heads, q_len, head_size)`
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

    Returns:
        Attention outputs, shape `(batch_size, n_heads, q_len, head_size)`

    """
    batch_size, n_head, _, q_len, kv_len, head_size = sdpa_check_args(
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
        sort_index = None
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
        key, value, sort_index = reorder_key_value(
            key,
            value,
            token_positions.detach(),
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
            sort_index=sort_index,
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


def reorder_key_value(
    key: torch.Tensor,
    value: torch.Tensor,
    token_positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not is_index_1d(token_positions):
        sort_index = torch.argsort(token_positions, dim=-1)
    else:
        # `token_positions` is essentially 1D
        sort_index = torch.argsort(token_positions[0, 0, :])
    return (
        reorder_buffer_given_sort_index(key, sort_index),
        reorder_buffer_given_sort_index(value, sort_index),
        sort_index,
    )


def reorder_buffer_given_sort_index(
    buffer: torch.Tensor,
    sort_index: torch.Tensor,
) -> torch.Tensor:
    if sort_index.ndim == 3:
        index = expand_index(sort_index, buffer.shape[-1])
        buffer = torch.gather(buffer, -2, index)
    else:
        buffer = buffer[:, :, sort_index, :]
    return buffer
