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
from typing import List, Optional, Union

import torch
from torch.nn.attention import SDPBackend

from keys_values.attention_utils import pytorch_scaled_dot_product_attention


def _extend_index(index: torch.Tensor, head_size: int) -> torch.Tensor:
    return index.unsqueeze(-1).expand(-1, -1, -1, head_size)


# TODO: If `index_ext` is not unique, the behaviour of `scatter` is
# non-deterministic for entries mapping to the same position, and the
# gradient propagated to these entries is wrong.
# ==> Is this a problem?
#
# Would be hard to avoid! We could first copy one, then the other, but this
# would need two copies of `x`. We could maybe isolate the critical positions
# and order only them?
def _reorder(
    x: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    """
    Exchange two parts of size `(batch_size, n_query_groups, q_len, head_size)`
    in `x`. One is determined by the gather index `index`, the other is
    `x[:, :, (-q_len):, :]`.

    """
    q_len = index.shape[-1]
    head_size = x.shape[-1]
    scatter_src = torch.cat(
        (
            x[:, :, (-q_len):, :],
            x.gather(2, _extend_index(index, head_size)),
        ),
        dim=2,
    )
    kv_len = x.shape[2]
    final_index = torch.arange(
        kv_len - q_len, kv_len, dtype=index.dtype, device=index.device,
    )[None, None, :].expand(*x.shape[:2], -1)
    pair_index = torch.cat((index, final_index), dim=-1)
    return torch.scatter(
        x, 2, _extend_index(pair_index, head_size), scatter_src,
    )


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: float,
    input_pos: int,
    token_positions: Optional[torch.Tensor],
    sdpa_kernels: Optional[Union[SDPBackend, List[SDPBackend]]] = None,
    check_token_pos: bool = False,
) -> torch.Tensor:
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
        input_pos: Position in input sequence, must be `> 0`
        token_positions: Contains token positions in KV cache, shape
            `(batch_size, n_query_groups, kv_len)`. See above. If not given,
            we must have `input_pos + q_len == kv_len`, and the new KV
            entries are on the right end. This happens when the cache is
            built up.
        sdpa_kernels: Kernels to be used for SDPA can be restricted by
            `sdpa_kernels`.
        check_token_pos: If `True`, check that `token_positions` is valid.
            Use this for testing.

    Returns:
        Attention outputs, shape `(batch_size, n_heads, q_len, head_size)`

    """
    if input_pos <= 0:
        raise ValueError("input_pos must be positive. Don't use this for prefill")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, value must be 4D tensors")
    if key.shape != value.shape:
        raise ValueError("key, value must have same shape")
    batch_size, n_head, q_len, head_size = query.shape
    if key.shape[0] != batch_size or key.shape[-1] != head_size:
        raise ValueError(f"key.shape = {key.shape}, must be ({batch_size}, _, _, {head_size})")
    _, n_query_groups, kv_len, _ = key.shape
    if not (0 < q_len < kv_len):
        raise ValueError(f"Must have 0 < q_len = {q_len} < kv_len = {kv_len}. Don't use this for prefill")
    if sdpa_kernels is None:
        sdpa_kernels = []

    if token_positions is None:
        if input_pos + q_len != kv_len:
            raise ValueError(f"Without token_positions, must have input_pos + q_len = {input_pos + q_len} == {kv_len} = kv_len")
    else:
        # Reorder entries in `key`, `value`, so that new entries are on the
        # right. New entries are those with `token_positions >= input_pos`.
        if token_positions.shape != key.shape[:-1]:
            raise ValueError(f"token_positions.shape = {token_positions.shape}, key.shape = {key.shape}: Not compatible")
        new_entries_mask = token_positions >= input_pos
        if check_token_pos:
            dummy = new_entries_mask.sum(dim=-1)
            if not (dummy == q_len).all().item():
                raise ValueError(f"token_positions must have entries [{input_pos}, {input_pos + q_len}) in every -1 slice. dummy = {dummy}")
        nz0, nz1, nz2 = new_entries_mask.nonzero(as_tuple=True)
        if check_token_pos:
            kwargs = dict(dtype=nz0.dtype, device=nz0.device)
            nz0_should_be = torch.arange(batch_size, **kwargs)[:, None, None].expand(
                -1, n_query_groups, q_len).flatten()
            if not nz0.equal(nz0_should_be):
                raise ValueError(f"nz0 = {nz0}, must equal to {nz0_should_be}")
            nz1_should_be = torch.arange(n_query_groups, **kwargs)[None, :, None].expand(
                batch_size, -1, q_len).flatten()
            if not nz1.equal(nz1_should_be):
                raise ValueError(f"nz1 = {nz1}, must equal to {nz1_should_be}")
        elif nz2.numel() != batch_size * n_query_groups * q_len:
            raise ValueError(f"Invalid token_positions: Number of entries in [{input_pos}, {input_pos + q_len}) must be {batch_size * n_query_groups * q_len}, but is {nz2.numel()}")
        # Index for `_reorder`
        new_entries_index = nz2.view(batch_size, n_query_groups, q_len)
        # Exchange new entries with final `q_len` ones
        key = _reorder(key, new_entries_index)
        value = _reorder(value, new_entries_index)

    # At this point, the new entries in `key`, `value`, corresponding to the
    # `query` tokens, are on the right end. Causal masking works if `query`
    # is zero-padded on the left
    fill_left = torch.zeros(
        (1, 1, 1, 1), dtype=query.dtype, device=query.device,
    ).expand(batch_size, n_head, kv_len - q_len, head_size)
    query = torch.cat((fill_left, query), dim=2)
    return pytorch_scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        scale_factor=scale_factor,
        sdpa_kernels=sdpa_kernels,
    )[:, :, (-q_len):, :].clone()
