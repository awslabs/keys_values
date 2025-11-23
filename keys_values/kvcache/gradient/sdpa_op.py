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
import math
from typing import Optional, List, Union, Tuple

import torch
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from litgpt.config import Config

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.attention import (
    scaled_dot_product_attention_in_blocks,
    DefaultKeysAndValues,
    MultiHeadSelfAttention,
)
from keys_values.attention_utils import (
    filter_sdpa_kernels,
    create_temp_array,
    sdpa_attention_weights,
    slice_as_flat,
)
from keys_values.use_eager_kernel import transform_mha_kwargs
from keys_values.utils import expand_index, repeat_interleave


# TODO: Remove once SDPAFunction._forward_new is established!
def sdpa_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_positions: Optional[torch.Tensor],
    input_pos: int,
    scale_factor: float,
    sliding_window_size: Optional[int],
    sdpa_kernels: Optional[Union[SDPBackend, List[SDPBackend]]],
    tmp_array_limit_gb: Optional[float] = None,
) -> torch.Tensor:
    is_causal = input_pos == 0
    _, n_query_groups, kv_len, _ = key.shape
    _, n_head, q_len, _ = query.shape
    enable_gqa = n_query_groups < n_head
    if is_causal and sliding_window_size is None:
        # Use `F.scaled_dot_product_attention`, which is optimized for the
        # causal case
        if enable_gqa:
            # Some efficient fused kernels have not implemented
            # `enabla_gqa=True`. It is better to extend keys, values in
            # this case.
            key = repeat_interleave(key, n_head)
            value = repeat_interleave(value, n_head)
            enable_gqa = key.shape[1] == n_query_groups
        # Run the right version of `F.scaled_dot_product_attention`
        kwargs = dict(
            query=query,
            key=key,
            value=value,
            attn_mask=None,
            dropout_p=0.0,
            scale=scale_factor,
            is_causal=True,
            enable_gqa=enable_gqa,
        )
        if sdpa_kernels is not None:
            if not isinstance(sdpa_kernels, list):
                sdpa_kernels = [sdpa_kernels]
            # Filter out kernels which are not supported
            sdpa_kernels = filter_sdpa_kernels(sdpa_kernels, **kwargs)
        else:
            sdpa_kernels = []
        if sdpa_kernels:
            with sdpa_kernel(sdpa_kernels):
                attn_output = F.scaled_dot_product_attention(**kwargs)
        else:
            attn_output = F.scaled_dot_product_attention(**kwargs)
    else:
        # Use own implementation, which limits GPU memory usage
        attn_output, _ = scaled_dot_product_attention_in_blocks(
            query=query,
            k_and_v=DefaultKeysAndValues(key, value),
            scale_factor=scale_factor,
            return_attn_weights=False,
            input_pos=input_pos,
            token_positions=token_positions,
            sliding_window_size=sliding_window_size,
            tmp_array_limit_gb=tmp_array_limit_gb,
        )
    return attn_output


def sdpa_backward_core(
    grad_attn_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    tmp_array: torch.Tensor,
    token_positions: Optional[torch.Tensor],
    input_pos: int,
    scale_factor: float,
    sliding_window_size: Optional[int],
    need_query: bool,
    need_key: bool,
    need_value: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    assert value.dtype == torch.float32
    grad_query = grad_key = grad_value = None
    # We use a buffer `tmp_array` of shape
    # `(batch_size, n_head, q_len, kv_len)`, which is in general the
    # largest size. We make sure that no extra copies of this size
    # are created.
    batch_size, n_head, q_len, head_size = query.shape
    _, n_query_groups, kv_len, _ = key.shape
    q_per_kv = n_head // n_query_groups
    # Compute attention weights f(S) -> `attn_weights_part`
    attn_weights_part = sdpa_attention_weights(
        query=query,
        key=key,
        tmp_array=tmp_array,
        token_positions=token_positions,
        input_pos=input_pos,
        scale_factor=scale_factor,
        sliding_window_size=sliding_window_size,
    )
    if need_value:
        # Avoid transpose of `attn_weights_part`, which may create copy
        if q_per_kv == 1:
            grad_value = torch.matmul(grad_attn_output.mT, attn_weights_part)
        else:
            # Try to use `tmp_array` for the larger temp array
            shape = (batch_size, n_head, head_size, kv_len)
            numel = batch_size * n_head * head_size * kv_len
            if tmp_array.numel() >= numel:
                tmp_grad_val = tmp_array.view(-1)[:numel].view(*shape)
            else:
                tmp_grad_val = torch.empty(
                    shape,
                    dtype=attn_weights_part.dtype,
                    device=attn_weights_part.device,
                )
            torch.matmul(grad_attn_output.mT, attn_weights_part, out=tmp_grad_val)
            grad_value = tmp_grad_val.view(
                batch_size, n_query_groups, q_per_kv, head_size, kv_len,
            ).sum(dim=2)
        grad_value = grad_value.mT
    if need_query or need_key:
        if q_per_kv == 1:
            torch.matmul(grad_attn_output, value.mT, out=tmp_array)
        else:
            q_shape = (batch_size, n_query_groups, q_per_kv, q_len, head_size)
            _arg1 = grad_attn_output.view(*q_shape)
            _arg2 = value.unsqueeze(2).mT
            o_shape = q_shape[:3] + (q_len, kv_len)
            # _arg1: (bs, nh_k, q_per_kv, q_len, hs)
            # _arg2: (bs, nh_k, 1, hs, kv_len)
            # o_shape: (bs, nh_k, q_per_kv, q_len, kv_len)
            torch.matmul(
                _arg1,
                _arg2,
                out=tmp_array.view(*o_shape)
            )
        tmp_array *= attn_weights_part
        attn_weights_part *= tmp_array.sum(dim=-1, keepdim=True)  # (diag e) f(S)
        # E: (bs, nh_q, q_len, kv_len)
        tmp_array -= attn_weights_part  # E
        if need_query:
            # Compute matmul(E, K)
            if q_per_kv == 1:
                grad_query = torch.matmul(tmp_array, key)
            else:
                e_shape = (batch_size, n_query_groups, q_per_kv, q_len, kv_len)
                _arg1 = tmp_array.view(*e_shape)
                _arg2 = key.unsqueeze(2)
                # _arg1: (bs, nh_k, q_per_kv, q_len, kv_len)
                # _arg2: (bs, nh_k, 1, kv_len, hs)
                grad_query = torch.matmul(_arg1, _arg2).view(*query.shape)
            grad_query *= scale_factor
        if need_key:
            # Compute matmul(E.mT, Q) = matmul(Q.mT, E).mT
            # Avoid transpose of `tmp_array`, which may create copy
            if q_per_kv == 1:
                grad_key = torch.matmul(query.mT, tmp_array)
            else:
                # Try to use `attn_weights_part` for the larger temp array
                shape = (batch_size, n_head, head_size, kv_len)
                numel = batch_size * n_head * head_size * kv_len
                if attn_weights_part.numel() >= numel:
                    tmp_grad_key = attn_weights_part.view(-1)[:numel].view(*shape)
                else:
                    tmp_grad_key = torch.empty(
                        shape, dtype=tmp_array.dtype, device=tmp_array.device,
                    )
                torch.matmul(query.mT, tmp_array, out=tmp_grad_key)
                grad_key = tmp_grad_key.view(
                    batch_size, n_query_groups, q_per_kv, head_size, kv_len,
                ).sum(dim=2)
            grad_key *= scale_factor
            grad_key = grad_key.mT
    return grad_query, grad_key, grad_value


def sdpa_backward(
    grad_attn_output: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_positions: Optional[torch.Tensor],
    input_pos: int,
    scale_factor: float,
    sliding_window_size: Optional[int],
    need_query: bool,
    need_key: bool,
    need_value: bool,
    tmp_array_limit_gb: Optional[float] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    One difficulty here is that we'd like to do computations in `float32`. We
    control the size of the two temporary arrays allocated here. This is done
    by slicing along the `q_len` dimension, see :func:`create_temp_arrays`.
    Note that if `tmp_len * num_slices > q_len`, we compute a bit more than
    needed and extract at the end.

    """
    grad_query = grad_key = grad_value = None
    device = query.device
    dtype = query.dtype
    if need_query or need_key or need_value:
        # Allocate temporary arrays and determine the number of slices.
        batch_size, n_head, q_len, _ = query.shape
        kv_len = key.shape[2]
        if grad_attn_output.shape != query.shape:
            raise ValueError(f"grad_attn_output.shape = {grad_attn_output.shape}, query.shape = {query.shape}, must be the same")
        tmp_array, num_splits, tmp_len = create_temp_array(
            batch_size=batch_size,
            n_head=n_head,
            q_len=q_len,
            kv_len=kv_len,
            device=device,
            tmp_array_limit_gb=tmp_array_limit_gb,
        )
        # Iterate over slices along `q_len` dimension
        grad_query_parts = []
        grad_key_parts = []
        grad_value_parts = []
        key32 = key.to(torch.float32)
        value32 = value.to(torch.float32)
        start = 0
        for _ in range(num_splits):
            end = min(start + tmp_len, q_len)
            sz = end - start
            _input_pos = input_pos + start
            # Subfunctions assume these arrays are flat:
            _tmp_array = slice_as_flat(tmp_array, sz)
            _grad_query, _grad_key, _grad_value = sdpa_backward_core(
                grad_attn_output=grad_attn_output[:, :, start:end, :].to(torch.float32),
                query=query[:, :, start:end, :].to(torch.float32),
                key=key32,
                value=value32,
                tmp_array=_tmp_array,
                token_positions=token_positions,
                input_pos=_input_pos,
                scale_factor=scale_factor,
                sliding_window_size=sliding_window_size,
                need_query=need_query,
                need_key=need_key,
                need_value=need_value,
            )
            if need_query:
                grad_query_parts.append(_grad_query.to(dtype))
            if need_query:
                grad_key_parts.append(_grad_key)
            if need_value:
                grad_value_parts.append(_grad_value)
            start = end

        # Combine
        if need_query:
            grad_query = torch.cat(grad_query_parts, dim=2)
        if need_key:
            grad_key = grad_key_parts[0]
            for x in grad_key_parts[1:]:
                grad_key += x
            grad_key = grad_key.to(dtype)
        if need_value:
            grad_value = grad_value_parts[0]
            for x in grad_value_parts[1:]:
                grad_value += x
            grad_value = grad_value.to(dtype)

    return grad_query, grad_key, grad_value


class SDPAFunction(Function):
    """
    Provides `scaled_dot_product_attention` as an `autograd` operator,
    ensuring that only its inputs are stored in the `autograd` graph, not the
    intermediates.

    """
    @staticmethod
    def forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_positions: Optional[torch.Tensor],
        input_pos: int,
        scale_factor: float,
        sliding_window_size: Optional[int] = None,
        sdpa_kernels: Optional[Union[SDPBackend, List[SDPBackend]]] = None,
        tmp_array_limit_gb: Optional[float] = None,
    ) -> torch.Tensor:
        # Check dimensions
        assert query.ndim == 4 and key.ndim == 4 and value.ndim == 4
        assert key.shape == value.shape
        batch_size, n_query_groups, kv_len, head_size = key.shape
        assert query.shape[0] == batch_size and query.shape[3] == head_size
        _, n_head, q_len, _ = query.shape
        assert q_len <= kv_len
        assert n_query_groups <= n_head and n_head % n_query_groups == 0
        if input_pos == 0:
            assert q_len == kv_len
            assert token_positions is None
        else:
            assert token_positions is not None
            assert token_positions.shape == key.shape[:-1]
        return SDPAFunction._forward_new(
            query=query,
            key=key,
            value=value,
            token_positions=token_positions,
            input_pos=input_pos,
            scale_factor=scale_factor,
            sliding_window_size=sliding_window_size,
            sdpa_kernels=sdpa_kernels,
            tmp_array_limit_gb=tmp_array_limit_gb,
        )

    @staticmethod
    def _forward_old(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_positions: Optional[torch.Tensor],
        input_pos: int,
        scale_factor: float,
        sliding_window_size: Optional[int] = None,
        sdpa_kernels: Optional[Union[SDPBackend, List[SDPBackend]]] = None,
        tmp_array_limit_gb: Optional[float] = None,
    ) -> torch.Tensor:
        return sdpa_forward(
            query=query,
            key=key,
            value=value,
            token_positions=token_positions,
            input_pos=input_pos,
            scale_factor=scale_factor,
            sliding_window_size=sliding_window_size,
            sdpa_kernels=sdpa_kernels,
            tmp_array_limit_gb=tmp_array_limit_gb,
        )

    @staticmethod
    def _forward_new(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_positions: Optional[torch.Tensor],
        input_pos: int,
        scale_factor: float,
        sliding_window_size: Optional[int] = None,
        sdpa_kernels: Optional[Union[SDPBackend, List[SDPBackend]]] = None,
        tmp_array_limit_gb: Optional[float] = None,
    ) -> torch.Tensor:
        head_size = query.shape[-1]
        temp = 1.0 / math.sqrt(head_size)
        if abs(temp - scale_factor) < 1e-7:
            attention_scores_scalar = None
        else:
            temp = 1.0 / scale_factor
            attention_scores_scalar = int(temp * temp)
        config = Config(
            n_head=query.shape[1],
            n_query_groups=key.shape[1],
            head_size=head_size,
            sliding_window_size=sliding_window_size,
            attention_logit_softcapping=None,
            attention_scores_scalar=attention_scores_scalar,
        )
        if tmp_array_limit_gb is not None:
            tmp_array_limit_forward = TemporaryArrayLimit(
                init_val=tmp_array_limit_gb,
                name="attention_forward_temp_size_gb",
            )
        else:
            tmp_array_limit_forward = None
        mha_kwargs = dict(
            sdpa_kernels=sdpa_kernels,
            tmp_array_limit_gb=tmp_array_limit_forward,
        )
        mha = MultiHeadSelfAttention(
            config, **transform_mha_kwargs(mha_kwargs, config),
        )
        return mha.scaled_dot_product_attention(
            query=query,
            k_and_v=DefaultKeysAndValues(key, value),
            input_pos=input_pos,
            token_positions=token_positions,
            sdpa_mode=None,
            sliding_window_size=sliding_window_size,
            return_attn_weights=False,
            transpose_result=False,
        )[0]

    @staticmethod
    def setup_context(ctx, inputs, output):
        query, key, value, token_positions, input_pos, scale_factor, sliding_window_size, _, tmp_array_limit_gb = inputs
        ctx.save_for_backward(query, key, value)
        ctx.extra_args = dict(
            token_positions=token_positions,
            input_pos=input_pos,
            scale_factor=scale_factor,
            sliding_window_size=sliding_window_size,
            tmp_array_limit_gb=tmp_array_limit_gb,
        )

    @staticmethod
    def backward(ctx, grad_attn_output: torch.Tensor):
        # Inputs from context
        query, key, value = ctx.saved_tensors
        # Main computation
        grad_query, grad_key, grad_value = sdpa_backward(
            grad_attn_output=grad_attn_output,
            query=query,
            key=key,
            value=value,
            token_positions=ctx.extra_args["token_positions"],
            input_pos=ctx.extra_args["input_pos"],
            scale_factor=ctx.extra_args["scale_factor"],
            sliding_window_size=ctx.extra_args["sliding_window_size"],
            need_query=ctx.needs_input_grad[0],
            need_key=ctx.needs_input_grad[1],
            need_value=ctx.needs_input_grad[2],
            tmp_array_limit_gb=ctx.extra_args["tmp_array_limit_gb"],
        )
        return grad_query, grad_key, grad_value, *([None] * 6)


def finalize_gradients_key_or_value(
    grad_buffer: torch.Tensor,
    index_e: torch.Tensor,
    positions_e: Optional[torch.Tensor],
    index_prime: Optional[torch.Tensor],
    positions_prime: Optional[torch.Tensor],
    need_buffer: bool,
    need_new: bool,
) -> Optional[torch.Tensor]:
    """
    Helper for :meth:`KVCacheUpdateAndSDPAFunction.backward`.

    `grad_buffer`is used as input and output, by being overwritten. For keys,
    it contains F_k. At the end, `grad_buffer` contains `grad_key_buffer`, and
    `grad_key` is returned.

    Without a grace period, `index` contains I, the others are not used. With
    grace period, `positions_e` is J, `index_prime`is I', `positions_prime` is
    J'.

    """
    do_grace_period = positions_e is not None
    grad_new = None
    if not do_grace_period:
        # No grace period: Just use `index_e`
        if need_new:
            grad_new = grad_buffer.gather(dim=-2, index=index_e)
        if need_buffer:
            grad_buffer.scatter_(dim=-2, index=index_e, value=0)
    else:
        # With grace period: More complicated
        assert index_prime is not None and positions_prime is not None
        if need_new:
            grad_new = grad_buffer.gather(dim=-2, index=index_prime)
        if need_buffer:
            # F_k -> F_k'
            grad_buffer.scatter_(dim=-2, index=index_prime, value=0)
            # g_{J'}(F_k') -> temp_mat
            temp_mat = grad_buffer.gather(dim=-2, index=positions_prime)
            # s_{J'}(F_k', 0)
            grad_buffer.scatter_(dim=-2, index=positions_prime, value=0)
            # The addition of p_J(temp_mat) can be done as scatter, because
            # entries at I' and J' have been zeroed, which includes entries
            # at J.
            grad_buffer.scatter_(dim=-2, index=positions_e, src=temp_mat)
    return grad_new


def scatter_on_buffers(
    key: torch.Tensor,
    value: torch.Tensor,
    key_buffer: torch.Tensor,
    value_buffer: torch.Tensor,
    index: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Check dimensions
    assert key.ndim == 4
    batch_size, n_query_groups, q_len, head_size = key.shape
    assert key.shape == value.shape
    kv_len = key_buffer.shape[2]
    assert key_buffer.shape == (batch_size, n_query_groups, kv_len, head_size)
    assert key_buffer.shape == value_buffer.shape
    assert q_len <= kv_len
    assert index.shape == key.shape[:-1]
    do_grace_period = positions is not None
    if do_grace_period:
        if positions.ndim == 1:
            positions = positions[None, None, :].expand(
                batch_size, n_query_groups, -1,
            )
        else:
            assert positions.shape[:-1] == (batch_size, n_query_groups)
        assert 0 < positions.shape[-1] <= q_len
    # Update KV cache buffers
    if not do_grace_period:
        index_e = expand_index(index, head_size)
        key_buffer_new = key_buffer.scatter(-2, index_e, key)
        value_buffer_new = value_buffer.scatter(-2, index_e, value)
    else:
        ext_index = expand_index(
            torch.cat((index, positions), dim=-1), head_size,
        )
        positions_e = expand_index(positions, head_size)
        key_buffer_new = key_buffer.scatter(
            -2,
            ext_index,
            torch.cat((key_buffer.gather(-2, positions_e), key), dim=-2),
        )
        value_buffer_new = value_buffer.scatter(
            -2,
            ext_index,
            torch.cat((value_buffer.gather(-2, positions_e), value), dim=-2),
        )
    return key_buffer_new, value_buffer_new


class KVCacheScatterUpdateAndSDPAFunction(Function):
    """
    This `autograd` operator combines the "scatter" update of KV cache buffers
    with `scaled_dot_product_attention`, ensuring that only its inputs are
    stored in the `autograd` graph, not the intermediates. It corresponds to
    what :meth:`KVCache.forward` is doing when the KV cache is full.

    Note that different to `SDPAFunction`, the prefill case is not supported,
    in that `input_pos > 0`.

    """
    @staticmethod
    def forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_buffer: torch.Tensor,
        value_buffer: torch.Tensor,
        index: torch.Tensor,
        token_positions: torch.Tensor,
        input_pos: int,
        scale_factor: float,
        positions: Optional[torch.Tensor] = None,
        sliding_window_size: Optional[int] = None,
        sdpa_kernels: Optional[Union[SDPBackend, List[SDPBackend]]] = None,
        tmp_array_limit_gb: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combines updates of K and V cache buffers with SDPA computation.

        Args:
            query: Queries for new tokens,
                `(batch_size, n_head, q_len, head_size)`
            key: Keys for new tokens,
                `(batch_size, n_query_heads, q_len, head_size)`
            value: Values for new tokens,
                `(batch_size, n_query_heads, q_len, head_size)`
            key_buffer: Keys cache buffer before the update,
                `(batch_size, n_query_heads, kv_len, head_size)`
            value_buffer: Values cache buffer before the update,
                `(batch_size, n_query_heads, kv_len, head_size)`
            index: Positions for new information in the cache buffers,
                `(batch_size, n_query_heads, q_len)`
            token_positions: Token positions for cache content,
                `(batch_size, n_query_heads, kv_len)`. This is *after*
                the update. It is needed in order to determine attention masks
            input_pos: New tokens have positions
                `input_pos:(input_pos + q_len)`
            scale_factor: Scale factor for SDPA
            positions: This index is given iff the KV cache supports a grace
                period, `(batch_size, n_query_heads, num2)`, where
                `num2 <= min(q_len, grace_period)`. In this case, the buffer
                update is a bit more complicated.
                `positions` can also be 1D, in which case it is extended by the
                first two dimensions.
            sliding_window_size: Affects the attention masking
            sdpa_kernels: Parameter for SDPA

        Returns:
            `(attn_output, key_buffer_new, value_buffer_new)`, the SDPA
            output and the new KV cache buffers after the update.

        """
        # Check dimensions and update KV cache buffers
        if input_pos <= 0:
            raise ValueError("Operator supports input_pos > 0 only")
        assert token_positions.shape == key_buffer.shape[:-1]
        key_buffer_new, value_buffer_new = scatter_on_buffers(
            key,
            value,
            key_buffer,
            value_buffer,
            index,
            positions,
        )
        # Compute SDPA
        attn_output = SDPAFunction.forward(
            query=query,
            key=key_buffer_new,
            value=value_buffer_new,
            token_positions=token_positions,
            input_pos=input_pos,
            scale_factor=scale_factor,
            sliding_window_size=sliding_window_size,
            sdpa_kernels=sdpa_kernels,
            tmp_array_limit_gb=tmp_array_limit_gb,
        )
        return attn_output, key_buffer_new, value_buffer_new

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            query,
            key,
            value,
            key_buffer,
            value_buffer,
            index,
            token_positions,
            input_pos,
            scale_factor,
            positions,
            sliding_window_size,
            _,
            tmp_array_limit_gb,
        ) = inputs
        if positions is not None and positions.ndim == 1:
            positions = positions[None, None, :].expand(
                *index.shape[:2], -1,
            )
        ctx.save_for_backward(
            query,
            key,
            value,
            key_buffer,
            value_buffer,
        )
        ctx.extra_args = dict(
            index=index,
            token_positions=token_positions,
            positions=positions,
            input_pos=input_pos,
            scale_factor=scale_factor,
            sliding_window_size=sliding_window_size,
            tmp_array_limit_gb=tmp_array_limit_gb,
        )

    @staticmethod
    def backward(
        ctx,
        grad_attn_output: torch.Tensor,
        grad_key_buffer_new: torch.Tensor,
        grad_value_buffer_new: torch.Tensor,
    ):
        """
        The computations here are derived in a technical report. Notation
        translates as follows:
        - query -> Q
        - key -> K
        - value -> V
        - key_buffer -> K_c, key_buffer_new -> K_c'
        - value_buffer -> V_c, value_buffer_new -> V_c'
        - attn_output -> Y
        - kv_len -> T_k
        - q_len -> T_q
        - head_size -> d_h
        - index -> I
        - positions -> J
        - index_prime -> I'
        - positions_prime -> J'
        - grad_* -> bar on top

        In general, indices have 3 dims. To be used in `gather`, `scatter`,
        we extend them by a 4th dimension of size `head_size`, and call
        these 4D versions `index_e`, `positions_e`, ...

        """
        # Inputs from context
        (
            query,
            key,
            value,
            key_buffer,
            value_buffer,
        ) = ctx.saved_tensors
        index = ctx.extra_args["index"]
        token_positions = ctx.extra_args["token_positions"]
        input_pos = ctx.extra_args["input_pos"]
        scale_factor = ctx.extra_args["scale_factor"]
        sliding_window_size = ctx.extra_args["sliding_window_size"]
        tmp_array_limit_gb = ctx.extra_args["tmp_array_limit_gb"]
        positions = ctx.extra_args["positions"]
        do_grace_period = positions is not None
        grad_query = grad_key = grad_value = None
        grad_key_buffer = grad_value_buffer = None
        # Prepare inputs
        head_size = query.shape[-1]
        need_query = ctx.needs_input_grad[0]
        need_key = ctx.needs_input_grad[1]
        need_value = ctx.needs_input_grad[2]
        need_key_buffer = ctx.needs_input_grad[3]
        need_value_buffer = ctx.needs_input_grad[4]
        need_one_of_key = need_key or need_key_buffer
        need_one_of_value = need_value or need_value_buffer
        if need_query or need_one_of_key or need_one_of_value:
            # Compute `key_buffer_new`, `value_buffer_new`.
            # Note: We tried to overwrite `key_buffer`, `value_buffer` with
            # these and restore them later. But `autograd` does not allow that:
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [5, 4, 512, 64]], which is output 0 of CatBackward0, is at version 2; expected version 0 instead
            index_e = expand_index(index, head_size)
            if not do_grace_period:
                key_buffer_new, value_buffer_new = scatter_on_buffers(
                    key,
                    value,
                    key_buffer,
                    value_buffer,
                    index,
                    positions,
                )
                positions_e = None
                index_prime = None
                positions_prime = None
            else:
                num2 = positions.shape[-1]
                positions_prime = expand_index(index[:, :, :num2], head_size)
                index_prime = expand_index(
                    torch.cat((index[:, :, num2:], positions), dim=-1),
                    head_size,
                )
                positions_e = expand_index(positions, head_size)
                key_buffer_new = key_buffer.scatter(
                    -2,
                    positions_prime,
                    key_buffer.gather(-2, positions_e),
                )
                key_buffer_new.scatter_(
                    -2,
                    index_prime,
                    key,
                )
                value_buffer_new = value_buffer.scatter(
                    -2,
                    positions_prime,
                    value_buffer.gather(-2, positions_e),
                )
                value_buffer_new.scatter_(
                    -2,
                    index_prime,
                    value,
                )
            # Backward of SDPA
            grad_query, grad_key_buffer, grad_value_buffer = sdpa_backward(
                grad_attn_output=grad_attn_output,
                query=query,
                key=key_buffer_new,
                value=value_buffer_new,
                token_positions=token_positions,
                input_pos=input_pos,
                scale_factor=scale_factor,
                sliding_window_size=sliding_window_size,
                need_query=need_query,
                need_key=need_one_of_key,
                need_value=need_one_of_value,
                tmp_array_limit_gb=tmp_array_limit_gb,
            )
            if need_one_of_key:
                grad_key_buffer += grad_key_buffer_new  # F_k in note
            if need_one_of_value:
                grad_value_buffer += grad_value_buffer_new  # F_v in note
            # At this point, `grad_query` is done, while `grad_key_buffer`,
            # `grad_value_buffer` contain F_k, F_v respectively.
            # Finalize gradients
            if need_one_of_key:
                grad_key = finalize_gradients_key_or_value(
                    grad_buffer=grad_key_buffer,
                    index_e=index_e,
                    positions_e=positions_e,
                    index_prime=index_prime,
                    positions_prime=positions_prime,
                    need_buffer=need_key_buffer,
                    need_new=need_key,
                )
                # Very unlikely
                if not need_key_buffer:
                    grad_key_buffer = None
            if need_one_of_value:
                grad_value = finalize_gradients_key_or_value(
                    grad_buffer=grad_value_buffer,
                    index_e=index_e,
                    positions_e=positions_e,
                    index_prime=index_prime,
                    positions_prime=positions_prime,
                    need_buffer=need_value_buffer,
                    need_new=need_value,
                )
                # Very unlikely
                if not need_value_buffer:
                    grad_value_buffer = None

        return grad_query, grad_key, grad_value, grad_key_buffer, grad_value_buffer, *([None] * 10)


def cat_on_buffers(
    key: torch.Tensor,
    value: torch.Tensor,
    key_buffer: Optional[torch.Tensor],
    value_buffer: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    # Check dimensions
    assert key.ndim == 4
    assert key.shape == value.shape
    batch_size, n_query_groups, q_len, head_size = key.shape
    buffers_none = key_buffer is None
    if not buffers_none:
        kv_len = key_buffer.shape[2]
        assert key_buffer.shape == value_buffer.shape
        assert key_buffer.shape == (batch_size, n_query_groups, kv_len, head_size)
        # Update KV cache buffers
        key_buffer_new = torch.cat((key_buffer, key), dim=-2)
        value_buffer_new = torch.cat((value_buffer, value), dim=-2)
    else:
        assert value_buffer is None
        kv_len = 0
        # Avoids the following error:
        # RuntimeError: A input that has been returned as-is as output is being saved for backward. This is not supported if you override setup_context. You should return and save a view of the input instead, e.g. with x.view_as(x) or setup ctx inside the forward function itself
        key_buffer_new = key
        value_buffer_new = value
    # Compute SDPA
    token_positions = torch.arange(
        0, kv_len + q_len, device=key.device, dtype=torch.int,
    ).view(1, 1, -1).expand(batch_size, n_query_groups, -1)
    return key_buffer_new, value_buffer_new, token_positions, kv_len


class KVCacheCatUpdateAndSDPAFunction(Function):
    """
    This `autograd` operator combines the "cat" update of KV cache buffers
    with `scaled_dot_product_attention`, ensuring that only its inputs are
    stored in the `autograd` graph, not the intermediates. It corresponds to
    what :meth:`KVCache.forward` is doing when the KV cache is not yet full.

    This operator cannot be used for the prefill call, because `key_buffer`,
    `value_buffer` must be given.

    """
    @staticmethod
    def forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_buffer: torch.Tensor,
        value_buffer: torch.Tensor,
        scale_factor: float,
        sliding_window_size: Optional[int] = None,
        sdpa_kernels: Optional[Union[SDPBackend, List[SDPBackend]]] = None,
        tmp_array_limit_gb: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combines updates of K and V cache buffers with SDPA computation.

        Args:
            query: Queries for new tokens,
                `(batch_size, n_head, q_len, head_size)`
            key: Keys for new tokens,
                `(batch_size, n_query_heads, q_len, head_size)`
            value: Values for new tokens,
                `(batch_size, n_query_heads, q_len, head_size)`
            key_buffer: Keys cache buffer before the update,
                `(batch_size, n_query_heads, kv_len, head_size)`. Can be `None`
            value_buffer: Values cache buffer before the update,
                `(batch_size, n_query_heads, kv_len, head_size)`. Can be `None`
            scale_factor: Scale factor for SDPA
            sliding_window_size: Affects the attention masking
            sdpa_kernels: Parameter for SDPA

        Returns:
            `(attn_output, key_buffer_new, value_buffer_new)`, the SDPA
            output and the new KV cache buffers after the update. The
            new buffers have shape `(batch_size, n_query_heads, kv_len + q_len,
            head_size)`. If `key_buffer`, `value_buffer` are `None`, so is
            `key_buffer_new`, `value_buffer_new`.

        """
        if key_buffer is None or value_buffer is None:
            raise ValueError("key_buffer, value_buffer must be given. Do not use this operator for the prefill call")
        # Check dimensions, compose new buffers
        key_buffer_new, value_buffer_new, token_positions, kv_len = cat_on_buffers(
            key, value, key_buffer, value_buffer,
        )
        # Compute SDPA
        attn_output = SDPAFunction.forward(
            query=query,
            key=key_buffer_new,
            value=value_buffer_new,
            token_positions=token_positions,
            input_pos=kv_len,
            scale_factor=scale_factor,
            sliding_window_size=sliding_window_size,
            sdpa_kernels=sdpa_kernels,
            tmp_array_limit_gb=tmp_array_limit_gb,
        )
        return attn_output, key_buffer_new, value_buffer_new

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            query,
            key,
            value,
            key_buffer,
            value_buffer,
            scale_factor,
            sliding_window_size,
            _,
            tmp_array_limit_gb,
        ) = inputs
        ctx.save_for_backward(
            query,
            key,
            value,
            key_buffer,
            value_buffer,
        )
        ctx.extra_args = dict(
            scale_factor=scale_factor,
            sliding_window_size=sliding_window_size,
            tmp_array_limit_gb=tmp_array_limit_gb,
        )

    @staticmethod
    def backward(
        ctx,
        grad_attn_output: torch.Tensor,
        grad_key_buffer_new: torch.Tensor,
        grad_value_buffer_new: torch.Tensor,
    ):
        """
        See comments of :meth:`KVCacheScatterUpdateAndSDPAFunction.backward`.

        """
        # Inputs from context
        (
            query,
            key,
            value,
            key_buffer,
            value_buffer,
        ) = ctx.saved_tensors
        scale_factor = ctx.extra_args["scale_factor"]
        sliding_window_size = ctx.extra_args["sliding_window_size"]
        tmp_array_limit_gb = ctx.extra_args["tmp_array_limit_gb"]
        grad_query = grad_key = grad_value = None
        grad_key_buffer = grad_value_buffer = None
        # Prepare inputs
        batch_size, n_query_groups, _, _ = key_buffer.shape
        need_query = ctx.needs_input_grad[0]
        need_key = ctx.needs_input_grad[1]
        need_value = ctx.needs_input_grad[2]
        need_key_buffer = ctx.needs_input_grad[3]
        need_value_buffer = ctx.needs_input_grad[4]
        need_one_of_key = need_key or need_key_buffer
        need_one_of_value = need_value or need_value_buffer
        if need_query or need_one_of_key or need_one_of_value:
            key_buffer_new, value_buffer_new, token_positions, kv_len = cat_on_buffers(
                key, value, key_buffer, value_buffer,
            )
            # Backward of SDPA
            grad_query, grad_key_buffer_new_indir, grad_value_buffer_new_indir = sdpa_backward(
                grad_attn_output=grad_attn_output,
                query=query,
                key=key_buffer_new,
                value=value_buffer_new,
                token_positions=token_positions,
                input_pos=kv_len,
                scale_factor=scale_factor,
                sliding_window_size=sliding_window_size,
                need_query=need_query,
                need_key=need_one_of_key,
                need_value=need_one_of_value,
                tmp_array_limit_gb=tmp_array_limit_gb,
            )
            if need_key_buffer:
                grad_key_buffer = (
                    grad_key_buffer_new[:, :, :kv_len, :] +
                    grad_key_buffer_new_indir[:, :, :kv_len, :]
                )
            if need_key:
                grad_key = (
                    grad_key_buffer_new[:, :, kv_len:, :] +
                    grad_key_buffer_new_indir[:, :, kv_len:, :]
                )
            if need_value_buffer:
                grad_value_buffer = (
                    grad_value_buffer_new[:, :, :kv_len, :] +
                    grad_value_buffer_new_indir[:, :, :kv_len, :]
                )
            if need_value:
                grad_value = (
                    grad_value_buffer_new[:, :, kv_len:, :] +
                    grad_value_buffer_new_indir[:, :, kv_len:, :]
                )

        return grad_query, grad_key, grad_value, grad_key_buffer, grad_value_buffer, *([None] * 4)
