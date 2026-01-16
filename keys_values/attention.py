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
from typing import List, Optional, Tuple, Union, Callable

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend

from litgpt.config import Config

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.attention_utils import (
    attention_compute_scores,
    attention_compute_weighted_values,
    build_mask_cache,
    build_mask_slice,
    create_temp_array,
    sdpa_attention_weights,
    slice_as_flat,
    pytorch_scaled_dot_product_attention,
)
from keys_values.pos_encoding import position_encoding_factory, PositionEncoding
from keys_values.sdpa_wrapper import scaled_dot_product_attention as qpadded_sdpa


class KeysAndValues:
    """
    Object passed to :meth:`MultiHeadSelfAttention.__call__`. Allows to access
    keys and values.

    """

    def keys(self) -> torch.Tensor:
        """
        Returns:
            keys tensor, shape `(batch_size, n_query_groups, T, head_size)`,
            where `T <= cache_length` is the current cache length)

        """
        raise NotImplementedError()

    def values(self) -> torch.Tensor:
        """
        Returns:
            values tensor, shape `(batch_size, n_query_groups, T, head_size)`,
            where `T <= cache_length` is the current cache length)

        """
        raise NotImplementedError()


class DefaultKeysAndValues(KeysAndValues):
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        # The final dimension of K and V can be different (in general)
        assert keys.shape[:-1] == values.shape[:-1] and keys.ndim == 4, (
            keys.shape,
            values.shape,
        )
        self._keys = keys
        self._values = values

    def keys(self) -> torch.Tensor:
        return self._keys

    def values(self) -> torch.Tensor:
        return self._values

    def clear(self):
        if self._keys is not None:
            del self._keys
            self._keys = None
            del self._values
            self._values = None


SDPA_IMPL_PYTORCH = 0

SDPA_IMPL_QPADDED_PYTORCH = 1

SDPA_IMPL_EAGER_BLOCKS = 2

SDPA_IMPL_EAGER_NO_BLOCKS = 3


UseEagerPredicate = Callable[[int, int], bool]


class MultiHeadSelfAttention:
    """
    Maintains code for the inner part of multi-head self-attention which is not
    parameterized. This is used both by :class:`CausalSelfAttention` and by the
    default KV cache implementation :class:`DefaultKVCache`.

    Kernels to be used for SDPA can be restricted by `sdpa_kernels`. By
    default, the choice is down to the method itself. If you supply a list,
    they are tried in that order, the first supported one is used.
    "Supported" depends on the arguments of calling SDPA with.

    If `filter_sdpa_kernels` is `True`, the kernels in `sdpa_kernels` are
    filtered in the first call, removing those which are not supported for
    the calls here. In subsequent calls, the filtered list is used.

    NOTE: `filter_sdpa_kernels=True` can lead to problems with `torch.compile`.
    In this case, set `filter_sdpa_kernels=False`.

    If `use_eager_sdpa_always=True`,
    `torch.nn.functional.scaled_dot_product_attention` is never used. Use
    this for debugging or testing only.

    Usage of different kernels:

    There are different ways how SDPA is computed, see also
    :meth:`_sdpa_mode`:
    - :const:`SDPA_IMPL_PYTORCH`: PyTorch SDPA kernel (see above). Only
        used for prefill (`input_pos=0`). This is because this is all
        PyTorch SDPA reliably supports at the moment (their C++
        default implementation is quite useless).
    - :const:`SDPA_IMPL_QPADDED_PYTORCH: Variant of PyTorch kernel, where
        `query` is zero-padded. This is implemented in
        :func:`sdpa_wrapper.scaled_dot_product_attention`. Cannot be used
        if attention weights are required.
    - :const:`SDPA_IMPL_EAGER_BLOCKS`: Eager (own) implementation, using
        blocking to limit GPU memory usage.
    - :const:`SDPA_IMPL_EAGER_NO_BLOCKS: Eager (own) implementation without
        blocking. This is the worst, it is chosen only if
        `config.attention_logit_softcapping` is not `None` (because I
        can't be bothered to extend the blocking implementation to this
        case).

    PyTorch kernels are most efficient for the `is_causal=True` case
    (where queries and keys are over the same tokens), but do not return
    attention weights. While they can be called for `is_causal=False`, they
    require an explicit mask matrix to be passed, which can lead to OOM errors.
    We do not use them in this case.

    If attention weights are not needed, the decision between
    :const:`SDPA_IMPL_EAGER_BLOCKS` and :const:`SDPA_IMPL_QPADDED_PYTORCH` is
    taken based on `use_eager_kernel(kv_len, q_len)`. If this returns `True`,
    we use :const:`SDPA_IMPL_EAGER_BLOCKS`, otherwise we use
    :const:`SDPA_IMPL_QPADDED_PYTORCH`. The rule should in general be of the
    form:
    ```
        use_eager_kernel(kv_len, q_len) = q_len <= thresh(kv_len)
    ```
    This is because :const:`SDPA_IMPL_QPADDED_PYTORCH` is faster from a
    certain value of `q_len` upwards. It also requires less temporary
    memory.

    Look at :class:`DefaultUseEagerKernel` for choosing `use_eager_kernel`.

    """

    def __init__(
        self,
        config: Config,
        pos_encoding: Optional[PositionEncoding] = None,
        sdpa_kernels: Optional[Union[SDPBackend, List[SDPBackend]]] = None,
        use_eager_sdpa_always: bool = False,
        tmp_array_limit_gb: Optional[TemporaryArrayLimit] = None,
        use_eager_kernel: Optional[UseEagerPredicate] = None,
        filter_sdpa_kernels: bool = True,
    ) -> None:
        self.config = config
        if pos_encoding is None:
            pos_encoding = position_encoding_factory(config)
        self.pos_encoding = pos_encoding
        self._sdpa_kernels = sdpa_kernels
        self._do_filter_kernels = filter_sdpa_kernels
        self.use_eager_sdpa_always = use_eager_sdpa_always
        self.set_tmp_array_limit_gb(tmp_array_limit_gb)
        if self.config.attention_logit_softcapping is not None:
            print(
                "Your model uses attention logit softcapping "
                "(config.attention_logit_softcapping != None). Time or memory "
                "efficient implementations of SDPA cannot be used, you may run "
                "out of GPU memory. Consider using a model without attention "
                "logit softcapping."
            )
        if use_eager_kernel is None and not use_eager_sdpa_always:
            # This is a good choice for `kv_len = 32768`
            use_eager_kernel = lambda kv_len, q_len: q_len < 512
        self._use_eager_kernel = use_eager_kernel

    @property
    def sdpa_kernels(self) -> Union[SDPBackend, List[SDPBackend]]:
        return self._sdpa_kernels if self._sdpa_kernels is not None else []

    @property
    def tmp_array_limit_gb(self) -> Optional[TemporaryArrayLimit]:
        return self._tmp_array_limit_gb

    def set_tmp_array_limit_gb(self, limit: Optional[TemporaryArrayLimit]):
        if limit is not None:
            assert isinstance(limit, TemporaryArrayLimit)
        self._tmp_array_limit_gb = limit

    def set_seq_length(
        self,
        value: int,
    ) -> None:
        self.pos_encoding.set_context_width(value)

    def __call__(
        self,
        query: torch.Tensor,
        k_and_v: KeysAndValues,
        block_idx: int,
        input_pos: Optional[int] = None,
        return_attn_weights: bool = False,
        token_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: Queries, shape `(batch_size, n_heads, q_len, head_size)`
            k_and_v: Access to keys and values, shape
                (batch_size, n_query_groups, kv_len, head_size)`
            block_idx: Index of block (or layer) in model
            input_pos: Position in input sequence. Defaults to 0
            return_attn_weights: If this is `True` and `input_pos > 0`, the
                attention weights (summed over query axis) are returned as
                second argument. Here, `attn_weights.dtype=float32` independent
                of `query.dtype`.
            token_positions: Required if `input_pos > 0`. Contains token
                positions in KV cache. This is needed to select the correct
                part of the mask matrix

        Returns:
            `attn_output, attn_weights`, where `attn_weights` is `None` if
            attention weights are not returned.

        """
        # We need the attention mask if there is sliding window attention.
        # Our eager implementation with blocking has masking built in, so
        # a mask is not needed then.
        for_prefill = input_pos == 0
        is_causal = input_pos is None or for_prefill
        if not is_causal and token_positions is None:
            raise ValueError("token_positions must be given if input_pos > 0")
        sliding_window_size = self._get_sliding_window_size(block_idx)
        batch_size, _, seq_length, _ = query.shape
        mask = None
        sdpa_mode = self._sdpa_mode(
            return_attn_weights=return_attn_weights,
            is_causal=is_causal,
            q_len=query.shape[2],
            kv_len=k_and_v.keys().shape[2],
            sliding_window_size=sliding_window_size,
        )
        if sdpa_mode in (SDPA_IMPL_PYTORCH, SDPA_IMPL_EAGER_NO_BLOCKS):
            sdpa_is_eager = sdpa_mode == SDPA_IMPL_EAGER_NO_BLOCKS
            if sdpa_is_eager or sliding_window_size is not None or not is_causal:
                # Build attention mask
                mask_dtype = torch.float32 if sdpa_is_eager else query.dtype
                if is_causal:
                    mask = (
                        build_mask_cache(
                            max_seq_length=seq_length,
                            sliding_window_size=sliding_window_size,
                            dtype=mask_dtype,
                            device=query.device,
                        )
                        .view(1, 1, seq_length, seq_length)
                        .detach()
                    )
                elif (not sdpa_is_eager) or seq_length > 1:
                    # We need a mask if T > 1, since inference needs to be causal
                    # for the new tokens
                    mask = build_mask_slice(
                        input_pos=input_pos,
                        num=seq_length,
                        token_positions=token_positions,
                        n_head=self.config.n_head,
                        dtype=mask_dtype,
                        sliding_window_size=sliding_window_size,
                    ).detach()

        attn_outputs, attn_weights = self.scaled_dot_product_attention(
            query=query,
            k_and_v=k_and_v,
            input_pos=input_pos if input_pos is not None else 0,
            token_positions=token_positions,
            sdpa_mode=sdpa_mode,
            sliding_window_size=sliding_window_size,
            mask=mask,
            return_attn_weights=return_attn_weights,
        )
        # Re-assemble all head outputs side by side.
        attn_outputs = attn_outputs.reshape(batch_size, seq_length, -1)
        return attn_outputs, attn_weights

    def _get_sliding_window_size(self, block_idx: int) -> Optional[int]:
        apply_sliding_window_attention = (
            self.config.sliding_window_size is not None
            and self.config.sliding_window_indices[block_idx] == 1
        )
        return (
            self.config.sliding_window_size if apply_sliding_window_attention else None
        )

    def _sdpa_mode(
        self,
        return_attn_weights: bool,
        is_causal: bool,
        q_len: int,
        kv_len: int,
        sliding_window_size: Optional[int],
    ) -> int:
        """
        Decides on what SDPA implementation can be used, depending on
        arguments.

        Args:
            return_attn_weights: Attention weights have to be returned?
            is_causal: Causal case (queries, keys over same tokens)?
            q_len: Length of queries
            kv_len: Length of keys, values

        Returns:
            Type of SDPA implementation: See `SDPA_IMPL_*` constants

        """
        if self.config.attention_logit_softcapping is not None:
            return SDPA_IMPL_EAGER_NO_BLOCKS
        must_eager = return_attn_weights or self.use_eager_sdpa_always
        if must_eager or not is_causal:
            if (
                must_eager
                or sliding_window_size is not None
                or self._use_eager_kernel(kv_len, q_len)
            ):
                return SDPA_IMPL_EAGER_BLOCKS
            else:
                return SDPA_IMPL_QPADDED_PYTORCH
        else:
            return SDPA_IMPL_PYTORCH

    def get_scale_factor(self):
        return self.pos_encoding.sdpa_scale_factor()

    def tmp_array_limit_gb_value(self) -> Optional[float]:
        return None if self._tmp_array_limit_gb is None else self._tmp_array_limit_gb()

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        k_and_v: KeysAndValues,
        input_pos: int,
        token_positions: Optional[torch.Tensor],
        sdpa_mode: Optional[int],
        sliding_window_size: Optional[int],
        mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
        transpose_result: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        scale_factor = self.get_scale_factor()
        # We cannot call PyTorch scaled_dot_product_attention if:
        # - Attention scores need to be returned; or
        # - Logit softcapping is required; or
        # - We cannot access keys and values from `k_and_v` in parallel
        attn_weights = None
        if sdpa_mode is None:
            sdpa_mode = self._sdpa_mode(
                return_attn_weights=return_attn_weights,
                is_causal=input_pos == 0,
                q_len=query.shape[2],
                kv_len=k_and_v.keys().shape[2],
                sliding_window_size=sliding_window_size,
            )
        if sdpa_mode == SDPA_IMPL_QPADDED_PYTORCH:
            attn_outputs, filtered_kernels = qpadded_sdpa(
                query=query,
                key=k_and_v.keys(),
                value=k_and_v.values(),
                scale_factor=scale_factor,
                input_pos=input_pos,
                token_positions=token_positions,
                sdpa_kernels=self.sdpa_kernels,
                do_filter_kernels=self._do_filter_kernels,
            )
            if self._do_filter_kernels:
                self._do_filter_kernels = False
                self._sdpa_kernels = filtered_kernels
        elif sdpa_mode == SDPA_IMPL_PYTORCH:
            # We need `key` and `value` at the same time here. For the training
            # use case, this will be the case, since `k_and_v` is the default
            # in this case.
            attn_outputs, filtered_kernels = pytorch_scaled_dot_product_attention(
                query=query,
                key=k_and_v.keys(),
                value=k_and_v.values(),
                scale_factor=scale_factor,
                sdpa_kernels=self.sdpa_kernels,
                do_filter_kernels=self._do_filter_kernels,
                mask=mask,
            )
            if self._do_filter_kernels:
                self._do_filter_kernels = False
                self._sdpa_kernels = filtered_kernels
        else:
            use_blocking = sdpa_mode == SDPA_IMPL_EAGER_BLOCKS
            if not use_blocking:
                assert mask is not None or query.shape[2] == 1
            attn_outputs, attn_weights = eager_scaled_dot_product_attention(
                query=query,
                k_and_v=k_and_v,
                scale_factor=scale_factor,
                use_blocking=use_blocking,
                return_attn_weights=return_attn_weights,
                input_pos=input_pos,
                token_positions=token_positions,
                sliding_window_size=sliding_window_size,
                mask=mask,
                attention_logit_softcapping=self.config.attention_logit_softcapping,
                tmp_array_limit_gb=self.tmp_array_limit_gb_value(),
            )
        if transpose_result:
            attn_outputs = attn_outputs.transpose(1, 2)
        return attn_outputs, attn_weights


def do_softcapping(x: torch.Tensor, thresh: Optional[float]) -> torch.Tensor:
    if thresh is not None:
        return torch.tanh(x / thresh) * thresh
    else:
        return x


def eager_scaled_dot_product_attention(
    query: torch.Tensor,
    k_and_v: KeysAndValues,
    scale_factor: float,
    use_blocking: bool,
    return_attn_weights: bool,
    input_pos: int,
    token_positions: Optional[torch.Tensor],
    sliding_window_size: Optional[int],
    mask: Optional[torch.Tensor] = None,
    attention_logit_softcapping: Optional[float] = None,
    tmp_array_limit_gb: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not use_blocking:
        dtype = query.dtype
        n_head = query.shape[1]
        query32 = query.to(torch.float32)
        key32 = k_and_v.keys().to(torch.float32)
        attn_weights = attention_compute_scores(query32, key32) * scale_factor
        attn_weights = do_softcapping(attn_weights, attention_logit_softcapping)
        if mask is not None:
            attn_weights = attn_weights + mask.to(torch.float32)
        attn_weights = F.softmax(attn_weights, dim=-1)
        del query32, key32
        value32 = k_and_v.values().to(torch.float32)
        result = attention_compute_weighted_values(attn_weights, value32).to(dtype)
        if return_attn_weights:
            batch_size, n_query_groups, kv_len, _ = value32.shape
            attn_weights = attn_weights.sum(dim=2)
            if n_head != n_query_groups:
                attn_weights = attn_weights.view(
                    batch_size,
                    n_query_groups,
                    -1,
                    kv_len,
                ).mean(dim=2)
        else:
            attn_weights = None
        return result, attn_weights
    else:
        assert attention_logit_softcapping is None  # Sanity check
        return scaled_dot_product_attention_in_blocks(
            query=query,
            k_and_v=k_and_v,
            scale_factor=scale_factor,
            return_attn_weights=return_attn_weights,
            input_pos=input_pos,
            token_positions=token_positions,
            sliding_window_size=sliding_window_size,
            tmp_array_limit_gb=tmp_array_limit_gb,
        )


def scaled_dot_product_attention_in_blocks(
    query: torch.Tensor,
    k_and_v: KeysAndValues,
    scale_factor: float,
    return_attn_weights: bool,
    input_pos: int,
    token_positions: Optional[torch.Tensor],
    sliding_window_size: Optional[int],
    tmp_array_limit_gb: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    device = query.device
    dtype = query.dtype
    key32 = k_and_v.keys().to(torch.float32)
    value32 = k_and_v.values().to(torch.float32)
    # Allocate temporary arrays and determine the number of slices.
    batch_size, n_head, q_len, _ = query.shape
    _, n_query_groups, kv_len, _ = key32.shape
    tmp_array, num_splits, tmp_len = create_temp_array(
        batch_size=batch_size,
        n_head=n_head,
        q_len=q_len,
        kv_len=kv_len,
        device=device,
        tmp_array_limit_gb=tmp_array_limit_gb,
    )
    # Iterate over slices along `q_len` dimension
    output_parts = []
    attn_weights = 0
    start = 0
    for _ in range(num_splits):
        end = min(start + tmp_len, q_len)
        sz = end - start
        _input_pos = input_pos + start
        # Functions assume these arrays are flat:
        _tmp_array = slice_as_flat(tmp_array, sz)
        # Attention weights -> `attn_weights_part`
        # Note: This creates a new matrix `attn_weights_part`, but this is
        # much faster than a in-place operation
        attn_weights_part = sdpa_attention_weights(
            query=query[:, :, start:end, :].to(torch.float32),
            key=key32,
            tmp_array=_tmp_array,
            token_positions=token_positions,
            input_pos=_input_pos,
            scale_factor=scale_factor,
            sliding_window_size=sliding_window_size,
        )
        if return_attn_weights:
            if n_head == n_query_groups:
                source = attn_weights_part
            else:
                # Need to average. Make sure to not create another
                # temporary array
                source = _tmp_array[:, :n_query_groups, :, :]
                torch.mean(
                    attn_weights_part.view(
                        batch_size,
                        n_query_groups,
                        -1,
                        sz,
                        kv_len,
                    ),
                    dim=2,
                    out=source,
                )
            attn_weights = source.sum(dim=2) + attn_weights
        # Compute attention outputs part
        # - attn_weights_part (bs, nh_q, sz, kv_len)
        # - value32 (bs, nh_k, kv_len, hs)
        # - output_part (bs, nh_q, sz, hs)
        output_parts.append(
            attention_compute_weighted_values(
                scores=attn_weights_part,
                value=value32,
            ).to(dtype)
        )
        start = end

    # Combine
    output = torch.cat(output_parts, dim=2)
    if not return_attn_weights:
        attn_weights = None
    return output, attn_weights
