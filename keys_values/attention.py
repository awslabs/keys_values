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
from typing import List, Optional, Tuple, Union

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
    slice_as_flat, pytorch_scaled_dot_product_attention,
)


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
        assert keys.shape[:-1] == values.shape[:-1] and keys.ndim == 4, (keys.shape, values.shape)
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

SDPA_IMPL_EAGER_BLOCKS = 1

SDPA_IMPL_EAGER_NO_BLOCKS = 2


class MultiHeadSelfAttention:
    """
    Maintains code for the inner part of multi-head self-attention which is not
    parameterized. This is used both by :class:`CausalSelfAttention` and by the
    default KV cache implementation :class:`DefaultKVCache`.

    Kernels to be used for SDPA can be restricted by `sdpa_kernels`. By
    default, the choice is down to the method itself. If GPU memory is a
    concern (e.g., if MHA is used in training mode, to compute gradients),
    `sdpa_kernels=SDPBackend.EFFICIENT_ATTENTION` is recommended.

    If `sdpa_kernels` is used, their availabilities are checked upon the
    first call, and a warning is printed if some are not available.

    If `use_eager_sdpa_always=True`,
    `torch.nn.functional.scaled_dot_product_attention` is never used.

    Usage of different kernels:

    There are different ways how SDPA is computed, see also
    :meth:`_use_eager_sdpa`:
    - 0: PyTorch kernel (see above).
    - 1: Eager (own) implementation, using blocking to limit GPU memory
        usage.
    - 2: Eager (own) implementation without blocking.

    PyTorch kernels are most efficient for the `is_causal=True` case
    (where queries and keys are over the same tokens), but do not return
    attention weights. While they can be called for `is_causal=False`, they
    require an explicit mask matrix to be passed, which can lead to OOM errors.
    We do not use them in this case. They also do not work with attention logit
    softcapping, or if `k_and_v` cannot return keys and values in parallel.

    Our eager implementation uses blocking in order to limit GPU memory
    usage (except for input and output arguments). This does not work
    with attention logit softcapping, or if `k_and_v` cannot return keys
    and values in parallel.

    """
    def __init__(
        self,
        config: Config,
        sdpa_kernels: Optional[Union[SDPBackend, List[SDPBackend]]] = None,
        use_eager_sdpa_always: bool = False,
        tmp_array_limit_gb: Optional[TemporaryArrayLimit] = None,
    ) -> None:
        self.config = config
        self._sdpa_kernels = sdpa_kernels
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
        device: torch.device,
    ) -> None:
        pass  # Currently, we don't use this

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
                attention weights (or scores) are returned as second argument
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
        B, _, T, _ = query.shape
        mask = None
        use_eager_sdpa = self._use_eager_sdpa(return_attn_weights, is_causal)
        if use_eager_sdpa != SDPA_IMPL_EAGER_BLOCKS:
            sdpa_is_eager = use_eager_sdpa != SDPA_IMPL_PYTORCH
            if sdpa_is_eager or sliding_window_size is not None or not is_causal:
                # Build attention mask
                mask_dtype = torch.float32 if sdpa_is_eager else query.dtype
                if is_causal:
                    mask = (
                        build_mask_cache(
                            max_seq_length=T,
                            sliding_window_size=sliding_window_size,
                            dtype=mask_dtype,
                            device=query.device,
                        )
                        .view(1, 1, T, T)
                        .detach()
                    )
                elif (not sdpa_is_eager) or T > 1:
                    # We need a mask if T > 1, since inference needs to be causal
                    # for the new tokens
                    mask = build_mask_slice(
                        input_pos=input_pos,
                        num=T,
                        token_positions=token_positions,
                        n_head=self.config.n_head,
                        dtype=mask_dtype,
                        sliding_window_size=sliding_window_size,
                    ).detach()

        y, scores = self.scaled_dot_product_attention(
            query=query,
            k_and_v=k_and_v,
            input_pos=input_pos if input_pos is not None else 0,
            token_positions=token_positions,
            sliding_window_size=sliding_window_size,
            mask=mask,
            return_attn_weights=return_attn_weights,
        )
        # Re-assemble all head outputs side by side.
        y = y.reshape(B, T, -1)
        return y, scores

    def _get_sliding_window_size(self, block_idx: int) -> Optional[int]:
        apply_sliding_window_attention = (
            self.config.sliding_window_size is not None and self.config.sliding_window_indices[block_idx] == 1
        )
        return self.config.sliding_window_size if apply_sliding_window_attention else None

    def _use_eager_sdpa(
        self,
        return_attn_weights: bool,
        is_causal: bool,
    ) -> int:
        """
        Decides on what SDPA implementation can be used, depending on
        arguments.

        Args:
            return_attn_weights: Attention weights have to be returned?
            is_causal: Causal case (queries, keys over same tokens)?

        Returns:
            Type of SDPA implementation: 0 (PyTorch kernel),
            1 (block-wise own implementation), 2 (own implementation
            without blocking; may result in OOM errors)

        """
        if self.config.attention_logit_softcapping is not None:
            return SDPA_IMPL_EAGER_NO_BLOCKS
        if return_attn_weights or self.use_eager_sdpa_always or not is_causal:
            return SDPA_IMPL_EAGER_BLOCKS
        else:
            return SDPA_IMPL_PYTORCH

    def _get_scale_factor(self):
        return 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)

    def tmp_array_limit_gb_value(self) -> Optional[float]:
        return None if self._tmp_array_limit_gb is None else self._tmp_array_limit_gb()

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        k_and_v: KeysAndValues,
        input_pos: int,
        token_positions: Optional[torch.Tensor],
        sliding_window_size: Optional[int],
        mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        scale_factor = self._get_scale_factor()
        # We cannot call PyTorch scaled_dot_product_attention if:
        # - Attention scores need to be returned; or
        # - Logit softcapping is required; or
        # - We cannot access keys and values from `k_and_v` in parallel
        use_eager_sdpa = self._use_eager_sdpa(return_attn_weights, input_pos == 0)
        if use_eager_sdpa != SDPA_IMPL_PYTORCH:
            if use_eager_sdpa == SDPA_IMPL_EAGER_NO_BLOCKS:
                assert mask is not None or query.shape[2] == 1
            y, scores = scaled_dot_product_attention(
                query=query,
                k_and_v=k_and_v,
                scale_factor=scale_factor,
                use_blocking=use_eager_sdpa == SDPA_IMPL_EAGER_BLOCKS,
                return_attn_weights=return_attn_weights,
                input_pos=input_pos,
                token_positions=token_positions,
                sliding_window_size=sliding_window_size,
                mask=mask,
                attention_logit_softcapping=self.config.attention_logit_softcapping,
                tmp_array_limit_gb=self.tmp_array_limit_gb_value(),
            )
            if not return_attn_weights:
                scores = None
        else:
            # We need `key` and `value` at the same time here. For the training
            # use case, this will be the case, since `k_and_v` is the default
            # in this case.
            y = pytorch_scaled_dot_product_attention(
                query=query,
                key=k_and_v.keys(),
                value=k_and_v.values(),
                scale_factor=scale_factor,
                sdpa_kernels=self.sdpa_kernels,
                mask=mask,
            )
            scores = None
        return y.transpose(1, 2), scores


def do_softcapping(x: torch.Tensor, thresh: Optional[float]) -> torch.Tensor:
    if thresh is not None:
        return torch.tanh(x / thresh) * thresh
    else:
        return x


def scaled_dot_product_attention(
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not use_blocking:
        dtype = query.dtype
        key = k_and_v.keys().to(torch.float32)
        query = query.to(torch.float32)
        scores = attention_compute_scores(query, key) * scale_factor
        scores = do_softcapping(scores, attention_logit_softcapping)
        if mask is not None:
            scores = scores + mask.to(torch.float32)
        scores = F.softmax(scores, dim=-1)
        value = k_and_v.values().to(torch.float32)
        result = attention_compute_weighted_values(scores, value).to(dtype)
        if return_attn_weights:
            _, n_head, q_len, _ = query.shape
            batch_size, n_query_groups, kv_len, _ = value.shape
            scores = scores.to(dtype)
            if n_head != n_query_groups:
                scores = scores.view(
                    batch_size, n_query_groups, -1, q_len, kv_len,
                ).mean(dim=2)
        else:
            scores = None
        return result, scores
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
    if return_attn_weights:
        # Note: Cannot use `torch.cat` to assemble from parts, this
        # gives OOM for large sizes
        attn_weights = torch.empty(
            (batch_size, n_query_groups, q_len, kv_len),
            dtype=dtype,
            device=device,
        )
    else:
        attn_weights = None
    start = 0
    for _ in range(num_splits):
        end = min(start + tmp_len, q_len)
        sz = end - start
        _input_pos = input_pos + start
        # Subfunctions assume these arrays are flat:
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
                        batch_size, n_query_groups, -1, sz, kv_len,
                    ),
                    dim=2,
                    out=source,
                )
            attn_weights[:, :, start:end, :] = source
        # Compute attention outputs part
        # - attn_weights_part (bs, nh_q, sz, kv_len)
        # - value32 (bs, nh_k, kv_len, hs)
        # - output_part (bs, nh_q, sz, hs)
        output_parts.append(
            attention_compute_weighted_values(
                scores=attn_weights_part, value=value32,
            ).to(dtype)
        )
        start = end

    # Combine
    output = torch.cat(output_parts, dim=2)
    # Sanity check:
    assert output.shape == (batch_size, n_head, q_len, value32.shape[-1]), (output.shape, (batch_size, n_head, q_len, value32.shape[-1]))
    return output, attn_weights
