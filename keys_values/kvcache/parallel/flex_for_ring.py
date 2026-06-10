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
import math
from typing import Optional, List, Tuple, Any

import torch
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    BlockMask,
    AuxRequest,
)

from keys_values.attention.flex_attention import (
    FlexAttnForPrefillManager,
    FlexAttnForChunkManager,
    FlexAttnWithBlockMask,
)
from keys_values.attention.sdpa_wrapper import (
    sdpa_check_args,
    zeropad_query,
)
from keys_values.utils import repeat_interleave


def causal_mask_for_prefill(
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    offset_positive: bool,
) -> torch.Tensor:
    return kv_idx < q_idx + int(offset_positive)


class RingFlexAttnForPrefillManager(FlexAttnForPrefillManager):
    """
    RingAttention requires to compute MHA for cells `(r, s)`, `r` looping
    over query parts, `s` over key-value parts. If `r == s`, this is
    standard SDPA with causal masking (after reordering keys and values),
    but for `r != s`, the causal masking is special. This case is implemented
    here. More details are given in the technical report.
    """

    def __init__(self):
        super().__init__()

    def _unpack_extra_kwargs(
        self,
        **kwargs,
    ) -> Tuple[int, bool]:
        unpacked_args = []
        for name in ("offset_positive",):
            if name not in kwargs:
                raise ValueError(f"{name} is required")
            unpacked_args.append(kwargs[name])
        return tuple(unpacked_args)

    def _get_args(
        self,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
        requires_grad: bool,
        sliding_window_size: Optional[int],
        als_signature: Optional[int],
        **kwargs,
    ) -> tuple:
        for val, name in (
            (sliding_window_size, "sliding_window_size"),
            (als_signature, "als_signature"),
        ):
            if val is not None:
                raise NotImplementedError(f"{name} not supported")
        if requires_grad:
            raise NotImplementedError("requires_grad=True not supported")
        result = super()._get_args(
            kv_len,
            device,
            dtype,
            requires_grad,
            sliding_window_size,
            als_signature,
            **kwargs,
        )
        offset_positive = self._unpack_extra_kwargs(**kwargs)
        return result + (offset_positive,)

    _ARGS_NAMES = {
        **FlexAttnForPrefillManager._ARGS_NAMES,
        "offset_positive": 6,
    }

    @staticmethod
    def _from_args(args: tuple, name: str) -> Any:
        return args[RingFlexAttnForPrefillManager._ARGS_NAMES[name]]

    def _args_to_str(self, *args) -> str:
        parts = [
            super()._args_to_str(*args),
            f"offset_positive:{int(self._from_args(args, 'offset_positive')):1d}",
        ]
        return ",".join(parts)

    def _create_block_mask(
        self,
        kv_len: int,
        device: torch.device,
        sliding_window_size: Optional[int],
        **kwargs,
    ) -> BlockMask:
        if sliding_window_size is not None:
            raise NotImplementedError("sliding_window_size not supported")
        offset_positive = self._unpack_extra_kwargs(**kwargs)
        mask_mod = partial(
            causal_mask_for_prefill,
            offset_positive=offset_positive,
        )
        return create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=kv_len,
            KV_LEN=kv_len,
            device=device,
        )


def causal_mask_for_chunk(
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    thresh: int,
    offset_positive: bool,
) -> torch.Tensor:
    """
    In the technical report, `q_idx` and `kv_idx` map to `i` and `j`, `thresh`
    is called `\bar{M}(s)`, and `offset_positive` is `True` iff
    `mod(r + U, N) − mod(s + U, N) > 0`. A position is filtered out (return
    `False` here) iff (1) and (2), where:
    - (1): j >= thresh
    - (2): j - thresh > i if `offset_positive == True`, j - thresh >= i otherwise

    """
    right_arg = kv_idx - thresh  # j - thresh
    return torch.logical_or(
        right_arg < 0,
        right_arg < q_idx + int(offset_positive),
    )


class RingFlexAttnForChunkManager(FlexAttnForChunkManager):
    """
    RingAttention requires to compute MHA for cells `(r, s)`, `r` looping
    over query parts, `s` over key-value parts. If `r == s`, this is
    standard SDPA with causal masking (after reordering keys and values),
    but for `r != s`, the causal masking is special. This case is implemented
    here. More details are given in the technical report.
    """

    def __init__(self, q_lens: Optional[List[int]] = None):
        super().__init__(q_lens, forward_return_lse=True)

    def _unpack_extra_kwargs(
        self,
        **kwargs,
    ) -> Tuple[int, bool]:
        unpacked_args = []
        for name in (
            "thresh",
            "offset_positive",
        ):
            if name not in kwargs:
                raise ValueError(f"{name} is required")
            unpacked_args.append(kwargs[name])
        return tuple(unpacked_args)

    def _get_args(
        self,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
        requires_grad: bool,
        sliding_window_size: Optional[int],
        als_signature: Optional[int],
        **kwargs,
    ) -> tuple:
        for val, name in (
            (sliding_window_size, "sliding_window_size"),
            (als_signature, "als_signature"),
        ):
            if val is not None:
                raise NotImplementedError(f"{name} not supported")
        if requires_grad:
            raise NotImplementedError("requires_grad=True not supported")
        result = super()._get_args(
            kv_len,
            device,
            dtype,
            requires_grad,
            sliding_window_size,
            als_signature,
            **kwargs,
        )
        thresh, offset_positive = self._unpack_extra_kwargs(**kwargs)
        return result + (thresh, offset_positive)

    _ARGS_NAMES = {
        **FlexAttnForChunkManager._ARGS_NAMES,
        "thresh": 10,
        "offset_positive": 11,
    }

    @staticmethod
    def _from_args(args: tuple, name: str) -> Any:
        return args[RingFlexAttnForChunkManager._ARGS_NAMES[name]]

    def _args_to_str(self, *args) -> str:
        parts = [
            super()._args_to_str(*args),
            f"thresh:{self._from_args(args, 'thresh'):5d}",
            f"offset_positive:{int(self._from_args(args, 'offset_positive')):1d}",
        ]
        return ",".join(parts)

    def _create_block_mask(
        self,
        kv_len: int,
        device: torch.device,
        sliding_window_size: Optional[int],
        **kwargs,
    ) -> BlockMask:
        if sliding_window_size is not None:
            raise NotImplementedError("sliding_window_size not supported")
        q_len, _, _, reverse = self._unpack_kwargs(**kwargs)
        if reverse:
            raise NotImplementedError("reverse=True not supported")
        thresh, offset_positive = self._unpack_extra_kwargs(**kwargs)
        q_len = self.transform_q_len(q_len)
        if q_len > kv_len:
            raise ValueError(
                f"q_len={q_len}, kv_len={kv_len}: Must have q_len <= kv_len"
            )
        mask_mod = partial(
            causal_mask_for_chunk,
            thresh=thresh,
            offset_positive=offset_positive,
        )
        return create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )


class RingOffdiagFlexAttentionArgs:
    """
    Maintains managers (for prefill and chunk computations) for the offdiagonal
    case `rank_r != rank_s`.

    Using `q_lens` is strongly recommended. You can use :func:`choose_q_lens`.

    Note: If `q_lens` is used, the expression returned by :meth:`attn_fn` for
    `input_pos > 0` must be used with `query` tensors of length
    `transform_q_len(q_len)`, which may be larger than `q_len`. The padding
    must be done by the caller.

    Args:
        num_devices: Number of devices. Ranks are in `range(num_devices)`.
        q_lens: If given, this is a list of `q_len` chunk lengths for which
            graphs are created. Any `q_len` passed to :meth:`attn_fn` with
            `input_pos > 0` is mapped to the smallest entry `>= q_len`.
        extend_kv: If `True` we always extend `key, value` to `n_head`. This
            needs more memory, only use if the default does not work.
    """

    def __init__(
        self,
        num_devices: int,
        q_lens: Optional[List[int]] = None,
        extend_kv: bool = False,
    ):
        self.attn_prefill_manager = RingFlexAttnForPrefillManager()
        self.attn_chunk_manager = RingFlexAttnForChunkManager(q_lens)
        self.extend_kv = extend_kv
        self.num_devices = num_devices

    def attn_fn(
        self,
        q_len: int,
        kv_len: int,
        batch_size: int,
        n_head: int,
        device: torch.device,
        dtype: torch.dtype,
        input_pos: int,
        rank_r: int,
        rank_s: int,
        q_len_for_s: int,
    ) -> Tuple[FlexAttnWithBlockMask, bool]:
        """
        We need to determine `thresh` and `offset_positive`. From the
        technical report, `thresh = kv_len - q_len_for_s`, where
        `M(s) = q_len_for_s`, and `offset_positive` is `True` iff
        `mod(rank_r + U, num_devices) − mod(rank_s + U, num_devices) > 0`.
        Also, `M(r) = q_len` and `P = input_pos`.

        """
        if not (0 <= rank_r < self.num_devices):
            raise ValueError(f"rank_r={rank_r}, must be in [0, {self.num_devices})")
        if not (0 <= rank_s < self.num_devices):
            raise ValueError(f"rank_s={rank_s}, must be in [0, {self.num_devices})")
        if rank_r == rank_s:
            raise ValueError(
                "Must have rank_r != rank_s. Use FlexAttentionArgs if they are the same"
            )
        if input_pos == 0:
            offset_positive = rank_r > rank_s
            return (
                self.attn_prefill_manager(
                    kv_len=kv_len,
                    device=device,
                    dtype=dtype,
                    requires_grad=False,
                    sliding_window_size=None,
                    attention_logit_softcapping=None,
                    offset_positive=offset_positive,
                )[0],
                False,
            )
        else:
            ndevs = self.num_devices  # N
            u_val = (ndevs - (input_pos % ndevs)) % ndevs  # U
            offset_positive = ((rank_r + u_val) % ndevs) > ((rank_s + u_val) % ndevs)
            thresh = kv_len - q_len_for_s
            _attn_fn, extend_kv = self.attn_chunk_manager(
                kv_len=kv_len,
                device=device,
                dtype=dtype,
                requires_grad=False,
                sliding_window_size=None,
                attention_logit_softcapping=None,
                q_len=q_len,
                batch_size=batch_size,
                n_head=n_head,
                reverse=False,
                thresh=thresh,
                offset_positive=offset_positive,
            )
            return _attn_fn, extend_kv or self.extend_kv

    def transform_q_len(self, q_len: int) -> int:
        return self.attn_chunk_manager.transform_q_len(q_len)

    def report(self) -> str:
        parts = []
        if self.attn_prefill_manager.num_hits:
            parts.extend(
                [
                    "RingFlexAttention: attn_fn graphs and usage for prefill:",
                    self.attn_prefill_manager.report(),
                ]
            )
        if self.attn_chunk_manager.num_hits:
            parts.extend(
                [
                    "RingFlexAttention: attn_fn graphs and usage for chunks:",
                    self.attn_chunk_manager.report(),
                ]
            )
        return "\n".join(parts)


class RingDiagFlexAttentionArgs:
    """
    Maintains managers (for prefill and chunk computations) for the diagonal
    case `rank_r == rank_s`. This is the same as treated by
    :class:`FlexAttentionArgs`, but the log_sum_exp array is returned as well.

    Using `q_lens` is strongly recommended. You can use :func:`choose_q_lens`.

    Note: If `q_lens` is used, the expression returned by :meth:`attn_fn` for
    `input_pos > 0` must be used with `query` tensors of length
    `transform_q_len(q_len)`, which may be larger than `q_len`. The padding
    must be done by the caller.

    Args:
        q_lens: If given, this is a list of `q_len` chunk lengths for which
            graphs are created. Any `q_len` passed to :meth:`attn_fn` with
            `input_pos > 0` is mapped to the smallest entry `>= q_len`.
        extend_kv: If `True` we always extend `key, value` to `n_head`. This
            needs more memory, only use if the default does not work.
    """

    def __init__(
        self,
        q_lens: Optional[List[int]] = None,
        extend_kv: bool = False,
    ):
        self.attn_prefill_manager = FlexAttnForPrefillManager(forward_return_lse=True)
        self.attn_chunk_manager = FlexAttnForChunkManager(
            q_lens, forward_return_lse=True
        )
        self.extend_kv = extend_kv

    def attn_fn(
        self,
        q_len: int,
        kv_len: int,
        batch_size: int,
        n_head: int,
        device: torch.device,
        dtype: torch.dtype,
        input_pos: int,
    ) -> Tuple[FlexAttnWithBlockMask, bool]:
        if input_pos == 0:
            return (
                self.attn_prefill_manager(
                    kv_len=kv_len,
                    device=device,
                    dtype=dtype,
                    requires_grad=False,
                    sliding_window_size=None,
                    attention_logit_softcapping=None,
                )[0],
                False,
            )
        else:
            _attn_fn, extend_kv = self.attn_chunk_manager(
                kv_len=kv_len,
                device=device,
                dtype=dtype,
                requires_grad=False,
                sliding_window_size=None,
                attention_logit_softcapping=None,
                q_len=q_len,
                batch_size=batch_size,
                n_head=n_head,
                reverse=False,
            )
            return _attn_fn, extend_kv or self.extend_kv

    def transform_q_len(self, q_len: int) -> int:
        return self.attn_chunk_manager.transform_q_len(q_len)

    def report(self) -> str:
        parts = []
        if self.attn_prefill_manager.num_hits:
            parts.extend(
                [
                    "FlexAttention: attn_fn graphs and usage for prefill:",
                    self.attn_prefill_manager.report(),
                ]
            )
        if self.attn_chunk_manager.num_hits:
            parts.extend(
                [
                    "FlexAttention: attn_fn graphs and usage for chunks:",
                    self.attn_chunk_manager.report(),
                ]
            )
        return "\n".join(parts)


def sdpa_ring_flexatt_offdiag(
    flexatt_args: RingOffdiagFlexAttentionArgs,
    rank_r: int,
    rank_s: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: Optional[float],
    input_pos: int,
    q_len_for_s: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes scaled dot product attention (SDPA) part for one cell
    `(rank_r, rank_s)` in RingAttention. Here, `rank_r != rank_s` (use
    `keys_values.attention.flex_attention.scaled_dot_product_attention_flexatt`
    for the standard case `rank_r == rank_s`. Here, `query` is for `rank_r`,
    while `key, value` is for `rank_s`.

    `key, value` must already have been updated and reordered, so that the new
    entries are on the right end (final `q_len_for_s` slots).

    Note: `flex_attention` is always called with `scale=None`, the default
    scale factor. If `scale_factor` is different, we multiply a factor into
    `query`. This avoids bogus graph re-compilations due to small numerical
    differences in `scale`.

    Padding of query: In general, we pad `query` to the nearest length in
    `flexatt_args.attn_chunk_manager.q_lens`, so that only a small number of
    graphs need to be maintained. In the standard case `rank_r == rank_s`,
    padding is done on the left, because the mask assumes `query` is
    right-aligned with `keys, values`. Here, we use padding on the right,
    because this works better with how the masks are defined here. Padding is
    done internally here.

    Args:
        flexatt_args: Arguments for `flex_attention`. Most important are the
            managers, see :class:`RingFlexAttnForPrefillManager` and
            :class:`RingFlexAttnForChunkManager`.
        rank_r: Device rank for query
        rank_s: Device rank for key, value
        query: Queries, shape `(batch_size, n_head, q_len, head_size)`
        key: Keys, shape `(batch_size, n_query_groups, kv_len, head_size)`
        value: Values, shape `(batch_size, n_query_groups, kv_len, head_size)`
        scale_factor: Scale factor for attention
        input_pos: Position in input sequence
        q_len_for_s: Number of new slots (or length of query) for rank `rank_s`.
            Need not be the same as `q_len = query.shape[2]` (at most 1
            different).

    Returns:
        `(output, lse)`, where `output` are attention outputs, shape
        `(batch_size, n_head, q_len, head_size)`, `lse` are the log-sum-exp
        values needed for RingAttention, shape `(batch_size, n_head, q_len)`.

    """
    batch_size, n_head, n_query_groups, q_len, kv_len, head_size = sdpa_check_args(
        query,
        key,
        value,
    )
    if not (0 <= rank_r < flexatt_args.num_devices):
        raise ValueError(f"rank_r={rank_r}, must be in [0, {flexatt_args.num_devices})")
    if not (0 <= rank_s < flexatt_args.num_devices):
        raise ValueError(f"rank_s={rank_s}, must be in [0, {flexatt_args.num_devices})")
    if rank_r == rank_s:
        raise ValueError(
            "Must have rank_r != rank_s. Use scaled_dot_product_attention_flexatt if they are the same"
        )
    if input_pos == 0:
        if q_len != kv_len:
            raise ValueError(
                f"For input_pos=0, must have q_len == kv_len, but have q_len = {q_len}, kv_len = {kv_len}"
            )
    enable_gqa = n_query_groups < n_head
    attn_fn, extend_kv = flexatt_args.attn_fn(
        q_len=q_len,
        kv_len=kv_len,
        batch_size=batch_size,
        n_head=n_head,
        device=query.device,
        dtype=query.dtype,
        input_pos=input_pos,
        rank_r=rank_r,
        rank_s=rank_s,
        q_len_for_s=q_len_for_s,
    )
    if enable_gqa and extend_kv:
        key = repeat_interleave(key, n_head)
        value = repeat_interleave(value, n_head)
        enable_gqa = False
        extend_kv = True
    else:
        extend_kv = False
    q_len_tr = flexatt_args.transform_q_len(q_len)
    if q_len_tr > q_len:
        # Use zero padding
        # Different to `scaled_dot_product_attention_flexatt`, padding is done
        # on the right here, since this works better with how masking is done
        # here.
        query = zeropad_query(query, q_len_tr - q_len, pad_on_left=False)
    # Deal with non-standard `scale_factor`
    if scale_factor is not None:
        diff = scale_factor * math.sqrt(head_size)
        if not (0.999 < diff < 1.001):
            query = query * diff

    output, aux = attn_fn(
        query=query,
        key=key,
        value=value,
        scale=None,
        enable_gqa=enable_gqa,
    )
    lse = aux.lse
    if q_len_tr > q_len:
        # Padding was on the right, not on the left
        output = output[:, :, :q_len, :].clone()
        lse = lse[:, :, :q_len].clone()
    return output, lse


def sdpa_ring_flexatt_diag(
    flexatt_args: RingDiagFlexAttentionArgs,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: Optional[float],
    input_pos: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes scaled dot product attention (SDPA) part for a diagonal cell
    `(rank_r, rank_r)` in RingAttention. This is essentially the same as
    `keys_values.attention.flex_attention.scaled_dot_product_attention_flexatt`,
    but the log_sum_exp values are returned as well, and `key, value` need
    to be reordered already.

    Args:
        flexatt_args: Arguments for `flex_attention`. Most important are the
            managers, see :class:`FlexAttnForPrefillManager` and
            :class:`FlexAttnForChunkManager`.
        query: Queries, shape `(batch_size, n_head, q_len, head_size)`
        key: Keys, shape `(batch_size, n_query_groups, kv_len, head_size)`
        value: Values, shape `(batch_size, n_query_groups, kv_len, head_size)`
        scale_factor: Scale factor for attention
        input_pos: Position in input sequence

    Returns:
        `(output, lse)`, where `output` are attention outputs, shape
        `(batch_size, n_head, q_len, head_size)`, `lse` are the log-sum-exp
        values needed for RingAttention, shape `(batch_size, n_head, q_len)`.

    """
    batch_size, n_head, n_query_groups, q_len, kv_len, head_size = sdpa_check_args(
        query,
        key,
        value,
    )
    if input_pos == 0:
        if q_len != kv_len:
            raise ValueError(
                f"For input_pos=0, must have q_len == kv_len, but have q_len = {q_len}, kv_len = {kv_len}"
            )
    enable_gqa = n_query_groups < n_head
    attn_fn, extend_kv = flexatt_args.attn_fn(
        q_len=q_len,
        kv_len=kv_len,
        batch_size=batch_size,
        n_head=n_head,
        device=query.device,
        dtype=query.dtype,
        input_pos=input_pos,
    )
    if enable_gqa and extend_kv:
        key = repeat_interleave(key, n_head)
        value = repeat_interleave(value, n_head)
        enable_gqa = False
        extend_kv = True
    else:
        extend_kv = False
    q_len_tr = flexatt_args.transform_q_len(q_len)
    if q_len_tr > q_len:
        # Use zero padding
        # Importantly, zeros are appended **on the left**, not on the right.
        # The real query entries must be right-aligned with key, value,
        # otherwise our causal attention masking does not work out properly.
        # See :func:`keys_values.sdpa_wrapper.scaled_dot_product_attention`.
        query = zeropad_query(query, q_len_tr - q_len)
    # Deal with non-standard `scale_factor`
    if scale_factor is not None:
        diff = scale_factor * math.sqrt(head_size)
        if not (0.999 < diff < 1.001):
            query = query * diff

    output, aux = attn_fn(
        query=query,
        key=key,
        value=value,
        scale=None,
        enable_gqa=enable_gqa,
    )
    lse = aux.lse
    if q_len_tr > q_len:
        # Padding was on the right, not on the left
        output = output[:, :, :q_len, :].clone()
        lse = lse[:, :, :q_len].clone()
    return output, lse
