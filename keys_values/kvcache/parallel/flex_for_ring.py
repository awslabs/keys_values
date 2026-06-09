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
            kv_len, device, dtype, requires_grad, sliding_window_size, als_signature, **kwargs,
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
        right_arg < 0, right_arg < q_idx + int(offset_positive),
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
            kv_len, device, dtype, requires_grad, sliding_window_size, als_signature, **kwargs,
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


class RingFlexAttentionArgs:
    """
    Maintains managers (for prefill and chunk computations).

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
        self._num_devices = num_devices

    # HIER: What about thresh and offset_positive? Do this based on
    # knowing r and s here?
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
        if not(0 <= rank_r < self._num_devices):
            raise ValueError(f"rank_r={rank_r}, must be in [0, {self._num_devices})")
        if not(0 <= rank_s < self._num_devices):
            raise ValueError(f"rank_s={rank_s}, must be in [0, {self._num_devices})")
        if rank_r == rank_s:
            raise ValueError("Must have rank_r != rank_s. Use FlexAttentionArgs if they are the same")
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
            ndevs = self._num_devices  # N
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
