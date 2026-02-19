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
from typing import Optional, Callable, Tuple

import torch
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    BlockMask,
)

from keys_values.sdpa_wrapper import (
    sdpa_check_args,
    reorder_key_value,
    ReorderAnnotationCallback,
)
from keys_values.utils import repeat_interleave


FlexAttnWithBlockMask = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float, bool],
    torch.Tensor,
]

ScoreModifier = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


def causal_mask_for_prefill(
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    sliding_window_size: Optional[int],
) -> torch.Tensor:
    result = q_idx >= kv_idx
    if sliding_window_size is not None:
        extra_mask = (q_idx - sliding_window_size) < kv_idx
        result = result & extra_mask
    return result


def logit_softcapping(
    score: torch.Tensor,
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    thresh: float,
) -> torch.Tensor:
    return torch.tanh(score / thresh) * thresh


def quantize_attention_logit_softcapping(
    x: Optional[float],
) -> Tuple[Optional[int], Optional[float]]:
    if x is None:
        return None, None
    else:
        qx = round(x * 128)
        return qx, qx / 128


class FlexAttnManager:
    """
    Base class for FlexAttention manager. Maintains `BlockMask` objects and
    compiled `flex_attention`.

    The idea for managers is to share compiled block masks and attention
    functions across several layers and iterations, so that compilations
    are required initially only.

    """

    def __init__(self):
        self._entries = dict()
        self.num_hits = dict()

    def _get_args(
        self,
        kv_len: int,
        device: torch.device,
        requires_grad: bool,
        sliding_window_size: Optional[int],
        als_signature: Optional[int],
        **kwargs,
    ) -> tuple:
        raise NotImplementedError

    def _create_block_mask(
        self,
        kv_len: int,
        device: torch.device,
        sliding_window_size: Optional[int],
        **kwargs,
    ) -> BlockMask:
        raise NotImplementedError

    def __call__(
        self,
        kv_len: int,
        device: torch.device,
        requires_grad: bool,
        sliding_window_size: Optional[int],
        attention_logit_softcapping: Optional[float],
        **kwargs,
    ) -> Tuple[FlexAttnWithBlockMask, bool]:
        als_signature, thresh = quantize_attention_logit_softcapping(
            attention_logit_softcapping
        )
        args = self._get_args(
            kv_len,
            device,
            requires_grad,
            sliding_window_size,
            als_signature,
            **kwargs,
        )
        result = self._entries.get(args)
        if result is None:
            torch._dynamo.reset()
            torch.compiler.reset()
            block_mask = self._create_block_mask(
                kv_len,
                device,
                sliding_window_size,
                **kwargs,
            )
            if attention_logit_softcapping is not None:
                score_mod = partial(logit_softcapping, thresh=thresh)
            else:
                score_mod = None
            attn_fn = partial(
                torch.compile(flex_attention, fullgraph=True),
                score_mod=score_mod,
                block_mask=block_mask,
            )
            extend_kv = True if self._entries else False
            result = (attn_fn, extend_kv)
            self._entries[args] = result
            self.num_hits[args] = 1
        else:
            self.num_hits[args] = self.num_hits[args] + 1
        return result


class FlexAttnForPrefillManager(FlexAttnManager):
    """
    FlexAttention manager for prefill case
    (`q_len == kv_len, input_pos == 0`).

    """

    def __init__(self):
        super().__init__()

    def _get_args(
        self,
        kv_len: int,
        device: torch.device,
        requires_grad: bool,
        sliding_window_size: Optional[int],
        als_signature: Optional[int],
        **kwargs,
    ) -> tuple:
        return kv_len, device, requires_grad, sliding_window_size, als_signature

    def _create_block_mask(
        self,
        kv_len: int,
        device: torch.device,
        sliding_window_size: Optional[int],
        **kwargs,
    ) -> BlockMask:
        mask_mod = partial(
            causal_mask_for_prefill,
            sliding_window_size=sliding_window_size,
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
    offset: int,
    sliding_window_size: Optional[int],
) -> torch.Tensor:
    left_arg = q_idx + offset
    result = left_arg >= kv_idx
    if sliding_window_size is not None:
        extra_mask = (left_arg - sliding_window_size) < kv_idx
        result = result & extra_mask
    return result


class FlexAttnForChunkManager(FlexAttnManager):
    def __init__(self):
        super().__init__()

    def _unpack_kwargs(
        self,
        **kwargs,
    ) -> Tuple[int, int, int]:
        unpacked_args = []
        for name in (
            "q_len",
            "batch_size",
            "n_head",
        ):
            if name not in kwargs:
                raise ValueError(f"{name} is required")
            unpacked_args.append(kwargs[name])
        return tuple(unpacked_args)

    def _get_args(
        self,
        kv_len: int,
        device: torch.device,
        requires_grad: bool,
        sliding_window_size: Optional[int],
        als_signature: Optional[int],
        **kwargs,
    ) -> tuple:
        q_len, batch_size, n_head = self._unpack_kwargs(**kwargs)
        return (
            q_len,
            kv_len,
            batch_size,
            n_head,
            device,
            requires_grad,
            sliding_window_size,
            als_signature,
        )

    def _create_block_mask(
        self,
        kv_len: int,
        device: torch.device,
        sliding_window_size: Optional[int],
        **kwargs,
    ) -> BlockMask:
        q_len = self._unpack_kwargs(**kwargs)[0]
        mask_mod = partial(
            causal_mask_for_chunk,
            offset=kv_len - q_len,
            sliding_window_size=sliding_window_size,
        )
        return create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )


# TODO: Further args concerning `flex_attention`
class FlexAttentionArgs:
    """
    Maintains managers (for prefill and chunk computations).

    Args:
        extend_kv: If `True` we always extend `key, value` to `n_head`. This
            needs more memory, only use if the default does not work.

    """

    def __init__(
        self,
        extend_kv: bool = False,
    ):
        self.attn_prefill_manager = FlexAttnForPrefillManager()
        self.attn_chunk_manager = FlexAttnForChunkManager()
        self.extend_kv = extend_kv

    def attn_fn(
        self,
        q_len: int,
        kv_len: int,
        batch_size: int,
        n_head: int,
        device: torch.device,
        requires_grad: bool,
        sliding_window_size: Optional[int],
        attention_logit_softcapping: Optional[float],
        input_pos: int,
    ) -> Tuple[FlexAttnWithBlockMask, bool]:
        if input_pos == 0:
            return (
                self.attn_prefill_manager(
                    kv_len=kv_len,
                    device=device,
                    requires_grad=requires_grad,
                    sliding_window_size=sliding_window_size,
                    attention_logit_softcapping=attention_logit_softcapping,
                )[0],
                False,
            )
        else:
            _attn_fn, extend_kv = self.attn_chunk_manager(
                kv_len=kv_len,
                device=device,
                requires_grad=requires_grad,
                sliding_window_size=sliding_window_size,
                attention_logit_softcapping=attention_logit_softcapping,
                q_len=q_len,
                batch_size=batch_size,
                n_head=n_head,
            )
            return _attn_fn, extend_kv or self.extend_kv


def scaled_dot_product_attention_flexatt(
    flexatt_args: FlexAttentionArgs,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: float,
    sliding_window_size: Optional[int],
    attention_logit_softcapping: Optional[float],
    input_pos: int,
    token_positions: Optional[torch.Tensor],
    annotation_callback: Optional[ReorderAnnotationCallback] = None,
) -> torch.Tensor:
    """
    Computes scaled dot product attention (SDPA) using PyTorch
    `flex_attention`. This does not need reordering of `key`, `value`, since
    we can build a specific `block_mask` based on `token_positions` and
    `input_pos`.

    Can be called for prefill (`input_pos == 0`) or chunk processing
    (`input_pos > 0`). In the latter case, the KV cache buffers `key`, `value`
    must have been updated, meaning that `range(input_pos, input_pos + q_len)`
    must be in each `token_positions[b, h]`.

    Args:
        flexatt_args: Arguments for `flex_attention`. Most important are the
            managers, see :class:`FlexAttnForPrefillManager` and
            :class:`FlexAttnForChunkManager`.
        query: Queries, shape `(batch_size, n_head, q_len, head_size)`
        key: Keys, shape `(batch_size, n_query_groups, kv_len, head_size)`
        value: Values, shape `(batch_size, n_query_groups, kv_len, head_size)`
        scale_factor: Scale factor for attention
        sliding_window_size: Sliding window size, optional
        attention_logit_softcapping: Attention logit softcapping threshold,
            optional. Note that this value is quantized so the manager can
            cache compiled graphs
        input_pos: Position in input sequence
        token_positions: Only if `input_pos > 0`. Contains token positions
            in KV cache, shape `(batch_size, n_query_groups, kv_len)`. If not
            given for `input_pos > 0`, it is equivalent to `arange(kv_len)`.
        annotation_callback: If this is given and `key, value` are reordered,
            the results are passed to this callback.

    Returns:
        Attention outputs, shape `(batch_size, n_heads, q_len, head_size)`

    """
    batch_size, n_head, n_query_groups, q_len, kv_len, _ = sdpa_check_args(
        query,
        key,
        value,
    )
    if input_pos == 0:
        if q_len != kv_len:
            raise ValueError(
                f"For input_pos=0, must have q_len == kv_len, but have q_len = {q_len}, kv_len = {kv_len}"
            )
        if token_positions is not None:
            raise ValueError("For input_pos=0, must have token_positions=None")
    else:
        if token_positions is not None and token_positions.shape != key.shape[:-1]:
            raise ValueError(
                f"token_positions.shape = {token_positions.shape}, key.shape = {key.shape}: Not compatible"
            )
    if token_positions is not None:
        key, value, sort_index = reorder_key_value(key, value, token_positions)
    else:
        sort_index = None
    enable_gqa = n_query_groups < n_head
    requires_grad = query.requires_grad or key.requires_grad or value.requires_grad
    attn_fn, extend_kv = flexatt_args.attn_fn(
        q_len=q_len,
        kv_len=kv_len,
        batch_size=batch_size,
        n_head=n_head,
        device=query.device,
        requires_grad=requires_grad,
        sliding_window_size=sliding_window_size,
        attention_logit_softcapping=attention_logit_softcapping,
        input_pos=input_pos,
    )
    if enable_gqa and extend_kv:
        key = repeat_interleave(key, n_head)
        value = repeat_interleave(value, n_head)
        enable_gqa = False
    if annotation_callback is not None:
        annotation_callback(key, value, sort_index)

    return attn_fn(
        query=query,
        key=key,
        value=value,
        scale=scale_factor,
        enable_gqa=enable_gqa,
    )
