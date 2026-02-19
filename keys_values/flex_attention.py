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
from typing import Optional, Callable, Tuple, Union

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
from keys_values.utils import is_index_1d, repeat_interleave

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

    def __init__(
        self,
        debug_flexatt_no_compile: bool = False,
    ):
        self._entries = dict()
        self.num_hits = dict()
        self._debug_flexatt_no_compile = debug_flexatt_no_compile

    def _get_args(
        self,
        kv_len: int,
        device: torch.device,
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
    ) -> Union[BlockMask, "BlockMaskForChunk"]:
        raise NotImplementedError

    def _create_final_attn_fn(
        self,
        attn_fn: Callable,
        score_mod: Optional[ScoreModifier],
        block_mask: Union[BlockMask, "BlockMaskForChunk"],
        kv_len: int,
        device: torch.device,
        sliding_window_size: Optional[int],
        **kwargs,
    ) -> FlexAttnWithBlockMask:
        raise NotImplementedError

    def __call__(
        self,
        kv_len: int,
        device: torch.device,
        sliding_window_size: Optional[int],
        attention_logit_softcapping: Optional[float],
        **kwargs,
    ) -> FlexAttnWithBlockMask:
        als_signature, thresh = quantize_attention_logit_softcapping(
            attention_logit_softcapping
        )
        args = self._get_args(
            kv_len,
            device,
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
            if not self._debug_flexatt_no_compile:
                attn_fn = torch.compile(flex_attention, fullgraph=True)
            else:
                attn_fn = None
            result = (block_mask, score_mod, attn_fn)
            self._entries[args] = result
            self.num_hits[args] = 1
        else:
            self.num_hits[args] = self.num_hits[args] + 1
        block_mask, score_mod, attn_fn = result
        if self._debug_flexatt_no_compile:
            attn_fn = flex_attention
        return self._create_final_attn_fn(
            attn_fn=attn_fn,
            score_mod=score_mod,
            block_mask=block_mask,
            kv_len=kv_len,
            device=device,
            sliding_window_size=sliding_window_size,
            **kwargs,
        )


class FlexAttnForPrefillManager(FlexAttnManager):
    """
    FlexAttention manager for prefill case
    (`q_len == kv_len, input_pos == 0`).

    """

    def __init__(
        self,
        debug_flexatt_no_compile: bool = False,
    ):
        super().__init__(debug_flexatt_no_compile)

    def _get_args(
        self,
        kv_len: int,
        device: torch.device,
        sliding_window_size: Optional[int],
        als_signature: Optional[int],
        **kwargs,
    ) -> tuple:
        return kv_len, device, sliding_window_size, als_signature

    def _create_block_mask(
        self,
        kv_len: int,
        device: torch.device,
        sliding_window_size: Optional[int],
        **kwargs,
    ) -> Union[BlockMask, "BlockMaskForChunk"]:
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

    def _create_final_attn_fn(
        self,
        attn_fn: Callable,
        score_mod: Optional[ScoreModifier],
        block_mask: Union[BlockMask, "BlockMaskForChunk"],
        kv_len: int,
        device: torch.device,
        sliding_window_size: Optional[int],
        **kwargs,
    ) -> FlexAttnWithBlockMask:
        return partial(
            attn_fn,
            score_mod=score_mod,
            block_mask=block_mask,
        )


def transform_token_positions(
    token_positions: torch.Tensor,
    n_head: int,
) -> torch.Tensor:
    """
    Transforms `token_positions` argument to
    :func:`scaled_dot_product_attention_flexatt` into tensor used in the
    kernel. If the argument is extended from 1D, we return the 1D slice.
    Otherwise, if `q_per_kv > 1`, we create an expanded copy of shape
    `(batch_size, n_head, kv_len)`.

    """
    if is_index_1d(token_positions):
        return token_positions[0, 0, :]
    else:
        batch_size, n_query_groups, _ = token_positions.shape
        q_per_kv = n_head // n_query_groups
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
                .contiguous()
            )
        return token_positions


# TODO: Does not work reliably. Figure this out, or remove!
def causal_mask_for_chunk_3d(
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    input_pos: torch.Tensor,
    token_positions: torch.Tensor,
    sliding_window_size: Optional[int],
) -> torch.Tensor:
    """
    This is based on :func:`keys_values.attention_utils.mask_slice_bool`,
    which implements `boolmask = A < B`, of shape
    `(batch_size, n_head, q_len, kv_len)`.

    The semantics of boolean is flipped here, we need to return
    `A[batch, head, q_idx, kv_idx] >= B[batch, head, q_idx, kv_idx]`.

    """
    # result = left_arg >= right_arg, where
    # left_arg = A[batch, head, q_idx, kv_idx],
    # right_arg = B[batch, head, q_idx, kv_idx]
    left_arg = q_idx + input_pos
    right_arg = token_positions[batch, head, kv_idx]
    result = left_arg >= right_arg
    if sliding_window_size is not None:
        extra_mask = (left_arg - sliding_window_size) < right_arg
        result = result & extra_mask
    return result


# TODO: Does not work reliably. Figure this out, or remove!
def causal_mask_for_chunk_1d(
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    input_pos: torch.Tensor,
    token_positions: torch.Tensor,
    sliding_window_size: Optional[int],
) -> torch.Tensor:
    """
    Variant for `token_positions.ndim == 1`.

    """
    left_arg = q_idx + input_pos
    right_arg = token_positions[kv_idx]
    result = left_arg >= right_arg
    if sliding_window_size is not None:
        extra_mask = (left_arg - sliding_window_size) < right_arg
        result = result & extra_mask
    return result


def causal_mask_for_chunk_notp(
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    input_pos: int,
    sliding_window_size: Optional[int],
) -> torch.Tensor:
    """
    Variant without `token_positions`, which means that
    `token_positions == arange(kv_len)`. Also, `input_pos` is a fixed argument
    here, not a variable tensor.

    """
    left_arg = q_idx + input_pos
    result = left_arg >= kv_idx
    if sliding_window_size is not None:
        extra_mask = (left_arg - sliding_window_size) < kv_idx
        result = result & extra_mask
    return result


def causal_mask_for_chunk_debug(
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    offset: int,
) -> torch.Tensor:
    return q_idx + offset >= kv_idx


# TODO: Case `tp_ndim > 0` does not work reliably. Figure this out or remove.
# The case `tp_ndim == 0` does not need a class, because the block mask has no
# variable inputs.
class BlockMaskForChunk:
    """
    Represents `BlockMask` object for inference with KV cache pattern
    (`q_len < kv_len`), along with captured tensors `input_pos` and
    `token_positions` (latter only if `tp_ndim > 0`). Depending on `tp_ndim`,
    `token_positions` is 3D (3) or 1D (1).

    """

    def __init__(
        self,
        q_len: int,
        kv_len: int,
        batch_size: int,
        n_head: int,
        device: torch.device,
        sliding_window_size: Optional[int],
        tp_ndim: int,
    ):
        assert tp_ndim in (0, 1, 3)
        tp_is_3d = tp_ndim == 3
        tp_is_none = tp_ndim == 0
        kwargs = dict(device=device, dtype=torch.int32)
        self.n_head = n_head
        if tp_is_none:
            self.input_pos = None
            self.token_positions = None
            if sliding_window_size is None:
                mask_mod = partial(
                    causal_mask_for_chunk_debug,
                    offset=kv_len - q_len,
                )
                assert not tp_is_3d
            else:
                mask_mod = partial(
                    causal_mask_for_chunk_notp,
                    input_pos=kv_len - q_len,
                    sliding_window_size=sliding_window_size,
                )
        else:
            self.input_pos = torch.tensor(0, **kwargs)
            self.token_positions = torch.zeros(
                (batch_size, n_head, kv_len) if tp_is_3d else (kv_len,),
                **kwargs,
            )
            mask_mod = partial(
                causal_mask_for_chunk_3d if tp_is_3d else causal_mask_for_chunk_1d,
                input_pos=self.input_pos,
                token_positions=self.token_positions,
                sliding_window_size=sliding_window_size,
            )
        self.block_mask = create_block_mask(
            mask_mod,
            B=batch_size if tp_is_3d else None,
            H=n_head if tp_is_3d else None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

    def __call__(
        self,
        input_pos: int,
        token_positions: Optional[torch.Tensor],
    ) -> BlockMask:
        if self.input_pos is not None:
            self.input_pos.copy_(input_pos)
            token_positions = transform_token_positions(token_positions, self.n_head)
            self.token_positions[:] = token_positions
        return self.block_mask


class FlexAttnForChunkManager(FlexAttnManager):
    """
    Maintains `BlockMask` objects and compiled `flex_attention` for inference
    with KV cache pattern (`q_len < kv_len`), for different
    `(q_len, kv_len, batch_size, n_head, device)` values.

    Each entry also has `input_pos` and `token_positions` as inputs, but
    they are captured tensors (they do not determine the shapes, just the
    mask content),
    see: https://pytorch.org/blog/flexattention-for-inference/.

    """

    def __init__(
        self,
        debug_flexatt_no_compile: bool = False,
    ):
        super().__init__(debug_flexatt_no_compile)

    def _unpack_kwargs(
        self,
        **kwargs,
    ) -> Tuple[int, int, int, int, Optional[torch.Tensor]]:
        unpacked_args = []
        for name in (
            "q_len",
            "batch_size",
            "n_head",
            "input_pos",
        ):
            if name not in kwargs:
                raise ValueError(f"{name} is required")
            unpacked_args.append(kwargs[name])
        unpacked_args.append(kwargs.get("token_positions"))
        return tuple(unpacked_args)

    @staticmethod
    def get_tp_ndim(token_positions: Optional[torch.Tensor]) -> int:
        if token_positions is None:
            return 0
        else:
            return 1 if is_index_1d(token_positions) else 3

    def _get_args(
        self,
        kv_len: int,
        device: torch.device,
        sliding_window_size: Optional[int],
        als_signature: Optional[int],
        **kwargs,
    ) -> tuple:
        q_len, batch_size, n_head, _, token_positions = self._unpack_kwargs(**kwargs)
        tp_ndim = self.get_tp_ndim(token_positions)
        return (
            q_len,
            kv_len,
            batch_size,
            n_head,
            device,
            sliding_window_size,
            tp_ndim,
            als_signature,
        )

    def _create_block_mask(
        self,
        kv_len: int,
        device: torch.device,
        sliding_window_size: Optional[int],
        **kwargs,
    ) -> Union[BlockMask, "BlockMaskForChunk"]:
        args = self._get_args(
            kv_len,
            device,
            sliding_window_size,
            None,
            **kwargs,
        )
        return BlockMaskForChunk(*args[:-1])

    def _create_final_attn_fn(
        self,
        attn_fn: Callable,
        score_mod: Optional[ScoreModifier],
        block_mask: Union[BlockMask, "BlockMaskForChunk"],
        kv_len: int,
        device: torch.device,
        sliding_window_size: Optional[int],
        **kwargs,
    ) -> FlexAttnWithBlockMask:
        _, _, _, input_pos, token_positions = self._unpack_kwargs(**kwargs)
        return partial(
            attn_fn,
            score_mod=score_mod,
            block_mask=block_mask(input_pos, token_positions),
        )


# TODO: Further args concerning `flex_attention`
# TODO: Figure out `reorder_kv=False` case or remove!
class FlexAttentionArgs:
    """
    Maintains managers (for prefill and chunk computations).

    Args:
        reorder_kv: If `True`, `token_positions` is taken into account by
            reordering `key`, `value` in
            :func:`keys_values.flex_attention.scaled_dot_product_attention_flexatt`,
            the argument is ignored here. Defaults to `True`.
            Note: `reorder_kv=False` not supported at the moment!
        extend_kv: If `True` we extend `key, value` to `n_head`. This avoids
            GQA, which may not be implemented correctly.
        debug_flexatt_no_compile: If true, `flex_attention` is not compiled.
            Only for debugging (slow and requires lots of memory).

    """

    def __init__(
        self,
        reorder_kv: bool = True,
        extend_kv: bool = False,
        debug_flexatt_no_compile: bool = False,
    ):
        if not reorder_kv:
            raise NotImplementedError(
                "At present, 'reorder_k=False' does not work reliably, do not use."
            )
        self.attn_prefill_manager = FlexAttnForPrefillManager(
            debug_flexatt_no_compile,
        )
        self.attn_chunk_manager = FlexAttnForChunkManager(
            debug_flexatt_no_compile,
        )
        self.reorder_kv = reorder_kv
        self.extend_kv = extend_kv

    def attn_fn(
        self,
        q_len: int,
        kv_len: int,
        batch_size: int,
        n_head: int,
        device: torch.device,
        sliding_window_size: Optional[int],
        attention_logit_softcapping: Optional[float],
        input_pos: int,
        token_positions: Optional[torch.Tensor],
    ) -> FlexAttnWithBlockMask:
        if input_pos == 0:
            return self.attn_prefill_manager(
                kv_len=kv_len,
                device=device,
                sliding_window_size=sliding_window_size,
                attention_logit_softcapping=attention_logit_softcapping,
            )
        else:
            return self.attn_chunk_manager(
                kv_len=kv_len,
                device=device,
                sliding_window_size=sliding_window_size,
                attention_logit_softcapping=attention_logit_softcapping,
                q_len=q_len,
                batch_size=batch_size,
                n_head=n_head,
                input_pos=input_pos,
                token_positions=None if self.reorder_kv else token_positions,
            )


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
    if token_positions is not None and flexatt_args.reorder_kv:
        key, value, sort_index = reorder_key_value(key, value, token_positions)
        token_positions = None
    else:
        sort_index = None
    enable_gqa = n_query_groups < n_head
    if flexatt_args.extend_kv and enable_gqa:
        key = repeat_interleave(key, n_head)
        value = repeat_interleave(value, n_head)
        enable_gqa = False
    if annotation_callback is not None:
        annotation_callback(key, value, sort_index)

    attn_fn = flexatt_args.attn_fn(
        q_len=q_len,
        kv_len=kv_len,
        batch_size=batch_size,
        n_head=n_head,
        device=query.device,
        sliding_window_size=sliding_window_size,
        attention_logit_softcapping=attention_logit_softcapping,
        input_pos=input_pos,
        token_positions=token_positions,
    )
    return attn_fn(
        query=query,
        key=key,
        value=value,
        scale=scale_factor,
        enable_gqa=enable_gqa,
    )
