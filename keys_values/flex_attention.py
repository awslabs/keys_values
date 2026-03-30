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
from typing import Optional, Callable, Tuple, List, Dict, Any

import torch
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    BlockMask,
    AuxRequest,
)

from keys_values.sdpa_wrapper import (
    sdpa_check_args,
    reorder_key_value,
    reorder_inverse,
    ReorderAnnotationCallback,
    zeropad_query_on_left,
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

    def _args_to_str(self, *args) -> str:
        raise NotImplementedError

    def _create_block_mask(
        self,
        kv_len: int,
        device: torch.device,
        sliding_window_size: Optional[int],
        **kwargs,
    ) -> BlockMask:
        raise NotImplementedError

    def _extra_flex_attention_kwargs(self, args: tuple) -> Dict[str, Any]:
        return dict()

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
            attn_fn = torch.compile(
                partial(
                    flex_attention,
                    score_mod=score_mod,
                    block_mask=block_mask,
                    **self._extra_flex_attention_kwargs(args),
                ),
                fullgraph=True,
            )
            extend_kv = True if self._entries else False
            result = (attn_fn, extend_kv)
            self._entries[args] = result
            self.num_hits[args] = 1
        else:
            self.num_hits[args] = self.num_hits[args] + 1
        return result

    def report(self) -> str:
        parts = [
            self._args_to_str(*args) + f": {num_hits}"
            for args, num_hits in sorted(
                self.num_hits.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        ]
        return "\n".join(parts)


class FlexAttnForPrefillManager(FlexAttnManager):
    """
    FlexAttention manager for prefill case (`q_len == kv_len, input_pos == 0`).

    Note that `flex_attention` is often not used for the prefill calls, because
    standard PyTorch SDPA is faster. It is used only with non-standard SDPA.
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

    def _args_to_str(self, *args) -> str:
        parts = [
            f"kv_len:{args[0]:5d}",
            f"requires_grad:{int(args[2]):1d}",
            f"device:{args[1]}",
        ]
        if args[3] is not None:
            parts.append(f"sliding_window_size:{args[3]:3d}")
        if args[4] is not None:
            parts.append(f"als_signature:{args[4]}")
        return ",".join(parts)

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


def causal_mask_for_chunk_reversed(
    batch: torch.Tensor,
    head: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    offset: int,
) -> torch.Tensor:
    left_arg = kv_idx + offset
    result = left_arg >= q_idx
    return result


class FlexAttnForChunkManager(FlexAttnManager):
    """
    FlexAttention manager for chunk case.

    If `q_lens` is provided, then for each `q_len` argument passed, we
    replace this with the smallest entry in `q_lens` which is `>= q_len`.
    If `q_len > max(q_lens)`, an exception is raised. This limits the number
    of compiled graphs, even if chunks with many different `q_len` sizes come
    in.

    `extend_kv` field: Hack used to get around issue with FlexAttention.
    Details: https://github.com/awslabs/keys_values/issues/34.
    For unclear reasons, `enable_gqa=True` does not work, except for the first
    graph to be stored in `_entries`. This is why `extend_kv=False` for the
    first entry, `extend_kv=True` for all others. Note that the first entry
    is the most used one in general.

    Support of :func:`sdpa_flexatt_with_attn_weights`:

    First, if `forward_return_lse == True`, the compute graphs returned
    for `requires_grad == False` output `attn_outputs, aux`, where `aux.lse`
    are log-sum-exp of attention weights. Second, an extra boolean argument
    `reverse` can be passed (defaults to `False`). If this is `True`, we flip
    `query` and `key` as inputs, so that the score matrix is transposed.
    Internally, this means that :func:`causal_mask_for_chunk_reversed` is used
    instead of :func:`causal_mask_for_chunk`. In this case, `sliding_window_size`
    and `als_signature` cannot be used.
    """

    def __init__(
        self,
        q_lens: Optional[List[int]] = None,
        forward_return_lse: bool = False,
    ):
        super().__init__()
        if q_lens is not None:
            q_lens = sorted(q_lens)
            assert all(x > 0 for x in q_lens)
            assert all(x < y for x, y in zip(q_lens[:-1], q_lens[1:]))
        self.q_lens = q_lens
        self.forward_return_lse = forward_return_lse

    def _unpack_kwargs(
        self,
        **kwargs,
    ) -> Tuple[int, int, int, bool]:
        unpacked_args = []
        for name in (
            "q_len",
            "batch_size",
            "n_head",
        ):
            if name not in kwargs:
                raise ValueError(f"{name} is required")
            unpacked_args.append(kwargs[name])
        unpacked_args.append(kwargs.get("reverse", False))
        return tuple(unpacked_args)

    def transform_q_len(self, q_len: int) -> int:
        if self.q_lens is None:
            return q_len
        else:
            try:
                return next(x for x in self.q_lens if x >= q_len)
            except StopIteration:
                raise ValueError(f"q_len={q_len}, must be <= {max(self.q_lens)}")

    def _get_args(
        self,
        kv_len: int,
        device: torch.device,
        requires_grad: bool,
        sliding_window_size: Optional[int],
        als_signature: Optional[int],
        **kwargs,
    ) -> tuple:
        q_len, batch_size, n_head, reverse = self._unpack_kwargs(**kwargs)
        if reverse:
            if sliding_window_size is not None:
                raise ValueError("Cannot use reverse=True and sliding_window_size")
            kv_len = self.transform_q_len(kv_len)
        else:
            q_len = self.transform_q_len(q_len)
        return (
            q_len,
            kv_len,
            batch_size,
            n_head,
            device,
            requires_grad,
            sliding_window_size,
            als_signature,
            reverse,
        )

    def _args_to_str(self, *args) -> str:
        parts = [
            f"q_len:{args[0]:5d}",
            f"kv_len:{args[1]:5d}",
            f"batch_size:{args[2]:3d}",
            f"n_head:{args[3]:3d}",
            f"requires_grad:{int(args[5]):1d}",
            f"device:{args[4]}",
        ]
        if args[6] is not None:
            parts.append(f"sliding_window_size:{args[6]:3d}")
        if args[7] is not None:
            parts.append(f"als_signature:{args[7]}")
        parts.append(f"reverse:{int(args[8]):1d}")
        return ",".join(parts)

    def _create_block_mask(
        self,
        kv_len: int,
        device: torch.device,
        sliding_window_size: Optional[int],
        **kwargs,
    ) -> BlockMask:
        q_len, _, _, reverse = self._unpack_kwargs(**kwargs)
        if not reverse:
            q_len = self.transform_q_len(q_len)
            if q_len > kv_len:
                raise ValueError(
                    f"q_len={q_len}, kv_len={kv_len}: Must have q_len <= kv_len"
                )
            mask_mod = partial(
                causal_mask_for_chunk,
                offset=kv_len - q_len,
                sliding_window_size=sliding_window_size,
            )
        else:
            kv_len = self.transform_q_len(kv_len)
            if q_len < kv_len:
                raise ValueError(
                    f"q_len={q_len}, kv_len={kv_len}: Must have q_len >= kv_len"
                )
            mask_mod = partial(
                causal_mask_for_chunk_reversed,
                offset=q_len - kv_len,
            )
        return create_block_mask(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

    def _extra_flex_attention_kwargs(self, args: tuple) -> Dict[str, Any]:
        requires_grad = args[5]
        if self.forward_return_lse and not requires_grad:
            return {"return_aux": AuxRequest(lse=True)}
        else:
            return dict()


class FlexAttentionArgs:
    """
    Maintains managers (for prefill and chunk computations).

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
        forward_return_lse: If `True`, the forward graphs also return the
            `lse` tensors. This is done for `input_pos > 0` only.
    """

    def __init__(
        self,
        q_lens: Optional[List[int]] = None,
        extend_kv: bool = False,
        forward_return_lse: bool = False,
    ):
        self.attn_prefill_manager = FlexAttnForPrefillManager()
        self.attn_chunk_manager = FlexAttnForChunkManager(
            q_lens,
            forward_return_lse,
        )
        self.extend_kv = extend_kv
        self.forward_return_lse = forward_return_lse

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
        reverse: bool = False,
    ) -> Tuple[FlexAttnWithBlockMask, bool]:
        """
        If `forward_return_lse == True`, `input_pos > 0`, and
        `requires_grad == False`, the returned graph outputs two arguments:
        `attn_outputs` as usual, and `aux` of type `AuxOutput`, with the
        `lse` attribute being set.

        """
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
            if reverse and sliding_window_size is not None:
                raise ValueError("Cannot use reverse=True and sliding_window_size")
            _attn_fn, extend_kv = self.attn_chunk_manager(
                kv_len=kv_len,
                device=device,
                requires_grad=requires_grad,
                sliding_window_size=sliding_window_size,
                attention_logit_softcapping=attention_logit_softcapping,
                q_len=q_len,
                batch_size=batch_size,
                n_head=n_head,
                reverse=reverse,
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


def scaled_dot_product_attention_flexatt(
    flexatt_args: FlexAttentionArgs,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: Optional[float],
    sliding_window_size: Optional[int],
    attention_logit_softcapping: Optional[float],
    input_pos: int,
    token_positions: Optional[torch.Tensor],
    annotation_callback: Optional[ReorderAnnotationCallback] = None,
    sort_if_3d: bool = False,
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

    Note: `flex_attention` is always called with `scale=None`, the default
    scale factor. If `scale_factor` is different, we multiply a factor into
    `query`. This avoids bogus graph re-compilations due to small numerical
    differences in `scale`.

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
        sort_if_3d: See :func:`reorder_key_value`.

    Returns:
        Attention outputs, shape `(batch_size, n_head, q_len, head_size)`

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
        if token_positions is not None:
            raise ValueError("For input_pos=0, must have token_positions=None")
    else:
        if token_positions is not None and token_positions.shape != key.shape[:-1]:
            raise ValueError(
                f"token_positions.shape = {token_positions.shape}, key.shape = {key.shape}: Not compatible"
            )
    if token_positions is not None:
        key, value, extra_info = reorder_key_value(
            key,
            value,
            token_positions.detach(),
            input_pos,
            q_len,
            sort_if_3d,
        )
    else:
        extra_info = dict()
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
        query = zeropad_query_on_left(query, q_len_tr - q_len)
    # Deal with non-standard `scale_factor`
    diff = scale_factor * math.sqrt(head_size)
    if not (0.999 < diff < 1.001):
        query = query * diff
    if annotation_callback is not None:
        annotation_callback(key, value, extra_info, extend_kv)

    result = attn_fn(
        query=query,
        key=key,
        value=value,
        scale=None,
        enable_gqa=enable_gqa,
    )
    if flexatt_args.forward_return_lse and not requires_grad:
        assert isinstance(result, tuple), (type(result), flexatt_args.forward_return_lse, requires_grad)
        result = result[0]
    else:
        assert isinstance(result, torch.Tensor), (type(result), flexatt_args.forward_return_lse, requires_grad)
    if q_len_tr > q_len:
        result = result[:, :, (-q_len):, :].clone()
    return result


def choose_q_lens(
    chunk_size: int,
    num_q_lens: int,
    add_one: bool = True,
) -> Optional[List[int]]:
    """
    Chooses `q_lens` argument for :class:`FlexAttentionArgs`. This supports
    `q_len` values up to `chunk_size`. The list is equi-spaced, containing
    `chunk_size`.

    Args:
        chunk_size: Maximum `q_len` size, is contained in `q_lens`
        num_q_lens: `len(q_lens) = num_q_lens + int(add_one)`
        add_one: If `True`, then `q_lens[0] = 1` as additional entry. This
            makes sense if the model is ever used to generate tokens. It does
            not hurt, because a graph for length 1 is created only if
            `q_len == 1` is encountered.

    """
    if num_q_lens < chunk_size:
        q_lens = [
            math.ceil(i * chunk_size / num_q_lens) for i in range(1, num_q_lens + 1)
        ]
        if add_one and q_lens[0] > 1:
            q_lens.insert(0, 1)
    else:
        q_lens = None
    return q_lens


MIN_HEAD_DIM = 16


def sdpa_flexatt_with_attn_weights(
    flexatt_args: FlexAttentionArgs,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: Optional[float],
    attention_logit_softcapping: Optional[float],
    input_pos: int,
    token_positions: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Baseline implementation for SDPA returning attention weights summed over
    query axis alongside attention outputs. This is done by calling PyTorch
    `flex_attention` twice. Details are provided in a technical report.

    Can be called for `input_pos > 0` and in `no_grad` mode only. Moreover,
    `sliding_window_size` and `attention_logit_softcapping` are not supported.
    And we must have `flexatt_args.forward_return_lse == True`.

    Args:
        flexatt_args: Arguments for `flex_attention`. Must have
            `forward_return_lse == True`.
        query: Queries, shape `(batch_size, n_head, q_len, head_size)`
        key: Keys, shape `(batch_size, n_query_groups, kv_len, head_size)`
        value: Values, shape `(batch_size, n_query_groups, kv_len, head_size)`
        scale_factor: Scale factor for attention
        attention_logit_softcapping: Attention logit softcapping threshold,
            optional. Note that this value is quantized so the manager can
            cache compiled graphs
        input_pos: Position in input sequence. Must be positive
        token_positions: Contains token positions in KV cache, shape
            `(batch_size, n_query_groups, kv_len)`. If not given, it is
            equivalent to `arange(kv_len)`.

    Returns:
        `(attn_outputs, attn_weights)`, where `attn_outputs` has shape
        `(batch_size, n_head, q_len, head_size)`, `attn_weights` has shape
        `(batch_size, n_head, kv_len)`.

    """
    batch_size, n_head, n_query_groups, q_len, kv_len, head_size = sdpa_check_args(
        query,
        key,
        value,
    )
    if query.requires_grad or key.requires_grad or value.requires_grad:
        raise ValueError("Cannot be used with autograd")
    if input_pos <= 0:
        raise ValueError("input_pos must be positive")
    if not flexatt_args.forward_return_lse:
        raise ValueError("flexatt_args.forward_return_lse must be True")
    if token_positions is not None and token_positions.shape != key.shape[:-1]:
        raise ValueError(
            f"token_positions.shape = {token_positions.shape}, key.shape = {key.shape}: Not compatible"
        )
    if token_positions is not None:
        key, value, extra_info = reorder_key_value(
            key,
            value,
            token_positions.detach(),
            input_pos,
            q_len,
            sort_if_3d=True,
        )
    else:
        extra_info = dict()
    enable_gqa = n_query_groups < n_head

    # (1) First call: SDPA(Q, K, V)
    attn_kwargs = dict(
        batch_size=batch_size,
        n_head=n_head,
        device=query.device,
        requires_grad=False,
        sliding_window_size=None,
        attention_logit_softcapping=attention_logit_softcapping,
        input_pos=input_pos,
    )
    attn_fn, extend_kv = flexatt_args.attn_fn(
        q_len=q_len,
        kv_len=kv_len,
        reverse=False,
        **attn_kwargs,
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
        query = zeropad_query_on_left(query, q_len_tr - q_len)
    # Deal with non-standard `scale_factor`
    diff = scale_factor * math.sqrt(head_size)
    if not (0.999 < diff < 1.001):
        query = query * diff
    attn_output, aux = attn_fn(
        query=query,
        key=key,
        value=value,
        scale=None,
        enable_gqa=enable_gqa,
    )
    if q_len_tr > q_len:
        attn_output = attn_output[:, :, (-q_len):, :].clone()

    # (2) Second call: SDPA_rev(K, Q, V_tilde)
    if not extend_kv:
        # Second call cannot use `enable_gqa=True`
        key = repeat_interleave(key, n_head)
    attn_fn, _ = flexatt_args.attn_fn(
        q_len=kv_len,
        kv_len=q_len,
        reverse=True,
        **attn_kwargs,
    )
    # Multiply by a factor `exp(mean_lse)` here, divide by the same below
    mean_lse = aux.lse.mean(dim=-1, keepdim=True)
    vtil_vec = torch.exp(-(aux.lse - mean_lse)).to(dtype=query.dtype)
    if q_len_tr > q_len:
        vtil_vec[:, :, : (q_len_tr - q_len)] = 0.0
    value_tilde = torch.cat(
        (
            vtil_vec.unsqueeze(-1),
            torch.zeros(
                (1, 1, 1, 1),
                dtype=query.dtype,
                device=query.device,
            ).expand(batch_size, n_head, q_len_tr, MIN_HEAD_DIM - 1),
        ),
        dim=-1,
    )
    output2, aux = attn_fn(
        query=key,
        key=query,
        value=value_tilde,
        scale=None,
        enable_gqa=False,
    )
    attn_weights = output2[:, :, :, 0].to(dtype=torch.float32) * torch.exp(
        aux.lse - mean_lse
    )
    if n_query_groups < n_head:
        # Undo `repeat_interleave`
        attn_weights = torch.mean(
            attn_weights.view(
                batch_size,
                n_query_groups,
                -1,
                kv_len,
            ),
            dim=2,
        )
    if token_positions is not None:
        # Undo reordering
        attn_weights = reorder_inverse(attn_weights, **extra_info)

    return attn_output, attn_weights
