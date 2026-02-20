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
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, Set, List

import torch

from keys_values.sdpa_wrapper import reorder_buffer_given_sort_index
from keys_values.utils import shape_to_tuple, expand_index, repeat_interleave


# The typical shape for `annotation.delta` in phase (1), matching annotations
# against pack arguments, is
# `(batch_size, n_query_groups, MAX_DELTA_TRANS_LENGTH, head_size)`. Making
# this smaller increases the probability of false matches, but speeds up the
# matching.
MAX_DELTA_TRANS_LENGTH = 32

_ANNOTATION_KIND_TO_SHORT = {
    "cat-key": "c-k",
    "cat-value": "c-v",
    "ext-key": "e-k",
    "ext-value": "e-v",
    "padded-query": "pad-q",
    "scatter-key": "s-k",
    "scatter-value": "s-v",
}


@dataclass()
class NodeAnnotation:
    """
    Note: If the node-creating operation is `x_new = f(x, index, delta)`, the
    information recorded is for reconstructing `x` (not `x_new`). For example,
    `shape = x.shape`, and `delta`, `index` refer to a part of `x`.

    The semantics of `index`, `delta` depend on `kind`. Only for "scatter-*",
    it is used to reconstruct `x` from `x_new`. For all other kinds, `index`
    and `delta` are used only to support matching (i.e., recognize whether a
    pack argument is equal to `x_new`)

    """

    kind: str
    layer_idx: int
    chunk_idx: int
    shape: Tuple[int, ...]
    index: Optional[torch.Tensor]
    delta: Optional[torch.Tensor]
    positions: Optional[torch.Tensor] = None
    extra_info: Optional[Dict[str, Any]] = None
    match_id: Optional[int] = None
    does_not_match: Set[int] = field(default_factory=set)
    debug_full_arg: Optional[torch.Tensor] = None
    debug_msg: Optional[str] = None

    def __post_init__(self):
        assert self.layer_idx >= 0
        assert self.chunk_idx >= 0
        self.kind_is_valid(self.kind)
        if self.index is not None:
            assert self.delta is not None
            device = self.delta.device
            assert (
                self.index.device == device
            ), f"delta.device = {device}, index.device = {self.index.device}, must be the same"
        else:
            assert self.delta is None
            device = None
        if self.positions is not None:
            assert self.is_scatter, "positions only with scatter"
            assert self.positions.ndim == 1, "positions must be a 1D tensor"
            if device is not None:
                assert (
                    self.positions.device == device
                ), f"delta.device = {device}, positions.device = {self.positions.device}, must be the same"

    def __str__(self) -> str:
        return f"{self.kind} ({self.layer_idx},{self.chunk_idx}): {self.shape}"

    @property
    def is_keys(self) -> bool:
        return self.kind_is_keys(self.kind)

    @staticmethod
    def kind_is_keys(kind: str) -> bool:
        return kind.endswith("key")

    @property
    def is_values(self) -> bool:
        return self.kind_is_values(self.kind)

    @staticmethod
    def kind_is_values(kind: str) -> bool:
        return kind.endswith("value")

    @property
    def is_scatter(self) -> bool:
        return self.kind_is_scatter(self.kind)

    @staticmethod
    def kind_is_scatter(kind: str) -> bool:
        return kind.startswith("scatter")

    @property
    def is_cat(self) -> bool:
        return self.kind_is_cat(self.kind)

    @staticmethod
    def kind_is_cat(kind: str) -> bool:
        return kind.startswith("cat")

    @property
    def is_ext(self) -> bool:
        return self.kind_is_ext(self.kind)

    @staticmethod
    def kind_is_ext(kind: str) -> bool:
        return kind.startswith("ext")

    @staticmethod
    def kind_is_valid(kind: str):
        assert (
            kind in _ANNOTATION_KIND_TO_SHORT
        ), f"kind = '{kind}', must be in {list(_ANNOTATION_KIND_TO_SHORT.keys())}"

    def fingerprint(self) -> str:
        kind_short = _ANNOTATION_KIND_TO_SHORT[self.kind]
        return f"{kind_short}({self.layer_idx},{self.chunk_idx})"


class NodeAnnotationForLog:
    def __init__(self, annotation: NodeAnnotation):
        self.kind = annotation.kind
        self.layer_idx = annotation.layer_idx
        self.chunk_idx = annotation.chunk_idx
        self.shape = annotation.shape
        self.index_shape = shape_to_tuple(annotation.index)
        self.delta_shape = shape_to_tuple(annotation.delta)
        self.dtype = annotation.delta.dtype
        self.positions_shape = (
            (
                None
                if annotation.positions is None
                else shape_to_tuple(annotation.positions)
            ),
        )
        self.debug_msg = annotation.debug_msg


def create_random_index(
    shape: Tuple[int, int, int, int],
    length: int,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    assert len(shape) == 4
    if dtype is None:
        dtype = torch.int64
    index_kwargs = dict(dtype=dtype, device=device)
    result = torch.empty(shape[:-1], **index_kwargs)
    num = min(shape[2], length)
    for b in range(shape[0]):
        for h in range(shape[1]):
            result[b, h, :] = torch.randperm(length, **index_kwargs)[:num]
    return expand_index(result, shape[-1])


def create_ext_annotations(
    key: torch.Tensor,
    value: torch.Tensor,
    sort_index: Optional[torch.Tensor],
    node_annotations: List[NodeAnnotation],
    annot_kwargs: Dict[str, Any],
):
    """
    Creates annotations "ext-key", "ext-value".

    Args:
        node_annotations: Annotations appended here
        annot_kwargs: Args for :class:`NodeAnnotation`, must contain
            "layer_idx", "chunk_idx"
        key: Keys after reordering (optional) and repeat_interleave
            (optional)
        value: Values after reordering (optional) and repeat_interleave
            (optional)
        sort_index: Sort index determining reordering (3D or 1D).
            Stored in `NodeAnnotation.extra_info`.

    """
    for buffer, name in ((key, "key"), (value, "value")):
        kind = "ext-" + name
        shape = shape_to_tuple(buffer)
        index_shape = shape[:2] + (MAX_DELTA_TRANS_LENGTH, shape[-1])
        index = create_random_index(
            shape=index_shape,
            length=shape[2],
            device=buffer.device,
            dtype=torch.int32,
        )
        delta = buffer.gather(2, index)
        if sort_index is not None:
            extra_info = {"sort_index": sort_index}
        else:
            extra_info = None
        node_annotations.append(
            NodeAnnotation(
                **annot_kwargs,
                kind=kind,
                shape=shape,
                index=index,
                delta=delta,
                extra_info=extra_info,
            )
        )


def apply_ext_annotation(
    buffer: torch.Tensor,
    annotation: NodeAnnotation,
    target_dim1: int,
):
    """
    Applies annotation to buffer, for kind "ext-*".

    Args:
        buffer: Buffer before reordering and/or extension
        annotation: Annotation of "ext-*" kind
        target_dim1: Returned buffer must have `shape[1] == target_dim1`

    Returns:
        Buffer after reordering and/or extension

    """
    allowed_kinds = ("ext-key", "ext-value")
    if annotation.kind not in allowed_kinds:
        raise ValueError(
            f"annotation.kind = {annotation.kind}, must be in {allowed_kinds}"
        )
    extra_info = annotation.extra_info
    sort_index = None if extra_info is None else extra_info.get("sort_index")
    if sort_index is not None:
        buffer = reorder_buffer_given_sort_index(buffer, sort_index)
    buffer = repeat_interleave(buffer, n_head=target_dim1)
    return buffer
