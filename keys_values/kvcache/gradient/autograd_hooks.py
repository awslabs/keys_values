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
from typing import List, Union, Tuple, Dict, Optional, Set, Any
from dataclasses import dataclass, field, replace
from collections import Counter

import torch

from litgpt.config import Config

from keys_values.kvcache.base import KVCacheReplayLog
from keys_values.kvcache.gradient.cleanup import ArraysForCleanup
from keys_values.kvcache.utils import shape_to_tuple, bytes_for_torch_dtype
from keys_values.utils import expand_index, repeat_interleave


_ANNOTATION_KIND_VALUES = {
    "cat-key",
    "cat-value",
    "ext-key",
    "ext-value",
    "padded-query",
    "scatter-key",
    "scatter-value",
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
            assert self.index.device == device, f"delta.device = {device}, index.device = {self.index.device}, must be the same"
        else:
            assert self.delta is None
            device = None
        if self.positions is not None:
            assert self.is_scatter, "positions only with scatter"
            assert self.positions.ndim == 1, "positions must be a 1D tensor"
            if device is not None:
                assert self.positions.device == device, f"delta.device = {device}, positions.device = {self.positions.device}, must be the same"

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
        assert kind in _ANNOTATION_KIND_VALUES, f"kind = '{kind}', must be in {_ANNOTATION_KIND_VALUES}"


class NodeAnnotationForLog:
    def __init__(self, annotation: NodeAnnotation):
        self.kind = annotation.kind
        self.layer_idx = annotation.layer_idx
        self.chunk_idx = annotation.chunk_idx
        self.shape = annotation.shape
        self.index_shape = shape_to_tuple(annotation.index)
        self.delta_shape = shape_to_tuple(annotation.delta)
        self.dtype = annotation.delta.dtype
        self.positions_shape = None if annotation.positions is None else shape_to_tuple(annotation.positions),
        self.debug_msg = annotation.debug_msg


@dataclass(frozen=True)
class Annotations:
    nodes: List[NodeAnnotation] = field(default_factory=list)
    final_keys: Dict[int, torch.Tensor] = field(default_factory=dict)
    final_values: Dict[int, torch.Tensor] = field(default_factory=dict)
    chunk_idx_keys: Dict[int, int] = field(default_factory=dict)
    chunk_idx_values: Dict[int, int] = field(default_factory=dict)

    def clear(self):
        self.nodes.clear()
        self.final_keys.clear()
        self.final_values.clear()
        self.chunk_idx_keys.clear()
        self.chunk_idx_values.clear()

    def _get_dicts(
        self, kind: str,
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
        if NodeAnnotation.kind_is_keys(kind):
            return self.final_keys, self.chunk_idx_keys
        else:
            return self.final_values, self.chunk_idx_values

    def set_final(
        self, x: torch.Tensor, layer_idx: int, chunk_idx: int, kind: str,
    ):
        x_dct, chidx_dct = self._get_dicts(kind)
        curr_val = x_dct.get(layer_idx)
        if curr_val is not None and curr_val.shape == x.shape:
            if x_dct[layer_idx] is not x:
                x_dct[layer_idx][:] = x
        else:
            x_dct[layer_idx] = x.clone()
        chidx_dct[layer_idx] = chunk_idx

    def get_final(self, layer_idx: int, kind: str) -> Tuple[torch.Tensor, int]:
        x_dct, chidx_dct = self._get_dicts(kind)
        return x_dct[layer_idx], chidx_dct[layer_idx]

    def find(self, annotation: NodeAnnotation) -> Optional[int]:
        return next(
            (
                pos
                for pos, annot in enumerate(self.nodes)
                if (
                    annot.layer_idx == annotation.layer_idx and
                    annot.chunk_idx == annotation.chunk_idx and
                    annot.kind == annotation.kind
                )
            ),
            None
        )

    def append_safe(self, annotation: NodeAnnotation):
        pos = self.find(annotation)
        if pos is not None:
            raise IndexError(
                "Annotation already in the list!\n"
                f"nodes[{pos}]: {str(self.nodes[pos])}\n"
                f"new: {str(annotation)}"
            )
        self.nodes.append(annotation)


@dataclass(frozen=True)
class PackHookArgument:
    id: int
    x: torch.Tensor
    shape: Tuple[int, ...]
    count: int = 0

    def increase_count(self) -> "PackHookArgument":
        return PackHookArgument(self.id, self.x, self.shape, self.count + 1)


@dataclass(frozen=True)
class UnmatchedPackHookArgument:
    id: int
    shape: Tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class PackArgumentAsAnnotation:
    annot: NodeAnnotation
    target_dtype: Optional[torch.dtype]


@dataclass(frozen=True)
class PackArgumentAsIndex:
    index_3d: torch.Tensor
    final_dim: int


PackedArgumentType = Union[PackArgumentAsAnnotation, PackArgumentAsIndex, torch.Tensor]


@dataclass(frozen=True)
class AnnotationUsageLog:
    """
    Statistics about autograd hooks mechanism, can be obtained by
    :meth:`CellComputationAutogradHooks.annotation_usage_log`.

    - `num_matched_annotations`: Ideally, all annotations are matched (and
        therefore used to pack arguments).
    - Number of unmatched pack args: A large number here means there are
        many pack args of the targeted shapes, which we cannot annotate and
        pack right now.
    - `num_unmatched_scatter_cat`: If this is non-zero, the value of
        `max_match_trials_pack_arg` may be too small. It means that some
        necessary annotations are not matched, most likely because the
        respective pack arguments are removed too early, or otherwise because
        they are not used in the `autograd` graph.
    - `num_comparisons`: If this is large, the value of
        `max_match_trials_pack_arg` may be too large. Pack arguments which do
        not match annotations, remain in the list for too long.
    - `num_4d_indexes`: Number of 4D broadcast-extended indexes which were
        matched.

    """
    num_matched_annotations: int
    num_comparisons: int
    num_4d_indexes: int
    num_unmatched_scatter_cat: int
    unmatched_pack_args: List[UnmatchedPackHookArgument] = field(default_factory=list)
    remaining_annotations: List[NodeAnnotationForLog] = field(default_factory=list)

    def report(self) -> str:
        lines: List[str] = [
            f"Number of matched annotations:           {self.num_matched_annotations}",
            f"Number of unmatched pack args:           {len(self.unmatched_pack_args)}",
            f"Number of delta comparisons:             {self.num_comparisons}",
            f"Number of 4D indexes packed:             {self.num_4d_indexes}",
            f"Number of unmatched scatter annotations: {self.num_unmatched_scatter_cat}",
        ]
        if self.remaining_annotations:
            lines.append("Remaining unmatched annotations:")
            for a in self.remaining_annotations:
                line = f"  ({a.layer_idx},{a.chunk_idx}): {a.shape}, {a.dtype}, {a.kind}"
                if a.debug_msg is not None:
                    line += f": {a.debug_msg} -- delta {a.delta_shape}"
                lines.append(line)
        else:
            lines.append("All annotations have been used.")
        if self.unmatched_pack_args:
            lines.append("Remaining unmatched pack arguments:")
            for a in self.unmatched_pack_args:
                lines.append(f"  {a.id}: {a.shape}, {a.dtype}")
        else:
            lines.append("All pack arguments have been used.")

        return "\n".join(lines)


# The typical shape for `annotation.delta` in phase (1), matching annotations
# against pack arguments, is
# `(batch_size, n_query_groups, MAX_DELTA_TRANS_LENGTH, head_size)`. Making
# this smaller increases the probability of false matches, but speeds up the
# matching.
MAX_DELTA_TRANS_LENGTH = 32


@dataclass(frozen=True)
class LargeNodesLogEntry:
    name: str
    shape: Tuple[int, ...]


@dataclass(frozen=True)
class LargeShapeEntry:
    shape: Tuple[int, ...]
    numel: int
    dtype: torch.dtype
    count: int = 1

    def process(self, x: torch.Tensor) -> "LargeShapeEntry":
        x_numel = x.numel()
        if x_numel > self.numel or self.numel == 0:
            return LargeShapeEntry(
                shape=tuple(x.shape),
                numel=x_numel,
                dtype=x.dtype,
                count=1,
            )
        elif x.shape == self.shape:
            return LargeShapeEntry(
                shape=self.shape,
                numel=self.numel,
                dtype=x.dtype,
                count=self.count + 1,
            )
        else:
            return self

    def size_in_mb(self) -> float:
        return self.numel * bytes_for_torch_dtype(self.dtype) / (2 ** 20)


class AutogradHooks:
    def pack_hook(self, x: torch.Tensor) -> Union[torch.Tensor, int]:
        raise NotImplementedError

    def unpack_hook(self, x: Union[torch.Tensor, int]) -> torch.Tensor:
        raise NotImplementedError

    @property
    def arrays_cleanup(self) -> Optional[ArraysForCleanup]:
        raise NotImplementedError

    def clear(self):
        pass


class CleanupArraysAutogradHooks(AutogradHooks):
    """
    Use these hooks instead of :class:`CellComputationAutogradHooks` if cell
    computations are not to be supported, but cleanup of autograd tensors
    should be.

    """
    def __init__(self, arrays_cleanup: ArraysForCleanup):
        self._arrays_cleanup = arrays_cleanup

    def pack_hook(self, x: torch.Tensor) -> Union[torch.Tensor, int]:
        self._arrays_cleanup.add(x)
        return x

    def unpack_hook(self, x: Union[torch.Tensor, int]) -> torch.Tensor:
        self._arrays_cleanup.remove(x)
        return x

    @property
    def arrays_cleanup(self) -> Optional[ArraysForCleanup]:
        return self._arrays_cleanup

    def clear(self):
        self._arrays_cleanup.reset()


class CellComputationAutogradHooks(AutogradHooks):
    """
    Attention: Everything below only works properly if the operators
    :class:`SDPAFunction` or :class:`KVCacheUpdateAndSDPAFunction` are used
    instead of the standard MHA.

    When running `autograd` on a cell, consisting of a number of layers and a
    number of token chunks, the largest variables are keys and values of the
    full KV cache size, with shape
    `(batch_size, n_query_groups, cache_length, head_size)`. Since these
    arise from `scatter` or `cat` with much smaller arguments, we can use
    packing and unpacking, so only a small number of these full-size tensors
    need to be stored. This is done by the autograd saved tensors hooks
    mechanism:

    https://pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html

    This works as interplay with :class:`TrainingAttnWeightsReplayCache`, which
    implements the part of the forward pass where these large tensor nodes are
    created. Each such node is given an annotation of type
    :class:`NodeAnnotation`.

    Matching annotations to :meth:`pack_hook` arguments:

    An annotation for `x` is created when a node is created as
    `x_new = f(x, index, delta)`. It is indexed as `(l, s)`. For each `l`,
    nodes form dependency chains: `(l, s - 1) -> (l, s)`. The annotation
    contains information to reconstruct `x` from `x_new`, or to map `x` to
    the pack argument.

    One difficulty is that :meth:`pack_hook` is often called for `x` before its
    annotation is created (because `x` is used before its usage to create
    `x_new`). In :meth:`pack_hook`, we check whether the argument `x` has a shape
    of an annotated tensor. If so, it is entered into `_pack_arguments`. We then
    call :meth:`_match_annotations` in order to match `_pack_arguments` entries
    with annotations.
    This method is called in subsequent :meth:`pack_hook` calls: the goal is to
    deplete `_pack_arguments` as soon as possible, so the GPU memory for the
    pack arguments is freed. :meth:`_match_annotations` is also called when
    :meth:`unpack_hook` is called for the first time, and remaining unmatched
    pack arguments are flushed, so they will not be packed. Ideally, there
    should be no such pack arguments, and no unmatched annotations either.

    We support some differences between pack arguments and annotations, which
    may arise in MHA implementations when called for the prefill:

    * Different type: Pack argument can be `float32`, annotation can have
      a different type (`float16`, `bfloat16`).

    Roles of `index`, `delta`:

    The `index, delta` entries of an annotation serve two different purposes:
    (1) matching against pack argument, (2) reconstruction of argument during
    unpacking. When an annotation is matched against a pack argument, the
    `delta` information is not needed for (2), since it can be extracted from
    the pack argument.
    However, for "scatter" annotations, `delta` contains (2) as well. This is
    because such annotations are used even if they cannot be matched against
    any pack argument, see :meth:`_flush_remaining_pack_arguments`.

    Arguments of expected shapes which cannot be packed:

    The logic here needs to be fit to how the computation graph for the model
    is created. In particular, you need to (1) know the shapes of all possible
    packed arguments (see `_initialize_shapes`), and (2) understand whether
    there are nodes with one of these shapes which cannot be packed. If there
    are such nodes, we may want to understand when they arise.

    For (1), we need to specialize certain annotations to how SDPA is being
    computed (for example, the "ext-*" or "padded-query" kind).
    For (2), we have a mechanism to remove pack arguments which cannot be
    matched to annotations after `max_match_trials_pack_arg` calls of
    :meth:`pack_hook`. This means we only create annotations for nodes which
    we know how to pack.

    Adjusting `max_match_trials_pack_arg`:

    It is important to properly choose `max_match_trials_pack_arg`, at least
    in situations where there are a substantial number of pack arguments of
    target shapes which cannot be matched against annotations. Namely, once
    a pack argument exists for `max_match_trials_pack_arg` calls of
    :meth:`pack_hook`, it will be removed as unmatched (and not packed).

    If `max_match_trials_pack_arg` is too large, pack arguments stick around
    for longer, and there will be more overhead in failed matchings. This can
    be substantial.
    If `max_match_trials_pack_arg` is too small, pack arguments may be removed
    before they can be matched against annotations, which reduces memory
    savings, and may lead to OOM errors.

    You can adjust `max_match_trials_pack_arg` by monitoring the
    `num_unmatched_scatter_cat` field in :meth:`annotation_usage_log`. As
    long as this is 0, `max_match_trials_pack_arg` can be decreased.

    Packing for scatter with grace period:

    A special case arises for nodes arising by `scatter` when
    `grace_period > 0` and the cache is full, see :class:`AttnWeightsKVCache`.
    We have `num = num1 + num2` and `num2 = min(num, grace_period)`. The
    operation is `x_new = f(x, index, delta, positions)`, where `positions` is
    an additional 1D index of length `num2`. Namely:

        `x_new = scatter(x, index_ext, delta_ext)`
        `delta_ext = cat(x.detach()[positions], delta)`
        `index_ext = cat(index, positions)`

    We reconstruct `x` from `x_new` using `index`, `positions`, and
    `delta_rev = gather(x, index)`:

        `a = gather(x_new, index[:num2])`
        `x = scatter(x_new, cat(index, positions), cat(delta_rev, a))`

    Annotations matched several times:

    Several pack arguments can match to the same annotation. We do not remove
    an annotation once matched, but just mark it (using `match_id`).
    We also count IDs return more than once in `_id_counts`. In
    :meth:`unpack_hook`, we store unpacked tensors in `_id_to_unpacked` so they
    can be returned once more for a duplicate.

    Dealing with different forward and backward traversal orderings:

    :meth:`pack_hook` is called along the forward traversal,
    :meth:`unpack_hook` along the backward traversal. It turns out the backward
    traversal ordering is not exactly the reverse of the forward traversal
    ordering. This is a problem: if "ext-key" (l_idx, c_idx) comes before
    "scatter-key" (l_idx, c_idx) in the backward traversal, we have noy yet
    reconstructed the input to "ext-key". We solve this by
    `prior_annotation` for the "ext-key" annotation referring to the
    "scatter-key" annotation, which can then be done earlier (and skipped
    later).

    Packing broadcast-extended indexes:

    A broadcast-extended index is a 4D tensor of integer dtype, obtained as
    `expand_index(index, head_size)` from a 3D index. We use such indexes a lot
    with scatter and gather. They are packed without needing annotations. Once
    a pack argument is determined to be a broadcast-extended index, it is packed
    with the 3D `index`.

    Logging:

    If `log_all_shapes` is set, we count the shapes of all nodes, as passed to
    the pack hook, both annotated ones and others. This can be used to develop
    a model for memory size.

    """
    def __init__(
        self,
        config: Config,
        batch_size: int,
        max_match_trials_pack_arg: int = 6,
        log_all_shapes: bool = False,
        debug_test_args: bool = False,
        arrays_cleanup: Optional[ArraysForCleanup] = None,
        track_largest_shape: bool = False,
    ):
        """
        Args:
            config: Model configuration
            batch_size: Batch size
            max_match_trials_pack_arg: Arguments of :meth:`pack_hook` are matched
                against annotations. A pack argument is removed (and not
                packed) if it is not matched after this number of
                :meth:`pack_hook` calls. This avoids running up costs trying to
                match pack args over and over, which can be significant
            log_all_shapes: See header comment. Defaults to `False`
            debug_test_args: If `True`, pack args are stored with annotations.
                This is only for testing!

        """
        self.batch_size = batch_size
        self.n_query_groups = config.n_query_groups
        self.n_head = config.n_head
        self.head_size = config.head_size
        self._max_comp_pack_arg = max_match_trials_pack_arg
        self.log_all_shapes = log_all_shapes
        self._debug_test_args = debug_test_args
        self._debug_log_args = []
        self._node_annotations = Annotations()
        self._arrays_cleanup = arrays_cleanup
        # To be initialized in `initialize_cell`
        self.eff_num_layers = None
        self.num_chunks = None
        self.first_layer_idx = None
        self.first_chunk_idx = None
        # This set contains shapes to look out for
        self._shapes_to_match = None
        # Maintains list of pack hook arguments not yet matched
        self._pack_arguments = None
        # Contains entry for every match of pack argument to annotation. This is
        # worked off in :meth:`unpack_hook`
        self._packed_arg_for_id = None
        self._id_counts = None
        self._id_to_unpacked = None
        # ID assigned to next pack hook argument
        self._next_id = None
        self._num_matched_annotations = None
        self._num_comparisons = None
        self._num_4d_indexes = None
        self._num_unmatched_scatter_cat = None
        self._unmatched_pack_args = None
        self._remaining_annotations = None
        self._shapes_counter = None
        self.replay_logs = None
        self.cache_lengths = None
        self.grace_period = None
        # If this is set, we log all annotations being matched
        self.debug_print_annotations = False
        self._track_largest_shape = track_largest_shape
        self._largest_shape = None

    def initialize_cell(
        self,
        eff_num_layers,
        num_chunks,
        first_layer_idx,
        first_chunk_idx,
        cache_lengths: List[int],
        replay_logs: List[KVCacheReplayLog],
    ):
        """
        Has to be called before a new cell is processed.

        Args:
            eff_num_layers: Number of layers in this cell
            num_chunks: Number of chunks in this cell
            first_layer_idx: Index of first layer in this cell
            first_chunk_idx: Index of first chunk in this cell
            cache_lengths: List of cache lengths, size `eff_num_layers`
            replay_logs: Replay logs for these layers

        """
        if eff_num_layers <= 0:
            raise ValueError(f"eff_num_layers {eff_num_layers} must be positive integer")
        if len(cache_lengths) != eff_num_layers or not all(x > 0 for x in cache_lengths):
            raise ValueError(f"cache_lengths = {cache_lengths} invalid, must have length {eff_num_layers}")
        assert num_chunks > 0
        assert first_layer_idx >= 0
        assert first_chunk_idx >= 0
        assert len(replay_logs) == eff_num_layers
        self.eff_num_layers = eff_num_layers
        self.num_chunks = num_chunks
        self.first_layer_idx = first_layer_idx
        self.first_chunk_idx = first_chunk_idx
        self.replay_logs = replay_logs
        self.cache_lengths = cache_lengths.copy()
        self.grace_period = replay_logs[0].grace_period
        self._node_annotations.clear()
        self._initialize_shapes(cache_lengths, first_chunk_idx)
        self._pack_arguments: List[PackHookArgument] = []
        self._packed_arg_for_id: Dict[int, PackedArgumentType] = dict()
        self._id_counts: Dict[int, int] = dict()
        self._id_to_unpacked: Dict[int, torch.Tensor] = dict()
        self._next_id = 0
        self._num_matched_annotations = 0
        self._num_comparisons = 0
        self._num_4d_indexes = 0
        self._num_unmatched_scatter_cat = 0
        self._unmatched_pack_args = []
        self._remaining_annotations = None
        if self.log_all_shapes:
            self._shapes_counter = Counter()
        if self._track_largest_shape:
            # Dummy: Replaced with first shape
            self._largest_shape = LargeShapeEntry(
                shape=(0,), numel=0, dtype=torch.float32, count=0,
            )

    @property
    def arrays_cleanup(self) -> Optional[ArraysForCleanup]:
        return self._arrays_cleanup

    def clear(self):
        self.eff_num_layers = None
        self.num_chunks = None
        self.first_layer_idx = None
        self.first_chunk_idx = None
        self._shapes_to_match = None
        self._pack_arguments = None
        if self._packed_arg_for_id is not None:
            self._packed_arg_for_id.clear()
            self._packed_arg_for_id = None
        self._id_counts = None
        self._id_to_unpacked = None
        self._next_id = None
        self._num_matched_annotations = None
        self._num_comparisons = None
        self._num_4d_indexes = None
        self._num_unmatched_scatter_cat = None
        self._unmatched_pack_args = None
        self._remaining_annotations = None
        self._shapes_counter = None
        self.replay_logs = None
        self.cache_lengths = None
        self.grace_period = None
        self._node_annotations.clear()

    def _add_shape(self, shape: List[int]):
        self._shapes_to_match.add(tuple(shape))
        if self.n_head > self.n_query_groups:
            shape[1] = self.n_head
            self._shapes_to_match.add(tuple(shape))
            shape[1] = self.n_query_groups

    def _initialize_shapes(
        self,
        cache_lengths: List[int],
        first_chunk_idx: int,
    ):
        """
        Initialize `_shapes_to_match` with shapes that a :meth:`pack_hook`
        argument must have to be mapped to a certain annotation. The shape
        is mapped to the `kind` value.

        If `n_head > n_query_groups`, we also include shapes with
        `n_head` in place of `n_query_groups`. This caters for annotations
        of padded query, or extended key, value.

        """
        # Initialize `_shapes_to_match`
        self._shapes_to_match: Set[Tuple[int, ...]] = set()
        # Only "cat" arguments >= this length can happen with this value
        # of `first_chunk_idx`
        min_cat_length = sum(
            chunk.shape[-1]
            for chunk in self.replay_logs[0].token_chunks[:first_chunk_idx]
        )
        key_shape = [
            self.batch_size,
            self.n_query_groups,
            42,
            self.head_size,
        ]
        for cache_length in cache_lengths:
            key_shape[2] = cache_length
            self._add_shape(key_shape)
            # "cat" has smaller shapes
            if min_cat_length < cache_length:
                cat_length = 0
                for chunk in self.replay_logs[0].token_chunks:
                    cat_length += chunk.shape[-1]
                    if cat_length >= cache_length:
                        break
                    if cat_length >= min_cat_length:
                        key_shape[2] = cat_length
                        self._add_shape(key_shape)

    @property
    def node_annotations(self) -> Annotations:
        return self._node_annotations

    def shapes_counter(self) -> Counter:
        assert self.log_all_shapes
        return self._shapes_counter

    def largest_shape(self) -> Optional[LargeShapeEntry]:
        return self._largest_shape

    def annotation_usage_log(self) -> AnnotationUsageLog:
        """
        The log can be used to understand why annotation matching does not work
        as intended. If `log.num_unmatched_pack_args == 0`, everything is fine,
        in that all pack arguments of supported shape can be packed using
        annotations. `log.remaining_annotations` contains annotations which were
        not used. If the cell is not the first in a row, we can expect
        `2 * num_layers` remaining annotations, where `num_layers` is the number
        of layers per cell. For the first call in a row, all annotations should
        be used.

        Returns:
            Annotation usage log
        """
        if self._remaining_annotations is None:
            remaining_annotations = [
                NodeAnnotationForLog(a) for a in self._node_annotations.nodes
            ]
        else:
            remaining_annotations = self._remaining_annotations
        return AnnotationUsageLog(
            num_matched_annotations=self._num_matched_annotations,
            num_comparisons=self._num_comparisons,
            num_4d_indexes=self._num_4d_indexes,
            num_unmatched_scatter_cat=self._num_unmatched_scatter_cat,
            unmatched_pack_args=self._unmatched_pack_args.copy(),
            remaining_annotations=remaining_annotations,
        )

    def debug_log_args(self) -> Optional[List[Tuple[torch.Tensor, NodeAnnotation]]]:
        return self._debug_log_args

    def pack_hook(self, x: torch.Tensor) -> Union[torch.Tensor, int]:
        # Shape logging
        if self.log_all_shapes:
            x_key = shape_to_tuple(x)
            extra = self._map_type(x)
            if 0 in x.stride():
                extra += f"-str{x.stride()}"
            x_key += (extra,)
            self._shapes_counter[x_key] += 1
        if self._track_largest_shape:
            self._largest_shape = self._largest_shape.process(x)
        # Pack argument if shape is matched
        new_id = self._pack_argument(x)
        if new_id is not None:
            # Try to pack as broadcast-extended index
            packed_index = self._pack_4d_index(x)
            if packed_index is not None:
                # Broadcast-extended index is packed without annotation
                self._num_4d_indexes += 1
                self._packed_arg_for_id[new_id] = packed_index
                if self.debug_print_annotations:
                    print(f"Pack broadcast-extended index: {x.shape}")
                # Remove from list to be matched against annotations
                assert self._pack_arguments[-1].id == new_id  # Sanity check
                del self._pack_arguments[-1]
            else:
                # Try to match annotations: We need to remove entries from
                # `_pack_arguments` as soon as possible
                _id = self._match_annotations()
                if _id is not None:
                    # Happens if `x` is directly matched by an annotation which
                    # matched before and has an ID already
                    new_id = _id
            return new_id
        else:
            if self._arrays_cleanup is not None:
                self._arrays_cleanup.add(x)
            return x

    def unpack_hook(self, x: Union[torch.Tensor, int]) -> torch.Tensor:
        if self._pack_arguments:
            # Match annotations with entries in `_pack_arguments`. All non-matched
            # entries are written to `_annotation_for_id` as is, so they will not
            # be packed.
            self._match_annotations(flush_pack_args=True)
        mark_cleanup = False
        if isinstance(x, int):
            idd = x
            if idd in self._id_to_unpacked:
                # Unpacked this one before: Just return it
                x = self._id_to_unpacked[idd]
                self._id_counts[idd] -= 1
                if self._id_counts[idd] == 0:
                    # Not needed anymore:
                    del self._id_to_unpacked[idd]
            else:
                value = self._packed_arg_for_id.pop(idd)
                if isinstance(value, PackArgumentAsAnnotation):
                    # Reconstruct buffer from annotation for successor
                    annotation = value.annot
                    x = self._unpack_from_annotation(annotation)
                    target_dtype = value.target_dtype
                    if target_dtype is not None and x.dtype != target_dtype:
                        x = x.to(dtype=target_dtype)
                    if self._debug_test_args:
                        self._debug_log_args.append((x.clone(), annotation))
                    if self._id_counts.get(idd, 0) >= 2:
                        # This ID will come up at least once more
                        self._id_to_unpacked[idd] = x
                        self._id_counts[idd] -= 1
                elif isinstance(value, PackArgumentAsIndex):
                     x = expand_index(value.index_3d, value.final_dim)
                else:
                    # Unmatched pack argument
                    assert isinstance(value, torch.Tensor)  # Sanity check
                    x = value
                    mark_cleanup = True
        else:
            mark_cleanup = True
        if self._arrays_cleanup is not None and mark_cleanup:
            self._arrays_cleanup.remove(x)
        return x

    def _unpack_from_annotation(
        self, annotation: NodeAnnotation,
    ) -> torch.Tensor:
        kind = annotation.kind
        if kind == "padded-query":
            if self.debug_print_annotations:
                print(f"_unpack_from_annotation: {str(annotation)}")
            buffer = self._unpack_padded_query(annotation)
        else:
            layer_idx = annotation.layer_idx
            chunk_idx = annotation.chunk_idx
            buffer, final_idx = self._node_annotations.get_final(
                layer_idx, kind,
            )
            if self.debug_print_annotations:
                print(f"_unpack_from_annotation: {str(annotation)}, buffer={buffer.shape}, final_idx={final_idx}")
            # This is complex, because it happens that "ext-*" appears before
            # "scatter-*" or "cat-*" for the same node. In this case, we need
            # to first execute this "prior annotation", since otherwise the
            # input to "ext-*" does not exist.
            annotations_todo = [annotation]
            if annotation.is_ext:
                if final_idx == chunk_idx + 1:
                    # ext-* annotation comes too early, need to do another one first
                    prior_annotation = self._find_prior_annotation(annotation)
                    if self.debug_print_annotations:
                        print(f"--> Doing {str(prior_annotation)} first")
                    annotations_todo.insert(0, prior_annotation)
                elif final_idx != chunk_idx:
                    raise ValueError(f"Annotation {str(annotation)}: final chunk_idx = {final_idx}, must be in [{chunk_idx}, {chunk_idx + 1}]")
            else:
                if final_idx == chunk_idx:
                    # Has already been done to support ext-* annotation
                    if self.debug_print_annotations:
                        print("--> Skip (already done)")
                    annotations_todo = []
                elif final_idx != chunk_idx + 1:
                    raise ValueError(f"Annotation {str(annotation)}: final chunk_idx = {final_idx}, must be in [{chunk_idx}, {chunk_idx + 1}]")
            for annot in annotations_todo:
                if annot.is_ext:
                    buffer = self._unpack_extended(
                        buffer, annot, self.n_head, self.head_size,
                    )
                else:
                    length = annot.shape[2]
                    if annot.is_scatter:
                        # Overwrites `buffer`:
                        self._unpack_scatter(buffer, annot, self.grace_period)
                        cache_length = self._get_cache_length(layer_idx)
                        assert length == cache_length  # Sanity check
                    else:
                        buffer = buffer[:, :, :length, :]
                    self._node_annotations.set_final(
                        buffer, layer_idx, chunk_idx, kind,
                    )
            # Sanity check
            final_idx = self._node_annotations.get_final(layer_idx, kind)[1]
            if final_idx != chunk_idx:
                raise IndexError(f"kind={kind}, layer_idx={layer_idx}, chunk_idx={chunk_idx}, final_chunk_idx={final_idx}")
        return buffer

    def _get_cache_length(self, layer_idx: int) -> int:
        return self.cache_lengths[layer_idx - self.first_layer_idx]

    def _find_prior_annotation(self, annotation: NodeAnnotation) -> NodeAnnotation:
        assert annotation.is_ext
        layer_idx = annotation.layer_idx
        chunk_idx = annotation.chunk_idx
        is_keys = annotation.is_keys
        result = next(
            (
                e.annot
                for e in self._packed_arg_for_id.values()
                if (
                    isinstance(e, PackArgumentAsAnnotation)
                    and e.annot.is_keys == is_keys
                    and e.annot.layer_idx == layer_idx
                    and e.annot.chunk_idx == chunk_idx
                    and (e.annot.is_scatter or e.annot.is_cat)
                )
            ),
            None
        )
        if result is None:
            raise IndexError(f"{str(annotation)}: Don't find prior annotation for this one!")
        return result

    @staticmethod
    def _unpack_extended(
        buffer: torch.Tensor,
        annotation: NodeAnnotation,
        n_head: int,
        head_size: int,
    ) -> torch.Tensor:
        # Extended keys, values may be permuted
        sort_index = None if annotation.extra_info is None else annotation.extra_info.get("sort_index")
        if sort_index is not None:
            buffer = torch.gather(
                buffer, 2, expand_index(sort_index, head_size),
            )
        buffer = repeat_interleave(buffer, n_head)
        return buffer

    @staticmethod
    def _unpack_scatter(
        buffer: torch.Tensor, annotation: NodeAnnotation, grace_period: int,
    ):
        if grace_period > 0:
            # If there is a grace period, the unpacking is slightly more complex,
            # see header comment of :class:`CellComputationAutogradHooks`
            assert annotation.positions is not None  # Sanity check
            num = annotation.delta.shape[2]
            num2 = min(num, grace_period)
            buff_cp = buffer.gather(2, annotation.index[:, :, :num2, :])
            positions = annotation.positions.view(1, 1, -1, 1).expand(
                *buffer.shape[:2], -1, buffer.shape[-1],
            )
            buffer.scatter_(2, positions, buff_cp)
        buffer.scatter_(2, annotation.index, annotation.delta)

    @staticmethod
    def _unpack_padded_query(annotation: NodeAnnotation) -> torch.Tensor:
        assert annotation.kind == "padded-query"  # Sanity check
        shape = annotation.shape
        delta = annotation.delta
        q_len = delta.shape[2]
        kv_len = shape[2]
        return torch.cat(
            (
                torch.zeros(
                    (1, 1, 1, 1), dtype=delta.dtype, device=delta.device,
                ).expand(*shape[:2], kv_len - q_len, shape[3]),
                delta,
            ),
            dim=2,
        )

    def _map_type(self, x: torch.Tensor) -> str:
        x_dtype = x.dtype
        if x_dtype == torch.int64:
            return "I64"
        elif x_dtype == torch.uint32:
            return "U32"
        elif x_dtype in (torch.float32, torch.float16, torch.bfloat16):
            return "D"
        else:
            return "O"

    def _pack_argument(self, x: torch.Tensor) -> Optional[int]:
        shape = shape_to_tuple(x)
        new_id = None
        if shape in self._shapes_to_match:
            new_id = self._next_id
            self._pack_arguments.append(
                PackHookArgument(id=new_id, shape=shape, x=x)
            )
            self._next_id += 1
        return new_id

    # TODO: Do we need to protect `self._node_annotations.nodes` against
    # multi-threaded access?
    def _match_annotations(
        self, flush_pack_args: bool = False,
    ) -> Optional[int]:
        """
        In :meth:`pack_hook` calls, we add candidates for packing into
        `_pack_arguments`. On the other side, annotations are entered by the
        forward code into `_node_annotations.nodes`. Here, we find all pairwise
        matches between these two sides.

        If `flush_pack_args == True`, the remaining `_pack_arguments`
        are entered into `_annotation_for_id` as they are. These arguments
        will not be packed. Ideally, there should be no unmatched pack
        arguments. Also, all remaining annotations are converted into a form
        for logging and then cleared.

        If the final entry of `_pack_arguments` is matched by an annotation
        with `annotation.match_id` set, this ID is returned. Otherwise, `None`
        is returned.

        """
        # Run pairwise matching: Outer over pack args, inner over annotations
        ret_id = None
        if self._node_annotations.nodes:
            new_entry_pos = len(self._pack_arguments) - 1
            rem_pack_args = []  # Collect unmatched pack args
            for pa_pos, pack_arg in enumerate(self._pack_arguments):
                parg_matched_to = None
                was_compared = False
                for an_pos, annotation in reversed(
                    list(enumerate(self._node_annotations.nodes))
                ):
                    parg_matched_to, was_compared = self._single_match(
                        pack_arg, annotation,
                    )
                    if parg_matched_to is not None:
                        parg_matched_to = (an_pos, parg_matched_to)
                        if annotation.match_id is not None and pa_pos == new_entry_pos:
                            ret_id = annotation.match_id
                        break
                    else:
                        # Avoids attempts to match in the future
                        annotation.does_not_match.add(pack_arg.id)

                if parg_matched_to is not None:
                    # We do not delete a matched annotation, but mark it. This
                    # way, we can detect multiple matches
                    ind, idd = parg_matched_to
                    annotation = self._node_annotations.nodes[ind]
                    if annotation.match_id is None:
                        annotation = replace(
                            annotation,
                            match_id=idd,
                        )
                        self._node_annotations.nodes[ind] = annotation
                else:
                    # Pack argument was not matched
                    if was_compared:
                        pack_arg = pack_arg.increase_count()
                    if pack_arg.count >= self._max_comp_pack_arg:
                        # We store full pack arguments in `_unmatched_pack_args`
                        # temporarily, until :meth:`_flush_remaining_pack_arguments`
                        # is called, and they are transformed to take less memory.
                        if self.debug_print_annotations:
                            print(f"Remove pack arg ID={pack_arg.id} after {pack_arg.count} match trials")
                        self._unmatched_pack_args.append(pack_arg)
                    else:
                        rem_pack_args.append(pack_arg)
            # Unmatched pack arguments:
            self._pack_arguments = rem_pack_args
        if flush_pack_args:
            self._flush_remaining_pack_arguments()
        return ret_id

    def _single_match(
        self,
        pack_arg: PackHookArgument,
        annotation: NodeAnnotation,
    ) -> Tuple[Optional[int], bool]:
        parg_shape = pack_arg.shape
        parg_dtype = pack_arg.x.dtype
        parg_matched_to = None
        was_compared = False
        idd = pack_arg.id  # ID of pack argument
        if annotation.shape == parg_shape and idd not in annotation.does_not_match:
            # Shapes matches, and no prior match failure
            assert annotation.delta is not None  # sanity check
            annot_dtype = annotation.delta.dtype
            same_dtype = parg_dtype == annot_dtype
            try_with_cast = (not same_dtype) and parg_dtype == torch.float32
            if same_dtype or try_with_cast:
                # Either same dtype, or `parg` is `float32`
                was_compared = True
                self._num_comparisons += 1
                parg_delta = self._delta_for_pack_argument(
                    x=pack_arg.x, annotation=annotation,
                )
                if try_with_cast:
                    parg_delta = parg_delta.to(dtype=annot_dtype)
                annot_delta = self._delta_for_annotation(annotation)
                if torch.allclose(
                    annot_delta, parg_delta, atol=1e-6, rtol=1e-4,
                ):
                    # Match confirmed: delta's the same
                    # Only used for testing:
                    if annotation.debug_full_arg is not None:
                        if self.debug_print_annotations:
                            print("Checking " + str(annotation))
                        torch.testing.assert_close(
                            pack_arg.x, annotation.debug_full_arg,
                        )
                    if self._debug_test_args:
                        annotation = replace(
                            annotation,
                            debug_full_arg=pack_arg.x.clone(),
                        )
                    if annotation.match_id is None:
                        # Annotation is transformed before
                        # being stored in `_packed_arg_for_id`
                        value = PackArgumentAsAnnotation(
                            annot=self._transform_annotation(
                                annotation, pack_arg.x,
                            ),
                            target_dtype=parg_dtype if try_with_cast else None,
                        )
                        if self.debug_print_annotations:
                            deb_msg = f"Matched {str(annotation)}"
                            if annotation.kind == "padded-query":
                                deb_msg += f", q_len={annotation.extra_info['q_len']}"
                            elif annotation.is_ext:
                                si_shape = annotation.extra_info
                                si_shape = None if si_shape is None else si_shape.get("sort_index")
                                si_shape = None if si_shape is None else si_shape.shape
                                deb_msg += f", sort_index={si_shape}"
                            print(deb_msg)
                        self._packed_arg_for_id[idd] = value
                    else:
                        # Annotation has been matched before, has ID already
                        if self.debug_print_annotations:
                            print(f"Matched again {str(annotation)}")
                        idd = annotation.match_id
                        # `_id_counts` contains IDs of annotations matched >= 2x
                        if idd in self._id_counts:
                            self._id_counts[idd] += 1
                        else:
                            self._id_counts[idd] = 2
                    # Count match and store info to mark annotation
                    self._num_matched_annotations += 1
                    parg_matched_to = idd
        return parg_matched_to, was_compared

    def _flush_remaining_pack_arguments(self):
        """
        This helper method is called with the first :meth:`unpack_hook` call.
        It marks the boundary between forward (pack) and backward (unpack).
        We do several things here:

        - Loop over unmatched annotations of kind "scatter" or "cat", try
          to match against args in `_unmatched_pack_args`. If this does
          not succeed, append to `_packed_arg_for_id` with new IDs. This
          ensures that the "annotation chain" needed to reconstruct all
          keys and values is complete in `_packed_arg_for_id`. Even if
          a "cat" or "scatter" annotation is not matched to a pack
          argument, it may be required to serve "ext" annotations. Namely,
          it could happen that `autograd` stores an "ext" node in its
          graph, but not the associated "cat" / "scatter" node.
        - All entries in `_unmatched_pack_args` and remaining pack args are
          flushed, i.e. they are not packed.
        - Annotations with `match_id is None` are converted to a cheaper
          form for logging and clear the annotations list.

        """
        # Flush remaining unmatched annotations
        self._remaining_annotations = []
        for annotation in self._node_annotations.nodes:
            if annotation.match_id is None:
                # One more attempt to match this annotation
                matched_pos = None
                for pa_pos, pack_arg in enumerate(self._unmatched_pack_args):
                    if self._single_match(
                        pack_arg, annotation,
                    )[0] is not None:
                        matched_pos = pa_pos
                        break
                if matched_pos is not None:
                    del self._unmatched_pack_args[matched_pos]
                else:
                    # Annotation remains not matched
                    self._remaining_annotations.append(NodeAnnotationForLog(annotation))
                    # We need all "cat", "scatter" annotations, also if they have
                    # not been matched
                    if annotation.is_scatter or annotation.is_cat:
                        self._num_unmatched_scatter_cat += 1
                        if self.debug_print_annotations:
                            print(f"Unmatched {str(annotation)}: Add with ID={self._next_id}")
                        self._packed_arg_for_id[self._next_id] = PackArgumentAsAnnotation(
                            annot=self._transform_annotation(annotation, None),
                            target_dtype=None,
                        )
                        self._next_id += 1
        self._node_annotations.nodes.clear()
        # Flush all remaining pack arguments (these will not be packed)
        rem_pack_args = self._unmatched_pack_args + self._pack_arguments
        self._unmatched_pack_args.clear()
        for pack_arg in rem_pack_args:
            self._packed_arg_for_id[pack_arg.id] = pack_arg.x
            # For logging (strip expensive content):
            self._unmatched_pack_args.append(
                UnmatchedPackHookArgument(
                    id=pack_arg.id,
                    shape=shape_to_tuple(pack_arg.x),
                    dtype=pack_arg.x.dtype,
                )
            )
            if self._arrays_cleanup is not None:
                self._arrays_cleanup.add(pack_arg.x)
        self._pack_arguments.clear()

    @staticmethod
    def _delta_for_annotation(annotation) -> torch.Tensor:
        delta = annotation.delta
        if annotation.is_scatter:
            delta = delta[:, :, :MAX_DELTA_TRANS_LENGTH, :]
        return delta

    @staticmethod
    def _delta_for_pack_argument(
        x: torch.Tensor, annotation: NodeAnnotation,
    ) -> torch.Tensor:
        index = annotation.index
        if annotation.is_scatter:
            index = index[:, :, :MAX_DELTA_TRANS_LENGTH, :]
        return x.gather(2, index)

    @staticmethod
    def _transform_annotation(
        annotation: NodeAnnotation,
        x: Optional[torch.Tensor],
    ) -> NodeAnnotation:
        """
        Helper for :meth:`_match_annotations`. For certain kinds of
        annotations, they need to be transformed before inserted into
        `_packed_arg_for_id`. This is because `index, delta` serve two
        different purposes for some annotations, namely (1) matching against
        pack arguments, and (2) reconstruction during unpacking.

        Args:
            annotation: Annotation to transform, (1) -> (2)
            x: Pack argument matched with this annotation. If
                `annotation.is_scatter`, this is not required, see also
                :meth:`_flush_remaining_pack_arguments`.

        Returns:
            Transformed annotation, fit for purpose (2)

        """
        if not annotation.is_scatter and x is None:
            raise ValueError("x must be given for non-scatter annotation")
        if annotation.kind == "padded-query":
            # We store the full query (without padding) in `delta`. This is
            # different from what is stored there for matching
            q_len = annotation.extra_info.get("q_len")
            assert q_len is not None and 0 < q_len <= x.shape[2], f"q_len={q_len}, x.shape[2]={x.shape[2]}"
            return replace(annotation, delta=x[:, :, (-q_len):, :])
        elif annotation.is_scatter:
            index_len = None if annotation.extra_info is None else annotation.extra_info.get("index_len")
            if index_len is not None:
                index = annotation.index[:, :, :index_len, :]
                delta = annotation.delta[:, :, :index_len, :]
                return replace(annotation, index=index, delta=delta)
            else:
                return annotation
        else:
            # Strip out `delta`, `index`. These are not needed for (2), and can
            # be large
            return replace(annotation, delta=None, index=None)

    @staticmethod
    def _pack_4d_index(x: torch.Tensor) -> Optional[PackArgumentAsIndex]:
        """
        Helper for :meth:`pack_hook`. `x` must be 4D of integer
        type. The assumption is that the final dimension is
        broadcast-extended. We run a (randomized) test for this (full test is
        too expensive). If positive, we return a packed version, otherwise
        `None`.

        """
        if x.ndim != 4 or x.dtype not in (torch.int64, torch.int32, torch.int):
            return None
        final_dim = x.shape[-1]
        rem_size = x.numel() // final_dim
        check_ind = torch.randperm(rem_size)[:5].to(device=x.device)
        arg1 = x.view(-1, final_dim)[check_ind, :]
        arg2 = arg1[:, 0].unsqueeze(-1).expand(-1, final_dim)
        if arg1.equal(arg2):
            return PackArgumentAsIndex(
                index_3d=x[..., 0].clone(), final_dim=final_dim,
            )
        else:
            return None


def debug_compute_ratio(a: torch.Tensor, b: torch.Tensor):
    """
    Helper function to check whether deltas from annotation and pack argument
    differ only by a scalar multiplier. This happens if inner products in
    self attention are done without normalizing the arguments first.

    """
    assert a.shape == b.shape
    a_dtype = a.dtype
    b_dtype = b.dtype
    a = a.to(dtype=torch.float32)
    b = b.to(dtype=torch.float32)
    thres = 1e-12
    ind = torch.logical_and(b >= 0, b < thres)
    b[ind] = thres
    ind = torch.logical_and(b < 0, b > -thres)
    b[ind] = -thres
    ratio = a / b
    max_val = ratio.max().item()
    min_val = ratio.min().item()
    print(f"{a.shape}, {a_dtype}, {b_dtype}: Ratio in [{min_val}, {max_val}]")


def create_random_index(
    shape: Tuple[int, int, int, int],
    length: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    assert len(shape) == 4
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.int64
    index_kwargs = dict(dtype=dtype, device=device)
    result = torch.empty(shape[:-1], **index_kwargs)
    num = min(shape[2], length)
    for b in range(shape[0]):
        for h in range(shape[1]):
            result[b, h, :] = torch.randperm(length, **index_kwargs)[:num]
    return expand_index(result, shape[-1])
