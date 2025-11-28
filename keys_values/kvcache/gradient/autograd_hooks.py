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
from typing import List, Union, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field, replace
from collections import Counter

import torch

from litgpt.config import Config

from keys_values.kvcache.base import KVCacheReplayLog
from keys_values.kvcache.gradient.cleanup import ArraysForCleanup
from keys_values.kvcache.utils import shape_to_tuple, bytes_for_torch_dtype
from keys_values.utils import expand_index

_ANNOTATION_KIND_VALUES = {
    "cat-key",
    "cat-value",
    "final-key",
    "final-value",
    "ignore-headgrad",
    "ignore-query",
    "scatter-key",
    "scatter-value",
}


@dataclass()
class NodeAnnotation:
    """
    Note: If the node-creating operation is `x_new = f(x, index, delta)`, the
    information recorded is for reconstructing `x` (not `x_new`). For example,
    `shape = x.shape`, and `delta`, `index` refer to a part of `x`.

    """
    kind: str
    layer_idx: int
    chunk_idx: int
    shape: Tuple[int, ...]
    index: torch.Tensor
    delta: torch.Tensor
    positions: Optional[torch.Tensor] = None
    debug_full_arg: Optional[torch.Tensor] = None
    debug_msg: Optional[str] = None
    match_id: Optional[int] = None

    def __post_init__(self):
        assert self.layer_idx >= 0
        assert self.chunk_idx >= 0
        self.kind_is_valid(self.kind)
        device = self.delta.device
        assert self.index.device == device, f"delta.device = {device}, index.device = {self.index.device}, must be the same"
        if self.positions is not None:
            assert self.is_scatter, "positions only with scatter"
            assert self.positions.ndim == 1, "positions must be a 1D tensor"
            assert self.positions.device == device, f"delta.device = {device}, positions.device = {self.positions.device}, must be the same"

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
    def is_final(self) -> bool:
        return self.kind_is_final(self.kind)

    @staticmethod
    def kind_is_final(kind: str) -> bool:
        return kind.startswith("final")

    @property
    def is_ignore(self) -> bool:
        return self.kind_is_ignore(self.kind)

    @staticmethod
    def kind_is_ignore(kind: str) -> bool:
        return kind.startswith("ignore")

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

    def clear(self):
        self.nodes.clear()
        self.final_keys.clear()
        self.final_values.clear()

    def set_final(self, x: torch.Tensor, layer_idx: int, kind: str):
        is_keys = NodeAnnotation.kind_is_keys(kind)
        target_dct = self.final_keys if is_keys else self.final_values
        curr_val = target_dct.get(layer_idx)
        if curr_val is not None and curr_val.shape == x.shape:
            target_dct[layer_idx][:] = x
        else:
            target_dct[layer_idx] = x.clone()

    def get_final(self, layer_idx: int, kind: str) -> torch.Tensor:
        is_keys = NodeAnnotation.kind_is_keys(kind)
        target_dct = self.final_keys if is_keys else self.final_values
        return target_dct[layer_idx]


@dataclass(frozen=True)
class PackHookArgument:
    id: int
    x: torch.Tensor
    shape: Tuple[int, ...]


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
    num_matched_annotations: int
    unmatched_pack_args: List[UnmatchedPackHookArgument] = field(default_factory=list)
    remaining_annotations: List[NodeAnnotationForLog] = field(default_factory=list)

    def report(self) -> str:
        lines: List[str] = [
            f"Number of matched annotations: {self.num_matched_annotations}",
            f"Number of unmatched pack args: {len(self.unmatched_pack_args)}",
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


MAX_DELTA_TRANS_LENGTH = 32


@dataclass(frozen=True)
class MatchingEntry:
    shape: Tuple[int, ...]
    dtype: torch.dtype
    annotation: NodeAnnotation

    def kind(self) -> str:
        return self.annotation.kind


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
        else:
            return LargeShapeEntry(
                shape=self.shape,
                numel=self.numel,
                dtype=x.dtype,
                count=self.count + 1,
            )

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
    nodes form dependency chains: `(l, s) -> (l, s + 1)`. The annotation
    contains information to reconstruct `x` from `x_new`.

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

    Arguments of expected shapes which cannot be packed:

    The logic here needs to be fit to how the computation graph for the model
    is created. In particular, you need to (1) know the shapes of all possible
    packed arguments (see `_initialize_shapes`), and (2) understand whether
    there are nodes with one of these shapes which cannot be packed. If there
    are such nodes, we may want to understand when they arise.

    For (1), we rely on :class:`SDPAFunction` being used for the MHA
    computation. This stores only `query`, `key`, `value` in the graph, so we
    only need to look for shapes of `key` (since `value` has the same shape).
    For (2), we know that such "outlier" nodes arise during the processing of
    the first chunk (prefill). Namely, there are `query` nodes in this step,
    one per layer, which have the same shape as the `value` nodes. We use
    "ignore" annotations to filter these out.

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

    Logging:

    If `log_all_shapes` is set, we count the shapes of all nodes, as passed to
    the pack hook, both annotated ones and others. This can be used to develop
    a model for memory size.

    """
    def __init__(
        self,
        config: Config,
        batch_size: int,
        log_all_shapes: bool = False,
        debug_test_args: bool = False,
        arrays_cleanup: Optional[ArraysForCleanup] = None,
        track_largest_shape: bool = False,
    ):
        """
        Args:
            config: Model configuration
            batch_size: Batch size
            log_all_shapes: See header comment. Defaults to `False`
            debug_test_args: If `True`, pack args are stored with annotations.
                This is only for testing!

        """
        self.batch_size = batch_size
        self.n_query_groups = config.n_query_groups
        self.n_head = config.n_head
        self.head_size = config.head_size
        self.log_all_shapes = log_all_shapes
        self._debug_test_args = debug_test_args
        self._debug_log_args = None
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
        self._unmatched_pack_args = []
        self._remaining_annotations = None
        if self.log_all_shapes:
            self._shapes_counter = Counter()
        if self._debug_test_args:
            self._debug_log_args = []
        if self._track_largest_shape:
            self._largest_shape = LargeShapeEntry(
                shape=(0,), numel=0,
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
        self._unmatched_pack_args = None
        self._remaining_annotations = None
        self._shapes_counter = None
        self.replay_logs = None
        self.cache_lengths = None
        self.grace_period = None
        self._node_annotations.clear()

    def _initialize_shapes(
        self,
        cache_lengths: List[int],
        first_chunk_idx: int,
    ):
        """
        Initialize `_shapes_to_match` with shapes that a :meth:`pack_hook`
        argument must have to be mapped to a certain annotation. The shape
        is mapped to the `kind` value.

        Several modifications have been done to go from annotated buffer to
        pack hook argument, see :meth:`_transform_annotated_tensor`.

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
            self._shapes_to_match.add(tuple(key_shape))
            # "cat" has smaller shapes
            if min_cat_length < cache_length:
                cat_length = 0
                for chunk in self.replay_logs[0].token_chunks:
                    cat_length += chunk.shape[-1]
                    if cat_length >= cache_length:
                        break
                    if cat_length >= min_cat_length:
                        key_shape[2] = cat_length
                        self._shapes_to_match.add(tuple(key_shape))

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
                # Broadcast-extended index is packed without annotation. `x` is
                # not tracked in `_pack_hook_arg_to_id`.
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
                    # already matched before and has an ID already
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
                #print(f"DEBUG: {idd} from _id_to_unpacked [new cnt={self._id_counts[idd]}]")  # DEBUG
                if self._id_counts[idd] == 0:
                    del self._id_to_unpacked[idd]
            else:
                value = self._packed_arg_for_id.pop(idd)
                if isinstance(value, PackArgumentAsAnnotation):
                    # Reconstruct buffer from annotation for successor
                    target_dtype = value.target_dtype
                    value = value.annot
                    kind = value.kind
                    buffer = self._node_annotations.get_final(value.layer_idx, kind)
                    length = value.shape[2]
                    cache_length = self._get_cache_length(value.layer_idx)
                    if value.is_scatter:
                        self._unpack_scatter(buffer, value)  # Overwrites `buffer`
                        assert length == cache_length  # Sanity check
                    # Buffer stays the same for "cat", but a smaller slice is
                    # returned
                    x = self._transform_annotated_tensor(
                        buffer=buffer[:, :, :length, :],
                        kind=kind,
                    )
                    if target_dtype is not None and x.dtype != target_dtype:
                        x = x.to(dtype=target_dtype)
                    if self._debug_test_args:
                        self._debug_log_args.append((x, value))
                    if self._id_counts.get(idd, 0) >= 2:
                        # This ID will come up at least once more
                        self._id_to_unpacked[idd] = x
                        self._id_counts[idd] -= 1
                        #print(f"DEBUG: {idd} -> _id_to_unpacked [new cnt={self._id_counts[idd]}]")  # DEBUG
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

    def _get_cache_length(self, layer_idx: int) -> int:
        return self.cache_lengths[layer_idx - self.first_layer_idx]

    def _unpack_scatter(self, buffer: torch.Tensor, value: NodeAnnotation):
        if self.grace_period > 0:
            # If there is a grace period, the unpacking is slightly more complex,
            # see header comment of :class:`CellComputationAutogradHooks`
            assert value.positions is not None  # Sanity check
            num = value.delta.shape[2]
            num2 = min(num, self.grace_period)
            buff_cp = buffer.gather(2, value.index[:, :, :num2, :])
            positions = value.positions.view(1, 1, -1, 1).expand(
                self.batch_size, self.n_query_groups, -1, self.head_size,
            )
            buffer.scatter_(2, positions, buff_cp)
        buffer.scatter_(2, value.index, value.delta)

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

    def _transform_shape(
        self, shape: Tuple[int, ...], kind: str,
    ) -> Tuple[int, ...]:
        """
        Allows to cater for cases where pack arguments are transformed from
        node annotations. At the moment, due to us using
        :class:`SDPAFunction` for MHA, there is no transformation.

        """
        return shape  # No transformation

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
        annot_keys = self._matching_entries()
        ret_id = None
        if annot_keys:
            new_entry_pos = len(self._pack_arguments) - 1
            rem_pack_args = []  # Collect unmatched pack args
            for pa_pos, pack_arg in enumerate(self._pack_arguments):
                parg_shape = pack_arg.shape
                parg_dtype = pack_arg.x.dtype
                parg_matched_to = None
                for an_pos, annot_key in enumerate(annot_keys):
                    if annot_key.shape == parg_shape:
                        # Shapes matches
                        annot_dtype = annot_key.dtype
                        annotation = annot_key.annotation
                        same_dtype = parg_dtype == annot_dtype
                        try_with_cast = (not same_dtype) and parg_dtype == torch.float32
                        if same_dtype or try_with_cast:
                            # Either same dtype, or `parg` is `float32`
                            parg_delta = self._delta_for_pack_argument(
                                x=pack_arg.x,
                                annotation=annotation,
                            )
                            if try_with_cast:
                                parg_delta = parg_delta.to(dtype=annot_dtype)
                            annot_delta = self._delta_for_annotation(annotation)
                            if torch.allclose(
                                annot_delta, parg_delta, atol=1e-6, rtol=1e-4,
                            ):
                                # Deltas match as well
                                if self._debug_test_args:
                                    annotation = replace(
                                        annotation,
                                        debug_full_arg=pack_arg.x,
                                    )
                                # If the annotation is "ignore", the pack
                                # argument counts as matched, but is dropped,
                                # as it cannot be packed. This is to avoid
                                # unmatched pack args (which otherwise are a
                                # sign that something could be wrong)
                                if not annotation.is_ignore:
                                    if annotation.match_id is None:
                                        value = PackArgumentAsAnnotation(
                                            annot=annotation,
                                            target_dtype=parg_dtype if try_with_cast else None,
                                        )
                                        self._packed_arg_for_id[
                                            pack_arg.id
                                        ] = value
                                        if self.debug_print_annotations:
                                            print(f"Match  ({annotation.layer_idx},{annotation.chunk_idx}): {annotation.shape}, {annotation.kind}")
                                    else:
                                        # Annotation has been matched before, has ID already
                                        if self.debug_print_annotations:
                                            print(f"Match again ({annotation.layer_idx},{annotation.chunk_idx}): {annotation.shape}, {annotation.kind}")
                                        idd = annotation.match_id
                                        if idd in self._id_counts:
                                            self._id_counts[idd] += 1
                                        else:
                                            self._id_counts[idd] = 2
                                        #print(f"DEBUG: {idd} matched again [new cnt={self._id_counts[idd]}]")  # DEBUG
                                        if pa_pos == new_entry_pos:
                                            #print(f"DEBUG: {idd} returned by pack_hook")  # DEBUG
                                            ret_id = idd
                                else:
                                    # Pack hook arg matched against "ignore"
                                    # annotation is treated like an unmatched
                                    # pack arg, but is not counted against them
                                    self._packed_arg_for_id[
                                        pack_arg.id
                                    ] = pack_arg.x
                                    if self.debug_print_annotations:
                                        print(f"Ignore ({annotation.layer_idx},{annotation.chunk_idx}): {annotation.shape}, {annotation.kind}")
                                    if self._arrays_cleanup is not None:
                                        self._arrays_cleanup.add(pack_arg.x)
                                self._num_matched_annotations += 1
                                parg_matched_to = (an_pos, pack_arg.id)
                                break
                if parg_matched_to is not None:
                    # We do not delete the annotation, but mark it
                    ind, idd = parg_matched_to
                    annotation = annot_keys[ind].annotation
                    if not annotation.is_ignore and annotation.match_id is None:
                        annotation = replace(
                            annotation,
                            match_id=idd,
                        )
                        self._node_annotations.nodes[ind] = annotation
                else:
                    rem_pack_args.append(pack_arg)
            self._pack_arguments = rem_pack_args
        if flush_pack_args:
            # Flush all remaining pack arguments (these will not be packed)
            for pack_arg in self._pack_arguments:
                self._packed_arg_for_id[pack_arg.id] = pack_arg.x
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
            # Convert all remaining annotations into a form for logging and
            # clear them
            self._remaining_annotations = [
                NodeAnnotationForLog(annotation)
                for annotation in self._node_annotations.nodes
                if annotation.match_id is None
            ]
            self._node_annotations.nodes.clear()
        return ret_id

    def _matching_entries(self) -> List[MatchingEntry]:
        return [
            MatchingEntry(
                shape=self._transform_shape(
                    annotation.shape,
                    kind="cat" if annotation.is_cat else "scatter",
                ),
                dtype=annotation.delta.dtype,
                annotation=annotation,
            )
            for annotation in self._node_annotations.nodes
        ]

    @staticmethod
    def _delta_for_annotation(
        annotation: NodeAnnotation,
    ) -> torch.Tensor:
        if (annotation.is_scatter or annotation.is_final) and annotation.delta.shape[2] > MAX_DELTA_TRANS_LENGTH:
            return annotation.delta[:, :, :MAX_DELTA_TRANS_LENGTH, :]
        else:
            return annotation.delta

    @staticmethod
    def _delta_for_pack_argument(
        x: torch.Tensor, annotation: NodeAnnotation,
    ) -> torch.Tensor:
        # There is currently no transformation between annotation and pack hook
        # argument, so the delta is extracted in the same way as for the
        # annotation
        if annotation.is_scatter or annotation.is_final:
            num = min(annotation.delta.shape[2], MAX_DELTA_TRANS_LENGTH)
            index = annotation.index[:, :, :num, :]
            part_of_x = x.gather(2, index)
        else:
            part_of_x = x[:, :, -1:, :]
        return part_of_x

    @staticmethod
    def _pack_4d_index(x: torch.Tensor) -> Optional[PackArgumentAsIndex]:
        """
        Helper for :meth:`_match_annotations`. `x` must be 4D of integer
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

    def _transform_annotated_tensor(
        self,
        buffer: torch.Tensor,
        kind: str,
    ) -> torch.Tensor:
        """
        Allows to cater for cases where pack arguments are transformed from
        node annotations. At the moment, due to us using
        :class:`SDPAFunction` for MHA, there is no transformation.

        """
        return buffer


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


def do_ignore_annotation(
    x: torch.Tensor,
    node_annotations: Annotations,
    kind: str,
    layer_idx: int,
    chunk_idx: int = 0,
    debug_print: bool = False,
):
    x_shape = shape_to_tuple(x)
    # Dummy:
    index = torch.tensor(
        [0], dtype=torch.int64, device=x.device,
    )
    delta = x[:, :, -1:, :].detach()
    node_annotations.nodes.append(
        NodeAnnotation(
            kind=kind,
            layer_idx=layer_idx,
            chunk_idx=chunk_idx,
            shape=x_shape,
            index=index,
            delta=delta,
        )
    )
    if debug_print:
        print(f"Create ({layer_idx},{chunk_idx}): {x_shape}, {kind}")


# TODO: Remove!
def debug_spda_callback(
    key: torch.Tensor,
    value: torch.Tensor,
    query: torch.Tensor,
    full_y: torch.Tensor,
    node_annotations: Annotations,
):
    do_ignore_annotation(
        x=key,
        node_annotations=node_annotations,
        kind="ignore-key-perm",
        layer_idx=0,
    )
    do_ignore_annotation(
        x=value,
        node_annotations=node_annotations,
        kind="ignore-value-perm",
        layer_idx=0,
    )
    do_ignore_annotation(
        x=query,
        node_annotations=node_annotations,
        kind="ignore-query-padded",
        layer_idx=0,
    )
    do_ignore_annotation(
        x=full_y,
        node_annotations=node_annotations,
        kind="ignore-sdpa-full",
        layer_idx=0,
    )
