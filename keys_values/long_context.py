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
import random
from itertools import accumulate
from typing import Optional, Dict, Any, Mapping, List, Set, Tuple
from dataclasses import dataclass, replace

import torch

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.attention import MultiHeadSelfAttention
from keys_values.head_model import HeadModel
from keys_values.kvcache.base import KVCache, DefaultKVCache
from keys_values.kvcache.basics import DenseKVCache
from keys_values.kvcache.factory import (
    KVCacheFactory,
    split_name,
    deallocate_kv_cache_buffers_of_model,
)
from keys_values.kvcache.stack_layers import DefaultCellBlocks
from keys_values.kvcache.utils import (
    wrap_tqdm_if_verbose,
    VerbosityLevels,
    bytes_for_torch_dtype,
)
from keys_values.model import GPT, block_iterator


HEAD_OR_INITIAL_TENSORS_MAX_BYTES = 2 ** 31

CLOSEBY_THRESHOLD = 4


@dataclass(frozen=True)
class KVCacheArgs:
    name: str  # TODO: Different per layer
    cache_length: int  # TODO: Different per layer
    layers_per_cell: int = 1
    chunk_size: int = 16
    cache_kwargs: Optional[Dict[str, Any]] = None
    randomize_chunk_sizes: bool = False
    chunks_per_cell_multiplier: float = 1.0
    single_tokens_for_targets: bool = False,
    verbose: str = VerbosityLevels.SOME.value
    allocate_buffers: bool = False
    attention_forward_temp_size_gb: Optional[float] = None
    attention_backward_temp_size_gb: Optional[float] = None
    use_new_cache: bool = False
    max_match_trials_pack_arg: Optional[int] = None

    def __post_init__(self):
        supported_names = KVCacheFactory.supported_names()
        assert self.name in supported_names, f"name = {self.name} not supported, must be in {supported_names}"
        assert self.verbose in VerbosityLevels, f"verbose = {self.verbose} not supported, must be in {VerbosityLevels}"
        assert self.cache_length >= 1
        assert self.attention_forward_temp_size_gb is None or self.attention_forward_temp_size_gb > 0
        assert self.attention_backward_temp_size_gb is None or self.attention_backward_temp_size_gb > 0
        assert self.chunks_per_cell_multiplier >= 0.1, f"chunks_per_cell_multiplier = {self.chunks_per_cell_multiplier}, must be >= 0.1"

    @property
    def verbosity_level(self) -> VerbosityLevels:
        return VerbosityLevels(self.verbose)

    @property
    def qname(self) -> str:
        return split_name(self.name)[1]


def create_chunk_sizes(
    gpt_model: GPT,
    seq_length: int,
    chunk_size: int,
    randomize_chunk_sizes: bool,
) -> List[int]:
    """
    Creates list of chunk sizes which are compatible with the KV caches
    assigned to `gpt_model`. The following constraints need to be met:

    - The first chunk has the size of the smallest `max_prefill_length` over
      all caches
    - For every cache, its cache length must be the union of initial chunks,
      so that no cache length falls in the middle of a chunk

    """
    def get_mpl(cache: KVCache) -> int:
        mlp = cache.max_prefill_length
        return cache.cache_length if mlp is None else mlp

    mpl = min(
        get_mpl(block.attn.kv_cache) for block in block_iterator(gpt_model)
    )
    if seq_length <= mpl:
        # Does not need anything special, but should still work. Maybe one
        # of the batches is short
        chunk_sizes = [seq_length]
    else:
        points_to_cover = list(
            set(
                block.attn.kv_cache.cache_length
                for block in block_iterator(gpt_model)
                if block.attn.kv_cache.cache_length < seq_length
            )
        )
        chunk_sizes = [mpl]  # First chunk (prefill)
        num_done = mpl
        step = chunk_size // 2
        min_val = max(chunk_size - step, 1)
        max_val = min(chunk_size + step, points_to_cover[0])
        while num_done < seq_length:
            if randomize_chunk_sizes:
                c_size = random.randint(min_val, max_val)
            else:
                c_size = chunk_size
            c_size = min(c_size, seq_length - num_done)
            next_pt = num_done + c_size
            if points_to_cover and next_pt >= points_to_cover[0] - CLOSEBY_THRESHOLD:
                c_size = points_to_cover.pop(0) - num_done
            if c_size > 0:
                chunk_sizes.append(c_size)
            num_done += c_size
    assert sum(chunk_sizes) == seq_length  # Sanity check
    _assert_chunk_sizes(
        chunk_sizes, gpt_model, chunk_size, randomize_chunk_sizes,
    )
    return chunk_sizes


def _assert_chunk_sizes(
    chunk_sizes: List[int],
    gpt_model: GPT,
    chunk_size: int,
    randomize_chunk_sizes: bool,
):
    cache_lengths = [
        block.attn.kv_cache.cache_length for block in block_iterator(gpt_model)
    ]
    min_cat_length = chunk_sizes[0]
    shape_lengths: Set[int] = set()
    for cache_length in set(cache_lengths):
        shape_lengths.add(cache_length)
        # "cat" has smaller shapes
        if min_cat_length < cache_length:
            cat_length = 0
            for c_size in chunk_sizes[1:]:
                if c_size in shape_lengths:
                    lines = [
                        "Chunk sizes give rise to KV buffer shapes conflicting with chunk shapes",
                        f"chunk_sizes:   {chunk_sizes}",
                        f"cache_lengths: {cache_lengths}",
                        "Some of your caches are too small, or your chunk size is too large. You can:",
                        f"- Increase cache lengths ({min(cache_lengths)} is too short)",
                        f"- Decrease chunk size ({chunk_size}) is too large), using --kv_cache.chunk_size",
                    ]
                    if randomize_chunk_sizes:
                        lines.append(
                            "- Do not use chunk size randomization (drop --kv_cache.randomize_chunk_sizes True)"
                        )
                    raise ValueError("\n".join(lines))
                cat_length += c_size
                if cat_length >= cache_length:
                    break
                if cat_length >= min_cat_length:
                    shape_lengths.add(cat_length)


def chunk_weights_for_loss(
    head_model: HeadModel,
    targets: torch.Tensor,
    chunk_sizes: List[int],
    num_input_tokens: int,
) -> List[float]:
    num_target_entries = []
    input_pos = 0
    for num in chunk_sizes:
        num_target_entries.append(
            head_model.num_target_entries(
                get_chunk_of_targets(
                    targets=targets,
                    input_pos=input_pos,
                    chunk_size=num,
                    num_input_tokens=num_input_tokens,
                )
            )
        )
        input_pos += num
    if num_target_entries[0] is None:
        weight_per_chunk = [1] * len(num_target_entries)
    else:
        total_sum = sum(num_target_entries)
        if total_sum == 0:
            raise ValueError(
                "Targets are invalid (all of them are masked out!):\n"
                f"targets:            {targets}\n"
                f"chunk_sizes:        {chunk_sizes}\n"
                f"num_target_entries: {num_target_entries}"
            )
        weight_per_chunk = [x / total_sum for x in num_target_entries]
    return weight_per_chunk


def write_back_cache_buffers(gpt_model: GPT):
    """
    This function should be called at the end of a loop over all layers,
    to make sure all quantized KV cache buffers are written back properly.

    This is maybe overly cautious, since KV cache contents may not be
    needed anymore at the end of such a loop.

    """
    for block in block_iterator(gpt_model):
        kv_cache = block.attn.kv_cache
        if kv_cache is not None:
            kv_cache.kv_buffers.write_back()


@dataclass(frozen=True)
class ChunksForCell:
    """
    Structure of grouping of chunks into a cell. :func:`get_chunks_for_cells`
    computes this from `chunks_per_cell` and `chunk_sizes`, over all cells.
    """
    input_range: Tuple[int, int]
    first_chunk_idx: int
    chunk_ranges: List[Tuple[int, int]]

    def __post_init__(self):
        assert self.chunk_ranges[0][0] == 0
        assert all(a < b for a, b in self.chunk_ranges)
        assert all(
            a[1] == b[0] for a, b in zip(
                self.chunk_ranges[:-1], self.chunk_ranges[1:]
            )
        )
        assert self.chunk_ranges[-1][1] == self.input_range[1] - self.input_range[0]

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_ranges)


def get_chunks_for_cells(
    chunks_per_cell: List[int],
    chunk_sizes: List[int],
) -> List[ChunksForCell]:
    chunk_numbers = [0] + list(accumulate(chunks_per_cell))[:-1] + [len(chunk_sizes)]
    part_lens = [
        sum(chunk_sizes[start:end])
        for start, end in zip(chunk_numbers[:-1], chunk_numbers[1:])
    ]
    part_numbers = [0] + list(accumulate(part_lens))
    input_ranges = [
        (start, end) for start, end in zip(part_numbers[:-1], part_numbers[1:])
    ]

    def get_chunk_ranges(sizes: List[int]) -> List[Tuple[int, int]]:
        numbers = [0] + list(accumulate(sizes))
        return list(zip(numbers[:-1], numbers[1:]))

    return [
        ChunksForCell(
            input_range=input_range,
            first_chunk_idx=first,
            chunk_ranges=get_chunk_ranges(chunk_sizes[first:(first + num)]),
        )
        for input_range, first, num in zip(
            input_ranges,
            chunk_numbers[:-1],
            chunks_per_cell,
        )
    ]


def get_chunk_of_targets(
    targets: torch.Tensor,
    input_pos: int,
    chunk_size: int,
    num_input_tokens: int,
) -> torch.Tensor:
    assert targets.ndim == 2
    num_output_tokens = targets.shape[1]
    start_output = num_input_tokens - num_output_tokens
    end = input_pos + chunk_size
    if end > start_output:
        start_rel = max(input_pos - start_output, 0)
        end_rel = end - start_output
        targets_chunk = targets[:, start_rel:end_rel]
    else:
        targets_chunk = None
    return targets_chunk


def compute_loss_for_chunk(
    head_model: HeadModel,
    model_outputs_for_chunk: torch.Tensor,
    targets: torch.Tensor,
    num_input_tokens: int,
    input_pos: int,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    assert model_outputs_for_chunk.ndim == 3
    targets_chunk = get_chunk_of_targets(
        targets=targets,
        input_pos=input_pos,
        chunk_size=model_outputs_for_chunk.shape[1],
        num_input_tokens=num_input_tokens,
    )
    if targets_chunk is not None:
        targets_chunk = targets_chunk.to(device=model_outputs_for_chunk.device)
    result = head_model(
        model_outputs=model_outputs_for_chunk,
        targets=targets_chunk,
        input_pos=input_pos,
    )
    return result * scale_factor


def compute_loss_with_limited_logits_tensor(
    gpt_model: GPT,
    head_model: HeadModel,
    model_outputs_for_chunk: torch.Tensor,
    targets: torch.Tensor,
    num_input_tokens: int,
    input_pos: int,
    scale_factor: float,
) -> torch.Tensor:
    """
    Helper for `LongContextGradientModel._forward_internal`, only if
    `head_model.needs_logits() == True`. Here, `model_outputs_for_chunk` have
    been computed with `skip_lm_head=True`. We ensure that the size of
    intermediate logits tensors remain below
    :const:`HEAD_OR_INITIAL_TENSORS_MAX_BYTES`.

    """
    assert head_model.needs_logits()
    assert model_outputs_for_chunk.ndim == 3
    config = gpt_model.config
    batch_size, chunk_size, _ = model_outputs_for_chunk.shape
    weights_dtype = gpt_model.transformer.wte.weight.dtype
    bytes_per_token = batch_size * config.padded_vocab_size * bytes_for_torch_dtype(weights_dtype)
    max_chunk_size = max(HEAD_OR_INITIAL_TENSORS_MAX_BYTES // bytes_per_token, 1)
    loss_all = 0
    for off in range(0, chunk_size, max_chunk_size):
        len = min(off + max_chunk_size, chunk_size) - off
        x = gpt_model.lm_head(model_outputs_for_chunk[:, off:(off + len), :])
        loss_part = compute_loss_for_chunk(
            head_model=head_model,
            model_outputs_for_chunk=x,
            targets=targets,
            num_input_tokens=num_input_tokens,
            input_pos=input_pos + off,
            scale_factor=scale_factor,
        )
        loss_all = loss_part + loss_all
    return loss_all


def oom_exception_action(
    ex: RuntimeError,
    tmp_array_limit_gb: TemporaryArrayLimit,
    print_message: bool = True,
):
    if "out of memory" in str(ex):
        if print_message:
            print("\nCaught out of memory error. Original message:\n" + str(ex))
        old_limit = tmp_array_limit_gb()
        ret_stat = tmp_array_limit_gb.reduce()
        if ret_stat is not None:
            # Cannot reduce any further
            print(ret_stat)
            raise ex
        else:
            if print_message:
                lines = [f"Reducing '{tmp_array_limit_gb.name}' limit:"]
            else:
                lines = [
                    "",
                    f"Caught out of memory error. Reducing '{tmp_array_limit_gb.name}' limit:",
                ]
            lines.extend(
                [
                    f"Old value: {old_limit:.3f}",
                    f"New value: {tmp_array_limit_gb():.3f}",
                ]
            )
            print("\n".join(lines))
    else:
        raise ex


class GPTAndHeadModel(torch.nn.Module):
    def __init__(
        self,
        gpt_model: GPT,
        head_model: HeadModel,
    ):
        super().__init__()
        self.gpt_model = gpt_model
        self.head_model = head_model

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        scale_factor: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        self.gpt_model.load_state_dict(
            state_dict["gpt_model"], strict=strict, assign=assign,
        )
        self.head_model.load_state_dict(
            state_dict["head_model"], strict=strict, assign=assign,
        )


class LongContextInferenceModel(GPTAndHeadModel):
    """
    Wraps a `GPT` model, provides inference computation for long contexts.
    For the moment, this means that a sequence batch is processed, so that
    an evaluation score can be computed.

    TODO: Fuse this with token generation.

    The GPT model `model` must have KV caches assigned to every layer. The
    caches can be of different type, but must have a fixed `cache_length`.

    All memory required here is allocated anew for every :meth:`forward` call,
    depending on the sequence length.

    Chunk and cell sizes:

    A long batch of sequences is split into chunks. We then process each
    chunk with a forward pass, updating KV caches. Chunk sizes are determined
    anew for each call of :meth:`forward`. The batched sequences have length
    `seq_length`, each KV cache has its own `cache_length`, `max_prefill_length`.
    The first chunk's length is the minimum over these `max_prefill_length`.
    Subsequent chunk lengths are chosen such that:

    * They have length at most `chunk_size` if `randomize_chunk_sizes == False`.
        If `randomize_chunk_sizes == True`, chunk sizes are randomized around
        the value `chunk_size`.
    * The `cache_length` values of all KV caches lie at chunk boundaries.
        We make sure that for all caches, if a chunk fills up a cache, it
        ends at `cache_length`. This simplifies cache management.

    Example: Say all caches have the same `cache_length`, but at least one of
    them is :class:`H2OKVCache`. Then,
    `max_prefill_length == cache_length - 1` for technical reasons. This means
    the first chunk has length `cache_length - 1`, the second has length 1,
    and if `randomize_chunk_sizes == False`, all subsequent chunks have length
    `chunk_size`, except possibly the last one.

    Also, chunks are grouped into cells. This is done so that the sum of
    chunk sizes in each cell is approximately equal to `cache_length`.
    This determines the loop structure in :meth:`forward`. The outermost loop
    is over cells. Then, we loop over layers. The innermost loop is over
    chunks within a cell. Why?

    * Why not outer over chunks, inner over layers? Most KV caches use
        quantization. When switching between model layers, the full cache
        content needs to be de-quantized. Doing this for every chunk is too
        costly. During the innermost loop over chunks within a cell, we
        only modify de-quantized buffers, writing them back only when
        switching layers.
    * Why not outer over layers, inner over chunks? This would require to
        maintain tensors of shape `(batch_size, sequence_length, n_embd)`,
        which is not tractable.

    The cell grouping is a compromise. Tensors of shape
    `(batch_size, cache_length, n_embd)` are tractable, and de-quantization
    is not needed too often.

    The same cell grouping is used in gradient computations as well, see
    :class:`LongContextGradientModel`.

    """
    def __init__(
        self,
        gpt_model: GPT,
        head_model: HeadModel,
        chunk_size: int = 16,
        randomize_chunk_sizes: bool = False,
        chunks_per_cell_multiplier: float = 1.0,
        single_tokens_for_targets: bool = False,
        verbose: VerbosityLevels = VerbosityLevels.SOME,
        tmp_array_limit_gb: Optional[TemporaryArrayLimit] = None,
        debug_single_cell_per_row: bool = False,
    ):
        """
        If `tmp_array_limit_gb` is given, it maintains a limit on
        temporary device memory used in forward computations. Objects
        such as KV caches in `gpt_model` must keep a reference. We
        catch out of memory exceptions during forward computations,
        reduce the limit and try again.

        Args:
            gpt_model: GPT model to train on sequence data. All layers must have
                KV caches assigned, and these must not be dense. For now, all
                caches must have the same `cache_length`.
            head_model: Head model and loss function
            chunk_size: Data batches are processed in chunks of this size
                (except the first one). See above.
            randomize_chunk_sizes: If `True`, chunk sizes are randomized (with
                mean `chunk_size`). This may have advantages for model
                training. Defaults to `False`.
            chunks_per_cell_multiplier: Each cell contains a number of chunks.
                The length of a cell is the sum of lengths of its cells. We
                assign chunks to cells so that cell lengths are close to
                `int(cache_length * chunks_per_cell_multiplier)`, but not
                larger. The larger this multiplier, the fewer cells per row,
                which speeds up computation, but also memory requirements of
                gradient computation per cell scales linearly in this value.
            single_tokens_for_targets: If `True`, the targets part of a
                sequence is processed token per token (i.e., with chunk size
                1). This is slower, but more realistic, mirroring how inference
                looks like.
            verbose: Verbosity level, defaults to ``VerbosityLevels.SOME``.
                For ``VerbosityLevels.ALL``, we print deep diagnostic
                information
            tmp_array_limit_gb: See above.
            debug_single_cell_per_row: Internal option, used for unit testing.

        """
        super().__init__(gpt_model, head_model)
        self._check_args(gpt_model, chunk_size, tmp_array_limit_gb)
        self.config = gpt_model.config
        self.chunk_size = chunk_size
        self.randomize_chunk_sizes = randomize_chunk_sizes
        if chunks_per_cell_multiplier < 0.1:
            raise ValueError(f"chunks_per_cell_multiplier = {chunks_per_cell_multiplier}, must be >=0.1")
        self.chunks_per_cell_multiplier = chunks_per_cell_multiplier
        self.single_tokens_for_targets = single_tokens_for_targets
        self.verbose = verbose
        self._debug_single_cell_per_row = debug_single_cell_per_row
        cache_params = self.gpt_model.get_kv_cache_params(0)
        self._max_batch_size = cache_params.max_batch_size
        # Set max_prefill_length as minimum over all caches
        caches = [block.attn.kv_cache for block in block_iterator(gpt_model)]
        min_cache_length = min(c.cache_length for c in caches)
        mpl_values = [min_cache_length] + [
            c.max_prefill_length
            for c in caches
            if c.max_prefill_length is not None
        ]
        self._max_prefill_length = min(mpl_values)
        self.chunk_sizes = None
        self.chunks_per_cell = None
        self.batch_size = None
        self._tmp_array_limit_gb = tmp_array_limit_gb
        self._record_gpu_memory_snapshots = None
        self._record_gpu_memory_kind = None

    def _check_args(
        self,
        model: GPT,
        chunk_size: int,
        tmp_array_limit_gb: Optional[TemporaryArrayLimit],
    ):
        if chunk_size < 1:
            raise ValueError(f"chunk_size = {chunk_size}, must be >= 1")
        if tmp_array_limit_gb is not None:
            mha = model.mha
            if mha.tmp_array_limit_gb is None:
                mha.set_tmp_array_limit_gb(tmp_array_limit_gb)
            elif not (mha.tmp_array_limit_gb is tmp_array_limit_gb):
                raise ValueError("tmp_array_limit_gb and gpt_model.mha.tmp_array_limit_gb must be the same object")

        for block_idx, block in enumerate(block_iterator(model)):
            kv_cache = block.attn.kv_cache
            prefix = f"Block {block_idx} of model: "
            if kv_cache is None:
                raise ValueError(prefix + "No KV cache assigned. Use 'model.assign_kv_caches'")
            if isinstance(kv_cache, DenseKVCache):
                raise ValueError(prefix + "Needs non-dense KV cache. Use 'model.assign_kv_caches'")
            if tmp_array_limit_gb is not None and isinstance(kv_cache, DefaultKVCache):
                mha = kv_cache.mha
                if mha.tmp_array_limit_gb is None:
                    mha.set_tmp_array_limit_gb(tmp_array_limit_gb)
                elif not (mha.tmp_array_limit_gb is tmp_array_limit_gb):
                    raise ValueError(prefix + "tmp_array_limit_gb and block.attn.kv_cache.mha.tmp_array_limit_gb must be the same object")

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        scale_factor: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Different to `GPT.forward`, this is processing a batch of full
        sequences. It also evaluates the head model and computes the loss
        function.

        Args:
            input_ids: Batch of full input token sequences
            targets: Targets, these are right-aligned with `input_ids`
            scale_factor: Loss is multiplied by this factor. Defaults to 1.

        Returns:
            Loss values, shape `(batch_size,)`

        """
        self._init_members_from_tokens(input_ids, targets)
        if not isinstance(self.gpt_model.mha, MultiHeadSelfAttention):
            raise ValueError(f"type(self.gpt_model.mha) = {type(self.gpt_model.mha)}, must be MultiHeadSelfAttention")
        return self._forward_only(input_ids, targets, scale_factor)

    def clear(self):
        """
        Resets members created in `_init_members_from_tokens` to `None`.

        """
        self.chunk_sizes = None
        self.chunks_per_cell = None
        self.batch_size = None
        self._record_gpu_memory_snapshots = None
        self._record_gpu_memory_kind = None

    def _init_members_from_tokens(
        self, input_ids: torch.Tensor, targets: torch.Tensor,
    ):
        """
        Initialize members required for processing the current batch.

        """
        # Checks
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids.shape = {input_ids.shape}, must be 2D")
        batch_size, seq_length = input_ids.shape
        if not (1 <= batch_size <= self._max_batch_size):
            raise ValueError(f"input_ids.batch_size = {batch_size}, must be in [1, {self._max_batch_size}]")
        self.batch_size = batch_size
        if seq_length > self.config.block_size:
            print(
                f"\nSequence length {seq_length} > {self.config.block_size} = "
                "config.block_size. Adjusting the latter"
            )
            self.config = replace(self.config, block_size=seq_length)
            self.gpt_model.config = self.config

        if targets.ndim != 2:
            raise ValueError(f"targets.shape = {targets.shape}: Must be 2D")
        num_output_tokens = targets.shape[-1]
        if self.batch_size != targets.shape[0] or not (1 <= num_output_tokens <= seq_length):
            raise ValueError(f"targets.shape = {targets.shape}: Not compatible with batch_size = {self.batch_size} or num_input_tokens = {seq_length}")
        self.gpt_model.max_seq_length = seq_length
        # Select chunk sizes and chunks per cell
        self._select_chunks_and_cells(num_output_tokens)

    def _select_chunks_and_cells(self, num_output_tokens: int):
        # Select chunk sizes
        seq_length = self.gpt_model.max_seq_length
        if self.single_tokens_for_targets:
            seq_length -= num_output_tokens
        chunk_sizes = create_chunk_sizes(
            gpt_model=self.gpt_model,
            seq_length=seq_length,
            chunk_size=self.chunk_size,
            randomize_chunk_sizes=self.randomize_chunk_sizes,
        )
        assert all(x > 0 for x in chunk_sizes), f"chunk_sizes = {chunk_sizes}, must all be positive"
        if self.single_tokens_for_targets:
            chunk_sizes += [1] * num_output_tokens
        self.chunk_sizes = chunk_sizes
        # Select chunks per cell. This is done in a way so that the length of
        # each cell (i.e., sum of chunk lengths) is close to `cache_length`,
        # but not larger.
        if self._debug_single_cell_per_row:
            # This is used for unit testing only: Force single cell per row.
            # Do not use!
            chunks_per_cell = [len(chunk_sizes)]
        else:
            max_cache_length = max(
                block.attn.kv_cache.cache_length
                for block in block_iterator(self.gpt_model)
            )
            max_cell_length = int(
                max_cache_length * self.chunks_per_cell_multiplier
            )
            chunks_per_cell = []
            cell_length = 0
            num_chunks = 0
            for chunk_size in chunk_sizes:
                new_length = cell_length + chunk_size
                if new_length > max_cell_length:
                    assert num_chunks > 0  # Sanity check
                    chunks_per_cell.append(num_chunks)
                    cell_length = chunk_size
                    num_chunks = 1
                else:
                    cell_length = new_length
                    num_chunks += 1
            chunks_per_cell.append(num_chunks)
            assert sum(chunks_per_cell) == len(chunk_sizes)
        self.chunks_per_cell = chunks_per_cell

    def _checkpoint_layer_input(
        self,
        x: torch.Tensor,
        layer_idx: int,
        input_pos: int,
    ):
        """
        Implemented in subclasses which need layer input checkpointing

        Args:
            x: Inputs to layer `layer_idx`, starting from `input_pos`.
                The length is that of the current cell
            layer_idx: See above
            input_pos: See above

        """
        pass

    def _forward_internal(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        """
        Wrapper around :meth:`_forward_internal_no_check`. If
        `tmp_array_limit_gb` is set, we catch out of memory errors,
        reduce the limit value and try again. Done only a limited
        number of times, see :class:`TemporaryArrayLimit`.

        """
        if self._tmp_array_limit_gb is None:
            return self._forward_internal_no_check(
                input_ids, targets, scale_factor,
            )
        else:
            if self._record_gpu_memory_snapshots is not None and self._record_gpu_memory_kind == 1:
                self._record_gpu_memory_snapshots.set_path(
                    self._record_gpu_memory_snapshots.path.parent / "snapshot_forward.pickle"
                )
                self._record_gpu_memory_snapshots.start_recording()
                print(f"Start profiling GPU memory: {self._record_gpu_memory_snapshots.path}")
            result = None
            retry_count = 0
            while result is None:
                try:
                    result = self._forward_internal_no_check(
                        input_ids, targets, scale_factor,
                    )
                except RuntimeError as ex:
                    oom_exception_action(ex, self._tmp_array_limit_gb)
                    result = None
                    deallocate_kv_cache_buffers_of_model(self.gpt_model)
                    torch.cuda.empty_cache()
                    retry_count += 1
                    if self._record_gpu_memory_kind == 1 and retry_count == 2:
                        self._record_gpu_memory_snapshots.store_current_snapshot()
                        self._record_gpu_memory_snapshots.stop_recording()
                        print(f"Stop profiling GPU memory: {self._record_gpu_memory_snapshots.path}")

            return result

    def _forward_internal_no_check(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        """
        We run a nested loop with 3 levels. Over cells, then over layers, then
        over chunks per cell. This is to speed up KV caches with quantization:
        encoding and decoding only happens in the loop over layers, not in the
        innermost loop over chunks.

        """
        loss_full = 0
        num_input_tokens = input_ids.shape[-1]
        weight_per_chunk = chunk_weights_for_loss(
            head_model=self.head_model,
            targets=targets,
            chunk_sizes=self.chunk_sizes,
            num_input_tokens=num_input_tokens,
        )
        # Need grouping of chunks into cells, for outermost loop
        chunks_for_cells = get_chunks_for_cells(
            self.chunks_per_cell, self.chunk_sizes,
        )

        # Need each layer separately
        model_blocks = [
            DefaultCellBlocks(
                model=self.gpt_model,
                first_layer_idx=idx,
                num_layers=1,
            )
            for idx in range(self.config.n_layer)
        ]
        wte = self.gpt_model.transformer.wte
        alpha = self.config.n_embd ** 0.5
        wte_device = wte.weight.device
        with torch.no_grad():
            # Outermost loop over cells (group of chunks)
            for chunks_for_cell in wrap_tqdm_if_verbose(
                chunks_for_cells, verbose=self.verbose
            ):
                start, end = chunks_for_cell.input_range
                # Input embeddings
                embeddings = wte(input_ids[:, start:end].to(device=wte_device))
                if self.config.scale_embeddings:
                    embeddings = embeddings * alpha
                # Loop over layers
                for block_idx, block in enumerate(model_blocks):
                    input_pos = start
                    # Layer input checkpointing
                    self._checkpoint_layer_input(
                        x=embeddings,
                        layer_idx=block_idx,
                        input_pos=input_pos,
                    )
                    new_embed_parts = []
                    # Innermost loop over chunks per cell
                    for rel_start, rel_end in chunks_for_cell.chunk_ranges:
                        ch_size = rel_end - rel_start
                        x = embeddings[:, rel_start:rel_end, :]
                        abs_start = start + rel_start
                        idx = input_ids[:, abs_start:(abs_start + ch_size)]
                        new_embed_parts.append(
                            block.forward(
                                x=x,
                                idx=idx,
                                input_pos=input_pos,
                            )
                        )
                        input_pos += ch_size
                    assert input_pos == end  # Sanity check
                    del embeddings
                    embeddings = torch.cat(new_embed_parts, dim=1)
                # Layer input checkpointing
                self._checkpoint_layer_input(
                    x=embeddings,
                    layer_idx=self.config.n_layer,
                    input_pos=start,
                )
                # Head model
                a = chunks_for_cell.first_chunk_idx
                b = a + chunks_for_cell.num_chunks
                input_pos = start
                for (rel_start, rel_end), weight in zip(
                    chunks_for_cell.chunk_ranges,
                    weight_per_chunk[a:b],
                ):
                    output_chunk = embeddings[:, rel_start:rel_end, :]
                    if self.head_model.needs_logits():
                        loss_part = compute_loss_with_limited_logits_tensor(
                            gpt_model=self.gpt_model,
                            head_model=self.head_model,
                            model_outputs_for_chunk=output_chunk,
                            targets=targets,
                            num_input_tokens=num_input_tokens,
                            input_pos=input_pos,
                            scale_factor=weight * scale_factor,
                        )
                    else:
                        loss_part = compute_loss_for_chunk(
                            head_model=self.head_model,
                            model_outputs_for_chunk=output_chunk,
                            targets=targets,
                            num_input_tokens=num_input_tokens,
                            input_pos=input_pos,
                            scale_factor=weight * scale_factor,
                        )
                    loss_full = loss_part + loss_full
                    input_pos += (rel_end - rel_start)

        write_back_cache_buffers(self.gpt_model)  # Just to be safe
        if self.verbose is not VerbosityLevels.NONE:
            print("\nDeallocate KV cache buffers")
        deallocate_kv_cache_buffers_of_model(self.gpt_model)
        return loss_full

    def _forward_only(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        scale_factor: float,
    ) -> torch.Tensor:
        # Ensure that all KV caches do not record replay logs
        for block in block_iterator(self.gpt_model):
            block.attn.kv_cache.switch_replay_logging(False)
        if self.verbose is not VerbosityLevels.NONE:
            print(f"\nForward pass over {len(self.chunk_sizes)} chunks, grouped into {len(self.chunks_per_cell)} cells (inference mode)")
        loss_full = self._forward_internal(input_ids, targets, scale_factor)
        self.clear()
        return loss_full
