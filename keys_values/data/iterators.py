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
from typing import Iterator, List, Optional

import torch
from torch.utils.data import Sampler


class BatchSampler(Sampler[List[int]]):
    @property
    def batch_size(self) -> int:
        raise NotImplementedError


class SimilarSequenceLengthIterator(Iterator[List[int]]):
    def __init__(
        self,
        sequence_lengths: List[int],
        micro_batch_size: int,
        seed: int,
        num_devices: int,
        rank: int,
        shuffle: bool,
        longest_first: bool,
        shortest_first: bool,
    ):
        global_batch_size = micro_batch_size * num_devices
        num_chunks = math.ceil(len(sequence_lengths) / global_batch_size)
        self.dataset_size = num_chunks * global_batch_size
        self.num_next = num_chunks
        self.micro_batch_size = micro_batch_size
        self.num_devices = num_devices
        self.rank = rank
        self._shuffle = shuffle
        self._longest_first = longest_first
        self._shortest_first = shortest_first
        self._prng = torch.Generator().manual_seed(seed)
        self._partition = None
        self._initialize(sequence_lengths)
        self._permutation = None
        self._create_permutation()
        self._pos = 0

    def _initialize(self, sequence_lengths: List[int]):
        # Sort from shortest to longest
        len_sl = len(sequence_lengths)
        inds_ascending = torch.argsort(torch.tensor(sequence_lengths))
        # These are positions of padding entries, they are used to complete
        # micro-batches
        extra_inds = torch.arange(
            len(sequence_lengths),
            self.dataset_size,
            dtype=inds_ascending.dtype,
            device=inds_ascending.device,
        )
        ndev = self.num_devices
        mbs = self.micro_batch_size
        global_batch_size = mbs * ndev
        num_batches = self.dataset_size // global_batch_size
        size_last = (
            0
            if len_sl == self.dataset_size
            else global_batch_size - self.dataset_size + len_sl
        )
        num_full_batches = num_batches
        if size_last > 0:
            num_full_batches -= 1
            if size_last < ndev:
                num_full_batches -= 1
        parts = []
        if num_full_batches > 0:
            start = num_full_batches * global_batch_size
            micro_batches = inds_ascending[:start].split(mbs)
            num_mb = len(micro_batches)
            assert num_mb == num_full_batches * ndev, (num_mb, num_full_batches * ndev)
            for off in range(0, num_mb, ndev):
                # Randomization between devices (otherwise, lower devices
                # receive shorter sequences)
                rind = torch.randperm(ndev, generator=self._prng)
                parts.append(micro_batches[off + rind[self.rank].item()].tolist())
        else:
            start = 0
        rem_batches = num_batches - num_full_batches
        if rem_batches > 0:
            size_rest = len(sequence_lengths) - start
            num_mb = rem_batches * ndev
            mbs = math.ceil(size_rest / num_mb)
            assert mbs >= 1  # Sanity check
            fix_me = num_mb * mbs - size_rest
            assert fix_me == 0 or mbs > 1, (fix_me, mbs)  # Sanity check
            sizes = [mbs] * (num_mb - fix_me) + [mbs - 1] * fix_me
            assert sum(sizes) == size_rest  # Sanity check
            # Randomization between devices (otherwise, lower devices
            # receive shorter sequences):
            rind = torch.randperm(ndev, generator=self._prng)
            if rem_batches > 1:
                rind = torch.cat((rind, torch.randperm(ndev, generator=self._prng)))
            off = 0
            for batch, pos in zip(
                inds_ascending[start:].split(sizes),
                rind,
            ):
                rem_sz = self.micro_batch_size - batch.numel()
                if pos == self.rank:
                    parts.append(
                        torch.cat((batch, extra_inds[off : off + rem_sz])).tolist()
                    )
                off += rem_sz
                pos += 1
            assert off == extra_inds.numel()  # Sanity check

        # Sanity check
        assert len(parts) == num_batches, (parts, num_batches)
        assert all(len(x) == self.micro_batch_size for x in parts)
        self._partition = parts

    def _create_permutation(self):
        def get_index(sz: int) -> List[int]:
            if self._shuffle:
                return torch.randperm(sz, generator=self._prng).tolist()
            else:
                return list(range(sz))

        if not self._shuffle and self._permutation is not None:
            return  # Ordering does not change
        num_outer = len(self._partition)
        out_inds = get_index(
            num_outer - int(self._longest_first or self._shortest_first)
        )
        if self._longest_first:
            out_inds.insert(0, num_outer - 1)
        elif self._shortest_first:
            out_inds = [0] + [x + 1 for x in out_inds]
        self._permutation = out_inds

    def __next__(self) -> List[int]:
        if self._pos >= self.num_next:
            raise StopIteration
        part_pos = self._permutation[self._pos]
        self._pos += 1
        return self._partition[part_pos]

    def __iter__(self) -> Iterator[List[int]]:
        return self


class SimilarSequenceLengthSampler(BatchSampler):
    """
    To be used as `batch_sampler` for :class:`torch.utils.data.DataLoader`.

    Use this to create balanced batches for distributed data parallel
    training over several devices. Batches are of size `micro_batch_size`
    per device, and there are `num_devices` devices. Also, `sequence_lengths`
    is the number of tokens per sequence.

    The sampler is aware of the process rank (via `rank`) and returns indexes
    of the corresponding part of the dataset only. The dataset length (with
    padding) is `micro_batch_size * num_chunks`, where
    `num_chunks = ceil(len(sequence_lengths) / global_batch_size)` and
    `global_batch_size = micro_batch_size * num_devices`.

    Note: We assume the size of the dataset this sampler is applied to, is a
    multiple of `global_batch_size`, possibly padded at the end to reach this
    size. This means that `sequence_lengths` can be shorter than the dataset
    size. When this happens, we ensure that the shorter macro-batch is the last
    one (if `shuffle=False`), containing the longest sequences (as well as pad
    entries).

    The method ensures that:

    * Sequences in a batch (or micro-batch) are mostly closest in length
    * Next, sequences in `num_devices` consecutive batches (called
      macro-batch) are close in length
    * If one macro-batch is shorter than the others, it contains the
      longest sequences
    * Given that, if `shuffle=True`, macro-batches are ordered randomly. In
      any case, the order of micro-batches in each macro-batch are random.

    """

    def __init__(
        self,
        sequence_lengths: List[int],
        micro_batch_size: int,
        seed: int,
        num_devices: int = 1,
        rank: Optional[int] = None,
        shuffle: bool = True,
        longest_first: bool = False,
        shortest_first: bool = False,
    ):
        assert micro_batch_size >= 1
        assert num_devices >= 1
        if micro_batch_size == 1 and num_devices == 1:
            raise ValueError(
                "This sampler requires micro_batch_size > 1 or num_devices > 1"
            )
        if shortest_first and longest_first:
            raise ValueError("Cannot set both shortest_first and longest_first")
        if num_devices > 1:
            if num_devices is None or not (0 <= rank < num_devices):
                raise ValueError(f"rank = {rank}, must be in [0, {num_devices})")
        else:
            rank = 0
        self._kwargs = {
            "sequence_lengths": sequence_lengths.copy(),
            "micro_batch_size": micro_batch_size,
            "seed": seed,
            "num_devices": num_devices,
            "rank": rank,
            "shuffle": shuffle,
            "longest_first": longest_first,
            "shortest_first": shortest_first,
        }
        global_batch_size = micro_batch_size * num_devices
        self._len = math.ceil(len(sequence_lengths) / global_batch_size)
        self._batch_size = micro_batch_size

    def __iter__(self) -> Iterator[List[int]]:
        return SimilarSequenceLengthIterator(**self._kwargs)

    def __len__(self) -> int:
        return self._len

    @property
    def batch_size(self) -> int:
        return self._batch_size
