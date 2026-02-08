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
from typing import Iterator, List

import torch
from torch.utils.data import Sampler


class LongestFirstIterator(Iterator[int]):
    def __init__(
        self,
        dataset_size: int,
        inds_longest: List[int],
    ):
        batch_size = len(inds_longest)
        if not all(0 <= x < dataset_size for x in inds_longest):
            raise ValueError(
                f"inds_longest = {inds_longest}: all entries must be in [0, {dataset_size})"
            )
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        remainder = list(set(range(dataset_size)) - set(inds_longest))
        assert len(remainder) == dataset_size - batch_size
        remainder = torch.tensor(remainder)[torch.randperm(len(remainder))]
        self._permutation = torch.cat((torch.tensor(inds_longest), remainder))
        self._pos = 0

    def __next__(self) -> int:
        if self._pos >= self.dataset_size:
            raise StopIteration
        result = self._permutation[self._pos].item()
        self._pos += 1
        return result

    def __iter__(self) -> Iterator[int]:
        return self


class LongestFirstIterable(Sampler[int]):
    """
    To be used as `sampler` for :class:`torch.utils.data.DataLoader`.
    Returns `inds_longest` first, then a random permutation of the remainder
    for the rest of the epoch.

    Use this to ensure that the first batch is given by the indexes in
    `inds_longest`.

    """

    def __init__(
        self,
        dataset_size: int,
        inds_longest: List[int],
    ):
        self._kwargs = {
            "dataset_size": dataset_size,
            "inds_longest": inds_longest.copy(),
        }
        self._len = dataset_size

    def __iter__(self) -> Iterator[int]:
        return LongestFirstIterator(**self._kwargs)

    def __len__(self) -> int:
        return self._len


class SimilarSequenceLengthIterator(Iterator[int]):
    def __init__(
        self,
        sequence_lengths: List[int],
        micro_batch_size: int,
        num_devices: int,
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
        global_batch_size = micro_batch_size * num_devices
        num_chunks = math.ceil(len(sequence_lengths) / global_batch_size)
        self.dataset_size = num_chunks * global_batch_size
        self.micro_batch_size = micro_batch_size
        self.num_devices = num_devices
        self._shuffle = shuffle
        self._longest_first = longest_first
        self._shortest_first = shortest_first
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
        if num_full_batches > 0:
            start = num_full_batches * global_batch_size
            micro_batches = inds_ascending[:start].split(mbs)
            num_mb = len(micro_batches)
            assert num_mb == num_full_batches * ndev, (num_mb, num_full_batches * ndev)
            parts = [
                micro_batches[off : (off + ndev)] for off in range(0, num_mb, ndev)
            ]
        else:
            parts = []
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
            off = 0
            micro_batches = []
            for batch in inds_ascending[start:].split(sizes):
                rem_sz = self.micro_batch_size - batch.numel()
                micro_batches.append(torch.cat((batch, extra_inds[off : off + rem_sz])))
                off += rem_sz
            assert off == extra_inds.numel()  # Sanity check
            parts.append(micro_batches[:ndev])
            if rem_batches > 1:
                parts.append(micro_batches[ndev:])

        # Sanity check
        assert len(parts) == num_batches, (parts, num_batches)
        assert all(len(x) == ndev for x in parts)
        self._partition = parts

    def _create_permutation(self):
        def get_index(sz: int) -> List[int]:
            if self._shuffle:
                return torch.randperm(sz).tolist()
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
        parts = [
            self._partition[out_ind][inn_ind]
            for out_ind in out_inds
            for inn_ind in get_index(len(self._partition[out_ind]))
        ]
        self._permutation = torch.cat(parts)
        assert len(self._permutation) == self.dataset_size  # Sanity check

    def __next__(self) -> int:
        if self._pos >= self.dataset_size:
            raise StopIteration
        result = self._permutation[self._pos].item()
        self._pos += 1
        return result

    def __iter__(self) -> Iterator[int]:
        return self


class SimilarSequenceLengthIterable(Sampler[int]):
    """
    To be used as `sampler` for :class:`torch.utils.data.DataLoader`.

    Use this to create balanced batches for distributed data parallel
    training over several devices. Batches are of size `micro_batch_size`
    per device, and there are `num_devices` devices. Also, `sequence_lengths`
    is the number of tokens per sequence.

    Note: We assume the size of the dataset this sampler is applied to, is a
    multiple of `micro_batch_size * num_devices`, possibly padded at the end
    to reach this size. This means that `sequence_lengths` can be shorter than
    the dataset size. When this happens, we ensure that the shorter macro-batch
    is the last one (if `shuffle=False`), containing the longest sequences (as
    well as pad entries).

    The method ensures that:

    * Sequences in a batch (or micro-batch) are mostly closest in length
    * Next, sequences in `num_devices` consecutive batches (called
      macro-batch) are close in length
    * If one macro-batch is shorter than the others, it contains the
      longest sequences
    * Given that, if `shuffle=True`, macro-batches are ordered randomly, and
      the order of micro-batches in each macro-batch are random as well

    """

    def __init__(
        self,
        sequence_lengths: List[int],
        micro_batch_size: int,
        num_devices: int,
        shuffle: bool = True,
        longest_first: bool = False,
        shortest_first: bool = False,
    ):
        self._kwargs = {
            "sequence_lengths": sequence_lengths.copy(),
            "micro_batch_size": micro_batch_size,
            "num_devices": num_devices,
            "shuffle": shuffle,
            "longest_first": longest_first,
            "shortest_first": shortest_first,
        }
        self._len = len(sequence_lengths)

    def __iter__(self) -> Iterator[int]:
        return SimilarSequenceLengthIterator(**self._kwargs)

    def __len__(self) -> int:
        return self._len
