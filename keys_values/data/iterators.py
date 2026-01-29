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
from typing import Iterator, List, Iterable

import torch


class LongestFirstIterator(Iterator):
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
            self._permutation = torch.randperm(self.dataset_size)
            self._pos = 0
        result = self._permutation[self._pos].item()
        self._pos += 1
        return result

    def __iter__(self) -> Iterator:
        return self


class LongestFirstIterable(Iterable):
    """
    To be used as `sampler` for :class:`torch.utils.data.DataLoader`.
    Returns `inds_longest` first, then a random permutation of the remainder
    for the rest of the epoch. Afterwards, each epoch is a random permutation,
    independent of `inds_longest`.

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

    def __iter__(self) -> Iterator:
        return LongestFirstIterator(**self._kwargs)

    def __len__(self) -> int:
        return self._len


class SimilarSequenceLengthIterator(Iterator):
    def __init__(
        self,
        sequence_lengths: List[int],
        micro_batch_size: int,
        num_devices: int,
        longest_first: bool = False,
    ):
        assert micro_batch_size >= 1
        assert num_devices >= 1
        if micro_batch_size == 1 and num_devices == 1:
            raise ValueError("This sampler requires micro_batch_size > 1 or num_devices > 1")
        self.dataset_size = len(sequence_lengths)
        self.micro_batch_size = micro_batch_size
        self.num_devices = num_devices
        self._longest_first = longest_first
        self._partition = None
        self._initialize(sequence_lengths)
        self._pos = self.dataset_size
        self._permutation = None

    def _initialize(self, sequence_lengths: List[int]):
        # Sort from shortest to longest
        inds_ascending = torch.argsort(torch.tensor(sequence_lengths))
        # If there is a shorter macro-batch, choose it in the middle
        global_batch_size = self.micro_batch_size * self.num_devices
        num_full_batches = self.dataset_size // global_batch_size
        num_left = math.ceil(num_full_batches / 2)
        start = num_left * global_batch_size
        if num_left > 0:
            parts_left = [
                global_batch.split(self.micro_batch_size)
                for global_batch in inds_ascending[:start].split(global_batch_size)
            ]
        else:
            parts_left = []
        num_right = num_full_batches - num_left
        end = self.dataset_size - num_right * global_batch_size
        if num_right > 0:
            parts_right = [
                global_batch.split(self.micro_batch_size)
                for global_batch in inds_ascending[end:].split(global_batch_size)
            ]
        else:
            parts_right = []
        if start < end:
            parts_mid = [
                global_batch.split(self.micro_batch_size)
                for global_batch in inds_ascending[start:end].split(global_batch_size)
            ]
        else:
            parts_mid = []
        self._partition = parts_left + parts_mid + parts_right

    def _create_permutation(self):
        num_outer = len(self._partition)
        if self._longest_first:
            out_inds = [num_outer - 1] + torch.randperm(num_outer - 1).tolist()
        else:
            out_inds = torch.randperm(num_outer).tolist()
        parts = [
            self._partition[out_ind][inn_ind]
            for out_ind in out_inds
            for inn_ind in torch.randperm(
                len(self._partition[out_ind])
            ).tolist()
        ]
        self._permutation = torch.cat(parts)

    def __next__(self) -> int:
        if self._pos >= self.dataset_size:
            self._create_permutation()
            self._pos = 0
            self._longest_first = False
        result = self._permutation[self._pos].item()
        self._pos += 1
        return result

    def __iter__(self) -> Iterator:
        return self


class SimilarSequenceLengthIterable(Iterable):
    """
    To be used as `sampler` for :class:`torch.utils.data.DataLoader`.

    Use this to create balanced batches for distributed data parallel
    training over several devices. Batches are of size `micro_batch_size`
    per device, and there are `num_devices` devices. Also, `sequence_lengths`
    is the number of tokens per sequence.

    The method ensures that:

    * Sequences in a batch (or micro-batch) are mostly closest in length
    * Next, sequences in `num_devices` consecutive batches (called
      macro-batch) are close in length
    * If one macro-batch is shorter than the others, it contains the
      longest sequences
    * Given that, macro-batches are ordered randomly, and the order of
      micro-batches in each macro-batch are random as well

    TODO: Another feature would be a preferred order of micro-batches
    inside macro-batches, to counter speed differences between the
    devices. Does this happen?

    """
    def __init__(
        self,
        sequence_lengths: List[int],
        micro_batch_size: int,
        num_devices: int,
        longest_first: bool = False,
    ):
        self._kwargs = {
            "sequence_lengths": sequence_lengths.copy(),
            "micro_batch_size": micro_batch_size,
            "num_devices": num_devices,
            "longest_first": longest_first,
        }
        self._len = len(sequence_lengths)

    def __iter__(self) -> Iterator:
        return SimilarSequenceLengthIterator(**self._kwargs)

    def __len__(self) -> int:
        return self._len
