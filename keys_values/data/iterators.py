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
from typing import Iterator, List, Iterable

import torch


class LongestFirstIterator(Iterator):
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        inds_longest: List[int],
    ):
        assert len(inds_longest) == batch_size
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
    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        inds_longest: List[int],
    ):
        self._kwargs = {
            "dataset_size": dataset_size,
            "batch_size": batch_size,
            "inds_longest": inds_longest.copy(),
        }
        self._len = dataset_size

    def __iter__(self) -> Iterator:
        return LongestFirstIterator(**self._kwargs)

    def __len__(self) -> int:
        return self._len
