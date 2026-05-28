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
from typing import Iterator, Dict, Any

import torch
from torch.utils.data import Dataset

from keys_values.data.constants import ORIG_IDX_NAME, Collator
from keys_values.data.iterators import BatchSampler, SimilarSequenceLengthIterator


class MyDataLoaderIterator(Iterator[Dict[str, Any]]):
    def __init__(
        self,
        dataset: Dataset,
        batch_sampler: BatchSampler,
        collate_fn: Collator,
    ):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn
        self._batch_iter = iter(batch_sampler)

    def __next__(self) -> Dict[str, Any]:
        inds = next(self._batch_iter)
        return {
            **self.collate_fn([self.dataset[idx] for idx in inds]),
            ORIG_IDX_NAME: inds,
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self

    def state_dict(self) -> Dict[str, torch.Tensor]:
        if not isinstance(self._batch_iter, SimilarSequenceLengthIterator):
            raise NotImplementedError("Only works for SimilarSequenceLengthIterator")
        return self._batch_iter.state_dict()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        if not isinstance(self._batch_iter, SimilarSequenceLengthIterator):
            raise NotImplementedError("Only works for SimilarSequenceLengthIterator")
        self._batch_iter.load_state_dict(state_dict)


class MyDataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_sampler: BatchSampler,
        collate_fn: Collator,
    ):
        """
        Replacement for PyTorch `DataLoader`, which seems to do odd things
        under the hood.

        Args:
            dataset: Dataset, supports random access
            batch_sampler: Samples batch indexes. Takes care of randomization
                (if any) and dataset partitioning when there are several
                processes
            collate_fn: Combines data items into a batch, also removes padding
                entries

        """
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self._iter_kwargs = {
            "dataset": dataset,
            "batch_sampler": batch_sampler,
            "collate_fn": collate_fn,
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return MyDataLoaderIterator(**self._iter_kwargs)

    def __len__(self) -> int:
        return len(self._iter_kwargs["batch_sampler"])

    @property
    def batch_size(self) -> int:
        return self._iter_kwargs["batch_sampler"].batch_size
