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
from typing import Tuple, List, Iterator, Dict, Any

import torch
from torch.utils.data import Dataset

from keys_values.data.dataloader import Collator
from keys_values.data.iterators import BatchSampler

ResultType = Tuple[List[int], int]

ORIG_IDX_NAME = "orig_idx"

TASK_NAME = "task"


class SimilarSequenceLengthWithTasksIterator(Iterator[ResultType]):
    def __init__(
        self,
        sequence_lengths: List[int],
        micro_batch_size: int,
        num_tasks: int,
    ):
        self.num_batches = math.ceil(len(sequence_lengths) / micro_batch_size)
        self.dataset_size = self.num_batches * micro_batch_size
        self.micro_batch_size = micro_batch_size
        self.num_tasks = num_tasks
        self._permutation = None
        self._initialize(sequence_lengths)
        self._pos = 0

    def _initialize(self, sequence_lengths: List[int]):
        # Sort from shortest to longest
        len_sl = len(sequence_lengths)
        inds_ascending = torch.argsort(torch.tensor(sequence_lengths))
        if len_sl == self.dataset_size:
            self._permutation = inds_ascending
        else:
            extra_inds = torch.arange(
                len_sl,
                self.dataset_size,
                dtype=inds_ascending.dtype,
                device=inds_ascending.device,
            )
            self._permutation = torch.cat((inds_ascending, extra_inds))
        self._permutation = self._permutation.tolist()

    def __next__(self) -> ResultType:
        if self._pos >= self.dataset_size * self.num_tasks:
            raise StopIteration
        task_idx = self._pos // self.dataset_size
        start = self._pos % self.dataset_size
        mbs = self.micro_batch_size
        self._pos += mbs
        return self._permutation[start : (start + mbs)], task_idx

    def __iter__(self) -> Iterator[ResultType]:
        return self


class SimilarSequenceLengthWithTasksSampler(BatchSampler):
    """
    Batch sampler for evaluation data iterator, where different tasks
    (i.e., model checkpoints) are evaluated for the same underlying dataset.

    The size of the dataset is a multiple of `micro_batch_size`, possibly
    padded at the end. `sequence_lengths` runs over the non-pad sequences,
    so can be shorter up to `micro_batch_size - 1`. Properties:

    * Items are grouped in sorted order according to `sequence_lengths`
      (shortest first). After sorting, we create batches of size
      `micro_batch_size`. Sorting ensures that items within a batch have
      similar length. The final batch may be shorter, but contains the longest
      items.
    * If there are `num_batches` batches and `num_tasks` tasks, the iterator
      produces `num_batches * num_tasks` batches from the cross product of
      dataset batches and tasks, where the outer loop is over tasks. Given
      this ordering, a batch is returned for rank `rank` if its position
      modulo `num_devices` is equal to `rank`.
    * The sampler only returns `List[int]` indexes into the dataset, of
      size `micro_batch_size`. Entries `>= len(sequence_lengths)`
      correspond to pad items. Fusing these with the tasks and collation
      is done in the dataset iterator.
    * The iterator returns the same sequence of batches on each rank. We
      then use locks to make sure ranks skip batches already picked up.
      This makes sure that if some ranks are faster than other, they also
      process more batches.

    """

    def __init__(
        self,
        sequence_lengths: List[int],
        micro_batch_size: int,
        num_tasks: int,
    ):
        assert micro_batch_size >= 1
        assert num_tasks >= 1
        assert len(sequence_lengths) > 0
        self._kwargs = {
            "sequence_lengths": sequence_lengths.copy(),
            "micro_batch_size": micro_batch_size,
            "num_tasks": num_tasks,
        }
        num_batches = math.ceil(len(sequence_lengths) / micro_batch_size)
        self._len = num_batches * num_tasks

    def __iter__(self) -> Iterator[ResultType]:
        return SimilarSequenceLengthWithTasksIterator(**self._kwargs)

    def __len__(self) -> int:
        return self._len

    @property
    def batch_size(self) -> int:
        return self._kwargs["micro_batch_size"]

    @property
    def num_tasks(self) -> int:
        return self._kwargs["num_tasks"]


class EvaluationDataLoaderIterator(Iterator[Dict[str, Any]]):
    def __init__(
        self,
        dataset: Dataset,
        batch_sampler: SimilarSequenceLengthWithTasksSampler,
        collate_fn: Collator,
        eval_tasks: List[str],
        delay_tokenization: bool,
    ):
        if len(eval_tasks) != batch_sampler.num_tasks:
            raise ValueError(
                f"len(eval_tasks) = {len(eval_tasks)} != {batch_sampler.num_tasks} = batch_sampler.num_tasks"
            )
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn
        self.eval_tasks = eval_tasks
        self.delay_tokenization = delay_tokenization
        self._batch_iter = iter(batch_sampler)
        dataset_size = self._batch_iter.dataset_size
        if len(dataset) != dataset_size:
            raise ValueError(
                f"len(dataset) = {len(dataset)} != {dataset_size} = batch_sampler.dataset_size"
            )

    def __next__(self) -> Dict[str, Any]:
        inds, task_idx = next(self._batch_iter)
        result = {
            TASK_NAME: self.eval_tasks[task_idx],
            ORIG_IDX_NAME: inds,
        }
        if not self.delay_tokenization:
            result = self.fetch_full(result)
        return result

    def fetch_full(self, partial_batch: Dict[str, Any]) -> Dict[str, Any]:
        inds = partial_batch[ORIG_IDX_NAME]
        result = self.collate_fn([self.dataset[idx] for idx in inds])
        return {**partial_batch, **result}

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self


class EvaluationDataLoader:
    """
    Data loader for pure evaluation runs over several tasks (i.e.,
    checkpoints).

    If `delay_tokenization == True`, the batch returned has only
    :const:`TASK_NAME` and :const:`ORIG_IDX_NAME` fields set, this
    does not require tokenization. The remaining fields can be obtained
    by calling :meth:`fetch_full`. Use this to be able to skip already
    processed or locked batches rapidly.

    """

    def __init__(
        self,
        dataset: Dataset,
        batch_sampler: SimilarSequenceLengthWithTasksSampler,
        collate_fn: Collator,
        eval_tasks: List[str],
        delay_tokenization: bool = False,
    ):
        self._iter_kwargs = {
            "dataset": dataset,
            "batch_sampler": batch_sampler,
            "collate_fn": collate_fn,
            "eval_tasks": eval_tasks,
            "delay_tokenization": delay_tokenization,
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return EvaluationDataLoaderIterator(**self._iter_kwargs)

    def __len__(self) -> int:
        return len(self._iter_kwargs["batch_sampler"])

    @property
    def batch_size(self) -> int:
        return self._iter_kwargs["batch_sampler"].batch_size

    @property
    def delay_tokenization(self) -> bool:
        return self._iter_kwargs["delay_tokenization"]
