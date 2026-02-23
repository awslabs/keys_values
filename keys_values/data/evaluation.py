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
from filelock import FileLock, Timeout
from functools import partial
import math
from pathlib import Path
import re
from typing import List, Dict, Any, Callable, Optional, Iterator, Tuple

import torch
from torch.utils.data import Dataset

from keys_values.data.base import (
    LIT_MODEL_FNAME,
    LORA_WEIGHTS_FNAME,
    LORA_WEIGHTS_FNAME_OLD,
)
from keys_values.data.dataloader import Collator
from keys_values.data.iterators import BatchSampler


EVAL_METRICS_FNAME = "eval/eval_metrics_{}.csv"

REGEX_TASKNAME = re.compile(r"step-[0-9]{6}|final")

_REQUIRED_FILES = [
    "hyperparameters.yaml",
    "model_config.yaml",
    "tokenizer.json",
    "tokenizer_config.json",
]

REQUIRED_FILES = {
    "full": _REQUIRED_FILES + [LIT_MODEL_FNAME],
    "lora": _REQUIRED_FILES + [LORA_WEIGHTS_FNAME],
}

ORIG_IDX_NAME = "orig_idx"

TASK_NAME = "task"


class EvaluationTasks:
    """
    Each evaluation task corresponds to a model checkpoint. It is represented
    by its directory name, starting from `out_dir`.

    """

    def __init__(self, out_dir: Path, model_type: str):
        self.model_type = model_type
        self._tasks = None
        self._init_task_names(out_dir)

    def _init_task_names(self, out_dir: Path):
        self._tasks = []
        include_final = False
        for child in out_dir.iterdir():
            if child.is_dir() and REGEX_TASKNAME.match(child.name):
                if self.check_complete(child, self.model_type):
                    if child.name != "final":
                        self._tasks.append(child.name)
                    else:
                        include_final = True
        # Sort to obtain unique ordering
        self._tasks = sorted(self._tasks)
        # If "final" is present, it should come first, so we get the final
        # eval results before others
        if include_final:
            self._tasks.insert(0, "final")

    @property
    def tasks(self) -> List[str]:
        return self._tasks

    @staticmethod
    def check_complete(task_path: Path, model_type: str) -> bool:
        missing_files = []
        for name in REQUIRED_FILES[model_type]:
            if not (task_path / name).exists():
                if name != LORA_WEIGHTS_FNAME:
                    missing_files.append(name)
                elif not (task_path / LORA_WEIGHTS_FNAME_OLD).exists():
                    missing_files.append(
                        f"{LORA_WEIGHTS_FNAME} or {LORA_WEIGHTS_FNAME_OLD}"
                    )
        if missing_files:
            print(f"{task_path.name}: Incomplete, did not find {missing_files}")
            return False
        else:
            return True


class EvaluationWithTasksHelper:
    """
    Helper to obtain path evaluation metrics file. Can be used to test
    whether the metrics file already exists, in which case the batch
    should be skipped.

    We also support file locking here, which enables the custom batch
    dataloader we use.

    """

    def __init__(self, out_dir: Path, tag: Optional[str] = None):
        self._out_dir = out_dir
        if tag is None:
            tag = ""
        self._tag = tag

    def evaluation_metrics_path(self, batch: Dict[str, Any]) -> Path:
        """
        Args:
            batch: Batch returned by data iterator. We only use entries
                :const:`ORIG_IDX_NAME` and :const:`TASK_NAME`.

        Returns:
            Evaluation metrics path

        """
        orig_idxs = batch.get(ORIG_IDX_NAME)
        task = batch.get(TASK_NAME)
        if not isinstance(orig_idxs, list) or not isinstance(task, str):
            raise ValueError(
                f"Batch needs to contain entries {ORIG_IDX_NAME}, {TASK_NAME}, "
                f"but got batch[{ORIG_IDX_NAME}] = {orig_idxs}, "
                f"batch[{TASK_NAME}] = {task}."
            )
        suffix = self._tag + str(orig_idxs[0])
        fname = EVAL_METRICS_FNAME.format(suffix)
        return self._out_dir / task / fname

    def get_lock(self, batch: Dict[str, Any]) -> Optional[Path]:
        """
        Tries to get a lock for the evaluation results on batch `batch`.
        If the lock is obtained, a bogus file is written, the lock is
        released, and the file path is returned. If we hit a lock or the
        file exists, returns `None`.

        Args:
            batch: Batch returned by data iterator.

        Returns:
            File path if evaluation metrics file does not exist and also
            has no lock on it. Otherwise, `None` is returned, and the
            batch should be skipped.

        """
        file_path = self.evaluation_metrics_path(batch)
        if file_path.exists():
            return None
        lock_path = file_path.with_suffix(".lock")
        lock = FileLock(lock_path, timeout=1)
        try:
            with lock.acquire(timeout=1):
                with file_path.open("w") as fp:
                    fp.write("CURRENTLY EVALUATING\n")
        except Timeout:
            return None
        finally:
            lock.release()
            lock_path.unlink()
            return file_path


ResultType = Tuple[List[int], int]


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
