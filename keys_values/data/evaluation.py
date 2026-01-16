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
from typing import List, Dict, Any, Callable, Optional

from torch.utils.data import Dataset

from keys_values.finetune.utils import (
    LIT_MODEL_FNAME,
    LORA_WEIGHTS_FNAME,
    LORA_WEIGHTS_FNAME_OLD,
)


EVAL_METRICS_FNAME = "eval_metrics_{}.csv"

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

CollateFnType = Callable[[List[Dict[str, Any]]], Dict[str, Any]]


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


class EvaluationWithTasksDataset(Dataset):
    """
    Wrapper class which enables evaluations over several tasks.

    Spans cross product of original dataset items with tasks. A task is given
    by a string, which in our case is a path.

    Why the dependence on `num_devices`? If more than 1 device is used, the
    case ordering needs to be modified by :class:`ReorderWrapperDataset`.
    This happens in periods of size `batch_size * num_devices`. We need to
    avoid that a batch sent to a device contains cases from different tasks.

    """

    def __init__(
        self,
        dataset: Dataset,
        tasks: List[str],
        batch_size: int,
        num_devices: int = 1,
    ):
        self._dataset = dataset
        self._tasks = tasks
        self._num_orig_cases = len(dataset)
        unit = batch_size * num_devices
        self._num_orig_cases_padded = int(math.ceil(self._num_orig_cases / unit)) * unit
        self._num_cases = self._num_orig_cases_padded * len(tasks)

    @property
    def orig_dataset(self) -> Dataset:
        return self._dataset

    def __len__(self) -> int:
        return self._num_cases

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not (0 <= idx < self._num_cases):
            raise IndexError(
                f"index {idx} out of range, must be in [0, {self._num_cases})"
            )
        task = self._tasks[idx // self._num_orig_cases_padded]
        orig_idx = idx % self._num_orig_cases_padded
        if orig_idx < self._num_orig_cases:
            return {
                **self._dataset[orig_idx],
                ORIG_IDX_NAME: orig_idx,
                TASK_NAME: task,
            }
        else:
            return dict()


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


def get_wrapped_collate_fn(orig_collate_fn: CollateFnType) -> CollateFnType:
    return partial(_wrapped_collate_fn, orig_collate_fn=orig_collate_fn)


def _wrapped_collate_fn(
    samples: List[Dict[str, Any]],
    orig_collate_fn: CollateFnType,
) -> Dict[str, Any]:
    samples = [elem for elem in samples if elem]
    if not samples:
        return dict()  # Empty batch can happen
    tasks = set(elem[TASK_NAME] for elem in samples)
    if len(tasks) > 1:
        raise IndexError(
            f"Batch must only have single {TASK_NAME} value, but has {tasks}"
        )
    task = next(iter(tasks))
    orig_collated_samples = orig_collate_fn(samples)
    orig_idxs = [elem[ORIG_IDX_NAME] for elem in samples]
    return {
        **orig_collated_samples,
        ORIG_IDX_NAME: orig_idxs,
        TASK_NAME: task,
    }


class ReorderWrapperDataset(Dataset):
    """
    Undoes the annoying property of interleaving cases into batches when
    a number of devices are used. For example, for 3 devices and batch size
    4, we obtain:

    * Device 0: [0, 3, 6, 9], [12, 15, 18, 21], ...
    * Device 1: [1, 4, 7, 10], [13, 16, 19, 22], ...
    * Device 2: [2, 5, 8, 11], [14, 17, 20, 23], ...

    This wrapper turns this into:

    * Device 0: [0, 1, 2, 3], [12, 13, 14, 15], ...
    * Device 1: [4, 5, 6, 7], [16, 17, 18, 19], ...
    * Device 2: [8, 9, 10, 11], [20, 21, 22, 23], ...

    """

    def __init__(
        self,
        dataset: Dataset,
        num_devices: int,
        batch_size: int,
    ):
        self._dataset = dataset
        self._num_devices = num_devices
        self._batch_size = batch_size
        assert num_devices >= 1 and batch_size >= 1

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        num_cases = self.__len__()
        if not (0 <= idx < num_cases):
            raise IndexError(f"index {idx} out of range, must be in [0, {num_cases})")
        if self._num_devices == 1 or self._batch_size == 1:
            orig_idx = idx
        else:
            ndev = self._num_devices
            bs = self._batch_size
            period = ndev * bs
            offset = (idx // period) * period
            orig_idx = idx % period
            orig_idx = (orig_idx % ndev) * bs + (orig_idx // ndev) + offset
        return self._dataset[orig_idx]
