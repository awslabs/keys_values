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
from pathlib import Path
import re
from typing import List, Dict, Any, Optional, Iterable, Tuple

from keys_values.data.base import (
    LIT_MODEL_FNAME,
    LORA_WEIGHTS_FNAME,
    LORA_WEIGHTS_FNAME_OLD,
)
from keys_values.data.evaluation import ORIG_IDX_NAME, TASK_NAME

EVAL_METRICS_FNAME = "eval/eval_metrics_{}.csv"

EVAL_METRICS_GLOB = EVAL_METRICS_FNAME.replace("{}", "*")

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

FILE_LOCK_TEXT = "CURRENTLY EVALUATING"


class EvaluationTasks:
    """
    Each evaluation task corresponds to a model checkpoint. It is represented
    by its directory name, starting from `out_dir`.

    If `collect_results == True`, we collect tasks for which evaluation results
    are available.
    """

    def __init__(
        self,
        out_dir: Path,
        model_type: str,
        tasks: Optional[List[str]] = None,
        collect_results: bool = False,
    ):
        self._out_dir = out_dir
        self.model_type = model_type
        self._tasks = tasks.copy() if tasks is not None else None
        self._init_task_names(collect_results)

    def _init_task_names(self, collect_results: bool):
        if self._tasks is None:
            self._tasks = []
            include_final = False
            for child in self._out_dir.iterdir():
                if child.is_dir() and REGEX_TASKNAME.match(child.name):
                    if not collect_results:
                        if self.check_complete(child, self.model_type):
                            if child.name != "final":
                                self._tasks.append(child.name)
                            else:
                                include_final = True
                    elif self._num_result_files(child) > 0:
                        self._tasks.append(child.name)
            # Sort to obtain unique ordering
            self._tasks = sorted(self._tasks)
            # If "final" is present, it should come first, so we get the final
            # eval results before others
            if include_final:
                self._tasks.insert(0, "final")
        else:
            for name in self._tasks:
                path = self._out_dir / name
                if not path.exists() or not path.is_dir():
                    raise ValueError(
                        f"{path} does not exist. tasks = {self._tasks} invalid"
                    )
                if not collect_results:
                    if not self.check_complete(path, self.model_type):
                        raise ValueError(
                            f"{path} is incomplete. tasks = {self._tasks} invalid"
                        )
                elif self._num_result_files(path) == 0:
                    raise ValueError(
                        f"{path} contains no evaluation result files"
                    )

    @staticmethod
    def _num_result_files(path: Path) -> int:
        return len(list(path.glob(EVAL_METRICS_GLOB)))

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

    def eval_result_files(self) -> Iterable[Tuple[str, List[Path]]]:
        """
        Yields:
            `(task_name, result_file_paths)`, where `result_file_paths`
            is list of paths of evaluation result files for this task name.
            These files are filtered to not contain incomplete lock files.

        """
        for task_name in self._tasks:
            result_file_paths = self._filter_incomplete_files(
                (self._out_dir / task_name).glob(EVAL_METRICS_GLOB)
            )
            if result_file_paths:
                yield task_name, result_file_paths

    @staticmethod
    def _filter_incomplete_files(paths: Iterable[Path]) -> List[Path]:
        result = []
        for path in paths:
            with path.open("r") as fp:
                if not fp.readline().startswith(FILE_LOCK_TEXT):
                    result.append(path)
        return result


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
                    fp.write(FILE_LOCK_TEXT + "\n")
        except Timeout:
            return None
        finally:
            lock.release()
            if lock_path.exists():
                lock_path.unlink()
            return file_path


