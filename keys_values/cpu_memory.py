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
import atexit
from datetime import datetime
from filelock import FileLock, Timeout
import math
import os
from pathlib import Path
import signal
import threading
from typing import Optional, Tuple

import psutil
import torch

from keys_values.utils import bytes_for_torch_dtype


def available_cpu_memory_in_bytes() -> int:
    vm = psutil.virtual_memory()
    return vm.available


NUM_FILES_PER_DIRECTORY = 3072


class FileBasedExtraMemoryManager:
    """
    Allocates CPU tensors from RAM, falling back to memory-mapped files on
    a large disk. This disk can be an external file system (such as AWS EFS).

    We allocate a tensor from RAM if free RAM space lies above
    `max(threshold, n_bytes * self._HEADROOM)`, otherwise we use a new
    memory-mapped file. It is recommended to set `threshold` as a fraction of
    free RAM space determined just before allocations start.

    """

    _HEADROOM = 1.1  # require 10% more available memory than requested

    def __init__(self, tmp_dir: str, threshold: Optional[int]) -> None:
        # Create unique subdirectory
        self.tmp_path = self._unique_path(tmp_dir)
        print(
            f"FileBasedExtraMemoryManager: Memory-mapped files will be written to {self.tmp_path}"
        )
        self.threshold = threshold
        self.num_files = 0
        self._lock = threading.Lock()
        # Ensure that when program ends, or when it is terminated,
        # :meth:`cleanup` is called, which removes the files.
        # atexit.register(self.cleanup)
        # self._install_sigterm_handler()

    @staticmethod
    def _unique_path(tmp_dir: str) -> Path:
        time_format = "%Y%m%d_%H%M%S"
        time_stamp = datetime.now().strftime(time_format)
        tmp_path = Path(tmp_dir)
        res_path = None
        runn_no = -1
        while res_path is None:
            runn_no += 1
            cand_path = tmp_path / (time_stamp + f"_{runn_no}")
            lock_path = cand_path.with_suffix(".lock")
            lock = FileLock(lock_path, timeout=1)
            try:
                with lock.acquire(timeout=1):
                    if not cand_path.exists():
                        cand_path.mkdir(parents=True)
                        res_path = cand_path
            except Timeout:
                pass
        return res_path

    def _path_for(self, num: int) -> Path:
        return (
            self.tmp_path
            / str(num // NUM_FILES_PER_DIRECTORY)
            / f"buf_{num % NUM_FILES_PER_DIRECTORY}.bin"
        )

    def _use_virtual_memory(self, n_bytes: int) -> bool:
        available = available_cpu_memory_in_bytes()
        limit = n_bytes * self._HEADROOM
        if self.threshold is not None:
            limit = max(limit, self.threshold)
        return available > limit

    def allocate(
        self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        n_bytes = math.prod(shape) * bytes_for_torch_dtype(dtype)
        if self._use_virtual_memory(n_bytes):
            return torch.empty(shape, dtype=dtype)
        return self._alloc_from_file(shape, dtype, n_bytes)

    def cleanup(self) -> None:
        """
        Removes all files and the directory. Afterwards, the object cannot be used
        anymore.

        """
        with self._lock:
            num_files, self.num_files = self.num_files, 0
            tmp_path, self.tmp_path = self.tmp_path, None
        if tmp_path is None:
            return
        for num in range(num_files):
            try:
                self._path_for(num).unlink()
            except FileNotFoundError:
                pass
            if num % NUM_FILES_PER_DIRECTORY == 0 and num > 0:
                try:
                    self._path_for(num - 1).parent.rmdir()
                except Exception:
                    pass
        if num_files > 0:
            final_dir = self._path_for(num_files - 1).parent
            if final_dir.exists():
                try:
                    final_dir.rmdir()
                except Exception:
                    pass
        try:
            tmp_path.rmdir()
        except Exception:
            pass
        lock_path = tmp_path.with_suffix(".lock")
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    def _alloc_from_file(
        self, shape: Tuple[int, ...], dtype: torch.dtype, n_bytes: int
    ) -> torch.Tensor:
        with self._lock:
            if self.tmp_path is None:
                raise RuntimeError(
                    "FileBasedExtraMemoryManager has already been cleaned up"
                )
            num = self.num_files
            path = self._path_for(num)
            if num % NUM_FILES_PER_DIRECTORY == 0:
                path.parent.mkdir(parents=True, exist_ok=True)
            self.num_files += 1
        storage = torch.UntypedStorage.from_file(str(path), shared=True, nbytes=n_bytes)
        if num == 0:
            print(f"\nStarting to write virtual memory to {self.tmp_path}")
        return torch.empty(0, dtype=dtype).set_(storage).reshape(shape)

    def _install_sigterm_handler(self) -> None:
        prev = signal.getsignal(signal.SIGTERM)

        def handler(signum, frame):
            self.cleanup()
            if callable(prev):
                prev(signum, frame)
            else:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGTERM)

        signal.signal(signal.SIGTERM, handler)


_manager: Optional[FileBasedExtraMemoryManager] = None


def get_memory_manager(
    tmp_dir: Optional[str] = None,
    threshold: Optional[int] = None,
) -> FileBasedExtraMemoryManager:
    """
    Return the process-wide singleton, creating it on first call.
    If this is called again, we try to return the singleton object, but sometimes
    have to recreate it:

    * If `tmp_dir` is given and different from `_manager.tmp_dir`: Recreate, copy
      `_manager.threshold`
    * If `threshold` is given: Set `_manager.threshold = threshold` and return
      `_manager`. This allows to delay setting `threshold`

    """
    global _manager
    if _manager is not None:
        if tmp_dir is not None:
            raise ValueError(
                "Singleton exists. Don't pass `tmp_dir` to `get_memory_manager`"
            )
        elif threshold is not None:
            _manager.threshold = threshold
    if _manager is None:
        if tmp_dir is None:
            raise ValueError("`tmp_dir` must be provided with first call")
        _manager = FileBasedExtraMemoryManager(tmp_dir, threshold)
    return _manager


def has_memory_manager() -> bool:
    return _manager is not None
