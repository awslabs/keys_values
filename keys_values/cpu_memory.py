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
        self.tmp_dir = self._unique_dir(tmp_dir)
        print(
            f"FileBasedExtraMemoryManager: Memory-mapped files will be written to {self.tmp_dir}"
        )
        self.threshold = threshold
        self.num_files = 0
        self._lock = threading.Lock()
        # Ensure that when program ends, or when it is terminated,
        # :meth:`cleanup` is called, which removes the files.
        atexit.register(self.cleanup)
        self._install_sigterm_handler()

    def _unique_dir(self, tmp_dir: str) -> str:
        time_format = "%Y%m%d_%H%M%S"
        time_stamp = datetime.now().strftime(time_format)
        tmp_path = Path(tmp_dir)
        res_dir = None
        runn_no = -1
        while res_dir is None:
            runn_no += 1
            cand_path = tmp_path / (time_stamp + f"_{runn_no}")
            lock_path = cand_path.with_suffix(".lock")
            lock = FileLock(lock_path, timeout=1)
            try:
                with lock.acquire(timeout=1):
                    if not cand_path.exists():
                        cand_path.mkdir(parents=True)
                        res_dir = str(cand_path)
            except Timeout:
                pass
        return res_dir

    def _filename(self, num: int) -> str:
        return os.path.join(self.tmp_dir, f"buf_{num}.bin")

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
        Removes all files. The directory itself is not removed.

        """
        with self._lock:
            num_files, self.num_files = self.num_files, 0
        for num in range(num_files):
            try:
                os.remove(self._filename(num))
            except FileNotFoundError:
                pass

    def _alloc_from_file(
        self, shape: Tuple[int, ...], dtype: torch.dtype, n_bytes: int
    ) -> torch.Tensor:
        with self._lock:
            path = self._filename(self.num_files)
            self.num_files += 1
        storage = torch.UntypedStorage.from_file(path, shared=True, nbytes=n_bytes)
        if self.num_files == 1:
            print(f"\nStarting to write virtual memory to {self.tmp_dir}")
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
        if tmp_dir is not None and tmp_dir != _manager.tmp_dir:
            print(
                f"tmp_dir = {tmp_dir} != {_manager.tmp_dir} = _manager.tmp_dir. Creating new manager"
            )
            _manager.cleanup()
            if threshold is None:
                threshold = _manager.threshold
            _manager = None
        elif threshold is not None:
            _manager.threshold = threshold
    if _manager is None:
        if tmp_dir is None:
            raise ValueError("tmp_dir must be provided with first call")
        _manager = FileBasedExtraMemoryManager(tmp_dir, threshold)
    return _manager


def has_memory_manager() -> bool:
    return _manager is not None
