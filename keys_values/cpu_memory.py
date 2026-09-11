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
import math
import os
import signal
import threading
from typing import Optional, Tuple, List

import psutil
import torch

from keys_values.utils import bytes_for_torch_dtype


class FileBasedExtraMemoryManager:
    """
    Allocates CPU tensors from RAM/swap, falling back to memory-mapped files on
    a large disk. This disk can be an external file system (such as AWS EFS).
    """

    _HEADROOM = 1.1  # require 10% more available memory than requested

    def __init__(self, tmp_dir: str):
        # TODO: Need to create something unique from `tmp_dir`, which does not
        # already exist. This must work concurrently
        self.tmp_dir = tmp_dir
        self._files: List[str] = []
        self._lock = threading.Lock()
        os.makedirs(tmp_dir, exist_ok=True)
        atexit.register(self.cleanup)
        self._install_sigterm_handler()

    def allocate(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        n_bytes = math.prod(shape) * bytes_for_torch_dtype(dtype)
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        if vm.available + sw.free > n_bytes * self._HEADROOM:
            return torch.empty(shape, dtype=dtype)
        return self._alloc_from_file(shape, dtype, n_bytes)

    def cleanup(self) -> None:
        with self._lock:
            files, self._files = self._files, []
        for path in files:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def _alloc_from_file(self, shape: Tuple[int, ...], dtype: torch.dtype, n_bytes: int) -> torch.Tensor:
        with self._lock:
            path = os.path.join(self.tmp_dir, f"buf_{len(self._files)}.bin")
            self._files.append(path)
        storage = torch.UntypedStorage.from_file(path, shared=True, nbytes=n_bytes)
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


def get_memory_manager(tmp_dir: Optional[str] = None) -> FileBasedExtraMemoryManager:
    """Return the process-wide singleton, creating it on first call."""
    global _manager
    if _manager is not None and tmp_dir is not None and tmp_dir != _manager.tmp_dir:
        print(f"tmp_dir = {tmp_dir} != {_manager.tmp_dir} = _manager.tmp_dir. Creating new manager")
        _manager.cleanup()
        _manager = None
    if _manager is None:
        if tmp_dir is None:
            raise ValueError("tmp_dir must be provided with first call")
        _manager = FileBasedExtraMemoryManager(tmp_dir)
    return _manager


def has_memory_manager() -> bool:
    return _manager is not None
