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
from pathlib import Path
from typing import Optional

import torch

from keys_values.kvcache.utils import VerbosityLevels

RECORD_MEMORY_MAX_MEM_EVENTS_PER_SNAPSHOT = 200000


class RecordGPUMemory:
    def __init__(
        self,
        path: Optional[str] = None,
        max_entries: int = RECORD_MEMORY_MAX_MEM_EVENTS_PER_SNAPSHOT,
        verbose: VerbosityLevels = VerbosityLevels.NONE,
    ):
        """
        This is more of a config, which also has methods to start and stop
        recording. However, only one object of this class can be used at
        any one time.

        """
        if max_entries <= 0:
            raise ValueError("max_entries must be greater than zero")
        if path is not None:
            Path(path).parent.mkdir(exist_ok=True, parents=True)
        self._path = path
        self.max_entries = max_entries
        self.verbose = verbose
        self._is_recording = False

    @property
    def path(self) -> Path:
        return Path(self._path)

    def set_path(self, path: str):
        self._path = path

    def _print_message(self) -> bool:
        return self.verbose is VerbosityLevels.MORE or self.verbose is VerbosityLevels.ALL

    def start_recording(self):
        if self._path is not None and self._print_message():
            print(f"Start recording GPU memory snapshot to {self._path}")
        torch.cuda.memory._record_memory_history(max_entries=self.max_entries)
        self._is_recording = True

    def stop_recording(self):
        if self._path is not None and self._print_message():
            print(f"Stop recording GPU memory snapshot to {self._path}")
        torch.cuda.memory._record_memory_history(enabled=None)
        self._is_recording = False

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def store_current_snapshot(self):
        if self._path is not None:
            try:
                torch.cuda.memory._dump_snapshot(self._path)
            except Exception as e:
                print(f"\nFailed to capture memory snapshot {e}")
