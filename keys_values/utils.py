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
import csv
from filelock import FileLock, Timeout
from pathlib import Path
import time
from typing import List, Dict, Any


def _append_results_to_csv(
    results: List[Dict[str, Any]],
    result_path: Path,
) -> bool:
    lock_path = result_path.with_suffix(".lock")
    lock = FileLock(lock_path, timeout=1)
    try:
        with lock.acquire(timeout=1):
            fieldnames = sorted(results[0].keys())
            mode = "a" if result_path.exists() else "w"
            with result_path.open(mode) as fp:
                writer = csv.writer(fp, delimiter=",")
                if mode == "w":
                    writer.writerow(fieldnames)
                for record in results:
                    row = [record[name] for name in fieldnames]
                    writer.writerow(row)
    except Timeout:
        return False
    finally:
        lock.release()
        lock_path.unlink()
        return True


def append_results_to_csv(
    results: List[Dict[str, Any]],
    result_path: Path,
    num_retrials: int = 100,
    sleep_time: float = 0.1,
):
    for _ in range(num_retrials):
        if _append_results_to_csv(results, result_path):
            break
        time.sleep(sleep_time)
