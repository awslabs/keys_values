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
from itertools import product
from pathlib import Path
from typing import Literal

from keys_values.evaluation.tasks import EvaluationTasks


def main(
    out_dir: Path,
    model_type: str,
    mode: Literal["non-lock", "lock", "all"],
    multiple_tasks: bool,
):
    total_removed = 0
    eval_tasks = EvaluationTasks(out_dir, model_type, multiple_tasks=multiple_tasks)
    print(f"Removing files for {out_dir}")
    for task_name, incomplete_file_paths in eval_tasks.eval_result_files(mode):
        print(f"{task_name}: Removing {len(incomplete_file_paths)} files (type {mode})")
        for path in incomplete_file_paths:
            path.unlink()
        total_removed += len(incomplete_file_paths)
    print(f"Removed {total_removed} files in total (type {mode})")


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"

    dataset_size = "64k"
    # dataset_size = "128k"
    # is_baseline = False
    is_baseline = True
    if is_baseline:
        base_path = base_path / "baseline"
    datasets = [
        f"helmet_nq_{dataset_size}",
        f"helmet_trivia_qa_{dataset_size}",
        f"helmet_hotpot_qa_{dataset_size}",
        f"helmet_pop_qa_{dataset_size}",
    ]
    cases = [
        "lr_4gpu_cs2048_lr5",
        "lr_4gpu_cs1024_lr5",
        "slr_4gpu_cs2048_lr5",
        "slr_4gpu_cs1024_lr5",
        "h2o_4gpu_cs2048_lr5",
        "h2o_4gpu_cs1024_lr5",
        "qh2o_4gpu_cs2048_lr5",
        "h2onorm_4gpu_cs2048_lr5",
        "h2onorm_4gpu_cs1024_lr5",
        "qh2onorm_4gpu_cs2048_lr5",
    ]
    # Use this to clean up lock files before restarting evaluation
    # mode = "lock"
    # Use this to remove all evaluation files
    mode: Literal["non-lock", "lock", "all"] = "all"
    model_type = "lora"
    for dataset, case in product(datasets, cases):
        out_dir = base_path / dataset / case
        if out_dir.exists():
            main(out_dir, model_type, mode, not is_baseline)
        else:
            print(f"\nResults for {dataset}/{case} do not exist")
