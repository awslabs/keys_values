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
from itertools import product
from pathlib import Path
from typing import List, Optional

from keys_values.evaluation.tasks import EvaluationTasks

EVAL_METRICS_ALL_FILENAME = "eval_metrics_all.csv"

SWEEP_TAR_FILENAME = "eval_metrics_transfer_{dataset_size}.tgz"


def main(
    out_dir: Path,
    model_type: str,
    tasks: Optional[List[str]] = None,
    multiple_tasks: bool = True,
):
    # Collect results from all files across all tasks
    print(f"\nLoading evaluation result files from {out_dir}")
    eval_tasks = EvaluationTasks(
        out_dir,
        model_type,
        tasks,
        collect_results=True,
        multiple_tasks=multiple_tasks,
    )
    all_data = []
    column_names = None
    for task_name, result_file_paths in eval_tasks.eval_result_files():
        print(f"{task_name}: {len(result_file_paths)}")
        sum_vals = 0
        num_vals = 0
        for path in result_file_paths:
            with open(path, "r") as fp:
                reader = csv.reader(fp, delimiter=",")
                first_row = True
                for row in reader:
                    if not first_row:
                        all_data.append(row)
                        sum_vals += float(row[-1])
                        num_vals += 1
                    elif column_names is None:
                        column_names = row
                    first_row = False
        print(f"    {column_names[-1]} = {(sum_vals / num_vals):.3f}")

    print(f"Total number of records: {len(all_data)}")
    if all_data:
        combined_path = out_dir / EVAL_METRICS_ALL_FILENAME
        if combined_path.exists():
            combined_path.unlink()
        with open(combined_path, "w") as fp:
            writer = csv.writer(fp, delimiter=",")
            writer.writerow(column_names)
            for row in sorted(all_data, key=lambda x: (x[1], int(x[0]))):
                writer.writerow(row)


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"

    mode = "collect"
    # mode = "sweep"
    dataset_size = "64k"
    # dataset_size = "128k"
    is_baseline = False
    # is_baseline = True
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
        "slr_4gpu_cs2048_lr5",
        "h2o_4gpu_cs2048_lr5",
        "qh2o_4gpu_cs2048_lr5",
        "h2onorm_4gpu_cs2048_lr5",
        "qh2onorm_4gpu_cs2048_lr5",
        "lr_4gpu_cs1024_lr5",
        "slr_4gpu_cs1024_lr5",
        "h2o_4gpu_cs1024_lr5",
        "h2onorm_4gpu_cs1024_lr5",
    ]
    model_type = "lora"
    if mode == "collect":
        for dataset, case in product(datasets, cases):
            out_dir = base_path / dataset / case
            if out_dir.exists():
                main(out_dir, model_type, multiple_tasks=not is_baseline)
            else:
                print(f"\nResults for {dataset}/{case} do not exist")
    elif mode == "sweep":
        names = []
        for dataset, case in product(datasets, cases):
            name = "/".join((dataset, case, EVAL_METRICS_ALL_FILENAME))
            if (base_path / name).exists():
                names.append(name)
        print(
            f"\nCollected {len(names)} result files. Run at {base_path}:\n"
            + "tar cfz "
            + SWEEP_TAR_FILENAME.format(dataset_size=dataset_size)
            + " "
            + " ".join(names)
        )
    else:
        raise NotImplementedError(f"Unknown mode: {mode}")
