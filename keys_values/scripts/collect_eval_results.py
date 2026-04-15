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

def main(
    out_dir: Path,
    model_type: str,
    tasks: Optional[List[str]] = None,
):
    # Collect results from all files across all tasks
    print(f"\nLoading evaluation result files from {out_dir}")
    eval_tasks = EvaluationTasks(
        out_dir, model_type, tasks, collect_results=True,
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
    combined_path = out_dir / "eval_metrics_all.csv"
    with open(combined_path, "w") as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerow(column_names)
        for row in sorted(all_data, key=lambda x: (x[1], int(x[0]))):
            writer.writerow(row)


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"
    datasets = [
        "helmet_nq_32k",
        "helmet_trivia_qa_32k",
        "helmet_hotpot_qa_32k",
        "helmet_pop_qa_32k",
    ]
    cases = [
        "lr_4gpu_lr5",
    ]
    model_type = "lora"
    for dataset, case in product(datasets, cases):
        out_dir = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b" / dataset / case
        if out_dir.exists():
            main(out_dir, model_type)
