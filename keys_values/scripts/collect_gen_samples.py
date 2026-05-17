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
from typing import List, Optional, Dict, Any
import yaml

from keys_values.evaluation.tasks import EvaluationTasks
from keys_values.finetune.longcontext_eval_ext import GENERATED_SAMPLES_FILENAME

GENERATED_SAMPLES_ALL_FILENAME = "generated_samples_all.yaml"

SWEEP_TAR_FILENAME = "generated_samples_transfer_{dataset_size}.tgz"


def main(
    out_dir: Path,
    model_type: str,
    tasks: Optional[List[str]] = None,
):
    # Collect results from all files across all tasks
    print(f"\nLoading generated samples files from {out_dir}")
    eval_tasks = EvaluationTasks(
        out_dir,
        model_type,
        tasks,
        collect_results=True,
        eval_metrics_filename="eval/" + GENERATED_SAMPLES_FILENAME,
    )
    all_data: Dict[str, List[Dict[str, Any]]] = dict()
    num_total = 0
    for task_name, result_file_paths in eval_tasks.eval_result_files():
        print(f"{task_name}: {len(result_file_paths)}")
        records: List[Dict[str, Any]] = []
        for path in result_file_paths:
            records.extend(yaml.safe_load(path.open()))
        print(f"    {len(records)} records")
        num_total += len(records)
        records = sorted(
            records,
            key=lambda x: (x["sub_exact_match"], x["idx"]),
        )
        all_data[task_name] = records

    print(f"Total number of records: {num_total}")
    if all_data:
        combined_path = out_dir / GENERATED_SAMPLES_ALL_FILENAME
        with open(combined_path, "w") as fp:
            yaml.safe_dump(all_data, fp)


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"

    mode = "collect"
    # mode = "sweep"
    dataset_size = "64k"
    # dataset_size = "128k"
    datasets = [
        f"helmet_nq_{dataset_size}",
        f"helmet_trivia_qa_{dataset_size}",
        f"helmet_hotpot_qa_{dataset_size}",
        f"helmet_pop_qa_{dataset_size}",
    ]
    cases = [
        "lr_4gpu_cs2048_lr5",
        "h2o_4gpu_cs2048_lr5",
        "slr_4gpu_cs2048_lr5",
        "qh2o_4gpu_cs2048_lr5",
        "h2onorm_4gpu_cs2048_lr5",
        "qh2onorm_4gpu_cs2048_lr5",
        "lr_4gpu_cs1024_lr5",
        "h2o_4gpu_cs1024_lr5",
    ]
    model_type = "lora"
    if mode == "collect":
        for dataset, case in product(datasets, cases):
            out_dir = base_path / dataset / case
            if out_dir.exists():
                main(out_dir, model_type)
            else:
                print(f"\nResults for {dataset}/{case} do not exist")
    elif mode == "sweep":
        names = []
        for dataset, case in product(datasets, cases):
            name = "/".join((dataset, case, GENERATED_SAMPLES_ALL_FILENAME))
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
