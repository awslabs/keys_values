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
from typing import Literal, Tuple, List, Union

from keys_values.evaluation.tasks import EvaluationTasks


def _append_old(case: str, multiple_tasks: bool) -> str:
    return case + "_old" if multiple_tasks else case


def datasets_and_cases(
    dataset_size: str,
    extra_data: bool,
    is_baseline: bool,
    is_base_model: bool,
    with_short: bool = False,
) -> Tuple[List[str], List[Union[str, Tuple[str, str]]]]:
    multiple_tasks = not is_baseline and not is_base_model
    if not extra_data:
        datasets = [
            f"helmet_nq_{dataset_size}",
            f"helmet_trivia_qa_{dataset_size}",
            f"helmet_hotpot_qa_{dataset_size}",
            f"helmet_pop_qa_{dataset_size}",
        ]
        if not with_short:
            cases = [
                _append_old("lr_4gpu_cs2048_lr5", multiple_tasks),
                "slr_4gpu_cs2048_lr5",
                "h2o_4gpu_cs2048_lr5",
                "h2onorm_4gpu_cs2048_lr5",
                "h2oorig_4gpu_cs2048_lr5",
                _append_old("lr_4gpu_cs1024_lr5", multiple_tasks),
                "slr_4gpu_cs1024_lr5",
                "h2o_4gpu_cs1024_lr5",
                "h2onorm_4gpu_cs1024_lr5",
                "h2oorig_4gpu_cs1024_lr5",
            ]
        else:
            cases = [
                (_append_old("lr_4gpu_cs2048_lr5", multiple_tasks), "lr_2048"),
                ("slr_4gpu_cs2048_lr5", "slr_2048"),
                ("h2o_4gpu_cs2048_lr5", "h2o_2048"),
                ("h2onorm_4gpu_cs2048_lr5", "h2onorm_2048"),
                ("h2oorig_4gpu_cs2048_lr5", "h2oorig_2048"),
                (_append_old("lr_4gpu_cs1024_lr5", multiple_tasks), "lr_1024"),
                ("slr_4gpu_cs1024_lr5", "slr_1024"),
                ("h2o_4gpu_cs1024_lr5", "h2o_1024"),
                ("h2onorm_4gpu_cs1024_lr5", "h2onorm_1024"),
                ("h2oorig_4gpu_cs1024_lr5", "h2oorig_1024"),
            ]
        if multiple_tasks:
            if not with_short:
                cases.extend(
                    [
                        "qh2o_4gpu_cs2048_lr5",
                        "qh2onorm_4gpu_cs2048_lr5",
                    ]
                )
                if dataset_size == "64k":
                    cases.extend(
                        [
                            "slr_4gpu_cs128_lr5",
                            "h2o_4gpu_cs128_lr5",
                            "h2onorm_4gpu_cs128_lr5",
                            "h2oorig_4gpu_cs128_lr5",
                        ]
                    )
            else:
                cases.extend(
                    [
                        ("qh2o_4gpu_cs2048_lr5", "qh2o_2048"),
                        ("qh2onorm_4gpu_cs2048_lr5", "qh2onorm_2048"),
                    ]
                )
                if dataset_size == "64k":
                    cases.extend(
                        [
                            ("slr_4gpu_cs128_lr5", "slr_128"),
                            ("h2o_4gpu_cs128_lr5", "h2o_128"),
                            ("h2onorm_4gpu_cs128_lr5", "h2onorm_128"),
                            ("h2oorig_4gpu_cs128_lr5", "h2oorig_128"),
                        ]
                    )
    else:
        datasets = [
            f"helmet_trec_coarse_{dataset_size}",
            f"helmet_nlu_{dataset_size}",
            f"helmet_clinc150_{dataset_size}",
            f"helmet_infinite_bench_qa_{dataset_size}",
            f"helmet_infinite_bench_mc_{dataset_size}",
            f"helmet_json_kv_{dataset_size}",
            #    f"helmet_trec_fine_{dataset_size}",
            #    f"helmet_banking77_{dataset_size}",
        ]
        if not with_short:
            cases = [
                "slr_4gpu_cs1024_lr5",
                "h2onorm_4gpu_cs1024_lr5",
                "h2oorig_4gpu_cs1024_lr5",
            ]
        else:
            cases = [
                ("slr_4gpu_cs1024_lr5", "slr_1024"),
                ("h2onorm_4gpu_cs1024_lr5", "h2onorm_1024"),
                ("h2oorig_4gpu_cs1024_lr5", "h2oorig_1024"),
            ]
    return datasets, cases


def main(
    out_dir: Path,
    model_type: str,
    mode: Literal["non-lock", "lock", "all"],
    multiple_tasks: bool,
    eval_dir: str = "eval",
):
    total_removed = 0
    eval_tasks = EvaluationTasks(
        out_dir=out_dir,
        model_type=model_type,
        multiple_tasks=multiple_tasks,
        eval_dir=eval_dir,
    )
    print(f"Removing files for {out_dir}")
    for task_name, incomplete_file_paths in eval_tasks.eval_result_files(mode):
        print(f"{task_name}: Removing {len(incomplete_file_paths)} files (type {mode})")
        for path in incomplete_file_paths:
            path.unlink()
        total_removed += len(incomplete_file_paths)
    print(f"Removed {total_removed} files in total (type {mode})")


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"

    eval_dir = "eval"
    # dataset_size = "64k"
    dataset_size = "128k"
    # is_rerun = False
    is_rerun = True
    is_baseline = False
    # is_baseline = True
    is_base_model = False
    # is_base_model = True
    extra_data = False
    # extra_data = True
    multiple_tasks = not is_baseline and not is_base_model
    if is_rerun:
        base_path = base_path / "rerun"
    elif is_baseline:
        base_path = base_path / "baseline"
    elif is_base_model:
        base_path = base_path / "basemod"
    datasets, cases = datasets_and_cases(
        dataset_size,
        extra_data,
        is_baseline,
        is_base_model,
    )

    # Use this to clean up lock files before restarting evaluation
    # mode = "lock"
    # Use this to remove all evaluation files
    mode: Literal["non-lock", "lock", "all"] = "all"
    model_type = "lora"
    for dataset, case in product(datasets, cases):
        out_dir = base_path / dataset / case
        if out_dir.exists():
            main(out_dir, model_type, mode, multiple_tasks, eval_dir)
        else:
            print(f"\nResults for {dataset}/{case} do not exist")
