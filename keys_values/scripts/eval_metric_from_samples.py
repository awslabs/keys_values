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
from typing import Tuple, Union
import yaml

from keys_values.evaluation.evaluator import (
    compute_metric,
    SampleBasedMetricsEvaluator,
)
from keys_values.evaluation.longcontext_eval_ext import GENERATED_SAMPLES_FILENAME

DATASETS = [
    "nq_64k",
    "trivia_qa_64k",
    "hotpot_qa_64k",
    "pop_qa_64k",
    "nq_128k",
    "trivia_qa_128k",
    "hotpot_qa_128k",
    "pop_qa_128k",
]


def _strip_name(name: str) -> str:
    return name[:-4] if name.endswith("_64k") else name[:-5]


def main(
    setup_path: Path,
    metric: str,
    eval_name: str,
    search_through_checkpoints: bool,
) -> Union[Tuple[float, int], str]:
    eval_path = None
    if not search_through_checkpoints:
        eval_path = setup_path / eval_name
        if not eval_path.exists():
            return f"{eval_path} does not exist. Skipping."
    else:
        for path in setup_path.glob("step-00*"):
            if path.is_dir():
                _eval_path = path / eval_name
                if _eval_path.exists():
                    if eval_path is not None:
                        return f"Found {eval_path} and {_eval_path}, must be one only. Skipping."
                    eval_path = _eval_path
        if eval_path is None:
            return f"No eval path step-*/{eval_name} found at {setup_path}. Skipping."

    metric_vals = []
    for path in eval_path.glob(GENERATED_SAMPLES_FILENAME.replace("{}", "*")):
        with open(path, "r") as f:
            records = yaml.safe_load(f)
        new_vals = [
            compute_metric(
                output=record["output"],
                targets=record["raw_target"],
                metric=metric,
            )
            for record in records
        ]
        metric_vals.extend(new_vals)
    num_vals = len(metric_vals)
    return sum(metric_vals) / num_vals, num_vals


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b/rerun"
    use_old_metrics = True
    fixed_metric = None
    eval_name = "eval_128"
    search_through_checkpoints = True

    skip_lines = []
    results = dict()
    for dataset in DATASETS:
        if fixed_metric is not None:
            metric = fixed_metric
        else:
            metric = SampleBasedMetricsEvaluator.metric_for_helmet_task(
                _strip_name(dataset),
                old_setup=use_old_metrics,
            )
        data_path = base_path / ("helmet_" + dataset)
        for setup_path in data_path.glob("*"):
            if setup_path.is_dir():
                result = main(setup_path, metric, eval_name, search_through_checkpoints)
                if isinstance(result, str):
                    skip_lines.append(result)
                else:
                    avg_metric_val, num_vals = result
                    setup_name = setup_path.stem
                    entries = results.get(setup_name, dict())
                    entries[dataset] = avg_metric_val
                    results[setup_name] = entries
                    print(
                        f"{dataset}/{setup_path.name}: {metric} = {(avg_metric_val * 100):.3f} [{num_vals}]"
                    )
    print("\n".join(skip_lines))
    # Print table entries from `results`
    print("\n")
    for setup_name, entries in results.items():
        print(setup_name)
        for i, dataset in enumerate(DATASETS):
            v = entries.get(dataset)
            is_last = i == len(DATASETS) - 1
            if v is not None:
                if search_through_checkpoints:
                    row = r" {\small\!" + f"{v * 100:.1f}" + "} & - "
                    if is_last:
                        row += r"\\"
                    else:
                        row += "&"
                else:
                    row = r" - & {\small\!" + f"{v * 100:.1f}" + "} "
                    if is_last:
                        row += r"\\"
                    else:
                        row += "&"
            else:
                row = " ... " + dataset + "..."
            print(row)
