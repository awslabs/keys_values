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
from typing import Tuple
import yaml

from keys_values.evaluation.evaluator import (
    compute_metric,
    METRICS_FOR_HELMET_TASKS,
)

DATASETS = [
    "clinc150_128k",
    "hotpot_qa_128k",
    "hotpot_qa_64k",
    "infinite_bench_mc_128k",
    "infinite_bench_qa_128k",
    "json_kv_128k",
    "nlu_128k",
    "nq_128k",
    "nq_64k",
    "pop_qa_128k",
    "pop_qa_64k",
    "trec_coarse_128k",
    "trivia_qa_128k",
]

SAMPLE_FNAME = "generated_samples_0.yaml"


def _strip_name(name: str) -> str:
    return name[:-4] if name.endswith("_64k") else name[:-5]


def main(path: Path, metric: str) -> Tuple[float, int]:
    with open(path, "r") as f:
        records = yaml.safe_load(f)
    metric_vals = [
        compute_metric(
            output=record["output"],
            targets=record["raw_target"],
            metric=metric,
        )
        for record in records
    ]
    num_vals = len(metric_vals)
    return sum(metric_vals) / num_vals, num_vals


if __name__ == "__main__":
    base_path = Path("/mnt/efs/kbenidis/results/20260807_000000")
    metric = "match_first_word_or_phrase"

    for dataset in DATASETS:
        metric = METRICS_FOR_HELMET_TASKS[_strip_name(dataset)]
        path = base_path / dataset / SAMPLE_FNAME
        if path.exists():
            avg_metric_val, num_vals = main(path, metric)
            print(f"{dataset}: {metric} = {(avg_metric_val * 100):.3f} [{num_vals}]")
        else:
            print(f"{dataset}: {path} not found")
