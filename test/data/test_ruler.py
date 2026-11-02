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
import json
from pathlib import Path
from typing import List

import pytest

from keys_values.data.load_ruler_dev_eval import (
    EVAL_QUESTIONS_NUM,
    SUPPORTED_DATASET_KEYS,
    format_target,
    get_instruction_template,
    load_ruler,
    source_data_path,
)
from keys_values.data.ruler import Ruler
from keys_values.evaluation.evaluator import (
    SampleBasedMetricsEvaluator,
    compute_metric,
)


@pytest.mark.parametrize(
    "dataset_key, outputs, expected",
    [
        ("niah_single_1", ["5663623"], "5663623"),
        ("vt", ["AAAAA", "BBBBB", "CCCCC"], "AAAAA BBBBB CCCCC"),
        ("cwe", ["apple", "river", "stone"], "1. apple 2. river 3. stone"),
        ("fwe", ["abcdef", "ghijkl", "mnopqr"], "abcdef ghijkl mnopqr"),
    ],
)
def test_format_target(dataset_key: str, outputs: List[str], expected: str):
    assert format_target(dataset_key, outputs) == expected


def _write_source_file(path: Path, num_rows: int):
    # Mimics the format written by RULER's `prepare.py`
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx in range(num_rows):
            row = {
                "index": idx,
                "input": f"Input {idx}. Question: Which words?",
                "outputs": [f"word{idx}", f"other{idx}"],
                "length": 100,
                "length_w_model_temp": 100,
                "answer_prefix": " Answer: The words are:",
            }
            f.write(json.dumps(row) + "\n")


@pytest.mark.parametrize("dataset_key", ["vt", "cwe"])
def test_load_ruler(tmp_path: Path, dataset_key: str):
    num_samples = EVAL_QUESTIONS_NUM + 5
    data_path = source_data_path(dataset_key, "8k", str(tmp_path))
    # More rows than needed: only the first `num_samples` are used
    _write_source_file(data_path, num_samples + 3)
    dev_data, eval_data = load_ruler(dataset_key, "8k", data_path, num_samples)
    assert len(eval_data) == EVAL_QUESTIONS_NUM
    assert len(dev_data) == num_samples - EVAL_QUESTIONS_NUM
    for data in (dev_data, eval_data):
        assert set(data.column_names) == {
            "input",
            "output",
            "outputs",
            "query_id",
            "max_length",
        }
    # Evaluation set is the first part, no shuffling
    for idx, instance in enumerate(list(eval_data) + list(dev_data)):
        outputs = [f"word{idx}", f"other{idx}"]
        assert instance["input"] == (
            f"Input {idx}. Question: Which words? Answer: The words are:"
        )
        assert instance["outputs"] == outputs
        assert instance["output"] == format_target(dataset_key, outputs)
        assert instance["query_id"] == f"{dataset_key}_8k_{idx}"
        assert instance["max_length"] == "8k"


def test_load_ruler_not_enough_rows(tmp_path: Path):
    data_path = source_data_path("vt", "8k", str(tmp_path))
    _write_source_file(data_path, EVAL_QUESTIONS_NUM)
    with pytest.raises(ValueError):
        load_ruler("vt", "8k", data_path, EVAL_QUESTIONS_NUM + 1)


@pytest.mark.parametrize(
    "output, targets, expected",
    [
        ("The value is 5663623.", ["5663623"], 1.0),
        ("They are: AAAAA, bbbbb and DDDDD", ["AAAAA", "BBBBB", "CCCCC"], 2 / 3),
        ("1. apple 2. river", ["apple", "river", "stone", "cloud"], 0.5),
        ("nothing here", ["abcdef"], 0.0),
    ],
)
def test_string_match_all(output: str, targets: List[str], expected: float):
    assert compute_metric(output, targets, "string_match_all") == pytest.approx(
        expected
    )


@pytest.mark.parametrize("dataset_key", SUPPORTED_DATASET_KEYS)
def test_ruler_dataset_keys(dataset_key: str):
    metric = SampleBasedMetricsEvaluator.metric_for_ruler_task(dataset_key)
    assert metric in SampleBasedMetricsEvaluator.supported_metrics()
    template, slot_names = get_instruction_template(dataset_key)
    assert all("{" + name + "}" in template for name in slot_names)
    data = Ruler(dataset_key=dataset_key, max_length="256k")
    assert data.encode().kwargs["max_length"] == "256k"


def test_ruler_invalid_arguments():
    with pytest.raises(ValueError):
        Ruler(dataset_key="nq")
    with pytest.raises(ValueError):
        Ruler(dataset_key="vt", max_length="512k")
