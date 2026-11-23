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
from itertools import batched
import json
from pathlib import Path
from typing import List
import yaml

ENTRY_SEPARATOR = "|||"

SOURCE_FILENAME = "evaluation.jsonl"

TARGET_FILENAME = "generated_samples_0.yaml"

DATA_KEYS = [("instance_id", "idx"), ("output", "output"), ("expected", "raw_target")]


def ruler_cwe_expected_to_raw_target(expected: str) -> List[str]:
    parts = expected.split()
    assert len(parts) == 20, (expected, parts)
    nums, result = zip(*batched(parts, 2))
    should_be = tuple(str(x) + "." for x in range(1, 11))
    assert nums == should_be, (expected, nums, should_be)
    return list(result)


def main(
    base_path: Path,
    metric_name: str,
    is_ruler_cwe: bool = False,
) -> None:
    expected_keys = DATA_KEYS[-1]
    output_key = DATA_KEYS[1][1]
    for setup_dir in base_path.iterdir():
        if not setup_dir.is_dir():
            continue
        src_path = setup_dir / SOURCE_FILENAME
        if not src_path.exists():
            print(f"{setup_dir.name}: No {SOURCE_FILENAME} file. Skipping")
            continue
        with open(src_path, "r") as fp:
            records = [json.loads(line) for line in fp]
        has_list_targets = any(ENTRY_SEPARATOR in r[expected_keys[0]] for r in records)
        yaml_records = []
        for src_record in records:
            trg_record = {trg: src_record[src] for src, trg in DATA_KEYS}
            trg_list = None
            if is_ruler_cwe:
                trg_list = ruler_cwe_expected_to_raw_target(
                    src_record[expected_keys[0]]
                )
            elif has_list_targets:
                trg_list = [
                    x.strip()
                    for x in trg_record[expected_keys[1]].split(ENTRY_SEPARATOR)
                ]
            if trg_list is not None:
                trg_record[expected_keys[1]] = trg_list
                trg_record[metric_name] = float(
                    any(t in trg_record[output_key] for t in trg_list)
                )
            yaml_records.append(trg_record)
        # Store result as YAML file
        trg_path = setup_dir / TARGET_FILENAME
        print(f"Write {str(trg_path)}")
        with open(trg_path, "w") as fp:
            yaml.dump(yaml_records, fp)


if __name__ == "__main__":
    metric_name = "string_match_all"
    for subdir, is_ruler_cwe in [
        ("20261118_134917", True),
        ("20261118_145829", False),
        ("20261118_165442", False),
        ("20261118_155631", False),
    ]:
        base_path = Path("/mnt/efs/kbenidis/results") / subdir / "finetuned"
        main(base_path, metric_name, is_ruler_cwe)
