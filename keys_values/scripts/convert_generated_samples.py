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
import yaml

ENTRY_SEPARATOR = "|||"

SOURCE_FILENAME = "evaluation.jsonl"

TARGET_FILENAME = "generated_samples_0.yaml"

DATA_KEYS = [("instance_id", "idx"), ("output", "output"), ("expected", "raw_target")]


def main(base_path: Path, metric_name: str) -> None:
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
            if has_list_targets:
                trg_list = [
                    x.strip()
                    for x in trg_record[expected_keys[1]].split(ENTRY_SEPARATOR)
                ]
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
    base_path = Path("/mnt/efs/kbenidis/results/20260807_000000/finetuned")
    metric_name = "match_first_word_or_phrase"  # New setup
    # metric_name = "sub_exact_match"  # Old setup

    main(base_path, metric_name)
