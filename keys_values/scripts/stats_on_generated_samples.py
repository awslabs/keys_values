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
from collections import Counter
import math
from pathlib import Path
from typing import List, Optional, Any, Tuple
import yaml

from transformers import AutoTokenizer

from keys_values.scripts.cleanup_evaluation import datasets_and_cases

BASE_MODEL_PATH = "/home/ubuntu/git/keys_values/checkpoints/Qwen/Qwen3-4B-Instruct-2507"


def _length_ratio(
    output: str,
    target: str,
    tokenizer: Optional[Any],
) -> Tuple[float, int]:
    if tokenizer is None:
        return len(output) / len(target), len(output)
    else:
        encoded_output = tokenizer.encode(output, add_special_tokens=False)
        encoded_target = tokenizer.encode(target, add_special_tokens=False)
        return len(encoded_output) / len(encoded_target), len(encoded_output)


def _extract_number(
    record: dict,
    tokenizer: Optional[Any],
) -> Tuple[float, int]:
    output = record["output"]
    raw_target = record["raw_target"]
    if isinstance(raw_target, str):
        target = raw_target
    elif record.get("sub_exact_match") == 1.0:
        target = max(
            (t for t in raw_target if t in output),
            key=len,
            default=max(raw_target, key=len),
        )
    else:
        if "sub_exact_match" not in record:
            print("UUPS!")
        target = max(raw_target, key=len)
    return _length_ratio(output, target, tokenizer)


def _find_yaml_files(
    base_path: Path,
    dataset: str,
    case: str,
    eval_dir: str,
    search_for_setups: bool,
) -> List[Path]:
    if not search_for_setups:
        directory = base_path / dataset / case / eval_dir
        if directory.exists():
            return sorted(directory.glob("generated_samples_*.yaml"))
        else:
            return []
    # Search for setup subdirectory that contains eval_dir
    case_path = base_path / dataset / case
    if case_path.exists():
        for setup_dir in case_path.iterdir():
            if not setup_dir.is_dir():
                continue
            name = setup_dir.name
            if name == "final" or (name.startswith("step-") and name[5:].isdigit()):
                candidate = setup_dir / eval_dir
                if candidate.is_dir():
                    return sorted(candidate.glob("generated_samples_*.yaml"))
    return []


def main(
    dataset: str,
    cases: List[str],
    base_path,
    eval_dir: str,
    search_for_setups: bool,
    tokenizer: Optional[Any],
    max_tokens: int,
):
    all_numbers: List[float] = []

    for case in cases:
        case_key = case[0] if isinstance(case, tuple) else case
        yaml_files = _find_yaml_files(
            base_path, dataset, case_key, eval_dir, search_for_setups
        )
        # if len(yaml_files) > 0:
        #     print(f"Found {len(yaml_files)} YAML files in {yaml_files[0].parent}")
        number_pairs: List[Tuple[float, int]] = []
        for path in yaml_files:
            with open(path) as f:
                records = yaml.safe_load(f)
            for record in records:
                number_pairs.append(_extract_number(record, tokenizer))
        numbers = [x[0] for x in number_pairs]
        num_entries = len(number_pairs)
        if numbers:
            mean = sum(numbers) / num_entries
            variance = sum((x - mean) ** 2 for x in numbers) / num_entries
            std = math.sqrt(variance)
            print(
                f"  ({dataset}, {case_key}): n={num_entries}, mean={mean:.4f}, std={std:.4f}"
            )
            if tokenizer is not None:
                num_tokens = [x[1] for x in number_pairs]
                num_eq_max = sum(x == max_tokens for x in num_tokens)
                num_gt_max = sum(x > max_tokens for x in num_tokens)
                hist_lt_max = Counter([x for x in num_tokens if x < max_tokens])
                print(
                    f"  ({dataset}, {case_key}): num_eq_max={num_eq_max}, num_gt_max={num_gt_max}, vals_lt_max={dict(hist_lt_max)}"
                )
        else:
            print(f"  ({dataset}, {case_key}): no data")
        all_numbers.extend(numbers)

    if all_numbers:
        mean = sum(all_numbers) / len(all_numbers)
        variance = sum((x - mean) ** 2 for x in all_numbers) / len(all_numbers)
        std = math.sqrt(variance)
        print(
            f"  ({dataset}, ALL): n={len(all_numbers)}, mean={mean:.4f}, std={std:.4f}"
        )


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"

    do_tokenize = True
    max_tokens = 128
    eval_dir = "eval_128"
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
    if is_rerun:
        base_path = base_path / "rerun"
    elif is_baseline:
        base_path = base_path / "baseline"
    elif is_base_model:
        base_path = base_path / "basemod"
    search_for_setups = not is_baseline and not is_base_model
    datasets, cases = datasets_and_cases(
        dataset_size,
        extra_data,
        is_baseline,
        is_base_model,
        with_short=False,
    )
    if do_tokenize:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    else:
        tokenizer = None
    for dataset in datasets:
        main(
            dataset,
            cases,
            base_path,
            eval_dir,
            search_for_setups,
            tokenizer,
            max_tokens,
        )
