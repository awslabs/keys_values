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
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from keys_values.finetune.resume_state import TRAINSTATE_REST_FNAME
from keys_values.scripts.cleanup_evaluation import datasets_and_cases


def _extract_train_data_index(
    case_path: Path,
) -> Optional[Tuple[int, ...]]:
    for setup_dir in case_path.iterdir():
        if not setup_dir.is_dir():
            continue
        name = setup_dir.name
        if name == "final" or (name.startswith("step-") and name[5:].isdigit()):
            state_path = setup_dir / TRAINSTATE_REST_FNAME
            if not state_path.exists():
                continue
            train_state = torch.load(state_path)
            if "data_state" not in train_state:
                print(f"{setup_dir}: {TRAINSTATE_REST_FNAME} has no 'data_state'")
                continue
            if "train_data_index" not in train_state["data_state"]:
                print(f"{setup_dir}: {TRAINSTATE_REST_FNAME}['data_state'] has no 'train_data_index'")
                continue
            return tuple(train_state["data_state"]["train_data_index"].tolist())
    return None


def main(
    dataset: str,
    base_paths: List[Path],
):
    histogram = Counter()
    for base_path in base_paths:
        data_path = base_path / dataset
        if data_path.exists() and data_path.is_dir():
            for case_path in data_path.iterdir():
                if not case_path.is_dir():
                    continue
                index = _extract_train_data_index(case_path)
                if index is not None:
                    histogram[index] += 1
    num_found = sum(histogram.values())
    if num_found > 0:
        tag = f"{dataset} [{num_found}]"
        if len(histogram) == 1:
            print(f"{tag}: Single train_data_index")
        else:
            print(f"{tag}: Multiple train_data_index's: {dict(histogram)}")


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"

    datasets = []
    for dataset_size in ("128k", "64k"):
        for extra_data in (False, True):
            _datasets, _ = datasets_and_cases(
                dataset_size=dataset_size,
                extra_data=extra_data,
                is_baseline=False,
                is_base_model=False,
            )
            datasets.extend(_datasets)
    base_paths = [base_path] + [
        base_path / name for name in ("rerun", "baseline", "basemode")
    ]
    for dataset in datasets:
        main(dataset, base_paths)
