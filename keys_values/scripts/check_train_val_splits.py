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
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import torch

from keys_values.data.constants import METADATA_TRAIN_VAL_SPLIT_KEY
from keys_values.data.helmet import METADATA_FNAME
from keys_values.finetune.resume_state import TRAINSTATE_REST_FNAME
from keys_values.scripts.cleanup_evaluation import datasets_and_cases
from keys_values.utils import get_dict, set_dict


def _extract_train_data_index(
    case_path: Path,
    indexes: Dict[str, Any],
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
            train_ind = train_state["data_state"]["train_data_index"].tolist()
            if indexes["train"] is None:
                indexes["train"] = train_ind
                indexes["val"] = train_state["data_state"]["val_data_index"].tolist()
            return tuple(train_ind)
    return None


def _metadata_keys_prefix(
    dataset: str,
    model_name: str,
) -> List[str]:
    parts = dataset.split("_")
    assert parts[0] == "helmet" and dataset[-1] == "k"
    dataset_size = parts[-1]
    dataset_key = "_".join(parts[1:-1])
    return [
        METADATA_TRAIN_VAL_SPLIT_KEY,
        dataset_key,
        dataset_size,
        model_name,
    ]


def _write_back_split(
    dataset: str,
    indexes: Dict[str, Any],
    model_name: str,
    metadata_dir: Path,
    val_split_fraction: str,
):
    meta_path = metadata_dir / METADATA_FNAME
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} does not exist")
    if indexes["train"] is None or indexes["val"] is None:
        raise ValueError(f"indexes must contain 'train' and 'val' entries:\n{indexes}")
    with meta_path.open("r") as fp:
        data = json.load(fp)
    keys = _metadata_keys_prefix(dataset, model_name) + [val_split_fraction]
    if get_dict(data, keys) is not None:
        print(f"Metadata file {meta_path} already has entries for {keys}: They are overwritten")
    set_dict(data, keys, indexes)
    with meta_path.open("w") as fp:
        json.dump(data, fp)
    print(f"Metadata stored in {meta_path} under {keys}")


def main(
    dataset: str,
    base_paths: List[Path],
    metadata_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    val_split_fraction: Optional[str] = None,
):
    if metadata_dir is not None:
        assert model_name is not None
        assert val_split_fraction is not None
    histogram = Counter()
    indexes: Dict[str, Any] = {"train": None, "val": None}
    for base_path in base_paths:
        data_path = base_path / dataset
        if data_path.exists() and data_path.is_dir():
            for case_path in data_path.iterdir():
                if not case_path.is_dir():
                    continue
                index = _extract_train_data_index(case_path, indexes)
                if index is not None:
                    histogram[index] += 1
    num_found = sum(histogram.values())
    if num_found > 0:
        tag = f"{dataset} [{num_found}]"
        if len(histogram) == 1:
            print(f"{tag}: Single train_data_index")
            if metadata_dir is not None:
                _write_back_split(
                    dataset,
                    indexes,
                    model_name,
                    metadata_dir,
                    val_split_fraction,
                )
        else:
            print(f"{tag}: Multiple train_data_index's: {dict(histogram)}")


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"
    # Set these to write back unique splits:
    metadata_dir = Path.home() / "out/finetune/data"
    model_name = "Qwen3-4B-Instruct-2507"
    val_split_fraction = "0.1"
    # metadata_dir = None
    # model_name = None
    # val_split_fraction = None

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
        main(
            dataset,
            base_paths,
            metadata_dir,
            model_name,
            val_split_fraction,
        )
