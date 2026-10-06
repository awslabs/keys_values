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
import yaml
from pathlib import Path
from typing import List

from keys_values.scripts.recompute_val_losses import RESULT_FILENAME

SWEEP_TAR_FILENAME = "val_loss_transfer.tgz"

CONTROL_FILENAME = "eval_{instance}{extra}_{dataset_size}.yaml"


def main(
    base_path: Path,
    datasets: List[str],
    control_path: Path,
):
    # Collect all result files
    result_paths = []
    for dataset in datasets:
        data_path = base_path / dataset
        if data_path.exists():
            for model_path in data_path.iterdir():
                if model_path.is_dir():
                    result_path = model_path / RESULT_FILENAME
                    if result_path.exists():
                        result_paths.append(result_path)
    print(f"Found {len(result_paths)} result files")
    entry_list = [
        yaml.safe_load(open(result_path, "r")) for result_path in result_paths
    ]
    print(f"Writing combined results into {control_path}")
    with open(control_path, "w") as fp:
        yaml.safe_dump(entry_list, fp, sort_keys=False)


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp/lora/qwen3_4b"

    # dataset_size = "64k"
    dataset_size = "128k"
    is_rerun = True
    # is_rerun = False
    # extra_data = True
    extra_data = False
    # is_instance1 = True
    is_instance1 = False
    if is_rerun:
        base_path = base_path / "rerun"
    if not extra_data:
        datasets = [
            f"helmet_nq_{dataset_size}",
            f"helmet_trivia_qa_{dataset_size}",
            f"helmet_hotpot_qa_{dataset_size}",
            f"helmet_pop_qa_{dataset_size}",
        ]
    else:
        datasets = [
            f"helmet_trec_coarse_{dataset_size}",
            f"helmet_ms_macro_{dataset_size}",
            f"helmet_nlu_{dataset_size}",
            f"helmet_clinc150_{dataset_size}",
            f"helmet_infinite_bench_qa_{dataset_size}",
            f"helmet_infinite_bench_mc_{dataset_size}",
            f"helmet_json_kv_{dataset_size}",
            f"helmet_ruler_mk_uuid_{dataset_size}",
        ]
    control_path = (
        Path.home()
        / ("sync" if is_instance1 else "git")
        / "keys_values"
        / CONTROL_FILENAME.format(
            instance="inst1" if is_instance1 else "inst2_3",
            extra="_extra" if extra_data else "",
            dataset_size=dataset_size,
        )
    )
    main(base_path, datasets, control_path)
