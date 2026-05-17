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
import yaml

from keys_values.finetune.longcontext_eval_ext import GENERATED_SAMPLES_FILENAME


def main(control_file: Path):
    setups = yaml.safe_load(control_file.open())
    num_total = 0
    for setup in setups:
        for task in setup["eval_tasks"]:
            num_task = 0
            base_path = Path(setup["out_dir"]) / task / "eval"
            for path in base_path.glob(GENERATED_SAMPLES_FILENAME.replace("{}", "*")):
                path.unlink()
                num_task += 1
            if num_task > 0:
                print(f"Removed {num_task} files from {base_path}")
            num_total += num_task
    print(f"\nRemoved {num_total} generated samples files in total")


if __name__ == "__main__":
    # dataset_size = "64k"
    dataset_size = "128k"
    control_file = (
        Path.home() / "sync" / "keys_values" / f"eval_inst1_{dataset_size}.yaml"
    )
    # control_file = Path.home() / "git" / "keys_values" / f"eval_inst2_3_{dataset_size}.yaml"
    main(control_file)
