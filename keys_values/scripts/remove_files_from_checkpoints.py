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

NAMES_TO_REMOVE = (
    "tokenizer.json",
    "tokenizer_config.json",
)


def main(base_path: Path):
    total_removed = 0
    num_removed = {k: 0 for k in NAMES_TO_REMOVE}
    for root, _, files in base_path.walk():
        for name in files:
            if name in NAMES_TO_REMOVE:
                num_removed[name] += 1
                (root / name).unlink()
                total_removed += 1
                if total_removed % 50 == 0:
                    print(total_removed)


if __name__ == "__main__":
    base_path = Path.home() / "out/finetune/neurips_exp"
    main(base_path)
