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
import json

import torch

from keys_values.data.constants import METADATA_SEQ_LENGTHS_KEY
from keys_values.data.longbench_v2 import (
    sample_stratified_split,
    LONGBENCH_NUM_CASES,
    METADATA_TRAIN_VAL_TEST_SPLIT_KEY,
)
from keys_values.utils import get_dict, set_dict

MODEL_NAME = "Qwen3-4B-Instruct-2507"


def main(
    path: Path,
    do_overwrite: bool,
    prng: torch.Generator,
):
    with open(path, "r") as f:
        metadata = json.load(f)
    seq_lens = metadata[METADATA_SEQ_LENGTHS_KEY][MODEL_NAME]
    if len(seq_lens) != LONGBENCH_NUM_CASES:
        raise ValueError(
            f"metadata['{METADATA_SEQ_LENGTHS_KEY}']['{MODEL_NAME}']: "
            f"Length {len(seq_lens)}, should be {LONGBENCH_NUM_CASES}"
        )
    meta_keys = [METADATA_TRAIN_VAL_TEST_SPLIT_KEY, MODEL_NAME]
    if not do_overwrite and get_dict(metadata, meta_keys) is not None:
        raise ValueError(
            f"metadata['{METADATA_TRAIN_VAL_TEST_SPLIT_KEY}']['{MODEL_NAME}'] already exists"
        )
    new_dict = sample_stratified_split(seq_lens, prng)
    set_dict(metadata, meta_keys, new_dict)
    print(f"Write back metadata to {path}")
    with open(path, "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    path = Path.home() / "out/finetune/data/longbench_v2_metadata.json"
    do_overwrite = False
    seed = 79366120

    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    prng = torch.Generator().manual_seed(seed)
    main(path, do_overwrite, prng)
