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

from keys_values.utils import get_dict, set_dict

LONGBENCH_NUM_CASES = 503

LONGBENCH_BUCKET_SIZES = [(20, 1, 4)] * 9 + [(22, 1, 5)] + [(20, 1, 4)] * 10

MODEL_NAME = "Qwen3-4B-Instruct-2507"


def main(
    path: Path,
    do_overwrite: bool,
    prng: torch.Generator,
):
    with open(path, "r") as f:
        metadata = json.load(f)
    seq_lens = metadata["sequence_lengths"][MODEL_NAME]
    if len(seq_lens) != LONGBENCH_NUM_CASES:
        raise ValueError(
            f"metadata['sequence_lengths']['{MODEL_NAME}']: "
            f"Length {len(seq_lens)}, should be {LONGBENCH_NUM_CASES}"
        )
    meta_keys = ["train_val_test_split", MODEL_NAME]
    if not do_overwrite and get_dict(metadata, meta_keys) is not None:
        raise ValueError(
            f"metadata['train_val_test_split']['{MODEL_NAME}'] already exists"
        )
    sort_ind, _ = zip(*sorted(enumerate(seq_lens), key=lambda x: x[1]))
    sort_ind = torch.tensor(sort_ind)
    train_ind = None
    val_ind = None
    test_ind = None
    pos = 0
    for train_sz, val_sz, test_sz in LONGBENCH_BUCKET_SIZES:
        sz = train_sz + val_sz + test_sz
        tpv_sz = train_sz + val_sz
        ind_slice = sort_ind[pos:(pos + sz)]
        rnd_ind = torch.randperm(sz, generator=prng)
        train_new = ind_slice[rnd_ind[:train_sz]]
        val_new = ind_slice[rnd_ind[train_sz:tpv_sz]]
        test_new = ind_slice[rnd_ind[tpv_sz:]]
        if train_ind is None:
            train_ind = train_new
            val_ind = val_new
            test_ind = test_new
        else:
            train_ind = torch.cat((train_ind, train_new))
            val_ind = torch.cat((val_ind, val_new))
            test_ind = torch.cat((test_ind, test_new))
        pos += sz
    assert pos == LONGBENCH_NUM_CASES, f"pos = {pos} != {LONGBENCH_NUM_CASES}"
    # Shuffle `train_ind` randomly (the others don't matter)
    rnd_ind = torch.randperm(train_ind.numel(), generator=prng)
    train_ind = train_ind[rnd_ind]
    new_dict = {
        "train": train_ind.tolist(),
        "val": val_ind.tolist(),
        "test": test_ind.tolist(),
    }
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
