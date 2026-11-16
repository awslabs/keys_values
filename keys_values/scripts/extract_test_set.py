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

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    auto_download_checkpoint,
    init_out_dir,
)

from keys_values.constants import DEFAULT_IGNORE_INDEX, DEFAULT_PAD_ID
from keys_values.data.constants import INPUT_IDS_NAME, ORIG_IDX_NAME, TARGETS_STRINGS_NAME
from keys_values.evaluation.longcontext_eval_ext import get_dataloader
from keys_values.head_model import CrossEntropyOnLogits


def setup(
    checkpoint_dir: Path,
    out_path: str,
    data: DataModule,
) -> None:
    """Write out test dataset as YAML file

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory.
            Just needed to load the tokenizer.
        out_path: Result YAML file is written there
        data: Data-related arguments. Mandatory

    """
    if data is None:
        raise ValueError("--data needs to be provided")
    checkpoint_dir = auto_download_checkpoint(
        model_name=checkpoint_dir,
        access_token=None,
    )
    out_path = Path(out_path)
    if data.metadata_dir is not None:
        data.metadata_dir = str(init_out_dir(Path(data.metadata_dir)))
    tokenizer = Tokenizer(checkpoint_dir)
    test_dataloader = get_dataloader(
        data=data,
        tokenizer=tokenizer,
        eval_tasks=None,
        head_model=CrossEntropyOnLogits.NAME,
        batch_size=1,
        devices=1,
        model_name=None,
    )
    ignore_index = getattr(data, "ignore_index", DEFAULT_IGNORE_INDEX)

    result = []
    for batch in test_dataloader:
        if not batch:
            print("Empty batch: Continue")
            continue
        orig_idxs = batch[ORIG_IDX_NAME]
        input_ids = batch[INPUT_IDS_NAME]
        targets = batch["targets"]
        raw_targets = batch[TARGETS_STRINGS_NAME]
        assert len(orig_idxs) == 1  # Sanity check
        assert len(input_ids) == 1
        assert len(targets) == 1
        assert len(raw_targets) == 1
        target = targets[0]
        prompt_len = input_ids.shape[1] - targets.shape[1] + 1
        prompt = input_ids[0, :prompt_len]
        record = {
            "idx": orig_idxs[0],
            "prompt": tokenizer.decode(prompt[prompt != DEFAULT_PAD_ID]),
            "raw_target": raw_targets[0],
            "sft_target": tokenizer.decode(target[target != ignore_index]),
        }
        result.append(record)

    print(f"Writing {len(result)} records to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fp:
        yaml.safe_dump(result, fp)
