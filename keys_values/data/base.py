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
from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import torch
from torch.utils.data import Dataset

from litgpt import Tokenizer, PromptStyle


INPUT_IDS_NAME = "input_ids"

LABELS_NAME = "labels"

POSITION_NAME = "position"

LIT_MODEL_FNAME = "lit_model.pth"

HEAD_MODEL_FNAME = "head_model.pth"

LORA_WEIGHTS_FNAME = "lit_model.lora.pth"

LORA_WEIGHTS_FNAME_OLD = "lit_model.pth.lora"


class LongContextDataset(Dataset):
    """
    Base class for some datasets we define here.

    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: Tokenizer,
        prompt_style: Union[str, PromptStyle],
        max_seq_length: int = -1,
        transform: Optional[Callable[[Dict[str, str]], Dict[str, str]]] = None,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_style = (
            prompt_style
            if isinstance(prompt_style, PromptStyle)
            else PromptStyle.from_name(prompt_style)
        )
        self.max_seq_length = max_seq_length
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)


def get_pad_datacase() -> Dict[str, Any]:
    return {"PADDING_DONT_USE": 31415927}


def is_pad_datacase(x: Dict[str, Any]) -> bool:
    return x.get("PADDING_DONT_USE") == 31415927


def pad_dataset(
    dataset: List[Dict[str, Any]],
    batch_size: int,
    num_devices: int = 1,
) -> List[Dict[str, Any]]:
    """
    Pads dataset `dataset` so its length becomes a multiple of
    `batch_size * num_devices`.

    We also add a field :const:`POSITION_NAME` to each entry, containing the
    position in the complete dataset.

    """
    factor = batch_size * num_devices
    remainder = len(dataset) % factor
    extra = [get_pad_datacase()] * ((factor - remainder) % factor)
    return [{**x, POSITION_NAME: i} for i, x in enumerate(dataset + extra)]


def common_collate_fn(
    samples: List[Dict[str, Any]],
    pad_id: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    # Batch can contain padding entries
    _samples = samples
    samples = [x for x in samples if not is_pad_datacase(x)]
    if not samples:
        raise ValueError(
            f"common_collate_fn received all-padding samples: Cannot return empty batch:\n{_samples}"
        )
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [sample[INPUT_IDS_NAME] for sample in samples],
        batch_first=True,
        padding_value=pad_id,
    )
    names = ("raw_plus_prompt_template",)
    if all("raw" in x["token_counts"] for x in samples):
        names += ("raw",)
    return {
        INPUT_IDS_NAME: input_ids,
        "token_counts": {
            name: torch.tensor(
                [sample["token_counts"][name] for sample in samples],
                dtype=torch.int64,
            ).unsqueeze(1)
            for name in names
        },
    }, samples
