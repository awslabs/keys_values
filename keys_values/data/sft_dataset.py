# Original: Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# Modification: Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from functools import partial
from typing import List, Dict, Optional, Callable, Union, Any

import torch
from torch.utils.data import Dataset

from litgpt.prompts import PromptStyle, Default
from litgpt.tokenizer import Tokenizer


INPUT_IDS_NAME = "input_ids"

LABELS_NAME = "labels"


class SFTDataset(Dataset):
    """
    Improved variant of :class:`litgpt.data.base.SFTDataset`.

    In particular, elem["token_counts"]["raw"] is not computed here, and
    included only if it is given or available from what is done anyway.
    Avoids extra costs due to tokenization.

    """
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: Tokenizer,
        prompt_style: Union[str, PromptStyle],
        max_seq_length: int = -1,
        mask_prompt: bool = True,
        ignore_index: int = -100,
        transform: Optional[Callable[[Dict[str, str]], Dict[str, str]]] = None,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_style = (
            prompt_style if isinstance(prompt_style, PromptStyle) else PromptStyle.from_name(prompt_style)
        )
        self.max_seq_length = max_seq_length
        self.mask_prompt = mask_prompt
        self.ignore_index = ignore_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.data[idx]
        if self.transform is not None:
            example = self.transform(example)
        prompt = self.prompt_style.apply(prompt=example["instruction"], **example)
        encoded_prompt = self.tokenizer.encode(
            prompt, max_length=self.max_seq_length,
        )
        encoded_response = self.tokenizer.encode(
            example["output"], bos=False, eos=True, max_length=self.max_seq_length,
        )
        encoded_prompt_and_response = torch.cat((encoded_prompt, encoded_response)).type(torch.int64)
        if 0 < self.max_seq_length < len(encoded_prompt_and_response):
            msl = self.max_seq_length
            encoded_prompt_and_response = encoded_prompt_and_response[:msl]
            encoded_prompt_and_response[msl - 1] = self.tokenizer.eos_id

        # The labels are the full prompt with response, but with the prompt masked out
        labels = encoded_prompt_and_response.clone()
        if self.mask_prompt:
            labels[: len(encoded_prompt)] = self.ignore_index

        token_counts = {"raw_plus_prompt_template": len(encoded_prompt_and_response)}
        raw_count = example.get("num_tokens_instruction")
        if raw_count is None and self.transform is None and isinstance(self.prompt_style, Default):
            raw_count = len(encoded_prompt)
        if raw_count is not None:
            token_counts["raw"] = raw_count + len(encoded_response)

        return {
            INPUT_IDS_NAME: encoded_prompt_and_response,
            LABELS_NAME: labels,
            "token_counts": token_counts,
        }


def get_sft_collate_fn(pad_id: int = 0, ignore_index: int = -100):
    """Returns the collate function for supervised finetuning (needed in the DataLoader).

    The collate function gets a list of dicts with keys `input_ids` and `labels`.
    It returns a dict with batched `input_ids` and `labels`. Also pads short sequences to the longest element in
    the batch. Optionally truncates all sequences to the specified maximum length.
    """
    return partial(_sft_collate_fn, pad_id=pad_id, ignore_index=ignore_index)


def common_collate_fn(
    samples: List[Dict[str, Union[torch.Tensor, Dict[str, Any]]]],
    pad_id: int = 0,
) -> Dict[str, Union[torch.Tensor, Dict[str, Any]]]:
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
        }
    }


def _sft_collate_fn(
    samples: List[Dict[str, Union[torch.Tensor, Dict[str, Any]]]],
    pad_id: int = 0,
    ignore_index: int = -100,
) -> Dict[str, Union[torch.Tensor, Dict[str, Any]]]:
    batched = common_collate_fn(samples, pad_id=pad_id)
    batched[LABELS_NAME] = torch.nn.utils.rnn.pad_sequence(
        [sample[key] for sample in samples],
        batch_first=True,
        padding_value=ignore_index,
    )
    return batched
