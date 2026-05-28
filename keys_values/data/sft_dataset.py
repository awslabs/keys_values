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
import math
from typing import List, Dict, Optional, Callable, Union, Any

import torch

from litgpt.prompts import PromptStyle, Default
from litgpt.tokenizer import Tokenizer

from keys_values.data.base import (
    LongContextDataset,
    common_collate_fn,
    is_pad_datacase,
)
from keys_values.data.constants import TARGETS_STRINGS_NAME, POSITION_NAME
from keys_values.data import INPUT_IDS_NAME, LABELS_NAME


class SFTDataset(LongContextDataset):
    """
    Improved variant of :class:`litgpt.data.base.SFTDataset`.

    In particular, elem["token_counts"]["raw"] is not computed here, and
    included only if it is given or available from what is done anyway.
    Avoids extra costs due to tokenization.

    It is admissible for `data[idx]["output"]` to be a list of strings.
    In this case, we choose one of them at random in each
    :meth:`__getitem__` call. The semantics is that any of the entries
    is a valid target sequence.

    If `retain_targets_strings == True`, we also append the original
    targets `data[idx]["output"]`, either string or list of strings,
    as :const:`TARGETS_STRINGS_NAME` field. This is needed by a number of
    evaluation metrics.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: Tokenizer,
        prompt_style: Union[str, PromptStyle],
        max_seq_length: Optional[int] = None,
        mask_prompt: bool = True,
        ignore_index: int = -100,
        transform: Optional[Callable[[Dict[str, str]], Dict[str, str]]] = None,
        target_choice: Optional[List[int]] = None,
        seed: Optional[int] = None,
        retain_targets_strings: bool = True,
    ) -> None:
        super().__init__(
            data,
            tokenizer,
            prompt_style,
            max_seq_length,
            transform,
        )
        self.mask_prompt = mask_prompt
        self.ignore_index = ignore_index
        if target_choice is None:
            if seed is None:
                seed = 31415927
            target_choice = sample_target_choice(
                data,
                generator=torch.Generator().manual_seed(seed),
            )
        else:
            if len(target_choice) != len(data):
                raise ValueError(
                    f"len(target_choice) = {len(target_choice)} != {len(data)} = len(data)"
                )
            if not all(x >= 0 for x in target_choice):
                raise ValueError("target_choice must all be non-negative")
        self.target_choice = target_choice
        self._retain_target_strings = retain_targets_strings

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.data[idx]
        if is_pad_datacase(example):
            return example
        if self.transform is not None:
            example = self.transform(example)
        prompt = self.prompt_style.apply(prompt=example["instruction"], **example)
        max_length = -1 if self.max_seq_length is None else self.max_seq_length
        encoded_prompt = self.tokenizer.encode(
            prompt,
            max_length=max_length,
        )
        num_tokens_prompt = encoded_prompt.numel()
        targets = example["output"]
        if isinstance(targets, list):
            _targets = targets[self.target_choice[idx]]
        else:
            _targets = targets
        encoded_response = self.tokenizer.encode(
            _targets,
            bos=False,
            eos=True,
            max_length=max_length,
        )
        num_tokens_response = encoded_response.numel()
        encoded_prompt_and_response = torch.cat(
            (encoded_prompt, encoded_response)
        ).type(torch.int64)
        if 0 < max_length < num_tokens_prompt + num_tokens_response:
            encoded_prompt_and_response = encoded_prompt_and_response[:max_length]
            encoded_prompt_and_response[max_length - 1] = self.tokenizer.eos_id

        # The labels are the full prompt with response, but with the prompt masked out
        labels = encoded_prompt_and_response.clone()
        if self.mask_prompt:
            labels[:num_tokens_prompt] = self.ignore_index

        token_counts = {
            "raw_plus_prompt_template": num_tokens_prompt + num_tokens_response
        }
        raw_count = example.get("num_tokens_instruction")
        if (
            raw_count is None
            and self.transform is None
            and isinstance(self.prompt_style, Default)
        ):
            raw_count = num_tokens_prompt
        if raw_count is not None:
            token_counts["raw"] = raw_count + num_tokens_response

        result = {
            INPUT_IDS_NAME: encoded_prompt_and_response,
            LABELS_NAME: labels,
            "token_counts": token_counts,
        }
        if POSITION_NAME in example:
            result[POSITION_NAME] = example[POSITION_NAME]
        if self._retain_target_strings:
            result[TARGETS_STRINGS_NAME] = targets
        return result


def sample_target_choice(
    data: List[Dict[str, Any]],
    generator: torch.Generator,
) -> List[int]:
    # For padding cases, we return 0 as target choice
    num_choices = [
        (
            len(example["output"])
            if not is_pad_datacase(example) and isinstance(example["output"], list)
            else 1
        )
        for example in data
    ]
    num_data = len(data)
    if all(x == num_choices[0] for x in num_choices):
        return torch.randint(
            0, num_choices[0], (num_data,), generator=generator
        ).tolist()
    else:
        unif_random = torch.rand((num_data,), dtype=torch.float64, generator=generator)
        return [
            min(int(math.floor(u * mx)), mx - 1)
            for u, mx in zip(unif_random.tolist(), num_choices)
        ]


def get_sft_collate_fn(pad_id: int = 0, ignore_index: int = -100):
    """Returns the collate function for supervised finetuning (needed in the DataLoader).

    The collate function gets a list of dicts with keys `input_ids` and `labels`.
    It returns a dict with batched `input_ids` and `labels`. Also pads short sequences to the longest element in
    the batch. Optionally truncates all sequences to the specified maximum length.
    """
    return partial(_sft_collate_fn, pad_id=pad_id, ignore_index=ignore_index)


def _sft_collate_fn(
    samples: List[Dict[str, Any]],
    pad_id: int = 0,
    ignore_index: int = -100,
) -> Dict[str, Union[torch.Tensor, Dict[str, Any]]]:
    batched, samples = common_collate_fn(samples, pad_id=pad_id)
    batched[LABELS_NAME] = torch.nn.utils.rnn.pad_sequence(
        [sample[LABELS_NAME] for sample in samples],
        batch_first=True,
        padding_value=ignore_index,
    )
    if TARGETS_STRINGS_NAME in samples[0]:
        batched[TARGETS_STRINGS_NAME] = [
            sample[TARGETS_STRINGS_NAME] for sample in samples
        ]
    return batched
