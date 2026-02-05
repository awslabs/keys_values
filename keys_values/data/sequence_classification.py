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
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union, Iterable

import torch
from torch import Tensor

from litgpt.prompts import PromptStyle, Default
from litgpt.tokenizer import Tokenizer

from keys_values.data.base import (
    INPUT_IDS_NAME,
    LABELS_NAME,
    LongContextDataset,
    common_collate_fn,
    is_pad_datacase,
)


class SequenceClassificationDataset(LongContextDataset):
    """
    An in-memory dataset for supervised finetuning of a sequence classification
    head.

    Args:
        data: A list of samples (dicts). The target/label must be stored under
            the key 'output', the instruction under the key 'instruction'. The
            latter is mapped to the prompt via `prompt_style`.
        tokenizer: The tokenizer to use. Should match the one that was used to
            pretrain the model.
        prompt_style: The style to apply to prompts. See `litgpt.prompts` for a
            list of available styles.
        class_labels: List of class labels. For each entry `x` of `data`,
            `x['output']` must be equal to an entry in this list.
        max_seq_length: Truncate sequences that are longer than this value. By
            default, no truncation is applied.

    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: Tokenizer,
        prompt_style: Union[str, PromptStyle],
        class_labels: Iterable[str],
        max_seq_length: int = -1,
        transform: Optional[Callable[[Dict[str, str]], Dict[str, str]]] = None,
    ) -> None:
        super().__init__(
            data,
            tokenizer,
            prompt_style,
            max_seq_length,
            transform,
        )
        self.class_labels = tuple(class_labels)
        self._label_indexes = None
        self._transform_labels()

    def _transform_labels(self):
        if len(set(self.class_labels)) != len(self.class_labels):
            raise ValueError(
                f"class_labels = {self.class_labels}, must not have duplicate entries"
            )
        self._label_indexes = []
        is_in_padding = False
        for idx, example in enumerate(self.data):
            if is_pad_datacase(example):
                is_in_padding = True
            else:
                if is_in_padding:
                    raise ValueError(
                        f"data can only contain pad entries at the end, but entry {idx} is not padding, while pad entries before"
                    )
                label = example["output"]
                pos = next(
                    (i for i, cl in enumerate(self.class_labels) if cl == label), None
                )
                if pos is None:
                    raise ValueError(
                        f"data[{idx}]['output'] = '{label}' invalid, must lie in {self.class_labels}"
                    )
                self._label_indexes.append(pos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not (0 <= idx < len(self.data)):
            raise IndexError(
                f"index {idx} out of range, must be in [0, {len(self.data)})"
            )
        example = self.data[idx]
        if is_pad_datacase(example):
            return example
        label_idx = self._label_indexes[idx]
        if self.transform is not None:
            example = self.transform(example)
        prompt = self.prompt_style.apply(prompt=example["instruction"], **example)
        encoded_prompt = self.tokenizer.encode(
            prompt, bos=False, eos=True, max_length=self.max_seq_length
        )
        token_counts = {"raw_plus_prompt_template": len(encoded_prompt)}
        raw_count = example.get("num_tokens_instruction")
        if (
            raw_count is None
            and self.transform is None
            and isinstance(self.prompt_style, Default)
        ):
            raw_count = len(encoded_prompt)
        if raw_count is not None:
            token_counts["raw"] = raw_count
        return {
            INPUT_IDS_NAME: encoded_prompt,
            LABELS_NAME: label_idx,
            "token_counts": token_counts,
        }


def get_seq_class_collate_fn(pad_id: int = 0):
    return partial(_seq_class_collate_fn, pad_id=pad_id)


def _seq_class_collate_fn(
    samples: List[Dict[str, Any]],
    pad_id: int = 0,
) -> Dict[str, Union[Tensor, Dict[str, Any]]]:
    batched, samples = common_collate_fn(samples, pad_id=pad_id)
    batched[LABELS_NAME] = torch.tensor(
        [sample[LABELS_NAME] for sample in samples],
        dtype=torch.int64,
    )
    return batched
