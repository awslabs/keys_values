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
from typing import Dict, Any, Optional

import torch

from keys_values.data.sft_dataset import INPUT_IDS_NAME, LABELS_NAME
from keys_values.head_model import CrossEntropyOnLogits
from keys_values.head_model_factory import SUPPORTED_HEAD_MODELS


class BatchTransform:
    """
    Transforms a batch emitted by a :class:`DataLoader` object into the form
    used by our fine-tuning code.

    Subclasses need to be specific to datasets and data loaders being used.

    """

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        The resulting dictionary has keys "input_ids" and "targets". Here,
        `input_ids.shape = (bs, n_inp)`, `targets.shape = (bs, n_trg)`, so that
        `n_inp >= n_trg`. Here, `targets[:, -k]` are targets corresponding to
        inputs `input_ids[:, -k]`, whereas there are no loss potentials on the
        initial `input_ids[:, :(n_inp - n_trg)]`, if any.

        Args:
            batch: Dictionary emitted by :class:`DataLoader` object

        Returns:
            Dictionary with keys "input_ids" and "targets", see above. If `batch`
            has entries other than corresponding to these, they are copied here
            as well.

        """
        raise NotImplementedError


class SFTBatchTransform(BatchTransform):
    """
    Batch transform for standard supervised fine-tuning, as represented by
    :class:`SFTDataset`.

    First, for every sequence, we have `input_ids` and `labels` of the same
    length. For same `n_prompt`, we have that `input_ids[n_prompt:] ==
    labels[n_prompt:]` and `labels[:n_prompt] == ignore_index`.

    Second, the batch collator does right padding, `input_ids` with `pad_id`,
    `labels` with `ignore_index`.

    """

    def __init__(
        self,
        ignore_index: int = -100,
        pad_id: int = 0,
        eos_id: Optional[int] = None,
    ):
        self.ignore_index = ignore_index
        self.pad_id = pad_id
        self._eos_id = eos_id

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = batch.get(INPUT_IDS_NAME)
        labels = batch.get(LABELS_NAME)
        if input_ids is None or labels is None:
            raise ValueError(
                f"batch.keys() = {list(batch.keys())}, must contain 'input_ids', 'labels'"
            )
        if (
            input_ids.ndim != 2
            or input_ids.shape[0] == 0
            or input_ids.shape[1] == 0
            or input_ids.shape != labels.shape
        ):
            raise ValueError(
                f"batch['input_ids'].shape = {input_ids.shape}, batch['labels'].shape = {labels.shape}: Must be 2D and the same"
            )
        batch_size, seq_length = input_ids.shape
        left_ignore = [
            next(i for i, x in enumerate(label) if x != self.ignore_index)
            for label in labels
        ]
        right_ignore = [
            next(i for i, x in enumerate(reversed(label)) if x != self.ignore_index)
            for label in labels
        ]
        if self._eos_id is not None:
            # Check whether <eos> tokens are in place
            kwargs = dict(dtype=input_ids.dtype, device=input_ids.device)
            for i, (input_id, label, ri) in enumerate(
                zip(input_ids, labels, right_ignore)
            ):
                sz = ri + 1
                should_be = torch.full((sz,), self.pad_id, **kwargs)
                should_be[0] = self._eos_id
                tail = input_id[(-sz):]
                if not (tail == should_be).all().item():
                    print(
                        f"Slot {i}: inputs_ids, wrong end: {tail} (should be {should_be})"
                    )
                should_be = torch.full((sz,), self.ignore_index, **kwargs)
                should_be[0] = self._eos_id
                tail = label[(-sz):]
                if not (tail == should_be).all().item():
                    print(
                        f"Slot {i}: labels, wrong end: {tail} (should be {should_be})"
                    )
        max_ignore = max(left_ignore)
        extra_left = [max_ignore - num for num in left_ignore]
        span = max(extra_left)
        max_ignore = max(max_ignore, 1)
        if span > 0:
            total_right = [ri + span - el for ri, el in zip(right_ignore, extra_left)]
            new_length = seq_length + span - min(total_right)
            new_input_ids = torch.full(
                (batch_size, new_length - 1),
                self.pad_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            new_labels = torch.empty(
                (batch_size, new_length - max_ignore),
                dtype=labels.dtype,
                device=labels.device,
            )
            temp_row = torch.empty(
                (new_length,), dtype=labels.dtype, device=labels.device
            )
            for i, (input_id, label, start) in enumerate(
                zip(input_ids, labels, extra_left)
            ):
                end = min(start + input_id.shape[0], new_length - 1)
                new_input_ids[i, start:end] = input_id[: (end - start)]
                temp_row.fill_(self.ignore_index)
                end = min(start + label.shape[0], new_length)
                temp_row[start:end] = label[: (end - start)]
                new_labels[i] = temp_row[max_ignore:]
        else:
            new_input_ids = input_ids[:, :-1]
            new_labels = labels[:, max_ignore:]
        return dict(
            {k: v for k, v in batch.items() if k not in (INPUT_IDS_NAME, LABELS_NAME)},
            input_ids=new_input_ids,
            targets=new_labels,
        )


class SequenceClassificationBatchTransform(BatchTransform):
    """
    Batch transform for asequence classification, as represented by
    :class:`SequenceClassificationDataset`.

    All we do here is convert right padding in `input_ids` into left
    padding. The model will learn to produce the class probabilities for the
    <eos> input.

    """

    def __init__(
        self,
        pad_id: int = 0,
        eos_id: Optional[int] = None,
    ):
        self.pad_id = pad_id
        self._eos_id = eos_id

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = batch.get(INPUT_IDS_NAME)
        labels = batch.get(LABELS_NAME).flatten()
        if input_ids is None or labels is None:
            raise ValueError(
                f"batch.keys() = {list(batch.keys())}, must contain 'input_ids', 'labels'"
            )
        if (
            input_ids.ndim != 2
            or input_ids.shape[0] == 0
            or input_ids.shape[1] == 0
            or input_ids.shape[0] != labels.shape[0]
        ):
            raise ValueError(
                f"batch['input_ids'].shape = {input_ids.shape}, batch['labels'].shape = {labels.shape}: Invalid"
            )
        batch_size, seq_length = input_ids.shape
        # Right padding size per slot
        right_pad = [
            next(i for i, x in enumerate(reversed(input_id)) if x != self.pad_id)
            for input_id in input_ids
        ]
        kwargs = dict(dtype=input_ids.dtype, device=input_ids.device)
        if self._eos_id is not None:
            # Check whether <eos> tokens are in place
            for i, (input_id, rp) in enumerate(zip(input_ids, right_pad)):
                sz = rp + 1
                should_be = torch.full((sz,), self.pad_id, **kwargs)
                should_be[0] = self._eos_id
                tail = input_id[(-sz):]
                if not (tail == should_be).all().item():
                    print(
                        f"Slot {i}: inputs_ids, wrong end: {tail} (should be {should_be}; pad_id={self.pad_id}, eos_id={self._eos_id})"
                    )
        if max(right_pad) > 0:
            new_input_ids = torch.full(
                (batch_size, seq_length),
                self.pad_id,
                **kwargs,
            )
            for i, (input_id, rp) in enumerate(zip(input_ids, right_pad)):
                head = input_id[:(-rp)] if rp > 0 else input_id
                new_input_ids[i, rp:] = head
        else:
            new_input_ids = input_ids
        return dict(
            {k: v for k, v in batch.items() if k not in (INPUT_IDS_NAME, LABELS_NAME)},
            input_ids=new_input_ids,
            targets=labels.unsqueeze(-1),
        )


class BatchTransformFactory:
    @staticmethod
    def from_head_model(
        head_model: str,
        pad_id: int = 0,
        eos_id: Optional[int] = None,
        ignore_index: int = -100,
    ) -> BatchTransform:
        if head_model not in SUPPORTED_HEAD_MODELS:
            raise ValueError(
                f"head_model={head_model} not supported, choose one of {SUPPORTED_HEAD_MODELS}"
            )
        if head_model == CrossEntropyOnLogits.NAME:
            return SFTBatchTransform(
                ignore_index=ignore_index,
                pad_id=pad_id,
                eos_id=eos_id,
            )
        else:
            return SequenceClassificationBatchTransform(
                pad_id=pad_id,
                eos_id=eos_id,
            )
