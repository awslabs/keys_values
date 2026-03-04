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
from typing import Optional, Dict, Any, Tuple

from tqdm import tqdm

from keys_values.data.dataloader import MyDataLoader
from keys_values.data.load_helmet_dev_eval import (
    load_helmet_dev_eval,
    DATASET_PARENT_DIR,
)
from keys_values.data.module import (
    SequenceLengthFilteredDataModule,
    RawDatasetType,
    NUM_TOKENS_NAME,
)
from keys_values.data.sft_dataset import SFTDataset, get_sft_collate_fn


class Helmet(SequenceLengthFilteredDataModule):
    """Data module for HELMET benchmark datasets.

    Loads development and evaluation splits via :func:`load_helmet_dev_eval`.
    The development split is further divided into train and validation sets
    using `val_split_fraction`. The evaluation split becomes the test set.

    Each HELMET instance already has its prompt fully formatted, so no prompt
    construction or truncation is performed here.

    """

    def __init__(
        self,
        dataset_key: str,
        max_length: str = "8k",
        dataset_parent_dir: str = DATASET_PARENT_DIR,
        mask_prompt: bool = True,
        val_split_fraction: float = 0.1,
        ignore_index: int = -100,
        max_seq_length: Optional[int] = None,
        seed: int = 42,
        trainloader_longest_first: bool = False,
        trainloader_shortest_first: bool = False,
    ):
        """
        Args:
            dataset_key: Name of the HELMET dataset to load (e.g. ``"nq"``,
                ``"json_kv"``). See :func:`load_helmet_dev_eval` for all
                supported keys.
            max_length: Context-length bucket to load. One of ``"8k"``,
                ``"16k"``, ``"32k"``, ``"64k"``, ``"128k"``.
            dataset_parent_dir: Directory where HELMET data is cached on disk.
                Defaults to ``~/.cache/huggingface/helmet/data``.
            mask_prompt: Whether to mask the prompt tokens in the labels
                (with ``ignore_index``) so that loss is computed only on the
                generated answer.
            val_split_fraction: Fraction of the development split to use for
                validation. The rest is used for training.
            ignore_index: Value used to mask prompt positions in the labels.
            max_seq_length: Sequences longer than this (in tokens) are filtered
                out. Defaults to no filtering (``100000``).
            seed: Random seed for the train/val split.
            trainloader_longest_first: If ``True``, the first training batch
                contains the longest sequences (useful for early OOM detection).
            trainloader_shortest_first: If ``True``, the first training batch
                contains the shortest sequences.

        """
        super().__init__(
            mask_prompt=mask_prompt,
            val_split_fraction=val_split_fraction,
            ignore_index=ignore_index,
            max_seq_length=max_seq_length,
            seed=seed,
            trainloader_longest_first=trainloader_longest_first,
            trainloader_shortest_first=trainloader_shortest_first,
        )
        self.dataset_key = dataset_key
        self.max_length = max_length
        self.dataset_parent_dir = dataset_parent_dir
        self.metadata_dir = None

    def _get_dataset(self) -> Tuple[RawDatasetType, Optional[RawDatasetType]]:
        dev_data, eval_data = load_helmet_dev_eval(
            self.dataset_key,
            max_length=self.max_length,
            dataset_parent_dir=self.dataset_parent_dir,
        )
        print(f"\nTransforming HELMET '{self.dataset_key}' ({self.max_length}) ...")
        train_data = self._transform(dev_data, split="dev")
        test_data = self._transform(eval_data, split="eval")
        return train_data, test_data

    def _transform(self, dataset: Any, split: str) -> RawDatasetType:
        """Convert HELMET instances to the internal record format.

        Each HELMET instance ``{"input": ..., "output": ..., "query_id": ...}``
        is converted to ``{"instruction": ..., "output": ...,
        "num_tokens_instruction": <int>}``.

        """
        results: RawDatasetType = []
        for instance in tqdm(dataset, desc=f"Tokenizing {split}"):
            instruction = instance["input"]
            seq_length = self.tokenizer.encode(instruction).numel()
            if seq_length > self.max_seq_length:
                continue
            results.append(
                {
                    "instruction": instruction,
                    "output": instance["output"],
                    NUM_TOKENS_NAME: seq_length,
                }
            )
        print(
            f"Kept {len(results)} of {len(dataset)} {split} records "
            f"(<= {self.max_seq_length} tokens)"
        )
        return results

    def _create_datasets(
        self,
        train_kwargs: Dict[str, Any],
        val_kwargs: Dict[str, Any],
        test_kwargs: Optional[Dict[str, Any]],
    ) -> None:
        self.train_dataset = SFTDataset(
            **train_kwargs,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.val_dataset = SFTDataset(
            **val_kwargs,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        if test_kwargs is not None:
            self.test_dataset = SFTDataset(
                **test_kwargs,
                mask_prompt=self.mask_prompt,
                ignore_index=self.ignore_index,
            )

    def _get_collate_fn(self) -> MyDataLoader:
        return get_sft_collate_fn(ignore_index=self.ignore_index)
