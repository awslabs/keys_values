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
import json
import random
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from tqdm import tqdm

from keys_values.data.dataloader import MyDataLoader
from keys_values.data.load_helmet_dev_eval import (
    load_helmet_dev_eval,
    DATASET_PARENT_DIR,
)
from keys_values.data.module import (
    SequenceLengthFilteredDataModule,
    METADATA_SEQ_LENGTHS_KEY,
    METADATA_KEYS,
    RawDatasetType,
    NUM_TOKENS_NAME,
)
from keys_values.data.sft_dataset import SFTDataset, get_sft_collate_fn

METADATA_FNAME = "helmet_metadata.json"


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
        metadata_dir: Optional[str] = None,
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
            metadata_dir: If given, sequence lengths for every case are stored
                in a JSON metadata file in this directory so that subsequent
                calls to :meth:`_transform` can skip re-tokenisation.
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
        self.metadata_dir = metadata_dir

    def _get_dataset(self) -> Tuple[RawDatasetType, Optional[RawDatasetType]]:
        dev_data, eval_data = load_helmet_dev_eval(
            self.dataset_key,
            max_length=self.max_length,
            dataset_parent_dir=self.dataset_parent_dir,
        )
        print(f"\nTransforming HELMET '{self.dataset_key}' ({self.max_length}) ...")
        metadata = self._load_metadata()
        model_name = self.tokenizer.model_name
        train_data, dev_seq_lengths, dev_needs_store = self._transform(
            dev_data, split="dev", seq_lengths=self._get_seq_lengths(metadata, "dev")
        )
        test_data, eval_seq_lengths, eval_needs_store = self._transform(
            eval_data, split="eval", seq_lengths=self._get_seq_lengths(metadata, "eval")
        )
        if dev_needs_store or eval_needs_store:
            if metadata is None or METADATA_SEQ_LENGTHS_KEY not in metadata:
                metadata = {METADATA_SEQ_LENGTHS_KEY: {}}
            if self.dataset_key not in metadata[METADATA_SEQ_LENGTHS_KEY]:
                metadata[METADATA_SEQ_LENGTHS_KEY][self.dataset_key] = {}
            if self.max_length not in metadata[METADATA_SEQ_LENGTHS_KEY][self.dataset_key]:
                metadata[METADATA_SEQ_LENGTHS_KEY][self.dataset_key][self.max_length] = {}
            if model_name not in metadata[METADATA_SEQ_LENGTHS_KEY][self.dataset_key][self.max_length]:
                metadata[METADATA_SEQ_LENGTHS_KEY][self.dataset_key][self.max_length][model_name] = {}
            if dev_needs_store:
                metadata[METADATA_SEQ_LENGTHS_KEY][self.dataset_key][self.max_length][model_name][
                    "dev"
                ] = dev_seq_lengths
            if eval_needs_store:
                metadata[METADATA_SEQ_LENGTHS_KEY][self.dataset_key][self.max_length][model_name][
                    "eval"
                ] = eval_seq_lengths
            self._store_metadata(metadata)
        return train_data, test_data

    def _transform(
        self,
        dataset: Any,
        split: str,
        seq_lengths: Optional[List[int]],
    ) -> Tuple[RawDatasetType, List[int], bool]:
        """Convert HELMET instances to the internal record format.

        Each HELMET instance ``{"input": ..., "output": ..., "query_id": ...}``
        is converted to ``{"instruction": ..., "output": ...,
        "num_tokens_instruction": <int>}``.

        If ``seq_lengths`` is ``None``, sequence lengths are computed by
        tokenising every instance and the results are returned so the caller
        can persist them. When ``seq_lengths`` is provided the tokenisation
        step is skipped entirely.

        Returns:
            A tuple ``(results, seq_lengths, needs_store)`` where
            ``needs_store`` is ``True`` when ``seq_lengths`` had to be
            recomputed and should be written to disk by the caller.

        """
        needs_store = seq_lengths is None and self.metadata_dir is not None
        if seq_lengths is not None and len(seq_lengths) != len(dataset):
            print(
                f"Cached seq_lengths length mismatch for split '{split}' "
                f"({len(seq_lengths)} vs {len(dataset)}); recomputing."
            )
            seq_lengths = None
            needs_store = self.metadata_dir is not None
        if needs_store:
            print(
                f"\nTokenizing HELMET '{self.dataset_key}' ({self.max_length}) split "
                f"'{split}'. Sequence lengths will be stored in {self.metadata_dir} "
                "so next time this split runs fast."
            )
        data_iter = (
            tqdm(dataset, desc=f"Tokenizing {split}")
            if seq_lengths is None
            else dataset
        )
        _list_output_datasets = {
            "nq", "trivia_qa", "hotpot_qa", "pop_qa",
            "narrative_qa", "ruler_mk_needle", "ruler_mk_uuid",
        }
        results: RawDatasetType = []
        new_seq_lengths: List[int] = []
        for idx, instance in enumerate(data_iter):
            instruction = instance["input"]
            if seq_lengths is None:
                seq_length = self.tokenizer.encode(instruction).numel()
                new_seq_lengths.append(seq_length)
            else:
                seq_length = seq_lengths[idx]
            if seq_length > self.max_seq_length:
                continue
            if self.dataset_key in _list_output_datasets:
                seed = idx * 10086 % 1024
                random.seed(seed)
                output = str(random.choice(instance["output"]))
            else:
                output = instance["output"]
            results.append(
                {
                    "instruction": instruction,
                    "output": output,
                    NUM_TOKENS_NAME: seq_length,
                }
            )
        final_seq_lengths = new_seq_lengths if seq_lengths is None else seq_lengths
        print(
            f"Kept {len(results)} of {len(dataset)} {split} records "
            f"(<= {self.max_seq_length} tokens)"
        )
        return results, final_seq_lengths, needs_store

    def _get_seq_lengths(
        self, metadata: Optional[Dict[str, Any]], split: str
    ) -> Optional[List[int]]:
        if metadata is None:
            return None
        result = metadata.get(METADATA_SEQ_LENGTHS_KEY)
        if result is not None:
            result = result.get(self.dataset_key)
        if result is not None:
            result = result.get(self.max_length)
        if result is not None:
            result = result.get(self.tokenizer.model_name)
        if result is not None:
            result = result.get(split)
        return result

    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        if self.metadata_dir is None:
            return None
        meta_path = Path(self.metadata_dir) / METADATA_FNAME
        if not meta_path.exists():
            return None
        with meta_path.open("r") as fp:
            data = json.load(fp)
        if not METADATA_KEYS.issubset(data.keys()):
            print(
                f"Metadata loaded from {meta_path} does not contain all keys "
                f"{METADATA_KEYS}:\n{data}"
            )
            return None
        return data

    def _store_metadata(self, data: Dict[str, Any]) -> None:
        if self.metadata_dir is not None:
            meta_path = Path(self.metadata_dir) / METADATA_FNAME
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with meta_path.open("w") as fp:
                json.dump(data, fp)
            print(f"Metadata stored in {meta_path}")

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
