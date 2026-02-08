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
import os
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from litgpt.data import DataModule
from litgpt.prompts import Default
from litgpt.tokenizer import Tokenizer

from keys_values.data.base import pad_dataset, ReorderWrapperDataset
from keys_values.data.evaluation import (
    EvaluationWithTasksDataset,
    get_wrapped_collate_fn,
)
from keys_values.data.iterators import SimilarSequenceLengthIterable
from keys_values.data.sequence_classification import (
    SequenceClassificationDataset,
    get_seq_class_collate_fn,
)
from keys_values.data.sft_dataset import (
    SFTDataset,
    get_sft_collate_fn,
)
from keys_values.head_model import (
    CrossEntropyOnLogits,
    SequenceClassificationOnLogits,
    SequenceClassification,
)


METADATA_FNAME = "longbench_v2_metadata.json"

METADATA_SEQ_LENGTHS_KEY = "sequence_lengths"

METADATA_KEYS = {METADATA_SEQ_LENGTHS_KEY}

SUPPORTED_HEAD_MODELS = (
    CrossEntropyOnLogits.NAME,
    SequenceClassificationOnLogits.NAME,
    SequenceClassification.NAME,
)

CLASS_LABELS = ("A", "B", "C", "D")

SUPPORTED_TEST_SET_TAGS = [
    "rest",
]


RawDatasetType = List[Dict[str, str]]


class LongBenchV2(DataModule):
    """LongBench-V2 data module for supervised finetuning.

    Depending on `head_model`, the dataset is treated as next token prediction
    or sequence classification.
    The dataset is filtered to contain only sequences whose prompt have
    `<= max_seq_length` tokens.

    If `metadata_dir` is given, a metadata file is loaded and/or stored. This
    is strongly recommended to save time. A dictionary is stored as JSON, with
    keys:
    - `METADATA_SEQ_LENGTHS_KEY`: List of sequence lengths (in tokens) for
      each record

    For the :meth:`train_dataloader` and :meth:`val_dataloader` iterators, we
    use a :class:`SimilarSequenceLengthIterable` sampler, which tries to form
    micro and macro batches with sequences of similar length.

    """

    def __init__(
        self,
        mask_prompt: bool = True,
        val_split_fraction: float = 0.1,
        ignore_index: int = -100,
        max_seq_length: Optional[int] = None,
        seed: int = 42,
        num_workers: int = 4,
        pin_memory: bool = False,
        include_multiturn_conversations: bool = False,
        repo_id: str = "THUDM/LongBench-v2",
        access_token: Optional[str] = None,
        metadata_dir: Optional[str] = None,
        debug_num_cases: Optional[int] = None,
        trainloader_longest_first: bool = False,
        trainloader_shortest_first: bool = False,
        test_set_tag: Optional[str] = None,
    ):
        """
        Args:
            mask_prompt: Whether to mask the prompt section from the label
                (with ``ignore_index``)
            val_split_fraction: The fraction of the dataset to use for the
                validation dataset. The rest is used for training.
            ignore_index: The index to use for elements to be ignored in the
                label.
            max_seq_length: Sequences longer than this number of tokens are
                filtered out. Defaults to 100000.
            seed: The random seed for creating the train/val splits and shuffling
                the dataset.
            num_workers: Option of `torch.utils.data.DataLoader`. How many
                processes to use for loading.
            pin_memory: Option of `torch.utils.data.DataLoader`. If `True`, data
                loading may be faster.
            include_multiturn_conversations: Whether to include multi-turn
                conversations in the dataset.
            repo_id: The Hugging Face dataset repository ID from where to
                download the data.
            access_token: The Hugging Face API token to use for authentication.
                Default is using the `HF_TOKEN` environment variable.
            metadata_dir: If given, we load store/load metadata from this
                directory. Strongly recommended to save time.
            debug_num_cases: If used, we only keep this number of records.
            trainloader_longest_first: If set, :meth:`train_dataloader` returns
                a data loader whose first batch contain the longest sequences in
                the dataset, otherwise uses :class:`SimilarSequenceLengthIterable`.
                This is useful to detect OOM errors early, since they are most
                likely to happen with the longest batch.
            trainloader_shortest_first: Same as `trainloader_longest_first`,
                but the first batch contain the shortest sequences.
            test_set_tag: If this is given, we also maintain a test dataset
                and serve a test dataloader. The tag determines how the test
                set is chosen. Current choices:
                - "rest": All cases with sequence length > `max_seq_length`,
                    sorted by token sequence length (non-decreasing).

        """
        if test_set_tag is not None and test_set_tag not in SUPPORTED_TEST_SET_TAGS:
            raise ValueError(
                f"test_set_tag = {test_set_tag} is not supported, must be None or in {SUPPORTED_TEST_SET_TAGS}"
            )
        if trainloader_longest_first and trainloader_shortest_first:
            raise ValueError(
                "Cannot use both trainloader_longest_first and trainloader_shortest_first"
            )
        self.mask_prompt = mask_prompt
        self.val_split_fraction = val_split_fraction
        self.ignore_index = ignore_index
        self.max_seq_length = 100000 if max_seq_length is None else max_seq_length
        self.seed = seed
        self._dataloader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        self.include_multiturn_conversations = include_multiturn_conversations
        self.repo_id = repo_id
        self.access_token = (
            os.getenv("HF_TOKEN") if access_token is None else access_token
        )
        self.metadata_dir = metadata_dir
        self.head_model = None
        self._is_sequence_classification = None
        self.prompt_style = Default()
        self.debug_num_cases = debug_num_cases
        self.tokenizer = None
        self.batch_size = None
        self.num_devices = None
        self.val_batch_size = None
        self.test_batch_size = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._trainloader_longest_first = trainloader_longest_first
        self._trainloader_shortest_first = trainloader_shortest_first
        self.test_set_tag = test_set_tag
        self._test_eval_tasks = None
        # Maintain sequence lengths (in tokens) for cases in training set.
        # This is used to support specialized data loaders.
        self._sequence_lengths = None

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        num_devices: int = 1,
        max_seq_length: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Specific arguments:
        - `head_model`: Head model name. Mandatory
        - `val_batch_size`: Batch size for :meth:`val_dataloader`.
            Defaults to `batch_size`.
        - `test_batch_size`: Batch size for :meth:`test_dataloader`.
            Only if `test_set_tag` is given. Defaults to `val_batch_size`.
        - `eval_tasks`: Only if `test_set_tag` is given. See
            :meth:`test_dataloader`.

        Args:
            tokenizer: Tokenizer
            batch_size: Batch size for :meth:`train_dataloader`
            num_devices: Number of GPU devices used for distributed
                data parallel training
            max_seq_length: Cutoff for sequence length
            **kwargs: See above

        """
        if tokenizer is None:
            raise ValueError("tokenizer must be provided")
        head_model = kwargs.get("head_model")
        if head_model is None:
            raise ValueError("head_model must be provided")
        if not head_model in SUPPORTED_HEAD_MODELS:
            raise ValueError(
                f"head_model '{head_model}' not supported, must be in {SUPPORTED_HEAD_MODELS}"
            )
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_devices = num_devices
        if max_seq_length is not None:
            self.max_seq_length = max_seq_length
        self.val_batch_size = kwargs.get("val_batch_size")
        if self.val_batch_size is None:
            self.val_batch_size = self.batch_size
        self.test_batch_size = None
        self._test_eval_tasks = None
        if self.test_set_tag is not None:
            self.test_batch_size = kwargs.get("test_batch_size")
            if self.test_batch_size is None:
                self.test_batch_size = self.val_batch_size
            self._test_eval_tasks = kwargs.get("eval_tasks")
        self.head_model = head_model
        self._is_sequence_classification = head_model != CrossEntropyOnLogits.NAME

    def _get_dataset(self) -> Tuple[RawDatasetType, Optional[RawDatasetType]]:
        from datasets import load_dataset

        dataset = load_dataset(self.repo_id, token=self.access_token)
        return self._filter_and_transform(dataset["train"])

    def prepare_data(self) -> None:
        # Make sure that if dataset has not yet been downloaded, the hard
        # work is done once only.
        self._get_dataset()

    def setup(self, stage: str = "") -> None:
        """
        Note: Datasets `train_dataset`, `val_dataset`, `test_dataset` are
        padded so that their size becomes a multiple of
        `batch_size * num_devices`, where `batch_size` is the respective
        micro-batch size. Such padding entries are filtered out by the
        collators. They are needed so that samplers can do their job
        properly.

        """
        data, test_data = self._get_dataset()
        # Partition the dataset into train and test
        train_data, val_data = random_split(
            data,
            [1.0 - self.val_split_fraction, self.val_split_fraction],
            generator=torch.Generator().manual_seed(self.seed),
        )
        train_data, val_data = list(train_data), list(val_data)
        self._sequence_lengths = {
            "train": [record["num_tokens_instruction"] for record in train_data],
            "valid": [record["num_tokens_instruction"] for record in val_data],
        }
        if test_data is not None:
            self._sequence_lengths["test"] = [
                record["num_tokens_instruction"] for record in test_data
            ]

        if not self._is_sequence_classification:
            self.train_dataset = SFTDataset(
                data=pad_dataset(
                    train_data,
                    batch_size=self.batch_size,
                    num_devices=self.num_devices,
                ),
                tokenizer=self.tokenizer,
                prompt_style=Default(),
                max_seq_length=self.max_seq_length,
                mask_prompt=self.mask_prompt,
                ignore_index=self.ignore_index,
            )
            self.val_dataset = SFTDataset(
                data=pad_dataset(
                    val_data,
                    batch_size=self.batch_size,
                    num_devices=self.num_devices,
                ),
                tokenizer=self.tokenizer,
                prompt_style=Default(),
                max_seq_length=self.max_seq_length,
                mask_prompt=self.mask_prompt,
                ignore_index=self.ignore_index,
            )
            if test_data is not None:
                self.test_dataset = SFTDataset(
                    data=pad_dataset(
                        test_data,
                        batch_size=self.batch_size,
                        num_devices=self.num_devices,
                    ),
                    tokenizer=self.tokenizer,
                    prompt_style=Default(),
                    max_seq_length=-1,
                    mask_prompt=self.mask_prompt,
                    ignore_index=self.ignore_index,
                )
        else:
            self.train_dataset = SequenceClassificationDataset(
                data=pad_dataset(
                    train_data,
                    batch_size=self.batch_size,
                    num_devices=self.num_devices,
                ),
                tokenizer=self.tokenizer,
                prompt_style=Default(),
                class_labels=CLASS_LABELS,
                max_seq_length=self.max_seq_length,
            )
            self.val_dataset = SequenceClassificationDataset(
                data=pad_dataset(
                    val_data,
                    batch_size=self.batch_size,
                    num_devices=self.num_devices,
                ),
                tokenizer=self.tokenizer,
                prompt_style=Default(),
                class_labels=CLASS_LABELS,
                max_seq_length=self.max_seq_length,
            )
            if test_data is not None:
                self.test_dataset = SequenceClassificationDataset(
                    data=pad_dataset(
                        test_data,
                        batch_size=self.batch_size,
                        num_devices=self.num_devices,
                    ),
                    tokenizer=self.tokenizer,
                    prompt_style=Default(),
                    class_labels=CLASS_LABELS,
                    max_seq_length=-1,
                )

    def _get_collate_fn(self):
        if not self._is_sequence_classification:
            return get_sft_collate_fn(ignore_index=self.ignore_index)
        else:
            return get_seq_class_collate_fn()

    def train_dataloader(self) -> DataLoader:
        assert self._sequence_lengths is not None
        len_sl = len(self._sequence_lengths["train"])
        assert 0 < len_sl <= len(self.train_dataset), (len_sl, len(self.train_dataset))
        print(f"train_dataloader: len_sl = {len_sl}, batch_size = {self.batch_size}, num_devices = {self.num_devices}")  # DEBUG
        torch.random.manual_seed(self.seed)
        return DataLoader(
            ReorderWrapperDataset(
                self.train_dataset,
                num_devices=self.num_devices,
                batch_size=self.batch_size,
            ),
            batch_size=self.batch_size,
            sampler=SimilarSequenceLengthIterable(
                sequence_lengths=self._sequence_lengths["train"],
                micro_batch_size=self.batch_size,
                num_devices=self.num_devices,
                shuffle=True,
                longest_first=self._trainloader_longest_first,
                shortest_first=self._trainloader_shortest_first,
            ),
            collate_fn=self._get_collate_fn(),
            **self._dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._sequence_lengths is not None
        len_sl = len(self._sequence_lengths["valid"])
        assert 0 < len_sl <= len(self.val_dataset), (len_sl, len(self.val_dataset))
        print(f"val_dataloader: len_sl = {len_sl}, batch_size = {self.val_batch_size}, num_devices = {self.num_devices}")  # DEBUG
        return DataLoader(
            ReorderWrapperDataset(
                self.val_dataset,
                num_devices=self.num_devices,
                batch_size=self.val_batch_size,
            ),
            batch_size=self.val_batch_size,
            sampler=SimilarSequenceLengthIterable(
                sequence_lengths=self._sequence_lengths["valid"],
                micro_batch_size=self.val_batch_size,
                num_devices=self.num_devices,
                shuffle=False,
            ),
            collate_fn=self._get_collate_fn(),
            **self._dataloader_kwargs,
        )

    def test_dataloader(self, num_devices: int = 1) -> DataLoader:
        if self.test_dataset is None:
            raise IndexError("Test dataset is not defined. Use 'test_set_tag'")
        assert self._sequence_lengths is not None
        len_sl = len(self._sequence_lengths["test"])
        assert 0 < len_sl <= len(self.test_dataset), (len_sl, len(self.test_dataset))
        if self._test_eval_tasks is None:
            return DataLoader(
                ReorderWrapperDataset(
                    self.test_dataset,
                    num_devices=num_devices,
                    batch_size=self.test_batch_size,
                ),
                batch_size=self.test_batch_size,
                sampler=SimilarSequenceLengthIterable(
                    sequence_lengths=self._sequence_lengths["test"],
                    micro_batch_size=self.test_batch_size,
                    num_devices=self.num_devices,
                    shuffle=False,
                ),
                collate_fn=self._get_collate_fn(),
                **self._dataloader_kwargs,
            )
        else:
            # TODO: Use _sequence_lengths["test"]
            # Cross product between test dataset and evaluation tasks (these
            # are typically different model checkpoints)
            collate_fn = get_wrapped_collate_fn(self._get_collate_fn())
            _test_dataset = EvaluationWithTasksDataset(
                dataset=self.test_dataset,
                tasks=self._test_eval_tasks,
                batch_size=self.test_batch_size,
                num_devices=num_devices,
            )
            return DataLoader(
                ReorderWrapperDataset(
                    _test_dataset,
                    num_devices=num_devices,
                    batch_size=self.test_batch_size,
                ),
                batch_size=self.test_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                **self._dataloader_kwargs,
            )

    def head_model_kwargs(self, name: str) -> Dict[str, Any]:
        result = dict()
        if name == CrossEntropyOnLogits.NAME:
            result["ignore_index"] = self.ignore_index
        elif name == SequenceClassificationOnLogits.NAME:
            if self.tokenizer is None:
                raise IndexError("Need `connect` to be called first")
            class_label_tokens = []
            for label in CLASS_LABELS:
                token_idx = self.tokenizer.encode(label)
                if len(token_idx) != 1:
                    raise ValueError(
                        f"Class label {label} maps to tokens {token_idx}, but must map to a single token"
                    )
                class_label_tokens.append(int(token_idx.item()))
            result["class_label_tokens"] = class_label_tokens
            print(f"Class label tokens: {CLASS_LABELS} -> {class_label_tokens}")
        elif name == SequenceClassification.NAME:
            result["num_classes"] = len(CLASS_LABELS)
        return result

    def _filter_and_transform(
        self,
        dataset: Any,
    ) -> Tuple[RawDatasetType, Optional[RawDatasetType]]:
        metadata = self._load_metadata(len(dataset))
        seq_lengths = None if metadata is None else self._get_seq_lengths(metadata)
        try_to_store = seq_lengths is None and self.metadata_dir is not None
        model_name = self.tokenizer.model_name
        # If `seq_lengths` could not be loaded, it is recomputed and stored below.
        # This takes more time.
        if try_to_store and self.metadata_dir is not None:
            print(
                "\nFiltering the dataset takes a while. I'll store the index in "
                f"{self.metadata_dir} under key '{model_name}', so next time "
                "this won't have to be done (if you use the same dataset and model)."
            )
        transformed_data, seq_lengths, test_data = filter_and_transform(
            dataset=dataset,
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            seq_lengths=seq_lengths,
            head_model=self.head_model,
            test_set_tag=self.test_set_tag,
            debug_num_cases=self.debug_num_cases,
        )
        if try_to_store:
            if metadata is None or METADATA_SEQ_LENGTHS_KEY not in metadata:
                metadata = {
                    METADATA_SEQ_LENGTHS_KEY: {model_name: seq_lengths},
                }
            else:
                metadata[METADATA_SEQ_LENGTHS_KEY][model_name] = seq_lengths
            self._store_metadata(metadata)
        return transformed_data, test_data

    def _get_seq_lengths(self, metadata: Dict[str, Any]) -> Optional[List[int]]:
        result = metadata.get(METADATA_SEQ_LENGTHS_KEY)
        if result is not None:
            result = result.get(self.tokenizer.model_name)
        return result

    def _load_metadata(self, num_records: int) -> Optional[Dict[str, Any]]:
        if self.metadata_dir is None:
            return None
        meta_path = Path(self.metadata_dir) / METADATA_FNAME
        if not meta_path.exists():
            return None
        with meta_path.open("r") as fp:
            data = json.load(fp)
        if not METADATA_KEYS.issubset(data.keys()):
            print(
                f"Metadata loaded from {meta_path} does not contain all keys {METADATA_KEYS}:\n{data}"
            )
            return None
        seq_lenghts = self._get_seq_lengths(data)
        if seq_lenghts is None:
            return data
        prefix = f"data['{METADATA_SEQ_LENGTHS_KEY}']['{self.tokenizer.model_name}'] = {seq_lenghts}"
        if not isinstance(seq_lenghts, list) or len(seq_lenghts) != num_records:
            print(prefix + f", must be list of length {num_records}")
            return None
        if any(int(x) != x or x <= 0 for x in seq_lenghts):
            print(prefix + ", must contain positive integers")
            return None
        return data

    def _store_metadata(self, data: Dict[str, Any]) -> None:
        if self.metadata_dir is not None:
            meta_path = Path(self.metadata_dir) / METADATA_FNAME
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with meta_path.open("w") as fp:
                json.dump(data, fp)
            print(f"Metadata stored in {meta_path}")


PROMPTLINES_PREFIX = [
    "Please read the following text and answer the question below.",
    "",
    "<text>",
]

PROMPTLINES_POSTFIX = [
    "</text>",
    "",
    "What is the correct answer to this question: {question}",
    "Choices:",
    "A: {choice_A}",
    "B: {choice_B}",
    "C: {choice_C}",
    "D: {choice_D}",
    "",
]

PROMPTLINES_FINAL = {
    CrossEntropyOnLogits.NAME: [
        'Format your response as follows: "The correct answer is (insert answer here)".',
        "",
        "Answer:",
        "The correct answer is ",
    ],
    SequenceClassificationOnLogits.NAME: [
        "The correct answer is ",
    ],
    SequenceClassification.NAME: [
        "The correct answer is ",
    ],
}


def filter_and_transform(
    dataset: Any,
    max_seq_length: int,
    tokenizer: Tokenizer,
    seq_lengths: Optional[List[int]],
    head_model: str,
    test_set_tag: Optional[str],
    debug_num_cases: Optional[int] = None,
) -> Tuple[RawDatasetType, List[int], Optional[RawDatasetType]]:
    # From https://huggingface.co/datasets/THUDM/LongBench-v2:
    # {
    #    "_id": "Unique identifier for each piece of data",
    #    "domain": "The primary domain category of the data",
    #    "sub_domain": "The specific sub-domain category within the domain",
    #    "difficulty": "The difficulty level of the task, either 'easy' or 'hard'",
    #    "length": "The length category of the task, which can be 'short', 'medium', or 'long'",
    #    "question": "The input/command for the task, usually short, such as questions in QA, queries in many-shot learning, etc",
    #    "choice_A": "Option A", "choice_B": "Option B", "choice_C": "Option C", "choice_D": "Option D",
    #    "answer": "The groundtruth answer, denoted as A, B, C, or D",
    #    "context": "The long context required for the task, such as documents, books, code repositories, etc."
    # }
    #
    # Prompt is from:
    # https://github.com/THUDM/LongBench/blob/main/prompts/0shot.txt
    train_results: RawDatasetType = []
    test_results: RawDatasetType = []
    num_used = 0
    num_total = 0
    print(f"\nProcessing dataset, filtering out records with > {max_seq_length} tokens")
    if seq_lengths is None:
        # Show progress bar: This takes a while
        data_iter = tqdm(dataset)
    else:
        if len(seq_lengths) != len(dataset):
            raise ValueError(
                f"len(seq_lengths) = {len(seq_lengths)} != {len(dataset)} = len(dataset)"
            )
        data_iter = dataset
    new_seq_lengths = []
    for idx, entry in enumerate(data_iter):
        num_total += 1
        values = {
            k: entry[k]
            for k in ("question", "choice_A", "choice_B", "choice_C", "choice_D")
        }
        instruction_list = (
            PROMPTLINES_PREFIX
            + [entry["context"]]
            + [line.format(**values) for line in PROMPTLINES_POSTFIX]
            + PROMPTLINES_FINAL[head_model]
        )
        instruction = "\n".join(instruction_list)
        output = entry["answer"]
        if seq_lengths is None:
            encoded_prompt = tokenizer.encode(instruction)
            seq_length = encoded_prompt.numel()
            new_seq_lengths.append(seq_length)
        else:
            seq_length = seq_lengths[idx]
        if seq_length <= max_seq_length:
            num_used += 1
            train_results.append(
                {
                    "instruction": instruction,
                    "output": output,
                    "num_tokens_instruction": seq_length,
                }
            )
            if debug_num_cases is not None and num_used >= debug_num_cases:
                print(f"DEBUG: Stop with {num_used} records.")
                break
        elif test_set_tag == "rest":
            test_results.append(
                {
                    "instruction": instruction,
                    "output": output,
                    "num_tokens_instruction": seq_length,
                }
            )
    print(f"\nKept {num_used} of {num_total} records: {max_seq_length} tokens or less")
    if test_set_tag == "rest":
        # Sort by increasing length
        test_results = sorted(
            test_results,
            key=lambda x: x["num_tokens_instruction"],
        )
        min_length = test_results[0]["num_tokens_instruction"]
        max_length = test_results[-1]["num_tokens_instruction"]
        print(
            f"Test dataset has {len(test_results)} records, token lengths between {min_length} and {max_length}"
        )
    else:
        test_results = None
    if seq_lengths is None:
        seq_lengths = new_seq_lengths
    return train_results, seq_lengths, test_results


METADATA_TRUNCATION_LENGHTS_KEY = "truncation_lengths"


class LongBenchV2Truncated(LongBenchV2):
    """
    Truncated version of :class:`LongBenchV2`, used for baseline comparisons.

    Here, `metadata_dir` must be given, and a metadata file must exist. We add
    the following keys:
    - `METADATA_TRUNCATION_LENGHTS_KEY`: This is a `Dict[int, Dict[int, int]]`
      which maps values of `truncation_context_width` to dictionaries mapping
      case index `idx` to truncation lengths, so that prompts have at most
      `truncation_context_width` tokens. See
      :func:`truncate_contexts_and_transform`.

    """

    def __init__(
        self,
        metadata_dir: str,
        mask_prompt: bool = True,
        val_split_fraction: float = 0.1,
        ignore_index: int = -100,
        max_seq_length: Optional[int] = None,
        seed: int = 42,
        num_workers: int = 4,
        pin_memory: bool = False,
        include_multiturn_conversations: bool = False,
        repo_id: str = "THUDM/LongBench-v2",
        access_token: Optional[str] = None,
        debug_num_cases: Optional[int] = None,
    ):
        if metadata_dir is None:
            raise ValueError(
                "metadata_dir must be given, and the metadata file must exist. "
                "Use `LongBenchV2` first to create it."
            )
        super().__init__(
            mask_prompt=mask_prompt,
            val_split_fraction=val_split_fraction,
            ignore_index=ignore_index,
            max_seq_length=max_seq_length,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            include_multiturn_conversations=include_multiturn_conversations,
            repo_id=repo_id,
            access_token=access_token,
            metadata_dir=metadata_dir,
            debug_num_cases=debug_num_cases,
        )
        self.truncation_context_width = None
        self.truncation_lengths = None

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().connect(tokenizer, batch_size, max_seq_length, **kwargs)
        truncation_context_width = kwargs.get("truncation_context_width")
        if truncation_context_width is None:
            raise ValueError("truncation_context_width must be provided")
        self.truncation_context_width = truncation_context_width

    def _filter_and_transform(self, dataset: Any) -> List[Dict[str, str]]:
        metadata = self._load_metadata(len(dataset))
        model_name = self.tokenizer.model_name
        if metadata is None:
            meta_path = Path(self.metadata_dir) / METADATA_FNAME
            raise FileNotFoundError(
                f"Error trying to load metadata from {meta_path}. The metadata "
                "file must exist. Use `LongBenchV2` first to create it."
            )
        seq_lengths = metadata[METADATA_SEQ_LENGTHS_KEY][model_name]
        truncation_lengths = None
        tl_map = metadata.get(METADATA_TRUNCATION_LENGHTS_KEY)
        if tl_map is not None:
            tl_map = {
                int(k): {int(inner_k): inner_v for inner_k, inner_v in v.items()}
                for k, v in tl_map.items()
            }
            truncation_lengths = tl_map.get(self.truncation_context_width)
        try_to_store = truncation_lengths is None
        # If `truncation_lengths` could not be loaded, it is recomputed and
        # stored below. This takes more time.
        if try_to_store:
            print(
                "\nTransforming the dataset to be truncated takes a while. I'll "
                "store the truncation lengths for truncation_context_width="
                f"{self.truncation_context_width} in {self.metadata_dir}, so "
                "next time this won't have to be done (make sure to use the "
                "same --data.truncation_context_width)."
            )
        transformed_data, truncation_lengths = truncate_contexts_and_transform(
            dataset=dataset,
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            seq_lengths=seq_lengths,
            head_model=self.head_model,
            truncation_context_width=self.truncation_context_width,
            truncation_lengths=truncation_lengths,
            debug_num_cases=self.debug_num_cases,
        )
        if try_to_store:
            tl_map = metadata.get(METADATA_TRUNCATION_LENGHTS_KEY)
            if tl_map is None:
                tl_map = {self.truncation_context_width: truncation_lengths}
            else:
                tl_map[self.truncation_context_width] = truncation_lengths
            metadata[METADATA_TRUNCATION_LENGHTS_KEY] = tl_map
            self._store_metadata(metadata)
        return transformed_data


def truncate_contexts_and_transform(
    dataset: Any,
    max_seq_length: Optional[int],
    tokenizer: Tokenizer,
    seq_lengths: List[int],
    head_model: str,
    truncation_context_width: int,
    truncation_lengths: Optional[Dict[int, int]] = None,
    debug_num_cases: Optional[int] = None,
) -> Tuple[List[Dict[str, str]], Optional[Dict[int, int]]]:
    """
    Truncation makes sure that the prompt in `instruction` has at most
    `truncation_context_width` tokens when encoded. It is done by truncating the
    `context` field in the prompt, keeping the tail and removing the head if
    necessary.

    Truncation is used to run a baseline to compare against long context
    fine-tuning. This baseline trains on the truncated prompts.

    """
    results: List[Dict[str, str]] = []
    num_used = 0
    num_total = 0
    print(
        f"\nProcessing dataset, truncating prompts for baseline with context width {truncation_context_width}"
    )
    if truncation_lengths is None:
        # Show progress bar: This takes a while
        data_iter = tqdm(dataset)
        encoded_prefix = tokenizer.encode("\n".join(PROMPTLINES_PREFIX) + "\n")
    else:
        data_iter = dataset
        encoded_prefix = None
    new_truncation_lengths = dict()
    for idx, entry in enumerate(data_iter):
        num_total += 1
        if max_seq_length is None or seq_lengths[idx] <= max_seq_length:
            values = dict(
                question=entry["question"],
                choice_A=entry["choice_A"],
                choice_B=entry["choice_B"],
                choice_C=entry["choice_C"],
                choice_D=entry["choice_D"],
            )
            if truncation_lengths is not None:
                len_truncated_context = truncation_lengths[idx]
            else:
                encoded_postfix = tokenizer.encode(
                    "\n"
                    + "\n".join(
                        [line.format(**values) for line in PROMPTLINES_POSTFIX]
                        + PROMPTLINES_FINAL[head_model]
                    )
                )
                num_tokens_context = (
                    truncation_context_width
                    - len(encoded_postfix)
                    - len(encoded_prefix)
                )
                if num_tokens_context <= 0:
                    raise ValueError(
                        f"idx={idx}: truncation_context_width={truncation_context_width} is too short (len(encoded_prefix)={len(encoded_prefix)}, len(encoded_postfix)={len(encoded_postfix)})"
                    )
                encoded_context = tokenizer.encode(entry["context"])
                len_truncated_context = len(
                    tokenizer.decode(encoded_context[(-num_tokens_context):])
                )
                new_truncation_lengths[idx] = len_truncated_context
            truncated_context = entry["context"][(-len_truncated_context):]
            instruction_list = (
                PROMPTLINES_PREFIX
                + [truncated_context]
                + [line.format(**values) for line in PROMPTLINES_POSTFIX]
                + PROMPTLINES_FINAL[head_model]
            )
            instruction = "\n".join(instruction_list)
            output = entry["answer"]
            num_used += 1
            results.append({"instruction": instruction, "output": output})
            if debug_num_cases is not None and num_used >= debug_num_cases:
                print(f"DEBUG: Stop with {num_used} records.")
                break
    print(f"\nKept {num_used} of {num_total} entries.")
    return results, (
        new_truncation_lengths if truncation_lengths is None else truncation_lengths
    )
