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
from tokenizers import Tokenizer as HFTokenizer
from tqdm import tqdm

from litgpt.tokenizer import Tokenizer

from keys_values.data.constants import (
    METADATA_SEQ_LENGTHS_KEY,
    METADATA_TRAIN_VAL_SPLIT_KEY,
    METADATA_KEYS,
    RawDatasetType,
)
from keys_values.data.dataloader import MyDataLoader
from keys_values.data.module import SequenceLengthFilteredDataModule
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
from keys_values.kvcache.smart_lastrec import (
    SmartInitialInformation,
    end_initial_regex_from_string,
)
from keys_values.utils import get_dict, set_dict

METADATA_FNAME = "longbench_v2_metadata.json"

SUPPORTED_HEAD_MODELS = (
    CrossEntropyOnLogits.NAME,
    SequenceClassificationOnLogits.NAME,
    SequenceClassification.NAME,
)

CLASS_LABELS = ("A", "B", "C", "D")

SUPPORTED_TEST_SET_TAGS = (
    "stratified",
    "rest",
)

METADATA_TRAIN_VAL_TEST_SPLIT_KEY = "train_val_test_split"

LONGBENCH_NUM_CASES = 503

LONGBENCH_BUCKET_SIZES = [(20, 1, 4)] * 9 + [(22, 1, 5)] + [(20, 1, 4)] * 10


class LongBenchV2(SequenceLengthFilteredDataModule):
    """LongBench-V2 data module for supervised finetuning.

    Depending on `head_model`, the dataset is treated as next token prediction
    or sequence classification.
    The dataset is filtered to contain only sequences whose prompt have
    `<= max_seq_length` tokens.

    LongBench-V2 does not prescribe a dev/eval split. Instead, this is
    determined by `test_set_tag`:

    * "stratified": We use a stratified random split with fixed bucket sizes.
      There are 20 buckets, 19 of size 25, 1 (middle) of size 28. Each bucket
      is randomly distributed between train (80%), validation (4%), test (16%),
      according to :const:`LONGBENCH_BUCKET_SIZES`. This train/valid/test split
      is written into the metadata file.
      Note: `max_seq_length`, `val_split_fraction` are ignored in this case.
      Each split contains short and long sequences.
    * "rest": The dev set contains sequences of length <= `max_seq_length`, the
      test set contains all remaining sequences. The dev set is randomly split
      to train/valid, using `val_split_fraction`. The train/valid split is
      written into the metadata file.

    If `metadata_dir` is given, a metadata file is loaded and/or stored. This
    is strongly recommended to save time. A dictionary is stored as JSON, with
    this structure:

    * `data[METADATA_SEQ_LENGTHS_KEY][model_name]`: List of sequence lengths
      (in tokens) for each record. Here, `model_name` because the tokenizer
      depends on the model.
    * `data[METADATA_TRAIN_VAL_TEST_SPLIT_KEY][model_name]`: This is used if
      `test_set_tag == "stratified"`. Determines the train/valid/test split.
    * `data[METADATA_TRAIN_VAL_SPLIT_KEY][model_name][str(max_seq_length)][str(val_split_fraction)]`:
      This is used if `test_set_tag == "rest"`. Determines the train/valid
      split. The metadata file can store entries for different
      `(max_seq_length, val_split_fraction)` pairs.

    """

    def __init__(
        self,
        mask_prompt: bool = True,
        val_split_fraction: float = 0.1,
        ignore_index: int = -100,
        max_seq_length: Optional[int] = 100000,
        seed: int = 42,
        repo_id: str = "THUDM/LongBench-v2",
        access_token: Optional[str] = None,
        metadata_dir: Optional[str] = None,
        store_split_in_metadata: bool = False,
        debug_num_cases: Optional[int] = None,
        trainloader_longest_first: bool = False,
        trainloader_shortest_first: bool = False,
        test_set_tag: Optional[str] = None,
        recompute_lengths: bool = False,
    ):
        """
        Args:
            mask_prompt: Whether to mask the prompt section from the label
                (with ``ignore_index``)
            val_split_fraction: The fraction of the dataset to use for the
                validation dataset. The rest is used for training.
                Note: This is ignored if `test_set_tag == "stratified"`
            ignore_index: The index to use for elements to be ignored in the
                label.
            max_seq_length: Sequences longer than this number of tokens are
                filtered out. Defaults to 100000.
                Note: This is ignored if `test_set_tag == "stratified"`
            seed: The random seed for creating the train/val splits and shuffling
                the dataset.
            repo_id: The Hugging Face dataset repository ID from where to
                download the data.
            access_token: The Hugging Face API token to use for authentication.
                Default is using the `HF_TOKEN` environment variable.
            metadata_dir: If given, we load store/load metadata from this
                directory. Strongly recommended to save time.
            store_split_in_metadata: If `True`, we sample and store the
                train/val split in the metadata and also load it from there,
                instead of sampling it anew each time.
            debug_num_cases: If used, we only keep this number of records.
            trainloader_longest_first: If set, :meth:`train_dataloader` returns
                a data loader whose first batch contain the longest sequences in
                the dataset, otherwise uses :class:`SimilarSequenceLengthIterable`.
                This is useful to detect OOM errors early, since they are most
                likely to happen with the longest batch.
            trainloader_shortest_first: Same as `trainloader_longest_first`,
                but the first batch contain the shortest sequences.
            test_set_tag: Determines how the complete dataset is split into
                train, validation, and test sets. Defaults to "stratified".
                Current choices:
                * "stratified": Stratified random split with fixed bucket sizes.
                  `val_split_fraction`, `max_seq_length` are ignored.
                * "rest": Test set is all cases with sequence length >
                  `max_seq_length`, sorted by token sequence length
                  (non-decreasing).
            recompute_lengths: If `True`, sequence lengths are recomputed even
                if they are in the metadata file. The previous information is
                overwritten.

        """
        if test_set_tag is None:
            test_set_tag = "stratified"
        elif test_set_tag not in SUPPORTED_TEST_SET_TAGS:
            raise ValueError(
                f"test_set_tag = {test_set_tag} is not supported, must be in {SUPPORTED_TEST_SET_TAGS}"
            )
        super().__init__(
            mask_prompt,
            val_split_fraction,
            ignore_index,
            max_seq_length,
            seed,
            trainloader_longest_first,
            trainloader_shortest_first,
        )
        self.repo_id = repo_id
        self.access_token = (
            os.getenv("HF_TOKEN") if access_token is None else access_token
        )
        self.metadata_dir = metadata_dir
        self.store_split_in_metadata = store_split_in_metadata
        self.head_model = None
        self._is_sequence_classification = None
        self.debug_num_cases = debug_num_cases
        self.test_set_tag = test_set_tag
        self._recompute_lengths = recompute_lengths
        self._stratified_split: Optional[Dict[str, List[int]]] = None
        self._split_from_metadata: Optional[Dict[str, List[int]]] = None

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        num_devices: int = 1,
        rank: Optional[int] = None,
        max_seq_length: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Extra specific arguments:
        - `head_model`: Head model name. Mandatory

        Args:
            tokenizer: Tokenizer
            batch_size: Batch size for :meth:`train_dataloader`
            num_devices: Number of GPU devices used for distributed data
                parallel training
            rank: Rank (only if `num_devices > 1`)
            max_seq_length: Cutoff for sequence length
            **kwargs: See above

        """
        super().connect(
            tokenizer,
            batch_size,
            num_devices,
            rank,
            max_seq_length,
            **kwargs,
        )
        head_model = kwargs.get("head_model")
        if head_model is None:
            raise ValueError("head_model must be provided")
        if not head_model in SUPPORTED_HEAD_MODELS:
            raise ValueError(
                f"head_model '{head_model}' not supported, must be in {SUPPORTED_HEAD_MODELS}"
            )
        self.head_model = head_model
        self._is_sequence_classification = head_model != CrossEntropyOnLogits.NAME

    def _get_dataset(self) -> Tuple[RawDatasetType, Optional[RawDatasetType]]:
        from datasets import load_dataset

        dataset = load_dataset(self.repo_id, token=self.access_token)
        return self._filter_and_transform(dataset["train"])

    def _create_datasets(
        self,
        train_kwargs: Dict[str, Any],
        val_kwargs: Dict[str, Any],
        test_kwargs: Optional[Dict[str, Any]],
    ):
        if not self._is_sequence_classification:
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
        else:
            self.train_dataset = SequenceClassificationDataset(
                **train_kwargs,
                class_labels=CLASS_LABELS,
            )
            self.val_dataset = SequenceClassificationDataset(
                **val_kwargs,
                class_labels=CLASS_LABELS,
            )
            if test_kwargs is not None:
                self.test_dataset = SequenceClassificationDataset(
                    **test_kwargs,
                    class_labels=CLASS_LABELS,
                )

    def _get_collate_fn(self) -> MyDataLoader:
        if not self._is_sequence_classification:
            return get_sft_collate_fn(ignore_index=self.ignore_index)
        else:
            return get_seq_class_collate_fn()

    def head_model_kwargs(
        self,
        name: str,
        tokenizer: Optional[Tokenizer] = None,
    ) -> Dict[str, Any]:
        result = dict()
        if name == CrossEntropyOnLogits.NAME:
            result["ignore_index"] = self.ignore_index
        elif name == SequenceClassificationOnLogits.NAME:
            if tokenizer is None:
                tokenizer = self.tokenizer
            if tokenizer is None:
                raise IndexError(
                    "Need `tokenizer` to be passed, or `connect` to be called first"
                )
            class_label_tokens = []
            for label in CLASS_LABELS:
                token_idx = tokenizer.encode(label)
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
        seq_lengths = self._get_seq_lengths(metadata)
        lens_need_store = seq_lengths is None
        # If `seq_lengths` could not be loaded, it is recomputed and stored below.
        # This takes more time.
        if lens_need_store and self.metadata_dir is not None:
            print(
                "\nFiltering the dataset takes a while. I'll store the index in "
                f"{self.metadata_dir} under key '{self.model_name}', so next time "
                "this won't have to be done (if you use the same dataset and model)."
            )
        max_seq_length = (
            None if self.test_set_tag == "stratified" else self.max_seq_length
        )
        transformed_data, seq_lengths, test_data = filter_and_transform(
            dataset=dataset,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            seq_lengths=seq_lengths,
            head_model=self.head_model,
            debug_num_cases=self.debug_num_cases,
        )
        stratified_needs_store = False
        split_needs_store = False
        if self.test_set_tag == "stratified":
            # Stratified random split
            assert test_data is None  # Sanity check
            transformed_data, test_data, stratified_needs_store = (
                self._load_or_sample_stratified_split(
                    metadata=metadata,
                    seq_lengths=seq_lengths,
                    transformed_data=transformed_data,
                )
            )
        else:
            split_needs_store = self._load_or_sample_split(
                metadata,
                devset_length=len(transformed_data),
            )
        if self.metadata_dir is not None and (
            lens_need_store or stratified_needs_store or split_needs_store
        ):
            if metadata is None:
                metadata = dict()
            if lens_need_store:
                set_dict(metadata, self._metakeys_seq_lengths, seq_lengths)
            if stratified_needs_store:
                assert self._stratified_split is not None
                set_dict(
                    metadata,
                    self._metakeys_train_val_test_split,
                    self._stratified_split,
                )
            if split_needs_store:
                assert self._split_from_metadata is not None
                set_dict(
                    metadata, self._metakeys_train_val_split, self._split_from_metadata
                )
            self._store_metadata(metadata)
        return transformed_data, test_data

    @property
    def _metakeys_seq_lengths(self) -> List[str]:
        return [METADATA_SEQ_LENGTHS_KEY, self.model_name]

    @property
    def _metakeys_train_val_test_split(self) -> List[str]:
        return [METADATA_TRAIN_VAL_TEST_SPLIT_KEY, self.model_name]

    @property
    def _metakeys_train_val_split(self) -> List[str]:
        return [
            METADATA_TRAIN_VAL_SPLIT_KEY,
            self.model_name,
            str(self.max_seq_length),
            str(self.val_split_fraction),
        ]

    def _get_seq_lengths(
        self, metadata: Optional[Dict[str, Any]]
    ) -> Optional[List[int]]:
        if not self._recompute_lengths:
            return get_dict(metadata, self._metakeys_seq_lengths)
        else:
            return None

    def _load_or_sample_stratified_split(
        self,
        metadata: Optional[Dict[str, Any]],
        seq_lengths: List[int],
        transformed_data: RawDatasetType,
    ) -> Tuple[RawDatasetType, RawDatasetType, bool]:
        needs_store = False
        self._stratified_split = get_dict(metadata, self._metakeys_train_val_test_split)
        if self._stratified_split is None:
            # Sample stratified split
            self._stratified_split = sample_stratified_split(
                seq_lengths,
                self._generator,
            )
            needs_store = True
        # Split dataset
        dev_data = [
            transformed_data[idx]
            for idx in self._stratified_split["train"] + self._stratified_split["val"]
        ]
        test_data = [transformed_data[idx] for idx in self._stratified_split["test"]]
        # Split of dev -> (train, val) is simple now
        len_train = len(self._stratified_split["train"])
        len_all = len_train + len(self._stratified_split["val"])
        self._split_from_metadata = {
            "train": list(range(len_train)),
            "val": list(range(len_train, len_all)),
        }
        return dev_data, test_data, needs_store

    def _load_or_sample_split(
        self,
        metadata: Optional[Dict[str, Any]],
        devset_length: int,
    ) -> bool:
        needs_store = False
        if not self.store_split_in_metadata:
            # Splits are not stored to / loaded from metadata
            self._split_from_metadata = None
            return False
        self._split_from_metadata = get_dict(metadata, self._metakeys_train_val_split)
        if self._split_from_metadata is None:
            # Sample train/val split -> store to metadata
            dev_perm = torch.randperm(
                devset_length,
                generator=self._generator,
            )
            val_size = max(int(devset_length * self.val_split_fraction), 1)
            self._split_from_metadata = {
                "train": dev_perm[val_size:].tolist(),
                "val": dev_perm[:val_size].tolist(),
            }
            needs_store = True
        return needs_store

    def _get_train_val_split_from_metadata(
        self,
    ) -> Optional[Dict[str, List[int]]]:
        return self._split_from_metadata

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
        prefix = (
            f"data['{METADATA_SEQ_LENGTHS_KEY}']['{self.model_name}'] = {seq_lenghts}"
        )
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

    def smart_lastrec_info(self, tokenizer: HFTokenizer) -> SmartInitialInformation:
        """
        Returns:
            Information used to detect the end of the initial part supposed to
            remain in the cache for
            :class:`SmartInitialLastRecentlyInsertedKVCache`.

        """
        return SmartInitialInformation(
            end_initial_regex=end_initial_regex_from_string(
                PROMPTLINES_PREFIX[-1],
                tokenizer=tokenizer,
            ),
            max_initial_fraction=0.1,
            include_end_string=True,
        )


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


def sample_stratified_split(
    seq_lengths: List[int],
    generator: torch.Generator,
) -> Dict[str, List[int]]:
    sort_ind, _ = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1]))
    sort_ind = torch.tensor(sort_ind)
    train_ind = None
    val_ind = None
    test_ind = None
    pos = 0
    for train_sz, val_sz, test_sz in LONGBENCH_BUCKET_SIZES:
        sz = train_sz + val_sz + test_sz
        tpv_sz = train_sz + val_sz
        ind_slice = sort_ind[pos : (pos + sz)]
        rnd_ind = torch.randperm(sz, generator=generator)
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
    rnd_ind = torch.randperm(train_ind.numel(), generator=generator)
    train_ind = train_ind[rnd_ind]
    return {
        "train": train_ind.tolist(),
        "val": val_ind.tolist(),
        "test": test_ind.tolist(),
    }


def filter_and_transform(
    dataset: Any,
    max_seq_length: Optional[int],
    tokenizer: Tokenizer,
    seq_lengths: Optional[List[int]],
    head_model: str,
    debug_num_cases: Optional[int] = None,
) -> Tuple[RawDatasetType, List[int], Optional[RawDatasetType]]:
    """
    Determines sequence lengths (in tokens), if not already given as `seq_lengths`.
    Also, if `max_seq_length` is given, `dataset` is split into a dev set (all
    sequences of length `<= max_seq_length`) and a test set (all remaining
    sequences, sorted in ascending order).

    """
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
    dev_results: RawDatasetType = []
    test_results: RawDatasetType = []
    num_used = 0
    num_total = 0
    if max_seq_length is not None:
        print(
            f"\nProcessing dataset, filtering out records with > {max_seq_length} tokens"
        )
    else:
        print(f"\nProcessing dataset")
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
        new_case = {
            "instruction": instruction,
            "output": output,
            "num_tokens_instruction": seq_length,
        }
        if max_seq_length is None or seq_length <= max_seq_length:
            num_used += 1
            dev_results.append(new_case)
            if debug_num_cases is not None and num_used >= debug_num_cases:
                print(f"DEBUG: Stop with {num_used} records.")
                break
        elif max_seq_length is not None:
            test_results.append(new_case)
    if max_seq_length is not None:
        print(f"\nKept {num_used} of {num_total} records")
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
    return dev_results, seq_lengths, test_results


def get_instruction_template(head_model: str) -> Tuple[str, Tuple[str, ...]]:
    template = "\n".join(
        PROMPTLINES_PREFIX
        + ["{context}"]
        + PROMPTLINES_POSTFIX
        + PROMPTLINES_FINAL[head_model]
    )
    return (
        template,
        (
            "context",
            "question",
            "choice_A",
            "choice_B",
            "choice_C",
            "choice_D",
        ),
    )


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
        repo_id: str = "THUDM/LongBench-v2",
        access_token: Optional[str] = None,
        debug_num_cases: Optional[int] = None,
        trainloader_longest_first: bool = False,
        trainloader_shortest_first: bool = False,
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
            repo_id=repo_id,
            access_token=access_token,
            metadata_dir=metadata_dir,
            debug_num_cases=debug_num_cases,
            trainloader_longest_first=trainloader_longest_first,
            trainloader_shortest_first=trainloader_shortest_first,
        )
        self.truncation_context_width = None
        self.truncation_lengths = None

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        num_devices: int = 1,
        rank: Optional[int] = None,
        max_seq_length: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().connect(
            tokenizer,
            batch_size,
            num_devices,
            rank,
            max_seq_length,
            **kwargs,
        )
        truncation_context_width = kwargs.get("truncation_context_width")
        if truncation_context_width is None:
            raise ValueError("truncation_context_width must be provided")
        self.truncation_context_width = truncation_context_width

    def _filter_and_transform(self, dataset: Any) -> List[Dict[str, str]]:
        metadata = self._load_metadata(len(dataset))
        if metadata is None:
            meta_path = Path(self.metadata_dir) / METADATA_FNAME
            raise FileNotFoundError(
                f"Error trying to load metadata from {meta_path}. The metadata "
                "file must exist. Use `LongBenchV2` first to create it."
            )
        seq_lengths = metadata[METADATA_SEQ_LENGTHS_KEY][self.model_name]
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
