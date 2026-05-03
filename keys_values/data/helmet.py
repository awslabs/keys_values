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
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Literal

from tokenizers import Tokenizer as HFTokenizer
from tqdm import tqdm

from keys_values.data.dataloader import MyDataLoader
from keys_values.data.load_helmet_dev_eval import (
    load_helmet_dev_eval,
    DATASET_PARENT_DIR,
    SUPPORTED_DATASET_KEYS,
)
from keys_values.data.module import (
    SequenceLengthFilteredDataModule,
    METADATA_SEQ_LENGTHS_KEY,
    METADATA_KEYS,
    RawDatasetType,
    NUM_TOKENS_NAME,
)
from keys_values.data.sft_dataset import SFTDataset, get_sft_collate_fn
from keys_values.kvcache.smart_lastrec import (
    SmartInitialInformation,
    end_initial_regex_from_string,
)
from keys_values.utils import get_dict, set_dict

METADATA_FNAME = "helmet_metadata.json"


class Helmet(SequenceLengthFilteredDataModule):
    """Data module for HELMET benchmark datasets.

    Loads development and evaluation splits via :func:`load_helmet_dev_eval`.
    The development split is further divided into train and validation sets
    using `val_split_fraction`. The evaluation split becomes the test set.

    Each HELMET instance already has its prompt fully formatted, so no prompt
    construction or truncation is performed here.

    If `metadata_dir` is given, a metadata file is loaded and/or stored. This
    is strongly recommended to save time. A dictionary is stored as JSON, with
    this structure:
    - `data[METADATA_SEQ_LENGTHS_KEY][dataset_key][max_length][model_name][split]`:
      List of sequence lengths (in tokens) for each record. Here, `model_name`
      because the tokenizer depends on the model, and `split` is "dev" or
      "eval".
    """

    def __init__(
        self,
        dataset_key: str,
        max_length: Literal["8k", "16k", "32k", "64k", "128k"] = "8k",
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
                ``"json_kv"``). Supported keys are
                :const:`keys_values.data.load_helmet_dev_eval.SUPPORTED_DATASET_KEYS`.
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
                out. Defaults to no filtering.
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
        if dataset_key not in SUPPORTED_DATASET_KEYS:
            raise ValueError(
                f"dataset_key = {dataset_key} is not supported. Choose from:\n"
                + str(SUPPORTED_DATASET_KEYS)
            )
        self.dataset_key = dataset_key
        self.max_length = max_length
        self.dataset_parent_dir = dataset_parent_dir
        self.metadata_dir = metadata_dir

    def _metadata_keys(self, split: str) -> List[str]:
        return [
            METADATA_SEQ_LENGTHS_KEY,
            self.dataset_key,
            self.max_length,
            self.tokenizer.model_name,
            split,
        ]

    def _get_dataset(self) -> Tuple[RawDatasetType, Optional[RawDatasetType]]:
        dev_data, eval_data = load_helmet_dev_eval(
            self.dataset_key,
            max_length=self.max_length,
            dataset_parent_dir=self.dataset_parent_dir,
        )
        print(f"\nTransforming HELMET '{self.dataset_key}' ({self.max_length}) ...")
        metadata = self._load_metadata()
        train_data, dev_seq_lengths, dev_needs_store = self._transform(
            dev_data, split="dev", seq_lengths=self._get_seq_lengths(metadata, "dev")
        )
        test_data, eval_seq_lengths, eval_needs_store = self._transform(
            eval_data, split="eval", seq_lengths=self._get_seq_lengths(metadata, "eval")
        )
        if dev_needs_store or eval_needs_store:
            if metadata is None:
                metadata = dict()
            if dev_needs_store:
                set_dict(metadata, self._metadata_keys("dev"), dev_seq_lengths)
            if eval_needs_store:
                set_dict(metadata, self._metadata_keys("eval"), eval_seq_lengths)
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
        results: RawDatasetType = []
        new_seq_lengths: List[int] = []
        for idx, instance in enumerate(data_iter):
            instruction = instance["input"]
            if seq_lengths is None:
                seq_length = self.tokenizer.encode(instruction).numel()
                new_seq_lengths.append(seq_length)
            else:
                seq_length = seq_lengths[idx]
            if self.max_seq_length is not None and seq_length > self.max_seq_length:
                continue
            output = instance["output"]
            results.append(
                {
                    "instruction": instruction,
                    "output": output,
                    NUM_TOKENS_NAME: seq_length,
                }
            )
        final_seq_lengths = new_seq_lengths if seq_lengths is None else seq_lengths
        print(f"Kept {len(results)} of {len(dataset)} {split} records")
        return results, final_seq_lengths, needs_store

    def _get_seq_lengths(
        self, metadata: Optional[Dict[str, Any]], split: str
    ) -> Optional[List[int]]:
        return get_dict(metadata, self._metadata_keys(split))

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
            seed=self.seed,
        )
        self.val_dataset = SFTDataset(
            **val_kwargs,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
            seed=self.seed,
        )
        if test_kwargs is not None:
            self.test_dataset = SFTDataset(
                **test_kwargs,
                mask_prompt=self.mask_prompt,
                ignore_index=self.ignore_index,
                seed=self.seed,
            )

    def _get_collate_fn(self) -> MyDataLoader:
        return get_sft_collate_fn(ignore_index=self.ignore_index)

    def smart_lastrec_info(self, tokenizer: HFTokenizer) -> SmartInitialInformation:
        """
        Returns:
            Information used to detect the end of the initial part supposed to
            remain in the cache for
            :class:`SmartInitialLastRecentlyInsertedKVCache`.

        """
        include_end_string = True
        max_initial_fraction = 0.1
        end_initial_regex = None
        substring = None
        if self.dataset_key in ("nq", "trivia_qa", "hotpot_qa", "pop_qa"):
            # See :func:`keys_values.data.load_helmet_dev_eval.load_rag`
            substring = "Answer: [answer]"
        elif self.dataset_key in ("alce_asqa", "alce_qampari"):
            # See :func:`keys_values.data.load_helmet_dev_eval.load_cited_generation`
            # These are instructions for the first demo shot (2 in total)
            substring = "Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents."
        elif self.dataset_key == "ms_macro":
            # See :func:`keys_values.data.load_helmet_dev_eval.load_rerank`
            substring = "Ranking: ID3 > ID1 > ID2"
        elif self.dataset_key in (
            "trec_coarse",
            "trec_fine",
            "nlu",
            "banking77",
            "clinc150",
        ):
            # See :func:`keys_values.data.load_helmet_dev_eval.load_icl`
            substring = 'Only output "label: {label}" and nothing else. '
        elif self.dataset_key in ("narrative_qa", "infinite_bench_qa"):
            # See :func:`keys_values.data.load_helmet_dev_eval.load_long_doc_qa`
            substring = "Answer the question as concisely as you can, using a single phrase if possible."
        elif self.dataset_key == "infinite_bench_mc":
            # See :func:`keys_values.data.load_helmet_dev_eval.load_long_doc_qa`
            substring = "output the answer using one single letter (A, B, C, or D). Don't say anything else."
        elif self.dataset_key == "infinite_bench_sum":
            # See :func:`keys_values.data.load_helmet_dev_eval.load_summarization`
            substring = "Do not provide any analysis or commentary."
        elif self.dataset_key == "multi_lex_sum":
            # See :func:`keys_values.data.load_helmet_dev_eval.load_summarization`
            substring = "the parties involved, and the outcomes of the case."
        elif self.dataset_key == "json_kv":
            # See :func:`keys_values.data.load_helmet_dev_eval.load_synthetic`
            # This will mostly not work, because the context precedes the
            # instructions (and may be long)
            substring = "Extract the value corresponding to the specified key in the JSON object below."
        elif self.dataset_key in ("ruler_mk_needle", "ruler_mk_uuid", "ruler_mv"):
            # See :func:`keys_values.data.load_helmet_dev_eval.load_synthetic`
            end_initial_regex = (
                end_initial_regex_from_string(
                    "I will quiz you about the", tokenizer=tokenizer
                )
                + r" [^ ]+ "
                + end_initial_regex_from_string("afterwards.", tokenizer=tokenizer)
            )
        else:
            raise AssertionError(f"Unrecognized dataset key: {self.dataset_key}")
        if end_initial_regex is None:
            end_initial_regex = end_initial_regex_from_string(
                substring, tokenizer=tokenizer
            )

        return SmartInitialInformation(
            end_initial_regex=end_initial_regex,
            max_initial_fraction=max_initial_fraction,
            include_end_string=include_end_string,
        )
