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
from typing import List, Optional, Dict, Any, Tuple, Literal

from tokenizers import Tokenizer as HFTokenizer

from keys_values.constants import DEFAULT_IGNORE_INDEX
from keys_values.data.base import DataModuleEncoding
from keys_values.data.constants import TARGETS_STRINGS_NAME
from keys_values.data.helmet import Helmet
from keys_values.data.load_ruler_dev_eval import (
    load_ruler_dev_eval,
    DATASET_PARENT_DIR,
    DEFAULT_NUM_SAMPLES,
    MAX_LENGTH_TO_NUM_TOKENS,
    SUPPORTED_DATASET_KEYS,
)
from keys_values.kvcache.smart_lastrec import (
    SmartInitialInformation,
    end_initial_regex_from_string,
)

METADATA_FNAME = "ruler_metadata.json"


class Ruler(Helmet):
    """Data module for synthetic RULER tasks.

    Instances are generated with the RULER repository and loaded via
    :func:`load_ruler_dev_eval`. Different to HELMET, the context length is
    not limited to 128k, and the number of instances can be chosen
    (`num_samples`). The first
    :const:`keys_values.data.load_ruler_dev_eval.EVAL_QUESTIONS_NUM` instances
    become the test set, the remaining ones the development set, which is
    further divided into train and validation sets using
    `val_split_fraction`.

    Each RULER instance has a list of reference strings, all of which have to
    be part of the answer. For training, they are joined into a single
    target, see :func:`keys_values.data.load_ruler_dev_eval.format_target`.
    Sample-based metrics are computed against the list, which is passed in
    :const:`TARGETS_STRINGS_NAME`.

    Apart from this, everything is the same as in :class:`Helmet`. Metadata
    is stored in a different file, and its keys contain `num_samples` as
    well:
    - `data[METADATA_SEQ_LENGTHS_KEY][dataset_key][max_length][num_samples][model_name][split]`

    """

    EncodedName = "Ruler"
    BenchmarkName = "RULER"
    MetadataFName = METADATA_FNAME
    SupportedDatasetKeys = SUPPORTED_DATASET_KEYS

    def __init__(
        self,
        dataset_key: str,
        max_length: Literal["8k", "16k", "32k", "64k", "128k", "256k"] = "128k",
        dataset_parent_dir: str = DATASET_PARENT_DIR,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        mask_prompt: bool = True,
        val_split_fraction: float = 0.1,
        ignore_index: int = DEFAULT_IGNORE_INDEX,
        max_seq_length: Optional[int] = None,
        seed: int = 42,
        metadata_dir: Optional[str] = None,
        store_split_in_metadata: bool = False,
        trainloader_longest_first: bool = False,
        trainloader_shortest_first: bool = False,
        recompute_lengths: bool = False,
    ):
        """
        Args:
            dataset_key: Name of the RULER task to load (e.g. ``"niah_single_1"``,
                ``"vt"``). Supported keys are
                :const:`keys_values.data.load_ruler_dev_eval.SUPPORTED_DATASET_KEYS`.
            max_length: Context-length bucket to load. One of ``"8k"``,
                ``"16k"``, ``"32k"``, ``"64k"``, ``"128k"``, ``"256k"``.
                Lengths are measured by the tokenizer used for generation, see
                :func:`keys_values.data.load_ruler_dev_eval.generate_source_data`.
            dataset_parent_dir: Directory where generated RULER data is stored
                on disk. Defaults to ``~/.cache/huggingface/ruler/data``.
            num_samples: Number of RULER instances to use, these are generated
                if needed.
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
            store_split_in_metadata: If `True`, we sample and store the
                train/val split in the metadata and also load it from there,
                instead of sampling it anew each time.
            trainloader_longest_first: If ``True``, the first training batch
                contains the longest sequences (useful for early OOM detection).
            trainloader_shortest_first: If ``True``, the first training batch
                contains the shortest sequences.
            recompute_lengths: If `True`, sequence lengths are recomputed even
                if they are in the metadata file. The previous information is
                overwritten.

        """
        if max_length not in MAX_LENGTH_TO_NUM_TOKENS:
            raise ValueError(
                f"max_length = {max_length} is not supported. Choose from:\n"
                + str(list(MAX_LENGTH_TO_NUM_TOKENS.keys()))
            )
        super().__init__(
            dataset_key=dataset_key,
            max_length=max_length,
            dataset_parent_dir=dataset_parent_dir,
            mask_prompt=mask_prompt,
            val_split_fraction=val_split_fraction,
            ignore_index=ignore_index,
            max_seq_length=max_seq_length,
            seed=seed,
            metadata_dir=metadata_dir,
            store_split_in_metadata=store_split_in_metadata,
            trainloader_longest_first=trainloader_longest_first,
            trainloader_shortest_first=trainloader_shortest_first,
            recompute_lengths=recompute_lengths,
        )
        self.num_samples = num_samples

    def _metadata_keys(
        self,
        root_key: str,
        split: str,
    ) -> List[str]:
        # The development set depends on `num_samples`
        return [
            root_key,
            self.dataset_key,
            self.max_length,
            str(self.num_samples),
            self.model_name,
            split,
        ]

    def _load_dev_eval(self) -> Tuple[Any, Any]:
        return load_ruler_dev_eval(
            self.dataset_key,
            max_length=self.max_length,
            dataset_parent_dir=self.dataset_parent_dir,
            num_samples=self.num_samples,
        )

    def _create_record(
        self,
        instance: Dict[str, Any],
        seq_length: int,
    ) -> Dict[str, Any]:
        record = super()._create_record(instance, seq_length)
        # Sample-based metrics need all reference strings
        record[TARGETS_STRINGS_NAME] = instance["outputs"]
        return record

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
        # See :func:`keys_values.data.load_ruler_dev_eval.get_instruction_template`.
        # For "vt" and "cwe", the in-context example starts with the same
        # instructions, so the first match is at the start
        if self.dataset_key == "niah_single_1":
            end_initial_regex = (
                end_initial_regex_from_string(
                    "I will quiz you about the", tokenizer=tokenizer
                )
                + r" [^ ]+ "
                + end_initial_regex_from_string("afterwards.", tokenizer=tokenizer)
            )
        elif self.dataset_key == "vt":
            substring = "Memorize and track the chain(s) of variable assignment hidden in the following text."
        elif self.dataset_key == "cwe":
            substring = "Memorize the ones that appear most often."
        elif self.dataset_key == "fwe":
            substring = "Find the three most frequently appeared coded words."
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

    def encode(self) -> DataModuleEncoding:
        if self.tokenizer is not None:
            raise RuntimeError("Cannot call encode after connect")
        kwargs = dict(
            dataset_key=self.dataset_key,
            max_length=self.max_length,
            dataset_parent_dir=self.dataset_parent_dir,
            num_samples=self.num_samples,
            mask_prompt=self.mask_prompt,
            val_split_fraction=self.val_split_fraction,
            ignore_index=self.ignore_index,
            max_seq_length=self.max_seq_length,
            seed=self.seed,
            metadata_dir=self.metadata_dir,
            store_split_in_metadata=self.store_split_in_metadata,
            trainloader_longest_first=self._trainloader_longest_first,
            trainloader_shortest_first=self._trainloader_shortest_first,
            recompute_lengths=self._recompute_lengths,
        )
        return DataModuleEncoding(name=self.EncodedName, kwargs=kwargs)
