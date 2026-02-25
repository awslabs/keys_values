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
from typing import List, Optional, Dict, Any, Tuple, Union, Callable

import torch
from torch.utils.data import random_split

from litgpt.data import DataModule
from litgpt.prompts import Default
from litgpt.tokenizer import Tokenizer

from keys_values.data.base import pad_dataset
from keys_values.data.dataloader import MyDataLoader
from keys_values.data.evaluation import (
    SimilarSequenceLengthWithTasksSampler,
    EvaluationDataLoader,
)
from keys_values.data.iterators import SimilarSequenceLengthSampler


METADATA_SEQ_LENGTHS_KEY = "sequence_lengths"

METADATA_KEYS = {METADATA_SEQ_LENGTHS_KEY}

RawDatasetType = List[Dict[str, str]]

CollateFnType = Callable[[List[Dict[str, Any]]], Dict[str, Any]]

NUM_TOKENS_NAME = "num_tokens_instruction"


class SequenceLengthFilteredDataModule(DataModule):
    """
    This data module requires the tokenized sequence length to be determined
    for every data case (or loaded from a meta data file).

    For the :meth:`train_dataloader`, :meth:`val_dataloader`,
    :meth:`test_dataloader` iterators, we use a
    :class:`SimilarSequenceLengthIterable` sampler, which tries to form micro
    and macro batches with sequences of similar length.

    For the test set, if evaluation tasks are passed in :meth:`connect`, we
    use a :class:`EvaluationDataLoader`, which returns batches coupled with
    tasks. Here, we try to form micro batches with sequences of similar length,
    but there is no concept of macro batches.

    """

    def __init__(
        self,
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
            trainloader_longest_first: If set, :meth:`train_dataloader` returns
                a data loader whose first batch contain the longest sequences in
                the dataset, otherwise uses :class:`SimilarSequenceLengthIterable`.
                This is useful to detect OOM errors early, since they are most
                likely to happen with the longest batch.
            trainloader_shortest_first: Same as `trainloader_longest_first`,
                but the first batch contain the shortest sequences.

        """
        if trainloader_longest_first and trainloader_shortest_first:
            raise ValueError(
                "Cannot use both trainloader_longest_first and trainloader_shortest_first"
            )
        self.mask_prompt = mask_prompt
        self.val_split_fraction = val_split_fraction
        self.ignore_index = ignore_index
        self.max_seq_length = 100000 if max_seq_length is None else max_seq_length
        self.seed = seed
        self.head_model = None
        self._is_sequence_classification = None
        self.prompt_style = Default()
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
        self._test_eval_tasks = None
        # Maintain sequence lengths (in tokens) for cases in training set.
        # This is used to support specialized data loaders.
        self._sequence_lengths = None

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
        Specific arguments:
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
            rank: Rank (only if `num_devices > 1`)
            max_seq_length: Cutoff for sequence length
            **kwargs: See above

        """
        if tokenizer is None:
            raise ValueError("tokenizer must be provided")
        if num_devices < 1:
            raise ValueError("num_devices must be >= 1")
        if num_devices > 1:
            if rank is None or not (0 <= rank < num_devices):
                raise ValueError(f"rank = {rank}, must be in [0, {num_devices})")
        else:
            rank = 0
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_devices = num_devices
        self.rank = rank
        if max_seq_length is not None:
            self.max_seq_length = max_seq_length
        self.val_batch_size = kwargs.get("val_batch_size")
        if self.val_batch_size is None:
            self.val_batch_size = self.batch_size
        self.test_batch_size = None
        self._test_eval_tasks = None
        self.test_batch_size = kwargs.get("test_batch_size")
        if self.test_batch_size is None:
            self.test_batch_size = self.val_batch_size
        self._test_eval_tasks = kwargs.get("eval_tasks")

    def _get_dataset(self) -> Tuple[RawDatasetType, Optional[RawDatasetType]]:
        """
        Returns training and test datasets (the latter is optional). These are
        lists of dictionaries, each item must contain the length of the
        tokenized sequence in :const:`NUM_TOKENS_NAME`. The training set
        returned here is split into training and validation set.

        Returns:
            `(train_dataset, test_dataset)`, where `test_dataset` can be `None`
            if a test set is not supported.

        """
        raise NotImplementedError()

    def prepare_data(self) -> None:
        # Make sure that if dataset has not yet been downloaded, the hard
        # work is done once only.
        self._get_dataset()

    def _create_datasets(
        self,
        train_kwargs: Dict[str, Any],
        val_kwargs: Dict[str, Any],
        test_kwargs: Optional[Dict[str, Any]],
    ):
        """
        Creates `self.train_dataset`, `self.val_dataset`, `self.test_dataset`
        (the latter only if `test_kwargs` is given).

        Args:
            train_kwargs: Arguments for `train_dataset` creation
            val_kwargs: Arguments for `val_dataset` creation
            test_kwargs: Arguments for `test_dataset` creation. Optional.

        """
        raise NotImplementedError()

    def setup(self, stage: str = "") -> None:
        """
        Note: Datasets `train_dataset`, `val_dataset`, `test_dataset` are
        padded so that their size becomes a multiple of
        `B * num_devices`, where `B` is
        `batch_size, val_batch_size, test_batch_size` respectively, i.e. the
        micro-batch size. Such padding entries are filtered out by the
        collators. They are needed so that samplers can do their job
        properly. For `test_dataset` with `_test_eval_tasks` in place, padding
        is done only to a multiple of `test_batch_size`.

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
            "train": [record[NUM_TOKENS_NAME] for record in train_data],
            "valid": [record[NUM_TOKENS_NAME] for record in val_data],
        }
        if test_data is not None:
            self._sequence_lengths["test"] = [
                record[NUM_TOKENS_NAME] for record in test_data
            ]

        # Padding for the test set depends on whether we run evaluation only
        # (`_test_eval_tasks` given) or as part of training
        if self._test_eval_tasks is not None:
            test_pad_num_devices = 1
        else:
            test_pad_num_devices = self.num_devices
        train_kwargs = dict(
            data=pad_dataset(
                train_data,
                batch_size=self.batch_size,
                num_devices=self.num_devices,
            ),
            tokenizer=self.tokenizer,
            prompt_style=Default(),
            max_seq_length=self.max_seq_length,
        )
        val_kwargs = dict(
            data=pad_dataset(
                val_data,
                batch_size=self.val_batch_size,
                num_devices=self.num_devices,
            ),
            tokenizer=self.tokenizer,
            prompt_style=Default(),
            max_seq_length=self.max_seq_length,
        )
        if test_data is not None:
            test_kwargs = dict(
                data=pad_dataset(
                    test_data,
                    batch_size=self.test_batch_size,
                    num_devices=test_pad_num_devices,
                ),
                tokenizer=self.tokenizer,
                prompt_style=Default(),
                max_seq_length=-1,
            )
        else:
            test_kwargs = None
        self._create_datasets(train_kwargs, val_kwargs, test_kwargs)

    def _get_collate_fn(self) -> CollateFnType:
        """
        Returns:
            Collator function, to create batch from list of cases. Must filter
            out pad cases.

        """
        raise NotImplementedError()

    def train_dataloader(self) -> MyDataLoader:
        assert self._sequence_lengths is not None
        len_sl = len(self._sequence_lengths["train"])
        assert 0 < len_sl <= len(self.train_dataset), (len_sl, len(self.train_dataset))
        return MyDataLoader(
            dataset=self.train_dataset,
            batch_sampler=SimilarSequenceLengthSampler(
                sequence_lengths=self._sequence_lengths["train"],
                micro_batch_size=self.batch_size,
                seed=self.seed,
                num_devices=self.num_devices,
                rank=self.rank,
                shuffle=True,
                longest_first=self._trainloader_longest_first,
                shortest_first=self._trainloader_shortest_first,
            ),
            collate_fn=self._get_collate_fn(),
        )

    def val_dataloader(self) -> MyDataLoader:
        assert self._sequence_lengths is not None
        len_sl = len(self._sequence_lengths["valid"])
        assert 0 < len_sl <= len(self.val_dataset), (len_sl, len(self.val_dataset))
        return MyDataLoader(
            dataset=self.val_dataset,
            batch_sampler=SimilarSequenceLengthSampler(
                sequence_lengths=self._sequence_lengths["valid"],
                micro_batch_size=self.val_batch_size,
                seed=self.seed,
                num_devices=self.num_devices,
                rank=self.rank,
                shuffle=False,
            ),
            collate_fn=self._get_collate_fn(),
        )

    def test_dataloader(self, num_devices: int = 1) -> Union[MyDataLoader, EvaluationDataLoader]:
        if self.test_dataset is None:
            raise IndexError("Test dataset is not defined")
        assert self._sequence_lengths is not None
        len_sl = len(self._sequence_lengths["test"])
        assert 0 < len_sl <= len(self.test_dataset), (len_sl, len(self.test_dataset))
        if self._test_eval_tasks is None:
            return MyDataLoader(
                dataset=self.test_dataset,
                batch_sampler=SimilarSequenceLengthSampler(
                    sequence_lengths=self._sequence_lengths["test"],
                    micro_batch_size=self.test_batch_size,
                    seed=self.seed,
                    num_devices=self.num_devices,
                    rank=self.rank,
                    shuffle=False,
                ),
                collate_fn=self._get_collate_fn(),
            )
        else:
            # Cross product between test dataset and evaluation tasks (these
            # are typically different model checkpoints)
            return EvaluationDataLoader(
                dataset=self.test_dataset,
                batch_sampler=SimilarSequenceLengthWithTasksSampler(
                    sequence_lengths=self._sequence_lengths["test"],
                    micro_batch_size=self.test_batch_size,
                    num_tasks=len(self._test_eval_tasks),
                ),
                collate_fn=self._get_collate_fn(),
                eval_tasks=self._test_eval_tasks,
                delay_tokenization=True,
            )
