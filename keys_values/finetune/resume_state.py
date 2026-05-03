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
from typing import Any, Dict, List

import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from litgpt.utils import CycleIterator

from keys_values.data.iterators import SimilarSequenceLengthIterator
from keys_values.data.module import SequenceLengthFilteredDataModule


def get_iterator(cycle_iter: CycleIterator) -> SimilarSequenceLengthIterator:
    if cycle_iter._iterator is not None:
        return cycle_iter._iterator
    else:
        return iter(cycle_iter.iterable)


# TODO: Currently, train_iterator and dataset treatment are highly specific.
# Make this more general.
class TrainingStateManager:
    """
    This class is responsible for extracting and storing the training state from
    training components. Here, *training state* means information on top of a
    checkpoint of model weights and configuration. This information is needed
    for resuming a training run which has been stopped or crashed.

    Since the training state can be large (in particular the optimizer state), we
    store it only alongside a fixed number of last recent checkpoints. This is done
    by storing training states with all checkpoints, but removing those which are
    too old.
    """

    def __init__(
        self,
        state: Dict[str, Any],
        train_iterator: CycleIterator,
        dataset: SequenceLengthFilteredDataModule,
    ):
        if not isinstance(train_iterator, CycleIterator) or not isinstance(get_iterator(train_iterator), SimilarSequenceLengthIterator):
            raise TypeError("train_iterator must be CycleIterator, wrapping a SimilarSequenceLengthIterator")
        self.state = state
        self.train_iterator = train_iterator
        self.dataset = dataset
        self._do_cpu_offload = None
        self._state_components = None
        self._check_state()

    def _check_state(self):
        iter_num = self.state.get("iter_num")
        if iter_num is None or int(iter_num) != iter_num or iter_num < 0:
            raise ValueError("state['iter_num'] must be nonnegative integer")
        optimizer = self.state.get("optimizer")
        if optimizer is not None:
            self._do_cpu_offload = False
            if not isinstance(optimizer, Optimizer):
                raise ValueError("state['optimizer'] must be torch.optim.optimize.Optimizer")
            scheduler = self.state.get("scheduler")
            if scheduler is None or not isinstance(scheduler, LRScheduler):
                raise ValueError("state['scheduler'] must be torch.optim.lr_scheduler.LRScheduler")
            self._state_components = ("optimizer", "scheduler",)
        else:
            self._do_cpu_offload = True
            gpu_as_well = "gpu_optimizer" in self.state
            if gpu_as_well:
                opt_names = ("cpu_optimizer", "gpu_optimizer",)
            else:
                opt_names = ("cpu_optimizer",)
            for name in opt_names:
                optimizer = self.state.get(name)
                if optimizer is None or not isinstance(optimizer, Optimizer):
                    raise ValueError(f"state['{name}'] must be torch.optim.optimize.Optimizer")
            if gpu_as_well:
                sched_names = ("cpu_scheduler", "gpu_scheduler",)
            else:
                sched_names = ("cpu_scheduler",)
            for name in sched_names:
                scheduler = self.state.get(name)
                if scheduler is None or not isinstance(scheduler, LRScheduler):
                    raise ValueError(f"state['{name}'] must be torch.optim.lr_scheduler.LRScheduler")
            self._state_components = opt_names + sched_names

    def _extract_training_state(self) -> Dict[str, Any]:
        train_ind, val_ind = self.dataset.train_val_split_indices()
        train_state = {
            name: getattr(self.state, name).state_dict()
            for name in self._state_components
        }
        kwargs = dict(dtype=torch.int64)
        train_state.update(
            {
                "iter_num": torch.tensor(self.state["iter_num"], **kwargs),
                "train_iterator": get_iterator(self.train_iterator).state_dict(),
                "train_data_index": torch.tensor(train_ind, **kwargs),
                "val_data_index": torch.tensor(val_ind, **kwargs),
            }
        )
        return train_state
