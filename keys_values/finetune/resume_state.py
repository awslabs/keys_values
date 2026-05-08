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
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import lightning as L
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from litgpt.utils import CycleIterator

from keys_values.data.dataloader import MyDataLoaderIterator
from keys_values.data.iterators import SimilarSequenceLengthIterator
from keys_values.data.module import SequenceLengthFilteredDataModule

TRAINSTATE_OPTIMIZER_FNAME = "training_state_optimizer.pth"

TRAINSTATE_REST_FNAME = "training_state.pth"

TRAINSTATE_ITERATOR_FNAME = "training_state_iterator_rank{rank}.pth"


def get_iterator(cycle_iter: CycleIterator) -> MyDataLoaderIterator:
    if cycle_iter._iterator is not None:
        return cycle_iter._iterator
    else:
        return iter(cycle_iter.iterable)


def check_train_iterator(train_iterator: CycleIterator):
    msg_parts = [
        "train_iterator must be CycleIterator, wrapping a MyDataLoaderIterator, which contains a batch iterator of type SimilarSequenceLengthIterator",
        f"type(train_iterator) = {type(train_iterator)}",
    ]
    if isinstance(train_iterator, CycleIterator):
        inner_iter = get_iterator(train_iterator)
        msg_parts.append(f"type(inner_iter) = {type(inner_iter)}")
        if isinstance(inner_iter, MyDataLoaderIterator):
            batch_iter = inner_iter._batch_iter
            if not isinstance(batch_iter, SimilarSequenceLengthIterator):
                msg_parts.append(f"type(batch_iter) = {type(batch_iter)}")
            else:
                msg_parts = None
    if msg_parts is not None:
        raise TypeError("\n".join(msg_parts))


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
        dataset: SequenceLengthFilteredDataModule,
        train_iterator: Optional[CycleIterator] = None,
    ):
        self.state = state
        self.train_iterator = None
        self.dataset = dataset
        self._state_components = None
        self._optimizer_names = None
        self._check_state()
        if train_iterator is not None:
            self.init_train_iterator(train_iterator)

    def _check_state(self):
        iter_num = self.state.get("iter_num")
        if iter_num is None or int(iter_num) != iter_num or iter_num < 0:
            raise ValueError("state['iter_num'] must be nonnegative integer")
        optimizer = self.state.get("optimizer")
        if optimizer is not None:
            opt_names = ("optimizer",)
            if not isinstance(optimizer, Optimizer):
                raise ValueError(
                    "state['optimizer'] must be torch.optim.optimize.Optimizer"
                )
            sched_names = ("scheduler",)
            scheduler = self.state.get("scheduler")
            if scheduler is None or not isinstance(scheduler, LRScheduler):
                raise ValueError(
                    "state['scheduler'] must be torch.optim.lr_scheduler.LRScheduler"
                )
        else:
            gpu_as_well = "gpu_optimizer" in self.state
            if gpu_as_well:
                opt_names = (
                    "cpu_optimizer",
                    "gpu_optimizer",
                )
            else:
                opt_names = ("cpu_optimizer",)
            for name in opt_names:
                optimizer = self.state.get(name)
                if optimizer is None or not isinstance(optimizer, Optimizer):
                    raise ValueError(
                        f"state['{name}'] must be torch.optim.optimize.Optimizer"
                    )
            if gpu_as_well:
                sched_names = (
                    "cpu_scheduler",
                    "gpu_scheduler",
                )
            else:
                sched_names = ("cpu_scheduler",)
            for name in sched_names:
                scheduler = self.state.get(name)
                if scheduler is None or not isinstance(scheduler, LRScheduler):
                    raise ValueError(
                        f"state['{name}'] must be torch.optim.lr_scheduler.LRScheduler"
                    )
        self._state_components = opt_names + sched_names
        self._optimizer_names = opt_names

    def init_train_iterator(self, train_iterator: CycleIterator):
        if self.train_iterator is not None:
            raise IndexError("train_iterator is already initialized")
        check_train_iterator(train_iterator)
        self.train_iterator = train_iterator

    def _extract_training_state(self) -> Dict[str, Any]:
        train_state = {
            name: self.state[name].state_dict() for name in self._state_components
        }
        kwargs = dict(dtype=torch.int64)
        train_state.update(
            {
                "data_state": self.dataset.training_state.state_dict(),
                "iter_num": torch.tensor(self.state["iter_num"], **kwargs),
                "train_iterator": get_iterator(self.train_iterator).state_dict(),
            }
        )
        return train_state

    def save_training_state(
        self,
        fabric: L.Fabric,
        file_dir: Path,
    ) -> Tuple[Path, ...]:
        if self.train_iterator is None:
            raise ValueError(
                "train_iterator must be initialized, call `init_train_iterator`"
            )
        train_state = self._extract_training_state()
        optim_state = {k: train_state[k] for k in self._optimizer_names}
        optim_path = file_dir / TRAINSTATE_OPTIMIZER_FNAME
        fabric.save(optim_path, state=optim_state)
        filter_names = self._optimizer_names
        # This part depends on the rank
        name = "train_iterator"
        rank = fabric.local_rank
        iter_state = {name: train_state[name]}
        iter_path = file_dir / TRAINSTATE_ITERATOR_FNAME.format(rank=rank)
        # Runs for all ranks, not just 0:
        torch.save(iter_state, iter_path)
        filter_names += (name,)
        rest_state = {k: v for k, v in train_state.items() if k not in filter_names}
        rest_path = file_dir / TRAINSTATE_REST_FNAME
        fabric.save(rest_path, state=rest_state)
        return optim_path, iter_path, rest_path


def load_training_state(file_dir: Path, rank: int) -> Dict[str, Any]:
    train_state = torch.load(file_dir / TRAINSTATE_OPTIMIZER_FNAME)
    train_state.update(
        torch.load(file_dir / TRAINSTATE_ITERATOR_FNAME.format(rank=rank))
    )
    train_state.update(torch.load(file_dir / TRAINSTATE_REST_FNAME))
    return train_state


_COMPONENT_NAMES = (
    "optimizer",
    "scheduler",
    "cpu_optimizer",
    "cpu_scheduler",
    "gpu_optimizer",
    "gpu_scheduler",
)


def restore_from_training_state(
    state: Dict[str, Any],
    train_iterator: CycleIterator,
    train_state: Dict[str, Any],
    rank: int,
    num_devices: int,
):
    """
    Restores components of `state` and `train_iterator` from training state
    `train_state`. This excludes the data part of the training state, which
    must be used elsewhere to restore the dataset (see
    :func:`restore_dataset_from_training_state`).

    Args:
        state: Components relevant for training
        train_iterator: Training iterator
        train_state: Training state to read from
        rank: Rank of device
        num_devices: Number of devices

    """
    ts_rank = SimilarSequenceLengthIterator.rank_from_state_dict(
        train_state["train_iterator"]
    )
    if ts_rank != rank:
        raise ValueError(
            f"train_state['train_iterator'] has rank {ts_rank}, but rank = {rank}"
        )
    ts_devices = SimilarSequenceLengthIterator.num_devices_from_state_dict(
        train_state["train_iterator"]
    )
    if ts_devices != num_devices:
        raise ValueError(
            f"train_state['train_iterator'] has num_devices {ts_devices}, but num_devices = {num_devices}"
        )
    check_train_iterator(train_iterator)
    state["iter_num"] = train_state["iter_num"].item()
    for name in _COMPONENT_NAMES:
        if name in state:
            if name not in train_state:
                raise ValueError(f"{name}: Contained in state, but not in train_state")
            state[name].load_state_dict(train_state[name])
        elif name in train_state:
            raise ValueError(f"{name}: Contained in train_state, but not in state")
    # Reconstruct the training iterator
    inner_iter = get_iterator(train_iterator)
    inner_iter.load_state_dict(train_state["train_iterator"])
    train_iterator._iterator = inner_iter


def restore_dataset_from_training_state(
    dataset: SequenceLengthFilteredDataModule,
    file_dir: Path,
):
    rest_state = torch.load(file_dir / TRAINSTATE_REST_FNAME)
    dataset.load_training_state(rest_state["data_state"])
