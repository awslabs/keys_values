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
from dataclasses import dataclass
from typing import Any, Set, Optional, Union, Dict, Tuple, Iterable

import torch

from keys_values.kvcache.utils import bytes_for_torch_dtype, storage_id


@dataclass(frozen=True)
class ArraysForCleanupStats:
    num: int
    max_num: int
    total_mem: float  # In GB


class ArraysForCleanup:
    """
    Inspired from:
    https://github.com/pytorch/pytorch/issues/82218 (tmm1, Aug 11, 2023).

    If an OOM error occurs during a gradient computation in
    :class:`LongContextGradientModel`, it can happen that tensors in the
    autograd graph are not released, even though they are marked in the
    garbage collector. We track these tensors with the autograd saved
    tensors hook mechanism, see :class:`CellComputationAutogradHooks`.

    The original "solution" proposed in the GitHub issue does not work for
    me, it affects too many tensors, not just those stored in the graph.

    """

    def __init__(
        self,
        protected_ids: Optional[Union[Set[int], Dict[int, str]]] = None,
    ):
        self._arrays = dict()
        self._num = 0
        self._max_size = 0
        if protected_ids is None:
            protected_ids = set()
        self._protected_ids = protected_ids

    def reset(self):
        self._arrays = dict()
        self._num = 0
        self._max_size = 0

    def add(self, x: Any):
        if isinstance(x, torch.Tensor):
            id_x = storage_id(x)
            if id_x not in self._arrays and id_x not in self._protected_ids:
                self._arrays[id_x] = x
                self._num += 1
                self._max_size = max(self._max_size, self._num)

    def remove(self, x: Any):
        id_x = storage_id(x)
        if id_x in self._arrays:
            del self._arrays[id_x]
            self._num -= 1

    def cleanup(self, verbose: bool = True):
        if verbose:
            stats = self.stats()
        for array in self._arrays.values():
            array.grad = None
            array.storage().resize_(0)
        self._arrays = dict()
        self._num = 0
        if verbose:
            if stats.num > 0:
                print(
                    f"Deallocated {stats.num} arrays [total: {stats.total_mem:.3f} GB]."
                )
            else:
                print("Autograd graph has been properly deallocated.")

    def stats(self) -> ArraysForCleanupStats:
        assert self._num == len(self._arrays), (self._num, len(self._arrays))
        total_mem = sum(
            x.numel() * bytes_for_torch_dtype(x.dtype) for x in self._arrays.values()
        ) / (2**30)
        return ArraysForCleanupStats(
            num=self._num,
            max_num=self._max_size,
            total_mem=total_mem,
        )


def _params_and_buffers(
    model: torch.nn.Module,
    map_names: bool,
) -> Union[Iterable[torch.Tensor], Iterable[Tuple[str, torch.Tensor]]]:
    if not map_names:
        for param in model.parameters():
            x = param.data
            if x is not None:
                yield x
        for buffer in model.buffers():
            yield buffer
    else:
        for name, param in model.named_parameters():
            x = param.data
            if x is not None:
                yield name, x
        for tup in model.named_buffers():
            yield tup


def protect_named_params_buffers_of_model(
    model: torch.nn.Module,
    map_names: bool = False,
) -> Union[Set[int], Dict[int, str]]:
    """
    Args:
        model: PyTorch model
        map_names: If `True`, return dictionary from IDs to names

    Returns:
        Object IDs of all weights tensors of named parameters and buffers
        of `model`. Can be used as `protected_ids` in
        :class:`ArraysForCleanup`.

    """
    if not map_names:
        return {storage_id(x) for x in _params_and_buffers(model, map_names)}
    else:
        return {
            storage_id(x): name for name, x in _params_and_buffers(model, map_names)
        }
