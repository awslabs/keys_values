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
import time
from typing import List, Tuple, Optional, Dict

import torch
import torch.distributed as dist
import lightning as L

from keys_values.finetune.utils import print_message
from keys_values.optimize.clone_model import copy_flat_vectors_to
from keys_values.optimize.module_wrapper import (
    AccessWeightsGradients,
    FlatVectors,
)


class DistributedPrimitives:
    @staticmethod
    def world_size(fabric: Optional[L.Fabric] = None) -> int:
        return dist.get_world_size() if fabric is None else fabric.world_size

    @staticmethod
    def rank(fabric: Optional[L.Fabric] = None) -> int:
        return dist.get_rank() if fabric is None else fabric.local_rank

    @staticmethod
    def all_reduce_sum(
        x: torch.Tensor,
        fabric: Optional[L.Fabric] = None,
        group: Optional[List[int]] = None,
    ):
        if fabric is not None:
            fabric.all_reduce(x, reduce_op="sum")
        else:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)


class CPUOffloadAccumulateGradients:
    """
    Represents data distributed parallel gradient accumulation over a number
    of ranks. If this group size is `> 1`, we use `dist.all_reduce`. In
    general, gradients per module are flattened (one vector per dtype), and
    the reductions are done for flat vectors. If the group size is 1, `dist`
    is not used at all, and `group[0]` does not matter.

    Usage:

    * Call :meth:`__call__` on different model shards
    * Call :meth:`finalize` at the end. Synchronization and data exchange
      between devices is either done in each :meth:`__call__`, or only
      at the end in :meth:`finalize`

    """

    def __init__(
        self,
        group: Optional[List[int]] = None,
        fabric: Optional[L.Fabric] = None,
    ):
        world_size = DistributedPrimitives.world_size(fabric)
        if group is None:
            group = list(range(world_size))
        else:
            group = sorted(group)
            if fabric is not None and group != list(range(world_size)):
                raise ValueError(f"group = {group}. If fabric is given, this must be {list(range(world_size))}")
            sz = len(group)
            if group[0] < 0 or any(x == y for x, y in zip(group[:-1], group[1:])):
                raise ValueError(
                    f"group = {group}, entries must be unique and non-negative"
                )
            if sz > 1:
                if group[-1] >= world_size:
                    raise ValueError(
                        f"group = {group}, entries must be < world_size = {world_size}"
                    )
        self.group = group
        self.fabric = fabric

    def __call__(
        self,
        module_pairs: List[Tuple[torch.nn.Module, torch.nn.Module]],
        module_on_device: Optional[torch.nn.Module] = None,
        debug_modules: Optional[List[torch.nn.Module]] = None,
    ):
        """
        Run gradient accumulation for module pairs `(mod_from, mod_to)`. This
        is called by every rank from `group`, and the ranks are synchronized
        here.

        Note that synchronization and exchange between devices can be
        delayed until :meth:`finalize` is called.

        Args:
            module_pairs: List of `(mod_from, mod_to)` tuples. Here, `mod_from`
                is on the device, `mod_to` is on the CPU.
            module_on_device: If given, this source module is on the device.
                Its gradients are accumulated if the group size is > 1.
            debug_modules: Use for debugging only. Only for group size 1.

        """
        idle_time = None
        use_dist = self.is_distributed
        if debug_modules is None:
            debug_modules = [None] * len(module_pairs)
        else:
            if use_dist:
                raise ValueError("debug_modules supported only if len(group) == 1")
            assert len(debug_modules) == len(module_pairs)
        for (mod_from, mod_to), mod_debug in zip(module_pairs, debug_modules):
            access = AccessWeightsGradients(mod_from)
            flat_vectors = access.get_gradients()
            if use_dist:
                start_time = time.perf_counter()
                # Depending on subclass, `all_reduce` is done here
                self._all_reduce_per_shard(flat_vectors)
                if start_time is not None:
                    idle_time = time.perf_counter() - start_time
                    start_time = None
                # At this point, `flat_vectors` on each rank is overwritten with
                # the sum of all `flat_vectors` on each rank
            flat_vectors = copy_flat_vectors_to(
                flat_vectors,
                device=torch.device("cpu"),
            )
            mod_from.zero_grad(set_to_none=True)
            AccessWeightsGradients(mod_to).accumulate_gradients(flat_vectors)
            if mod_debug is not None:
                for name, param in mod_debug.named_parameters():
                    param_comp = mod_from.get_parameter(name)
                    print_message(f"Compare {name}", self.fabric)
                    torch.testing.assert_close(param.data, param_comp.data)
                    if param.requires_grad:
                        src_arg = mod_from.get_parameter(name).grad.data
                        if param.grad is None:
                            param.grad = torch.nn.Parameter(src_arg)
                        else:
                            param.grad.data.copy_(src_arg)

        if module_on_device is not None and use_dist:
            access = AccessWeightsGradients(module_on_device)
            flat_vectors = access.get_gradients()
            for vec in flat_vectors.values():
                DistributedPrimitives.all_reduce_sum(
                    vec, self.fabric, self.group,
                )
            AccessWeightsGradients(module_on_device).accumulate_gradients(flat_vectors)

        if use_dist:
            self._update_idle_time(idle_time)

    def _update_idle_time(self, idle_time: float):
        pass

    def _all_reduce_per_shard(self, flat_vectors: FlatVectors):
        pass

    def finalize(self, model: torch.nn.Module) -> Optional[float]:
        """
        Ensures that gradient accumulation across devices is finalized. For
        implementations which synchronize with every :meth:`__call__`,
        nothing is done here.

        Args:
            model: Module to run `all_reduce` on gradients for. Its blocks
                must have been covered by `mod_to` submodules in previous
                :meth:`__call__` calls.

        Returns:
            In the distributed case, we return the idle time, i.e. the
            time between start and end of `all_reduce`.

        """
        raise NotImplementedError()

    def test_all_reduce(self):
        my_rank = DistributedPrimitives.rank(self.fabric)
        if my_rank not in self.group:
            raise AssertionError(f"Rank {my_rank} not in group {self.group}")
        device = torch.device("cuda", my_rank)
        vec = torch.arange(1, 10, dtype=torch.int32, device=device) * my_rank
        DistributedPrimitives.all_reduce_sum(vec, self.fabric, self.group)
        all_factor = sum(self.group)
        should_be = torch.arange(1, 10, dtype=torch.int32, device=device) * all_factor
        if not (vec == should_be).all().item():
            raise AssertionError(f"Rank {my_rank}: Have {vec} after all_reduce, should have {should_be}")

    @property
    def is_distributed(self) -> bool:
        return len(self.group) > 1

    def rank(self) -> int:
        return DistributedPrimitives.rank(self.fabric) if self.is_distributed else 0


# TODO: Remove this option: It cannot be competitive!
class CPUOffloadSyncPerShardAccumulateGradients(CPUOffloadAccumulateGradients):
    """
    Implementation for which each :meth:`__call__` call synchronizes
    between devices. May run up more idle time.

    """
    def __init__(
        self,
        group: Optional[List[int]] = None,
        fabric: Optional[L.Fabric] = None,
    ):
        super().__init__(group, fabric)
        self._idle_time = 0

    def _update_idle_time(self, idle_time: float):
        self._idle_time += idle_time

    def _all_reduce_per_shard(self, flat_vectors: FlatVectors):
        for vec in flat_vectors.values():
            DistributedPrimitives.all_reduce_sum(
                vec, self.fabric, self.group,
            )

    def finalize(self, model: torch.nn.Module) -> Optional[float]:
        # All synchronizations have been done already
        result = self._idle_time if self.is_distributed else None
        self._idle_time = 0  # reset
        return result


class CPUOffloadSyncOnceAccumulateGradients(CPUOffloadAccumulateGradients):
    """
    Implementation for which a single synchronization is done with the
    :meth:`finalize` call. May use more memory.

    """
    def __init__(
        self,
        group: Optional[List[int]] = None,
        fabric: Optional[L.Fabric] = None,
        flat_vecs_on_gpu: bool = True,
        debug_store_grads_for_name: Optional[str] = None,
    ):
        """
        TODO: Try to eliminate flat_vecs_on_gpu, checking what is faster!

        Args:
            group: Group of ranks. Defaults to all ranks (world)
            fabric: Fabric object. Defaults to None, in which case
                `torch.distributed` is used directly.
            flat_vecs_on_gpu: If `True`, flat vectors created from model
                gradients are copied to GPU device of rank before
                `all_reduce` is called. Defaults to `True`.
                T

        """
        super().__init__(group, fabric)
        self._flat_vecs_on_gpu = flat_vecs_on_gpu
        self._debug_store_grads_for_name = debug_store_grads_for_name
        self._debug_iter_count = 0

    @staticmethod
    def extract_grads(
        part_of_name: str,
        model: torch.nn.Module,
        access: AccessWeightsGradients,
        flat_vectors: FlatVectors,
        prefix: str,
    ) -> Dict[str, torch.Tensor]:
        debug_info = dict()
        dtype_str = None
        for name, param in model.named_parameters():
            if part_of_name in name and param.grad is not None:
                debug_info[prefix + ":" + name] = param.grad.data.clone()
                if dtype_str is None:
                    dtype_str = str(param.dtype)
        if dtype_str is None:
            raise AssertionError(f"No gradients in model for params containing {part_of_name}")
        if dtype_str not in flat_vectors:
            raise AssertionError(f"dtype_str = {dtype_str}, keys = {list(flat_vectors.keys())}")
        fvec = flat_vectors[dtype_str]
        for entry in access._grad_structure[dtype_str].entries:
            name = entry.name
            if part_of_name in name:
                start = entry.offset
                end = start + entry.size
                debug_info[prefix + "_fvec:" + name] = fvec[start:end].clone().reshape(
                    *entry.shape
                )
        return debug_info

    def finalize(self, model: torch.nn.Module) -> Optional[float]:
        # Synchronization of all gradients are done here. Calls of
        # :meth:`__call__` have written gradients into buffers of `model`.
        # We flatten them into a single vector and run `all_reduce`
        if not self.is_distributed:
            return None  # Nothing to do
        access = AccessWeightsGradients(model)
        flat_vectors = access.get_gradients()
        debug_info = None
        if self._debug_store_grads_for_name is not None:
            debug_info = self.extract_grads(
                self._debug_store_grads_for_name,
                model,
                access,
                flat_vectors,
                "before",
            )
        model.zero_grad(set_to_none=True)
        if self._flat_vecs_on_gpu:
            flat_vectors = copy_flat_vectors_to(
                flat_vectors,
                device=torch.device("cuda", self.rank()),
            )
        idle_time = None
        start_time = time.perf_counter()
        for vec in flat_vectors.values():
            DistributedPrimitives.all_reduce_sum(
                vec, self.fabric, self.group,
            )
            if start_time is not None:
                idle_time = time.perf_counter() - start_time
                start_time = None
        if self._flat_vecs_on_gpu:
            flat_vectors = copy_flat_vectors_to(
                flat_vectors,
                device=torch.device("cpu"),
            )
        AccessWeightsGradients(model).accumulate_gradients(flat_vectors)
        if self._debug_store_grads_for_name is not None:
            debug_info.update(
                self.extract_grads(
                    self._debug_store_grads_for_name,
                    model,
                    access,
                    flat_vectors,
                    "after",
                )
            )
            fname = f"./grad_accumulate_iter{self._debug_iter_count}_rank{self.rank()}.pth"
            print(f"DEBUG: Storing {self._debug_store_grads_for_name} gradients before/after to {fname}")
            torch.save(debug_info, fname)
            self._debug_iter_count += 1
        return idle_time
