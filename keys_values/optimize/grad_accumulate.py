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
from typing import List, Tuple, Optional, Callable

import torch
import torch.distributed as dist
import lightning as L

from keys_values.finetune.utils import print_message
from keys_values.optimize.module_wrapper import AccessWeightsGradients


class DistributedPrimitives:
    @staticmethod
    def world_size(fabric: Optional[L.Fabric] = None) -> int:
        if fabric is not None:
            return fabric.world_size
        elif torch.cuda.is_available():
            return dist.get_world_size()
        else:
            return 1

    @staticmethod
    def rank(fabric: Optional[L.Fabric] = None) -> int:
        if fabric is not None:
            return fabric.local_rank
        elif torch.cuda.is_available():
            return dist.get_rank()
        else:
            return 0

    @staticmethod
    def device(fabric: Optional[L.Fabric] = None) -> torch.device:
        if fabric is not None:
            return fabric.device
        elif torch.cuda.is_available():
            return torch.device("cuda", DistributedPrimitives.rank(fabric))
        else:
            return torch.device("cpu")

    @staticmethod
    def all_reduce_sum(
        x: torch.Tensor,
        fabric: Optional[L.Fabric] = None,
        group: Optional[List[int]] = None,
    ):
        if fabric is not None or torch.cuda.is_available():
            if x.device != DistributedPrimitives.device(fabric):
                raise ValueError(f"x.device = {x.device}, must be {DistributedPrimitives.device(fabric)}")
            if fabric is not None:
                fabric.all_reduce(x, reduce_op="sum")
            else:
                dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)


DebugStoreGradsNamePredicate = Callable[[str], bool]


class CPUOffloadAccumulateGradients:
    """
    Represents data distributed parallel gradient accumulation over a number
    of ranks. If this group size is `> 1`, we use `dist.all_reduce`. In
    general, gradients per module are flattened (one vector per dtype), and
    the reductions are done for flat vectors. If the group size is 1, `dist`
    is not used at all, and `group[0]` does not matter.

    """

    def __init__(
        self,
        group: Optional[List[int]] = None,
        fabric: Optional[L.Fabric] = None,
        debug_store_grads_name_predicate: Optional[DebugStoreGradsNamePredicate] = None,
    ):
        if group is None:
            world_size = DistributedPrimitives.world_size(fabric)
            group = list(range(world_size))
        elif len(group) > 1:
            world_size = DistributedPrimitives.world_size(fabric)
            group = sorted(group)
            if fabric is not None and group != list(range(world_size)):
                raise ValueError(
                    f"group = {group}. If fabric is given, this must be {list(range(world_size))}"
                )
            if group[0] < 0 or any(x == y for x, y in zip(group[:-1], group[1:])):
                raise ValueError(
                    f"group = {group}, entries must be unique and non-negative"
                )
            if group[-1] >= world_size:
                raise ValueError(
                    f"group = {group}, entries must be < world_size = {world_size}"
                )
        else:
            group = [0]
        self.group = group
        self.fabric = fabric
        self._debug_store_grads_name_predicate = debug_store_grads_name_predicate
        self._debug_iter_count = 0

    def __call__(
        self,
        module_pairs: List[Tuple[torch.nn.Module, torch.nn.Module]],
        module_on_device: Optional[torch.nn.Module] = None,
        debug_modules: Optional[List[torch.nn.Module]] = None,
    ) -> Optional[float]:
        """
        Run gradient accumulation for module pairs `(mod_from, mod_to)`. This
        is called by every rank from `group`, and the ranks are synchronized
        here.

        Args:
            module_pairs: List of `(mod_from, mod_to)` tuples. Here, `mod_from`
                is on the device, `mod_to` is on the CPU.
            module_on_device: If given, this source module is on the device.
                Its gradients are accumulated if the group size is > 1.
            debug_modules: Use for debugging only. Only for group size 1.

        Returns:
            Idle time in seconds at `all_reduce` sync point, or `None` if not
            distributed.

        """
        use_dist = self.is_distributed
        if debug_modules is None:
            debug_modules = [None] * len(module_pairs)
        else:
            if use_dist:
                raise ValueError("debug_modules supported only if len(group) == 1")
            assert len(debug_modules) == len(module_pairs)
        idle_time = 0
        for (mod_from, mod_to), mod_debug in zip(module_pairs, debug_modules):
            # DEBUG
            if use_dist:
                print("\n*** Checking lora_B gradients before all_reduce ***")
            for name, param in mod_from.named_parameters():
                if "attn.qkv.lora_B" in name:
                    if param.requires_grad:
                        gradient = param.grad.data
                        temp = torch.abs(gradient).flatten()
                        vals, ind = torch.topk(temp, k=8)
                        print(f"Parameter {name}. Largest gradient entries:")
                        print(gradient.flatten()[ind])
                    else:
                        print(f"{name} has no gradient")
            # END DEBUG
            access = AccessWeightsGradients(mod_from)
            flat_vectors = access.get_gradients()
            if use_dist:
                idle_time_now = None
                start_time = time.perf_counter()
                for vec in flat_vectors.values():
                    DistributedPrimitives.all_reduce_sum(
                        vec, self.fabric, self.group,
                    )
                    if idle_time_now is None:
                        idle_time_now = time.perf_counter() - start_time
                idle_time += idle_time_now
            mod_from.zero_grad(set_to_none=True)
            flat_vectors = {
                k: v.to(device=torch.device("cpu"))
                for k, v in flat_vectors.items()
            }
            AccessWeightsGradients(mod_to).accumulate_gradients(flat_vectors)
            # DEBUG
            if use_dist:
                print("\n*** Checking lora_B gradients after all_reduce ***")
                for name, param in mod_from.named_parameters():
                    if "attn.qkv.lora_B" in name:
                        if param.requires_grad:
                            gradient = param.grad.data
                            temp = torch.abs(gradient).flatten()
                            vals, ind = torch.topk(temp, k=8)
                            print(f"Parameter {name}. Largest gradient entries:")
                            print(gradient.flatten()[ind])
                        else:
                            print(f"{name} has no gradient")
            # END DEBUG

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

        if module_on_device is not None:
            access = AccessWeightsGradients(module_on_device)
            flat_vectors = access.get_gradients()
            if use_dist:
                for vec in flat_vectors.values():
                    DistributedPrimitives.all_reduce_sum(
                        vec,
                        self.fabric,
                        self.group,
                    )
            AccessWeightsGradients(module_on_device).accumulate_gradients(flat_vectors)

        return idle_time if use_dist else None

    def test_all_reduce(self):
        device = DistributedPrimitives.device(self.fabric)
        my_rank = DistributedPrimitives.rank(self.fabric)
        vec = (
            torch.arange(
                1,
                10,
                dtype=torch.int32,
                device=device,
            )
            * my_rank
        )
        DistributedPrimitives.all_reduce_sum(vec, self.fabric, self.group)
        all_factor = sum(self.group)
        should_be = (
            torch.arange(
                1,
                10,
                dtype=torch.int32,
                device=device,
            )
            * all_factor
        )
        if not (vec == should_be).all().item():
            raise AssertionError(
                f"Rank {my_rank}, device {device}: Have {vec} after all_reduce, should have {should_be}"
            )

    @property
    def is_distributed(self) -> bool:
        return len(self.group) > 1

    def rank(self) -> int:
        return DistributedPrimitives.rank(self.fabric) if self.is_distributed else 0
