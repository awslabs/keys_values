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

from keys_values.distributed.fabric import Fabric
from keys_values.distributed.module_wrapper import AccessWeightsGradients

DebugStoreGradsNamePredicate = Callable[[str], bool]


class CPUOffloadAccumulateGradients:
    """
    Represents data distributed parallel gradient accumulation over the default
    process group. If this group size is `> 1` and `use_dist == True`, we
    use `dist.all_reduce`. In general, gradients per module are flattened
    (one vector per dtype), and the reductions are done for flat vectors.
    If `use_dist == False` or the group size is 1, `dist` is not used at all.

    This class is used to implement distributed data parallel (DDP) optimization
    with CPU offloading, where the full model and optimizer state resides on the
    host (CPU), but gradients are accumulated on the device (GPU).

    We also use it to implement standard DDP, with model and optimizer states
    on devices, see :meth:`__call__`.
    """

    def __init__(self, use_dist: bool = True):
        self.use_dist = use_dist

    @staticmethod
    def _is_mean_reducible(dtype: torch.dtype) -> bool:
        return (
            dtype == torch.float16
            or dtype == torch.bfloat16
            or dtype == torch.float32
            or dtype == torch.float64
        )

    def _all_reduce(self, vec: torch.Tensor, mean_reduction: bool):
        if self.use_dist:
            if mean_reduction and self._is_mean_reducible(vec.dtype):
                Fabric.all_reduce_avg(vec)
            else:
                Fabric.all_reduce_sum(vec)

    def __call__(
        self,
        module_pairs: List[Tuple[torch.nn.Module, Optional[torch.nn.Module]]],
        module_on_device: Optional[torch.nn.Module] = None,
        debug_modules: Optional[List[torch.nn.Module]] = None,
        mean_reduction: bool = False,
    ) -> Optional[float]:
        """
        Run gradient accumulation for module pairs `(mod_from, mod_to)`. This
        is called by every rank, and the ranks are synchronized here.

        By default, for the tuples `(mod_from, mod_to)`, `mod_from` is on the
        device, `mod_to` on the host (CPU). We also support DDP without CPU
        offloading, in which case `mod_to == None`, and gradients are written
        back to `mod_from` after accumulation.

        Args:
            module_pairs: List of `(mod_from, mod_to)` tuples. Here, `mod_from`
                is on the device, `mod_to` is on the CPU. If `mod_to == None`,
                gradients are written back to `mod_from`.
            module_on_device: If given, this source module is on the device.
                Its gradients are accumulated if the group size is > 1.
            debug_modules: Use for debugging only. Only for group size 1.
            mean_reduction: If `True`, use mean reduction in `all_reduce`, otherwise
                sum reduction. Mean reduction is done only for floating point types.

        Returns:
            Idle time in seconds at `all_reduce` sync point, or `None` if not
            distributed.

        """
        _use_dist = self.is_distributed
        num_none = sum(mod_to is None for _, mod_to in module_pairs)
        do_offload = num_none == 0
        if not do_offload and num_none != len(module_pairs):
            raise ValueError(
                "Entries of module_pairs: Either all mod_to are None, or none"
            )
        if debug_modules is None:
            debug_modules = [None] * len(module_pairs)
        else:
            if _use_dist:
                raise ValueError("debug_modules supported only if len(group) == 1")
            assert len(debug_modules) == len(module_pairs)
        idle_time = 0
        for (mod_from, mod_to), mod_debug in zip(module_pairs, debug_modules):
            access = AccessWeightsGradients(mod_from)
            flat_vectors = access.get_gradients()
            if _use_dist:
                idle_time_now = None
                start_time = time.perf_counter()
                for vec in flat_vectors.values():
                    self._all_reduce(vec, mean_reduction)
                    if idle_time_now is None:
                        idle_time_now = time.perf_counter() - start_time
                if idle_time_now is not None:
                    idle_time += idle_time_now
            mod_from.zero_grad(set_to_none=True)
            if do_offload:
                flat_vectors = {
                    k: v.to(device=torch.device("cpu")) for k, v in flat_vectors.items()
                }
            else:
                mod_to = mod_from
            AccessWeightsGradients(mod_to).accumulate_gradients(flat_vectors)

            if mod_debug is not None:
                for name, param in mod_debug.named_parameters():
                    param_comp = mod_from.get_parameter(name)
                    print(f"Compare {name}")
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
            if _use_dist:
                for vec in flat_vectors.values():
                    self._all_reduce(vec, mean_reduction)
            AccessWeightsGradients(module_on_device).accumulate_gradients(flat_vectors)

        return idle_time if _use_dist else None

    def test_all_reduce(self):
        if self.use_dist:
            device = Fabric.device()
            my_rank = Fabric.rank()
            vec = (
                torch.arange(
                    1,
                    10,
                    dtype=torch.int32,
                    device=device,
                )
                * my_rank
            )
            Fabric.all_reduce_sum(vec)
            all_factor = sum(range(Fabric.world_size()))
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
        return self.use_dist and Fabric.world_size() > 1

    def rank(self) -> int:
        return Fabric.rank() if self.use_dist else 0

    def world_size(self) -> int:
        return Fabric.world_size() if self.use_dist else 1
