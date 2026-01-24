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
from typing import List, Tuple, Optional

import torch
import torch.distributed as dist

from keys_values.optimize.clone_model import copy_flat_vectors_to
from keys_values.optimize.module_wrapper import AccessWeightsGradients


class CPUOffloadAccumulateGradients:
    """
    Represents data distributed parallel gradient accumulation over a number
    of ranks. If this group size is `> 1`, we use `dist.all_reduce`. In
    general, gradients per module are flattened (one vector per dtype), and
    the reductions are done for flat vectors. If the group size is 1, `dist`
    is not used at all, and `group[0]` does not matter.

    """
    def __init__(self, group: List[int]):
        group = sorted(group)
        sz = len(group)
        if group[0] < 0 or any(x == y for x, y in zip(group[:-1], group[1:])):
            raise ValueError(f"group = {group}, entries must be unique and non-negative")
        if sz > 1:
            if group[-1] >= dist.get_world_size():
                raise ValueError(f"group = {group}, entries must be < world_size = {dist.get_world_size()}")
        self.group = group

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

        Args:
            module_pairs: List of `(mod_from, mod_to)` tuples. Here, `mod_from`
                is on the device, `mod_to` is on the CPU.
            module_on_device: If given, this source module is on the device.
                Its gradients are accumulated if the group size is > 1.
            debug_modules: Use for debugging only. Only for group size 1.

        """
        use_dist = len(self.group) > 1
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
                # We run all-reduce on all parts of `flat_vectors`
                for vec in flat_vectors.values():
                    dist.all_reduce(vec, op=dist.ReduceOp.SUM, group=self.group)
                # At this point, `flat_vectors` on each rank is overwritten with
                # the sum of all `flat_vectors` on each rank
            flat_vectors = copy_flat_vectors_to(
                flat_vectors, device=torch.device("cpu"),
            )
            mod_from.zero_grad(set_to_none=True)
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

        if module_on_device is not None and use_dist:
            access = AccessWeightsGradients(module_on_device)
            flat_vectors = access.get_gradients()
            for vec in flat_vectors.values():
                dist.all_reduce(vec, op=dist.ReduceOp.SUM, group=self.group)
            AccessWeightsGradients(module_on_device).accumulate_gradients(
                flat_vectors
            )
