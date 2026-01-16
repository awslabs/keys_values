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
from typing import List, Dict, Optional, Tuple
import math

import torch


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    offset: int
    size: int
    shape: Tuple[int, ...]
    requires_grad: bool

    def __post_init__(self):
        assert self.offset >= 0
        assert self.size >= 0
        assert math.prod(self.shape) == self.size

    @property
    def range(self) -> Tuple[int, int]:
        return self.offset, self.offset + self.size


@dataclass
class ParameterStructure:
    entries: List[ParameterSpec]
    dtype: torch.dtype
    size: int

    def __post_init__(self):
        sum_sizes = sum(x.size for x in self.entries)
        if sum_sizes != self.size:
            raise ValueError(
                f"size = {self.size}, sum_sizes = {sum_sizes}. Must be the same"
            )

    def append(self, name: str, shape: Tuple[int, ...], requires_grad: bool):
        size = math.prod(shape)
        self.entries.append(
            ParameterSpec(
                name=name,
                offset=self.size,
                size=size,
                shape=shape,
                requires_grad=requires_grad,
            )
        )
        self.size += size


FlatVectors = Dict[str, torch.Tensor]


class AccessWeightsGradients:
    """
    Allows to get or set all weights and all gradients of an :class:`torch.nn.Module`
    as flat vectors.

    Note: In principle, the flat vectors can be on devices other than the model
    parameters. But then, they are communicated in small units, which is
    inefficient. We strongly recommend to ensure flat vectors are on the same
    device as model parameters.

    """

    def __init__(self, module: torch.nn.Module):
        """
        Args:
            module (torch.nn.Module): Module to be wrapped. It is OK for the
                module not to have named parameters with gradients, or no
                named parameters at all.

        """
        self._module = module
        self._param_structure = None
        self._grad_structure = None
        self._device = None
        self._init_structure()

    def _init_structure(self):
        self._param_structure: Dict[str, ParameterStructure] = dict()
        self._grad_structure: Dict[str, ParameterStructure] = dict()
        device = None
        name_device = None
        for name, param in self._module.named_parameters():
            dtype = str(param.dtype)
            shape = tuple(param.shape)
            if math.prod(shape) != param.numel():
                raise ValueError(
                    f"name={name}, shape={shape}, numel={param.numel()}: Parameter is not flat"
                )
            requires_grad = param.requires_grad
            if dtype not in self._param_structure:
                self._param_structure[dtype] = ParameterStructure(
                    entries=[],
                    dtype=param.dtype,
                    size=0,
                )
            if device is None:
                device = param.device
                name_device = name
            elif device != param.device:
                raise ValueError(
                    f"All parameters must be on the same device (but {name_device} on {device}; {name} on {param.device})"
                )
            self._param_structure[dtype].append(name, shape, requires_grad)
            if requires_grad:
                if dtype not in self._grad_structure:
                    self._grad_structure[dtype] = ParameterStructure(
                        entries=[],
                        dtype=param.dtype,
                        size=0,
                    )
                self._grad_structure[dtype].append(name, shape, requires_grad)
        self._device = device

    @property
    def size_weights(self) -> Dict[str, int]:
        return {
            dtype: structure.size for dtype, structure in self._param_structure.items()
        }

    @property
    def size_gradients(self) -> Dict[str, int]:
        return {
            dtype: structure.size for dtype, structure in self._grad_structure.items()
        }

    def param_structure(self) -> Dict[str, ParameterStructure]:
        return self._param_structure

    def _check_sizes(
        self,
        structures: Dict[str, ParameterStructure],
        vecs: FlatVectors,
        vname: str,
    ):
        for dtype, vec in vecs.items():
            if dtype not in structures:
                raise ValueError(
                    f"{vname}[{dtype}] exists, but {dtype} not a key in structures"
                )
            structure = structures[dtype]
            vsize = vec.numel()
            if vsize != structure.size:
                raise ValueError(
                    f"{vname}[{dtype}] has size {vsize}, must be {structure.size}"
                )
            if vec.shape != (vsize,):
                raise ValueError(
                    f"{vname}[{dtype}] has shape {vec.shape}, numel()={vsize}. Must be flat"
                )

    def _get_internal(
        self,
        do_gradients: bool,
        out: Optional[FlatVectors] = None,
        device: Optional[torch.device] = None,
    ) -> FlatVectors:
        if device is None:
            device = self._device
        structures = self._grad_structure if do_gradients else self._param_structure
        if not structures:
            return dict()
        if out is None:
            out = {
                dtype: torch.empty(
                    (structure.size,),
                    dtype=structure.dtype,
                    device=device,
                )
                for dtype, structure in structures.items()
            }
        else:
            self._check_sizes(structures, out, "out")
        for dtype, target_vec in out.items():
            for pspec in structures[dtype].entries:
                start, end = pspec.range
                param = self._module.get_parameter(pspec.name)
                if do_gradients:
                    if param.grad is None:
                        raise IndexError(f"Parameter {pspec.name} has no gradient")
                    src_vec = param.grad.data.flatten()
                else:
                    src_vec = param.data.flatten()
                target_vec[start:end].copy_(src_vec)
        return out

    def get_weights(
        self,
        out: Optional[FlatVectors] = None,
        device: Optional[torch.device] = None,
    ) -> FlatVectors:
        return self._get_internal(
            do_gradients=False,
            out=out,
            device=device,
        )

    def get_gradients(
        self,
        out: Optional[FlatVectors] = None,
        device: Optional[torch.device] = None,
    ) -> FlatVectors:
        return self._get_internal(
            do_gradients=True,
            out=out,
            device=device,
        )

    def _set_internal(
        self,
        do_gradients: bool,
        src_vecs: FlatVectors,
    ):
        structures = self._grad_structure if do_gradients else self._param_structure
        self._check_sizes(structures, src_vecs, "src_vecs")
        for dtype, src_vec in src_vecs.items():
            for pspec in structures[dtype].entries:
                start, end = pspec.range
                param = self._module.get_parameter(pspec.name)
                src_arg = src_vec[start:end].view(param.shape)
                if do_gradients:
                    src_arg = src_arg.to(param.data.device)
                    if param.grad is None:
                        param.grad = torch.nn.Parameter(src_arg)
                    else:
                        param.grad.data.add_(src_arg)
                else:
                    param.data.copy_(src_arg)

    def set_weights(self, src_vecs: FlatVectors):
        self._set_internal(do_gradients=False, src_vecs=src_vecs)

    def accumulate_gradients(self, src_vecs: FlatVectors):
        self._set_internal(do_gradients=True, src_vecs=src_vecs)
