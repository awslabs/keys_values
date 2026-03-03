# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file exc ept in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import torch

from keys_values.debug_utils import size_quantiles_internal
from keys_values.utils import append_results_to_csv


TABLE_FNAMES = {
    "weight": "sizes/size_weights.csv",
    "grad": "sizes/size_gradients.csv",
}

NAMES_FNAMES = {
    "weight": "sizes/weight_names.csv",
    "grad": "sizes/gradient_names.csv",
}

ENTRIES_KEYS = ("weight", "grad")


@dataclass(frozen=True)
class SizeLogMapperRule:
    postfix: str
    sizes_names: Tuple[Tuple[int, str], ...]
    dim: int = -1

    def __post_init__(self):
        assert all(x[0] > 0 for x in self.sizes_names)

    def match(self, name: str) -> bool:
        return name.endswith(self.postfix)

    @property
    def sizes(self) -> Tuple[int, ...]:
        return tuple(x[0] for x in self.sizes_names)

    @property
    def full_size(self) -> int:
        return sum(x[0] for x in self.sizes_names)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(x[1] for x in self.sizes_names)

    def sub_names(self, name: str) -> Tuple[str, ...]:
        assert self.match(name)
        prefix = name[: -len(self.postfix)]
        return tuple(prefix + n for n in self.names)

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        len = x.shape[self.dim]
        if len != self.full_size:
            raise ValueError(f"x.shape = {x.shape}: {len} != {self.full_size}")
        return x.split(self.sizes, dim=self.dim)


class SizeLogMapper:
    """
    Helper for :class:`SizeWeightsGradientsLog`. Used to split certain
    parameter tensors into several parts, giving them distinct names.

    """

    def __init__(
        self,
        rules: List[SizeLogMapperRule],
    ):
        self.rules = rules

    def extend_names(self, names: List[str]) -> List[str]:
        new_names = []
        for name in names:
            for rule in self.rules:
                if rule.match(name):
                    new_names.extend(rule.sub_names(name))
                    name = None
                    break
            if name is not None:
                new_names.append(name)
        return new_names

    def rule(self, name: str) -> Optional[SizeLogMapperRule]:
        for _rule in self.rules:
            if _rule.match(name):
                return _rule
        return None


class SizeWeightsGradientsLog:
    """
    Maintains a log in the form of two tables (stored as CSV files), containing
    quantiles for absolute values of (1) weights and (2) gradients for a fixed
    model architecture along a number of steps: call them weights and gradients
    tables.

    Rows of the weights table are indexed by `(step, weight_idx, q)`, `step`
    the step number (running counter), `weight_idx` the index of the weight in
    `weight_names`, `q` the quantile level from `quantiles`. Rows of the
    gradients table are indexed by `(step, grad_idx, q)`, `grad_idx` the index
    of the gradient in `grad_names`. Here, `weight_names` and `grad_names` are
    set with the first call.

    Weights and gradients sizes are updated together, by passing `module`. This
    should be called when gradients have been computed, but before weights are
    updated by the optimizer.

    """

    def __init__(
        self,
        quantiles: List[float],
        path: Path,
        mapper: Optional[SizeLogMapper] = None,
        weight_column_names: Optional[Tuple[str, ...]] = None,
        grad_column_names: Optional[Tuple[str, ...]] = None,
    ):
        self.quantiles = quantiles
        self._quantile_names = tuple(f"{q:.3f}" for q in self.quantiles)
        self.path = path
        if weight_column_names is None:
            weight_column_names = ("step", "weight_idx", "q", "value")
        else:
            weight_column_names = tuple(weight_column_names)
        if len(weight_column_names) != 4:
            raise ValueError(
                f"weight_column_names = {weight_column_names}, must have length 4"
            )
        if grad_column_names is None:
            grad_column_names = ("step", "grad_idx", "q", "value")
        else:
            grad_column_names = tuple(grad_column_names)
        if len(grad_column_names) != 4:
            raise ValueError(
                f"grad_column_names = {grad_column_names}, must have length 4"
            )
        self.column_names = {
            "weight": weight_column_names,
            "grad": grad_column_names,
        }
        self.names: Dict[str, List[str]] = {k: None for k in ENTRIES_KEYS}
        self._name_pos: Dict[str, Dict[str, int]] = {k: None for k in ENTRIES_KEYS}
        self.mapper = mapper
        self.step = 0

    def _initialize(self, module: torch.nn.Module):
        if self.names[ENTRIES_KEYS[0]] is None:
            for k in ENTRIES_KEYS:
                if k == ENTRIES_KEYS[0]:
                    names = [name for name, _ in module.named_parameters()]
                else:
                    names = [
                        name
                        for name, param in module.named_parameters()
                        if param.requires_grad
                    ]
                if self.mapper is not None:
                    names = self.mapper.extend_names(names)
                self.names[k] = sorted(names)
                self._name_pos[k] = {name: i for i, name in enumerate(self.names[k])}
                names_path = self.path / NAMES_FNAMES[k]
                names_path.parent.mkdir(parents=True, exist_ok=True)
                with open(names_path, "w") as fp:
                    writer = csv.writer(fp, delimiter=",")
                    writer.writerow((self.column_names[k][1], "name"))
                    writer.writerows(list(enumerate(self.names[k])))

    @staticmethod
    def _get_tensor(k: str, param: torch.nn.Parameter) -> torch.Tensor:
        if k == ENTRIES_KEYS[0]:
            return param.data
        else:
            return param.grad.data

    def __call__(self, module: torch.nn.Module):
        self._initialize(module)
        new_entries = {k: [] for k in ENTRIES_KEYS}
        for name, param in module.named_parameters():
            for k in ENTRIES_KEYS:
                inputs = []
                rule = None if self.mapper is None else self.mapper.rule(name)
                if rule is not None:
                    sub_names = rule.sub_names(name)
                    pos = tuple(self._name_pos[k].get(n) for n in sub_names)
                    if not any(p is None for p in pos):
                        inputs = list(zip(pos, rule(self._get_tensor(k, param))))
                    elif k == ENTRIES_KEYS[0]:
                        raise ValueError(
                            f"Parameters {sub_names} are not valid names. Do not pass different models!"
                        )
                else:
                    pos = self._name_pos[k].get(name)
                    if pos is not None:
                        inputs = [(pos, self._get_tensor(k, param))]
                    elif k == ENTRIES_KEYS[0]:
                        raise ValueError(
                            f"Parameter {name} is not a valid name. Do not pass different models!"
                        )
                for pos, x in inputs:
                    qvals = size_quantiles_internal(x, self.quantiles)
                    new_entries[k].extend(
                        [
                            (self.step, pos, q, val)
                            for q, val in zip(self._quantile_names, qvals)
                        ]
                    )

        sort_key = lambda x: (x[1], float(x[2]))
        for k in ENTRIES_KEYS:
            new_entries[k] = sorted(new_entries[k], key=sort_key)
            table_path = self.path / TABLE_FNAMES[k]
            print(f"Append {len(new_entries[k])} entries to {table_path}")
            append_results_to_csv(
                [
                    {k: v for k, v in zip(self.column_names[k], row)}
                    for row in new_entries[k]
                ],
                table_path,
            )
        self.step += 1
