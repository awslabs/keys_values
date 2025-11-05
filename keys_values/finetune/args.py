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
from typing import Optional, Tuple, Dict, Any, List


def _check_positive(value: Optional[float], name: str):
    if value is not None and value <= 0.0:
        raise ValueError(f"`{name}` must be positive, got {value}")


def _check_nonnegative(value: Optional[float], name: str):
    if value is not None and value < 0.0:
        raise ValueError(f"`{name}` must be nonnegative, got {value}")


def _set_attr(kwargs: Dict[str, Any], key: Optional[str], value: Optional[Any]):
    if key is not None and value is not None:
        kwargs[key] = value


def _append_line(lines: List[str], name: str, value: Optional[Any]):
    if value is not None:
        lines.append(f"  {name}: {value}")


HAS_LEARNING_RATE = {
    "Adam": "lr",
    "AdamW": "lr",
    "Adamax": "lr",
    "Adadelta": "lr",
    "RMSprop": "lr",
}


HAS_WEIGHT_DECAY = {
    "Adam": "weight_decay",
    "AdamW": "weight_decay",
    "Adamax": "weight_decay",
    "Adadelta": "weight_decay",
    "RMSprop": "weight_decay",
}


HAS_EPS = {
    "Adam": "eps",
    "AdamW": "eps",
    "Adamax": "eps",
    "Adadelta": "eps",
    "RMSprop": "eps",
}


HAS_MOMENTUM = {
    "RMSprop": "momentum",
    "SGD": "momentum",
}


@dataclass
class OptimizerArgs:
    name: Optional[str] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    eps: Optional[float] = None
    momentum: Optional[float] = None
    adam_betas: Optional[Tuple[float, float]] = None
    adadelta_rho: Optional[float] = None
    rmspprop_alpha: Optional[float] = None

    def __post_init__(self):
        if self.name is None:
            self.name = "AdamW"  # Default optimizer
        _check_positive(self.learning_rate, "learning_rate")
        _check_nonnegative(self.weight_decay, "weight_decay")
        _check_positive(self.eps, "eps")
        _check_nonnegative(self.momentum, "momentum")
        if self.adam_betas is not None:
            if not isinstance(self.adam_betas, tuple) or len(self.adam_betas) != 2:
                raise ValueError(f"adam_betas = {self.adam_betas} must be tuple of size 2")
            if any (not 0 <= x < 1 for x in self.adam_betas):
                raise ValueError(f"adam_betas = {self.adam_betas}, entries must be in [0, 1)")
        if self.adadelta_rho is not None and not (0 <= self.adadelta_rho <= 1):
            raise ValueError(f"adadelta_rho = {self.adadelta_rho}, must be in [0, 1]")
        _check_nonnegative(self.rmspprop_alpha, "rmspprop_alpha")

    def optimizer_kwargs(self):
        kwargs = dict()
        _set_attr(kwargs, HAS_LEARNING_RATE.get(self.name), self.learning_rate)
        _set_attr(kwargs, HAS_WEIGHT_DECAY.get(self.name), self.weight_decay)
        _set_attr(kwargs, HAS_EPS.get(self.name), self.eps)
        _set_attr(kwargs, HAS_MOMENTUM.get(self.name), self.momentum)
        if self.name in ("Adam", "AdamW", "Adamax"):
            _set_attr(kwargs, "betas", self.adam_betas)
        if self.name == "Adadelta":
            _set_attr(kwargs, "rho", self.adadelta_rho)
        if self.name == "RMSprop":
            _set_attr(kwargs, "alpha", self.rmspprop_alpha)

        return kwargs

    def __str__(self) -> str:
        lines = [
            "OptimizerArgs:",
            f"  name: {self.name}",
        ]
        _append_line(lines, "learning_rate", self.learning_rate)
        _append_line(lines, "weight_decay", self.weight_decay)
        _append_line(lines, "eps", self.eps)
        _append_line(lines, "momentum", self.momentum)
        _append_line(lines, "adam_betas", self.adam_betas)
        _append_line(lines, "adadelta_rho", self.adadelta_rho)
        _append_line(lines, "rmspprop_alpha", self.rmspprop_alpha)
        return "\n".join(lines)


@dataclass
class LoRAARgs:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    query: bool = True
    key: bool = False
    value: bool = True
    projection: bool = False
    mlp: bool = False
    head: bool = False
