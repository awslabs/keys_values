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
"""
Single-device runtime for the RL / benchmark example drivers.

The drivers used `lightning.Fabric(devices=1, ...)` only for three things: the
target device, `init_module(empty_init=True)` at the run precision, and
`load_checkpoint`. Lightning Fabric was removed from the library (#167); this
is the minimal replacement on top of `keys_values.finetune.utils`, so the call
sites (`fabric.device`, `with fabric.init_module(...)`) stay unchanged.
"""

from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch

from keys_values.finetune.utils import init_module


@dataclass(frozen=True)
class SingleDevice:
    device: torch.device
    precision: str

    @staticmethod
    def create(accelerator: str) -> "SingleDevice":
        """`accelerator` is "cpu" or "cuda". Same precision rule as before:
        fp32 on CPU, bf16 on GPU."""
        if accelerator == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda requested but CUDA is not available")
            return SingleDevice(torch.device("cuda", 0), "bf16-true")
        if accelerator == "cpu":
            return SingleDevice(torch.device("cpu"), "32-true")
        raise ValueError(f"accelerator={accelerator!r}, must be 'cpu' or 'cuda'")

    def init_module(self, empty_init: bool = False) -> AbstractContextManager:
        return init_module(
            empty_init=empty_init, precision=self.precision, device=self.device
        )
