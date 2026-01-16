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
from typing import Optional


REDUCTION_FACTORS = [
    3 / 4,
    2 / 4,
    1 / 4,
    3 / 16,
    2 / 16,
    1 / 16,
    3 / 64,
    2 / 64,
    1 / 64,
]


class TemporaryArrayLimit:
    """
    In order to limit the size of device memory spikes, several classes accept
    a temporary array limit (in GB). An object of this class maintains this
    limit.

    The limit can be reduced when an out of memory exception is caught. The
    current value is maintained here. Participating classes keep a reference
    to this object and read the limit from here.

    """

    def __init__(self, init_val: float, name: str):
        if init_val <= 0:
            raise ValueError("Initial value must be positive (unit is GB)")
        self._init_val = init_val
        self._curr_val = init_val
        self._name = name
        self._pos = 0

    def __call__(self, *args, **kwargs):
        return self._curr_val

    @property
    def name(self) -> str:
        return self._name

    def reduce(self) -> Optional[str]:
        if self._pos >= len(REDUCTION_FACTORS):
            return (
                f"Cannot reduce {self._name} any further. Started with "
                f"{self._init_val}, reduced until {self._curr_val}."
            )
        self._curr_val = self._init_val * REDUCTION_FACTORS[self._pos]
        self._pos += 1
        return None
