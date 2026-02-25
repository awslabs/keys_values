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
from typing import List, Tuple

import torch


def smallest_covering_ranges(
    slots: List[int],
    max_num_ranges: int,
) -> List[Tuple[int, int]]:
    """
    Given list of integers `slots`, determine a list of ranges `(a, b)`,
    of size at most `max_num_ranges`, so that the union of `range(a, b)`
    contains `slots`, and the sum of `b - a` is minimum.

    Args:
        slots: List of integers
        max_num_ranges: Maximum number of ranges to return

    Returns:
        List of ranges `(a, b)` covering `slots`

    """
    assert max_num_ranges >= 1
    slots = sorted(slots)
    mn, mx = slots[0], slots[-1]
    if max_num_ranges == 1:
        return [(mn, mx + 1)]
    if mx < mn + len(slots) - 1:
        raise ValueError(f"Slots {slots} must not have duplicates")
    diffs = sorted(
        [(a, b - a) for a, b in zip(slots[:-1], slots[1:]) if b > a + 1],
        key=lambda x: x[1],
        reverse=True,
    )[: (max_num_ranges - 1)]
    holes = (
        [(None, mn)]
        + sorted(
            [(a, a + l) for a, l in diffs],
            key=lambda x: x[0],
        )
        + [(mx, None)]
    )
    result = [(hole1[1], hole2[0] + 1) for hole1, hole2 in zip(holes[:-1], holes[1:])]
    return result


def storage_id(x: torch.Tensor) -> int:
    """
    Provides a unique ID for a tensor. See discussion on:
    https://stackoverflow.com/questions/67289617/pytorch-tensor-storages-have-the-same-id-when-calling-the-storage-method

    Weaknesses:
    - Tensors which use the same underlying storage, have the same ID. Can be
        resolved if needed
    - Tensors on different devices can in principle have the same ID. Very
        unlikely

    Args:
        x: PyTorch tensor

    Returns:
        ID based on the underlying storage

    """
    return x.untyped_storage().data_ptr()
