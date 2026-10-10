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
from typing import List, Tuple, Union, Optional

import torch

from keys_values.utils import index_to_3d


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


PositionsType = Union[torch.Tensor, Tuple[int, int]]


def positions_wrap_around(
    num: int,
    current: int,
    start: int,
    end: int,
    batch_size: int,
    n_query_groups: int,
    device: torch.device,
    return_tensor: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> PositionsType:
    """
    Returns positions which form a range of length `num`, starting from
    `current`, and lying in `[min(current, start), end)`. If
    `current + num >= end`, the range wraps around and consists of two
    parts. The wrap-around starts at `start`.

    Example: `start = 2, current = 3, end = 8, num = 7` results in
    `[3, 4, 5, 6, 7, 2, 3]`.

    Args:
        num: Length of positions, must be `<= end - start`
        current: First entry of positions
        start: Wrap around starts here
        end: End of domain range (not inclusive)
        batch_size: Dimension size for expanding `positions`
        n_query_groups: Dimension size for expanding `positions`
        device: Device used for index returned
        return_tensor: If `True`, returns a `torch.Tensor` even if range has
            one part only

    Returns:
        positions, which is either a tuple `(current, current + num)` if
        `current + num <= end` and `return_tensor=False`, or a position index
        of shape `(batch_size, n_query_groups, num)`, which is broadcast
        from shape `(1, 1, num)`.

    """
    assert start <= current < end, f"current = {current}, must be in [{start}, {end})"
    assert num <= end - start, f"num = {num}, must be <= {end - start}"
    num1 = min(num, end - current)
    diff = num - num1
    if diff == 0 and not return_tensor:
        return current, current + num
    if device is None:
        raise ValueError("device must be given")
    if dtype is None:
        dtype = torch.int64
    kwargs = dict(device=device, dtype=dtype)
    positions = torch.arange(current, current + num1, **kwargs)
    if diff > 0:
        positions = torch.cat((positions, torch.arange(start, start + diff, **kwargs)))
    return index_to_3d(positions, batch_size, n_query_groups)
