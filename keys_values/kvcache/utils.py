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
from enum import unique, Enum
import gc
from typing import Iterable, Optional, Iterator, Union, List, Tuple, Dict

import torch
from tqdm import tqdm


@unique
class VerbosityLevels(str, Enum):
    NONE = "none"
    SOME = "some"
    MORE = "more"
    ALL = "all"


def wrap_tqdm_if_verbose(
    iterator: Iterable,
    verbose: VerbosityLevels,
    total: Optional[int] = None,
) -> Union[Iterable, Iterator]:
    if verbose is VerbosityLevels.NONE:
        return iterator
    if isinstance(iterator, Iterator):
        return tqdm(iterator, total=total)
    else:
        return tqdm(iterator)


def expand_index(index: torch.Tensor, head_size: int) -> torch.Tensor:
    assert index.ndim == 3
    return index.unsqueeze(-1).expand(-1, -1, -1, head_size)


_PRECISION_TO_DTYPE = {
    "16-true": torch.float16,
    "16-mixed": torch.float16,
    "bf16-true": torch.bfloat16,
    "bf16-mixed": torch.bfloat16,
    "32-true": torch.float32,
}

_PRECISION_NOT_SUPPORTED = (
    "transformer-engine",
    "transformer-engine-float16",
    "64-true",
)


def fabric_precision_to_dtype(precision: str) -> torch.dtype:
    result = _PRECISION_TO_DTYPE.get(precision)
    if result is None:
        if precision in _PRECISION_NOT_SUPPORTED:
            raise ValueError(f"Precision {precision} not yet supported")
        else:
            raise ValueError(f"Precision {precision} is not valid")
    return result


def map_model_weights_from_precision(
    model: torch.nn.Module,
    precision: str,
) -> torch.nn.Module:
    result = _PRECISION_TO_DTYPE.get(precision)
    if result is None:
        return model
    elif result == torch.float16:
        return model.half()
    elif result == torch.bfloat16:
        return model.bfloat16()
    elif result == torch.float32:
        return model.float()


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
    )[:(max_num_ranges - 1)]
    holes = [(None, mn)] + sorted(
        [(a, a + l) for a, l in diffs], key=lambda x: x[0],
    ) + [(mx, None)]
    result = [
        (hole1[1], hole2[0] + 1)
        for hole1, hole2 in zip(holes[:-1], holes[1:])
    ]
    return result


def message_with_device_memory(device: torch.device) -> str:
    free, total = torch.cuda.mem_get_info(device)
    used_in_gb = (total - free) / (1024 ** 3)
    free_in_gb = free / (1024 ** 3)
    return f"Memory on {device}: Used {used_in_gb:.3f} GB, Free {free_in_gb:.3f} GB"


def message_memory_all_devices() -> str:
    num_devices = torch.cuda.device_count()
    assert num_devices > 0, "There are no CUDA devices"
    lines = [
        message_with_device_memory(torch.device("cuda", i))
        for i in range(num_devices)
    ]
    return "\n".join(lines)


def log_memory_all_devices() -> Dict[str, float]:
    num_devices = torch.cuda.device_count()
    result = dict()
    for i in range(num_devices):
        device = torch.device("cuda", i)
        free, total = torch.cuda.mem_get_info(device)
        used_in_gb = (total - free) / (1024 ** 3)
        result[f"memory_cuda{i}"] = used_in_gb
    return result


def bytes_for_torch_dtype(dtype: torch.dtype) -> int:
    """
    Args:
        dtype: Torch data type

    Returns:
        Number of bytes used to represent one number of this type.

    """
    return torch.tensor([], dtype=dtype).element_size()


def bits_for_torch_dtype(dtype: torch.dtype) -> int:
    """
    Args:
        dtype: Torch data type

    Returns:
        Number of bits used to represent one number of this type.

    """
    return bytes_for_torch_dtype(dtype) * 8


def bitsize_of(x: torch.Tensor) -> int:
    return x.numel() * x.element_size() * 8


def shape_to_tuple(x: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(d) for d in x.shape)


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
    return x.storage().data_ptr()
