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
import csv
import sys
from enum import unique, Enum

from filelock import FileLock, Timeout
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Iterable, Union, Iterator, Tuple

import torch
from tqdm import tqdm


# Currently, `F.scaled_dot_product_attention` does not properly support the
# case `enabla_gqa=True` (i.e., keys and values have less heads than
# queries). In this case, it is best to extend keys and values, which requires
# extra memory, but allows for efficient kernels to be used.
# Once PyTorch supports `enabla_gqa=True` properly at least with some fused
# kernels (such as flash attention), this flag can be switched to `False`.
FUSED_SDPA_DOES_NOT_SUPPORT_ENABLE_GQA = True


def _append_results_to_csv(
    results: List[Dict[str, Any]],
    result_path: Path,
) -> bool:
    lock_path = result_path.with_suffix(".lock")
    lock = FileLock(lock_path, timeout=1)
    try:
        with lock.acquire(timeout=1):
            fieldnames = sorted(results[0].keys())
            mode = "a" if result_path.exists() else "w"
            with result_path.open(mode) as fp:
                writer = csv.writer(fp, delimiter=",")
                if mode == "w":
                    writer.writerow(fieldnames)
                for record in results:
                    row = [record[name] for name in fieldnames]
                    writer.writerow(row)
    except Timeout:
        return False
    finally:
        lock.release()
        lock_path.unlink()
        return True


def append_results_to_csv(
    results: List[Dict[str, Any]],
    result_path: Path,
    num_retrials: int = 100,
    sleep_time: float = 0.1,
):
    for _ in range(num_retrials):
        if _append_results_to_csv(results, result_path):
            break
        time.sleep(sleep_time)


def expand_index(index: torch.Tensor, head_size: int) -> torch.Tensor:
    assert index.ndim == 3
    return index.unsqueeze(-1).expand(-1, -1, -1, head_size)


def index_to_3d(index: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    assert index.ndim == 1
    return index.view(1, 1, -1).expand(dim0, dim1, -1)


def is_index_1d(index: torch.Tensor) -> bool:
    ndim = index.ndim
    return index.stride() == (0,) * (ndim - 1) + (1,)


def need_repeat_interleave(n_head: int, n_query_groups: int) -> bool:
    return n_query_groups < n_head and FUSED_SDPA_DOES_NOT_SUPPORT_ENABLE_GQA


def repeat_interleave(x: torch.Tensor, n_head: int) -> torch.Tensor:
    n_query_groups = x.shape[1]
    if need_repeat_interleave(n_head, n_query_groups):
        q_per_kv = n_head // n_query_groups
        assert n_head == n_query_groups * q_per_kv
        x = torch.repeat_interleave(x, q_per_kv, dim=1)
    return x


def copy_parameters(
    from_model: torch.nn.Module,
    to_model: torch.nn.Module,
    copy_requires_grad: bool = True,
):
    """
    Copies parameter values from `from_model` to `to_model`.

    Args:
        from_model (torch.nn.Module): Source model
        to_model (torch.nn.Module): Target model
        copy_requires_grad (bool): Should `param.requires_grad` be copied as well?
            Defaults to `True`.

    """
    # Note: Don't use `from_model.state_dict`, this does not retain the
    # `requires_grad` flag!
    for name, param in to_model.named_parameters():
        if param is not None:
            src_param = from_model.get_parameter(name)
            param.data.copy_(src_param.data, non_blocking=True)
            if copy_requires_grad:
                param.requires_grad_(src_param.requires_grad)


def flush_io_streams():
    sys.stdout.flush()
    sys.stderr.flush()


def randint_torch(a: int, b: int) -> int:
    return torch.randint(a, b + 1, (1,)).item()


def check_for_nan(
    x: torch.Tensor,
    meth_name: str,
    key_name: str,
    extra_txt: Optional[str] = None,
    do_boom: bool = False,
) -> int:
    x = x.detach()
    num_nan = torch.isnan(x).sum().item()
    if num_nan > 0:
        if extra_txt is not None:
            extra_txt = ": " + extra_txt
        else:
            extra_txt = ""
        print(
            f"From {meth_name}: {key_name} has {num_nan} NaNs [shape={x.shape}, numel={x.numel()}]"
            + extra_txt
        )
        if do_boom:
            raise AssertionError("BOOM")
    return num_nan


def check_for_nan_module_weights(
    module: torch.nn.Module,
    do_grads: bool = False,
    extra_msg: Optional[str] = None,
    do_boom: bool = False,
):
    is_boom = False
    for name, param in module.named_parameters():
        if param is not None:
            if (
                check_for_nan(
                    param.data,
                    "check_for_nan_model_weights",
                    name,
                    extra_msg,
                    do_boom=False,
                )
                > 0
            ):
                is_boom = True
            if do_grads and param.grad is not None:
                if (
                    check_for_nan(
                        param.grad.data,
                        "check_for_nan_model_weights",
                        name + ".grad",
                        extra_msg,
                        do_boom=False,
                    )
                    > 0
                ):
                    is_boom = True
    if do_boom and is_boom:
        raise AssertionError("BOOM")


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


def message_with_device_memory(device: torch.device) -> str:
    free, total = torch.cuda.mem_get_info(device)
    used_in_gb = (total - free) / (1024**3)
    free_in_gb = free / (1024**3)
    return f"Memory on {device}: Used {used_in_gb:.3f} GB, Free {free_in_gb:.3f} GB"


def message_memory_all_devices() -> str:
    num_devices = torch.cuda.device_count()
    assert num_devices > 0, "There are no CUDA devices"
    lines = [
        message_with_device_memory(torch.device("cuda", i)) for i in range(num_devices)
    ]
    return "\n".join(lines)


def log_memory_all_devices() -> Dict[str, float]:
    num_devices = torch.cuda.device_count()
    result = dict()
    for i in range(num_devices):
        device = torch.device("cuda", i)
        free, total = torch.cuda.mem_get_info(device)
        used_in_gb = (total - free) / (1024**3)
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
