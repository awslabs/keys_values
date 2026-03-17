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
from math import ceil, floor
from typing import Dict, Optional, Callable, Any, List

import torch


def for_debug(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device=torch.device("cpu")).clone()


QUANTILES = [0.01, 0.5, 0.9, 0.99]


def torch_quantile(  # noqa: PLR0913 (too many arguments)
    tensor: torch.Tensor,
    q: float,
    dim: int | None = None,
    *,
    keepdim: bool = False,
    interpolation: str = "linear",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Improved ``torch.quantile`` for one scalar quantile.

    Taken from:
    https://github.com/pytorch/pytorch/issues/157431#issuecomment-3026856373

    Arguments
    ---------
    tensor: ``Tensor``
        See ``torch.quantile``.
    q: ``float``
        See ``torch.quantile``. Supports only scalar values currently.
    dim: ``int``, optional
        See ``torch.quantile``.
    keepdim: ``bool``
        See ``torch.quantile``. Supports only ``False`` currently.
        Defaults to ``False``.
    interpolation: ``{"linear", "lower", "higher", "midpoint", "nearest"}``
        See ``torch.quantile``. Defaults to ``"linear"``.
    out: ``Tensor``, optional
        See ``torch.quantile``. Currently not supported.

    Notes
    -----
    Uses ``torch.kthvalue``. Better than ``torch.quantile`` since:

    #. it has no :math:`2^{24}` tensor `size limit <https://github.com/pytorch/pytorch/issues/64947#issuecomment-2304371451>`_;
    #. it is much faster, at least on big tensor sizes.

    """
    # Sanitization of: q
    q_float = float(q)  # May raise an (unpredictible) error
    if not 0 <= q_float <= 1:
        msg = f"Only values 0<=q<=1 are supported (got {q_float!r})"
        raise ValueError(msg)

    # Sanitization of: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        tensor = tensor.reshape((-1, *(1,) * (tensor.ndim - 1)))

    # Sanitization of: inteporlation
    idx_float = q_float * (tensor.shape[dim] - 1)
    if interpolation == "nearest":
        idxs = [round(idx_float)]
    elif interpolation == "lower":
        idxs = [floor(idx_float)]
    elif interpolation == "higher":
        idxs = [ceil(idx_float)]
    elif interpolation in {"linear", "midpoint"}:
        low = floor(idx_float)
        idxs = [low] if idx_float == low else [low, low + 1]
        weight = idx_float - low if interpolation == "linear" else 0.5
    else:
        msg = (
            "Currently supported interpolations are {'linear', 'lower', 'higher', "
            f"'midpoint', 'nearest'}} (got {interpolation!r})"
        )
        raise ValueError(msg)

    # Sanitization of: out
    if out is not None:
        msg = f"Only None value is currently supported for out (got {out!r})"
        raise ValueError(msg)

    # Logic
    outs = [torch.kthvalue(tensor, idx + 1, dim, keepdim=True)[0] for idx in idxs]
    out = outs[0] if len(outs) == 1 else outs[0].lerp(outs[1], weight)

    # Rectification of: keepdim
    if keepdim:
        return out
    return out.squeeze() if dim_was_none else out.squeeze(dim)


def size_quantiles_internal(
    x: torch.Tensor,
    quantiles: List[float],
) -> List[float]:
    abs_x = x.detach().flatten().abs()
    return [torch_quantile(abs_x, q=q).item() for q in quantiles]


def size_quantiles(x: torch.Tensor) -> str:
    qvals_x = size_quantiles_internal(x, QUANTILES)
    return "|".join([f"{k:.2f}:{v:.2e}" for k, v in zip(QUANTILES, qvals_x)])


MAX_NUM_CATCHES = 50


def debug_compare_dicts(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    sort_key: Optional[Callable[[str], Any]] = None,
    verbose: bool = False,
    print_size: bool = False,
):
    a_names = set(a.keys())
    b_names = set(b.keys())
    diff_ab = a_names - b_names
    if diff_ab:
        raise ValueError(f"Names in A, not in B: {diff_ab}")
    diff_ba = b_names - a_names
    if diff_ba:
        raise ValueError(f"Names in B, not in A: {diff_ba}")
    exc_caught = []
    rethrow_ex = None
    if a_names:
        if print_size:
            max_name_len = max(len(name) for name in a_names)
            prefix = "    {name:" + str(max_name_len + 1) + "} "
        else:
            prefix = None
        for name in sorted(a_names, key=sort_key):
            a_val = a[name]
            b_val = b[name]
            try:
                if print_size:
                    print(prefix.format(name=name + ":") + size_quantiles(a_val))
                elif verbose:
                    print("    " + name)
                torch.testing.assert_close(a_val, b_val)
            except AssertionError as e:
                rethrow_ex = e
                exc_caught.append((name, str(e)))
                if len(exc_caught) >= MAX_NUM_CATCHES:
                    break
        if exc_caught:
            print(f"\nCaught {len(exc_caught)} exceptions:")
            for name, msg in exc_caught:
                print(f"\n    [{name}]\n{msg}")
            raise rethrow_ex
