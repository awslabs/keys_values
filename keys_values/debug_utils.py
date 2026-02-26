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
import re
from pathlib import Path
from typing import Dict, Optional, Callable, Any, Tuple

import torch

from keys_values.data import INPUT_IDS_NAME


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


def size_quantiles(x: torch.Tensor) -> str:
    abs_x = x.detach().to(device=torch.device("cpu"), dtype=torch.float32).flatten().abs()
    qvals_x = [torch_quantile(abs_x, q=q).item() for q in QUANTILES]
    return "|".join([f"{k:.2f}:{v:.2e}" for k, v in zip(QUANTILES, qvals_x)])


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
    if print_size:
        max_name_len = max(len(name) for name in a_names)
        prefix = "    {name:" + str(max_name_len) + "}: "
    else:
        prefix = None
    for name in sorted(a_names, key=sort_key):
        a_val = a[name]
        b_val = b[name]
        try:
            if print_size:
                print(prefix.format(name=name) + size_quantiles(a_val))
            elif verbose:
                print("    " + name)
            torch.testing.assert_close(a_val, b_val)
        except AssertionError as e:
            print(f"Significant differences A vs B for {name}")
            raise e


# Keys in debug_intermediates are:
#   forward_wte_{start}:{end}
#   forward_block{block_idx}_{start}:{end}_{rel_start}:{rel_end}
#   forward_loss_{start}:{end}_{rel_start}:{rel_end}

REGEX_FORW_WTE = re.compile(r"forward_wte_(\d+):\d+$")

REGEX_FORW_BLOCK = re.compile(r"forward_block(\d+)_(\d+):\d+_(\d+):\d+$")

REGEX_FORW_LOSS = re.compile(r"forward_loss_(\d+):\d+_(\d+):\d+$")


def sort_key_debug_intermediates(name: str) -> Tuple[int, int, int, int]:
    m = REGEX_FORW_WTE.match(name)
    if m:
        return 0, int(m.group(1)), 0, 0
    m = REGEX_FORW_BLOCK.match(name)
    if m:
        return 1, int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = REGEX_FORW_LOSS.match(name)
    if m:
        return 2, int(m.group(1)), int(m.group(2)), 0
    raise ValueError(f"Invalid name: {name}")


def debug_store_or_compare_state(
    batch: Dict[str, Any],
    loss_values: torch.Tensor,
    gpt_state_dict: Dict[str, torch.Tensor],
    head_state_dict: Dict[str, torch.Tensor],
    eval_metrics_path: Path,
    debug_intermediates: Optional[Dict[str, torch.Tensor]] = None,
):
    state_path = eval_metrics_path.parent / (eval_metrics_path.stem + ".pth")
    do_compare = state_path.exists()
    cpu_device = torch.device("cpu")
    state = {
        INPUT_IDS_NAME: batch[INPUT_IDS_NAME].to(cpu_device),
        "targets": batch["targets"].to(cpu_device),
        "gpt_state_dict": gpt_state_dict,
        "head_state_dict": head_state_dict,
        "loss_values": loss_values.to(cpu_device),
    }
    if debug_intermediates is not None:
        state["intermediates"] = debug_intermediates
    if not do_compare:
        print(f"DEBUG: Storing state to {state_path}")
        if "intermediates" in state:
            print("       [also intermediates]")
        torch.save(state, state_path)
    else:
        print(f"DEBUG: Loading A state from {state_path}")
        comp_state = torch.load(state_path, map_location=cpu_device)
        for name in (
            INPUT_IDS_NAME,
            "targets",
            "gpt_state_dict",
            "head_state_dict",
            "loss_values",
        ):
            if name not in comp_state:
                raise IndexError(f"A state does not contain {name} field")
        do_intermediates = (
            debug_intermediates is not None and "intermediates" in comp_state
        )
        if debug_intermediates is not None and not do_intermediates:
            print(
                "DEBUG: WARNING: State A does not contain debug_intermediates, so cannot compare them"
            )
        print("DEBUG: Comparing A state against current state (B):")
        a_data = {
            INPUT_IDS_NAME: comp_state[INPUT_IDS_NAME],
            "targets": comp_state["targets"],
        }
        b_data = {
            INPUT_IDS_NAME: state[INPUT_IDS_NAME],
            "targets": state["targets"],
        }
        dict_tuples = [
            ("data_batch", a_data, b_data, None, False),
            ("gpt_state_dict", comp_state["gpt_state_dict"], state["gpt_state_dict"], None, True),
            (
                "head_state_dict",
                comp_state["head_state_dict"],
                state["head_state_dict"],
                None,
                True,
            ),
        ]
        if do_intermediates:
            dict_tuples.append(
                ("intermediates", comp_state["intermediates"], state["intermediates"], sort_key_debug_intermediates, False),
            )
        for name, a_dict, b_dict, sort_key, print_size in dict_tuples:
            print(f"Comparing {name} dictionaries:")
            debug_compare_dicts(
                a_dict, b_dict, sort_key, verbose=True, print_size=print_size,
            )
        if do_intermediates:
            print("Comparing loss_values:")
            torch.testing.assert_close(comp_state["loss_values"], state["loss_values"])
