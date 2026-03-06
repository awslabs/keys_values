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
import re
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Any

import torch

from keys_values.data import INPUT_IDS_NAME
from keys_values.tools.debug_utils import for_debug, debug_compare_dicts


DebugIntermediatesPredicate = Callable[[str, int, int, int, int, int], bool]


def debug_intermediates_all(
    kind: str,
    block_idx: int,
    start: int,
    end: int,
    rel_start: int,
    rel_end: int,
) -> bool:
    return True


REGEX_FORW_WTE = {
    False: re.compile(r"forward_wte_(\d+):(\d+)$"),
    True: re.compile(r"forward_wte_(\d+):(\d+)"),
}
REGEX_FORW_BLOCK = {
    False: re.compile(r"forward_block(\d+)_(\d+):(\d+)_(\d+):(\d+)$"),
    True: re.compile(r"forward_block(\d+)_(\d+):(\d+)_(\d+):(\d+)"),
}
REGEX_FORW_LOSS = {
    False: re.compile(r"forward_loss_(\d+):(\d+)_(\d+):(\d+)$"),
    True: re.compile(r"forward_loss_(\d+):(\d+)_(\d+):(\d+)"),
}


class DebugIntermediates:
    """
    Maintains a dictionary of tensors, used to store and compare intermediate
    results obtained during a long context inference forward pass. Used mainly
    in :class:`LongContextInferenceModel`, or in components of the model for
    fine-grained intermediates.

    Values are stored only if
    `predicate(kind, block_idx, start, end, rel_start, rel_end)` evaluates to
    `True`. Here, `kind` is "wte", "block", "loss". Arguments which are
    undefined for a `kind` value, should be ignored by the predicate.

    Note: At the moment, the structured and selective variant here is for the
    forward pass only. The backward pass is treated by passing the `entries`
    dictionary down to classes, which will write to it independent of
    `predicate`.

    """

    def __init__(self, predicate: DebugIntermediatesPredicate):
        self._predicate = predicate
        self.entries: Dict[str, torch.Tensor] = dict()

    def clear(self):
        self.entries.clear()

    def should_store_wte(
        self,
        start: int,
        end: int,
    ) -> bool:
        return self._predicate("wte", 0, start, end, 0, 0)

    def store_wte(
        self,
        value: torch.Tensor,
        start: int,
        end: int,
        postfix: Optional[str] = None,
    ):
        args = (start, end)
        if self.should_store_wte(*args):
            name = self.wte_name(*args)
            if postfix is not None:
                name += postfix
            self.entries[name] = for_debug(value)

    @staticmethod
    def wte_name(start: int, end: int) -> str:
        return f"forward_wte_{start}:{end}"

    @staticmethod
    def wte_match(name: str, as_prefix: bool = False) -> Optional[Tuple[int, int]]:
        m = REGEX_FORW_WTE[as_prefix].match(name)
        if m is not None:
            return tuple(int(x) for x in m.groups())
        else:
            return None

    def should_store_block(
        self,
        block_idx: int,
        start: int,
        end: int,
        rel_start: int,
        rel_end: int,
    ):
        return self._predicate("block", block_idx, start, end, rel_start, rel_end)

    def store_block(
        self,
        value: torch.Tensor,
        block_idx: int,
        start: int,
        end: int,
        rel_start: int,
        rel_end: int,
        postfix: Optional[str] = None,
    ):
        args = (block_idx, start, end, rel_start, rel_end)
        if self.should_store_block(*args):
            name = self.block_name(*args)
            if postfix is not None:
                name += postfix
            self.entries[name] = for_debug(value)

    @staticmethod
    def block_name(
        block_idx: int, start: int, end: int, rel_start: int, rel_end: int
    ) -> str:
        return f"forward_block{block_idx}_{start}:{end}_{rel_start}:{rel_end}"

    @staticmethod
    def block_match(
        name: str, as_prefix: bool = False
    ) -> Optional[Tuple[int, int, int, int, int]]:
        m = REGEX_FORW_BLOCK[as_prefix].match(name)
        if m is not None:
            return tuple(int(x) for x in m.groups())
        else:
            return None

    def should_store_loss(
        self,
        start: int,
        end: int,
        rel_start: int,
        rel_end: int,
    ):
        return self._predicate("loss", 0, start, end, rel_start, rel_end)

    def store_loss(
        self,
        value: torch.Tensor,
        start: int,
        end: int,
        rel_start: int,
        rel_end: int,
        postfix: Optional[str] = None,
    ):
        args = (start, end, rel_start, rel_end)
        if self.should_store_loss(*args):
            name = self.loss_name(*args)
            if postfix is not None:
                name += postfix
            self.entries[name] = for_debug(value)

    @staticmethod
    def loss_name(start: int, end: int, rel_start: int, rel_end: int) -> str:
        return f"forward_loss_{start}:{end}_{rel_start}:{rel_end}"

    @staticmethod
    def loss_match(
        name: str, as_prefix: bool = False
    ) -> Optional[Tuple[int, int, int, int]]:
        m = REGEX_FORW_LOSS[as_prefix].match(name)
        if m is not None:
            return tuple(int(x) for x in m.groups())
        else:
            return None


def sort_key_debug_intermediates(name: str) -> Tuple[int, int, int, int]:
    res = DebugIntermediates.wte_match(name, as_prefix=True)
    if res is not None:
        return 0, res[0], 0, 0
    res = DebugIntermediates.block_match(name, as_prefix=True)
    if res is not None:
        return 1, res[0], res[1], res[3]
    res = DebugIntermediates.loss_match(name, as_prefix=True)
    if res is not None:
        return 2, res[0], res[2], 0
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
            (
                "gpt_state_dict",
                comp_state["gpt_state_dict"],
                state["gpt_state_dict"],
                None,
                True,
            ),
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
                (
                    "intermediates",
                    comp_state["intermediates"],
                    state["intermediates"],
                    sort_key_debug_intermediates,
                    False,
                ),
            )
        for name, a_dict, b_dict, sort_key, print_size in dict_tuples:
            print(f"Comparing {name} dictionaries:")
            debug_compare_dicts(
                a_dict,
                b_dict,
                sort_key,
                verbose=True,
                print_size=print_size,
            )
        if do_intermediates:
            print("Comparing loss_values:")
            torch.testing.assert_close(comp_state["loss_values"], state["loss_values"])
