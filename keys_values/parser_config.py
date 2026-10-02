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
from pathlib import Path
import sys
from typing import List, Callable, Dict, Any

from litgpt.parser_config import parser_commands as parser_commands_litgpt
from litgpt.utils import CLI

HYPERPARAMETERS_FILENAME = "hyperparameters.yaml"


def parser_commands() -> List[str]:
    return parser_commands_litgpt() + [
        "eval_long",
        "eval_long_ext",
        "finetune_long_full",
        "finetune_long_lora",
        "finetune_offload_full",
        "finetune_offload_lora",
        "recomp_val_losses",
    ]


# From `litgpt.parser_config`. Apart from storing the hyperparameters instead
# of returning them, the function there has a serious side effect: It modifies
# `sys.argv` and does not restore it, which implies that subsequent calls of
# `Fabric.launch` fail.
def _get_hyperparameters_internal(
    function: Callable,
    known_commands: list[str] | None = None,
) -> Any:
    """
    Captures the CLI parameters passed to `function` without running `function`.
    """
    from jsonargparse import capture_parser

    # TODO: Make this more robust
    # This hack strips away the subcommands from the top-level CLI
    # to parse the file as if it was called as a script
    if known_commands is None:
        known_commands = parser_commands()
    _restore = None
    if sys.argv[1] in known_commands:
        _restore = sys.argv.pop(1)
        print(f"\n*** sys.argv after modification:\n{sys.argv}")

    parser = capture_parser(lambda: CLI(function))
    # Restore
    if _restore is not None:
        sys.argv.insert(1, _restore)
    print(f"\n*** sys.argv restored:\n{sys.argv}")
    return parser


def get_hyperparameters_from_parser(
    function: Callable,
    known_commands: list[str] | None = None,
) -> Dict[str, Any]:
    """
    Captures CLI parameters passed to `function` without running `function`.
    These should be stored as hyperparameters alongside a checkpoint.

    """
    parser = _get_hyperparameters_internal(function, known_commands)
    config = parser.parse_args()
    return config.__dict__


def save_hyperparameters(
    function: Callable,
    checkpoint_dir: Path,
    known_commands: list[str] | None = None,
) -> None:
    """
    Use this instead of `litgpt.parser_commands.save_hyperparameters`, the
    latter has serious side effects!

    """
    parser = _get_hyperparameters_internal(function, known_commands)
    config = parser.parse_args()
    parser.save(config, checkpoint_dir / HYPERPARAMETERS_FILENAME, overwrite=True)
