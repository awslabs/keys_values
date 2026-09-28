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
from jsonargparse import ArgumentParser, capture_parser, Namespace
from pathlib import Path
import sys
from typing import List, Callable, Tuple

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
def capture_parser_from_script(
    function: Callable,
    known_commands: list[str] | None = None,
) -> Tuple[ArgumentParser, Namespace]:
    """
    Captures the CLI parameters passed to `function` without running `function`.

    """
    # TODO: Make this more robust
    # This hack strips away the subcommands from the top-level CLI
    # to parse the file as if it was called as a script
    if known_commands is None:
        known_commands = parser_commands()
    _restore = None
    if sys.argv[1] in known_commands:
        _restore = sys.argv.pop(1)

    parser = capture_parser(lambda: CLI(function))
    config = parser.parse_args()
    # Restore
    if _restore is not None:
        sys.argv.insert(1, _restore)

    return parser, config


def save_hyperparameters(
    parser: ArgumentParser,
    config: Namespace,
    checkpoint_dir: Path,
) -> None:
    """
    Use this instead of `litgpt.parser_commands.save_hyperparameters`, the
    latter has serious side effects!

    """
    parser.save(config, checkpoint_dir / HYPERPARAMETERS_FILENAME, overwrite=True)
