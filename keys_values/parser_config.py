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
from typing import List, Optional
from pathlib import Path

from litgpt.parser_config import (
    parser_commands as parser_commands_litgpt,
    save_hyperparameters as save_hyperparameters_litgpt,
)


def parser_commands() -> List[str]:
    return parser_commands_litgpt() + [
        "finetune_long_full",
        "finetune_long_lora",
        "eval_long",
        "finetune_offload_full",
        "finetune_offload_lora",
    ]


def save_hyperparameters(
    function: callable,
    checkpoint_dir: Path,
    known_commands: Optional[List[str]] = None,
) -> None:
    if known_commands is None:
        known_commands = parser_commands()
    save_hyperparameters_litgpt(function, checkpoint_dir, known_commands)
