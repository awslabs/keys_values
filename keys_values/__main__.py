# Original Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# Modification Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import atexit
import os, sys
import warnings

import torch
from jsonargparse import auto_cli, set_config_read_mode, set_docstring_parse_options

from litgpt.__main__ import PARSER_DATA as PARSER_DATA_LITGPT

from keys_values.finetune.longcontext_eval import setup as eval_long_fn
from keys_values.finetune.longcontext_full import setup as finetune_long_full_fn
from keys_values.finetune.longcontext_lora import setup as finetune_long_lora_fn
from keys_values.finetune.longcon_offload_full import setup as finetune_offload_full_fn
from keys_values.finetune.longcon_offload_lora import setup as finetune_offload_lora_fn
from keys_values.parser_config import parser_commands


ENV_VAR_LOG_DIR = "KEYSVALS_LOG_DIR"

ENV_VAR_LOG_DIR_LEGACY = "VALKEYRIE_LOG_DIR"


PARSER_DATA = {
    **PARSER_DATA_LITGPT,
    "finetune_long_full": finetune_long_full_fn,
    "finetune_long_lora": finetune_long_lora_fn,
    "eval_long": eval_long_fn,
    "finetune_offload_full": finetune_offload_full_fn,
    "finetune_offload_lora": finetune_offload_lora_fn,
}


def _check_commands():
    assert set(parser_commands()) == set(PARSER_DATA.keys()), (
        "PARSER_DATA has to be kept in sync with "
        "keys_values.parser_config.parser_commands().\n\n"
        f"{set(parser_commands())}\n\n"
        f"{set(PARSER_DATA.keys())}"
    )


class TeeOutput:
    """Utility class to duplicate output to both file and stream (stdout/stderr)"""

    def __init__(self, file_obj, stream):
        self.file = file_obj
        self.stream = stream

    def write(self, data):
        """Write data to both file and stream"""
        self.file.write(data)
        self.stream.write(data)

    def flush(self):
        """Ensure both file and stream buffers are flushed"""
        try:
            self.file.flush()
        finally:
            self.stream.flush()

    @property
    def encoding(self):
        """Return stream encoding or default to utf-8"""
        return getattr(self.stream, "encoding", "utf-8")


def _setup_rank_logs(base_directory: str | None = None):
    """Set up logging for distributed training by redirecting stdout/stderr to rank-specific file

    Args:
        base_directory: Root directory for log files. Defaults to './logs' if not specified
    """
    # Get rank information from environment
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    base_directory = os.environ.get(ENV_VAR_LOG_DIR)
    if base_directory is None:
        base_directory = os.environ.get(ENV_VAR_LOG_DIR_LEGACY)
    if base_directory is None:
        base_directory = "./logs"
    os.makedirs(base_directory, exist_ok=True)

    # Configure log file paths
    prefix = f"{base_directory}/gpu{local_rank}"
    log_params = {"buffering": 1, "encoding": "utf-8", "errors": "replace", "mode": "a"}

    # Open single log file for both stdout and stderr
    f_log = open(f"{prefix}.log", **log_params)

    # Store original streams and set up Tee objects
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeOutput(f_log, original_stdout)
    sys.stderr = TeeOutput(f_log, original_stderr)

    def _restore():
        """Cleanup function to restore original streams and close log files"""
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            f_log.close()

    # Register cleanup function to run at exit
    atexit.register(_restore)
    print(f"[rank={local_rank}] logging to {prefix}.log]", flush=True)


def main() -> None:
    _check_commands()
    _setup_rank_logs()
    set_docstring_parse_options(attribute_docstrings=True)
    set_config_read_mode(urls_enabled=True)

    # PyTorch bug that raises a false-positive warning
    # More info: https://github.com/Lightning-AI/litgpt/issues/1561
    warning_message = r"The epoch parameter in `scheduler.step\(\)` was not necessary and is being deprecated.*"

    warnings.filterwarnings(
        action="ignore",
        message=warning_message,
        category=UserWarning,
        module=r".*torch\.optim\.lr_scheduler.*",
    )

    torch.set_float32_matmul_precision("high")
    auto_cli(PARSER_DATA)


if __name__ == "__main__":
    main()
