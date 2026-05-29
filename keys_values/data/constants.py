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
from typing import Tuple, List, Callable, Any, Dict

ResultType = Tuple[List[int], int]

Collator = Callable[[List[Dict[str, Any]]], Dict[str, Any]]

RawDatasetType = List[Dict[str, str]]

CollateFnType = Callable[[List[Dict[str, Any]]], Dict[str, Any]]

METADATA_SEQ_LENGTHS_KEY = "sequence_lengths"

METADATA_KEYS = {METADATA_SEQ_LENGTHS_KEY}

NUM_TOKENS_NAME = "num_tokens_instruction"

INPUT_IDS_NAME = "input_ids"

LABELS_NAME = "labels"

ORIG_IDX_NAME = "orig_idx"

TASK_NAME = "task"

TARGETS_STRINGS_NAME = "targets_as_strings"

POSITION_NAME = "position"

LIT_MODEL_FNAME = "lit_model.pth"

HEAD_MODEL_FNAME = "head_model.pth"

LORA_WEIGHTS_FNAME = "lit_model.lora.pth"

LORA_WEIGHTS_FNAME_OLD = "lit_model.pth.lora"
