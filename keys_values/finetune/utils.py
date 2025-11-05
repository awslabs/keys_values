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
from keys_values.model import GPT


LIT_MODEL_FNAME = "lit_model.pth"

HEAD_MODEL_FNAME = "head_model.pth"

LORA_WEIGHTS_FNAME = "lit_model.lora.pth"

LORA_WEIGHTS_FNAME_OLD = "lit_model.pth.lora"


def debug_print_param_names(model: GPT):
    rows = ["", "Names of model (GPT)", ""]
    rows.extend([name for name, _ in model.named_parameters()])
    for i, block in enumerate(model.transformer.h):
        rows.extend(["", f"Names of block {i} (Block)", ""])
        rows.extend([name for name, _ in block.named_parameters()])
    for pname, block in [
        ("lm_head (Linear)", model.lm_head),
        ("wte (Embedding)", model.transformer.wte),
    ]:
        rows.extend(["", f"Names of {pname}", ""])
        rows.extend([name for name, _ in block.named_parameters()])
    print("\n".join(rows))


