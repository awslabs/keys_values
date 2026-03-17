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
from typing import Tuple

from keys_values.kvcache.quantize import (
    TorchBasicQuantizer,
    BitsAndBytesQuantizer,
)

SUPPORTED_QUANTIZERS = {
    "default": None,
    "torch-quantized4": TorchBasicQuantizer,
    "torch-quantized8": TorchBasicQuantizer,
    "bnb-quantized4": BitsAndBytesQuantizer,
    "bnb-quantized8": BitsAndBytesQuantizer,
}


def split_name(name: str) -> Tuple[str, str]:
    for qname in SUPPORTED_QUANTIZERS.keys():
        if name.endswith(qname):
            return name[: -(len(qname) + 1)], qname
    raise ValueError(f"Name {name} is not supported")
