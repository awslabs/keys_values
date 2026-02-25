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
from dataclasses import dataclass
from functools import partial
from typing import Type

import torch

from litgpt.config import Config as _Config


@dataclass
class Config(_Config):
    own_rms_norm_implementation: bool = False

    @property
    def norm_class(self) -> Type:
        if self.norm_class_name == "RMSNorm":
            from keys_values.model import RMSNorm as RMSNormNew
            from litgpt.model import RMSNorm as RMSNormOld

            norm_type = RMSNormNew if self.own_rms_norm_implementation else RMSNormOld
            return partial(
                norm_type,
                add_unit_offset="Gemma" in self.name,
            )
        if self.norm_class_name == "LayerNorm" and "OLMo" in self.name:
            # this makes it equivalent to `torch.nn.functional.layer_norm`
            # that is used by OLMo
            # Table 5 caption in the OLMo paper shows this - https://aclanthology.org/2024.acl-long.841
            return partial(torch.nn.LayerNorm, elementwise_affine=False)
        return getattr(torch.nn, self.norm_class_name)
