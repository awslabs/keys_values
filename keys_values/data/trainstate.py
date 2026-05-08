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
from typing import Dict

import torch


class DataTrainState:
    """
    Base class for part of training state related to a dataset. This does not
    include the training start part for the data iterator.

    Any random choices made when loading the dataset (which are not already
    fixed in its metadata) need to be included here.

    """

    def state_dict(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        raise NotImplementedError
