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
from typing import Tuple, Optional

from litgpt.config import Config
from litgpt.data import DataModule

from keys_values.data import LongBenchV2
from keys_values.head_model import (
    HeadModel,
    CrossEntropyOnLogits,
    SequenceClassificationOnLogits,
    SequenceClassification,
)


SUPPORTED_HEAD_MODELS = {
    CrossEntropyOnLogits.NAME: CrossEntropyOnLogits,
    SequenceClassificationOnLogits.NAME: SequenceClassificationOnLogits,
    SequenceClassification.NAME: SequenceClassification,
}


class HeadModelFactory:
    @staticmethod
    def create(
        name: str,
        config: Config,
        data: Optional[DataModule] = None,
        **kwargs,
    ) -> HeadModel:
        """
        Args:
            name: Name of head model to create, see :const:`SUPPORTED_HEAD_MODELS`
            config: Config object for backbone model
            data: :class:`DataModule` object. Optional
            kwargs: Extra arguments passed to head model constructor

        Returns:
            Head model object

        """
        model_cls = SUPPORTED_HEAD_MODELS.get(name)
        if model_cls is None:
            raise ValueError(
                f"name = {name} not supported, must be in {HeadModelFactory.supported_names()}"
            )
        head_kwargs = dict()
        if data is not None and isinstance(data, LongBenchV2):
            head_kwargs = data.head_model_kwargs(name)
        return model_cls(config, **head_kwargs, **kwargs)

    @staticmethod
    def supported_names() -> Tuple[str, ...]:
        return tuple(SUPPORTED_HEAD_MODELS.keys())
