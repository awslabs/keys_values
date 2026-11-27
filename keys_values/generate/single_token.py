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

import torch

from keys_values.long_context import LongContextInferenceModel


class SingleTokenLongContextInference:
    """
    Base class for single token inference for a long-context model
    :class:`LongContextInferenceModel`. Supports prompt processing (any
    length) as well as inference for a single additional token.

    Usage:
    - Process prompt by :meth:`process_prompt`
    - Generate tokens by :func:`batched_generate_fn`. This is iterating
      between calling :meth:`inference_for_token` and sampling a new batch
      of single tokens.

    NOTE: Use `model.gpt_model.max_seq_length = X` in order to adjust the
    sequence length (where `X = prompt.shape[-1] + max_returned_tokens`), ideally
    before calling :meth:`process_prompt`. This is important for position
    encoding.
    """

    @property
    def model(self) -> LongContextInferenceModel:
        raise NotImplementedError

    def process_prompt(self, prompt: torch.Tensor) -> torch.Tensor:
        """
        Process prompt by running a forward pass on the model. Afterwards,
        single token inference can be run with :meth:`inference_for_token`.

        Args:
            prompt: Inputs to be processes, `(batch_size, input_length)`

        Returns:
            Logits for final token, `(batch_size, 1,  padded_vocab_size)`

        """
        raise NotImplementedError

    def inference_for_token(self, token_idx: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass for single additional token.

        Args:
            token_idx: New token, `(batch_size, 1)` or `(batch_size,)`

        Returns:
            Logits for new token, `(batch_size, 1,  padded_vocab_size)`

        """
        raise NotImplementedError


class DefaultSingleTokenInference(SingleTokenLongContextInference):
    """
    Default implementation. Each :meth:`inference_for_token` call runs the
    GPT model forward. This can be slow, because the forward computation is
    not a compiled graph, and the KV caches decisions are done for each
    single token.
    """

    def __init__(self, model: LongContextInferenceModel):
        self._model = model

    @property
    def model(self) -> LongContextInferenceModel:
        return self._model

    def process_prompt(self, prompt: torch.Tensor) -> torch.Tensor:
        return self._model(prompt, targets=None)

    def inference_for_token(self, token_idx: torch.Tensor) -> torch.Tensor:
        return self._model.gpt_model(token_idx)
