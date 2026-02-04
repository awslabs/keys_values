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
from typing import Optional, Union, List

import torch
import torch.nn.functional as F

from litgpt.config import Config

from keys_values.utils import copy_parameters


class HeadModel(torch.nn.Module):
    """
    Abstraction for head model and loss function. Can contain trainable
    weights.

    Supports computation in chunks, see :meth:`forward`.

    """

    def __init__(self):
        super().__init__()

    def needs_logits(self) -> bool:
        """
        Returns:
            If `True`, inputs are logits of the transformer model (after
            `GPT.lm_head`). If false, inputs are final layer outputs (before
            `GPT.lm_head`).
        """
        raise NotImplementedError()

    def forward(
        self,
        model_outputs: torch.Tensor,
        targets: Optional[torch.Tensor],
        input_pos: int,
    ) -> torch.Tensor:
        """
        This is called sequentially over chunks, from left to right, and
        `input_pos=0` starts a new batch.

        Args:
            model_outputs: Logits over vocabulary if `needs_logits` returns
                `True`, otherwise final layer outputs
            targets: Corresponding targets. If shorter than `model_outputs`,
                they align with `model_outputs` on the right. If `None`, we
                just process `model_outputs` and return 0.
            input_pos: Token position of first entry in `model_outputs` along
                the sequence. If `input_pos == 0`, this starts a new loss
                computation.

        Returns:
            Loss function values, one entry per batch dimension. If the loss
            is normalized over number of targets, the value just considers
            the length of `targets`. If loss values over chunks are added
            up, the appropriate weighting needs to be applied.

        """
        raise NotImplementedError()

    def num_target_entries(self, targets: Optional[torch.Tensor]) -> Optional[int]:
        """
        This is used in order to ensure correct combination of the loss value
        parts returned by :meth:`forward` over all chunks. Namely, if
        `loss[j] = forward(chunk[j])`, `num[j] = num_target_entries(chunk[j])` over
        all chunks `j`, the combined loss value is
        `sum(num * loss) / sum(num)`.

        For some loss functions, the loss parts are simply summed up, in which
        case this method returns `None` for all chunks.

        """
        raise NotImplementedError()

    @staticmethod
    def _check_model_outputs_targets(
        model_outputs: torch.Tensor,
        targets: Optional[torch.Tensor],
        final_dim: int,
    ) -> Optional[int]:
        if model_outputs.ndim != 3 or model_outputs.shape[-1] != final_dim:
            raise ValueError(
                f"model_outputs.shape = {model_outputs.shape}, must be 3D and final size must be {final_dim}"
            )
        if targets is None:
            return None
        if targets.ndim != 2 or targets.shape[0] != model_outputs.shape[0]:
            raise ValueError(
                f"model_outputs.shape = {model_outputs.shape}, targets.shape = {targets.shape}: Not compatible"
            )
        diff = model_outputs.shape[-2] - targets.shape[-1]
        if diff < 0:
            raise ValueError(
                f"model_outputs.length = {model_outputs.shape[-2]}, must not be smaller than targets.length = {targets.shape[-1]}"
            )
        return diff

    def _empty_clone(self, device: Optional[torch.device] = None) -> "HeadModel":
        """
        Creates copy of this object on the same device. Parameters are not
        copied.

        """
        raise NotImplementedError()

    def clone(self, device: Optional[torch.device] = None) -> "HeadModel":
        """
        Creates and returns a copy of this object, situated on device `device`.
        All named parameter tensors are copied.

        Args:
            device: Device on which the copy is created.

        Returns:
            Copy of this object on device `device`

        """
        # Create empty copy
        model_copy = self._empty_clone(device)
        copy_parameters(self, model_copy)
        return model_copy


class CrossEntropyOnLogits(HeadModel):
    """
    Default loss function for instruction tuning with GPT model, namely
    cross entropy on the logits, with mean reduction.

    Say your data is a token sequence `tokens` of length `n = n1 + n2`, so that
    `tokens[:, :n1]` are prompts, `tokens[:, n1:]` are desired responses. The
    respective `n1, n2` are the same across the batch: use padding tokens at
    left and right to make this work.

    This translates into `input_ids = tokens[:, :-1]` and `targets =
    tokens[:, n1:]`. This means that `input_ids[:, -1] = tokens[:, -2]` is
    paired with `targets[:, -1] = tokens[:, -1]`, etc., so that next token
    prediction is enforced over the response range.

    More general masking:

    This class also supports instruction data with more general masking, such
    as several switches between prompt and desired targets. To this end, just
    use `ignore_token` in `targets` for outputs to be masked out in the loss
    function. For example, teaching a model to use tools, we can use data cases
    coming from trajectories with several tool calls. Apart from the initial
    prompt, we also mask out the tool outputs in `targets`.

    """

    NAME = "next_token_prediction"

    def __init__(
        self,
        config: Config,
        ignore_index: int = -100,
    ):
        super().__init__()
        self._vocab_size = config.padded_vocab_size
        self._ignore_index = ignore_index

    def needs_logits(self) -> bool:
        return True

    def forward(
        self,
        model_outputs: torch.Tensor,
        targets: Optional[torch.Tensor],
        input_pos: int,
    ) -> torch.Tensor:
        diff = self._check_model_outputs_targets(
            model_outputs,
            targets,
            final_dim=self._vocab_size,
        )
        if diff is not None:
            logits = model_outputs[:, diff:, :]
            losses = F.cross_entropy(
                logits.reshape(-1, self._vocab_size),
                targets.reshape(-1),
                ignore_index=self._ignore_index,
                reduction="none",
            )
            return losses.view(*logits.shape[:2]).mean(dim=-1)
        else:
            return torch.zeros(
                model_outputs.shape[0],
                device=model_outputs.device,
                dtype=model_outputs.dtype,
            )

    def num_target_entries(self, targets: Optional[torch.Tensor]) -> Optional[int]:
        """
        This is the sum of entries not equal to `_ignore_index`.

        """
        return (
            0 if targets is None else int(targets.ne(self._ignore_index).sum().item())
        )

    def _empty_clone(self, device: Optional[torch.device] = None) -> "HeadModel":
        config = Config()
        config.padded_vocab_size = self._vocab_size
        if device is None:
            model_copy = CrossEntropyOnLogits(
                config,
                ignore_index=self._ignore_index,
            )
        else:
            with torch.device(device):
                model_copy = CrossEntropyOnLogits(
                    config,
                    ignore_index=self._ignore_index,
                )
        return model_copy


class SequenceClassificationOnLogits(HeadModel):
    """
    Variant of :class:`CrossEntropyOnLogits` for sequence classification,
    where the class labels are single tokens. Here, `targets` is a sequence
    of length 1, with values in `[0, num_classes)`, where
    `num_classes == len(class_label_tokens)`. Here, token value
    `class_label_tokens[j]` corresponds to target value `j`.

    The loss function is cross-entropy on the logits for token values
    `class_label_tokens`. In other words, the distribution over the
    vocabulary predicted for the final sequence position is reduced to the
    tokens in `class_label_tokens` and taken as multi-class probability.

    Compared to :class:`SequenceClassification`, this does not need a
    separate linear head to be trained. On the other hand, predictions are
    more computationally wasteful, because the logits over the whole
    vocabulary are determined.

    """

    NAME = "seq_classification_on_logits"

    def __init__(
        self,
        config: Config,
        class_label_tokens: List[int],
    ):
        super().__init__()
        self._vocab_size = config.padded_vocab_size
        self.num_classes = len(class_label_tokens)
        if self.num_classes < 2:
            raise ValueError(
                f"class_label_tokens = {class_label_tokens} must have at least 2 elements"
            )
        if not all(
            x == int(x) and 0 <= x < self._vocab_size for x in class_label_tokens
        ):
            raise ValueError(
                f"class_label_tokens = {class_label_tokens} must be in [0, {self._vocab_size})"
            )
        if len(set(class_label_tokens)) != self.num_classes:
            raise ValueError(
                f"class_label_tokens = {class_label_tokens} must not have duplicates"
            )
        self.class_label_tokens = tuple(class_label_tokens)

    def needs_logits(self) -> bool:
        return True

    def forward(
        self,
        model_outputs: torch.Tensor,
        targets: Optional[torch.Tensor],
        input_pos: int,
    ) -> torch.Tensor:
        if targets is None:
            return torch.zeros(
                model_outputs.shape[0],
                device=model_outputs.device,
                dtype=model_outputs.dtype,
            )
        if targets.shape[-1] != 1:
            raise ValueError(f"targets.length = {targets.shape[-1]}, must be 1")
        diff = self._check_model_outputs_targets(
            model_outputs,
            targets,
            final_dim=self._vocab_size,
        )
        selected_logits = model_outputs[:, diff:, self.class_label_tokens]
        losses = F.cross_entropy(
            selected_logits.reshape(-1, self.num_classes),
            targets.reshape(-1),
            reduction="none",
        )
        return losses.view(*selected_logits.shape[:2]).mean(dim=-1)

    def num_target_entries(self, targets: Optional[torch.Tensor]) -> Optional[int]:
        return None

    def _empty_clone(self, device: Optional[torch.device] = None) -> "HeadModel":
        config = Config()
        config.padded_vocab_size = self._vocab_size
        if device is None:
            model_copy = SequenceClassificationOnLogits(
                config,
                class_label_tokens=list(self.class_label_tokens),
            )
        else:
            with torch.device(device):
                model_copy = SequenceClassificationOnLogits(
                    config,
                    class_label_tokens=list(self.class_label_tokens),
                )
        return model_copy


SUPPORTED_POOL_TYPES = ("last", "mean")


class SequenceClassification(HeadModel):
    """
    Head model for sequence classification. Here, `targets` is a sequence
    of length 1, with values in `[0, num_classes)`. We place a linear head
    with biases on top of a state vector of the embedding dimension. This
    state vector is pooled according to `pool_type`:

    - "last": Outpyt embedding for final input token
    - "mean": Mean of output embeddings for all tokens

    """

    NAME = "seq_classification"

    def __init__(
        self,
        config: Config,
        num_classes: int = 2,
        pool_type: str = "last",
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes = {num_classes} must be >= 2")
        if pool_type not in SUPPORTED_POOL_TYPES:
            raise ValueError(
                f"pool_type = {pool_type} must be one of {SUPPORTED_POOL_TYPES}"
            )
        self.num_classes = num_classes
        self.n_embd = config.n_embd
        self.pool_type = pool_type
        self.linear_head = torch.nn.Linear(config.n_embd, num_classes, bias=True)
        self._num_tokens = None
        self._accumulated_state = None

    def needs_logits(self) -> bool:
        return False

    def forward(
        self,
        model_outputs: torch.Tensor,
        targets: Optional[torch.Tensor],
        input_pos: int,
    ) -> Union[torch.Tensor, float]:
        self._check_model_outputs_targets(
            model_outputs,
            targets,
            final_dim=self.n_embd,
        )
        if targets is not None and targets.shape[-1] != 1:
            raise ValueError(f"targets.length = {targets.shape[-1]}, must be 1")
        num = model_outputs.shape[-2]
        if input_pos == 0:
            # First chunk in sequence
            if self.pool_type == "mean":
                self._accumulated_state = 0
            self._num_tokens = 0
        else:
            assert self._num_tokens is not None, "First call must be with input_pos=0"
            assert (
                input_pos == self._num_tokens
            ), f"input_pos = {input_pos} != {self._num_tokens} = num_tokens"
        if self.pool_type == "mean":
            self._accumulated_state = (
                model_outputs.sum(dim=-2) + self._accumulated_state
            )
        self._num_tokens += num
        if targets is None:
            return 0
        if self.pool_type == "mean":
            self._accumulated_state = self._accumulated_state / self._num_tokens
        else:
            assert self.pool_type == "last"  # Sanity check
            self._accumulated_state = model_outputs[..., -1, :]
        logits = self.linear_head(self._accumulated_state)
        self._accumulated_state = None
        losses = F.cross_entropy(
            logits.reshape(-1, self.num_classes),
            targets.reshape(-1),
            reduction="none",
        )
        return losses.view(*logits.shape[:2]).mean(dim=-1)

    def num_target_entries(self, targets: Optional[torch.Tensor]) -> Optional[int]:
        return None

    def _empty_clone(self, device: Optional[torch.device] = None) -> "HeadModel":
        config = Config()
        config.n_embd = self.n_embd
        if device is None:
            model_copy = SequenceClassification(
                config,
                num_classes=self.num_classes,
                pool_type=self.pool_type,
            )
        else:
            with torch.device(device):
                model_copy = SequenceClassification(
                    config,
                    num_classes=self.num_classes,
                    pool_type=self.pool_type,
                )
        return model_copy
