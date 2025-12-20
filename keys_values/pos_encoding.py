# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file exc ept in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from typing import Optional

from litgpt.config import Config
from litgpt.model import build_rope_cache, apply_rope

import torch


class PositionEncoding:
    """
    Base class for RoPE position encoding techniques which adapt to the current
    context width.

    """
    def __call__(self, x: torch.Tensor, input_pos: int) -> torch.Tensor:
        """
        Encodes `x` (queries, keys) corresponding to token positions
        `range(input_pos, input_pos + x_len)`, where `x_len = x.shape[2]`.

        Args:
            x (torch.Tensor): Input tensor, shape
                `(batch_size, n_head, x_len, n_elem)`, where
                `n_elem <= head_size`
            input_pos (int): Determines token positions

        Returns:
            Position encoded tensor, same shape as `x`

        """
        raise NotImplementedError

    def sdpa_scale_factor(self) -> float:
        """
        Returns:
            Scale factor to be used in scaled dot product attention. Inner
            products between queries and keys are multiplied with this factor
            before the softmax.

        """
        raise NotImplementedError

    @property
    def context_width(self) -> int:
        """
        Returns:
            Current context width

        """
        raise NotImplementedError

    def set_context_width(self, width: int):
        """
        Args:
            width (int): Context width to which position encoding should be
                adapted to.

        """
        raise NotImplementedError

    @property
    def device(self) -> Optional[torch.device]:
        raise NotImplementedError


DEFAULT_YARN_ALPHA = 1.0

DEFAULT_YARN_BETA = 32.0


class YaRNPositionEncoding(PositionEncoding):
    """
    Implements YaRN, as detailed in:

        Peng, B. etal.
        YaRN: Efficient Context Window Extension of Large Language Models
        ICLR 2024

    """
    def __init__(
        self,
        config: Config,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        """
        The pre-training context width must be `config.block_size`, the
        RoPE base b used during training must be `config.rope_base`.

        If `config.rope_adjustments` is given, parameters are taken from
        there. In this case, `alpha`, `beta` must not be given.

        Args:
            config (Config): Configuration of model
            alpha (float): YaRN parameter, must have `0 < alpha < beta`
            beta (float): YaRN parameter, must have `0 < alpha < beta`

        """
        self.rope_base = config.rope_base
        self.head_size = config.head_size
        context_width = None
        train_context_width = None
        if config.rope_adjustments is not None:
            _alpha = config.rope_adjustments.get("low_freq_factor")
            if _alpha is not None:
                if alpha is not None and alpha != _alpha:
                    raise ValueError("Cannot have config.rope_adjustments['low_freq_factor'] and alpha")
                alpha = _alpha
            _beta = config.rope_adjustments.get("high_freq_factor")
            if _beta is not None:
                if beta is not None and beta != _beta:
                    raise ValueError("Cannot have config.rope_adjustments['high_freq_factor'] and beta")
                beta = _beta
            train_context_width = config.rope_adjustments.get("original_max_seq_len")
            factor = config.rope_adjustments.get("factor")
            if factor is not None:
                if train_context_width is None:
                    raise ValueError("config.rope_adjustments: Must have both or none of 'factor', 'original_max_seq_len'")
                context_width = int(train_context_width * factor)
        if context_width is None:
            context_width = config.block_size
        self._context_width = context_width
        if train_context_width is None:
            train_context_width = config.block_size
        self.train_context_width = train_context_width
        if alpha is None:
            alpha = DEFAULT_YARN_ALPHA
        if beta is None:
            beta = DEFAULT_YARN_BETA
        if not (0 < alpha < beta):
            raise ValueError(f"alpha = {alpha}, beta = {beta}: Must be 0 < alpha < beta")
        self.alpha = alpha
        self.beta = beta
        if device is None:
            device = torch.get_default_device()
        self._device = device
        self._cos = None
        self._sin = None
        self._sdpa_scale_factor = None
        self.set_context_width(self._context_width)

    @property
    def context_width(self) -> int:
        return self._context_width

    def set_context_width(self, width: int):
        if width <= 0:
            raise ValueError(f"width = {width}: Must be positive")
        self._context_width = width
        self._precompute()

    def sdpa_scale_factor(self) -> float:
        return self._sdpa_scale_factor

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    def _factor(self) -> float:
        return max(1.0, self.context_width / self.train_context_width)

    def _precompute(self):
        """
        Precomputations, based on context width. Must be called whenever the
        context width changes.

        """
        extra_config = {
            "original_max_seq_len": self.train_context_width,
            "low_freq_factor": self.alpha,
            "high_freq_factor": self.beta,
            "factor": self._factor(),
        }
        self._cos, self._sin = build_rope_cache(
            seq_len=self.context_width,
            n_elem=self.head_size,
            device=self.device,
            base=self.rope_base,
            extra_config=extra_config,
        )
        sqrt_inv_t = 0.1 * math.log(self._factor()) + 1.0
        self._sdpa_scale_factor = sqrt_inv_t * sqrt_inv_t / math.sqrt(self.head_size)

    def __call__(self, x: torch.Tensor, input_pos: int) -> torch.Tensor:
        if x.ndim < 2 or x.shape[-1] > self.head_size:
            raise ValueError(f"x.shape = {x.shape}, must be at least 2D, and last dimension must be <= {self.head_size}")
        x_len = x.shape[-2]
        if input_pos < 0 or input_pos + x_len > self.context_width:
            raise ValueError(f"input_pos = {input_pos}, x_len = {x_len}, must have 0 <= input_pos, input_pos + x_len <= {self.context_width}")
        if x.device != self._cos.device:
            self._device = x.device
            self._cos = self._cos.to(device=self._device)
            self._sin = self._sin.to(device=self._device)
        return apply_rope(
            x=x,
            cos = self._cos[input_pos:(input_pos + x_len), :],
            sin = self._sin[input_pos:(input_pos + x_len), :],
        )
