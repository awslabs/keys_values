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
from typing import Optional, Dict, Any

from litgpt.config import Config
from litgpt.model import build_rope_cache, apply_rope

import torch


class PositionEncoding:
    """
    Base class for RoPE position encoding techniques which adapt to the current
    context width.

    """

    def __call__(
        self,
        x: torch.Tensor,
        input_pos: int,
        block_idx: int,
    ) -> torch.Tensor:
        """
        Encodes `x` (queries, keys) corresponding to token positions
        `range(input_pos, input_pos + x_len)`, where `x_len = x.shape[2]`.

        Args:
            x (torch.Tensor): Input tensor, shape
                `(batch_size, n_head, x_len, n_elem)`, where
                `n_elem <= head_size`
            input_pos (int): Determines token positions
            block_idx (int): Index of layer. Allows for encodings dependent on
                the layer

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


class LinearPositionEncoding(PositionEncoding):
    """
    Implements linear interpolation with a fixed scale factor and a fixed
    attention scale factor, as used (for example) in Gemma-2 and Gemma-3.

    """

    def __init__(
        self,
        config: Config,
        device: Optional[torch.device] = None,
        do_set_context_width: bool = True,
    ):
        """
        Fixed scale factor in `config.rope_adjustments['factor']`, fixed
        attention scale factor in `config.attention_scores_scalar`. Supports
        `config.rope_local_base_freq`, `config.rope_indices`.

        Args:
            config (Config): Configuration of model
            device (Optional[torch.device]): Device for state

        """
        self.rope_base = config.rope_base
        self.n_elem = config.rope_n_elem
        self.rope_local_base_freq = config.rope_local_base_freq
        self.rope_indices = config.rope_indices
        self._fixed_factor = None
        if config.rope_adjustments is not None:
            self._fixed_factor = config.rope_adjustments.get("factor")
            if self._fixed_factor is None:
                raise ValueError("config.rope_adjustments must include 'factor'")
        self.train_context_width = config.block_size
        if device is None:
            device = torch.get_default_device()
        self._device = device
        self._cos = None
        self._sin = None
        if config.attention_scores_scalar is not None:
            temp = config.attention_scores_scalar
        else:
            temp = config.head_size
        self._sdpa_scale_factor = 1.0 / math.sqrt(temp)
        self._context_width = config.block_size
        if do_set_context_width:
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
        if self._fixed_factor is None:
            return max(1.0, self.context_width / self.train_context_width)
        else:
            return self._fixed_factor

    def _precompute_extra_args(self) -> Dict[str, Any]:
        return {"factor": self._factor()}

    def _precompute(self):
        """
        The state consists of `_cos`, `_sin`, with shapes
        `(context_width, n_elem)` or `(context_width, n_elem, 2)`,
        the latter if `rope_indices` is given.

        """
        self._cos, self._sin = build_rope_cache(
            seq_len=self.context_width,
            n_elem=self.n_elem,
            device=self.device,
            base=self.rope_base,
            extra_config=self._precompute_extra_args(),
            rope_local_base_freq=self.rope_local_base_freq,
        )

    def __call__(
        self,
        x: torch.Tensor,
        input_pos: int,
        block_idx: int,
    ) -> torch.Tensor:
        if x.ndim < 2 or x.shape[-1] != self.n_elem:
            raise ValueError(
                f"x.shape = {x.shape}, must be at least 2D, and last dimension must be {self.n_elem}"
            )
        x_len = x.shape[-2]
        if input_pos < 0 or input_pos + x_len > self.context_width:
            raise ValueError(
                f"input_pos = {input_pos}, x_len = {x_len} (x.shape = {x.shape}), must have 0 <= input_pos, input_pos + x_len <= {self.context_width}"
            )
        if x.device != self._cos.device:
            self._device = x.device
            self._cos = self._cos.to(device=self._device)
            self._sin = self._sin.to(device=self._device)
        if self.rope_local_base_freq is None:
            cos = self._cos[input_pos : (input_pos + x_len), :].unsqueeze(0)
            sin = self._sin[input_pos : (input_pos + x_len), :].unsqueeze(0)
        else:
            ind = self.rope_indices[block_idx]
            cos = self._cos[input_pos : (input_pos + x_len), :, ind].unsqueeze(0)
            sin = self._sin[input_pos : (input_pos + x_len), :, ind].unsqueeze(0)
        return apply_rope(x=x, cos=cos, sin=sin)


DEFAULT_YARN_ALPHA = 1.0

DEFAULT_YARN_BETA = 32.0


class AdjustedPositionEncoding(LinearPositionEncoding):
    """
    Implements non-dynamic "adjusted RoPE", given by setting fields
    "factor", "original_max_seq_len", "low_freq_factor", "high_freq_factor" in
    `config.rope_adjustments`. Calling :meth:`set_context_width` does not
    scale the encoding, different to :class:`YaRNPositionEncoding`.

    """

    def __init__(
        self,
        config: Config,
        device: Optional[torch.device] = None,
    ):
        """
        The pre-training context width must be `config.block_size`, the
        RoPE base b used during training must be `config.rope_base`.

        Args:
            config (Config): Configuration of model

        """
        super().__init__(config, device, do_set_context_width=False)
        if config.rope_adjustments is None:
            raise ValueError("config.rope_adjustments must be given")
        self.train_context_width = config.rope_adjustments.get("original_max_seq_len")
        if self.train_context_width is None:
            raise ValueError(
                "Must have config.rope_adjustments['original_max_seq_len']"
            )
        assert self._fixed_factor is not None  # Sanity check
        alpha = config.rope_adjustments.get("low_freq_factor")
        beta = config.rope_adjustments.get("high_freq_factor")
        if alpha is None:
            alpha = DEFAULT_YARN_ALPHA
        if beta is None:
            beta = DEFAULT_YARN_BETA
        if not (0 < alpha < beta):
            raise ValueError(
                f"alpha = {alpha}, beta = {beta}: Must be 0 < alpha < beta"
            )
        self.alpha = alpha
        self.beta = beta
        if config.attention_scores_scalar is not None:
            temp = config.attention_scores_scalar
        else:
            temp = config.head_size
        self._sdpa_scale_factor = 1.0 / math.sqrt(temp)
        self._context_width = int(self._fixed_factor * self.train_context_width)
        self.set_context_width(self._context_width)

    def _precompute_extra_args(self) -> Dict[str, Any]:
        return {
            **super()._precompute_extra_args(),
            "original_max_seq_len": self.train_context_width,
            "low_freq_factor": self.alpha,
            "high_freq_factor": self.beta,
        }


class YaRNPositionEncoding(LinearPositionEncoding):
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
        super().__init__(config, device, do_set_context_width=False)
        self.head_size = config.head_size
        if config.attention_scores_scalar is not None:
            print(
                "You have set config.attention_scores_scalar. This is not "
                "supported, since for YaRN the scale_factor is determined by "
                "the position encoding. The value will be ignored."
            )
        train_context_width = None
        factor = None
        if config.rope_adjustments is not None:
            _alpha = config.rope_adjustments.get("low_freq_factor")
            if _alpha is not None:
                if alpha is not None and alpha != _alpha:
                    raise ValueError(
                        "Cannot have config.rope_adjustments['low_freq_factor'] and alpha"
                    )
                alpha = _alpha
            _beta = config.rope_adjustments.get("high_freq_factor")
            if _beta is not None:
                if beta is not None and beta != _beta:
                    raise ValueError(
                        "Cannot have config.rope_adjustments['high_freq_factor'] and beta"
                    )
                beta = _beta
            train_context_width = config.rope_adjustments.get("original_max_seq_len")
            factor = config.rope_adjustments.get("factor")
            if factor is not None:
                print(
                    "You have set config.rope_adjustments['factor']. For YaRN, "
                    "the scale factor is dynamic. Value here will be used to "
                    "initialize context_width, but will be updated with each "
                    "call of set_context_width."
                )
        if train_context_width is not None:
            self.train_context_width = train_context_width
        if factor is not None:
            self._context_width = int(factor * self.train_context_width)
        if alpha is None:
            alpha = DEFAULT_YARN_ALPHA
        if beta is None:
            beta = DEFAULT_YARN_BETA
        if not (0 < alpha < beta):
            raise ValueError(
                f"alpha = {alpha}, beta = {beta}: Must be 0 < alpha < beta"
            )
        self.alpha = alpha
        self.beta = beta
        self._sdpa_scale_factor = None
        self._fixed_factor = None
        self.set_context_width(self._context_width)

    def _precompute_extra_args(self) -> Dict[str, Any]:
        return {
            **super()._precompute_extra_args(),
            "original_max_seq_len": self.train_context_width,
            "low_freq_factor": self.alpha,
            "high_freq_factor": self.beta,
        }

    def _precompute(self):
        super()._precompute()
        sqrt_inv_t = 0.1 * math.log(self._factor()) + 1.0
        self._sdpa_scale_factor = sqrt_inv_t * sqrt_inv_t / math.sqrt(self.head_size)


def position_encoding_factory(
    config: Config,
    do_yarn: bool = False,
    **kwargs,
) -> PositionEncoding:
    if do_yarn:
        return YaRNPositionEncoding(config, **kwargs)
    else:
        rope_adjustments = config.rope_adjustments
        if rope_adjustments is not None and "original_max_seq_len" in rope_adjustments:
            return AdjustedPositionEncoding(config, **kwargs)
        else:
            return LinearPositionEncoding(config, **kwargs)
