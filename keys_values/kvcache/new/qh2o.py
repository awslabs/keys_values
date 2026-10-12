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
from typing import Tuple, Dict, Any, List, Optional

import torch

from keys_values.config import Config
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.h2o import VLengthInstantScoreMixin
from keys_values.kvcache.qh2o import (
    DEFAULT_COMBINATION_CONSTANT,
    DEFAULT_SCRATCH_BLOCKSIZE,
)
from keys_values.kvcache.quant_buffers import QuantizedKVCacheBuffers
from keys_values.kvcache.new.basics import KVCachePolicy
from keys_values.kvcache.new.h2o import H2OKVCachePolicy
from keys_values.utils import bits_for_torch_dtype, bitsize_of


class QuantizedH2OKVCachePolicy(H2OKVCachePolicy):
    """
    Implements improved variant of the Q-Hitter KV cache, see

        Zhang et al
        Q-Hitter: A better token oracle for efficient LLM inference via sparse-quantized KV cache
        Proceedings of Machine Learning and Systems (MLSys) 2024, volume 6, pages 381–394
        https://proceedings.mlsys.org/paper_files/paper/2024/hash/bbb7506579431a85861a05fff048d3e1-Abstract-Conference.html

     Our implementation contains some improvements over their code at
     https://github.com/VITA-Group/Q-Hitter/blob/main/utils/modify_llama.py:

    * They average scores over the batch dimension and occupy slots
      independent of the batch dimension. We make eviction decisions
      for each batch entry independently, which is not more expensive. This is
      the same improvement as we do for H2O (:class:`H2OKVCachePolicy` versus
      :class:`H2OOriginalKVCachePolicy`).
    * They sum scores over all rounds, which may favor earlier tokens.
      We allow this as well, but also support normalization of
      cumulative scores by the number of rounds a token is in the
      cache (if `normalize_scores=True`).
    * They score quantization error on K, V content of batch dimension
      0, which is arbitrary. We compute quantization errors for each
      batch dimension.
    * Their code normalizes the two components of the score differently, which
      is quite odd. We normalize both components in the same way.
    """

    def __init__(
        self,
        config: Config,
        kv_buffers: QuantizedKVCacheBuffers,
        grace_period: int = 0,
        replay_log_blocksize: Optional[int] = None,
        detach_attn_weights: bool = False,
        keep_initial_fraction: Optional[float] = None,
        max_chunk_size: Optional[int] = None,
        normalize_scores: bool = False,
        combination_constant: float = DEFAULT_COMBINATION_CONSTANT,
        scratch_blocksize: int = DEFAULT_SCRATCH_BLOCKSIZE,
    ):
        super().__init__(
            config,
            kv_buffers.cache_length,
            kv_buffers.max_batch_size,
            grace_period,
            replay_log_blocksize,
            detach_attn_weights,
            keep_initial_fraction,
            max_chunk_size,
            normalize_scores,
        )
        self.kv_buffers = kv_buffers
        if not (0 <= combination_constant <= 1):
            raise ValueError(
                f"combination_constant = {combination_constant}, must be in [0, 1]"
            )
        if scratch_blocksize < 1:
            raise ValueError("scratch_blocksize must be positive int")
        scratch_blocksize = min(scratch_blocksize, kv_buffers.cache_length)
        self._scratch_blocksize = scratch_blocksize
        self.combination_constant = combination_constant
        shape = (kv_buffers.max_batch_size, self.n_query_groups, kv_buffers.cache_length)
        device = self._device_internal()
        self.register_buffer(
            "q_errors",
            torch.zeros(shape, device=device, dtype=torch.float32),
            persistent=False,
        )

    @classmethod
    def _parameter_names(cls) -> List[str]:
        return super()._parameter_names() + ["q_errors"]

    def _score_buffers(self) -> List[Tuple[torch.Tensor, str]]:
        return super()._score_buffers() + [(self.q_errors, "q_errors")]

    @classmethod
    def _score_buffer_names(cls) -> List[str]:
        return super()._score_buffer_names() + ["q_errors"]

    def fix_dtype_of_score_buffers(self):
        super().fix_dtype_of_score_buffers()
        if self.q_errors.dtype != torch.float32:
            self.q_errors = self.q_errors.to(torch.float32)

    def _compute_quantization_errors(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        write_out = out is not None
        if write_out:
            shape = key.shape[:-1]
            if out.shape != shape:
                raise ValueError(f"out.shape = {out.shape}, must be {shape}")
            if out.dtype != torch.float32:
                raise ValueError(f"out.dtype = {out.dtype}, must be torch.float32")
        num = key.shape[2]
        bsz = self._scratch_blocksize
        parts = []
        for start in range(0, num, bsz):
            end = min(start + bsz, num)
            _key = key[:, :, start:end, :]
            _value = value[:, :, start:end, :]
            part_kandv = self.kv_buffers.quantization_error(_key, _value)
            part = part_kandv[0].to(torch.float32) + part_kandv[1].to(torch.float32)
            if write_out:
                out[:, :, start:end] = part
            else:
                parts.append(part)

        return out if write_out else torch.cat(parts, dim=-1)

    def _initial_scores_in_forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        quant_error = self._compute_quantization_errors(key, value)
        return {"q_errors": quant_error}

    def _initial_scores_in_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        super()._initial_scores_in_prefill(key, value)
        init_length = key.shape[2]
        self._compute_quantization_errors(
            key,
            value,
            out=self.q_errors[: self.batch_size, :, :init_length],
        )

    def _modify_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """
        `scores` are the H2O scores computed in `H2OKVCache._compute_scores`.
        We add the quantization error parts here as a convex combination.
        """
        limit = self.cache_length - self.grace_period
        assert scores.shape[2] == limit  # Sanity check
        _scores = self._min_max_normalize(scores)
        _q_errors = self._min_max_normalize(self.q_errors[: self.batch_size, :, :limit])
        return (
            self.combination_constant * _scores
            + ((self.combination_constant - 1) / 2) * _q_errors
            + (1 - self.combination_constant)
        )

    @staticmethod
    def _min_max_normalize(scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: Array to be normalized, shape `(batch_size, ...)`

        Returns:
            Min-max normalization along the dimensions other than the first
            (batch dimension)

        """
        _scores = scores.reshape(scores.size(0), -1)
        _min = _scores.min(dim=-1, keepdim=True)[0]
        _denom = _scores.max(dim=-1, keepdim=True)[0] - _min
        _scores = (_scores - _min) / _denom
        return _scores.view(*scores.shape)

    def _base_kwargs_for_clone(self) -> Dict[str, Any]:
        return dict(
            super()._base_kwargs_for_clone(),
            combination_constant=self.combination_constant,
            scratch_blocksize=self._scratch_blocksize,
        )

    def clone(self) -> KVCachePolicy:
        return QuantizedH2OKVCachePolicy(**self._base_kwargs_for_clone())


class QuantizedVLengthH2OKVCachePolicy(
    QuantizedH2OKVCachePolicy,
    VLengthInstantScoreMixin,
):
    """
    Variant of :class:`QuantizedH2OKVCachePolicy`. Derived from there in the
    same way as :class:`VLengthH2OKVCachePolicy` is derived from
    :class:`H2OKVCachePolicy`.
    """

    def __init__(
        self,
        config: Config,
        kv_buffers: QuantizedKVCacheBuffers,
        grace_period: int = 0,
        replay_log_blocksize: Optional[int] = None,
        detach_attn_weights: bool = False,
        keep_initial_fraction: Optional[float] = None,
        max_chunk_size: Optional[int] = None,
        normalize_scores: bool = False,
        combination_constant: float = DEFAULT_COMBINATION_CONSTANT,
        scratch_blocksize: int = DEFAULT_SCRATCH_BLOCKSIZE,
    ):
        super().__init__(
            config,
            kv_buffers,
            grace_period,
            replay_log_blocksize,
            detach_attn_weights,
            keep_initial_fraction,
            max_chunk_size,
            normalize_scores,
            combination_constant,
            scratch_blocksize,
        )
        shape = (kv_buffers.max_batch_size, self.n_query_groups, kv_buffers.cache_length)
        device = self._device_internal()
        self.register_buffer(
            self.get_name_v_norm(),
            torch.zeros(shape, device=device, dtype=torch.float32),
            persistent=False,
        )

    @classmethod
    def get_name_v_norm(cls) -> str:
        return "v_norm"

    def _score_buffers(self) -> List[Tuple[torch.Tensor, str]]:
        return super()._score_buffers() + [(self.v_norm, self.get_name_v_norm())]

    @classmethod
    def _score_buffer_names(cls) -> List[str]:
        return super()._score_buffer_names() + [cls.get_name_v_norm()]

    def fix_dtype_of_score_buffers(self):
        super().fix_dtype_of_score_buffers()
        if self.v_norm.dtype != torch.float32:
            self.v_norm = self.v_norm.to(torch.float32)

    @classmethod
    def _parameter_names(cls) -> List[str]:
        return super()._parameter_names() + [cls.get_name_v_norm()]

    def _initial_scores_in_forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            **QuantizedH2OKVCachePolicy._initial_scores_in_forward(self, key, value),
            **VLengthInstantScoreMixin._initial_scores_in_forward(self, key, value),
        }

    def _initial_scores_in_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        QuantizedH2OKVCachePolicy._initial_scores_in_prefill(self, key, value)
        VLengthInstantScoreMixin._initial_scores_in_prefill(self, key, value)

    def get_v_norm(self) -> torch.Tensor:
        return self.v_norm[: self.batch_size, :, : self._current_length]

    def clone(self) -> KVCachePolicy:
        return QuantizedVLengthH2OKVCachePolicy(**self._base_kwargs_for_clone())
