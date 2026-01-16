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
from typing import Tuple, List, Dict, Optional, Any

import torch

from litgpt.config import Config

from keys_values.kvcache.basics import KVCache
from keys_values.kvcache.buffers import KVCacheBuffers
from keys_values.kvcache.h2o import H2OKVCache, VLengthInstantScoreMixin
from keys_values.kvcache.quant_buffers import QuantizedKVCacheBuffers


DEFAULT_COMBINATION_CONSTANT = 0.5

DEFAULT_SCRATCH_BLOCKSIZE = 1024


class QuantizedH2OKVCache(H2OKVCache):
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
      the same improvement as we do for H2O (:class:`H2OKVCache` versus
      :class:`H2OOriginalKVCache`).
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
        buffers: QuantizedKVCacheBuffers,
        block_idx: int,
        grace_period: int = 0,
        replay_log_blocksize: Optional[int] = None,
        detach_attn_weights: bool = False,
        keep_initial_fraction: Optional[float] = None,
        max_chunk_size: Optional[int] = None,
        normalize_scores: bool = False,
        combination_constant: float = DEFAULT_COMBINATION_CONSTANT,
        scratch_blocksize: int = DEFAULT_SCRATCH_BLOCKSIZE,
        **base_kwargs,
    ):
        """
        Additional args:
            combination_constant: Constant for convex combination of H2O and
                quantization error scores. Defaults to 0.5.
            scratch_blocksize: Quantization errors are computed in blocks of
                this length. The larger, the faster, but also more scratch
                memory is required.

        """
        super().__init__(
            config,
            buffers,
            block_idx,
            grace_period,
            replay_log_blocksize,
            detach_attn_weights,
            keep_initial_fraction,
            max_chunk_size,
            normalize_scores,
            **base_kwargs,
        )
        if not isinstance(buffers, QuantizedKVCacheBuffers):
            raise TypeError("buffers must be of type QuantizedKVCacheBuffers")
        if not (0 <= combination_constant <= 1):
            raise ValueError(
                f"combination_constant = {combination_constant}, must be in [0, 1]"
            )
        if scratch_blocksize < 1:
            raise ValueError("scratch_blocksize must be positive int")
        scratch_blocksize = min(scratch_blocksize, buffers.cache_length)
        self._scratch_blocksize = scratch_blocksize
        self.combination_constant = combination_constant
        shape = (buffers.max_batch_size, self.n_query_groups, buffers.cache_length)
        device = self._default_device_for_new_params()
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
        num = key.shape[2]
        bsz = self._scratch_blocksize
        parts = []
        for start in range(0, num, bsz):
            end = min(start + bsz, num)
            _key = key[:, :, start:end, :]
            _value = value[:, :, start:end, :]
            part_kandv = self.kv_buffers.quantization_error(_key, _value)
            part = part_kandv[0] + part_kandv[1]
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

    def clone(self) -> KVCache:
        return QuantizedH2OKVCache(**self._base_kwargs_for_clone())

    def _base_kwargs_for_clone(self) -> Dict[str, Any]:
        return dict(
            super()._base_kwargs_for_clone(),
            combination_constant=self.combination_constant,
            scratch_blocksize=self._scratch_blocksize,
        )


class QuantizedVLengthH2OKVCache(QuantizedH2OKVCache, VLengthInstantScoreMixin):
    """
    Variant of :class:`QuantizedH2OKVCache`. Derived from there in the same
    way as :class:`VLengthH2OKVCache` is derived from :class:`H2OKVCache`.

    """

    def __init__(
        self,
        config: Config,
        buffers: QuantizedKVCacheBuffers,
        block_idx: int,
        grace_period: int = 0,
        replay_log_blocksize: Optional[int] = None,
        keep_initial_fraction: Optional[float] = None,
        max_chunk_size: Optional[int] = None,
        normalize_scores: bool = False,
        combination_constant: float = DEFAULT_COMBINATION_CONSTANT,
        scratch_blocksize: int = DEFAULT_SCRATCH_BLOCKSIZE,
        **base_kwargs,
    ):
        super().__init__(
            config=config,
            buffers=buffers,
            block_idx=block_idx,
            grace_period=grace_period,
            replay_log_blocksize=replay_log_blocksize,
            keep_initial_fraction=keep_initial_fraction,
            max_chunk_size=max_chunk_size,
            normalize_scores=normalize_scores,
            combination_constant=combination_constant,
            scratch_blocksize=scratch_blocksize,
            **base_kwargs,
        )
        shape = (buffers.max_batch_size, self.n_query_groups, buffers.cache_length)
        device = self._default_device_for_new_params()
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

    @classmethod
    def _parameter_names(cls) -> List[str]:
        return super()._parameter_names() + [cls.get_name_v_norm()]

    def _initial_scores_in_forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            **QuantizedH2OKVCache._initial_scores_in_forward(self, key, value),
            **VLengthInstantScoreMixin._initial_scores_in_forward(self, key, value),
        }

    def _initial_scores_in_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        QuantizedH2OKVCache._initial_scores_in_prefill(self, key, value)
        VLengthInstantScoreMixin._initial_scores_in_prefill(self, key, value)

    def get_v_norm(self) -> torch.Tensor:
        return self.v_norm[: self.batch_size, :, : self.current_length]

    def get_kv_buffers(self) -> KVCacheBuffers:
        return self.kv_buffers

    def clone(self) -> KVCache:
        return QuantizedVLengthH2OKVCache(**self._base_kwargs_for_clone())
