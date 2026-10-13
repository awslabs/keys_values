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
from keys_values.kvcache.h2o import VLengthInstantScoreMixin, REMOVE_ARG_NAMES
from keys_values.kvcache.new.attn_weights import AttnWeightsKVCachePolicy
from keys_values.kvcache.new.basics import KVCachePolicy, KVCacheReplayLog
from keys_values.utils import bits_for_torch_dtype, bitsize_of


class H2OKVCachePolicy(AttnWeightsKVCachePolicy):
    """
    Implements some variants of the heavy hitter oracle (H2O) KV cache, see

        Zhang et al
        H2O: Heavy-hitter oracle for efficient generative inference of large language models
        Advances in Neural Information Processing Systems 37, 2024
        https://openreview.net/forum?id=RkRrPp7GKO

    Our implementation contains some improvements over their code:

    * They average scores over the batch dimension and occupy slots
      independent of the batch dimension. We make eviction decisions
      for each batch entry independently, which is not more expensive.
    * They sum scores over all rounds, which may favor earlier tokens.
      We allow this as well, but also support normalization of
      cumulative scores by the number of rounds a token is in the
      cache (if `normalize_scores=True`).
    * Our implementation is easy to generalize to other scores based on
      attention weights, see :meth:`_instantaneous_score`.

    The original H2O method as published is provided in
    :class:`H2OOriginalKVCachePolicy`.
    """

    def __init__(
        self,
        config: Config,
        cache_length: int,
        max_batch_size: int,
        use_replay_logging: bool = True,
        replay_log: Optional[KVCacheReplayLog] = None,
        grace_period: int = 0,
        detach_attn_weights: bool = False,
        keep_initial_fraction: Optional[float] = None,
        max_chunk_size: Optional[int] = None,
        normalize_scores: bool = False,
    ):
        super().__init__(
            config,
            cache_length,
            max_batch_size,
            use_replay_logging,
            replay_log,
            grace_period,
            detach_attn_weights,
            keep_initial_fraction,
            max_chunk_size,
        )
        self.normalize_scores = normalize_scores
        shape = (max_batch_size, self.n_query_groups, cache_length)
        device = self._device_internal()
        self.register_buffer(
            "scores",
            torch.zeros(shape, device=device, dtype=torch.float32),
            persistent=False,
        )

    @classmethod
    def _parameter_names(cls) -> List[str]:
        return super()._parameter_names() + cls._score_buffer_names()

    def _score_buffers(self) -> List[Tuple[torch.Tensor, str]]:
        return [(self.scores, "scores")]

    @classmethod
    def _score_buffer_names(cls) -> List[str]:
        return ["scores"]

    def fix_dtype_of_score_buffers(self):
        if self.scores.dtype != torch.float32:
            self.scores = self.scores.to(torch.float32)

    def _initial_scores_in_forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return dict()

    def _compute_scores(
        self,
        attn_weights: torch.Tensor,
        query_length: int,
    ) -> Optional[torch.Tensor]:
        if attn_weights.dtype != torch.float32:
            raise ValueError(
                f"attn_weights.dtype={attn_weights.dtype}, must be {torch.float32}"
            )
        self.scores[
            : self.batch_size, :, : self._current_length
        ] += self._instantaneous_score(attn_weights, query_length)
        scores = None
        if self._current_length == self.cache_length:
            # Exclude the grace region, so that these tokens are not evicted
            limit = self.cache_length - self.grace_period
            scores = self.scores[: self.batch_size, :, :limit]
            if self.normalize_scores:
                # Normalize cumulative scores
                token_pos = self.token_pos[: self.batch_size, :, :limit]
                assert token_pos.shape == scores.shape  # Sanity check
                other = torch.full(
                    (1, 1, 1),
                    self.prefill_length - 1,
                    dtype=self.token_pos.dtype,
                    device=self.device,
                ).expand(*token_pos.shape)
                token_pos = token_pos.maximum(other)
                # Note: `input_pos` has been increased already
                denom = (self.input_pos - token_pos).to(torch.float32)
                scores = scores / denom
            # Subclasses may modify the scores:
            scores = self._modify_scores(scores)
        return scores

    def _instantaneous_score(
        self,
        attn_weights: torch.Tensor,
        query_length: int,
    ) -> torch.Tensor:
        """
        Computes score values for this round from attention weights. These score
        values are accumulated.

        Args:
            attn_weights: Attention weights, shape
                `(batch_size, n_query_heads, current_length)`
            query_length: Size of query axis

        Returns:
            Instantaneous score values, same shape and dtype as `attn_weights`.

        """
        return attn_weights  # H2O accumulates attention weights directly

    def _modify_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Transform score values at the end of :meth:`_compute_scores`.

        Args:
            scores: Score values, shape
                `(batch_size, n_query_groups, cache_length - grace_period)`

        Returns:
            Modified values, same shape

        """
        return scores  # Default: no modification

    def _base_kwargs_for_clone(self) -> Dict[str, Any]:
        return dict(
            super()._base_kwargs_for_clone(),
            normalize_scores=self.normalize_scores,
        )

    def clone(self) -> KVCachePolicy:
        return H2OKVCachePolicy(**self._base_kwargs_for_clone())

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        sz_total, dct_sz = super().size_estimate()
        sz_sc = 0
        for scores, name in self._score_buffers():
            sz = int(bitsize_of(scores))
            dct_sz[name] = sz
            sz_sc += sz
        return sz_total + sz_sc, dict(dct_sz, scores=sz_sc)

    @classmethod
    def size_estimate_apriori(
        cls, params: KVCacheParams, **kwargs
    ) -> Tuple[int, Dict[str, int]]:
        sz_total, dct_sz = super().size_estimate_apriori(params, **kwargs)
        numel = params.max_batch_size * params.n_query_groups * params.cache_length
        sz = int(numel * bits_for_torch_dtype(torch.float32))
        sz_sc = 0
        for name in cls._score_buffer_names():
            dct_sz[name] = sz
            sz_sc += sz
        return sz_total + sz_sc, dict(dct_sz, scores=sz_sc)


class VLengthH2OKVCachePolicy(H2OKVCachePolicy, VLengthInstantScoreMixin):
    """
    Variant of H2O. Instead of cumulative attention weights, we use
    cumulative expected v vector sizes, where the expectation is over the
    attention weights. The instantaneous score is:

    ..math::
        S(j) = n_Q * m_j |v_j| / Z,  Z = sum_j m_j |v_j|,
         m_j = sum_i m_{i,j}

    Namely, `m_j` is `attn_weights` summed over the query axis, `|v_j|` the
    norm of the v vector, n_Q the length of the query axis (`query_length`
    passed to :meth:`_instantaneous_score`).

    Different to H2O, this score takes the length of v vectors into account as
    well, which in the end determine the attention outputs alongside the
    attention weights.
    """

    def __init__(
        self,
        config: Config,
        cache_length: int,
        max_batch_size: int,
        use_replay_logging: bool = True,
        replay_log: Optional[KVCacheReplayLog] = None,
        grace_period: int = 0,
        detach_attn_weights: bool = False,
        keep_initial_fraction: Optional[float] = None,
        max_chunk_size: Optional[int] = None,
        normalize_scores: bool = False,
    ):
        super().__init__(
            config,
            cache_length,
            max_batch_size,
            use_replay_logging,
            replay_log,
            grace_period,
            detach_attn_weights,
            keep_initial_fraction,
            max_chunk_size,
            normalize_scores,
        )
        shape = (max_batch_size, self.n_query_groups, cache_length)
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

    def _instantaneous_score(
        self,
        attn_weights: torch.Tensor,
        query_length: int,
    ) -> torch.Tensor:
        return VLengthInstantScoreMixin._instantaneous_score(
            self,
            attn_weights,
            query_length,
        )

    def _initial_scores_in_forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return VLengthInstantScoreMixin._initial_scores_in_forward(self, key, value)

    def _initial_scores_in_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        H2OKVCachePolicy._initial_scores_in_prefill(self, key, value)
        VLengthInstantScoreMixin._initial_scores_in_prefill(self, key, value)

    def get_v_norm(self) -> torch.Tensor:
        return self.v_norm[: self.batch_size, :, : self._current_length]

    def clone(self) -> KVCachePolicy:
        return VLengthH2OKVCachePolicy(**self._base_kwargs_for_clone())


class H2OOriginalKVCachePolicy(AttnWeightsKVCachePolicy):
    """
    Implements the heavy hitter oracle (H2O) KV cache, see

        Zhang et al
        H2O: Heavy-hitter oracle for efficient generative inference of large language models
        Advances in Neural Information Processing Systems 37, 2024
        https://openreview.net/forum?id=RkRrPp7GKO

    This is the original version, equivalent to their published code. This
    class is mostly for comparisons, we recommend to use
    :class:`H2OKVCachePolicy` instead, which has some simple improvements.

    The original version sums scores over the batch dimension and makes
    decisions independent of this dimension. This is why `self.next_positions`
    and `self.token_pos` are broadcast here over the batch dimension. Their
    shapes remain the same, for compatibility with the parent class. Also,
    the score buffer `scores` has a batch dimension, even if it is not used.
    """

    def __init__(
        self,
        config: Config,
        cache_length: int,
        max_batch_size: int,
        use_replay_logging: bool = True,
        replay_log: Optional[KVCacheReplayLog] = None,
    ):
        super().__init__(
            config,
            cache_length,
            max_batch_size,
            use_replay_logging,
            replay_log,
        )
        # Note: `scores` has a batch dimension, even though it is not used.
        # This is because all score buffers in :class:`AttnWeightsKVCache` have
        # the same shape, which has a batch dimension.
        shape = (max_batch_size, self.n_query_groups, cache_length)
        device = self._device_internal()
        self.register_buffer(
            "scores",
            torch.zeros(shape, device=device, dtype=torch.float32),
            persistent=False,
        )

    def _compute_scores(
        self,
        attn_weights: torch.Tensor,
        query_length: int,
    ) -> Optional[torch.Tensor]:
        if attn_weights.dtype != torch.float32:
            raise ValueError(
                f"attn_weights.dtype={attn_weights.dtype}, must be {torch.float32}"
            )
        # Sum over the batch dimension
        aggregated_weights = attn_weights.sum(0, keepdim=True)
        self.scores[: self.batch_size, :, : self._current_length] += aggregated_weights
        if self._current_length == self.cache_length:
            return self.scores[: self.batch_size]
        else:
            return None

    @classmethod
    def _parameter_names(cls) -> List[str]:
        return super()._parameter_names() + cls._score_buffer_names()

    def _score_buffers(self) -> List[Tuple[torch.Tensor, str]]:
        return [(self.scores, "scores")]

    @classmethod
    def _score_buffer_names(cls) -> List[str]:
        return ["scores"]

    def _base_kwargs_for_clone(self) -> Dict[str, Any]:
        return {
            k: v
            for k, v in super()._base_kwargs_for_clone().items()
            if k not in REMOVE_ARG_NAMES
        }

    def clone(self) -> KVCachePolicy:
        return H2OOriginalKVCachePolicy(**self._base_kwargs_for_clone())

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        sz_total, dct_sz = super().size_estimate()
        sz_sc = bitsize_of(self.scores)
        return sz_total + sz_sc, dict(dct_sz, scores=sz_sc)

    @classmethod
    def size_estimate_apriori(
        cls, params: KVCacheParams, **kwargs
    ) -> Tuple[int, Dict[str, int]]:
        numel = params.n_query_groups * params.cache_length
        sz_sc = numel * bits_for_torch_dtype(torch.float32)
        sz_total, dct_sz = super().size_estimate_apriori(params, **kwargs)
        return sz_total + sz_sc, dict(dct_sz, scores=sz_sc)
