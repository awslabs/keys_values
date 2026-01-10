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
from typing import Optional, Tuple, Dict, List, Any

import torch
from torch.linalg import vector_norm

from litgpt.config import Config

from keys_values.kvcache.attn_weights import AttnWeightsKVCache
from keys_values.kvcache.base import KVCacheParams, KVCache
from keys_values.kvcache.basics import KVCacheWithBuffers
from keys_values.kvcache.buffers import KVCacheBuffers
from keys_values.kvcache.utils import bitsize_of, bits_for_torch_dtype


class H2OKVCache(AttnWeightsKVCache):
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
    :class:`H2OOriginalKVCache`.
    """
    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        grace_period: int = 0,
        replay_log_blocksize: Optional[int] = None,
        detach_attn_weights: bool = False,
        keep_initial_fraction: Optional[float] = None,
        max_chunk_size: Optional[int] = None,
        normalize_scores: bool = False,
        **base_kwargs,
    ):
        super().__init__(
            config=config,
            buffers=buffers,
            block_idx=block_idx,
            grace_period=grace_period,
            replay_log_blocksize=replay_log_blocksize,
            detach_attn_weights=detach_attn_weights,
            keep_initial_fraction=keep_initial_fraction,
            max_chunk_size=max_chunk_size,
            **base_kwargs,
        )
        self.normalize_scores = normalize_scores
        shape = (buffers.max_batch_size, self.n_query_groups, buffers.cache_length)
        device = self._default_device_for_new_params()
        self.register_buffer(
            "scores",
            torch.zeros(shape, device=device, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def from_config(
        config: Config,
        max_batch_size: int,
        cache_length: int,
        block_idx: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        grace_period: int = 0,
        replay_log_blocksize: Optional[int] = None,
        detach_attn_weights: bool = False,
        keep_initial_fraction: Optional[float] = None,
        max_chunk_size: Optional[int] = None,
        normalize_scores: bool = False,
        **base_kwargs,
    ) -> "H2OKVCache":
        buffers_kwargs = KVCacheWithBuffers.extract_default_buffers_kwargs(base_kwargs)
        buffers = KVCacheWithBuffers.create_default_buffers(
            config=config,
            max_batch_size=max_batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
            **buffers_kwargs,
        )
        return H2OKVCache(
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

    @classmethod
    def _parameter_names(cls) -> List[str]:
        return super()._parameter_names() + cls._score_buffer_names()

    def _score_buffers(self) -> List[Tuple[torch.Tensor, str]]:
        return [(self.scores, "scores")]

    @classmethod
    def _score_buffer_names(cls) -> List[str]:
        return ["scores"]

    def _compute_scores(
        self,
        attn_weights: torch.Tensor,
        query_length: int,
    ) -> Optional[torch.Tensor]:
        # Map weights to instantaneous scores, accumulate them. Sum over
        # dimension 2 (`num` tokens)
        if attn_weights.dtype != torch.float32:
            raise ValueError(f"attn_weights.dtype={attn_weights.dtype}, must be {torch.float32}")
        self.scores[
            :self.batch_size, :, :self.current_length
        ] += self._instantaneous_score(attn_weights, query_length)
        scores = None
        if self.current_length == self.cache_length:
            # Exclude the grace region, so that these tokens are not evicted
            limit = self.cache_length - self.grace_period
            scores = self.scores[:self.batch_size, :, :limit]
            if self.normalize_scores:
                # Normalize cumulative scores
                token_pos = self.token_pos[:self.batch_size, :, :limit]
                assert token_pos.shape == scores.shape  # Sanity check
                other = torch.full(
                    (1, 1, 1),
                    self.prefill_length - 1,
                    dtype=self.token_pos.dtype,
                    device=self.device
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
        return scores

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        sz_total, dct_sz = super().size_estimate()
        sz_sc = 0.0
        for scores, name in self._score_buffers():
            sz = int(bitsize_of(scores))
            dct_sz[name] = sz
            sz_sc += sz
        return sz_total + sz_sc, dict(dct_sz, scores=sz_sc)

    @classmethod
    def size_estimate_apriori(cls, params: KVCacheParams, **kwargs) -> Tuple[int, Dict[str, int]]:
        sz_total, dct_sz = super().size_estimate_apriori(params, **kwargs)
        numel = params.max_batch_size * params.n_query_groups * params.cache_length
        sz = int(numel * bits_for_torch_dtype(torch.float32))
        sz_sc = 0
        for name in cls._score_buffer_names():
            dct_sz[name] = sz
            sz_sc += sz
        return sz_total + sz_sc, dict(dct_sz, scores=sz_sc)

    def clone(self) -> KVCache:
        return H2OKVCache(**self._base_kwargs_for_clone())

    def _base_kwargs_for_clone(self) -> Dict[str, Any]:
        base_kwargs = super()._base_kwargs_for_clone()
        base_kwargs["normalize_scores"] = self.normalize_scores
        return base_kwargs


class VLengthInstantScoreMixin:
    """
    Same as H2O in :class:`H2OKVCache`, but the instantaneous score is
    modified to take the length of V vectors into account.

    """
    def get_v_norm(self) -> torch.Tensor:
        """
        Returns:
            Norms of V along final dimension, shape
            `(batch_size, n_query_heads, current_length)`, dtype`torch.float32`.

        """
        raise NotImplementedError()

    @classmethod
    def get_name_v_norm(cls) -> str:
        raise NotImplementedError()

    def get_kv_buffers(self) -> KVCacheBuffers:
        raise NotImplementedError()

    @property
    def batch_size(self) -> int:
        raise NotImplementedError()

    @property
    def next_positions(self) -> torch.Tensor:
        raise NotImplementedError()

    def _instantaneous_score(
        self,
        attn_weights: torch.Tensor,
        query_length: int,
    ) -> torch.Tensor:
        """
        The score is the attention weight times the v vector norm, normalized
        to sum to 1 over the cache length dimension, then multiplied by
        `query_length`.

        NOTE: If we obtained the attention weights NOT summed over the query
        axis, we could first normalize to sum to 1 over the cache length
        dimension, then sum over the query axis, which may be a better score.
        But we do not obtain the full attention weights.

        """
        scores = self.get_v_norm() * attn_weights
        return (scores / scores.sum(dim=-1, keepdim=True)) * query_length

    def _initial_scores_in_forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            self.get_name_v_norm(): vector_norm(
                value[:self.batch_size],
                dim=-1,
                dtype=torch.float32,
            )
        }

    def _initial_scores_in_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        v_norm_buffer = self.get_v_norm()
        vector_norm(
            value[:self.batch_size],
            dim=-1,
            dtype=torch.float32,
            out=v_norm_buffer,
        )


class VLengthH2OKVCache(H2OKVCache, VLengthInstantScoreMixin):
    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        grace_period: int = 0,
        replay_log_blocksize: Optional[int] = None,
        detach_attn_weights: bool = False,
        keep_initial_fraction: Optional[float] = None,
        max_chunk_size: Optional[int] = None,
        normalize_scores: bool = False,
        **base_kwargs,
    ):
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
        shape = (buffers.max_batch_size, self.n_query_groups, buffers.cache_length)
        device = self._default_device_for_new_params()
        self.register_buffer(
            self.get_name_v_norm(),
            torch.zeros(shape, device=device, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def from_config(
        config: Config,
        max_batch_size: int,
        cache_length: int,
        block_idx: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        grace_period: int = 0,
        replay_log_blocksize: Optional[int] = None,
        detach_attn_weights: bool = False,
        keep_initial_fraction: Optional[float] = None,
        max_chunk_size: Optional[int] = None,
        normalize_scores: bool = False,
        **base_kwargs,
    ) -> "VLengthH2OKVCache":
        buffers_kwargs = KVCacheWithBuffers.extract_default_buffers_kwargs(base_kwargs)
        buffers = KVCacheWithBuffers.create_default_buffers(
            config=config,
            max_batch_size=max_batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
            **buffers_kwargs,
        )
        return VLengthH2OKVCache(
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

    def _instantaneous_score(
        self,
        attn_weights: torch.Tensor,
        query_length: int,
    ) -> torch.Tensor:
        return VLengthInstantScoreMixin._instantaneous_score(
            self, attn_weights, query_length,
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
        H2OKVCache._initial_scores_in_prefill(self, key, value)
        VLengthInstantScoreMixin._initial_scores_in_prefill(self, key, value)

    def get_v_norm(self) -> torch.Tensor:
        return self.v_norm[:self.batch_size, :, :self.current_length]

    def get_kv_buffers(self) -> KVCacheBuffers:
        return self.kv_buffers

    def clone(self) -> KVCache:
        return VLengthH2OKVCache(**self._base_kwargs_for_clone())


class H2OOriginalKVCache(AttnWeightsKVCache):
    """
    Implements the heavy hitter oracle (H2O) KV cache, see

        Zhang et al
        H2O: Heavy-hitter oracle for efficient generative inference of large language models
        Advances in Neural Information Processing Systems 37, 2024
        https://openreview.net/forum?id=RkRrPp7GKO

    This is the original version, equivalent to their published code. This
    class is mostly for comparisons, we recommend to use :class:`H2OKVCache`
    instead, which has some simple improvements.

    The original version sums scores over the batch dimension and makes
    decisions independent of this dimension. This is why `self.next_positions`
    and `self.token_pos` are broadcast here over the batch dimension. Their
    shapes remain the same, for compatibility with the parent class. Also,
    the score buffer `scores` has a batch dimension, even if it is not used.

    """
    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        **base_kwargs,
    ):
        super().__init__(config, buffers, block_idx, **base_kwargs)
        # Note: `scores` has a batch dimension, even though it is not used.
        # This is because all score buffers in :class:`AttnWeightsKVCache` have
        # the same shape, which has a batch dimension.
        shape = (self.max_batch_size, self.n_query_groups, buffers.cache_length)
        device = self._default_device_for_new_params()
        self.register_buffer(
            "scores",
            torch.zeros(shape, device=device, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def from_config(
        config: Config,
        max_batch_size: int,
        cache_length: int,
        block_idx: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **base_kwargs,
    ) -> "H2OOriginalKVCache":
        buffers_kwargs = KVCacheWithBuffers.extract_default_buffers_kwargs(base_kwargs)
        buffers = KVCacheWithBuffers.create_default_buffers(
            config=config,
            max_batch_size=max_batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
            **buffers_kwargs,
        )
        return H2OOriginalKVCache(config, buffers, block_idx, **base_kwargs)

    def _prefill_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        # We do not compute H2O scores and cumulate scores for tokens inserted
        # by the prefill. Instead, we set the scores to 0.
        super()._prefill_internal(key, value, token_idx)
        self.scores.fill_(0.0)

    def _compute_scores(
        self,
        attn_weights: torch.Tensor,
        query_length: int,
    ) -> Optional[torch.Tensor]:
        if attn_weights.dtype != torch.float32:
            raise ValueError(f"attn_weights.dtype={attn_weights.dtype}, must be {torch.float32}")
        # Sum over the batch dimension 0
        aggregated_weights = attn_weights.sum(0, keepdim=True)
        self.scores[:self.batch_size, :, :self.current_length] += aggregated_weights
        if self.current_length == self.cache_length:
            return self.scores[:self.batch_size]
        else:
            return None

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        sz_total, dct_sz = super().size_estimate()
        sz_sc = bitsize_of(self.scores)
        return sz_total + sz_sc, dict(dct_sz, scores=sz_sc)

    @classmethod
    def size_estimate_apriori(cls, params: KVCacheParams, **kwargs) -> Tuple[int, Dict[str, int]]:
        numel = params.n_query_groups * params.cache_length
        sz_sc = numel * bits_for_torch_dtype(torch.float32)
        sz_total, dct_sz = super().size_estimate_apriori(params, **kwargs)
        return sz_total + sz_sc, dict(dct_sz, scores=sz_sc)

    def _score_buffers(self) -> List[Tuple[torch.Tensor, str]]:
        return [(self.scores, "scores")]

    @classmethod
    def _score_buffer_names(cls) -> List[str]:
        return ["scores"]

    @classmethod
    def _parameter_names(cls) -> List[str]:
        return super()._parameter_names() + ["scores"]

    def clone(self) -> KVCache:
        return H2OOriginalKVCache(**self._base_kwargs_for_clone())
