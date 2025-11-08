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
from typing import Optional, Dict, Tuple, List

import torch

from litgpt.config import Config

from keys_values.kvcache.attn_weights import (
    AttnWeightsKVCache, AttnWeightsReplayLog,
)
from keys_values.kvcache.base import DefaultKVCacheReplayLog, KVCacheReplayLog
from keys_values.kvcache.basics import (
    DenseKVCache,
    LastRecentlyInsertedKVCache,
    LastRecentlyInsertedKVCacheReplayLog,
    KVCacheWithBuffers,
)
from keys_values.kvcache.buffers import KVCacheBuffers, PositionsType
from keys_values.kvcache.lastrec_alt import LastRecentlyInsertedAltKVCache
from keys_values.model import GPT


class InferenceReplayCacheMixin:
    """
    Mixin for :class:`KVCacheWithBuffers` subclasses, to be used for replaying
    in inference mode.

    This class is used for the inference forward pass on a row of cells, to
    obtain the KV cache buffer checkpoints. It is a trimmed version of the
    base class :class:`AttnWeightsKVCache`, where all scoring is removed, and
    decisions are taken by following the replay log.

    Note: Forward passes with an inference replay cache result in slightly
    different outcomes than with a training replay cache (i.e.,
    :class:`TrainingAttnWeightsReplayCache`). This is because here, we
    quantize and dequantize KV cache buffers after every update, this is not
    done in training mode.

    """
    def __init__(self):
        self.replay_log = None
        self.token_chunk_pos = 0

    @property
    def next_token_pos(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def current_length(self) -> int:
        raise NotImplementedError

    @property
    def cache_length(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        raise NotImplementedError

    @property
    def batch_size(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def n_query_groups(self) -> int:
        raise NotImplementedError

    def next_positions(self, num: int) -> Optional[torch.Tensor]:
        kwargs = {"dtype": torch.int64, "device": self.device}
        if self.next_token_pos is None:
            return None
        elif self.current_length < self.cache_length:
            assert num <= self.cache_length - self.current_length
            return torch.arange(
                self.current_length,
                min(self.cache_length, self.current_length + num),
                **kwargs,
            ).view(1, 1, -1).expand(self.batch_size, self.n_query_groups, -1)
        else:
            return self.replay_log.extract_index(
                self.next_token_pos, num, **kwargs,
            )

    def _validate_token_idx(self, token_idx: torch.Tensor):
        if self.replay_log is None:
            raise IndexError("Replay log must be set at construction")
        other = self.replay_log.token_chunks[self.token_chunk_pos].to(
            device=self.device,
        )
        if not token_idx.equal(other):
            raise ValueError(f"token_idx:\n{token_idx} -- {token_idx.shape}\nreplay_log.token_chunks[{self.token_chunk_pos}]:\n{other} -- {other.shape}\nShould be the same!")
        self.token_chunk_pos += 1


class InferenceAttnWeightsReplayCache(
    AttnWeightsKVCache, InferenceReplayCacheMixin
):
    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        replay_log: AttnWeightsReplayLog,
        **base_kwargs,
    ):
        AttnWeightsKVCache.__init__(
            self,
            config=config,
            buffers=buffers,
            block_idx=block_idx,
            grace_period=replay_log.grace_period,
            **base_kwargs,
        )
        InferenceReplayCacheMixin.__init__(self)
        if replay_log is None or len(replay_log) == 0:
            raise ValueError("replay_log must not be empty")
        if self.device != replay_log.device:
            raise ValueError(f"self.device = {self.device}, replay_log.device = {self.replay_log.device}: must be the same")
        self.replay_log = replay_log

    @property
    def next_token_pos(self) -> Optional[int]:
        return super().next_token_pos

    @property
    def current_length(self) -> int:
        return super().current_length

    @property
    def cache_length(self) -> Optional[int]:
        return super().cache_length

    @property
    def device(self) -> torch.device:
        return super().device

    @property
    def batch_size(self) -> Optional[int]:
        return super().batch_size

    @property
    def n_query_groups(self) -> int:
        return super().n_query_groups

    def next_positions(self, num: int) -> Optional[torch.Tensor]:
        return InferenceReplayCacheMixin.next_positions(self, num)

    def _update(self, *args, **kwargs):
        pass

    def update_requires_attn_weights(self) -> bool:
        return False

    def _initial_scores_in_prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        pass

    def _update_score_buffers_in_forward(
        self,
        grace_period_case: bool,
        positions: Optional[PositionsType],
        index: torch.Tensor,
        init_score_values: Dict[str, torch.Tensor],
        num1: Optional[int],
    ):
        pass

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        self._next_positions = None
        return super().size_estimate()

    def _validate_token_idx(self, token_idx: torch.Tensor):
        InferenceReplayCacheMixin._validate_token_idx(self, token_idx)


class InferenceDenseReplayCache(
    DenseKVCache, InferenceReplayCacheMixin
):
    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        replay_log: DefaultKVCacheReplayLog,
        **base_kwargs,
    ):
        DenseKVCache.__init__(
            self,
            config=config,
            buffers=buffers,
            block_idx=block_idx,
            **base_kwargs,
        )
        InferenceReplayCacheMixin.__init__(self)
        if len(replay_log) == 0:
            raise ValueError("replay_log must not be empty")
        if self.device != self.replay_log.device:
            raise ValueError(f"self.device = {self.device}, replay_log.device = {self.replay_log.device}: must be the same")
        self.replay_log = replay_log

    @property
    def next_token_pos(self) -> Optional[int]:
        return super().next_token_pos

    @property
    def current_length(self) -> int:
        return super().current_length

    @property
    def cache_length(self) -> Optional[int]:
        return super().cache_length

    @property
    def device(self) -> torch.device:
        return super().device

    @property
    def batch_size(self) -> Optional[int]:
        return super().batch_size

    @property
    def n_query_groups(self) -> int:
        return super().n_query_groups

    def next_positions(self, num: int) -> Optional[torch.Tensor]:
        return InferenceReplayCacheMixin.next_positions(self, num)

    def _validate_token_idx(self, token_idx: torch.Tensor):
        InferenceReplayCacheMixin._validate_token_idx(self, token_idx)


class InferenceLastRecentlyInsertedReplayCache(
    LastRecentlyInsertedKVCache, InferenceReplayCacheMixin
):
    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        replay_log: LastRecentlyInsertedKVCacheReplayLog,
        **base_kwargs,
    ):
        LastRecentlyInsertedKVCache.__init__(
            self,
            config=config,
            buffers=buffers,
            block_idx=block_idx,
            **base_kwargs,
        )
        InferenceReplayCacheMixin.__init__(self)
        if len(replay_log) == 0:
            raise ValueError("replay_log must not be empty")
        if self.device != replay_log.device:
            raise ValueError(f"self.device = {self.device}, replay_log.device = {self.replay_log.device}: must be the same")
        self.replay_log = replay_log

    @property
    def next_token_pos(self) -> Optional[int]:
        return super().next_token_pos

    @property
    def current_length(self) -> int:
        return super().current_length

    @property
    def cache_length(self) -> Optional[int]:
        return super().cache_length

    @property
    def device(self) -> torch.device:
        return super().device

    @property
    def batch_size(self) -> Optional[int]:
        return super().batch_size

    @property
    def n_query_groups(self) -> int:
        return super().n_query_groups

    def next_positions(self, num: int) -> Optional[torch.Tensor]:
        return InferenceReplayCacheMixin.next_positions(self, num)

    def _validate_token_idx(self, token_idx: torch.Tensor):
        InferenceReplayCacheMixin._validate_token_idx(self, token_idx)


class InferenceLastRecentlyInsertedAltReplayCache(
    LastRecentlyInsertedAltKVCache, InferenceReplayCacheMixin
):
    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        replay_log: LastRecentlyInsertedKVCacheReplayLog,
        **base_kwargs,
    ):
        LastRecentlyInsertedAltKVCache.__init__(
            self,
            config=config,
            buffers=buffers,
            block_idx=block_idx,
            **base_kwargs,
        )
        InferenceReplayCacheMixin.__init__(self)
        if len(replay_log) == 0:
            raise ValueError("replay_log must not be empty")
        if self.device != replay_log.device:
            raise ValueError(f"self.device = {self.device}, replay_log.device = {self.replay_log.device}: must be the same")
        self.replay_log = replay_log

    @property
    def next_token_pos(self) -> Optional[int]:
        return super().next_token_pos

    @property
    def current_length(self) -> int:
        return super().current_length

    @property
    def cache_length(self) -> Optional[int]:
        return super().cache_length

    @property
    def device(self) -> torch.device:
        return super().device

    @property
    def batch_size(self) -> Optional[int]:
        return super().batch_size

    @property
    def n_query_groups(self) -> int:
        return super().n_query_groups

    def next_positions(self, num: int) -> Optional[torch.Tensor]:
        return InferenceReplayCacheMixin.next_positions(self, num)

    def _validate_token_idx(self, token_idx: torch.Tensor):
        InferenceReplayCacheMixin._validate_token_idx(self, token_idx)


def inference_replay_cache_factory(
    kv_cache: KVCacheWithBuffers,
    config: Config,
    buffers: KVCacheBuffers,
    block_idx: int,
    replay_log: KVCacheReplayLog,
    **base_kwargs,
) -> Optional[KVCacheWithBuffers]:
    kwargs = dict(
        base_kwargs,
        config=config,
        buffers=buffers,
        block_idx=block_idx,
        replay_log=replay_log,
    )
    if isinstance(kv_cache, DenseKVCache):
        return InferenceDenseReplayCache(**kwargs)
    elif isinstance(kv_cache, LastRecentlyInsertedAltKVCache):
        return InferenceLastRecentlyInsertedAltReplayCache(**kwargs)
    elif isinstance(kv_cache, LastRecentlyInsertedKVCache):
        return InferenceLastRecentlyInsertedReplayCache(**kwargs)
    elif isinstance(kv_cache, AttnWeightsKVCache):
        return InferenceAttnWeightsReplayCache(**kwargs)
    else:
        return None


def get_replay_logs(model: GPT) -> List[KVCacheReplayLog]:
    kv_caches = [block.attn.kv_cache for block in model.transformer.h]
    if any(
        c is None or not isinstance(c, KVCacheWithBuffers)
        for c in kv_caches
    ):
        raise IndexError("All blocks of GPT model must have KV caches of type KVCacheWithBuffers assigned")
    if not all(c.do_replay_logging for c in kv_caches):
        raise IndexError("All KV caches of GPT model must have replay logging activated")
    return [c.get_replay_log() for c in kv_caches]
