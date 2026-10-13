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
import copy
from typing import Optional, Tuple, Dict, Any, List

import torch

from keys_values.config import Config
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.new.basics import (
    KVCachePolicy,
    KVCacheReplayLog,
    KVCacheReplayLogEntry,
    KVCacheReplayLogStatus,
    PositionsNewEntriesType,
)
from keys_values.kvcache.utils import positions_wrap_around
from keys_values.utils import (
    index_to_3d,
    bits_for_torch_dtype,
    bitsize_of,
)


class DefaultKVCachePolicy(KVCachePolicy):
    """
    Abstract implementation of :class:`KVCachePolicy`, contains common
    code. In particular, replay logging is implemented generically.
    """

    def __init__(
        self,
        config: Config,
        cache_length: int,
        use_replay_logging: bool = True,
        replay_log: Optional[KVCacheReplayLog] = None,
    ):
        """
        Args:
            config: Configuration object
            cache_length: Cache length
            use_replay_logging: Whether to use replay logging (so
                `replay_log_status != "none"`)
            replay_log: If given, we start with
                `replay_log_status = "play"`, replaying from this log.
                Otherwise, we start with `replay_log_status = "record"`
                and an empty log.

        """
        super().__init__()
        self._config = config
        self._cache_length = cache_length
        self._batch_size = None
        # Need to track this here as well:
        self._input_pos = 0
        self._replay_log: Optional[KVCacheReplayLog] = None
        self._replay_log_status: KVCacheReplayLogStatus = "none"
        self._replay_log_nextpos = None
        if use_replay_logging:
            if replay_log is None:
                self._replay_log = []
                self._replay_log_status = "record"
            else:
                self._replay_log = replay_log
                self._replay_log_status = "play"
                self._replay_log_nextpos = 0

    @property
    def cache_length(self) -> int:
        return self._cache_length

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @property
    def n_query_groups(self) -> Optional[int]:
        return self._config.n_query_groups

    def _device_internal(self) -> torch.device:
        return torch.get_default_device() if self.device is None else self.device

    def _base_kwargs_for_clone(self) -> Dict[str, Any]:
        """
        Supports :meth:`clone` implementations in subclasses.
        Note that the copy created by :meth:`clone` uses the same `self.mha`
        object (shallow copy).

        Returns:
            Keyword arguments for calling the constructor in :meth:`clone`

        """
        return dict(
            config=self._config,
            cache_length=self._cache_length,
            use_replay_logging=self._replay_log_status != "none",
        )

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        return 0, dict()

    @classmethod
    def size_estimate_apriori(
        cls, params: KVCacheParams, **kwargs
    ) -> Tuple[int, Dict[str, int]]:
        return 0, dict()

    def replay_log_status(self) -> KVCacheReplayLogStatus:
        return self._replay_log_status

    def reset(self, replay_log_status: KVCacheReplayLogStatus):
        if replay_log_status != self.replay_log_status and (self.replay_log_status == "none" or replay_log_status == "none"):
            raise ValueError(f"Cannot switch replay_log_status from '{self.replay_log_status}' to '{replay_log_status}'")
        if replay_log_status == "play":
            self._replay_log_nextpos = 0
        elif replay_log_status == "record":
            self._replay_log_nextpos = None
            self._replay_log = []
        self._input_pos = 0

    def get_replay_log(self) -> Optional[KVCacheReplayLog]:
        return copy.copy(self._replay_log)

    def _get_next_replay_entry(self) -> Optional[KVCacheReplayLogEntry]:
        if self._replay_log_status == "play":
            if self._replay_log_nextpos >= len(self._replay_log):
                raise IndexError(f"replay_log_nextpos = {self._replay_log_nextpos}, len(replay_log) = {len(self._replay_log)}: Not enough entries to replay!")
            return self._replay_log[self._replay_log_nextpos]
        else:
            return None

    def _validate_token_idx(self, token_idx: torch.Tensor) -> Optional[KVCacheReplayLogEntry]:
        next_entry = self._get_next_replay_entry()
        if next_entry is not None:
            other = next_entry.token_idx.to(device=token_idx.device)
            if not token_idx.equal(other):
                raise ValueError(
                    f"token_idx:\n{token_idx} -- {token_idx.shape}\nreplay_log.token_chunks[{self.token_chunk_pos}]:\n{other} -- {other.shape}\nShould be the same!"
                )
            self._replay_log_nextpos += 1
        return next_entry

    def prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        self._validate_token_idx(token_idx)
        self._batch_size = token_idx.shape[0]
        self._input_pos = 0
        self._prefill(query, key, value, token_idx)
        self._input_pos = token_idx.shape[-1]

    def _prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        raise NotImplementedError

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> PositionsNewEntriesType:
        if self._replay_log_status == "play":
            entry = self._validate_token_idx(token_idx)
            positions = entry.positions.to(device=query.device)
        else:
            positions = None
        positions = self._forward(
            query,
            key,
            value,
            token_idx,
            positions=positions,
        )
        if self._replay_log_status == "record":
            self._replay_log.append(
                KVCacheReplayLogEntry(
                    token_idx=token_idx.to(device=query.device),
                    positions=positions,
                )
            )
        self._input_pos += token_idx.shape[-1]
        return positions

    def _forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        positions: Optional[PositionsNewEntriesType] = None,
    ) -> PositionsNewEntriesType:
        """
        If `positions is None`, this does the work of :meth:`forward`.
        Otherwise, `positions` are passed in and returned as is, and the
        state changes *other* than determining `positions` is done.

        """
        raise NotImplementedError


class DenseKVCachePolicy(DefaultKVCachePolicy):
    """
    Policy for dense attention. Key and value tensors for all past tokens are
    maintained. This cache can only process as many tokens as its cache
    length. It requires a lot of memory, so can only be used for moderate
    cache lengths.
    """

    def __init__(
        self,
        config: Config,
        cache_length: int,
        use_replay_logging: bool = True,
        replay_log: Optional[KVCacheReplayLog] = None,
    ):
        super().__init__(config, cache_length, use_replay_logging, replay_log)

    def token_positions(self) -> torch.Tensor:
        if self.batch_size is None:
            raise RuntimeError("batch_size is not set. Call `prefill` first.")
        return index_to_3d(
            torch.arange(self._input_pos, device=self._device_internal()),
            self.batch_size,
            self.n_query_groups,
        )

    def _forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        positions: Optional[PositionsNewEntriesType] = None,
    ) -> PositionsNewEntriesType:
        if positions is None:
            np = self._input_pos
            num = key.shape[2]
            positions = torch.arange(
                np, np + num, device=query.device, dtype=torch.int,
            )
        return positions

    def _prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        pass

    def max_forward_length(self) -> int:
        return self.cache_length - self._input_pos

    def clone(self) -> "KVCachePolicy":
        return DenseKVCachePolicy(**self._base_kwargs_for_clone())


DEFAULT_INIT_GRACE_TOKENS = 16


class LastRecentlyInsertedKVCachePolicy(DefaultKVCachePolicy):
    """
    KV cache policy which stores the last recently inserted `cache_length` key
    and value tensors. When the cache is full, those slots are overwritten
    whose content is in the cache for the longest time.

    Information for the initial `init_grace_tokens` tokens remains in the cache
    indefinitely. The default is positive, see :const:`DEFAULT_INIT_GRACE_TOKENS`,
    catering for the "attention sink" property exhibited by pre-trained LLMs:

    Xiao etal.
    Efficient Streaming Language Models with Attention Sinks
    https://arxiv.org/abs/2309.17453
    """

    def __init__(
        self,
        config: Config,
        cache_length: int,
        use_replay_logging: bool = True,
        replay_log: Optional[KVCacheReplayLog] = None,
        init_grace_tokens: Optional[int] = None,
    ):
        super().__init__(config, cache_length, use_replay_logging, replay_log)
        if init_grace_tokens is None:
            init_grace_tokens = min(
                DEFAULT_INIT_GRACE_TOKENS,
                max(cache_length // 8, 1),
            )
        elif init_grace_tokens < 0:
            raise ValueError(f"init_grace_tokens={init_grace_tokens}, must be >= 0")
        self.init_grace_tokens = init_grace_tokens
        # Note: We could generate `token_pos` on the fly from `input_pos`
        # and `next_position`, but it is simpler just to maintain it.
        device = self._device_internal()
        self.register_buffer(
            "token_pos",
            torch.zeros(cache_length, device=device, dtype=torch.int),
            persistent=False,
        )
        # Position of first slot to overwrite with next :meth:`forward`
        self.next_position = None
        # Tracked here as well:
        self._current_length = 0

    @classmethod
    def _parameter_names(cls) -> List[str]:
        return super()._parameter_names() + ["token_pos"]

    def token_positions(self) -> torch.Tensor:
        if self.batch_size is None:
            raise RuntimeError("batch_size is not set. Call `prefill` first.")
        return index_to_3d(
            self.token_pos[: self._current_length],
            self.batch_size,
            self.n_query_groups,
        )

    def _forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        positions: Optional[PositionsNewEntriesType] = None,
    ) -> PositionsNewEntriesType:
        num = key.shape[2]
        start = self.init_grace_tokens
        device = self._device_internal()
        ind_kwargs = dict(device=device, dtype=torch.int)
        if positions is None:
            positions = positions_wrap_around(
                num=num,
                current=self.next_position,
                start=start,
                end=self.cache_length,
                batch_size=self.batch_size,
                n_query_groups=self.n_query_groups,
                device=device,
            )
            if isinstance(positions, tuple):
                positions = torch.arange(positions[0], positions[1], **ind_kwargs)
        np = self.next_position
        num1 = min(num, self.cache_length - np)
        diff = num - num1
        ntp = self._input_pos
        self.token_pos[np : (np + num1)] = torch.arange(
            ntp, ntp + num1, **ind_kwargs,
        )
        np += num1
        if diff > 0 or np == self.cache_length:
            self.token_pos[start : (start + diff)] = torch.arange(
                ntp + num1, ntp + num, **ind_kwargs,
            )
            np = start + diff
        self.next_position = np
        self._current_length = min(self._input_pos + num, self.cache_length)
        return positions

    def _prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        init_length = token_idx.shape[-1]
        if init_length < self.cache_length:
            self.next_position = init_length
        else:
            assert init_length == self.cache_length, (
                init_length,
                self.cache_length,
            )  # Sanity check
            self.next_position = self.init_grace_tokens
        self.token_pos[:init_length] = torch.arange(
            init_length,
            dtype=self.token_pos.dtype,
            device=self._device_internal(),
        )
        self._batch_size = token_idx.shape[0]
        self._current_length = init_length

    def max_forward_length(self) -> int:
        return self.cache_length - self.init_grace_tokens

    def _base_kwargs_for_clone(self) -> Dict[str, Any]:
        return dict(
            super()._base_kwargs_for_clone(),
            init_grace_tokens=self.init_grace_tokens,
        )

    def clone(self) -> "KVCachePolicy":
        return LastRecentlyInsertedKVCachePolicy(**self._base_kwargs_for_clone())

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        tk_p = bitsize_of(self.token_pos)
        return tk_p, {"token_pos": tk_p}

    @classmethod
    def size_estimate_apriori(
        cls, params: KVCacheParams, **kwargs
    ) -> Tuple[int, Dict[str, int]]:
        tk_p = params.cache_length * bits_for_torch_dtype(torch.int)
        return tk_p, {"token_pos": tk_p}

    def reset(self, replay_log_status: KVCacheReplayLogStatus):
        super().reset(replay_log_status)
        self._current_length = 0
