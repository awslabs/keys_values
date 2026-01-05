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
from typing import Optional, Tuple, Dict, Callable, List, Any

import torch

from litgpt.config import Config

from keys_values.attention import KeysAndValues
from keys_values.kvcache.base import (
    KVCache,
    DefaultKVCache,
    DefaultKVCacheReplayLog,
    KVCacheParams,
    KVCacheReplayLog,
)
from keys_values.kvcache.buffers import (
    DefaultKVCacheBuffers,
    KVCacheBuffers,
    KVCacheBuffersParams,
    positions_wrap_around,
)
from keys_values.kvcache.utils import bitsize_of, bits_for_torch_dtype


class KVCacheWithBuffers(DefaultKVCache):
    """
    Base class of all KV caches supported by KV cache buffers of type
    :class:`KVCacheBuffers`. We recommend that all KV caches separate out
    buffers, in order to keep selection orthogonal to content compression
    (e.g., by quantization).

    Checkpoint hook:

    We also support a checkpoint hook, in the form of `checkpoint_hook`. If
    this is given, we count the number of :meth:`forward` calls starting with
    a call to :meth:`prefill` (the prefill call is 0, first forward is 1, ...)
    in `chunk_idx`. We then call
    `checkpoint_hook(self.kv_buffers, self.chunk_idx)` at the start of
    :meth:`forward`. An important use case is activation checkpointing for
    gradient computation.

    Buffers, dependence on batch size:

    Subclasses support buffer allocation on demand, as well as their
    deallocation, so that GPU memory is freed for other uses. Buffers are
    allocated (if not already present) with a call of :meth:`prefill`,
    and the effective batch size `batch_size` is used then. Also, if
    buffers exist, but differ in shape due to `batch_size`, they are
    reallocated. This allows us to support different batch sizes (e.g.
    for training and validation), without having to allocate buffers
    for `max_batch_size`.

    """
    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        **base_kwargs,
    ):
        """
        Args:
            config: Model config
            buffers: KV cache buffers to be used
            block_idx: Index of model block (or layer)

        """
        super().__init__(
            config=config,
            max_batch_size=buffers.max_batch_size,
            cache_length=buffers.cache_length,
            block_idx=block_idx,
            dtype=buffers.dtype,
            **base_kwargs,
        )
        self.config = config  # Needed for :meth:`clone`
        self.kv_buffers = buffers
        self._checkpoint_hook = None
        self._chunk_idx = None
        if buffers.current_length is not None and buffers.current_length > 0:
            print(f"WARNING: buffers.current_length = {buffers.current_length} > 0")
        # If `buffers.device` already determined, this fixes the device of the cache
        self._device = buffers.device

    @property
    def current_length(self) -> int:
        return self.kv_buffers.current_length

    @property
    def batch_size(self) -> Optional[int]:
        return self.kv_buffers.batch_size

    @staticmethod
    def extract_default_buffers_kwargs(
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = dict()
        name = "allocate_buffers"
        if name in kwargs:
            result[name] = kwargs.pop(name)
        return result

    @staticmethod
    def create_default_buffers(
        config: Config,
        max_batch_size: int,
        cache_length: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **buffers_kwargs,
    ) -> DefaultKVCacheBuffers:
        return DefaultKVCacheBuffers(
            params=KVCacheBuffersParams.from_config(
                config=config,
                max_batch_size=max_batch_size,
                device=device,
                dtype=dtype,
            ),
            cache_length=cache_length,
            **buffers_kwargs,
        )

    def deallocate_buffers(self):
        """
        Deallocates the underlying buffers. They are automatically reallocated
        with the first :meth:`_prefill` call.

        Use this method if inference is iterated with another computation which
        needs lots of device memory, in order to share the device memory between
        the two.

        Another use case is if a model is moved to a different device, including
        its KV caches. This works only if KV cache buffers are deallocated.

        """
        self.kv_buffers.deallocate()
        self._device = None

    @property
    def buffers_are_allocated(self) -> bool:
        return self.kv_buffers.buffers_are_allocated

    def set_checkpoint_hook(
        self,
        checkpoint_hook: Optional[Callable[[KVCacheBuffers, int], None]],
    ):
        """
        Args:
            checkpoint_hook: See header comment. If `None`, the hook is
                removed.

        """
        self._checkpoint_hook = checkpoint_hook
        self._chunk_idx = None

    def _prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        self._validate_token_idx(token_idx)
        self._prefill_internal(key, value, token_idx)
        if self._checkpoint_hook is not None:
            self._chunk_idx = 0

    def _prefill_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        raise NotImplementedError

    def _forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> KeysAndValues:
        if not self.buffers_are_allocated:
            raise IndexError("Buffers are not allocated. Call 'prefill' first.")
        if self._checkpoint_hook is not None:
            if self._chunk_idx is None:
                raise IndexError("Chunk index not initialized. Call 'prefill' first.")
            self._chunk_idx += 1
            self._checkpoint_hook(self.kv_buffers, self._chunk_idx)
        self._validate_token_idx(token_idx)
        return self._forward_internal(key, value, token_idx)

    def _forward_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> KeysAndValues:
        raise NotImplementedError

    def get_keys_values(self) -> Optional[KeysAndValues]:
        return self.kv_buffers.get_keys_values()

    def _validate_token_idx(self, token_idx: torch.Tensor):
        """
        Called at the start of :meth:`forward` and :meth:`prefill`, can be used
        in subclasses.

        Args:
            token_idx: Argument to :meth:`forward`

        """
        pass

    def switch_replay_logging(self, status: bool):
        """
        By default, replay logging is switched off.

        Args:
            status: Do replay logging?

        """
        raise NotImplementedError

    @property
    def do_replay_logging(self) -> bool:
        """
        Returns:
            Is replay logging enabled?

        """
        raise NotImplementedError

    def get_replay_log(self) -> Optional[KVCacheReplayLog]:
        """
        Returns:
            Cache replay log which is required for gradient computations. If
            `do_replay_logging == False`, `None` is returned.

        """
        raise NotImplementedError


class DenseKVCache(KVCacheWithBuffers):
    """
    Key-value cache for dense attention. Key and value tensors for all
    past tokens are maintained. The cache length is the maximum sequence
    length. This cache requires a lot of memory, it can only be used for
    moderate cache lengths.

    Note: If the cache is full, :meth:`forward` raises an exception. The cache
    buffers are allocated up front and are not enlarged later on.

    """
    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        **base_kwargs,
    ):
        super().__init__(config, buffers, block_idx, **base_kwargs)
        self.next_position = None
        self._replay_log = None

    @staticmethod
    def from_config(
        config: Config,
        max_batch_size: int,
        cache_length: int,
        block_idx: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **base_kwargs,
    ) -> "DenseKVCache":
        buffers_kwargs = KVCacheWithBuffers.extract_default_buffers_kwargs(base_kwargs)
        buffers = KVCacheWithBuffers.create_default_buffers(
            config=config,
            max_batch_size=max_batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
            **buffers_kwargs,
        )
        return DenseKVCache(config, buffers, block_idx, **base_kwargs)

    @property
    def next_token_pos(self) -> Optional[int]:
        return self.next_position

    @property
    def max_tokens_forward(self) -> int:
        return self.cache_length

    @property
    def max_prefill_length(self) -> Optional[int]:
        return self.cache_length

    def _forward_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> KeysAndValues:
        if self.next_position is None:
            raise IndexError("Cache needs to be initialized with 'prefill' before being used")
        num = key.shape[2]
        np = self.next_position
        if np + num > self.cache_length:
            raise IndexError(
                f"Cache has at most {self.cache_length - np} free slots, cannot add {num} entries")
        self.next_position += num
        if self._replay_log is not None:
            if not isinstance(self._replay_log, DefaultKVCacheReplayLog):
                raise IndexError("Cannot switch on replay logging in the middle of inference run. Call 'prefill'.")
            self._replay_log.append_token_chunk(token_idx)
        return self.kv_buffers.forward(
            positions=(np, np + num),
            key=key,
            value=value,
        )

    def _update(self, *args, **kwargs):
        pass

    def _prefill_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        self.kv_buffers.prefill(key, value)
        init_length = key.shape[2]
        self.next_position = init_length
        if self._replay_log is not None:
            self._replay_log = DefaultKVCacheReplayLog(
                token_chunks=[token_idx],
                cache_length=self.cache_length,
                max_prefill_length=self.max_prefill_length,
                dtype=self.dtype,
            )

    def token_positions(self) -> torch.Tensor:
        return torch.arange(self.next_position, device=self.device).reshape(
            1, 1, -1
        ).expand(self.batch_size, self.n_query_groups, -1)

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        return self.kv_buffers.size_estimate()

    @classmethod
    def size_estimate_apriori(cls, params: KVCacheParams, **kwargs) -> Tuple[int, Dict[str, int]]:
        """
        `cache_length` is required in `kwargs`. If `buffer_type` is given in
        `kwargs`, the size for this type is used, otherwise for the default
        type `DefaultKVCacheBuffers`.

        """
        buff_params = KVCacheBuffersParams.from_params(params)
        buffer_type = kwargs.get("buffer_type", DefaultKVCacheBuffers)
        return buffer_type.size_estimate_apriori(
            buff_params, cache_length=params.cache_length, **kwargs,
        )

    def switch_replay_logging(self, status: bool):
        if status:
            if self._replay_log is None:
                self._replay_log = []  # Dummy, created later
        else:
            self._replay_log = None

    @property
    def do_replay_logging(self) -> bool:
        return self._replay_log is not None

    def get_replay_log(self) -> Optional[KVCacheReplayLog]:
        return self._replay_log

    def clone(self, device: Optional[torch.device] = None) -> KVCache:
        if self.kv_buffers.buffers_are_allocated:
            raise ValueError(f"Buffers must be deallocated, use `deallocate_buffers`")
        result = DenseKVCache(
            config=self.config,
            buffers=self.kv_buffers,
            block_idx=self.block_idx,
            **self._base_kwargs_for_clone(),
        )
        result._device = device
        return result


class LastRecentlyInsertedKVCacheReplayLog(DefaultKVCacheReplayLog):
    """
    Baseline key-value cache which stores the last recently inserted
    `cache_length` key, value tensors.

    """
    def __init__(
        self,
        token_chunks: List[torch.Tensor],
        cache_length: int,
        batch_size: int,
        n_query_groups: int,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            token_chunks,
            cache_length,
            max_prefill_length=None,
            grace_period=0,
            dtype=dtype,
        )
        self._shape = (batch_size, n_query_groups)

    def extract_index(
        self,
        token_pos: int,
        num: int,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = self.__len__()
        if num <= 0 or token_pos < 0 or token_pos > seq_length - num:
            raise ValueError(f"token_pos = {token_pos}, num = {num}, seq_length = {seq_length}: Out of range")
        if token_pos < self.cache_length:
            raise ValueError(f"token_pos = {token_pos} must be >= {self.cache_length} = cache_length")
        device = kwargs.get("device", self.device)
        return positions_wrap_around(
            num=num,
            current=token_pos % self.cache_length,
            start=0,
            end=self.cache_length,
            batch_size=self._shape[0],
            n_query_groups=self._shape[1],
            device=device,
            return_tensor=True,
        ).to(**kwargs)


class LastRecentlyInsertedKVCache(KVCacheWithBuffers):
    """
    Baseline key-value cache which stores the last recently inserted
    `cache_length` key, value tensors.

    """
    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        **base_kwargs,
    ):
        """
        Args:
            config: Model config
            buffers: KV cache buffers to be used
        """
        super().__init__(config, buffers, block_idx, **base_kwargs)
        device = buffers.device
        if device is None:
            device = torch.get_default_device()
        self.register_buffer(
            "token_pos",
            torch.zeros(buffers.cache_length, device=device, dtype=torch.int),
            persistent=False,
        )
        self.next_position = None
        self._next_token_pos = None
        self._replay_log = None

    @staticmethod
    def from_config(
        config: Config,
        max_batch_size: int,
        cache_length: int,
        block_idx: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **base_kwargs,
    ) -> "LastRecentlyInsertedKVCache":
        """
        Creates KV cache with default buffers.

        Args:
            config: Model config
            max_batch_size: Maximum batch size supported
            cache_length: Number of slots (i.e., tokens) in cache
            device: Device for buffers
            dtype: Data type for buffers

        """
        buffers_kwargs = KVCacheWithBuffers.extract_default_buffers_kwargs(base_kwargs)
        buffers = KVCacheWithBuffers.create_default_buffers(
            config=config,
            max_batch_size=max_batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
            **buffers_kwargs,
        )
        return LastRecentlyInsertedKVCache(
            config, buffers, block_idx, **base_kwargs,
        )

    def _parameter_names(self) -> List[str]:
        return ["token_pos"]

    @property
    def next_token_pos(self) -> Optional[int]:
        return self._next_token_pos

    @property
    def max_prefill_length(self) -> Optional[int]:
        return None  # This KV cache can be prefilled with any length

    @property
    def max_tokens_forward(self) -> int:
        return self.cache_length

    def _forward_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> KeysAndValues:
        if self.next_position is None:
            raise IndexError("Cache needs to be initialized with 'prefill' before being used")
        num = key.shape[2]
        positions = positions_wrap_around(
            num=num,
            current=self.next_position,
            start=0,
            end=self.cache_length,
            batch_size=self.batch_size,
            n_query_groups=self.n_query_groups,
            device=self.device,
        )
        k_and_v = self.kv_buffers.forward(
            positions=positions,
            key=key,
            value=value,
        )
        np = self.next_position
        num1 = min(num, self.cache_length - np)
        diff = num - num1
        ntp = self._next_token_pos
        self.token_pos[np:(np + num1)] = torch.arange(
            ntp, ntp + num1, device=self.device, dtype=torch.int
        )
        if diff > 0:
            self.token_pos[:diff] = torch.arange(
                ntp + num1, ntp + num, device=self.device, dtype=torch.int
            )
        self.next_position = (np + num) % self.cache_length
        if self._replay_log is not None:
            if not isinstance(self._replay_log, DefaultKVCacheReplayLog):
                raise IndexError("Cannot switch on replay logging in the middle of inference run. Call 'prefill'.")
            self._replay_log.append_token_chunk(token_idx)
        self._next_token_pos += num
        return k_and_v

    def _update(self, *args, **kwargs):
        pass

    def _prefill_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        init_length = key.shape[2]
        eff_init_length = min(init_length, self.cache_length)
        if eff_init_length < init_length:
            key = key[:, :, -eff_init_length:, :]
            value = value[:, :, -eff_init_length:, :]
        self.kv_buffers.prefill(key, value)
        self._next_token_pos = init_length
        self.next_position = eff_init_length % self.cache_length
        self.token_pos[:eff_init_length] = torch.arange(
            init_length - eff_init_length,
            init_length,
            dtype=self.token_pos.dtype,
            device=self.device,
        )
        if self._replay_log is not None:
            self._replay_log = LastRecentlyInsertedKVCacheReplayLog(
                token_chunks=[token_idx],
                cache_length=self.cache_length,
                batch_size=self.batch_size,
                n_query_groups=self.n_query_groups,
                dtype=self.dtype,
            )

    def token_positions(self) -> torch.Tensor:
        return self.token_pos[:self.current_length].reshape(1, 1, -1).expand(
            self.batch_size, self.n_query_groups, -1
        )

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        tk_p = bitsize_of(self.token_pos)
        sz_total, dct_sz = self.kv_buffers.size_estimate()
        return sz_total + tk_p, {**dct_sz, "token_pos": tk_p}

    @classmethod
    def size_estimate_apriori(cls, params: KVCacheParams, **kwargs) -> Tuple[int, Dict[str, int]]:
        """
        `cache_length` is required in `kwargs`. If `buffer_type` is given in
        `kwargs`, the size for this type is used, otherwise for the default
        type `DefaultKVCacheBuffers`.

        """
        buff_params = KVCacheBuffersParams.from_params(params)
        buffer_type = kwargs.get("buffer_type", DefaultKVCacheBuffers)
        sz_total, dct_sz = buffer_type.size_estimate_apriori(
            buff_params, cache_length=params.cache_length, **kwargs,
        )
        tk_p = params.cache_length * bits_for_torch_dtype(torch.int)
        return sz_total + tk_p, {**dct_sz, "token_pos": tk_p}

    def switch_replay_logging(self, status: bool):
        if status:
            if self._replay_log is None:
                self._replay_log = []  # Dummy, created later
        else:
            self._replay_log = None

    @property
    def do_replay_logging(self) -> bool:
        return self._replay_log is not None

    def get_replay_log(self) -> Optional[KVCacheReplayLog]:
        return self._replay_log

    def clone(self, device: Optional[torch.device] = None) -> KVCache:
        if self.kv_buffers.buffers_are_allocated:
            raise ValueError(f"Buffers must be deallocated, use `deallocate_buffers`")
        result = LastRecentlyInsertedKVCache(
            config=self.config,
            buffers=self.kv_buffers,
            block_idx=self.block_idx,
            **self._base_kwargs_for_clone(),
        )
        result._device = device
        return result
