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
from typing import Optional, Tuple, Dict, Callable, List, Any, Union, Type

import torch

from keys_values.config import Config

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
from keys_values.utils import index_to_3d, bits_for_torch_dtype, bitsize_of

NOT_NEEDED_ARGS = ("max_batch_size", "cache_length", "dtype")


class KVCacheWithBuffersState:
    """
    Represents state of a :class:`KVCacheWithBuffers` cache, except the
    buffers.

    Part of the state is not variable. This is used for validation, to
    avoid that :meth:`KVCacheWithBuffers.switch_buffers` introduces mistakes.

    The remaining fields are variable, overwriting state variables when calling
    :meth:`KVCacheWithBuffers.switch_buffers`.
    """

    def __init__(
        self,
        n_head: int,
        n_query_groups: int,
        head_size: int,
        max_batch_size: int,
        cache_length: int,
        block_idx: int,
        dtype: Optional[torch.dtype],
        device: Optional[torch.device],
        input_pos: int,
    ):
        self.n_head = n_head
        self.n_query_groups = n_query_groups
        self.head_size = head_size
        self.max_batch_size = max_batch_size
        self.cache_length = cache_length
        self.block_idx = block_idx
        self.dtype = dtype
        self.device = device
        self.input_pos = input_pos

    def asdict(self) -> Dict[str, Any]:
        return dict(
            n_head=self.n_head,
            n_query_groups=self.n_query_groups,
            head_size=self.head_size,
            max_batch_size=self.max_batch_size,
            cache_length=self.cache_length,
            block_idx=self.block_idx,
            dtype=self.dtype,
            device=self.device,
            input_pos=self.input_pos,
        )

    def _is_compatible(
        self,
        state_or_buffers: Union["KVCacheWithBuffersState", KVCacheBuffers],
        extra_names: Optional[Tuple[str, ...]] = None,
    ) -> Optional[str]:
        """
        Checks whether `self` and `state` are the same in all non-variable
        fields.

        Args:
            state_or_buffers: State or buffers to compare with

        Returns:
            Name of field with different values, or `None` if all values are
            the same.

        """
        names = (
            "n_query_groups",
            "head_size",
            "max_batch_size",
            "cache_length",
            "dtype",
        )
        if not isinstance(state_or_buffers, KVCacheBuffers):
            names += (
                "n_head",
                "block_idx",
                "device",
            )
            if extra_names is not None:
                names += extra_names
        for name in names:
            if getattr(self, name) != getattr(state_or_buffers, name):
                return name
        return None

    def is_compatible(
        self,
        state_or_buffers: Union["KVCacheWithBuffersState", KVCacheBuffers],
    ) -> Optional[str]:
        return self._is_compatible(state_or_buffers)


class KVCacheWithBuffers(DefaultKVCache):
    """
    Base class of all KV caches supported by KV cache buffers of type
    :class:`KVCacheBuffers`. We recommend that all KV caches separate out
    buffers, in order to keep selection orthogonal to content compression
    (e.g., by quantization). Also, buffers can be deallocated to free up
    GPU memory, and reallocated later on.

    Checkpoint hook:

    We also support a checkpoint hook, in the form of `checkpoint_hook`. If
    this is given, we count the number of :meth:`forward` calls starting with
    a call to :meth:`prefill` (the prefill call is 0, first forward is 1, ...)
    in `chunk_idx`. We then call
    `checkpoint_hook(self.kv_buffers, self.chunk_idx)` at the start of
    :meth:`forward`. An important use case is KV cache checkpointing for
    gradient computation.

    Buffers, dependence on batch size:

    Subclasses support buffer allocation on demand, as well as their
    deallocation, so that GPU memory is freed for other uses. Buffers are
    allocated (if not already present) with a call of :meth:`_prefill`,
    and the effective batch size `batch_size` is used then. Also, if
    buffers exist, but differ in shape due to `batch_size`, they are
    reallocated. This allows us to support different batch sizes (e.g.
    for training and validation), without having to allocate buffers
    for `max_batch_size`.

    Switching buffers, cache state:

    The cache can be switched to different buffers with :meth:`switch_buffers`.
    This includes (optionally) a cache state, containing variables determining
    the current cache state other than the buffers.

    An important use case for buffer switching is if several token sequences
    are to be generated for the same prompt, or if both a sample-based metric
    and a loss function are to be computed for the same (long) prompt. In
    this case, we want to process the prompt only once. Use
    :class:`ReplayKVCacheBuffers` buffers for the generation.
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
            base_kwargs: Further args passed to superclass constructor

        """
        for name in NOT_NEEDED_ARGS:
            if name in base_kwargs:
                print(f"Removing {name} from base_kwargs (taken from buffers")
                base_kwargs.pop(name)
        super().__init__(
            config=config,
            max_batch_size=buffers.max_batch_size,
            cache_length=buffers.cache_length,
            block_idx=block_idx,
            dtype=buffers.dtype,
            **base_kwargs,
        )
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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def get_keys_values(self) -> Optional[KeysAndValues]:
        return self.kv_buffers.get_keys_values()

    def switch_buffers(
        self,
        new_buffers: KVCacheBuffers,
        cache_state: Optional[KVCacheWithBuffersState] = None,
    ):
        """
        Switches cache buffers to `new_buffers`. If `cache_state` is given,
        the state variables of the cache are also changed, otherwise they
        remain the same. The new buffers `new_buffers` must be compatible with
        `cache_state` or the current state.

        The content of the current buffers is NOT copied to `new_buffers`. If
        this is intended, it must be done before.

        Args:
            new_buffers: Buffers to be used from now on
            cache_state: If given, the current cache state is overwritten by
                this one

        """
        if cache_state is not None and not isinstance(cache_state, self._state_type()):
            raise TypeError(
                f"type(cache_state) = {type(cache_state)}, must be {str(self._state_type())}"
            )
        self_state = self.get_state()
        name = self_state.is_compatible(new_buffers)
        if name is not None:
            raise ValueError(f"new_buffers not compatible: {name}")
        if cache_state is not None:
            name = self_state.is_compatible(cache_state)
            if name is not None:
                raise ValueError(f"cache_state not compatible: {name}")
            self._input_pos = cache_state.input_pos
        self.kv_buffers = new_buffers

    def get_state(self) -> KVCacheWithBuffersState:
        return KVCacheWithBuffersState(
            n_head=self.n_head,
            n_query_groups=self.n_query_groups,
            head_size=self.head_size,
            max_batch_size=self.max_batch_size,
            cache_length=self.cache_length,
            block_idx=self.block_idx,
            dtype=self.dtype,
            device=self.device,
            input_pos=self.input_pos,
        )

    @staticmethod
    def _state_type() -> Type[KVCacheWithBuffersState]:
        return KVCacheWithBuffersState

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

    def _default_device_for_new_params(self) -> torch.device:
        """
        Returns:
            Device on which new parameters should be created first.

        """
        device = self.kv_buffers.device
        if device is None:
            device = torch.get_default_device()
        return device

    def _reset(self):
        self.kv_buffers.reset()

    def _base_kwargs_for_clone(self) -> Dict[str, Any]:
        """
        Supports :meth:`clone` implementations in subclasses.
        Note that the copy created by :meth:`clone` uses the same `self.mha`
        object (shallow copy).

        Returns:
            Keyword arguments for calling the constructor in :meth:`clone`

        """
        if self.kv_buffers.buffers_are_allocated:
            raise ValueError(f"Buffers must be deallocated, use `deallocate_buffers`")
        result = super()._base_kwargs_for_clone()
        for name in NOT_NEEDED_ARGS:
            del result[name]
        result["buffers"] = self.kv_buffers
        return result


class DenseKVCache(KVCacheWithBuffers):
    """
    Key-value cache for dense attention. Key and value tensors for all
    past tokens are maintained. This cache can only process as many tokens
    as its cache length. It requires a lot of memory, so can only be used for
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
        """
        Creates cache with default buffers (no quantization).

        Args:
            config: Model config
            max_batch_size: Inference batch size (maximum)
            cache_length: Number of slots in cache. This is also the maximum
                number of tokens which can be stored for this cache
            block_idx: Index of model block (or layer). Multi-head attention
                needs to know this.
            device: Device for buffers. If not given, it is set with the
                first :meth:`forward` call, based on the input arguments.
            dtype: Data type for buffers. If not given, it is set with the
                first :meth:`forward` call, based on the input arguments.
            base_kwargs: Extra keyword arguments for cache and default buffer

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
        return DenseKVCache(config, buffers, block_idx, **base_kwargs)

    def max_forward_length(self) -> int:
        return self.cache_length - self.input_pos

    def _forward_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> KeysAndValues:
        if self._replay_log is not None:
            if not isinstance(self._replay_log, DefaultKVCacheReplayLog):
                raise IndexError(
                    "Cannot switch on replay logging in the middle of inference run. Call 'prefill'."
                )
            self._replay_log.append_token_chunk(token_idx)
        np = self.input_pos
        num = key.shape[2]
        return self.kv_buffers.forward(
            positions=(np, np + num),
            key=key,
            value=value,
        )

    def _prefill_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        self.kv_buffers.prefill(key, value)
        if self._replay_log is not None:
            self._replay_log = DefaultKVCacheReplayLog(
                token_chunks=[token_idx],
                cache_length=self.cache_length,
                max_prefill_length=self.max_prefill_length,
            )

    def token_positions(self) -> torch.Tensor:
        device = torch.get_default_device() if self.device is None else self.device
        return index_to_3d(
            torch.arange(self.input_pos, device=device),
            self.batch_size,
            self.n_query_groups,
        )

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        return self.kv_buffers.size_estimate()

    @classmethod
    def size_estimate_apriori(
        cls, params: KVCacheParams, **kwargs
    ) -> Tuple[int, Dict[str, int]]:
        """
        `cache_length` is required in `kwargs`. If `buffer_type` is given in
        `kwargs`, the size for this type is used, otherwise for the default
        type `DefaultKVCacheBuffers`.

        """
        buff_params = KVCacheBuffersParams.from_params(params)
        buffer_type = kwargs.get("buffer_type", DefaultKVCacheBuffers)
        return buffer_type.size_estimate_apriori(
            buff_params,
            cache_length=params.cache_length,
            **kwargs,
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
        return copy.copy(self._replay_log)

    def clone(self) -> KVCache:
        return DenseKVCache(**self._base_kwargs_for_clone())

    def switch_buffers(
        self,
        new_buffers: KVCacheBuffers,
        cache_state: Optional[KVCacheWithBuffersState] = None,
    ):
        super().switch_buffers(new_buffers, cache_state)
        self._replay_log = None


class LastRecentlyInsertedKVCacheReplayLog(DefaultKVCacheReplayLog):
    """
    Replay log for :class:`LastRecentlyInsertedKVCache`.

    """

    def __init__(
        self,
        token_chunks: List[torch.Tensor],
        cache_length: int,
        batch_size: int,
        n_query_groups: int,
        init_grace_tokens: int,
    ):
        super().__init__(
            token_chunks,
            cache_length,
            max_prefill_length=cache_length,
            grace_period=0,
        )
        self._shape = (batch_size, n_query_groups)
        self.init_grace_tokens = init_grace_tokens

    def extract_index(
        self,
        input_pos: int,
        num: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        seq_length = self.__len__()
        if num <= 0 or input_pos < 0 or input_pos > seq_length - num:
            raise ValueError(
                f"token_pos = {input_pos}, num = {num}, seq_length = {seq_length}: Out of range"
            )
        if input_pos < self.cache_length:
            raise ValueError(
                f"input_pos = {input_pos} must be >= {self.cache_length} = cache_length"
            )
        if device is None:
            device = torch.get_default_device()
        start = self.init_grace_tokens
        mod = self.cache_length - start
        current = (input_pos - self.cache_length) % mod + start
        result = positions_wrap_around(
            num=num,
            current=current,
            start=start,
            end=self.cache_length,
            batch_size=self._shape[0],
            n_query_groups=self._shape[1],
            device=device,
            return_tensor=True,
            dtype=dtype,
        )
        return result


class LastRecentlyInsertedKVCacheState(KVCacheWithBuffersState):
    def __init__(
        self,
        n_head: int,
        n_query_groups: int,
        head_size: int,
        max_batch_size: int,
        cache_length: int,
        block_idx: int,
        dtype: Optional[torch.dtype],
        device: Optional[torch.device],
        input_pos: int,
        init_grace_tokens: int,
        next_position: int,
    ):
        super().__init__(
            n_head,
            n_query_groups,
            head_size,
            max_batch_size,
            cache_length,
            block_idx,
            dtype,
            device,
            input_pos,
        )
        self.init_grace_tokens = init_grace_tokens
        self.next_position = next_position

    def asdict(self) -> Dict[str, Any]:
        return dict(
            super().asdict(),
            init_grace_tokens=self.init_grace_tokens,
            next_position=self.next_position,
        )

    def is_compatible(
        self,
        state_or_buffers: Union["KVCacheWithBuffersState", KVCacheBuffers],
    ) -> Optional[str]:
        return self._is_compatible(state_or_buffers, ("init_grace_tokens",))


class LastRecentlyInsertedKVCache(KVCacheWithBuffers):
    """
    Baseline key-value cache which stores the last recently inserted
    `cache_length` key and value tensors. When the cache is full,
    those slots are overwritten whose content is in the cache for the
    longest time.

    If `init_grace_tokens > 0`, this number of initial keys and values are kept
    in the cache indefinitely. Use this in order to cater for initial prompt
    tokens which may be important.
    """

    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        init_grace_tokens: int = 0,
        **base_kwargs,
    ):
        if init_grace_tokens < 0:
            raise ValueError(f"init_grace_tokens={init_grace_tokens}, must be >= 0")
        super().__init__(config, buffers, block_idx, **base_kwargs)
        self.init_grace_tokens = init_grace_tokens
        # Note: We could generate `token_pos` on the fly from `input_pos`
        # and `next_position`, but it is simpler just to maintain it.
        device = self._default_device_for_new_params()
        self.register_buffer(
            "token_pos",
            torch.zeros(buffers.cache_length, device=device, dtype=torch.int),
            persistent=False,
        )
        # Position of first slot to overwrite with next :meth:`forward`
        self.next_position = None
        self._replay_log = None

    @staticmethod
    def from_config(
        config: Config,
        max_batch_size: int,
        cache_length: int,
        block_idx: int,
        init_grace_tokens: int = 0,
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
            block_idx: Block index
            init_grace_tokens: This number of initial keys and values are
                kept in the cache indefinitely. Defaults to 0.
            device: Device for buffers. If not given, it is set with the
                first :meth:`forward` call, based on the input arguments.
            dtype: Data type for buffers. If not given, it is set with the
                first :meth:`forward` call, based on the input arguments.
            base_kwargs: Extra keyword arguments for cache and default buffer

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
            config,
            buffers,
            block_idx,
            init_grace_tokens=init_grace_tokens,
            **base_kwargs,
        )

    @classmethod
    def _parameter_names(cls) -> List[str]:
        return super()._parameter_names() + ["token_pos"]

    def max_forward_length(self) -> int:
        return self.cache_length - self.init_grace_tokens

    def _forward_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> KeysAndValues:
        num = key.shape[2]
        start = self.init_grace_tokens
        positions = positions_wrap_around(
            num=num,
            current=self.next_position,
            start=start,
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
        ntp = self.input_pos
        self.token_pos[np : (np + num1)] = torch.arange(
            ntp, ntp + num1, device=self.device, dtype=torch.int
        )
        if diff > 0:
            self.token_pos[start : (start + diff)] = torch.arange(
                ntp + num1, ntp + num, device=self.device, dtype=torch.int
            )
        np += num
        diff = np - self.cache_length
        if diff >= 0:
            np = diff + start
        self.next_position = np
        if self._replay_log is not None:
            if not isinstance(self._replay_log, DefaultKVCacheReplayLog):
                raise IndexError(
                    "Cannot switch on replay logging in the middle of inference run. Call 'prefill'."
                )
            self._replay_log.append_token_chunk(token_idx)
        return k_and_v

    def _prefill_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        init_length = key.shape[2]
        self.kv_buffers.prefill(key, value)
        self.next_position = init_length % self.cache_length
        self.token_pos[:init_length] = torch.arange(
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
                init_grace_tokens=self.init_grace_tokens,
            )

    def token_positions(self) -> torch.Tensor:
        result = index_to_3d(
            self.token_pos[: self.current_length],
            self.batch_size,
            self.n_query_groups,
        )
        return result

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        tk_p = bitsize_of(self.token_pos)
        sz_total, dct_sz = self.kv_buffers.size_estimate()
        return sz_total + tk_p, {**dct_sz, "token_pos": tk_p}

    @classmethod
    def size_estimate_apriori(
        cls, params: KVCacheParams, **kwargs
    ) -> Tuple[int, Dict[str, int]]:
        """
        `cache_length` is required in `kwargs`. If `buffer_type` is given in
        `kwargs`, the size for this type is used, otherwise for the default
        type `DefaultKVCacheBuffers`.

        """
        buff_params = KVCacheBuffersParams.from_params(params)
        buffer_type = kwargs.get("buffer_type", DefaultKVCacheBuffers)
        sz_total, dct_sz = buffer_type.size_estimate_apriori(
            buff_params,
            cache_length=params.cache_length,
            **kwargs,
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
        return copy.copy(self._replay_log)

    def _base_kwargs_for_clone(self) -> Dict[str, Any]:
        result = super()._base_kwargs_for_clone()
        result["init_grace_tokens"] = self.init_grace_tokens
        return result

    def clone(self) -> KVCache:
        return LastRecentlyInsertedKVCache(**self._base_kwargs_for_clone())

    def get_state(self) -> KVCacheWithBuffersState:
        super_state = super().get_state()
        return LastRecentlyInsertedKVCacheState(
            **super_state.asdict(),
            init_grace_tokens=self.init_grace_tokens,
            next_position=self.next_position,
        )

    def _create_token_pos(self):
        kwargs = dict(dtype=self.token_pos.dtype, device=self.token_pos.device)
        cl = self.cache_length
        ip = self.input_pos
        if ip <= cl:
            self.token_pos[:ip] = torch.arange(ip, **kwargs)
        else:
            igt = self.init_grace_tokens
            np = self.next_position
            assert np >= igt, (np, igt)  # Sanity check
            if igt > 0:
                self.token_pos[:igt] = torch.arange(igt, **kwargs)
            rsz = cl - igt  # Remaining entries to set
            entries = torch.arange(ip - rsz, ip, **kwargs)
            sz = cl - np
            self.token_pos[np:] = entries[:sz]
            rsz -= sz  # Remaining entries to set
            if rsz > 0:
                self.token_pos[igt : (igt + rsz)] = entries[sz:]

    def switch_buffers(
        self,
        new_buffers: KVCacheBuffers,
        cache_state: Optional[KVCacheWithBuffersState] = None,
    ):
        super().switch_buffers(new_buffers, cache_state)
        if cache_state is not None:
            self.next_position = cache_state.next_position
            self._create_token_pos()
        self._replay_log = None

    @staticmethod
    def _state_type() -> Type[KVCacheWithBuffersState]:
        return LastRecentlyInsertedKVCacheState
