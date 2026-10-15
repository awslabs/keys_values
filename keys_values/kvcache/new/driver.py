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
from typing import Optional, Tuple, Dict, Any, Callable

import torch

from keys_values.attention import (
    DefaultKeysAndValues,
    KeysAndValues,
    MultiHeadSelfAttention,
)
from keys_values.attention.use_eager_kernel import transform_mha_kwargs
from keys_values.config import Config
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.buffers import (
    KVCacheBuffers,
    DefaultKVCacheBuffers,
    KVCacheBuffersParams,
)
from keys_values.kvcache.new.basics import (
    KVCacheDriver,
    PositionsNewEntriesType,
)
from keys_values.utils import index_to_3d


class DefaultKVCacheDriver(KVCacheDriver):
    """
    Abstract KV cache driver implementation for the default case, where
    buffers are kept on a single device. We implement the SDPA part here.
    Methods :meth:`_prefill`, :meth:`_forward` remain abstract, they need a
    concrete buffer implementation. See :class:`KVCacheDriverWithBuffers`.
    """

    def __init__(
        self,
        config: Config,
        max_batch_size: int,
        cache_length: int,
        block_idx: int,
        dtype: Optional[torch.dtype] = None,
        head_size: Optional[int] = None,
        mha: Optional[MultiHeadSelfAttention] = None,
        **mha_kwargs,
    ):
        super().__init__(config)
        self._max_batch_size = max_batch_size
        self._block_idx = block_idx
        self._cache_length = cache_length
        if head_size is None:
            head_size = config.head_size
        self._head_size = head_size
        self.n_head = config.n_head
        self._dtype = dtype
        if mha is None:
            self.mha = MultiHeadSelfAttention(
                config,
                **transform_mha_kwargs(mha_kwargs, config),
            )
        else:
            self.mha = mha

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """
        Returns:
            Data type of the KV cache buffers. This can be set at construction
            or is otherwise set with the first :meth:`forward` call. Cannot
            be changed afterwards.

        """
        return self._dtype

    @property
    def cache_length(self) -> int:
        """
        Returns:
            Maximum cache length.

        """
        return self._cache_length

    @property
    def head_size(self) -> Optional[int]:
        return self._head_size

    @property
    def block_idx(self) -> int:
        return self._block_idx

    def prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self._prefill_or_forward_mha(
            query=query,
            key=key,
            value=value,
            token_idx=token_idx,
            input_pos=0,
        )[0]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        positions: PositionsNewEntriesType,
        token_positions: torch.Tensor,
        input_pos: int,
        return_attn_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert input_pos > 0  # Sanity check
        return self._prefill_or_forward_mha(
            query=query,
            key=key,
            value=value,
            token_idx=token_idx,
            input_pos=input_pos,
            positions=positions,
            token_positions=token_positions,
            return_attn_weights=return_attn_weights,
        )

    def _prefill_or_forward_mha(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        input_pos: int,
        positions: Optional[PositionsNewEntriesType] = None,
        token_positions: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if input_pos == 0:
            self._prefill(key, value, token_idx)
            k_and_v = DefaultKeysAndValues(key, value)
        else:
            # Extend KV cache and retrieve key, value tensors to be used.
            # Instead of asking for the key and value tensors as such,
            # `k_and_v` allows access to them.
            k_and_v = self._forward(key, value, positions)

        # Multi-head self-attention main computation
        attn_outputs, attn_weights = self.mha(
            query=query,
            k_and_v=k_and_v,
            block_idx=self.block_idx,
            input_pos=input_pos,
            return_attn_weights=return_attn_weights,
            token_positions=token_positions,
        )

        return attn_outputs, attn_weights

    def _prefill(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        """
        Buffer update part of :meth:`prefill`. The length of arguments must be
        `num <= max_prefill_length`. The effective batch size must be
        `batch_size <= max_batch_size`. This batch size is then fixed for
        subsequent calls of :meth:`forward`.

        Args:
            key: Prefill keys, `(batch_size, n_query_groups, num, head_size)`
            value: Prefill values, `(batch_size, n_query_groups, num, head_size)`
            token_idx: Token indices of input sequence, `(batch_size, num)`.

        """
        raise NotImplementedError

    def _forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: PositionsNewEntriesType,
    ) -> KeysAndValues:
        """
        Buffer update part of :meth:`forward` (`input_pos > 0`.) Namely, `key`
        and `value` are written into the cache, overwriting entries at
        `positions`. An object is returned which provides read access to the
        full keys and values buffers.

        Args:
            key: New keys, `(batch_size, n_query_groups, num, head_size)`,
                where `1 <= num <= max_forward_length()`
            value: New values, `(batch_size, n_query_groups, num, head_size)`
            positions: Positions for the new entries

        Returns:
            key_cached, value_cached, `(batch_size, n_query_groups, T,
                head_size)`, where `T <= cache_length` is the current cache
                length

        """
        raise NotImplementedError

    def set_seq_length(self, seq_length: int) -> None:
        self.mha.set_seq_length(seq_length)

    def _base_kwargs_for_clone(self) -> Dict[str, Any]:
        """
        Supports :meth:`clone` implementations in subclasses.

        Returns:
            Keyword arguments for calling the constructor in :meth:`clone`

        """
        return dict(
            config=self.config,
            max_batch_size=self.max_batch_size,
            cache_length=self.cache_length,
            block_idx=self.block_idx,
            dtype=self.dtype,
            head_size=self.head_size,
            mha=self.mha,
        )


NOT_NEEDED_ARGS = ("max_batch_size", "cache_length", "dtype")


class KVCacheDriverWithBuffers(DefaultKVCacheDriver):
    """
    Concrete implementation of :class:`DefaultKVCacheDriver`, where K and
    V buffers are provided by a :class:`KVCacheBuffers` object `buffers`.
    This works for buffers stored on a single device.
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
        if buffers.device is not None:
            self.set_or_check_device(buffers.device)

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
        self.kv_buffers.prefill(key, value)
        if self._checkpoint_hook is not None:
            self._chunk_idx = 0

    def _forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: PositionsNewEntriesType,
    ) -> KeysAndValues:
        if not self.buffers_are_allocated:
            raise IndexError("Buffers are not allocated. Call 'prefill' first.")
        if self._checkpoint_hook is not None:
            if self._chunk_idx is None:
                raise IndexError("Chunk index not initialized. Call 'prefill' first.")
            self._chunk_idx += 1
            self._checkpoint_hook(self.kv_buffers, self._chunk_idx)
        if not isinstance(positions, list):
            assert isinstance(positions, torch.Tensor)
            positions = [(positions, 0, key.shape[2])]
        k_and_v = None
        for index, off, num in positions:
            if index.ndim == 1:
                index = index_to_3d(index, self.batch_size, self.n_query_groups)
            _key = key[:, :, off : (off + num), :]
            _value = value[:, :, off : (off + num), :]
            index_sz = index.shape[-1]
            if index_sz < num:
                raise ValueError(
                    f"index.shape = {index.shape}, num = {num}: Index is too short"
                )
            if index_sz > num:
                key_cp, value_cp = self.kv_buffers.get_slots(index[:, :, num:])
                _key = torch.cat((key_cp, _key), dim=-2)
                _value = torch.cat((value_cp, _value), dim=-2)
            k_and_v = self.kv_buffers.forward(
                positions=index,
                key=_key,
                value=_value,
            )
        return k_and_v

    def reset(self) -> None:
        self.kv_buffers.reset()

    def get_keys_values(self) -> Optional[KeysAndValues]:
        return self.kv_buffers.get_keys_values()

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

    def clone(self) -> "KVCacheDriver":
        return KVCacheDriverWithBuffers(**self._base_kwargs_for_clone())

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        return self.kv_buffers.size_estimate()

    @classmethod
    def size_estimate_apriori(
        cls, params: KVCacheParams, **kwargs
    ) -> Tuple[int, Dict[str, int]]:
        buff_params = KVCacheBuffersParams.from_params(params)
        buffer_type = kwargs.get("buffer_type", DefaultKVCacheBuffers)
        return buffer_type.size_estimate_apriori(
            buff_params,
            cache_length=params.cache_length,
            **kwargs,
        )
