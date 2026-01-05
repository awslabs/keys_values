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
from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass

import torch

from litgpt.config import Config

from keys_values.attention import KeysAndValues, DefaultKeysAndValues
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.utils import (
    bitsize_of,
    bits_for_torch_dtype,
)
from keys_values.utils import expand_index


@dataclass(frozen=True)
class KVCacheBuffersParams:
    """
    Note that `device` need not be set. As long as a buffer remains not
    allocated, its device can be unspecified. It is then determined upon
    allocation, based on input arguments to the first KV cache call.

    `dtype` need not be set either. In this case, the buffer dtype is
    determined from input arguments to the first KV cache call. This dtype
    cannot be changed later.

    """
    max_batch_size: int
    n_query_groups: int
    head_size: int
    device: Optional[torch.device]
    dtype: Optional[torch.dtype]

    @staticmethod
    def from_config(
        config: Config,
        max_batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCacheBuffersParams":
        return KVCacheBuffersParams(
            max_batch_size=max_batch_size,
            n_query_groups = config.n_query_groups,
            head_size = config.head_size,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def from_params(params: KVCacheParams) -> "KVCacheBuffersParams":
        return KVCacheBuffersParams(
            max_batch_size=params.max_batch_size,
            n_query_groups=params.n_query_groups,
            head_size=params.head_size,
            device=None,
            dtype=params.dtype,
        )


PositionsType = Union[torch.Tensor, Tuple[int, int]]


def positions_wrap_around(
    num: int,
    current: int,
    start: int,
    end: int,
    batch_size: int,
    n_query_groups: int,
    device: torch.device,
    return_tensor: bool = False,
) -> PositionsType:
    """
    Returns positions which form a range of length `num`, starting from
    `current`, and lying in `[start, end)`. If `current + num > end`, the range
    wraps around and consists of two parts.

    Args:
        num: Length of positions
        current: First entry of positions
        start: Start of domain range
        end: End of domain range (not inclusive)
        batch_size: Dimension size for expanding `positions`
        n_query_groups: Dimension size for expanding `positions`
        device: Device used for index returned
        return_tensor: If `True`, returns a `torch.Tensor` even if range has
            one part only

    Returns:
        positions, which is either a tuple `(current, current + num)` if
        `current + num <= end` and `return_tensor=False`, or a position index
        of shape `(batch_size, n_query_groups, num)`, which is broadcast
        from shape `(1, 1, num)`.

    """
    assert start <= current < end
    assert num <= end - start
    num1 = min(num, end - current)
    diff = num - num1
    if diff == 0 and not return_tensor:
        return current, current + num
    if device is None:
        raise ValueError("device must be given")
    kwargs = dict(device=device, dtype=torch.int64)
    positions = torch.arange(current, current + num1, **kwargs)
    if diff > 0:
        positions = torch.cat(
            (positions, torch.arange(start, start + diff, **kwargs))
        )
    return positions.view(1, 1, -1).expand(batch_size, n_query_groups, -1)


class KVCacheBuffers(torch.nn.Module):
    """
    Base class for key-value cache buffers.

    In general, there are two buffers (keys, values) of shape
    `(batch_size, n_query_groups, cache_length, head_size)`. Instead of
    allocating them in the :class:`KVCache` class, we abstract them here. The
    reason is that some KV caches maintain compressed (e.g., quantized)
    versions of these buffers.

    The general structure for quantized buffers allocates memory for the
    quantized buffers as members here, and several objects (for different model
    layers) share a single object with normal arrays, which is used to mediate
    between the user and quantized storage.

    Deallocating and allocating buffers:

    By default, allocation is delayed until first usage. In this case, the
    device for the buffers is determined from the input arguments.
    Buffers can be deallocated by calling :meth:`deallocate`, to save GPU
    memory when they are not needed. In general, buffers are allocated when
    not present with the next recent call of :meth:`prefill`. There are some
    advantages:

    * If inference with KV caches is iterated with other operations requiring
        a lot of device memory, the available device memory can be shared.
    * Delayed buffer allocation works better with parallel training
        frameworks, which may move models to a device after creation. A
        buffer object can change its device if buffers are deallocated.

    """
    def __init__(
        self,
        params: KVCacheBuffersParams,
        cache_length: int,
    ):
        super().__init__()
        self.max_batch_size = params.max_batch_size
        self.n_query_groups = params.n_query_groups
        self.cache_length = cache_length
        self.head_size = params.head_size
        self.dtype = params.dtype
        self.batch_size = None
        # Number of slots which are occupied. Grows until `cache_length`, then
        # stays there. Initialized by :meth:`prefill`.
        self.current_length = None

    @property
    def device(self) -> Optional[torch.device]:
        """
        Returns:
            Device the KV cache buffers are kept on. If buffers are not
            allocated, this can be `None`.

        """
        raise NotImplementedError()

    def get_slots(
        self,
        positions: PositionsType,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use :meth:`get_keys_values()` to obtain all or most of the buffers, as
        this can be more economical.

        Args:
            positions: Slot positions, either `(start, end)` with
                `num = end - start` or batched of shape
                `(batch_size, n_query_groups, num)`

        Returns:
            key, value, `(batch_size, n_query_groups, num, head_size)`

        """
        raise NotImplementedError()

    def set_slots(
        self,
        positions: PositionsType,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """
        Use :meth:`prefill` to set all or most slots, as this can be more
        economical.

        Args:
            positions: Slot positions, either `(start, end)` with
                `num = end - start` or batched of shape
                `(batch_size, n_query_groups, num)`
            key: Keys to write, `(batch_size, n_query_groups, num, head_size)`
            value: Values to write, `(batch_size, n_query_groups, num, head_size)`

        """
        raise NotImplementedError()

    def forward(
        self,
        positions: PositionsType,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> KeysAndValues:
        """
        First, `key` and `value` are written to slot positions `positions`. Then,
        the full buffers are returned.

        Args:
            positions: Slot positions, either `(start, end)` with
                `num = end - start` or batched of shape
                `(batch_size, n_query_groups, num)`
            key: New keys, `(batch_size, n_query_groups, num, head_size)`
            value: New values, `(batch_size, n_query_groups, num, head_size)`

        Returns:
            key_cached, value_cached, `(batch_size, n_query_groups, T,
                head_size)`, where `T <= cache_length` is the current cache
                length

        """
        num = key.shape[2]
        self.current_length = min(self.cache_length, self.current_length + num)
        return self._forward(positions, key, value)

    def _forward(
        self,
        positions: PositionsType,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> KeysAndValues:
        """
        Does job of :meth:`forward`. Note that `current_length` is already
        increased before.

        """
        raise NotImplementedError()

    def get_keys_values(self) -> Optional[KeysAndValues]:
        """
        Same response as :meth:`forward`, but buffers are not modified.

        Returns:
            Full keys and values, `(batch_size, n_query_groups, T,
                head_size)`, where `T <= cache_length` is the current cache
                length.

        """
        raise NotImplementedError()

    def prefill(self, key: torch.Tensor, value: torch.Tensor):
        """
        Prefills buffers with key and value tensors. The length must be
        `T <= max_prefill_length`. The effective batch size must be
        `batch_size <= batch_size`.

        If buffers are not allocated, this is done here, using `key.device`
        as device.

        Args:
            key: Prefill keys, `(batch_size, n_query_groups, T, head_size)`
            value: Prefill values, `(batch_size, n_query_groups, T, head_size)`
        """
        self.batch_size, self.current_length = self._check_prefill(key, value)
        self._prefill(key, value)

    def _check_prefill(self, key: torch.Tensor, value: Optional[torch.Tensor]) -> Tuple[int, int]:
        if key.dim() != 4:
            raise ValueError("key must have 4 dimensions")
        batch_size, _, init_length, _ = key.shape
        if not (1 <= batch_size <= self.max_batch_size):
            raise ValueError(f"key.shape[0] = {batch_size} must be in [1, {self.max_batch_size}]")
        shape = (batch_size, self.n_query_groups, init_length, self.head_size)
        if key.shape != shape or (value is not None and value.shape != shape):
            msg = f"Shapes of key, value must be {shape}, but key.shape = {key.shape}"
            if value is not None:
                msg += f", value.shape = {value.shape}"
            raise ValueError(msg)
        return batch_size, init_length

    def _prefill(self, key: torch.Tensor, value: torch.Tensor):
        """
        For caches which support late buffer allocation and deallocation, this
        method needs to do the allocation based on `batch_size` (not
        `max_batch_size`), and also reallocate existing buffers if they do not
        fit `batch_size`.

        """
        raise NotImplementedError()

    def write_back(self):
        """
        This method should be called at the end of a loop over all layers of
        a model. Buffers involving quantization may have delayed write-back
        of information, which otherwise may not happen.

        """
        pass

    def deallocate(self):
        """
        Deallocates the buffers. They are automatically reallocated with the
        next :meth:`prefill` call. Use this method only if device memory is
        scarce and is needed by other operations in between inference calls.

        """
        self._deallocate()
        self.current_length = 0

    def _deallocate(self):
        raise NotImplementedError()

    @property
    def buffers_are_allocated(self) -> bool:
        """
        Returns:
            Are buffers currently allocated?

        """
        raise NotImplementedError()

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        """
        Estimate of storage taken by the buffers.

        Note that if in the case of quantized storage, a shared object is used
        for the de-quantized content, this must not be counted here.

        Returns:
            num_bits_total, bits_by_part (unit is bit)
        """
        raise NotImplementedError()

    @staticmethod
    def size_estimate_apriori(params: KVCacheBuffersParams, **kwargs) ->Tuple[int, Dict[str, int]]:
        """
        Same semantics as :meth:`size_estimate`, but can be called without a
        cache being created. Results may not be exactly the same, but should
        be very close.

        Args:
            params: KV cache buffers parameters
            **kwargs: Extra arguments (optional)

        Returns:
            num_bits_total, bits_by_part (unit is bit)
        """
        raise NotImplementedError()

    def get_params(self) -> KVCacheBuffersParams:
        return KVCacheBuffersParams(
            max_batch_size=self.max_batch_size,
            n_query_groups=self.n_query_groups,
            head_size=self.head_size,
            device=self.device,
            dtype=self.dtype,
        )


class DefaultKVCacheBuffers(KVCacheBuffers):
    """
    Default implementation, where KV cache buffers are simply allocated as
    such (no compression or clever storage).

    """
    def __init__(
        self,
        params: KVCacheBuffersParams,
        cache_length: int,
        allocate_buffers: bool = False,
        debug_device_warning: bool = False,
    ):
        super().__init__(params, cache_length)
        self.k = None
        self.v = None
        if allocate_buffers:
            self._allocate_buffers(params.device)
        self._debug_device_warning = debug_device_warning

    @property
    def device(self) -> Optional[torch.device]:
        return None if self.k is None else self.k.device

    def _allocate_buffers(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            device: If given, the buffer must be on this device. It is
                reallocated if not.
            dtype: If `self.dtype is None`, it is set to this value. Otherwise,
                this value is ignored.

        """
        if device is None:
            if self.buffers_are_allocated:
                device = self.device
            else:
                device = torch.get_default_device()
        if self.dtype is None:
            self.dtype = dtype
        batch_size = self.batch_size
        if not self.buffers_are_allocated:
            if batch_size is None:
                batch_size = self.max_batch_size
        elif batch_size is not None:
            if self.k.shape[0] >= batch_size and device == self.device:
                # Buffer exists and is large enough: No need to reallocate
                batch_size = None
            elif self.k.shape[0] < batch_size:
                print(f"Batch size increased from {self.k.shape[0]} to {batch_size}: Re-allocating buffers")
        if batch_size is not None:
            shape = (batch_size, self.n_query_groups, self.cache_length, self.head_size)
            self.k = torch.zeros(shape, device=device, dtype=self.dtype)
            self.v = torch.zeros(shape, device=device, dtype=self.dtype)

    def _deallocate(self):
        if self.k is not None:
            del self.k
            self.k = None
            del self.v
            self.v = None

    @property
    def buffers_are_allocated(self) -> bool:
        return self.k is not None

    def _assert_buffers_allocated_and_initialized(self):
        assert self.buffers_are_allocated, "Buffers are not allocated. Call 'prefill' first"

    def get_slots(
        self,
        positions: PositionsType,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._assert_buffers_allocated_and_initialized()
        positions = self._check_positions(positions)
        if isinstance(positions, torch.Tensor):
            index = expand_index(positions, self.head_size)
            res_k = self.k[:self.batch_size, ...].gather(-2, index)
            res_v = self.v[:self.batch_size, ...].gather(-2, index)
        else:
            start, end = positions
            res_k = self.k[:self.batch_size, :, start:end, :]
            res_v = self.v[:self.batch_size, :, start:end, :]
        return res_k, res_v

    def set_slots(
        self,
        positions: PositionsType,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        self._assert_buffers_allocated_and_initialized()
        if self._debug_device_warning and key.device != self.device:
            print(f"WARNING DefaultKVCacheBuffers.set_slots: {key.device} -> {self.device}")
        key = key.to(device=self.device, dtype=self.dtype)
        value = value.to(device=self.device, dtype=self.dtype)
        positions = self._check_positions(positions)
        if isinstance(positions, torch.Tensor):
            index = expand_index(positions, self.head_size)
            self.k[:self.batch_size, ...].scatter_(-2, index, key)
            self.v[:self.batch_size, ...].scatter_(-2, index, value)
        else:
            start, end = positions
            self.k[:self.batch_size, :, start:end, :] = key
            self.v[:self.batch_size, :, start:end, :] = value

    def _forward(
        self,
        positions: PositionsType,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> KeysAndValues:
        self.set_slots(positions, key, value)
        return self.get_keys_values()

    def get_keys_values(self) -> Optional[KeysAndValues]:
        if not self.buffers_are_allocated or self.batch_size is None or self.current_length is None:
            return None
        else:
            return DefaultKeysAndValues(
                self.k[:self.batch_size, :, :self.current_length, :],
                self.v[:self.batch_size, :, :self.current_length, :],
            )

    def _prefill(self, key: torch.Tensor, value: torch.Tensor):
        if self.buffers_are_allocated and key.device != self.device:
            raise ValueError(f"key.device = {key.device}, must be {self.device}")
        # Note: `self.batch_size` already set here
        self._allocate_buffers(device=key.device, dtype=key.dtype)
        # Initialize cache buffers
        init_length = key.shape[2]
        self.k[:self.batch_size, :, :init_length, :] = key.to(dtype=self.dtype)
        self.v[:self.batch_size, :, :init_length, :] = value.to(dtype=self.dtype)

    def prefill_from_keys_values(self, k_and_v: KeysAndValues):
        """
        Same as :meth:`prefill`, but using `k_and_v`. This can save memory,
        because `k_and_v.keys()` and `k_and_v.values()` are accessed one after
        the other.

        If `keys.shape[2] > self.cache_length`, only the initial part is copied.

        """
        keys = k_and_v.keys()
        if self.buffers_are_allocated and keys.device != self.device:
            raise ValueError(f"k_and_v.keys.device = {keys.device}, must be {self.device}")
        self.batch_size, init_length = self._check_prefill(keys, None)
        init_length = min(init_length, self.cache_length)
        self._allocate_buffers(device=keys.device, dtype=keys.dtype)
        self.current_length = init_length
        keys = keys[:, :, :init_length, :].to(dtype=self.dtype)
        self.k[:self.batch_size, :, :init_length, :] = keys
        values = k_and_v.values()[:, :, :init_length, :].to(dtype=self.dtype)
        self.v[:self.batch_size, :, :init_length, :] = values

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        if not self.buffers_are_allocated:
            raise IndexError("Buffers are not allocated. Call 'prefill' first")
        sz_buffs = bitsize_of(self.k) + bitsize_of(self.v)
        return sz_buffs, dict(buffers=sz_buffs)

    @staticmethod
    def size_estimate_apriori(params: KVCacheBuffersParams, **kwargs) -> Tuple[int, Dict[str, int]]:
        cache_length = kwargs.get("cache_length")
        if cache_length is None:
            raise IndexError("Argument 'cache_length' is missing")
        else:
            cache_length = int(cache_length)
        dtype = params.dtype
        if dtype is None:
            raise ValueError("params.dtype must be provided")
        numel = params.max_batch_size * params.n_query_groups * cache_length * params.head_size
        sz_buffs = 2 * numel * bits_for_torch_dtype(dtype)
        return sz_buffs, dict(buffers=sz_buffs)

    def _check_positions(self, positions: PositionsType) -> PositionsType:
        if not isinstance(positions, torch.Tensor):
            if not isinstance(positions, tuple) or len(positions) != 2:
                raise TypeError("positions must be tuple (start, end) or tensor")
            start, end = positions
            if not 0 <= start < end <= self.cache_length:
                raise ValueError(f"positions = {positions} is out of bounds")
        else:
            assert (
                positions.ndim == 3 and
                positions.shape[0] in (1, self.batch_size) and
                positions.shape[1] in (1, self.n_query_groups) and
                positions.shape[2] <= self.cache_length
            ), f"positions.shape = {positions.shape} not compatible with batch_size = {self.batch_size}, n_query_groups = {self.n_query_groups}"
            if positions.dtype != torch.int64:
                positions = positions.to(dtype=torch.int64)
        return positions
