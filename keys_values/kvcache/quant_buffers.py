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
from typing import Tuple, Optional, Dict, List
from inspect import isclass
from dataclasses import dataclass

import torch

from keys_values.attention import KeysAndValues
from keys_values.kvcache.buffers import (
    KVCacheBuffers,
    KVCacheBuffersParams,
    PositionsType,
)
from keys_values.kvcache.quantize.quantization import Quantizer
from keys_values.kvcache.utils import (
    smallest_covering_ranges,
    bitsize_of,
    bits_for_torch_dtype,
)
from keys_values.utils import expand_index


class QuantizedKVCacheBuffers(KVCacheBuffers):
    """
    Quantization is delegated to `quantizer`. The member `dequant_buffers` is
    used to return de-quantized content to the user. In order for quantization
    to make sense, this object should be shared across several quantized
    buffers, in particular for different layers of the model.

    Association with `dequant_buffers`, write back:

    `dequant_buffers` obtains de-quantized content once it gets associated
    with a new object of this class here. Its content is quantized and
    written back only once the association is changed again. Try to group
    several modifications of a KV cache buffer before moving on to another
    layer, since otherwise a lot of time is spent on quantization
    and dequantization.

    Since `dequant_buffers` is typically shared by several KV caches of this
    type (with the same parameters), it is important to call
    :code:`self.dequant_buffers.set_quantized_cache(self)` every time the
    shared buffer is used here.

    Important: The association logic assumes that the quantized buffers are
    modified only via `dequant_buffers`. If quantized buffers are modified
    elsewhere (e.g., restored from a checkpoint), call
    :meth:`drop_association`.

    Note that :meth:`deallocate` only deallocates `quantizer_k`, `quantizer_v`,
    but not `dequant_buffers`. This is because typically, several KV caches
    share the same `dequant_buffers`. It remains allocated.

    """
    def __init__(
        self,
        quantizer_k: Quantizer,
        quantizer_v: Quantizer,
        dequant_buffers: "DequantizedKVCacheBuffers",
        debug_label: Optional[str] = None,
    ):
        """
        Args:
            quantizer_k: Quantizer for keys
            quantizer_v: Quantizer for values
            dequant_buffers: Object used to store de-quantized content. Should
                be shared between several quantized buffers.

        """
        cache_length = quantizer_k.shape[2]
        super().__init__(dequant_buffers.get_params(), cache_length)
        self._check_init_args(quantizer_k, quantizer_v, dequant_buffers)
        self.quantizer_k = quantizer_k
        self.quantizer_v = quantizer_v
        self.dequant_buffers = dequant_buffers
        self._debug_label = debug_label

    @property
    def device(self) -> torch.device:
        return self.quantizer_k.device

    @property
    def buffers_are_allocated(self) -> bool:
        return self.quantizer_k.buffers_are_allocated and self.quantizer_v.buffers_are_allocated

    @property
    def debug_label(self) -> Optional[str]:
        return self._debug_label

    def deallocate(self):
        self.quantizer_k.deallocate()
        self.quantizer_v.deallocate()

    def _allocate_buffers(self, device: Optional[torch.device] = None):
        batch_size = self.batch_size
        if not self.buffers_are_allocated:
            if batch_size is None:
                batch_size = self.max_batch_size
        elif batch_size is not None:
            if self.quantizer_k.batch_size == batch_size:
                # No need to adjust buffer size
                batch_size = None
            elif self.quantizer_k.batch_size < batch_size:
                print(f"Batch size increased from {self.quantizer_k.batch_size} to {batch_size}: Re-allocating buffers")
        if batch_size is not None:
            # Note: We also call `allocate_buffers` of quantizers if batch size
            # decreases. In general, this does not lead to re-allocation, but
            # the quantizers need to know the effective batch size as well.
            self.quantizer_k.allocate_buffers(
                batch_size=batch_size, device=device,
            )
            self.quantizer_v.allocate_buffers(
                batch_size=batch_size, device=device,
            )

    def _check_init_args(
        self,
        quantizer_k: Quantizer,
        quantizer_v: Quantizer,
        dequant_buffers: "DequantizedKVCacheBuffers",
    ):
        if not (0 < self.cache_length <= dequant_buffers.cache_length):
            raise ValueError(f"quantizer_k.cache_length = {self.cache_length}, must be <= dequant_buffers.cache_length = {dequant_buffers.cache_length}")
        params = dequant_buffers.get_params()
        quant_shape_k = quantizer_k.shape
        quant_shape_v = quantizer_v.shape
        dequant_shape = (
            quant_shape_k[0],
            params.n_query_groups,
            self.cache_length,
            params.head_size,
        )
        if quant_shape_k != dequant_shape or quant_shape_v != dequant_shape:
            raise ValueError(
                f"quantizer_k.shape = {quant_shape_k}, "
                f"quantizer_v.shape = {quant_shape_v}, "
                f"dequant_buffers.shape = {dequant_shape}, must be the same"
            )
        if quantizer_k.blocks_over_heads != quantizer_v.blocks_over_heads:
            raise ValueError(
                f"quantizer_k.blocks_over_heads = {quantizer_k.blocks_over_heads}, "
                f"quantizer_v.blocks_over_heads = {quantizer_v.blocks_over_heads}, "
                "must be the same"
            )

    @property
    def blocks_over_heads(self) -> bool:
        return self.quantizer_k.blocks_over_heads

    def _assert_buffers_allocated(self):
        assert self.buffers_are_allocated, "Buffers are not allocated. Call 'prefill' first"

    def get_slots(
        self,
        positions: PositionsType,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._assert_buffers_allocated()
        self.dequant_buffers.set_quantized_cache(self)
        return self.dequant_buffers.get_slots(positions)

    def set_slots(
        self,
        positions: PositionsType,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        self._assert_buffers_allocated()
        self.dequant_buffers.set_quantized_cache(self)
        self.dequant_buffers.set_slots(positions, key, value)

    def _forward(
        self,
        positions: PositionsType,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> KeysAndValues:
        self._assert_buffers_allocated()
        self.dequant_buffers.set_quantized_cache(self)
        return self.dequant_buffers.forward(positions, key, value)

    def write_back(self):
        """
        Normally, the buffer content of `dequant_buffers` is quantized and
        written back to its associated quantized cache once the association
        is changed. But this may not happen at the end of an iteration over
        all layers.

        To be safe, call this method for all quantized caches. This makes
        sure that content is written back for all `dequant_buffers` being
        used. Associations are not changed here.

        """
        self.dequant_buffers.write_back()

    def drop_association(self):
        """
        Drops association to `dequant_buffers`. Call this method if the
        quantized cache content is modified *not* through `dequant_buffers`.
        Once the association is dropped, the next access here will reset the
        association and populate the `dequant_buffers`.

        If the quantized content here is changed without dropping the
        association, `dequant_buffers` is not re-populated and wrong content
        may be served.

        """
        self.dequant_buffers.set_quantized_cache(None)

    def get_keys_values(self) -> Optional[KeysAndValues]:
        self._assert_buffers_allocated()
        self.dequant_buffers.set_quantized_cache(self)
        return self.dequant_buffers.get_keys_values()

    def _prefill(self, key: torch.Tensor, value: torch.Tensor):
        self._allocate_buffers(device=key.device)
        self.dequant_buffers.set_quantized_cache(self)
        return self.dequant_buffers.prefill(key, value)

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        self._assert_buffers_allocated()
        total_k, parts_k = self.quantizer_k.size_estimate()
        total_v, parts_v = self.quantizer_v.size_estimate()
        bits_by_part = {
            **{"quant_k_" + name: num for name, num in parts_k.items()},
            **{"quant_v_" + name: num for name, num in parts_v.items()},
        }
        return total_k + total_v, bits_by_part

    @staticmethod
    def size_estimate_apriori(
        params: KVCacheBuffersParams, **kwargs,
    ) -> Tuple[int, Dict[str, int]]:
        quantizer_type = kwargs.get("quantizer_type")
        if quantizer_type is None:
            raise IndexError("Argument 'quantizer_type' is missing")
        else:
            assert isclass(quantizer_type)
        total_k, parts_k = quantizer_type.size_estimate_apriori(
            params, **kwargs,
        )
        bits_by_part = {
            **{"quant_k_" + name: num for name, num in parts_k.items()},
            **{"quant_v_" + name: num for name, num in parts_k.items()},
        }
        return 2 * total_k, bits_by_part

    def quantization_error(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method can be used to score a token in the cache in terms of
        how well its K, V content can be quantized. Note that for caches
        which do not keep slots in contiguous slices, this is not exactly
        correct, since we quantize along contiguous slices. Still, this is
        probably the best surrogate.

        Args:
            keys: Array to be quantized. Here, `keys.shape[-1] == head_size`,
                and if `quantizer_k.blocks_over_heads==True`, then
                `keys.shape[-2] == n_query_groups`.
            values: Same as `keys`, but for values.

        Returns:
            L2 quantization errors `vector_norm(x - dequant(quant(x)), dim=-1)`,
            where `x` is `keys`, `values`

        """
        return (
            self.quantizer_k.quantization_error(keys),
            self.quantizer_v.quantization_error(values),
        )


@dataclass(frozen=True)
class DebugDequantBuffEvent:
    kind: str
    label: Optional[str]
    num_slots: int
    slots: Optional[List[int]] = None

    def __post_init__(self):
        assert self.num_slots > 0
        assert self.kind in {"read", "write"}
        assert self.slots is None or len(self.slots) <= self.num_slots

    def __str__(self) -> str:
        result = f"* {self.kind:5}: {self.label}, {self.num_slots}"
        if self.slots is not None:
            result += f" - {self.slots}"
        return result


MAX_SLOTS_TRACKED_FRACTION = 0.2

MAX_SIZE_POSITIONS_TO_CHECK = 32

DEFAULT_MAX_NUM_RANGES = 4


class DequantizedKVCacheBuffers:
    """
    Special implementation just for use in :class:`QuantizedKVCacheBuffers`.

    In general, several :class:`QuantizedKVCacheBuffers` cache buffers share
    a common object of this class. The association is done by calling
    :meth:`set_quantized_cache`. The object here holds the dequantized
    buffer contents during the association. They are quantized and written
    back only once the association is changed, or :meth:`write_back` is called.
    This supports a number of change to the cache buffers before the content
    needs to be quantized again.

    This mechanism is slow when single tokens are generated, since this
    object rotates through layers with every update of size 1. We deal with
    this by tracking modified slots of the buffers, so that only these have
    to be written back. This tracking is limited by
    :const:`MAX_SLOTS_TRACKED_FRACTION` and
    :const:`MAX_SIZE_POSITIONS_TO_CHECK`, since it takes too much time
    otherwise.

    Note: This is not a subclass of :class:`KVCacheBuffers`, even if this would
    be convenient, since we need things like `current_length`, `batch_size`
    here and have to copy the logic. If this class was a child of
    :class:`KVCacheBuffers`, there would be a cyclic dependence between
    `QuantizedKVCacheBuffers.dequant_buffers` and
    `DequantizedKVCacheBuffers._quantized_cache`, which makes methods like
    :meth:`train` fail due to their recursive implementations, iterating over
    `self.children()`. There is apparently no way to tell PyTorch not to
    register a member, in order to break such cyclic dependencies: you must
    just not have any. In our case, `self._quantized_cache` is not even set
    in :meth:``__init__`, but only in :meth:`set_quantized_cache`, but even
    this leads to registration and infinite loops.

    This class also does not take part in buffer allocation on demand or
    buffer deallocation. Its buffers are allocated at construction, using
    `params.max_batch_size`. This is reasonable, since in general, several
    :class:`QuantizedKVCacheBuffers` use a shared object of this class
    here.

    """
    def __init__(
        self,
        params: KVCacheBuffersParams,
        cache_length: int,
        max_num_ranges: Optional[int] = None,
    ):
        # Copied from `KVCacheBuffers.__init__`:
        self.max_batch_size = params.max_batch_size
        self.n_query_groups = params.n_query_groups
        self.cache_length = cache_length
        self.head_size = params.head_size
        self._device = params.device
        self.dtype = params.dtype
        self.current_length = None
        self.k_buff = None
        self.v_buff = None
        self._quantized_cache = None
        self._max_num_ranges = DEFAULT_MAX_NUM_RANGES if max_num_ranges is None else max_num_ranges
        self._needs_write_back = False
        self._slots_to_write_back = None
        # Do not track more than this many slots
        self._max_slots_tracked = int(cache_length * MAX_SLOTS_TRACKED_FRACTION)
        self._allocate_buffers()
        self.debug_events = None

    @property
    def device(self) -> torch.device:
        if self.k_buff is not None:
            self._device = self.k_buff.device
        return self._device

    def start_debug_event_protocol(self):
        self.debug_events = []

    def _allocate_buffers(self, device: Optional[torch.device] = None):
        if device is None:
            device = self._device
        shape = (self.max_batch_size, self.n_query_groups, self.cache_length, self.head_size)
        kwargs = dict(device=device, dtype=self.dtype)
        self.k_buff = torch.zeros(shape, **kwargs)
        self.v_buff = torch.zeros(shape, **kwargs)

    def set_quantized_cache(self, cache: Optional[QuantizedKVCacheBuffers]):
        """
        This method has to be called by the KV cache before other methods. This
        is because the same object here is in general used with different KV caches,
        e.g. those of different layers of a model.

        Args:
            cache: :class:`QuantizedKVCacheBuffers` object whose quantized
                buffers are used here. Can be `None`, to drop the association.

        """
        if cache is self._quantized_cache:
            return  # Assignment remains unchanged
        if cache is not None and cache.cache_length > self.cache_length:
            raise ValueError(f"cache.cache_length={cache.cache_length}, must be <= {self.cache_length}")
        self.write_back()
        self._quantized_cache = cache  # Change association
        if cache is not None:
            # Buffers here must be on same device as `_quantized_cache`
            self._device = self._quantized_cache.device
            if self.k_buff is not None and self._device != self.k_buff.device:
                self._allocate_buffers()
            self._dequantize()  # Dequantize and copy

    def write_back(self):
        """
        Normally, the buffer content is quantized and written back to the
        associated quantized cache buffer once the association is changed.
        Call this method to force the write-back, e.g. at the end of a
        loop over layers.

        If the buffer content has already been written back and not been
        changed since then, nothing is done here.

        """
        # It can happen that `self._quantized_cache.buffers_are_allocated` is
        # False here, because buffers have been deallocated due to OOM error.
        # In this case, we don't need to write back anything
        if self._quantized_cache is not None and self._needs_write_back and self._quantized_cache.buffers_are_allocated:
            self._quantize()  # Quantize and write back

    def _check_quantized_cache(self):
        if self._quantized_cache is None:
            raise IndexError("Quantized cache is not assigned. Call `set_quantized_cache` first.")
        if self._quantized_cache.batch_size is None:
            raise IndexError("Quantized cache does not have effective batch size set. Call `prefill` first.")

    @property
    def eff_cache_length(self) -> int:
        cache = self._quantized_cache
        return self.cache_length if cache is None else cache.cache_length

    @property
    def batch_size(self) -> Optional[int]:
        """
        Returns:
            Current effective batch size of the associated quantized
            cache buffer

        """
        if self._quantized_cache is not None:
            return self._quantized_cache.batch_size
        else:
            return None

    # Copied from `DefaultKVCacheBuffers.get_slots`
    def get_slots(
        self,
        positions: PositionsType,
    )  -> Tuple[torch.Tensor, torch.Tensor]:
        self._check_quantized_cache()
        positions = self._check_positions(positions)
        if isinstance(positions, torch.Tensor):
            # `index[i, j, k, l] = positions[i, j, k]
            index = expand_index(positions, self.head_size)
            res_k = self.k_buff[:self.batch_size, ...].gather(-2, index)
            res_v = self.v_buff[:self.batch_size, ...].gather(-2, index)
        else:
            start, end = positions
            res_k = self.k_buff[:self.batch_size, :, start:end, :]
            res_v = self.v_buff[:self.batch_size, :, start:end, :]
        return res_k, res_v

    def _track_slots(self, positions: PositionsType):
        if not self._needs_write_back:
            self._slots_to_write_back = set()  # Initiate tracking
        elif self._slots_to_write_back is None:
            return  # Everything is written back
        if isinstance(positions, torch.Tensor):
            if positions.shape[2] > MAX_SIZE_POSITIONS_TO_CHECK:
                # `positions` too large: Write back everything
                self._slots_to_write_back = None
            else:
                # Track all new slots
                self._slots_to_write_back.update(positions.flatten().tolist())
        else:
            start, end = positions
            if end - start > self._max_slots_tracked:
                self._slots_to_write_back = None
            else:
                self._slots_to_write_back.update(range(start, end))
        if self._slots_to_write_back is not None and len(self._slots_to_write_back) > self._max_slots_tracked:
            # Too many slots: Write back everything
            self._slots_to_write_back = None
        self._needs_write_back = True

    def set_slots(
        self,
        positions: PositionsType,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        self._check_quantized_cache()
        key = key.to(device=self.device, dtype=self.dtype)
        value = value.to(device=self.device, dtype=self.dtype)
        positions = self._check_positions(positions)
        self._track_slots(positions)
        if isinstance(positions, torch.Tensor):
            # `index[i, j, k, l] = positions[i, j, k]`
            index = expand_index(positions, self.head_size)
            self.k_buff[:self.batch_size, ...].scatter_(-2, index, key)
            self.v_buff[:self.batch_size, ...].scatter_(-2, index, value)
        else:
            start, end = positions
            self.k_buff[:self.batch_size, :, start:end, :] = key
            self.v_buff[:self.batch_size, :, start:end, :] = value

    # Copied from `KVCacheBuffers.forward`:
    def forward(
        self,
        positions: PositionsType,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> KeysAndValues:
        num = key.shape[2]
        self.current_length = min(self.eff_cache_length, self.current_length + num)
        return self._forward(positions, key, value)

    def _forward(
        self,
        positions: PositionsType,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> KeysAndValues:
        self.set_slots(positions, key, value)
        return DequantizedBufferKeysAndValues(self)

    def get_keys_values(self) -> Optional[KeysAndValues]:
        if self.batch_size is None or self.current_length is None:
            return None
        else:
            return DequantizedBufferKeysAndValues(self)

    # Copied from `KVCacheBuffers.prefill`:
    def prefill(self, key: torch.Tensor, value: torch.Tensor):
        self._check_quantized_cache()
        self.current_length = self._check_prefill(key, value)
        self._prefill(key, value)

    # Copied from `KVCacheBuffers._check_prefill`:
    def _check_prefill(self, key: torch.Tensor, value: Optional[torch.Tensor]) -> int:
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
        return init_length

    def _prefill(self, key: torch.Tensor, value: torch.Tensor):
        # Sanity check
        if self.batch_size is None or self.batch_size != key.shape[0]:
            raise IndexError(f"self.batch_size = {self.batch_size}, key.shape[0] = {key.shape[0]}. Must be the same!")
        # Initialize cache buffers
        init_length = key.shape[2]
        self.set_slots((0, init_length), key, value)

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        sz_buffs = bitsize_of(self.k_buff) + bitsize_of(self.v_buff)
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

    # Copied from `KVCacheBuffers.get_params`:
    def get_params(self) -> KVCacheBuffersParams:
        return KVCacheBuffersParams(
            max_batch_size=self.max_batch_size,
            n_query_groups=self.n_query_groups,
            head_size=self.head_size,
            device=self.device,
            dtype=self.dtype,
        )

    def _check_positions(self, positions: PositionsType) -> PositionsType:
        if not isinstance(positions, torch.Tensor):
            assert isinstance(positions, tuple) and len(positions) == 2
            start, end = positions
            assert 0 <= start < end <= self.eff_cache_length
        else:
            assert (
                positions.ndim == 3 and
                positions.shape[0] in (1, self.batch_size) and
                positions.shape[1] in (1, self.n_query_groups)
            ), f"positions.shape = {positions.shape} not compatible with batch_size = {self.batch_size}, n_query_groups = {self.n_query_groups}"
            if positions.dtype != torch.int64:
                positions = positions.to(dtype=torch.int64)
        return positions

    def _quantize(self) -> None:
        cache = self._quantized_cache
        assert cache is not None
        assert self.batch_size is not None
        assert self._needs_write_back
        if self._slots_to_write_back is None:
            # Write back everything
            ranges = [(0, self.eff_cache_length)]
        else:
            ranges = smallest_covering_ranges(
                self._slots_to_write_back, self._max_num_ranges,
            )
        for a, b in ranges:
            cache.quantizer_k.quantize(
                a, b, self.k_buff[:self.batch_size, :, a:b, :],
            )
            cache.quantizer_v.quantize(
                a, b, self.v_buff[:self.batch_size, :, a:b, :],
            )
        if self.debug_events is not None:
            slots = None if self._slots_to_write_back is None else list(self._slots_to_write_back)
            self.debug_events.append(
                DebugDequantBuffEvent(
                    kind="write",
                    label=cache.debug_label,
                    num_slots=sum(b - a for a, b in ranges),
                    slots=slots,
                )
            )
        self._needs_write_back = False
        self._slots_to_write_back = None

    def _dequantize(self) -> None:
        cache = self._quantized_cache
        assert cache is not None
        assert self.batch_size is not None
        ecl = self.eff_cache_length
        cache.quantizer_k.dequantize(
            start=0,
            end=ecl,
            out=self.k_buff[:self.batch_size, :, :ecl, :],
        )
        cache.quantizer_v.dequantize(
            start=0,
            end=ecl,
            out=self.v_buff[:self.batch_size, :, :ecl, :],
        )
        if self.debug_events is not None:
            self.debug_events.append(
                DebugDequantBuffEvent(
                    kind="read",
                    label=cache.debug_label,
                    num_slots=ecl,
                    slots=None,
                )
            )
        self._needs_write_back = False
        self._slots_to_write_back = None


class DequantizedBufferKeysAndValues(KeysAndValues):
    def __init__(self, buffers: DequantizedKVCacheBuffers):
        self._buffers = buffers
        self._assoc_quant_buffers = buffers._quantized_cache
        if self._assoc_quant_buffers is None:
            raise ValueError("buffers must have associated cache. Use 'set_quantized_cache' first")

    def keys(self) -> torch.Tensor:
        if not (self._assoc_quant_buffers is self._buffers._quantized_cache):
            raise IndexError("buffers has been associated with different cache")
        current_length = self._buffers.current_length
        batch_size = self._buffers.batch_size
        if batch_size is None:
            raise IndexError("Associated buffer still has undefined batch size")
        return self._buffers.k_buff[:batch_size, :, :current_length, :]

    def values(self) -> torch.Tensor:
        if not (self._assoc_quant_buffers is self._buffers._quantized_cache):
            raise IndexError("buffers has been associated with different cache")
        current_length = self._buffers.current_length
        batch_size = self._buffers.batch_size
        if batch_size is None:
            raise IndexError("Associated buffer still has undefined batch size")
        return self._buffers.v_buff[:batch_size, :, :current_length, :]
