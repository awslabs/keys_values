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
from typing import List, Optional, Tuple, Dict, Any

import torch

from keys_values.attention import DefaultKeysAndValues
from keys_values.kvcache.buffers import KVCacheBuffersParams, DefaultKVCacheBuffers
from keys_values.kvcache.factory import create_quantized_kv_buffers_for_checkpoints
from keys_values.kvcache.quant_buffers import (
    QuantizedKVCacheBuffers,
)
from keys_values.kvcache.stack_layers import DefaultCellBlocks
from keys_values.model import GPT


class KVCacheBufferCheckpoints:
    """
    Interface for classes which collect checkpoints of KV cache buffers for
    a subset of token chunks.

    """
    def __init__(
        self,
        chunk_numbers: List[int],
    ):
        """
        Args:
            chunk_numbers: List of chunk indexes for which checkpoints are to
                be stored. Must not contain 0

        """
        KVCacheBufferCheckpoints.set_chunk_numbers(self, chunk_numbers)
        self._debug_layer_idx = None

    def set_chunk_numbers(self, chunk_numbers: List[int]):
        assert len(chunk_numbers) >= 1
        assert all(x >= 0 for x in chunk_numbers)
        self.chunk_numbers = chunk_numbers.copy()
        self._chunk_pos = {i: p for p, i in enumerate(chunk_numbers)}
        assert len(self._chunk_pos) == len(chunk_numbers)

    def set_debug_layer_idx(self, debug_layer_idx: int):
        self._debug_layer_idx = debug_layer_idx

    def pos_for_chunk_idx(self, chunk_idx: int) -> Optional[int]:
        return self._chunk_pos.get(chunk_idx)

    def set_checkpoint(
        self,
        chunk_idx: int,
        buffers: DefaultKVCacheBuffers,
    ) -> Optional[int]:
        """
        Args:
            chunk_idx: Index of chunk. The checkpoint is written only if this
                value is in `self.chunk_numbers`.
            buffers: KV cache buffers to be checkpointed. Can be on GPU. Note
                that `buffers.current_length` is stored along with the
                checkpoint.

        Returns:
            Slot position of `layer_idx` in `self.layer_numbers` if checkpoint
            is set, or `None` otherwise.

        """
        if not isinstance(buffers, DefaultKVCacheBuffers):
            raise ValueError(f"type(value) = {type(buffers)}, must be DefaultKVCacheBuffers")
        pos = self._chunk_pos.get(chunk_idx)
        if pos is None:
            return None
        if self._debug_layer_idx is not None:
            print(f"set_checkpoint: layer {self._debug_layer_idx}, chunk {chunk_idx} -> pos {pos}")
        return self._set_checkpoint(pos, buffers)

    def _set_checkpoint(
        self,
        pos: int,
        buffers: DefaultKVCacheBuffers,
    ) -> int:
        raise NotImplementedError

    def get_checkpoint(
        self,
        chunk_idx: int,
        out: DefaultKVCacheBuffers,
    ) -> DefaultKVCacheBuffers:
        """
        Args:
            chunk_idx: Index of layer, must be in `self.chunk_numbers`.
            out: KV cache buffers to write checkpoint to. Can be on GPU. Note
                that `out.current_length` is set to the length stored with
                :meth:`set_checkpoint`.

        Returns:
            `out` for convenience.

        """
        if not isinstance(out, DefaultKVCacheBuffers):
            raise ValueError(f"type(out) = {type(out)}, must be DefaultKVCacheBuffers")
        pos = self._chunk_pos.get(chunk_idx)
        if pos is None:
            raise IndexError(f"chunk_idx = {chunk_idx} must be in {self.chunk_numbers}")
        self._get_checkpoint(pos, out)
        return out

    def _get_checkpoint(
        self,
        pos: int,
        out: DefaultKVCacheBuffers,
    ):
        raise NotImplementedError

    def set_checkpoint_slice(
        self,
        chunk_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        input_pos: int,
    ) -> Optional[int]:
        raise NotImplementedError

    def get_checkpoint_slice(
        self,
        chunk_idx: int,
        input_pos: int,
        num: int,
    ) -> DefaultKeysAndValues:
        raise NotImplementedError


class KVCacheBufferQuantizedCheckpoints(KVCacheBufferCheckpoints):
    """
    Collects checkpoints of KV cache buffers for a subset of token chunks.

    Note that this class interacts with cache buffers of type
    :class:`DefaultKVCacheBuffers`, which do not quantize their content as
    such. Quantization is done only here, when a checkpoint is stored, and
    dequantization is done when a checkpoint is restored. This ensures that
    quantization and dequantization errors are avoided for the gradient
    computations. Different to the usage in pure inference mode, we need to
    maintain one full KV cache buffer per layer in training mode anyway.

    How to activate checkpointing:

    We checkpoint cache buffers for a row of cells, consisting of a number of
    layers. Each layer has its checkpoint object. All these can share a single
    `quant_buffers` object to do the quantization/dequantization, since this
    is never done in parallel over several layers. If several rows of cells
    are processed in sequence, the same checkpointing objects can be reused.

    For each layer in the row:
    ```
    # Share `quant_buffers`
    checkpoints = KVCacheBufferCheckpoints(..., quant_buffers=quant_buffers)

    # kv_cache is the cache for the layer in question
    kv_cache.set_checkpoint_hook(
        checkpoint_hook = lambda buffers, chunk_idx: checkpoints.set_checkpoint(
            chunk_idx=chunk_idx,
            value=buffers,
        )
    )
    ```

    Here, the KV caches must use :class:`DefaultKVCacheBuffers` buffers, not
    :class:`QuantizedKVCacheBuffers`.

    """
    def __init__(
        self,
        chunk_numbers: List[int],
        quant_buffers: QuantizedKVCacheBuffers,
        cache_length: Optional[int] = None,
    ):
        """
        Args:
            chunk_numbers: List of chunk indexes for which checkpoints are to
                be stored. Must not contain 0
            quant_buffers: Quantized KV cache buffers. This object is used to
                do the quantization in :meth:`set_checkpoint` and the
                dequantization in :meth:`get_checkpoint`. We also use related
                quantizer states as checkpoints. The object can be shared
                between different checkpoint objects. It must be on the same
                device as the arguments of :meth:`set_checkpoint` and
                :meth:`get_checkpoint`.
            cache_length: Determine the checkpoint size. May be smaller than
                `quant_buffers.cache_length`.

        """
        if cache_length is None:
            cache_length = quant_buffers.cache_length
        elif not (1 <= cache_length <= quant_buffers.cache_length):
            raise ValueError(f"cache_length={cache_length}, must be in [1, {quant_buffers.cache_length}]")
        self.cache_length = cache_length
        self.quant_buffers = quant_buffers
        self.checkpoints = None
        self._checkpoint_lengths = None
        super().__init__(chunk_numbers)
        self.set_chunk_numbers(chunk_numbers)

    @property
    def batch_size(self) -> Optional[int]:
        return self.quant_buffers.batch_size

    def set_chunk_numbers(self, chunk_numbers: List[int]):
        """
        If `chunk_numbers` is longer than the current list, extra buffers are
        allocated. Buffers are not deallocated if `chunk_numbers` is shorter.

        Args:
            chunk_numbers: List of chunk indexes for which checkpoints are to
                be stored. Must not contain 0

        """
        super().set_chunk_numbers(chunk_numbers)
        if self.checkpoints is None:
            num_to_create = len(self.chunk_numbers)
        else:
            num_to_create = max(len(self.chunk_numbers) - len(self.checkpoints), 0)
        kwargs = dict(device=torch.device("cpu"), cache_length=self.cache_length)
        new_checkpoints = [
            (
                self.quant_buffers.quantizer_k.create_quantizer_state(**kwargs),
                self.quant_buffers.quantizer_v.create_quantizer_state(**kwargs),
            )
            for _ in range(num_to_create)
        ]
        if self.checkpoints is None:
            self.checkpoints = new_checkpoints
            self._checkpoint_lengths = [self.cache_length] * num_to_create
        else:
            self.checkpoints.extend(new_checkpoints)
            self._checkpoint_lengths.extend([self.cache_length] * num_to_create)

    def _set_checkpoint(
        self,
        pos: int,
        buffers: DefaultKVCacheBuffers,
    ) -> int:
        k_and_v = buffers.get_keys_values()
        keys, values = k_and_v.keys(), k_and_v.values()
        current_length = buffers.current_length
        self.quant_buffers.prefill(
            keys[:, :, :current_length, :], values[:, :, :current_length, :],
        )
        # Ensure that content is quantized and written into buffers:
        self.quant_buffers.write_back()
        self.checkpoints[pos][0].copy_(end=current_length)
        self.checkpoints[pos][1].copy_(end=current_length)
        self._checkpoint_lengths[pos] = current_length
        return pos

    def _get_checkpoint(
        self,
        pos: int,
        out: DefaultKVCacheBuffers,
    ):
        assert out.cache_length == self.cache_length
        current_length = self._checkpoint_lengths[pos]
        self.checkpoints[pos][0].restore(end=current_length)
        self.checkpoints[pos][1].restore(end=current_length)
        # Dropping the assignment is important, since the quantized buffers
        # are modified (by `restore`) without `quant_buffers.dequant_buffers`
        # being notified. With the assignment dropped, it is recreated in
        # the `get_keys_values` call, and `dequant_buffers` hosts the correct
        # content.
        self.quant_buffers.drop_association()
        k_and_v = self.quant_buffers.get_keys_values()
        if k_and_v is None:
            raise IndexError(f"Failed to fetch dequantized buffer contents (pos={pos})")
        out.prefill_from_keys_values(k_and_v)
        out.current_length = current_length

    def set_checkpoint_slice(
        self,
        chunk_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        input_pos: int,
    ) -> Optional[int]:
        pos = self._chunk_pos.get(chunk_idx)
        if pos is None:
            return None
        assert key.ndim == 4
        num = key.shape[2]
        batch_size = self.batch_size
        if batch_size is None:
            if input_pos > 0:
                raise IndexError(f"quant_buffers must have batch_size set")
            # `quant_buffers.batch_size` will be set with `prefill`
            batch_size = key.shape[0]
        shape = (
            batch_size,
            self.quant_buffers.n_query_groups,
            num,
            self.quant_buffers.head_size,
        )
        if key.shape != shape:
            raise ValueError(f"key.shape = {key.shape}, must be {shape}")
        if value.shape != shape:
            raise ValueError(f"value.shape = {value.shape}, must be {shape}")
        if not (0 <= input_pos and num > 0 and input_pos + num <= self.quant_buffers.cache_length):
            raise ValueError(f"input_pos = {input_pos}, num = {num}, does not fit into [0, {self.quant_buffers.cache_length}]")
        if input_pos == 0:
            self.quant_buffers.prefill(key, value)
        else:
            self.quant_buffers.set_slots(
                (input_pos, input_pos + num), key, value,
            )
        # Ensure that content is quantized and written into buffers:
        self.quant_buffers.write_back()
        self.checkpoints[pos][0].copy_(start=input_pos, end=input_pos + num)
        self.checkpoints[pos][1].copy_(start=input_pos, end=input_pos + num)
        self._checkpoint_lengths[pos] = max(
            self._checkpoint_lengths[pos], input_pos + num,
        )
        return pos

    def get_checkpoint_slice(
        self,
        chunk_idx: int,
        input_pos: int,
        num: int,
        device: Optional[torch.device] = None,
    ) -> DefaultKeysAndValues:
        pos = self._chunk_pos.get(chunk_idx)
        if pos is None:
            raise IndexError(f"chunk_idx = {chunk_idx} must be in {self.chunk_numbers}")
        if not (0 <= input_pos and num > 0 and input_pos + num <= self._checkpoint_lengths[pos]):
            raise ValueError(f"input_pos = {input_pos}, num = {num}, does not fit into [0, {self._checkpoint_lengths[pos]}]")
        self.checkpoints[pos][0].restore(start=input_pos, end=input_pos + num)
        self.checkpoints[pos][1].restore(start=input_pos, end=input_pos + num)
        # See comments in :meth:`_get_checkpoint`
        self.quant_buffers.drop_association()
        keys, values = self.quant_buffers.get_slots((input_pos, input_pos + num))
        if device is not None:
            keys = keys.to(device)
            values = values.to(device)
        return DefaultKeysAndValues(keys, values)


class KVCacheBufferDefaultCheckpoints(KVCacheBufferCheckpoints):
    """
    Collects checkpoints of KV cache buffers for a subset of token chunks.

    The checkpoints are stored as they are, without quantization. This is
    recommended mostly for testing, or if CPU memory is not scarce.

    """
    def __init__(
        self,
        chunk_numbers: List[int],
        params: KVCacheBuffersParams,
        cache_length: int,
        batch_size: Optional[int] = None,
    ):
        """
        Args:
            chunk_numbers: List of chunk indexes for which checkpoints are to
                be stored. Must not contain 0
            params: KV cache buffer parameters
            cache_length: Cache length

        """
        super().__init__(chunk_numbers)
        self._kwargs = dict(dtype=params.dtype, device=torch.device("cpu"))
        if  batch_size is None:
            batch_size = params.max_batch_size
        shape = (batch_size, params.n_query_groups, cache_length, params.head_size)
        num_slots = len(chunk_numbers)
        self.k = [torch.zeros(shape, **self._kwargs) for _ in range(num_slots)]
        self.v = [torch.zeros(shape, **self._kwargs) for _ in range(num_slots)]
        self._checkpoint_lengths = [cache_length] * num_slots

    @property
    def batch_size(self) -> int:
        return self.k[0].shape[0]

    @property
    def cache_length(self) -> int:
        return self.k[0].shape[2]

    def set_chunk_numbers(self, chunk_numbers: List[int]):
        """
        If `chunk_numbers` is longer than the current list, extra buffers are
        allocated. Buffers are not deallocated if `chunk_numbers` is shorter.

        Args:
            chunk_numbers: List of chunk indexes for which checkpoints are to
                be stored. Must not contain 0

        """
        super().set_chunk_numbers(chunk_numbers)
        shape = self.k[0].shape
        num_extra = max(len(self.chunk_numbers) - len(self.k), 0)
        for _ in range(num_extra):
            self.k.append(torch.zeros(shape, **self._kwargs))
            self.v.append(torch.zeros(shape, **self._kwargs))
        if num_extra > 0:
            self._checkpoint_lengths.extend([self.cache_length] * num_extra)

    def _set_checkpoint(
        self,
        pos: int,
        buffers: DefaultKVCacheBuffers,
    ) -> int:
        k_and_v = buffers.get_keys_values()
        current_length = buffers.current_length
        self.k[pos][:, :, :current_length, :] = k_and_v.keys()[:, :, :current_length, :]
        self.v[pos][:, :, :current_length, :] = k_and_v.values()[:, :, :current_length, :]
        self._checkpoint_lengths[pos] = current_length
        return pos

    def _get_checkpoint(
        self,
        pos: int,
        out: DefaultKVCacheBuffers,
    ):
        out.prefill(
            key=self.k[pos][:, ...],
            value=self.v[pos][:, ...],
        )
        out.current_length = self._checkpoint_lengths[pos]

    def set_checkpoint_slice(
        self,
        chunk_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        input_pos: int,
    ) -> Optional[int]:
        pos = self._chunk_pos.get(chunk_idx)
        if pos is None:
            return None
        assert key.ndim == 4
        num = key.shape[2]
        _shape = self.k[0].shape
        shape = _shape[:2] + (num, _shape[3])
        if key.shape != shape:
            raise ValueError(f"key.shape = {key.shape}, must be {shape}")
        if value.shape != shape:
            raise ValueError(f"value.shape = {value.shape}, must be {shape}")
        if not (0 <= input_pos and num > 0 and input_pos + num <= self.cache_length):
            raise ValueError(f"input_pos = {input_pos}, num = {num}, does not fit into [0, {self.cache_length}]")
        self.k[pos][:, :, input_pos:(input_pos + num), :] = key
        self.v[pos][:, :, input_pos:(input_pos + num), :] = value
        self._checkpoint_lengths[pos] = max(
            self._checkpoint_lengths[pos], input_pos + num,
        )
        return pos

    def get_checkpoint_slice(
        self,
        chunk_idx: int,
        input_pos: int,
        num: int,
        device: Optional[torch.device] = None,
    ) -> DefaultKeysAndValues:
        pos = self._chunk_pos.get(chunk_idx)
        if pos is None:
            raise IndexError(f"chunk_idx = {chunk_idx} must be in {self.chunk_numbers}")
        current_length = self._checkpoint_lengths[pos]
        if not (0 <= input_pos and num > 0 and input_pos + num <= current_length):
            raise ValueError(f"input_pos = {input_pos}, num = {num}, does not fit into [0, {current_length}]")
        if device is None:
            device = torch.device("cpu")
        return DefaultKeysAndValues(
            keys=self.k[pos][:, :, input_pos:(input_pos + num), :].to(device=device),
            values=self.v[pos][:, :, input_pos:(input_pos + num), :].to(device=device),
        )


class LayerInputCheckpoints:
    """
    Collects checkpoints of layer inputs for a subset of the layers.

    During the inference forward pass, call :meth:`set_checkpoint` with the
    input of every layer. The object will store inputs for layer indexes in
    `layer_numbers`.

    """
    def __init__(self, layer_numbers: List[int]):
        """
        Args:
            layer_numbers: List of layer numbers for which inputs checkpoints
                are stored. One entry can be equal to `n_layer`, for which the
                output of the final layer `n_layer - 1` is stored.

        """
        assert len(layer_numbers) >= 1
        assert all(x >= 0 for x in layer_numbers)
        self.layer_numbers = layer_numbers.copy()

    def set_checkpoint(
        self,
        layer_idx: int,
        buffers: torch.Tensor,
        input_pos: int,
    ) -> Optional[int]:
        """
        Args:
            layer_idx: Index of layer. The checkpoint is written only if this
                value is in `self.layer_numbers`.
            buffers: Tensor part to write, of shape `(batch_size, num, n_embd)`.
                Can be on GPU.
            input_pos: Position in sequence. `value` is written to
                `range(input_pos, input_pos + num)` along dimension 1.

        Returns:
            Slot position of `layer_idx` in `self.layer_numbers` if checkpoint
            is set, or `None` otherwise.

        """
        raise NotImplementedError

    def get_checkpoint(
        self,
        layer_idx: int,
        input_pos: int,
        num: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Args:
            layer_idx: Index of layer. The checkpoint is returned only if this
                value is in `self.layer_numbers`.
            input_pos: Position in sequence. Returns part of checkpoint
                corresponding to `range(input_pos, input_pos + num)` along
                dimension 1.
            num: Length of slice to be returned
            device: Device for return argument

        Returns:
            Slice of checkpoint requested

        """
        raise NotImplementedError


def create_layers_and_buffers_for_model(
    model: GPT,
    layer_numbers: List[int],
    batch_size: int,
    qname: str,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Tuple[List[int], QuantizedKVCacheBuffers]]:
    # Determine the devices
    # Note: `n_layer` may be in `layer_numbers`, for which we use the device
    # of the final layer `n_layer - 1`
    n_layer = model.config.n_layer
    lnums_and_device = dict()
    for l_ix in layer_numbers:
        params = model.get_kv_cache_params(min(l_ix, n_layer - 1))
        device = params.device
        lnums = lnums_and_device.get(device)
        if lnums is None:
            lnums_and_device[device] = [l_ix]
        else:
            lnums.append(l_ix)
    # Create quantization buffers, one per device
    model_part = DefaultCellBlocks(
        model=model,
        first_layer_idx=0,
        num_layers=n_layer,
    )
    dequant_kwargs = dict(max_num_ranges=cache_kwargs.get("max_num_ranges"))
    quant_buffers = create_quantized_kv_buffers_for_checkpoints(
        model_part=model_part,
        qname=qname,
        batch_size=batch_size,
        layer_activations=True,
        cache_kwargs=cache_kwargs,
        dequant_kwargs=dequant_kwargs,
    )
    return [
        (lnums, quant_buffers[device])
        for device, lnums in lnums_and_device.items()
    ]


class LayerInputQuantizedCheckpoints(LayerInputCheckpoints):
    """
    Internally, we use :class:`KVCacheBufferQuantizedCheckpoints` objects,
    splitting two halves to keys and values.

    """
    def __init__(
        self,
        layers_and_buffers: List[Tuple[List[int], QuantizedKVCacheBuffers]],
    ):
        """
        If `layers_and_buffers` has more than one entry, the layers in different
        entries reside on different devices (model-parallel), and the
        `quant_buffers` object is on the respective device as well. For
        convenience, use :func:`create_layers_and_buffers_for_model`.

        Args:
            layers_and_buffers: List of `(layer_numbers, quant_buffers)` tuples.
                The `layer_numbers` lists are pairwise disjoint. For one entry,
                all layers and `quant_buffers` are on the same device. The
                `quant_buffers` object is used internally with an
                :class:`KVCacheBufferQuantizedCheckpoints` object. It must have
                `n_query_groups=1`, `cache_length=max_sequence_length`, and
                `head_size=n_embd // 2`.

        """
        self._layer_to_pos = {
            l_ix: pos for pos, (l_nums, _) in enumerate(layers_and_buffers) for l_ix in l_nums
        }
        layer_numbers = list(self._layer_to_pos.keys())
        if len(layer_numbers) < sum(len(x[0]) for x in layers_and_buffers):
            raise ValueError("layer_numbers entries must be pairwise disjoint")
        super().__init__(layer_numbers)
        if not all(x[1].n_query_groups == 1 for x in layers_and_buffers):
            raise ValueError(f"quant_buffers.n_query_groups must be 1 for all entries")
        cache_lengths = set(x[1].cache_length for x in layers_and_buffers)
        if len(cache_lengths) > 1:
            raise ValueError(f"quant_buffers.cache_length must be the same for all entries [{cache_lengths}]")
        head_sizes = set(x[1].head_size for x in layers_and_buffers)
        if len(head_sizes) > 1:
            raise ValueError(f"quant_buffers.head_size must be the same for all entries [{head_sizes}]")
        self.max_sequence_length = next(iter(cache_lengths))
        self.n_embd = next(iter(head_sizes)) * 2
        # Internally, we use :class:`KVCacheBufferQuantizedCheckpoints` objects
        self._checkpoints_int = [
            KVCacheBufferQuantizedCheckpoints(
                chunk_numbers=layer_numbers,
                quant_buffers=quant_buffers,
            )
            for layer_numbers, quant_buffers in layers_and_buffers
        ]

    def clear(self):
        """
        The object is defunct after this method is called. Just part of an
        attempt to avoid GPU memory leaks.

        """
        for checkpoint in self._checkpoints_int:
            checkpoint.quant_buffers.deallocate()
        self._checkpoints_int = None

    def set_checkpoint(
        self,
        layer_idx: int,
        buffers: torch.Tensor,
        input_pos: int,
    ) -> Optional[int]:
        pos = self._layer_to_pos.get(layer_idx)
        if pos is None:
            return None
        cp_int = self._checkpoints_int[pos]
        if buffers.ndim != 3:
            raise ValueError(f"buffers.shape = {buffers.shape}, must be 3D")
        num = buffers.shape[1]
        batch_size = cp_int.batch_size
        if batch_size is None and input_pos == 0:
            batch_size = buffers.shape[0]
        shape = (batch_size, num, self.n_embd)
        if buffers.shape != shape:
            raise ValueError(f"buffers.shape = {buffers.shape}, must be {shape}")
        if not (0 <= input_pos and num > 0 and input_pos + num <= self.max_sequence_length):
            raise ValueError(f"input_pos = {input_pos}, num = {num}, does not fit into [0, {self.max_sequence_length}]")
        ne2 = self.n_embd // 2
        return cp_int.set_checkpoint_slice(
            chunk_idx=layer_idx,
            key=buffers[:, None, :, :ne2],
            value=buffers[:, None, :, ne2:],
            input_pos=input_pos,
        )

    def get_checkpoint(
        self,
        layer_idx: int,
        input_pos: int,
        num: int,
        device: torch.device,
    ) -> torch.Tensor:
        pos = self._layer_to_pos.get(layer_idx)
        if pos is None:
            raise ValueError(f"layer_idx = {layer_idx} not in layer numbers [{self.layer_numbers}]")
        cp_int = self._checkpoints_int[pos]
        if not (0 <= input_pos and num > 0 and input_pos + num <= self.max_sequence_length):
            raise ValueError(f"input_pos = {input_pos}, num = {num}, does not fit into [0, {self.max_sequence_length}]")
        k_and_v = cp_int.get_checkpoint_slice(
            chunk_idx=layer_idx,
            input_pos=input_pos,
            num=num,
            device=device,
        )
        return torch.cat(
            (k_and_v.keys().squeeze(1), k_and_v.values().squeeze(1)),
            dim=-1,
        )


class LayerInputDefaultCheckpoints(LayerInputCheckpoints):
    """
    Internally, we use a :class:`KVCacheBufferDefaultCheckpoints` object,
    splitting two halves to keys and values.

    """
    def __init__(
        self,
        layer_numbers: List[int],
        batch_size: int,
        max_sequence_length: int,
        n_embd: int,
        dtype: Optional[torch.dtype],
    ):
        """
        Args:
            layer_numbers: List of layer numbers for which inputs checkpoints
                are stored
            batch_size: Batch size
            max_sequence_length: Maximum sequence length
            n_embd: Number of embedding dimensions
            dtype: Data type

        """
        super().__init__(layer_numbers)
        self._buffer_params = KVCacheBuffersParams(
            max_batch_size=batch_size,
            n_query_groups=1,
            head_size=n_embd // 2,
            dtype=dtype,
            device=torch.device("cpu"),
        )
        self._checkpoints_int = KVCacheBufferDefaultCheckpoints(
            chunk_numbers=layer_numbers,
            params=self._buffer_params,
            cache_length=max_sequence_length,
        )
        self.n_embd = n_embd

    def pos_for_layer_idx(self, layer_idx: int) -> Optional[int]:
        return self._checkpoints_int.pos_for_chunk_idx(layer_idx)

    def set_checkpoint(
        self,
        layer_idx: int,
        buffers: torch.Tensor,
        input_pos: int,
    ) -> Optional[int]:
        if buffers.ndim != 3:
            raise ValueError(f"buffers.shape = {buffers.shape}, must be 3D")
        num = buffers.shape[1]
        batch_size = self._checkpoints_int.batch_size
        if batch_size is None and input_pos == 0:
            batch_size = buffers.shape[0]
        shape = (batch_size, num, self.n_embd)
        if buffers.shape != shape:
            raise ValueError(f"buffers.shape = {buffers.shape}, must be {shape}")
        max_seq_length = self._checkpoints_int.cache_length
        if not (0 <= input_pos and num > 0 and input_pos + num <= max_seq_length):
            raise ValueError(f"input_pos = {input_pos}, num = {num}, does not fit into [0, {max_seq_length}]")
        ne2 = self.n_embd // 2
        device = torch.device("cpu")
        return self._checkpoints_int.set_checkpoint_slice(
            chunk_idx=layer_idx,
            key=buffers[:, None, :, :ne2].to(device=device),
            value=buffers[:, None, :, ne2:].to(device=device),
            input_pos=input_pos,
        )

    def get_checkpoint(
        self,
        layer_idx: int,
        input_pos: int,
        num: int,
        device: torch.device,
    ) -> torch.Tensor:
        max_seq_length = self._checkpoints_int.cache_length
        if not (0 <= input_pos and num > 0 and input_pos + num <= max_seq_length):
            raise ValueError(f"input_pos = {input_pos}, num = {num}, does not fit into [0, {max_seq_length}]")
        pos = self._checkpoints_int.pos_for_chunk_idx(layer_idx)
        if pos is None:
            raise ValueError(f"layer_idx = {layer_idx} is not in layer numbers [{self.layer_numbers}]")
        k_and_v = self._checkpoints_int.get_checkpoint_slice(
            chunk_idx=layer_idx,
            input_pos=input_pos,
            num=num,
            device=device,
        )
        return torch.cat(
            (k_and_v.keys().squeeze(1), k_and_v.values().squeeze(1)),
            dim=-1,
        )
