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
from dataclasses import replace
from typing import List, Optional, Tuple, Dict, Any

import torch

from keys_values.attention import DefaultKeysAndValues
from keys_values.kvcache.buffers import KVCacheBuffersParams, DefaultKVCacheBuffers
from keys_values.kvcache.quant_buffers import (
    QuantizedKVCacheBuffers,
    create_quantized_kv_buffers,
)
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
                checkpoint. Also, `buffers.cache_length` can be smaller than
                the cache length used here.

        Returns:
            Slot position of `layer_idx` in `self.layer_numbers` if checkpoint
            is set, or `None` otherwise.

        """
        if not isinstance(buffers, DefaultKVCacheBuffers):
            raise ValueError(
                f"type(value) = {type(buffers)}, must be DefaultKVCacheBuffers"
            )
        pos = self._chunk_pos.get(chunk_idx)
        if pos is None:
            return None
        if self._debug_layer_idx is not None:
            print(
                f"set_checkpoint: layer {self._debug_layer_idx}, chunk {chunk_idx} -> pos {pos}"
            )
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
                :meth:`set_checkpoint`. `out.cache_length` must not be smaller
                than this length.

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
        pin_memory: Optional[List[bool]] = None,
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
            pin_memory: If given, must have the same length as `chunk_numbers`.
                Checkpoints for chunks with `True` entries are pinned in CPU
                memory. Default: No checkpoints are pinned.

        """
        if cache_length is None:
            cache_length = quant_buffers.cache_length
        elif not (1 <= cache_length <= quant_buffers.cache_length):
            raise ValueError(
                f"cache_length={cache_length}, must be in [1, {quant_buffers.cache_length}]"
            )
        self.cache_length = cache_length
        self.quant_buffers = quant_buffers
        self.checkpoints = None
        self._checkpoint_lengths = None
        super().__init__(chunk_numbers)
        self.set_chunk_numbers(chunk_numbers, pin_memory)

    @property
    def batch_size(self) -> Optional[int]:
        return self.quant_buffers.batch_size

    def set_chunk_numbers(
        self,
        chunk_numbers: List[int],
        pin_memory: Optional[List[bool]] = None,
    ):
        """
        If `chunk_numbers` is longer than the current list, extra buffers are
        allocated. Buffers are not deallocated if `chunk_numbers` is shorter.

        Args:
            chunk_numbers: List of chunk indexes for which checkpoints are to
                be stored. Must not contain 0
            pin_memory: See :meth:`__init__`

        """
        super().set_chunk_numbers(chunk_numbers)
        if pin_memory is not None:
            if len(pin_memory) != len(self.chunk_numbers):
                raise ValueError(
                    f"pin_memory = {pin_memory}, chunk_numbers = {chunk_numbers}: Must have same length"
                )
        else:
            pin_memory = [False] * len(self.chunk_numbers)
        if self.checkpoints is None:
            num_to_create = len(self.chunk_numbers)
        else:
            num_to_create = max(len(self.chunk_numbers) - len(self.checkpoints), 0)
        kwargs = dict(device=torch.device("cpu"), cache_length=self.cache_length)
        if num_to_create > 0:
            new_checkpoints = [
                (
                    self.quant_buffers.quantizer_k.create_quantizer_state(
                        **kwargs,
                        pin_memory=pm,
                    ),
                    self.quant_buffers.quantizer_v.create_quantizer_state(
                        **kwargs,
                        pin_memory=pm,
                    ),
                )
                for pm in pin_memory[(-num_to_create):]
            ]
        else:
            new_checkpoints = []
        new_lengths = [self.cache_length] * num_to_create
        if self.checkpoints is None:
            self.checkpoints = new_checkpoints
            self._checkpoint_lengths = new_lengths
        else:
            self.checkpoints.extend(new_checkpoints)
            self._checkpoint_lengths.extend(new_lengths)

    def _set_checkpoint(
        self,
        pos: int,
        buffers: DefaultKVCacheBuffers,
    ) -> int:
        k_and_v = buffers.get_keys_values()
        keys, values = k_and_v.keys(), k_and_v.values()
        current_length = buffers.current_length
        self.quant_buffers.prefill(
            keys[:, :, :current_length, :],
            values[:, :, :current_length, :],
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
        if not (
            0 <= input_pos
            and num > 0
            and input_pos + num <= self.quant_buffers.cache_length
        ):
            raise ValueError(
                f"input_pos = {input_pos}, num = {num}, does not fit into [0, {self.quant_buffers.cache_length}]"
            )
        if input_pos == 0:
            self.quant_buffers.prefill(key, value)
        else:
            self.quant_buffers.set_slots(
                (input_pos, input_pos + num),
                key,
                value,
            )
        # Ensure that content is quantized and written into buffers:
        self.quant_buffers.write_back()
        self.checkpoints[pos][0].copy_(start=input_pos, end=input_pos + num)
        self.checkpoints[pos][1].copy_(start=input_pos, end=input_pos + num)
        self._checkpoint_lengths[pos] = max(
            self._checkpoint_lengths[pos],
            input_pos + num,
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
        if not (
            0 <= input_pos
            and num > 0
            and input_pos + num <= self._checkpoint_lengths[pos]
        ):
            raise ValueError(
                f"input_pos = {input_pos}, num = {num}, does not fit into [0, {self._checkpoint_lengths[pos]}]"
            )
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
        pin_memory: Optional[List[bool]] = None,
    ):
        """
        Args:
            chunk_numbers: List of chunk indexes for which checkpoints are to
                be stored. Must not contain 0
            params: KV cache buffer parameters
            cache_length: Cache length
            pin_memory: If given, must have the same length as `chunk_numbers`.
                Checkpoints for chunks with `True` entries are pinned in CPU
                memory. Default: No checkpoints are pinned.

        """
        super().__init__(chunk_numbers)
        self._kwargs = dict(dtype=params.dtype, device=torch.device("cpu"))
        if batch_size is None:
            batch_size = params.max_batch_size
        self._shape = (
            batch_size,
            params.n_query_groups,
            cache_length,
            params.head_size,
        )
        self.k = None
        self.v = None
        self._checkpoint_lengths = None
        self.set_chunk_numbers(chunk_numbers, pin_memory)

    @property
    def batch_size(self) -> int:
        return self._shape[0]

    @property
    def cache_length(self) -> int:
        return self._shape[2]

    def set_chunk_numbers(
        self,
        chunk_numbers: List[int],
        pin_memory: Optional[List[bool]] = None,
    ):
        """
        If `chunk_numbers` is longer than the current list, extra buffers are
        allocated. Buffers are not deallocated if `chunk_numbers` is shorter.

        Args:
            chunk_numbers: List of chunk indexes for which checkpoints are to
                be stored. Must not contain 0
            pin_memory: See :meth:`__init__`

        """
        super().set_chunk_numbers(chunk_numbers)
        if pin_memory is not None:
            if len(pin_memory) != len(self.chunk_numbers):
                raise ValueError(
                    f"pin_memory = {pin_memory}, chunk_numbers = {chunk_numbers}: Must have same length"
                )
        else:
            pin_memory = [False] * len(chunk_numbers)
        if self.k is None:
            num_to_create = len(self.chunk_numbers)
        else:
            num_to_create = max(len(self.chunk_numbers) - len(self.k), 0)
        if num_to_create > 0:
            new_k = [
                torch.zeros(self._shape, **self._kwargs, pin_memory=pm)
                for pm in pin_memory[(-num_to_create):]
            ]
            new_v = [
                torch.zeros(self._shape, **self._kwargs, pin_memory=pm)
                for pm in pin_memory[(-num_to_create):]
            ]
        else:
            new_k = []
            new_v = []
        new_lengths = [self.cache_length] * num_to_create
        if self.k is None:
            self.k = new_k
            self.v = new_v
            self._checkpoint_lengths = new_lengths
        else:
            self.k.extend(new_k)
            self.v.extend(new_v)
            self._checkpoint_lengths.extend(new_lengths)

    def _set_checkpoint(
        self,
        pos: int,
        buffers: DefaultKVCacheBuffers,
    ) -> int:
        k_and_v = buffers.get_keys_values()
        current_length = buffers.current_length
        self.k[pos][:, :, :current_length, :].copy_(
            k_and_v.keys()[:, :, :current_length, :],
            non_blocking=True,
        )
        self.v[pos][:, :, :current_length, :].copy_(
            k_and_v.values()[:, :, :current_length, :],
            non_blocking=True,
        )
        self._checkpoint_lengths[pos] = current_length
        return pos

    def _get_checkpoint(
        self,
        pos: int,
        out: DefaultKVCacheBuffers,
    ):
        key = self.k[pos][:, ...]
        value = self.v[pos][:, ...]
        device = out.device
        if device is not None:
            key = key.to(device, non_blocking=True)
            value = value.to(device, non_blocking=True)
        out.prefill(key=key, value=value)
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
            raise ValueError(
                f"input_pos = {input_pos}, num = {num}, does not fit into [0, {self.cache_length}]"
            )
        self.k[pos][:, :, input_pos : (input_pos + num), :].copy_(
            key,
            non_blocking=True,
        )
        self.v[pos][:, :, input_pos : (input_pos + num), :].copy_(
            value,
            non_blocking=True,
        )
        self._checkpoint_lengths[pos] = max(
            self._checkpoint_lengths[pos],
            input_pos + num,
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
            raise ValueError(
                f"input_pos = {input_pos}, num = {num}, does not fit into [0, {current_length}]"
            )
        if device is None:
            device = torch.get_default_device()
        return DefaultKeysAndValues(
            keys=self.k[pos][:, :, input_pos : (input_pos + num), :].to(
                device=device,
                non_blocking=True,
            ),
            values=self.v[pos][:, :, input_pos : (input_pos + num), :].to(
                device=device,
                non_blocking=True,
            ),
        )


class LayerInputCheckpoints:
    """
    Collects checkpoints of layer inputs for a subset of the layers.

    During the inference forward pass, call :meth:`set_checkpoint` with the
    input of every layer. The object will store inputs for layer indexes in
    `layer_numbers`.

    `cell_ranges` contains tuples `(start, end)`, so that
    `start[k + 1] == end[k]`, `start[0] == 0`. Internally, buffers for
    different cell ranges are kept separate (i.e., different tensors).
    This simplifies CPU-GPU transfers. While it is most efficient to call
    :meth:`get_checkpoint` and :meth:`set_checkpoint` with
    `range(input_pos, input_pos + num)` being a cell range, they can be
    called with any range.
    """

    def __init__(
        self,
        layer_numbers: List[int],
        cell_ranges: List[Tuple[int, int]],
    ):
        """
        Args:
            layer_numbers: List of layer numbers for which inputs checkpoints
                are stored. One entry can be equal to `n_layer`, for which the
                output of the final layer `n_layer - 1` is stored. We also use
                an entry `n_layer + 1` to store the head gradient during the
                backward pass.
            cell_ranges: List of tuples `(start, end)`, see above.
        """

        assert len(layer_numbers) >= 1
        assert all(x >= 0 for x in layer_numbers)
        self.layer_numbers = layer_numbers.copy()
        assert cell_ranges[0][0] == 0
        for rng1, rng2 in zip(cell_ranges[:-1], cell_ranges[1:]):
            assert rng1[0] < rng1[1]
            assert rng2[0] == rng1[1]
        assert cell_ranges[-1][0] < cell_ranges[-1][1]
        self.cell_ranges = cell_ranges.copy()
        self.max_seq_length = cell_ranges[-1][1]

    def get_ranges(
        self,
        input_pos: int,
        num: int,
    ) -> List[Tuple[int, int, int, int, int]]:
        if not (0 <= input_pos and num > 0 and input_pos + num <= self.max_seq_length):
            raise ValueError(
                f"input_pos = {input_pos}, num = {num}, does not fit into [0, {self.max_seq_length}]"
            )
        result = []
        _start = input_pos
        _end = input_pos + num
        left_ind = None
        sum_sizes = 0
        for ind, (start, end) in enumerate(self.cell_ranges):
            if start <= _start < end:
                result.append((ind, _start, min(end, _end), start))
                left_ind = ind
                sum_sizes += min(end, _end) - _start
            if start < _end <= end:
                if ind > left_ind:
                    for _ind in range(left_ind + 1, ind):
                        s, e = self.cell_ranges[_ind]
                        result.append((_ind, s, e, s))
                        sum_sizes += e - s
                    result.append((ind, start, _end, start))
                    sum_sizes += _end - start
        assert sum_sizes == num, (sum_sizes, num)
        return [
            (ind, l - input_pos, r - input_pos, l - off, r - off)
            for ind, l, r, off in result
        ]

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
        result = None
        for ind, start1, end1, start2, end2 in self.get_ranges(
            input_pos, buffers.shape[1]
        ):
            result = self._set_checkpoint(
                layer_idx,
                buffers[:, start1:end1, :],
                ind,
                start2,
            )
        return result

    def _set_checkpoint(
        self,
        layer_idx: int,
        buffers: torch.Tensor,
        ind: int,
        rstart: int,
    ) -> Optional[int]:
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
        parts = [
            self._get_checkpoint(
                layer_idx,
                ind,
                start2,
                end2,
                device,
            )
            for ind, _, _, start2, end2 in self.get_ranges(input_pos, num)
        ]
        return torch.cat(parts, dim=1)

    def _get_checkpoint(
        self,
        layer_idx: int,
        ind: int,
        rstart: int,
        rend: int,
        device: torch.device,
    ) -> torch.Tensor:
        raise NotImplementedError


class LayerInputQuantizedCheckpoints(LayerInputCheckpoints):
    """
    Internally, we use :class:`KVCacheBufferQuantizedCheckpoints` objects,
    splitting two halves to keys and values.

    In fact, we use a list of :class:`KVCacheBufferQuantizedCheckpoints`
    objects, one for each cell range, which share their
    :class:`QuantizedKVCacheBuffers` object, in order to limit the amount
    of GPU memory being used.
    """

    def __init__(
        self,
        model: GPT,
        layer_numbers: List[int],
        cell_ranges: List[Tuple[int, int]],
        batch_size: int,
        qname: str,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        allocate_buffers: bool = False,
        device: Optional[torch.device] = None,
        pin_memory: Optional[List[bool]] = None,
    ):
        """
        We create a list of :class:`KVCacheBufferQuantizedCheckpoints` objects,
        one for each cell range.

        Args:
            model: GPT model
            layer_numbers: List of layer numbers for which inputs checkpoints
                are stored
            cell_ranges: List of tuples `(start, end)`, see above.
            batch_size: Batch size
            qname: Determines quantization buffers
            cache_kwargs: Additional keyword arguments for
                :class:`QuantizedKVCacheBuffers`.
            allocate_buffers: If `True`, we allocate buffer buffers here.
                Otherwise, they are allocated at first use
            device: Device for buffer allocations, needed if
                `allocate_buffers=True`
            pin_memory: If given, must have the same length as `layer_numbers`.
                Checkpoints for layers with `True` entries are pinned in CPU
                memory. Default: No checkpoints are pinned.
        """

        super().__init__(layer_numbers, cell_ranges)
        if pin_memory is not None and len(pin_memory) != len(layer_numbers):
            raise ValueError(
                f"pin_memory = {pin_memory}, layer_numbers = {layer_numbers}: Must have same length"
            )
        self.max_seq_length = model.max_seq_length
        max_cell_length = max(end - start for start, end in cell_ranges)
        # Create quantization buffer to be used
        cache_params = replace(
            model.get_kv_cache_params(0),
            max_batch_size=batch_size,
            n_query_groups=1,
            n_head=1,
            head_size=model.config.n_embd // 2,
            cache_length=max_cell_length,
        )
        if cache_kwargs is not None:
            dequant_kwargs = dict(max_num_ranges=cache_kwargs.get("max_num_ranges"))
        else:
            dequant_kwargs = None
        quant_buffers = create_quantized_kv_buffers(
            qname=qname,
            cache_lengths=[max_cell_length],
            cache_params=cache_params,
            cache_kwargs=cache_kwargs,
            dequant_kwargs=dequant_kwargs,
            allocate_buffers=allocate_buffers,
            device=device,
        )[0]
        # Internally, we use :class:`KVCacheBufferQuantizedCheckpoints` objects
        self._checkpoints_int = [
            KVCacheBufferQuantizedCheckpoints(
                chunk_numbers=layer_numbers,
                quant_buffers=quant_buffers,
                cache_length=end - start,
                pin_memory=pin_memory,
            )
            for start, end in cell_ranges
        ]
        self.n_embd = model.config.n_embd

    def clear(self):
        """
        The object is defunct after this method is called. Just part of an
        attempt to avoid GPU memory leaks.

        """
        self._checkpoints_int[0].quant_buffers.deallocate()
        self._checkpoints_int = None

    def _set_checkpoint(
        self,
        layer_idx: int,
        buffers: torch.Tensor,
        ind: int,
        rstart: int,
    ) -> Optional[int]:
        if layer_idx not in self.layer_numbers:
            return None
        cp_int = self._checkpoints_int[ind]
        if buffers.ndim != 3:
            raise ValueError(f"buffers.shape = {buffers.shape}, must be 3D")
        num = buffers.shape[1]
        batch_size = cp_int.batch_size
        if batch_size is None:
            batch_size = buffers.shape[0]
        shape = (batch_size, num, self.n_embd)
        if buffers.shape != shape:
            raise ValueError(f"buffers.shape = {buffers.shape}, must be {shape}")
        ne2 = self.n_embd // 2
        return cp_int.set_checkpoint_slice(
            chunk_idx=layer_idx,
            key=buffers[:, None, :, :ne2],
            value=buffers[:, None, :, ne2:],
            input_pos=rstart,
        )

    def _get_checkpoint(
        self,
        layer_idx: int,
        ind: int,
        rstart: int,
        rend: int,
        device: torch.device,
    ) -> torch.Tensor:
        if layer_idx not in self.layer_numbers:
            raise ValueError(
                f"layer_idx = {layer_idx} not in layer numbers [{self.layer_numbers}]"
            )
        cp_int = self._checkpoints_int[ind]
        k_and_v = cp_int.get_checkpoint_slice(
            chunk_idx=layer_idx,
            input_pos=rstart,
            num=rend - rstart,
            device=device,
        )
        return torch.cat(
            (k_and_v.keys().squeeze(1), k_and_v.values().squeeze(1)),
            dim=-1,
        )


class LayerInputDefaultCheckpoints(LayerInputCheckpoints):
    """
    Internally, we use :class:`KVCacheBufferDefaultCheckpoints` objects
    (one for each cell range), splitting two halves to keys and values.
    """

    def __init__(
        self,
        layer_numbers: List[int],
        cell_ranges: List[Tuple[int, int]],
        batch_size: int,
        n_embd: int,
        dtype: Optional[torch.dtype],
        pin_memory: Optional[List[bool]] = None,
    ):
        """
        Args:
            layer_numbers: List of layer numbers for which inputs checkpoints
                are stored
            cell_ranges: List of tuples `(input_pos, num)`, see above.
            batch_size: Batch size
            n_embd: Number of embedding dimensions
            dtype: Data type
            pin_memory: If given, must have the same length as `layer_numbers`.
                Checkpoints for layers with `True` entries are pinned in CPU
                memory. Default: No checkpoints are pinned.
        """

        super().__init__(layer_numbers, cell_ranges)
        if pin_memory is not None and len(pin_memory) != len(layer_numbers):
            raise ValueError(
                f"pin_memory = {pin_memory}, layer_numbers = {layer_numbers}: Must have same length"
            )
        self._buffer_params = KVCacheBuffersParams(
            max_batch_size=batch_size,
            n_query_groups=1,
            head_size=n_embd // 2,
            dtype=dtype,
            device=torch.device("cpu"),
        )
        self._checkpoints_int = [
            KVCacheBufferDefaultCheckpoints(
                chunk_numbers=layer_numbers,
                params=self._buffer_params,
                cache_length=end - start,
                pin_memory=pin_memory,
            )
            for start, end in cell_ranges
        ]
        self.n_embd = n_embd

    def pos_for_layer_idx(self, layer_idx: int) -> Optional[int]:
        return self._checkpoints_int[0].pos_for_chunk_idx(layer_idx)

    def _set_checkpoint(
        self,
        layer_idx: int,
        buffers: torch.Tensor,
        ind: int,
        rstart: int,
    ) -> Optional[int]:
        cp_int = self._checkpoints_int[ind]
        if buffers.ndim != 3:
            raise ValueError(f"buffers.shape = {buffers.shape}, must be 3D")
        num = buffers.shape[1]
        batch_size = cp_int.batch_size
        if batch_size is None:
            batch_size = buffers.shape[0]
        shape = (batch_size, num, self.n_embd)
        if buffers.shape != shape:
            raise ValueError(f"buffers.shape = {buffers.shape}, must be {shape}")
        ne2 = self.n_embd // 2
        device = torch.device("cpu")
        return cp_int.set_checkpoint_slice(
            chunk_idx=layer_idx,
            key=buffers[:, None, :, :ne2].to(device=device),
            value=buffers[:, None, :, ne2:].to(device=device),
            input_pos=rstart,
        )

    def _get_checkpoint(
        self,
        layer_idx: int,
        ind: int,
        rstart: int,
        rend: int,
        device: torch.device,
    ) -> torch.Tensor:
        cp_int = self._checkpoints_int[ind]
        if cp_int.pos_for_chunk_idx(layer_idx) is None:
            raise ValueError(
                f"layer_idx = {layer_idx} is not in layer numbers [{self.layer_numbers}]"
            )
        k_and_v = cp_int.get_checkpoint_slice(
            chunk_idx=layer_idx,
            input_pos=rstart,
            num=rend - rstart,
            device=device,
        )
        return torch.cat(
            (k_and_v.keys().squeeze(1), k_and_v.values().squeeze(1)),
            dim=-1,
        )
