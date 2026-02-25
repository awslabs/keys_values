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
from typing import Tuple, Iterable, Dict, Any, List, Union

import torch

from litgpt.config import Config

from keys_values.kvcache.base import KVCache
from keys_values.model import Block, GPT


class CellBlocks:
    """
    Base class for representing the stack of blocks underlying a cell, see
    :class:`CellComputation`.

    """

    def __init__(self, config: Config):
        self.config = config

    @property
    def max_seq_length(self) -> int:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, chunk_len, n_embd = x.shape
        if n_embd != self.config.n_embd:
            raise ValueError(
                f"x.shape[2] = {n_embd} != {self.config.n_embd} = config.n_embd"
            )
        if idx.shape != (batch_size, chunk_len):
            raise ValueError(
                f"idx.shape = {idx.shape}, must be {(batch_size, chunk_len)}"
            )
        if chunk_len > self.max_seq_length:
            raise ValueError(
                f"Cannot forward chunk of length {chunk_len}, max seq length is only {self.max_seq_length}"
            )
        for block_idx, cache in self.get_kv_caches():
            self._check_kv_cache(cache, block_idx, batch_size, chunk_len)
        # Loop over blocks
        for block_idx, block, block_kwargs in self.blocks_with_kwargs():
            x = block(x, token_idx=idx, **block_kwargs)

        return x

    @property
    def first_layer_idx(self) -> int:
        raise NotImplementedError

    @property
    def num_layers(self) -> int:
        raise NotImplementedError

    def blocks_with_kwargs(self) -> Iterable[Tuple[int, Block, Dict[str, Any]]]:
        """
        Returns:
            Sequence of `(block_idx, block, block_kwargs)` of model blocks,
            in increasing order. We call
            `x = block(x, token_idx, **block_kwargs)`.

        """
        raise NotImplementedError

    def blocks(self) -> Iterable[Tuple[int, Block]]:
        return [(a, b) for a, b, _ in self.blocks_with_kwargs()]

    def get_kv_caches(self) -> List[Tuple[int, KVCache]]:
        return [
            (block_idx, block.attn.kv_cache)
            for block_idx, block, _ in self.blocks_with_kwargs()
        ]

    @staticmethod
    def _check_kv_cache(
        kv_cache: KVCache,
        block_idx: int,
        batch_size: int,
        chunk_len: int,
    ):
        if kv_cache is not None:
            if kv_cache.input_pos == 0:
                if kv_cache.max_batch_size < batch_size:
                    raise ValueError(
                        f"Batch size {batch_size} is too large for KV cache layer {block_idx} (batch size {kv_cache.max_batch_size}). Use 'assign_kv_caches' or `set_kv_caches'"
                    )
            else:
                if kv_cache.max_forward_length() < chunk_len:
                    print(f"UUPS: kv_cache.input_pos = {kv_cache.input_pos}")  # DEBUG
                    raise ValueError(
                        f"KV cache for layer {block_idx}: chunk_len = {chunk_len}, must be <= max_forward_length() = {kv_cache.max_forward_length()}"
                    )

    def assign_kv_caches(
        self,
        kv_caches: Union[List[KVCache], List[Tuple[int, KVCache]]],
    ):
        if len(kv_caches) != self.num_layers:
            raise ValueError(
                f"kv_caches must have one entry per layer, so {self.num_layers} entries"
            )
        if isinstance(kv_caches[0], KVCache):
            for cache, (_, block) in zip(kv_caches, self.blocks()):
                block.attn.kv_cache = cache
        else:
            for (_, cache), (_, block) in zip(kv_caches, self.blocks()):
                block.attn.kv_cache = cache

    def clear_kv_caches(self):
        for _, block in self.blocks():
            block.attn.kv_cache = None


class DefaultCellBlocks(CellBlocks):
    def __init__(
        self,
        model: GPT,
        first_layer_idx: int,
        num_layers: int,
    ):
        super().__init__(model.config)
        self._model = model
        self._first_layer_idx = first_layer_idx
        self._num_layers = num_layers

    @property
    def max_seq_length(self) -> int:
        return self._model.max_seq_length

    def blocks_with_kwargs(self) -> Iterable[Tuple[int, Block, Dict[str, Any]]]:
        start = self._first_layer_idx
        end = start + self._num_layers
        block_kwargs = dict(mha=self._model.mha)
        return zip(
            range(start, end),
            self._model.transformer.h[start:end],
            [block_kwargs] * self._num_layers,
        )

    @property
    def first_layer_idx(self) -> int:
        return self._first_layer_idx

    @property
    def num_layers(self) -> int:
        return self._num_layers
