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
from typing import Optional, Tuple

import torch

from keys_values.config import Config
from keys_values.kvcache.attn_weights import (
    update_token_positions,
    UpdateTokenPositionsGracePeriod,
)
from keys_values.kvcache.base import DefaultKVCache
from keys_values.kvcache.next_positions import NextPositionsMixin
from keys_values.model import GPT


class SingleTokenInferenceKVCache(DefaultKVCache):
    """
    Specific :class:`DefaultKVCache` implementation used to build the single
    token inference graph in :class:`SingleTokenFastInference`. What is
    special here:

    - `input_pos >= cache_length`, KV buffers must be full
    - :meth:`forward` can be called once only, with the chunk size given by
      `next_positions` (for single token inference, this is 1)
    - KV buffers represented by specific tensors: `_orig_keys`, `_orig_values`
      passed at construction, and `_new_keys`, `_new_values` after the
      update in the :meth:`forward` call.
    """

    def __init__(
        self,
        config: Config,
        keys: torch.Tensor,
        values: torch.Tensor,
        next_positions: torch.Tensor,
        token_positions: torch.Tensor,
        input_pos: int,
        layer_idx: int,
        **base_kwargs,
    ):
        assert keys.ndim == 4
        assert keys.shape == values.shape
        assert token_positions.shape == keys.shape[:-1]
        assert next_positions.shape[:-1] == token_positions.shape[:-1]
        batch_size, n_query_groups, cache_length, head_size = keys.shape
        assert n_query_groups == config.n_query_groups
        assert head_size == config.head_size
        if input_pos < cache_length:
            raise ValueError(f"input_pos = {input_pos}, must be >= cache_length = {cache_length}")
        super().__init__(
            config=config,
            max_batch_size=batch_size,
            cache_length=cache_length,
            block_idx=layer_idx,
            dtype=keys.dtype,
            **base_kwargs,
        )
        self._orig_keys = keys
        self._orig_values = values
        self._next_positions = next_positions
        self._orig_token_positions = token_positions
        self._input_pos = input_pos
        self._new_keys = None
        self._new_values = None
        self._new_token_positions = None

    @property
    def batch_size(self) -> Optional[int]:
        return self._orig_keys.shape[0]

    def token_positions(self) -> torch.Tensor:
        if self._new_token_positions is None:
            return self._orig_token_positions
        else:
            return self._new_token_positions

    def keys(self) -> torch.Tensor:
        if self._new_keys is None:
            raise IndexError("Call 'forward' first")
        return self._new_keys

    def values(self) -> torch.Tensor:
        if self._new_values is None:
            raise IndexError("Call 'forward' first")
        return self._new_values

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self._new_keys is not None:
            raise IndexError("'forward' can be called once only")
        self._forward_check_args(query, key, value, token_idx)
        num = token_idx.shape[-1]
        if self._next_positions.shape[-1] != num:
            raise ValueError(f"Number of tokens {num} != {self._next_positions.shape[-1]} = next_positions.shape[-1]")
        # HIER! What about index, positions?
        # ==> Would not have trouble with new_kvcache3 !


class SingleTokenFastInference:
    """
    Driver for fast inference in a single new token. This is needed during
    token generation.

    Concepts:
    - Group single token calls into chunks of size `chunk_size`. All expensive
      computations are done at chunk boundaries only.
    - Use compiled graph (`torch.compile`) for the complete single token forward
      expression
    - Within a chunk, KV buffers are maintained as tensors on GPU and are not
      written back. If KV caches use quantization, this is not used.
    - Both writing back cache buffers and policy decisions (i.e., which slots to
      overwrite next) are done at chunk boundaries only.

    Choices:
    - The inference graph returns the updated KV buffers as outputs, and they
      replace the tensors used as input. An alternative would be to only return
      the new keys, values and write them into existing buffers (in place).
      TODO: Could compare w.r.t. speed and peak GPU memory usage.
    - Once `chunk_size` tokens are sampled, we need to update the KV caches (not
      just the buffer nodes replicated here) and make decisions for the next
      `chunk_size` slots. This is done by running a normal forward pass through
      the model, updating internal states of caches as needed.
      An alternative would be to write back the buffer node replicas and change
      inner states according to the fixed decisions. We'd not have to recompute
      the updates then. But this would be difficult to support generically, and
      a chunk update of KV caches is cheap anyways.
      TODO: If the added time for the extra update is significant, we could
      explore this in the future.

    TODO:
    - Method to create graph from current KV cache buffers and model weights:
      Batch size can be determined there as well
    """

    def __init__(
        self,
        model: GPT,
        chunk_size: int,
    ):
        self.model = model
        self.batch_size = None
        self.chunk_size = chunk_size
        self._next_positions = None
        self._num_tokens_left_in_chunk = 0

    # - All caches must implement NextPositionsMixin
    # - All caches must have the same length (current restriction)
    def _check_init_args(self):
        cache_length = None
        for block_idx, cache in enumerate(self.model.get_kv_caches()):
            if not isinstance(cache, NextPositionsMixin):
                raise TypeError(f"Block {block_idx}: type(cache) = {type(cache)}, must inherit from NextPositionsMixin")
            if cache_length is None:
                cache_length = cache.cache_length
            elif cache.cache_length != cache_length:
                raise ValueError(f"Block {block_idx}: cache_length = {cache.cache_length}, must all be the same (but also {cache_length})")

    def _check_is_initialized(self):
        if self.batch_size is None:
            raise IndexError("Call 'initialize' to initialize the object")

    # TODO!
    # - Take batch_size from caches
    # - Check that all cache buffers are full
    def initialize(self):
        """
        This method initializes the object, in that current KV cache buffers
        are replicated into tensor nodes, and the single token inference
        graph is built and compiled.

        """
        pass

    @property
    def num_tokens_left_in_chunk(self) -> int:
        return self._num_tokens_left_in_chunk

    def update(self, token_idx: torch.Tensor):
        """
        Runs forward pass `model.forward` with tokens `token_idx`. Afterwards,
        `_next_positions` is updated from the KV caches. These indexes cover
        the next `chunk_size` tokens. The defaylt use case is to call this
        method once `num_tokens_left_in_chunk == 0`, passing the `num_chunks`
        tokens which have been sampled for the chunk. But this method can be
        called also if `num_tokens_left_in_chunk > 0`, and `token_idx` can
        have a length different from `chunk_size - num_tokens_left_in_chunk`.

        Args:
            token_idx: New token indices, `(batch_size, num)`

        """
        if token_idx.ndim != 2 or token_idx.shape[0] != self.batch_size:
            raise ValueError(f"token_idx.shape = {token_idx.shape}: Must be 2D and first axis {self.batch_size}")
        # Run forward pass. This updates all KV caches
        logits = self.model(token_idx)
        self._next_positions = [
            cache.next_positions(self.chunk_size).clone()
            for cache in self.model.get_kv_caches()
        ]
        self._num_tokens_left_in_chunk = self.chunk_size
