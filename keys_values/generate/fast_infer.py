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
from functools import partial
from typing import Optional, Tuple

import torch

from keys_values.config import Config
from keys_values.attention.attention_utils import (
    range_from_input_pos,
    InputPosType,
)
from keys_values.attention.base import (
    MultiHeadSelfAttention,
    DefaultKeysAndValues,
    SDPA_IMPL_QPADDED_PYTORCH,
    SDPA_IMPL_EAGER_BLOCKS,
    SDPA_IMPL_FLEXATTENTION,
)
from keys_values.attention.sdpa_op import scatter_on_buffers
from keys_values.generate.single_token import SingleTokenLongContextInference
from keys_values.kvcache.attn_weights import AttnWeightsKVCache
from keys_values.kvcache.base import DefaultKVCache
from keys_values.kvcache.next_positions import NextPositionsMixin
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT
from keys_values.utils import index_to_3d


class SingleTokenInferenceKVCache(DefaultKVCache):
    """
    Specific :class:`DefaultKVCache` implementation used to build the single
    token inference graph in :class:`SingleTokenFastInferenceDriver`. What is
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
        token_positions: torch.Tensor,
        next_positions: torch.Tensor,
        input_pos: torch.Tensor,
        layer_idx: int,
        mha: MultiHeadSelfAttention,
    ):
        assert keys.ndim == 4
        assert keys.shape == values.shape
        assert token_positions.shape == keys.shape[:-1]
        assert next_positions.shape[:-1] == token_positions.shape[:-1]
        batch_size, n_query_groups, cache_length, head_size = keys.shape
        assert n_query_groups == config.n_query_groups
        assert head_size == config.head_size
        if not torch.compiler.is_compiling() and input_pos.item() < cache_length:
            raise ValueError(
                f"input_pos = {input_pos.item()}, must be >= cache_length = {cache_length}"
            )
        assert input_pos.ndim == 1 and input_pos.numel() == 1
        super().__init__(
            config=config,
            max_batch_size=batch_size,
            cache_length=cache_length,
            block_idx=layer_idx,
            dtype=keys.dtype,
            mha=mha,
        )
        self._orig_keys = keys
        self._orig_values = values
        self._next_positions = next_positions
        self._orig_token_positions = token_positions
        self._input_pos = input_pos
        self._new_keys = None
        self._new_values = None
        self._new_token_positions = None
        sliding_window_size = self.mha._get_sliding_window_size(layer_idx)
        if self.mha.has_flex_attention:
            sdpa_mode = SDPA_IMPL_FLEXATTENTION
        elif sliding_window_size is not None:
            print(
                "WARNING: config.sliding_window_size is used. This means that "
                "a naive SDPA kernel has to be used, for which computations "
                "can be slow. Consider switching to a model which does not use "
                "config.sliding_window_size, or use FlexAttention by passing "
                "flexatt_args when creating the MultiHeadSelfAttention object."
            )
            sdpa_mode = SDPA_IMPL_EAGER_BLOCKS
        else:
            sdpa_mode = SDPA_IMPL_QPADDED_PYTORCH
        self._sdpa_kwargs = dict(
            sdpa_mode=sdpa_mode,
            sliding_window_size=sliding_window_size,
        )

    @property
    def batch_size(self) -> Optional[int]:
        return self._orig_keys.shape[0]

    @property
    def input_pos(self) -> InputPosType:
        return self._input_pos, False

    def _forward_check_args(self, query, key, value, token_idx):
        if not torch.compiler.is_compiling():
            super()._forward_check_args(query, key, value, token_idx)

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
            raise IndexError("'forward' can be called only once")
        self._forward_check_args(query, key, value, token_idx)
        num = token_idx.shape[-1]
        # "scatter" update of KV cache buffers
        index = self._next_positions
        if index.shape[-1] > num:
            assert index.shape[-1] == num + 1, (index.shape, num)
            positions = index[..., num:]
            index = index[..., :num]
        else:
            if index.shape[-1] < num:
                raise ValueError(
                    f"Number of tokens {num} != {index.shape[-1]} = next_positions.shape[-1]"
                )
            positions = None
        self._new_keys, self._new_values = scatter_on_buffers(
            key,
            value,
            self._orig_keys,
            self._orig_values,
            index,
            positions=positions,
        )
        # Update `token_positions`
        new_tps = index_to_3d(
            range_from_input_pos(
                self._input_pos,
                0,
                num,
                device=self._orig_token_positions.device,
                dtype=self._orig_token_positions.dtype,
            ),
            *self._orig_token_positions.shape[:2],
        )
        if positions is not None:
            new_tps = torch.cat(
                (
                    self._orig_token_positions.gather(-1, positions),
                    new_tps,
                ),
                dim=-1,
            )
            index = self._next_positions
        self._new_token_positions = self._orig_token_positions.scatter(
            -1,
            index,
            new_tps,
        )
        attn_outputs, _ = self.mha.scaled_dot_product_attention(
            query=query,
            k_and_v=DefaultKeysAndValues(self._new_keys, self._new_values),
            input_pos=self.input_pos,
            token_positions=self._new_token_positions,
            return_attn_weights=False,
            **self._sdpa_kwargs,
        )
        return attn_outputs.reshape(self.batch_size, num, -1)


def single_token_forward(
    token_idx: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    token_positions: torch.Tensor,
    next_positions: torch.Tensor,
    input_pos: torch.Tensor,
    model: GPT,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Expression to be compiled in :class:`SingleTokenFastInferenceDriver`.

    Args:
        token_idx: New input tokens, `(batch_size, num)`
        keys: Keys buffers for all layers,
            `(n_layer, batch_size, n_query_groups, cache_length, head_size)`
        values: Values buffers for all layers,
            `(n_layer, batch_size, n_query_groups, cache_length, head_size)`
        token_positions: Indexes for all layers,
            `(n_layer, batch_size, n_query_groups, cache_length)`
        next_positions: Indexes for all layers,
            `(n_layer, batch_size, n_query_groups, num)`
        input_pos: Current input position as tensor, `(1,)`
        model: Model, whose KV caches are temporarily replaced by
            :class:`SingleTokenInferenceKVCache` objects

    Returns:
         `(logits, new_keys, new_values, new_token_positions)`

    """
    assert (
        keys.ndim == 5
        and values.ndim == 5
        and token_positions.ndim == 4
        and next_positions.ndim == 4
    )
    n_layer = keys.shape[0]
    config = model.config
    assert config.n_layer == n_layer
    # Create replacement KV caches
    orig_kv_caches = model.get_kv_caches()
    kv_caches = [
        SingleTokenInferenceKVCache(
            config=config,
            keys=keys[layer_idx],
            values=values[layer_idx],
            token_positions=token_positions[layer_idx],
            next_positions=next_positions[layer_idx],
            input_pos=input_pos,
            layer_idx=layer_idx,
            mha=cache.mha,
        )
        for layer_idx, cache in enumerate(orig_kv_caches)
    ]
    orig_kv_caches = model.get_kv_caches()
    model.assign_kv_caches(kv_caches)
    # Run forward pass
    logits = model(token_idx)
    # Collect new tensors to be returned
    model.assign_kv_caches(orig_kv_caches)
    new_keys = torch.cat(
        [cache.keys().unsqueeze(0) for cache in kv_caches],
        dim=0,
    )
    new_values = torch.cat(
        [cache.values().unsqueeze(0) for cache in kv_caches],
        dim=0,
    )
    new_token_positions = torch.cat(
        [cache.token_positions().unsqueeze(0) for cache in kv_caches],
        dim=0,
    )
    return logits, new_keys, new_values, new_token_positions


class SingleTokenFastInferenceDriver:
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

    Usage:
    - Pass GPT model and chunk size at construction. The graph is compiled if
      the KV caches of the model have allocated buffers already. Otherwise,
      graph compilation is delayed until the first :meth:`initialize` call.
    - Call :meth:`initialize` once the prompt for a new sequence batch has
      been processed and token generation is to be started.
    - Call :meth:`forward` up to `chunk_size` times. This method can be
      called as long as `num_tokens_left_in_chunk > 0`.
    - Once `num_tokens_left_in_chunk == 0`, call :meth:`update` and pass
      the `(batch_size, chunk_size)` tokens generated. This updates the
      original KV cache buffers and determines `next_positions` for the next
      chunk.
    """

    def __init__(
        self,
        model: GPT,
        chunk_size: int,
    ):
        self.model = model
        self.chunk_size = chunk_size
        self._check_init_args()
        self.batch_size = None
        self._next_positions = None
        self._pos_in_chunk = None
        self._keys_buffers = None
        self._values_buffers = None
        self._token_positions = None
        self._input_pos = None
        self._inference_graph = None
        if model.get_kv_caches()[0].batch_size is not None:
            # If KV caches are already initialized and allocated, we can compile
            # the graph here. Otherwise, it is compiled with the first call of
            # :meth:`initialize`
            self._init_buffers_and_compile_graph()

    def _check_init_args(self):
        cache_length = None
        for block_idx, cache in enumerate(self.model.get_kv_caches()):
            if not isinstance(cache, NextPositionsMixin):
                raise TypeError(
                    f"Block {block_idx}: type(cache) = {type(cache)}, must inherit from NextPositionsMixin"
                )
            if not isinstance(cache, DefaultKVCache):
                raise TypeError(
                    f"Block {block_idx}: type(cache) = {type(cache)}, must inherit from DefaultKVCache"
                )
            if cache_length is None:
                cache_length = cache.cache_length
            elif cache.cache_length != cache_length:
                raise ValueError(
                    f"Block {block_idx}: cache_length = {cache.cache_length}, must all be the same (but also {cache_length})"
                )

    def _init_buffers_and_compile_graph(self, recreate_graph: bool = True):
        fst_cache = self.model.get_kv_caches()[0]
        batch_size = fst_cache.batch_size
        cache_length = fst_cache.cache_length
        device = fst_cache.device
        dtype = fst_cache.dtype
        tp_dtype = fst_cache.token_positions().dtype
        if (
            device is not None
            and dtype is not None
            and (
                recreate_graph
                or self.batch_size is None
                or self.batch_size != batch_size
            )
        ):
            # Create buffers
            self.batch_size = batch_size
            config = self.model.config
            shape = (
                config.n_layer,
                batch_size,
                config.n_query_groups,
                cache_length,
                config.head_size,
            )
            kwargs = dict(device=device, dtype=dtype)
            self._keys_buffers = torch.zeros(shape, **kwargs)
            self._values_buffers = torch.zeros(shape, **kwargs)
            kwargs["dtype"] = tp_dtype
            self._token_positions = torch.zeros(shape[:-1], **kwargs)
            self._input_pos = torch.zeros((1,), **kwargs)
            # Compile the graph
            torch._dynamo.reset()
            torch.compiler.reset()
            self._inference_graph = torch.compile(
                partial(single_token_forward, model=self.model),
                fullgraph=True,
                dynamic=False,
            )

    def _check_is_initialized(self):
        if self.batch_size is None:
            raise IndexError("Call 'initialize' to initialize the object")

    def _copy_from_caches(self):
        kv_caches = self.model.get_kv_caches()
        for block_idx, cache in enumerate(kv_caches):
            self._token_positions[block_idx] = cache.token_positions()
            k_and_v = cache.get_keys_values()
            self._keys_buffers[block_idx] = k_and_v.keys()
            self._values_buffers[block_idx] = k_and_v.values()

    def _set_next_positions(self):
        """
        Sets `_next_positions` and `_input_pos` from current caches

        """
        parts = [
            cache.next_positions_ext(self.chunk_size).clone().unsqueeze(0)
            for cache in self.model.get_kv_caches()
        ]
        shapes = {tuple(p.shape) for p in parts}
        if len(shapes) > 1:
            raise IndexError(
                f"next_positions: Shapes must be the same for all layers, but: {shapes}"
            )
        self._next_positions = torch.cat(parts, dim=0)
        self._input_pos[0] = self.model.get_kv_caches()[0].input_pos

    def initialize(self, recreate_graph: bool = True):
        """
        This method initializes the object, in that current KV cache buffers
        are replicated into tensor nodes, and the single token inference
        graph is built and compiled.

        NOTE: If this is called with `recreate_graph=False`, the graph is not
        recompiled (unless the batch size changed). Use this only as long as
        the same batch sequence is processed. Reasons for the graph to become
        stale:

        - Model weights have changed
        - Sequence length changed: Position encoding changed

        Args:
            recreate_graph: If `False` and the batch size stays the same,
                the graph is not recreated. Use this with care.

        """
        fst_cache = self.model.get_kv_caches()[0]
        self.batch_size = fst_cache.batch_size
        input_pos = fst_cache.input_pos
        if input_pos < fst_cache.cache_length:
            raise IndexError("Can only initialize when all caches are full")
        for cache in self.model.get_kv_caches():
            assert cache.input_pos == input_pos, (
                cache.input_pos,
                input_pos,
            )
            assert cache.batch_size == self.batch_size, (
                cache.batch_size,
                self.batch_size,
            )
            assert cache.device is not None
            assert cache.dtype is not None
        # Allocate buffers and compile graph if not yet done before, or if
        # batch size has changed
        self._init_buffers_and_compile_graph(recreate_graph)
        # Copy buffers/indexes from current caches and set `next_positions`
        self._copy_from_caches()
        self._set_next_positions()
        self._pos_in_chunk = 0

    def forward(self, token_idx: torch.Tensor) -> torch.Tensor:
        """
        Inference for batch of single tokens, using the compiled graph.
        Works only if `num_tokens_left_in_chunk > 0`. If the chunk has
        been processed, call :meth:`update`.

        Args:
            token_idx: Batch of single tokens, `(batch_size, 1)`

        Returns:
              Logits for single token batch

        """
        self._check_is_initialized()
        if self.num_tokens_left_in_chunk <= 0:
            raise IndexError(f"Call 'update' for another {self.chunk_size} tokens")
        if token_idx.shape != (self.batch_size, 1):
            raise ValueError(
                f"token_idx.shape = {token_idx.shape}, must be {(self.batch_size, 1)}"
            )
        cp = self._pos_in_chunk
        next_positions = self._next_positions[:, :, :, cp : (cp + 1)]
        logits, self._keys_buffers, self._values_buffers, self._token_positions = (
            self._inference_graph(
                token_idx=token_idx,
                keys=self._keys_buffers,
                values=self._values_buffers,
                token_positions=self._token_positions,
                next_positions=next_positions,
                input_pos=self._input_pos,
            )
        )
        self._input_pos[0] += 1
        self._pos_in_chunk += 1
        return logits

    @property
    def num_tokens_left_in_chunk(self) -> int:
        return self.chunk_size - self._pos_in_chunk

    def update(self, token_idx: torch.Tensor):
        """
        Runs forward pass `model.forward` with tokens `token_idx`. Afterwards,
        `_next_positions` is updated from the KV caches. These indexes cover
        the next `chunk_size` tokens. The defaylt use case is to call this
        method once `num_tokens_left_in_chunk == 0`, passing the `num_chunks`
        tokens which have been sampled for the chunk. But this method can be
        called also if `num_tokens_left_in_chunk > 0`, and `token_idx` can
        have a length different from `chunk_size - num_tokens_left_in_chunk`.

        Note: Importantly, this update of KV cache states is not the same as
        what is obtained by single token updates done with each :meth:`forward`
        call, for several reasons:
        - The KV cache buffers may use quantization, whereas the replicas
          updated in :meth:`forward` are exact
        - More important: A sequence of `chunk_size` single token updates is
          not the same as a single update with `chunk_size` tokens (i.e., what
          is done here)

        Args:
            token_idx: New token indices, `(batch_size, num)`

        """
        self._check_is_initialized()
        if token_idx.ndim != 2 or token_idx.shape[0] != self.batch_size:
            raise ValueError(
                f"token_idx.shape = {token_idx.shape}: Must be 2D and first axis {self.batch_size}"
            )
        # Forward pass: Update all KV caches
        logits = self.model(token_idx)
        # Copy new buffer/index contents from caches, set `next_positions`
        self._copy_from_caches()
        self._set_next_positions()
        self._pos_in_chunk = 0


class FastSingleTokenInference(SingleTokenLongContextInference):
    """
    Wraps :class:`SingleTokenFastInferenceDriver`. The driver object is
    created here.
    """

    def __init__(
        self,
        model: LongContextInferenceModel,
        chunk_size: int,
        recreate_graph: bool = True,
    ):
        self._model = model
        self._driver = SingleTokenFastInferenceDriver(
            model=model.gpt_model,
            chunk_size=chunk_size,
        )
        self._recreate_graph = recreate_graph
        self._token_idx_since_last_update = None

    @property
    def model(self) -> LongContextInferenceModel:
        return self._model

    def process_prompt(self, prompt: torch.Tensor) -> torch.Tensor:
        logits = self._model(prompt, targets=None)
        self._driver.initialize(recreate_graph=self._recreate_graph)
        self._token_idx_since_last_update = []
        return logits

    def inference_for_token(self, token_idx: torch.Tensor) -> torch.Tensor:
        if self._driver.num_tokens_left_in_chunk == 0:
            self.update_model_with_remaining_tokens()
        self._token_idx_since_last_update.append(token_idx)
        return self._driver.forward(token_idx)

    def update_model_with_remaining_tokens(self):
        """
        Can be called to update `model` with tokens returned by
        :meth:`inference_for_token` since the last recent update due to the
        chunk being completed.

        """
        if self._token_idx_since_last_update:
            token_idx_since_last_update = torch.cat(
                [x.view(-1, 1) for x in self._token_idx_since_last_update],
                dim=-1,
            )
            self._driver.update(token_idx_since_last_update)
            self._token_idx_since_last_update = []
