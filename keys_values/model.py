# Original Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# Modification Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""
from typing import Any, List, Optional, Union, Callable
from typing_extensions import Self

import torch
import torch.nn as nn

from litgpt.config import Config
from litgpt.scripts.convert_hf_checkpoint import qkv_reassemble

from keys_values.attention import (
    DefaultKeysAndValues,
    MultiHeadSelfAttention,
    do_softcapping,
)
from keys_values.kvcache.base import KVCacheParams, KVCache, DefaultKVCache
from keys_values.kvcache.basics import KVCacheWithBuffers
from keys_values.use_eager_kernel import transform_mha_kwargs
from keys_values.utils import copy_parameters


# See `GPT.set_start_of_layer_hook`. A start of layer hook is called just before
# a layer is computed. The call is `hook(x, block_idx)`, where `x` is the layer
# input, `block_idx` the number of the layer. The position in the sequence
# is not passed, it must be tracked by the caller of the hook.
StartOfLayerHook = Callable[[Any, int], None]


class GPT(nn.Module):
    def __init__(
        self,
        config: Config,
        **mha_kwargs,
    ) -> None:
        """
        Args:
            config: Configuration parameters
            mha_kwargs: Extra arguments passed to :class:`MultiHeadSelfAttention`.
                For example, `pos_encoding` sets the position encoding.

        """
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(
            config.n_embd,
            config.padded_vocab_size,
            bias=config.lm_head_bias,
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, block_idx) for block_idx in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.mha = MultiHeadSelfAttention(
            config, **transform_mha_kwargs(mha_kwargs, config),
        )
        self.max_seq_length = config.block_size
        self._start_of_layer_hook = None
        # Have dense KV caches been created by `set_kv_caches`?
        self._default_kv_cache = False

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        Calls to :meth:`forward` must be such that `idx.shape[-1] + input_pos`
        is no larger than the maximum sequence length.

        This length determines the position encoding. For fine-tuning, we
        recommend to set it to the batch length for every new batch processed.
        If KV caches are of type `DenseKVCache`, and they are too small to hold
        `value` entries, a warning message is printed.

        Note: Do not change the maximum sequence length in the middle of an
        inference run, consisting of several processing and generation steps.
        The keys stored in the KV cache are already encoded and would not be
        recoded. We plan to support dynamic position encoding in the future.

        Args:
            value: New value for `max_seq_length`. This can be larger than
                `config.block_size`, which is the context width used during
                training. It can also be smaller.

        """
        from keys_values.kvcache.basics import DenseKVCache

        if value <= 0:
            raise ValueError(f"value = {value}, must be positive")
        self._max_seq_length = value
        # KV caches and sequence length.
        # We do not change them here, but output a warning if default caches are
        # too small
        for l_ix, kv_cache in enumerate(self.get_kv_caches()):
            if kv_cache is not None:
                if isinstance(kv_cache, DenseKVCache) and kv_cache.cache_length < value:
                    print(
                        f"KV cache for layer {l_ix} too small: Call 'set_kv_caches(batch_size={kv_cache.max_batch_size}, max_seq_length={value}) before inference"
                    )
                    break
                if isinstance(kv_cache, DefaultKVCache):
                    # Multi-head attention (includes position encoding)
                    kv_cache.mha.set_seq_length(value)

        # Multi-head attention (includes position encoding)
        self.mha.set_seq_length(value)

    def are_kv_caches_assigned(self) -> bool:
        status = [kv_cache is not None for kv_cache in self.get_kv_caches()]
        result = any(status)
        if result and not all(status):
            raise IndexError("Some layers have KV caches assigned, but not all")
        return result

    def _num_layers(self) -> int:
        has_no_layers = self.transformer is None or not hasattr(self.transformer,"h")
        return 0 if has_no_layers else len(self.transformer.h)

    def assign_kv_caches(self, kv_caches: List[KVCache]):
        """
        Assigns specific KV caches to the multi-head attention blocks
        of each layer. KV caches are required for inference. If no KV caches
        are assigned, inference calls fail.

        Args:
            kv_caches: KV caches, one for each layer of the model

        """
        num_layers = self._num_layers()
        if len(kv_caches) != num_layers:
            raise ValueError(f"len(kv_caches) = {len(kv_caches)}, must be {num_layers}")
        num_none = sum(c is None for c in kv_caches)
        if num_none == 0:
            batch_size = kv_caches[0].max_batch_size
            dtype = kv_caches[0].dtype
            for cache in kv_caches:
                self._check_kv_cache(self.config, cache, batch_size, dtype)
        elif num_none != num_layers:
            raise ValueError(f"kv_caches must not contain None or all be None")
        for cache, block in zip(kv_caches, self.transformer.h):
            block.attn.kv_cache = cache

    def set_kv_caches(
        self,
        batch_size: int,
        dtype: Optional[torch.dtype] = None,
        max_seq_length: Optional[int] = None,
    ):
        """
        This method can be called only if KV caches have not been assigned
        with :meth:`assign_kv_caches`. It creates default (dense) KV caches
        for every layer. These may require a lot of memory. If this is an
        issue, consider :meth:`assign_kv_caches` with KV caches of restricted
        size.

        KV caches are required for inference. If no KV caches are assigned,
        inference calls fail.

        Args:
            batch_size: Inference batch size
            dtype: Data type for buffers
            max_seq_length: Cache length. If not given, we use
                `self.max_seq_length`

        """
        if self.are_kv_caches_assigned() and not self._default_kv_cache:
            raise ValueError("Model has KV caches assigned already")
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        for block in self.transformer.h:
            attn = block.attn
            kv_cache = attn.kv_cache
            if (
                kv_cache is None
                or kv_cache.max_batch_size != batch_size
                or kv_cache.cache_length != max_seq_length
                or kv_cache.dtype != dtype
            ):
                if kv_cache is not None:
                    dtype = kv_cache.dtype if dtype is None else dtype
                attn.create_default_kv_cache(
                    batch_size=batch_size,
                    dtype=dtype,
                    max_sequence_length=max_seq_length,
                )
        self._default_kv_cache = True

    def get_kv_caches(self) -> List[KVCache]:
        return [block.attn.kv_cache for block in self.transformer.h]

    def reset(self) -> None:
        """
        Should be called before a new batch of sequences is processed. Passes
        `max_seq_length` to `mha`, and resets all KV caches (if any).

        """
        # Trigger resetting the rope-cache
        self.mha.set_seq_length(self.max_seq_length)
        for cache in self.get_kv_caches():
            if cache is not None:
                cache.reset()

    @property
    def start_of_layer_hook(self) -> Optional[StartOfLayerHook]:
        return self._start_of_layer_hook

    def set_start_of_layer_hook(
        self,
        hook: Optional[StartOfLayerHook],
    ):
        """
        Sets a function `hook(x, block_idx)`, which is called in
        :meth:`forward` at the start of each layer. Here, `x` is the layer
        input, `block_idx` the number of the layer. The hook is also called
        with the output of the final layer (input of head model), where
        `block_idx=self.config.n_layer`.

        The default start of layer hook is `self.config.start_of_layer_hook`.
        This is overwritten here.

        Args:
            hook: Hook function to be set, or `None` to remove hook

        """
        self._start_of_layer_hook = hook

    @staticmethod
    def _check_kv_cache(
        config: Config,
        kv_cache: KVCache,
        batch_size: int,
        dtype: torch.dtype,
    ):
        params = kv_cache.get_params()
        if config.n_query_groups != params.n_query_groups:
            raise ValueError(
                f"config and kv_cache not compatible: config.n_query_groups = {config.n_query_groups} != {params.n_query_groups} = kv_cache.n_query_groups"
            )
        if config.n_head != params.n_head:
            raise ValueError(
                f"config and kv_cache not compatible: config.n_head = {config.n_head} != {params.n_head} = kv_cache.n_head"
            )
        head_size = config.head_size
        if head_size != params.head_size:
            raise ValueError(
                f"config and kv_cache not compatible: config.head_size = {head_size} != {params.head_size} = kv_cache.head_size"
            )
        if batch_size != params.max_batch_size:
            raise ValueError(f"kv_cache.batch_size = {params.max_batch_size}, must be {batch_size}")
        if dtype != params.dtype:
            raise ValueError(f"kv_cache.dtype = {params.dtype}, must be {dtype}")

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        skip_lm_head: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        There are two different contexts in which this method is called:

        - Training: No KV caches assigned.
        - Inference and generation: KV caches must be assigned. In this case,
          a long batch of token sequences is processed by multiple
          :meth:`forward` calls. The number of tokens processed so far is
          tracked by `input_pos` in the KV caches. Internally, they treat
          the case `input_pos == 0` (prefill) different from `input_pos > 0`.
          Note that :meth:`reset` must be called before a new batch of
          sequences is processed.

        Args:
            idx: Token indices of input sequences, shape `(batch_size, num)`
            skip_lm_head: If `True`, we do not apply the final LM head
                `self.lm_head`.

        Returns:
            Logit outputs, shape `(batch_size, num, config.padded_vocab_size)`.
            If `skip_lm_head` is `True`, we return the final layer outputs,
            shape `(batch_size, num, config.n_embd)`.

        """
        if idx.ndim == 1:
            idx = idx.unsqueeze(0)
        elif idx.ndim != 2:
            raise ValueError(f"idx must be 1D or 2D tensor, but idx.shape = {idx.shape}")
        num = idx.size(1)
        if self.max_seq_length < num:
            raise ValueError(f"Cannot forward sequence of length {num}, max seq length is only {self.max_seq_length}.")

        x = self.transformer.wte(idx)  # (batch_size, num, n_embd)
        if self.config.scale_embeddings:
            x = x * (self.config.n_embd**0.5)

        hook = self._start_of_layer_hook
        for block_idx, block in enumerate(self.transformer.h):
            if hook is not None:
                # Call start of layer hook, passing detached layer input
                hook(x.detach(), block_idx)
            x = block(x, idx, self.mha)

        if hook is not None:
            # Hook is also called for the input to the head block
            hook(x.detach(), self.config.n_layer)
        x = self.transformer.ln_f(x)
        if skip_lm_head:
            return x
        return do_softcapping(
            self.lm_head(x),
            thresh=self.config.final_logit_softcapping,
        )

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def clear_kv_caches(self) -> None:
        for block in self.transformer.h:
            block.attn.kv_cache = None
        self._default_kv_cache = False

    def get_kv_cache_params(self, block_idx: int) -> Optional[KVCacheParams]:
        """
        Args:
            block_idx: Layer for which KV cache params are requested

        Returns:
            Parameters for KV caches (see above), or `None` if KV caches are
            not assigned.

        """
        num_layers = self._num_layers()
        if not (0 <= block_idx < num_layers):
            raise IndexError(f"block_idx={block_idx}, must be in [0, {num_layers}])")
        kv_cache = self.transformer.h[block_idx].attn.kv_cache
        return None if kv_cache is None else kv_cache.get_params()

    def kv_cache_max_tokens_forward(self) -> Optional[int]:
        """
        Returns:
            Smallest `max_tokens_forward` over all KV caches, or `None` if KV
            caches are not assigned.

        """
        if not self.are_kv_caches_assigned():
            return None
        else:
            return min(kvc.max_tokens_forward for kvc in self.get_kv_caches())

    def kv_cache_max_prefill_length(self) -> Optional[int]:
        """
        Returns:
            Smallest `max_prefill_length` over all KV caches, or `None` if KV
            caches are not assigned.

        """
        if not self.are_kv_caches_assigned():
            return None
        else:
            return min(c.max_prefill_length for c in self.get_kv_caches())

    def _empty_clone(self, device: Optional[torch.device] = None) -> "GPT":
        """
        Creates empty clone of this object. Parameters are not copied. The
        clone uses the same `mha`.

        Args:
            device: Device to create clone on. If not given, the default
                device is used.

        """
        if device is None:
            model_copy = GPT(self.config)
        else:
            with torch.device(device):
                model_copy = GPT(self.config)
        model_copy.mha = self.mha
        return model_copy

    def clone(self, device: Optional[torch.device] = None) -> "GPT":
        """
        Creates and returns a copy of this object, situated on device `device`.
        All named parameter tensors are copied. The copy is not entirely deep:

        - Copy uses the same `self.mha` than this object
        - For KV caches (if any), we use :meth:`KVCache.clone`, which produce
          shallow copies, in that the underlying buffers are the same, but set
          to a different device. For this to work, all buffers must be
          deallocated.

        Do not use the copy and the original at the same time! A typical use
        case is to create a copy of the model on a device, run computations
        there, and delete the copy before using the original model again.

        Args:
            device: Device on which the copy is created.

        Returns:
            Copy of this object on device `device` (or the default device).

        """
        kv_caches = []
        try:
            # Remove KV caches before copy is created
            for l_ix, block in enumerate(self.transformer.h):
                kv_cache = block.attn.kv_cache
                if kv_cache is not None and isinstance(kv_cache, KVCacheWithBuffers) and kv_cache.buffers_are_allocated:
                    raise ValueError(f"KV cache of layer {l_ix} has buffers allocated. Deallocate buffers with `deallocate_kv_cache_buffers_of_model`")
                kv_caches.append(kv_cache)
                block.attn.kv_cache = None
            # Create empty copy
            model_copy = self._empty_clone(device)
            copy_parameters(self, model_copy)
        finally:
            self.assign_kv_caches(kv_caches)
        # Deal with KV caches
        model_copy.assign_kv_caches(
            [None if c is None else c.clone() for c in kv_caches]
        )
        return model_copy


class Block(nn.Module):
    def __init__(
        self,
        config: Config,
        block_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> None:
        super().__init__()
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No checkpoint amongst the ones we support uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = nn.Identity() if not config.norm_1 else config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, block_idx, kv_cache=kv_cache)
        self.post_attention_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_attention_norm else nn.Identity()
        )
        self.norm_2 = (
            nn.Identity()
            if not config.norm_2
            else (None if config.shared_attention_norm else config.norm_class(config.n_embd, eps=config.norm_eps))
        )
        self.mlp = config.mlp_class(config)
        self.post_mlp_norm = (
            config.norm_class(config.n_embd, eps=config.norm_eps) if config.post_mlp_norm else nn.Identity()
        )
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        token_idx: torch.Tensor,
        mha: MultiHeadSelfAttention,
    ) -> torch.Tensor:
        """
        Non-parallel residual       Parallel residual
           ┌─ x                     ┌─ x ──────────────────┐             Note: if `shared_attention_norm` is True,
           │  ↓                     │  ↓                   ↓                   the output from `norm_1` is reused
           │  norm_1                │  norm_1  ───────►    norm_2
           │  ↓                     │  ↓                   ↓
           │  attn                  │  attn                MLP
           │  ↓                     │  ↓                   ↓
           |  post_attn_norm        |  post_attn_norm      post_mlp_norm
           |  ↓                     |  ↓                   ↓
        ┌─ └► +                     └► + ◄─────────────────┘
        |     ↓
        │     norm_2
        │     ↓
        │     MLP
        │     ↓
        |     post_mlp_norm
        |     ↓
        └───► +
        """

        x_normed = self.norm_1(x)
        attention_output = self.post_attention_norm(
            self.attn(x_normed, token_idx=token_idx, mha=mha)
        )
        if self.config.parallel_residual:
            if not self.config.shared_attention_norm:
                x_normed = self.norm_2(x)
            x = attention_output + x
        else:
            x = attention_output + x
            x_normed = self.norm_2(x)
        return self.post_mlp_norm(self.mlp(x_normed)) + x


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        config: Config,
        block_idx: int,
        kv_cache: Optional[KVCache] = None,
    ) -> None:
        super().__init__()
        # key, query and value projections for all heads, but in a batch
        self.qkv = nn.Linear(
            config.n_embd,
            (config.n_head + 2 * config.n_query_groups) * config.head_size,  # support for grouped/multi queries
            bias=config.bias or config.attn_bias,
        )
        # output projection
        self.proj = nn.Linear(config.head_size * config.n_head, config.n_embd, bias=config.bias)
        # KV cache (needed for inference)
        self.kv_cache = kv_cache

        if config.norm_qk:
            norm_q_size = config.n_head * config.head_size if config.norm_qk_type == "olmo2" else config.head_size
            norm_k_size = (
                config.n_query_groups * config.head_size if config.norm_qk_type == "olmo2" else config.head_size
            )
            self.norm_q = config.norm_class(norm_q_size, eps=config.norm_eps)
            self.norm_k = config.norm_class(norm_k_size, eps=config.norm_eps)
        else:
            self.norm_q = self.norm_k = None

        self.config = config
        self.block_idx = block_idx

    def forward(
        self,
        x: torch.Tensor,
        token_idx: torch.Tensor,
        mha: MultiHeadSelfAttention,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            token_idx: Token indexes corresponding to `x`
            mha: Multi-head self-attention code

        Returns:
            Output tensor

        """
        # Notation:
        # - B          | batch size
        # - T          | time-step (sequence length)
        # - C          | model's embeddings size (n_embd)
        # - C*         | attentions's embeddings size
        # - hs         | head size
        # - nh_(q,k,v) | number of heads for query, key and value
        # - n_query_groups = nh_k = nh_v | number of query groups sharing key and value heads
        # alternative notation: num_kv_groups = n_query_groups
        # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
        # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        #   │    │    │    │         │        │                 │
        # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
        # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
        # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
        #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
        # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
        # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
        # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
        # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
        #         MHA                    GQA                   MQA
        #   n_query_groups=4       n_query_groups=2      n_query_groups=1
        #
        # credit https://arxiv.org/pdf/2305.13245.pdf
        head_size = self.config.head_size
        n_head = self.config.n_head
        n_query_groups = self.config.n_query_groups
        rope_n_elem = self.config.rope_n_elem
        batch_size, num, _ = x.size()
        input_pos = None if self.kv_cache is None else self.kv_cache.input_pos

        # Perform a single multiplication operation using a combined QKV matrix to calculate `query`, `key`, and `value`
        # instead of individually multiplying the input `x` with the respective weight matrices.
        qkv = self.qkv(x)

        # Define query, key and value sizes.
        # If grouped/multi query is enabled, these sizes are not equal (see the diagram above).
        query_size = n_head * head_size
        key_size = n_query_groups * head_size
        # Split qkv into query, key and value matrices.
        q, k, v = qkv.split((query_size, key_size, key_size), dim=-1)

        if self.config.norm_qk and self.config.norm_qk_type == "olmo2":
            q = self.norm_q(q)
            k = self.norm_k(k)

        # The original GQA paper is followed here and the term query groups is used.
        # alternative notation: Query groups are also referred to as KV groups.
        q = q.view(batch_size, num, n_head, head_size)
        k = k.view(batch_size, num, n_query_groups, head_size)
        v = v.view(batch_size, num, n_query_groups, head_size)

        # The tensors `query`, `key`, and `value` are now accurately structured: within each batch element (B), there are
        # multiple heads (n_head), and within each head, there is a sequence of elements (T), each represented by a vector
        # of size `hs`.
        # Note that `n_query_groups` can be smaller than `n_head` (but the latter must be a
        # multiple of the former). This works with the
        # `scaled_dot_product_attention` implementations below.
        q = q.transpose(1, 2)  # (batch_size, n_head, num, hs)
        k = k.transpose(1, 2)  # (batch_size, n_query_groups, num, hs)
        v = v.transpose(1, 2)  # (batch_size, n_query_groups, num, hs)

        if self.config.norm_qk and self.config.norm_qk_type == "default":
            q = self.norm_q(q)
            k = self.norm_k(k)

        # Unlike standard positional embeddings rotary embeddings must be applied at every layer.
        if rope_n_elem > 0:
            _input_pos = 0 if input_pos is None else input_pos
            q_roped = mha.pos_encoding(
                q[..., :rope_n_elem],
                input_pos=_input_pos,
                block_idx=self.block_idx,
            )
            k_roped = mha.pos_encoding(
                k[..., :rope_n_elem],
                input_pos=_input_pos,
                block_idx=self.block_idx,
            )
            q = torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1)
            k = torch.cat((k_roped, k[..., rope_n_elem:]), dim=-1)

        # Inner part of multi-head self-attention computation
        if self.kv_cache is None:
            # Default causal self-attention
            y, _ = mha(
                query=q,
                k_and_v=DefaultKeysAndValues(k, v),
                block_idx=self.block_idx,
            )
        else:
            # Defer this to KV cache
            y = self.kv_cache(
                query=q,
                key=k,
                value=v,
                token_idx=token_idx,
            )

        # Output projection
        y = self._transform_output(y, query=q, mha=mha)
        return self.proj(y)  # (batch_size, num, n_embd)

    def _transform_output(
        self,
        y: torch.Tensor,
        query: torch.Tensor,
        mha: MultiHeadSelfAttention,
    ) -> torch.Tensor:
        return y

    def create_default_kv_cache(
        self,
        batch_size: int,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: Optional[int] = None,
    ):
        from keys_values.kvcache.basics import DenseKVCache

        if max_sequence_length is None:
            max_sequence_length = self.config.block_size
        self.kv_cache = DenseKVCache.from_config(
            config=self.config,
            max_batch_size=batch_size,
            cache_length=max_sequence_length,
            block_idx=self.block_idx,
            dtype=dtype,
        )

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args: Any, **kwargs: Any) -> None:
        """For compatibility with legacy checkpoints."""

        for attr in ("weight", "bias"):
            legacy_key = f"{prefix}attn.{attr}"
            current_key = f"{prefix}qkv.{attr}"
            if legacy_key in state_dict:
                state_dict[current_key] = qkv_reassemble(state_dict.pop(legacy_key), self.config)

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
