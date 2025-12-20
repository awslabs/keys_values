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
from itertools import product
import math
import random
from typing import Optional, Tuple

import pytest
import torch
from torch.nn import functional as F

from litgpt.config import Config
from litgpt.model import (
    apply_rope,
    build_rope_cache,
    batched_index_select,
)

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.attention import (
    DefaultKeysAndValues,
    MultiHeadSelfAttention,
    do_softcapping,
    eager_scaled_dot_product_attention,
)
from keys_values.attention_utils import (
    build_mask_cache,
    build_mask_slice,
    sample_token_positions,
    ENTRIES_PER_GB,
)
from keys_values.kvcache.base import KVCache
from keys_values.kvcache.test_utils import (
    product_with_devices,
    available_backends,
)
from keys_values.model import GPT, CausalSelfAttention
from keys_values.pos_encoding import YaRNPositionEncoding
from keys_values.use_eager_kernel import transform_mha_kwargs
from keys_values.utils import repeat_interleave


@pytest.mark.parametrize(
    *product_with_devices(
        [
            (2, 1),
            (4, 1),
            (8, 4),
            (12, 4),
            (24, 8),
            (9, 3),
        ],
        "n_head, n_query_groups",
    ),
)
@torch.inference_mode()
def test_scaled_dot_product_attention(device, n_head, n_query_groups):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    num_repeats = 32
    dtype = torch.bfloat16
    mask_kwargs = dict(dtype=dtype, device=device)
    assert_kwargs = dict(atol=0.0005, rtol=0.05)
    sliding_window_size = None

    for repeat in range(num_repeats):
        head_size = 2 ** random.randint(3, 6)
        batch_size = random.randint(1, 5)
        len_key = random.randint(16, 128)
        mask = None
        if repeat % 2 == 0:
            len_query = len_key
            mask = build_mask_cache(
                max_seq_length=len_key,
                sliding_window_size=sliding_window_size,
                **mask_kwargs,
            )
            input_pos = 0
            token_positions = None
        elif repeat % 4 == 1:
            len_query = random.randint(1, len_key // 2)
            token_positions = torch.arange(
                0,
                len_key,
                dtype=torch.int64,
                device=device,
            ).view(1, 1, -1).expand(batch_size, n_query_groups, -1)
            input_pos = len_key // 2
            mask = build_mask_slice(
                input_pos=len_key - len_query,
                num=len_query,
                token_positions=token_positions,
                n_head=n_head,
                dtype=dtype,
            )
        else:
            len_query = 1
            input_pos = len_key // 2
            token_positions = None
        shape = (batch_size, n_head, len_query, head_size)
        query = torch.randn(shape, **mask_kwargs)
        shape = (batch_size, n_query_groups, len_key, head_size)
        key = torch.randn(shape, **mask_kwargs)
        value = torch.randn(shape, **mask_kwargs)
        k_and_v = DefaultKeysAndValues(key, value)
        scale = 1.0 / math.sqrt(head_size)

        result, attn_weights = eager_scaled_dot_product_attention(
            query,
            k_and_v,
            scale_factor=scale,
            use_blocking=False,
            return_attn_weights=True,
            input_pos=input_pos,
            token_positions=token_positions,
            sliding_window_size=sliding_window_size,
            mask=mask,
        )
        key_bc = repeat_interleave(key, n_head)
        value_bc = repeat_interleave(value, n_head)
        k_and_v_bc = DefaultKeysAndValues(key_bc, value_bc)
        result_cmp, attn_weights_cmp = eager_scaled_dot_product_attention(
            query,
            k_and_v_bc,
            scale_factor=scale,
            use_blocking=False,
            return_attn_weights=True,
            input_pos=input_pos,
            token_positions=token_positions,
            sliding_window_size=sliding_window_size,
            mask=mask,
        )
        attn_weights_cmp = attn_weights_cmp.view(
            batch_size, n_query_groups, -1, len_key,
        ).mean(dim=2)
        msg = (
            f"bs={batch_size}, hs={head_size}, nh_q={n_head}, nh_k={n_query_groups}, len_q={len_query}, len_k={len_key}"
        )
        torch.testing.assert_close(result, result_cmp, **assert_kwargs), msg
        torch.testing.assert_close(attn_weights, attn_weights_cmp, **assert_kwargs), msg


@pytest.mark.parametrize(
    *product_with_devices(
        [
            (None, 1, 1),
            (None, 4, 16),
            (4, 1, 1),
            (4, 2, 32),
            (128, 1, 1),
            (128, 4, 16),
        ],
        "sliding_window_size, batch_size, n_query_groups",
    ),
)
@torch.inference_mode()
def test_build_mask_slice(
    device: torch.device,
    sliding_window_size: Optional[int],
    batch_size: int,
    n_query_groups: int,
):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    num_repeats = 30
    dtype = torch.bfloat16

    for _ in range(num_repeats):
        seq_len = random.randint(16, 256)
        full_mask = build_mask_cache(seq_len, sliding_window_size, device, dtype)
        input_pos = random.randint(1, seq_len - 1)
        num = random.randint(1, min(16, seq_len - input_pos))
        cache_length = random.randint(8, seq_len - 4)
        token_positions = torch.zeros(
            (batch_size, n_query_groups, cache_length),
            dtype=torch.int64,
            device=device,
        )
        for bs in range(batch_size):
            for nq in range(n_query_groups):
                token_positions[bs, nq, :] = torch.randperm(
                    seq_len,
                    device=device,
                )[:cache_length]
        mask = build_mask_slice(
            input_pos=input_pos,
            num=num,
            token_positions=token_positions,
            n_head=n_query_groups,
            dtype=dtype,
            sliding_window_size=sliding_window_size,
        )
        mask_cmp = batched_index_select(
            full_mask[input_pos : (input_pos + num), :],
            dim=1,
            idx=token_positions,
        )
        torch.testing.assert_close(mask, mask_cmp)


@pytest.mark.parametrize(
    "device, dtype",
    product(
        available_backends(),
        [torch.float32, torch.float16, torch.bfloat16],
    ),
)
def test_mask_sliding_window(device, dtype):
    """
    Compares `mask` used in MHA in training mode in old code (using
    `mask_cache`) and new code, using a setup from
    :func:`test_against_original_gemma_2` above.

    """
    T = 20
    model_name = "gemma-2-27b"
    config = Config.from_name(
        model_name,
        block_size=T,
        sliding_window_size=T // 2,
        n_layer=2,
        n_head=16,
        n_embd=32,
        intermediate_size=86,
        rotary_percentage=1.0,
    )
    # Determine mask used in forward call for length `T` input (old code)
    # neg_infty = float("-inf")
    neg_infty = torch.finfo(dtype).min
    old_mask = torch.ones(T, T, dtype=dtype, device=device).triu(diagonal=1)
    old_mask.masked_fill_(old_mask.bool(), neg_infty)
    old_mask = old_mask.view(1, 1, *old_mask.shape)
    sliding_window_bias = torch.ones_like(old_mask).tril(diagonal=-config.sliding_window_size)
    sliding_window_bias.masked_fill_(sliding_window_bias.bool(), neg_infty)
    old_mask += sliding_window_bias
    # Determine mask as in new code
    new_mask = build_mask_cache(
        max_seq_length=T,
        sliding_window_size=config.sliding_window_size,
        dtype=dtype,
        device=device,
    ).view(1, 1, T, T)
    torch.testing.assert_close(old_mask, new_mask)


# Old code before `attention.py` was factored out
class CausalSelfAttention_OLD(torch.nn.Module):
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        # key, query and value projections for all heads, but in a batch
        self.qkv = torch.nn.Linear(
            config.n_embd,
            (config.n_head + 2 * config.n_query_groups) * config.head_size,  # support for grouped/multi queries
            bias=config.bias or config.attn_bias,
        )
        # output projection
        self.proj = torch.nn.Linear(config.head_size * config.n_head, config.n_embd, bias=config.bias)
        # disabled by default
        self.kv_cache: Optional[KVCache] = None
        self.apply_sliding_window_attention = False
        if config.sliding_window_size is not None and config.sliding_window_indices is not None:
            self.apply_sliding_window_attention = config.sliding_window_indices[block_idx]

        if config.norm_qk:
            self.norm_q = config.norm_class(config.head_size, eps=config.norm_eps)
            self.norm_k = config.norm_class(config.head_size, eps=config.norm_eps)
        else:
            self.norm_q = self.norm_k = None

        self.config = config
        self.block_idx = block_idx

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_pos_maxp1: Optional[int] = None,
    ) -> torch.Tensor:
        # Notation:
        # - B          | batch size
        # - T          | time-step (sequence length)
        # - C          | model's embeddings size (n_embd)
        # - C*         | attentions's embeddings size
        # - nh_(q,k,v) | number of heads for query, key and value
        # - hs         | head size
        head_size = self.config.head_size
        n_head = self.config.n_head
        n_query_groups = self.config.n_query_groups
        rope_n_elem = self.config.rope_n_elem
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # Perform a single multiplication operation using a combined QKV matrix to calculate `query`, `key`, and `value`
        # instead of individually multiplying the input `x` with the respective weight matrices.
        qkv = self.qkv(x)  # (B, T, 3xC*)

        # Define query, key and value sizes.
        # If grouped/multi query is enabled, these sizes are not equal (see the diagram in `lit_gpt/config.py::Config`).
        query_size = n_head * head_size
        key_size = value_size = n_query_groups * head_size
        # Split qkv into query, key and value matrices.
        q, k, v = qkv.split((query_size, key_size, value_size), dim=-1)  # 3x(B, T, C*)

        # To place the num_heads (nh) dimension right after the batch (B) dimension, the first step is to decouple the
        # embedding size (C) into num_heads (nh) and head_size (hs).
        q = q.view(B, T, n_head, head_size)  # (B, T, nh_q, hs)
        k = k.view(B, T, n_query_groups, head_size)  # (B, T, nh_k, hs)
        v = v.view(B, T, n_query_groups, head_size)  # (B, T, nh_v, hs)

        # The tensors `query`, `key`, and `value` are now accurately structured: within each batch element (B), there are
        # multiple heads (nh), and within each head, there is a sequence of elements (T), each represented by a vector
        # of size `hs`.
        q = q.transpose(1, 2)  # (B, nh_q, T, hs)
        k = k.transpose(1, 2)  # (B, nh_k, T, hs)
        v = v.transpose(1, 2)  # (B, nh_v, T, hs)

        if self.config.norm_qk:
            q = self.norm_q(q)
            k = self.norm_k(k)

        # Unlike standard positional embeddings rotary embeddings must be applied at every layer.
        q_roped = apply_rope(q[..., :rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., :rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1)  # (B, nh_q, T, hs)
        k = torch.cat((k_roped, k[..., rope_n_elem:]), dim=-1)  # (B, nh_k, T, hs)

        # Apply kv-cache during inference.
        if input_pos is not None:
            if not isinstance(self.kv_cache, KVCache):
                raise TypeError("You need to call `gpt.set_kv_caches()`")
            k, v = self.kv_cache(input_pos, k, v)
            if input_pos_maxp1 is not None:
                # Subselect along sequence dimension
                k = k[..., :input_pos_maxp1, :]
                v = v[..., :input_pos_maxp1, :]
            # k, v: (B, nh_k, input_pos_maxp1, hs)
            # If input_pos_maxp1 is None -> max_seq_length

        # Grouped queries: balance the number of heads across all three matrices.
        # NOTE: flash attention requires it in training mode.
        # Multi-query: this step can be skipped since there is only 1 head, allowing us to use broadcasting.
        if n_query_groups != n_head and (input_pos is None or n_query_groups != 1):
            q_per_kv = n_head // n_query_groups
            k = k.repeat_interleave(q_per_kv, dim=1)  # (B, nh_q, T, hs)
            v = v.repeat_interleave(q_per_kv, dim=1)  # (B, nh_q, T, hs)

        if self.apply_sliding_window_attention:
            """
                  Global Window              Sliding window             Sliding window
                  attention mask      +            bias          =      attention mask
            ┌────────────────────────┐  ┌───────────────────────┐  ┌─────────────────────────┐
            │ True False False False │  │ True  True  True True │  │ True  False False False │
            │ True True  False False │  │ True  True  True True │  │ True  True  False False │
            │ True True  True  False │  │ False True  True True │  │ False True  True  False │
            │ True True  True  True  │  │ False False True True │  │ False False True  True  │
            └────────────────────────┘  └───────────────────────┘  └─────────────────────────┘
            """
            minus_infty = torch.finfo(q.dtype).min
            # minus_infty = float("-inf")
            if mask is None:
                mask = torch.ones(T, T, dtype=q.dtype, device=q.device).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), minus_infty)
                mask = mask.view(1, 1, *mask.shape)
            sliding_window_bias = torch.ones_like(mask).tril(diagonal=-self.config.sliding_window_size)
            sliding_window_bias.masked_fill_(sliding_window_bias.bool(), minus_infty)
            mask += sliding_window_bias

        # Efficient attention using Flash Attention CUDA kernels.
        # NOTE: efficient implementation is disabled if `mask` is not None or softcapping is enabled.
        # ↓ (B, nh, T, hs) @ (B, nh, T, hs).mT --> (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)
        y = self.scaled_dot_product_attention(q, k, v, mask)

        # Re-assemble all head outputs side by side.
        y = y.reshape(B, T, head_size * n_head)

        # Output projection.
        return self.proj(y)  # (B, T, C)

    # Note: All internal computations are done in `float32`. This is also done
    # in `F.scaled_dot_product_attention`.
    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(self.config.attention_scores_scalar or self.config.head_size)

        # with softcapping we cannot use SDPA
        if self.config.attention_logit_softcapping is not None:
            dtype = torch.float32
            scores = q.to(dtype) @ k.mT.to(dtype) * scale
            scores = do_softcapping(scores, self.config.attention_logit_softcapping)
            if mask is None:
                q_len = q.shape[2]
                mask = torch.ones(
                    q_len,
                    q_len,
                    dtype=dtype,
                    device=q.device,
                ).triu(diagonal=1)
                mask.masked_fill_(mask.bool(), torch.finfo(dtype).min)
                mask = mask.view(1, 1, *mask.shape)
            scores = scores + mask
            scores = F.softmax(scores, dim=-1)
            y = (scores @ v.to(dtype)).to(q.dtype)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
            )
        return y.transpose(1, 2)


def rope_cache_OLD(
    config: Config,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if config.rope_adjustments is None:
        extra_config = None

    else:
        adjusted_params_required = ["factor", "low_freq_factor", "high_freq_factor", "original_max_seq_len"]
        params_present = [param in config.rope_adjustments for param in adjusted_params_required]
        num_params_present = sum(params_present)

        if num_params_present == 0:
            extra_config = None  # uses standard RoPE
        elif num_params_present == 4:
            # These parameters should always be used together so that we don't interfere with standard rope
            extra_config = {name: config.rope_adjustments[name] for name in adjusted_params_required}
        elif "factor" in config.rope_adjustments:
            # linear RoPE
            adjusted_params_required = ["factor"]
            extra_config = {name: config.rope_adjustments[name] for name in adjusted_params_required}
        else:
            # Some but not all parameters are specified; raise an error
            missing_params = [param for param, present in zip(adjusted_params_required, params_present) if not present]
            raise ValueError(
                f"The following adjusted RoPE parameters are missing in rope_adjustments: {', '.join(missing_params)}. "
                "All adjusted RoPE parameters must be specified together."
            )

    return build_rope_cache(
        seq_len=config.block_size,
        n_elem=config.rope_n_elem,
        device=device,
        condense_ratio=config.rope_condense_ratio,
        base=config.rope_base,
        extra_config=extra_config,
        rope_local_base_freq=config.rope_local_base_freq,
    )


@pytest.mark.parametrize(
    "device, model_name, dtype",
    product(
        available_backends(),
        ["gemma-2-27b", "gemma-3-27b-it"],
        [torch.float32, torch.float16, torch.bfloat16],
    ),
)
def test_multi_head_attention_for_gemma(device, model_name, dtype):
    """
    Compares multi-head attention in old and current code, using a
    setup from :func:`test_against_original_gemma_2` above.

    """
    num_repeats = 20
    T = 20
    batch_size = 4
    is_gemma_3 = model_name.startswith("gemma-3")
    config = Config.from_name(
        model_name,
        block_size=T,
        sliding_window_size=T // 2,
        n_layer=2,
        n_query_groups=16,
        n_head=16,
        n_embd=32,
        intermediate_size=86,
        rotary_percentage=1.0,
        rope_indices=[0, 1] if is_gemma_3 else None,
    )
    kwargs = dict(dtype=dtype, device=device)

    # Obtain RoPE parameters and compare
    model_new = GPT(config).to(**kwargs)
    model_new.max_seq_length = T
    mha = model_new.mha
    pos_encoding = mha.pos_encoding
    assert isinstance(pos_encoding, YaRNPositionEncoding)
    cos_new = pos_encoding._cos.unsqueeze(0)
    sin_new = pos_encoding._sin.unsqueeze(0)
    cos_old, sin_old = rope_cache_OLD(config)
    cos_old = cos_old.unsqueeze(0).to(**kwargs)
    sin_old = sin_old.unsqueeze(0).to(**kwargs)
    torch.testing.assert_close(cos_new, cos_old)
    torch.testing.assert_close(sin_new, sin_old)

    shape = (batch_size, T, config.n_embd)
    for rep in range(num_repeats):
        block_idx = rep % 2
        attn_new = CausalSelfAttention(
            config,
            block_idx=block_idx,
        ).to(**kwargs)
        attn_old = CausalSelfAttention_OLD(
            config,
            block_idx=block_idx,
        ).to(**kwargs)
        # Ensure they have the same weights
        attn_old.load_state_dict(attn_new.state_dict())
        inputs = torch.randn(shape, **kwargs)
        token_idx = torch.randint(
            0,
            config.padded_vocab_size,
            (batch_size, T),
            dtype=torch.int64,
            device=device,
        )
        outputs_new = attn_new(
            x=inputs,
            token_idx=token_idx,
            mha=mha,
        )
        if is_gemma_3:
            _cos = cos_old[..., config.rope_indices[block_idx]]
            _sin = sin_old[..., config.rope_indices[block_idx]]
        else:
            _cos = cos_old
            _sin = sin_old
        outputs_old = attn_old(
            x=inputs,
            cos=_cos,
            sin=_sin,
            mask=None,
        )
        torch.testing.assert_close(outputs_new, outputs_old)


def _get_token_positions(
    start: int,
    end: int,
    batch_size: int,
    n_query_groups: int,
    device: torch.device,
) -> torch.Tensor:
    return (
        torch.arange(start, end, dtype=torch.int64, device=device)
        .view(
            1,
            1,
            -1,
        )
        .expand(batch_size, n_query_groups, -1)
    )


@pytest.mark.parametrize(
    *product_with_devices(
        [
            (128, None),
            (21, None),
            (128, 16),
            (21, 12),
        ],
        "seq_len, sliding_window_size",
    ),
)
def test_build_mask(device, seq_len, sliding_window_size):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    num_repeats = 4
    batch_size = 2
    n_query_groups = 4
    tp_kwargs = dict(
        batch_size=batch_size,
        n_query_groups=n_query_groups,
        device=device,
    )

    mask_full = build_mask_cache(
        max_seq_length=seq_len,
        sliding_window_size=sliding_window_size,
        device=device,
        dtype=torch.float32,
    )[None, None, :, :].expand(batch_size, n_query_groups, -1, -1)
    token_positions = _get_token_positions(0, seq_len, **tp_kwargs)
    for _ in range(num_repeats):
        mask_parts = []
        num_prefill = random.randint(1, seq_len - 1)
        mask_parts.append(
            build_mask_slice(
                input_pos=0,
                num=num_prefill,
                token_positions=token_positions,
                n_head=n_query_groups,
                dtype=torch.float32,
                sliding_window_size=sliding_window_size,
            )
        )
        for pos in range(num_prefill, seq_len):
            mask_parts.append(
                build_mask_slice(
                    input_pos=pos,
                    num=1,
                    token_positions=token_positions,
                    n_head=n_query_groups,
                    dtype=torch.float32,
                    sliding_window_size=sliding_window_size,
                )
            )
        mask_comp = torch.cat(mask_parts, dim=2)
        torch.testing.assert_close(mask_full, mask_comp)


@pytest.mark.parametrize(
    *product_with_devices(
        [
            (4, 2, 128, 512, torch.float16, None),
            (4, 4, 8, 256, torch.bfloat16, None),
            (8, 4, 128, 128, torch.float16, None),
            (12, 4, 16, 512, torch.bfloat16, None),
            (24, 8, 2, 512, torch.float16, None),
            (9, 3, 128, 512, torch.bfloat16, None),
            (12, 4, 16, 512, torch.float16, 12),
            (24, 8, 2, 512, torch.bfloat16, 64),
            (9, 3, 128, 512, torch.float16, 96),
        ],
        "n_head, n_query_groups, q_len, kv_len, dtype, sliding_window_size",
    ),
)
def test_attention_in_blocks(device, n_head, n_query_groups, q_len, kv_len, dtype, sliding_window_size):
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    num_repeats = 32
    seq_len = 2 * kv_len
    is_causal = q_len == kv_len
    input_pos = seq_len - q_len if not is_causal else 0
    config = Config.from_name(
        "gemma-2-27b",
        block_size=3 * kv_len,
        sliding_window_size=sliding_window_size,
        n_layer=1,
        n_query_groups=n_query_groups,
        n_head=n_head,
        n_embd=n_head * 32,
        intermediate_size=n_head * 32 * 3,
        rotary_percentage=1.0,
    )

    print(f"n_head={n_head}, n_query_groups={n_query_groups}, q_len={q_len}, kv_len={kv_len}, is_causal={is_causal}, dtype={dtype}, device={device}")
    kwargs = dict(dtype=dtype, device=device)
    for repeat in range(num_repeats):
        head_size = 2 ** random.randint(3, 6)
        batch_size = random.randint(1, 5)
        if q_len % 2 != 0 and batch_size % 2 != 0:
            batch_size += 1
        print(f"repeat={repeat}, head_size={head_size}, batch_size={batch_size}")
        if is_causal:
            token_positions = None
        else:
            token_positions = sample_token_positions(
                batch_size,
                n_query_groups,
                q_len,
                kv_len,
                input_pos,
                device=device,
            )
        shape = (batch_size, n_head, q_len, head_size)
        query = torch.randn(shape, **kwargs)
        shape = (batch_size, n_query_groups, kv_len, head_size)
        key = torch.randn(shape, **kwargs)
        value = torch.randn(shape, **kwargs)
        k_and_v = DefaultKeysAndValues(key, value)
        print(f"query {query.shape}, key {key.shape}, value {value.shape}")
        if token_positions is not None:
            print(f"token_positions {token_positions.shape}")
        results = {"outputs": [], "weights": []}
        numel_tmp = batch_size * n_head * q_len * kv_len
        # Different MHA objects w/o a tight entry limit
        mha = {
            "yes": MultiHeadSelfAttention(
                config,
                use_eager_sdpa_always=True,
                tmp_array_limit_gb=TemporaryArrayLimit(
                    init_val=numel_tmp / 3 / ENTRIES_PER_GB,
                    name="bogus",
                ),
            ),
            "no": MultiHeadSelfAttention(
                config,
                use_eager_sdpa_always=True,
                tmp_array_limit_gb=TemporaryArrayLimit(
                    init_val=2 * numel_tmp / ENTRIES_PER_GB,
                    name="bogus",
                )
            ),
        }
        for kind in ("no", "yes"):
            attn_outputs, attn_weights = mha[kind](
                query=query,
                k_and_v=k_and_v,
                block_idx=0,
                input_pos=input_pos,
                return_attn_weights=True,
                token_positions=token_positions,
            )
            results["outputs"].append(attn_outputs)
            results["weights"].append(attn_weights)
        # Compare
        for name, result in results.items():
            print(f"Compare gradients for {name}")
            torch.testing.assert_close(
                result[0], result[1], atol=0.0005, rtol=0.05,
            )
