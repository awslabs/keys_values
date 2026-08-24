import math
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from litgpt.config import Config


class RotaryPositionEncoding:
    def __init__(self, config: Config, context_width: int):
        self.context_width = context_width
        n_elem = config.head_size
        theta = 1.0 / (config.rope_base ** (torch.arange(0, n_elem, 2).float() / n_elem))
        seq_idx = torch.arange(context_width).float()
        idx_theta = (seq_idx[:, None] * theta[None, :]).repeat(1, 2)
        self._cos = torch.cos(idx_theta)
        self._sin = torch.sin(idx_theta)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        head_size_half = x.size(-1) // 2
        x1 = x[..., :head_size_half]
        x2 = x[..., head_size_half:]
        rotated = torch.cat((-x2, x1), dim=-1)  # (B, ..., T, head_size)
        cos = self._cos.to(device=x.device)
        sin = self._sin.to(device=x.device)
        dims_diff = x.ndim - self._cos.ndim
        if dims_diff > 0:
            # Ensure that shapes of `x`, `cos`, `sin` align
            new_shape = cos.shape[0:1] + (1,) * dims_diff + cos.shape[1:]
            cos = cos.view(*new_shape)
            sin = sin.view(*new_shape)
        roped = (x * cos) + (rotated * sin)
        return roped.to(dtype=x.dtype)


class Transformer(nn.Module):
    """
    Implements complete transformer model.

    """
    def __init__(
        self,
        config: Config,
        sdpa_type: Literal["naive", "torch"] = "torch",
        has_head: bool = True,
        context_width: Optional[int] = None,
    ):
        super().__init__()
        self._check_supported(config)
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(
                    TransformerLayer(config, sdpa_type) for _ in range(config.n_layer)
                ),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        if has_head:
            self.lm_head = nn.Linear(
                config.n_embd,
                config.padded_vocab_size,
                bias=config.lm_head_bias,
            )
        else:
            self.output_head = nn.Identity()
        self._rope = None
        if context_width is not None:
            self.set_context_width(context_width)

    @property
    def context_width(self) -> Optional[int]:
        return None if self._rope is None else self._rope.context_width

    def set_context_width(self, context_width: int):
        self._rope = RotaryPositionEncoding(self.config, context_width)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Batch of input token sequences, shape
                `(batch_size, seq_length)`

        Returns:
            Logits, shape `(batch_size, seq_length, config.vocab_size)`, if
            `has_head == True`, final layer outputs, shape
            `(batch_size, seq_length, config.n_embd)` otherwise.

        """
        if self._rope is None:
            raise ValueError("Context width not set. Use `set_context_width`")
        x = self.transformer.wte(input_ids)
        if self.config.scale_embeddings:
            x = x * (self.config.n_embd ** 0.5)
        for layer in self.transformer.h:
            x = layer(x, self._rope)
        return self.lm_head(self.transformer.ln_f(x))

    @staticmethod
    def _check_supported(config: Config):
        for name in (
            "final_logit_softcapping",
        ):
            if getattr(config, name) is not None:
                raise ValueError(f"config.{name} is not supported")
        for name, val in (
            ("parallel_residual", False),
            ("shared_attention_norm", False),
            ("mlp_class_name", "LLaMAMLP"),
            ("rotary_percentage", 1.0),
        ):
            if getattr(config, name) != val:
                raise ValueError(f"config.{name} == {getattr(config, name)} is not supported (must be {val})")


class TransformerLayer(nn.Module):
    """
    Implements transformer layer.

    """
    def __init__(self, config: Config, sdpa_type: str):
        super().__init__()
        self.attn = MultiHeadAttention(config, sdpa_type)
        self.mlp = FeedForwardNetwork(config)
        self.norm_1 = self._create_norm(config, config.norm_1)
        self.norm_2 = self._create_norm(config, config.norm_2)
        self.post_attention_norm = self._create_norm(
            config, config.post_attention_norm,
        )
        self.post_mlp_norm = self._create_norm(config, config.post_mlp_norm)

    @staticmethod
    def _create_norm(config: Config, do_norm: bool) -> nn.Module:
        return (
            nn.Identity()
            if not do_norm
            else config.norm_class(config.n_embd, eps=config.norm_eps)
        )

    def forward(self, x: torch.Tensor, rope: RotaryPositionEncoding) -> torch.Tensor:
        # Use residual link and normalization
        attn_output = self.post_attention_norm(self.attn(self.norm_1(x), rope))
        x = x + attn_output
        return x + self.post_mlp_norm(self.mlp(self.norm_2(x)))


class FeedForwardNetwork(nn.Module):
    """
    Implements feed-forward network block.

    Taken from `litgpt.model.LLaMAMLP`. This is the form introduced by
    LLaMA, which is also used by Qwen3 models.

    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        intermediate_size = config.intermediate_size
        self.fc_1 = nn.Linear(config.n_embd, intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, intermediate_size, bias=config.bias)
        self.proj = nn.Linear(intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fc_1(x)
        x2 = self.fc_2(x)
        x = F.silu(x1) * x2
        return self.proj(x)


def sdpa_naive(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """
    Scaled dot product attention: Naive implementation.

    Attention: This needs a huge amount of memory, since scores and
    attention weights are fully materialized. Use `sdpa_torch` instead.

    """
    bs, nh_q, q_len, head_size = query.shape
    _, nh_k, kv_len, _ = key.shape
    q_per_kv = nh_q // nh_k
    assert q_len <= kv_len
    assert q_per_kv >= 1 and nh_q == nh_k * q_per_kv
    scale_factor = 1.0 / math.sqrt(head_size)
    # Compute inner products in `scores`
    if scale_factor == 1.0:
        arg1 = query
        arg2 = key.mT
    elif query.numel() <= key.numel():
        arg1 = query * scale_factor
        arg2 = key.mT
    else:
        arg1 = query
        arg2 = key.mT * scale_factor
    if q_per_kv == 1:
        scores = torch.matmul(arg1, arg2)
    else:
        # Grouped query attention (GQA): Using broadcasting with `matmul`
        q_shape = (bs, nh_k, q_per_kv) + query.shape[2:]
        arg1 = arg1.view(*q_shape)
        arg2 = arg2.unsqueeze(2)
        # At this point:
        # - arg1: (bs, nh_k, q_per_kv, q_len, head_size)
        # - arg2: (bs, nh_k, 1, head_size, kv_len)
        # - scores: (bs, nh_k, q_per_kv, q_len, kv_len)
        scores = torch.matmul(arg1, arg2)
        s_shape = query.shape[:-1] + (kv_len,)
        scores = scores.view(*s_shape)  # (bs, nh_q, q_len, kv_len)

    # Causal masking:
    # Q pos `q_pos` can attend to KV pos `k_pos` only if `q_pos >= k_pos`.
    # Adding `mask` to `score` ensures that
    # `scores[:, :, q_pos, k_pos] == -infty` if `q_pos < k_pos`.
    # For training, `q_len == kv_len` and `offset == 0`, but for inference,
    # `query` is right-aligned with `key`.
    mask = torch.zeros_like(scores[0, 0, :, :])
    offset = kv_len - q_len
    kwargs = dict(device=query.device)
    mask[
        torch.arange(offset, kv_len, **kwargs)
        < torch.arange(kv_len, **kwargs)
    ] = float("-inf")
    # Softmax to compute attention weights
    scores = F.softmax(scores + mask, dim=-1)
    if q_per_kv == 1:
        return torch.matmul(scores, value)
    else:
        # Grouped query attention (GQA): Using broadcasting with `matmul`
        s_shape = (bs, nh_k, q_per_kv) + scores.shape[2:]
        _scores = scores.view(*s_shape)
        _value = value.unsqueeze(2)
        # At this point:
        # - _scores: (bs, nh_k, q_per_kv, q_len, kv_len)
        # - _value: (bs, nh_k, 1, kv_len, head_size)
        # - result: (bs, nh_k, q_per_kv, q_len, head_size)
        result = torch.matmul(_scores, _value)
        r_shape = scores.shape[:-1] + (head_size,)
        return result.view(*r_shape)


def sdpa_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    if query.shape[-2] != key.shape[-2]:
        raise NotImplementedError(
            "Only supports training case (q_len == kv_len). For inference, "
            "we'd need query to be right-aligned with key, value."
        )
    n_head = query.shape[1]
    n_query_groups = key.shape[1]
    enable_gqa = n_query_groups < n_head
    if enable_gqa:
        # Some efficient kernels have not implemented `enabla_gqa=True`. It is
        # better to extend keys, values in this case.
        q_per_kv = n_head // n_query_groups
        key = torch.repeat_interleave(key, q_per_kv, dim=1)
        value = torch.repeat_interleave(value, q_per_kv, dim=1)
        enable_gqa = key.shape[1] == n_query_groups

    return F.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        is_causal=True,
        enable_gqa=enable_gqa,
    )


class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention block.

    """
    def __init__(
        self,
        config: Config,
        sdpa_type: str,
    ):
        super().__init__()
        # `qkv` combines linear maps for query, key, value into one.
        # With GQA, query scales as `n_head`, while key, value scale as
        # `n_query_groups`.
        self.n_head = config.n_head
        self.n_query_groups = config.n_query_groups
        self.head_size = config.head_size
        attn_n_embd = config.n_head * config.head_size
        qkv_out_size = (config.n_head + 2 * config.n_query_groups) * config.head_size
        self.qkv = nn.Linear(
            config.n_embd,
            qkv_out_size,
            bias=config.bias or config.attn_bias,
        )
        # `proj` maps SDPA outputs to outputs of the block
        self.proj = nn.Linear(
            config.n_head * config.head_size,
            config.n_embd,
            bias=False,
        )
        self._sdpa = sdpa_torch if sdpa_type == "torch" else sdpa_naive
        if config.norm_qk:
            self.norm_q = config.norm_class(config.head_size, eps=config.norm_eps)
            self.norm_k = config.norm_class(config.head_size, eps=config.norm_eps)
        else:
            self.norm_q = self.norm_k = None

    def forward(self, x: torch.Tensor, rope: RotaryPositionEncoding) -> torch.Tensor:
        bs = x.shape[0]
        q_size = self.n_head * self.head_size
        k_size = self.n_query_groups * self.head_size
        q, k, v = self.qkv(x).split((q_size, k_size, k_size), dim=-1)
        # Transpose to shape used in SDPA
        q_shape = (bs, -1, self.n_head, self.head_size)
        kv_shape = (bs, -1, self.n_query_groups, self.head_size)
        q = q.view(*q_shape).transpose(1, 2)
        k = k.view(*kv_shape).transpose(1, 2)
        v = v.view(*kv_shape).transpose(1, 2)
        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)
        # Position encoding
        # Also, inputs into SDPA must be contiguous
        q = rope(q).contiguous()
        k = rope(k).contiguous()
        v = v.contiguous()
        # Scaled dot product attention
        # Note: Inputs should all be contiguous
        sdpa_output = self._sdpa(q, k, v)
        # Reverse transpose, final projection
        return self.proj(
            sdpa_output.transpose(1, 2).reshape(bs, -1, q_size)
        )
