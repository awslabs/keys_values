from dataclasses import dataclass
import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    n_layers: int
    n_embd: int
    n_heads: int
    head_size: int
    hidden_size: int
    vocab_size: int
    norm_eps: float
    sdpa_type: Literal["naive", "torch"] = "torch"
    rope_base: int = 10000


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
        has_head: bool = True,
        context_width: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.init_embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = nn.ModuleList(
            TransformerLayer(config) for _ in range(config.n_layers)
        )
        self.output_norm = nn.LayerNorm(config.n_embd, eps=config.norm_eps)
        if has_head:
            self.output_head = nn.Linear(config.n_embd, config.vocab_size, bias=True)
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
        x = self.init_embd(input_ids)
        for layer in self.layers:
            x = layer(x, self._rope)
        return self.output_head(self.output_norm(x))


class TransformerLayer(nn.Module):
    """
    Implements transformer layer.

    """
    def __init__(self, config: Config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForwardNetwork(config)
        self.norm_1 = nn.LayerNorm(config.n_embd, eps=config.norm_eps)
        self.norm_2 = nn.LayerNorm(config.n_embd, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, rope: RotaryPositionEncoding) -> torch.Tensor:
        # Use residual link and normalization
        x = x + self.attn(self.norm_1(x), rope)
        return x + self.ffn(self.norm_2(x))


class FeedForwardNetwork(nn.Module):
    """
    Implements feed-forward network block.

    Note: Different LLM families use different network structures
    or activation functions. This is a simple choice here.

    """
    def __init__(self, config: Config):
        super().__init__()
        self.hidden = nn.Linear(config.n_embd, config.hidden_size, bias=True)
        self.activation = nn.GELU()
        self.output = nn.Linear(config.hidden_size, config.n_embd, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.hidden(x)))


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
    q_len = query.shape[-2]
    kv_len = key.shape[-2]
    assert q_len <= kv_len
    scale = 1.0 / math.sqrt(query.shape[-1])
    # Weakness of naive: Intermediate tensors are huge!
    # `(bs, n_heads, q_len, kv_len)`
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
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
    attn_weights = F.softmax(scores + mask, dim=-1)
    return torch.matmul(attn_weights, value)


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
    return F.scaled_dot_product_attention(
        query, key, value, is_causal=True,
    )


class MultiHeadAttention(nn.Module):
    """
    Implements multi-head attention block.

    """
    def __init__(self, config: Config):
        super().__init__()
        # `qkv` combines linear maps for query, key, value into one.
        # Normally, `n_embd == n_heads * head_size`, but different values
        # are permitted as well.
        self.n_heads = config.n_heads
        self.head_size = config.head_size
        attn_n_embd = config.n_heads * config.head_size
        self.qkv = nn.Linear(
            config.n_embd,
            3 * attn_n_embd,
            bias=False,
        )
        # `proj` maps SDPA outputs to outputs of the block
        self.proj = nn.Linear(
            attn_n_embd,
            config.n_embd,
            bias=False,
        )
        self._sdpa = sdpa_torch if config.sdpa_type == "torch" else sdpa_naive

    def forward(self, x: torch.Tensor, rope: RotaryPositionEncoding) -> torch.Tensor:
        bs = x.shape[0]
        q, k, v = self.qkv(x).split(3, dim=-1)
        # Transpose to shape `(bs, n_heads, clen, head_size)`
        shape = (bs, -1, self.n_heads, self.head_size)
        q = q.view(*shape).transpose(1, 2)
        k = k.view(*shape).transpose(1, 2)
        v = v.view(*shape).transpose(1, 2).contiguous()
        # Position encoding
        q = rope(q).contiguous()
        k = rope(k).contiguous()
        # Scaled dot product attention
        # Note: Inputs should all be contiguous
        sdpa_output = self._sdpa(q, k, v)
        # Reverse transpose, final projection
        return self.proj(
            sdpa_output.transpose(1, 2).reshape(bs, -1, self.n_heads * self.head_size)
        )
