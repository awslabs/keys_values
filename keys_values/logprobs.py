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
"""
Memory-efficient per-token log-probability computation using KV cache.

This module provides :func:`chunked_per_token_logps`, which computes
per-token log-probabilities using KeysAndValues' chunked forward pass
with a KV cache policy. Instead of materializing the full
``(batch, seq_length, vocab_size)`` logits tensor, it processes the
sequence in chunks and extracts log-probs incrementally.

Memory usage is bounded by ``O(batch * chunk_size * vocab_size)``
regardless of total sequence length.

For TRL integration, see :class:`keys_values.finetune.grpo.GRPOLongContextTrainer`.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from keys_values.kvcache.factory import (
    KVCacheFactory,
    deallocate_kv_cache_buffers_of_model,
)
from keys_values.kvcache.stack_layers import DefaultCellBlocks
from keys_values.long_context import (
    create_chunk_sizes,
    get_chunks_for_cells,
    write_back_cache_buffers,
)
from keys_values.model import GPT


def chunked_per_token_logps(
    gpt_model: GPT,
    input_ids: torch.Tensor,
    logits_to_keep: int,
    cache_name: str = "h2o-torch-quantized8",
    cache_length: int = 16384,
    chunk_size: int = 1024,
    cache_kwargs: Optional[dict] = None,
    temperature: float = 1.0,
    compute_entropy: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Compute per-token log-probs using chunked KV-cache forward pass.

    This is the KeysAndValues alternative to TRL's full-sequence forward.
    It processes ``input_ids`` chunk by chunk through the model with a KV
    cache active, only materializing logits for one chunk at a time. For
    positions corresponding to completion tokens (the last ``logits_to_keep``
    positions), it extracts log-probs on the fly.

    Memory usage is bounded by ``O(batch * chunk_size * vocab_size)`` for
    the logits tensor, regardless of total sequence length.

    Args:
        gpt_model: A KeysAndValues GPT model (with or without KV caches
            already assigned). If caches are not assigned, they are created
            and assigned temporarily.
        input_ids: Full input token IDs (prompt + completion),
            shape ``(batch_size, seq_length)``.
        logits_to_keep: Number of completion tokens at the end of the
            sequence for which log-probs are needed.
        cache_name: KV cache policy name (e.g. "h2o-torch-quantized8",
            "lastrec-default", "dense-default").
        cache_length: Number of slots in the KV cache.
        chunk_size: Size of each processing chunk after the prefill.
        cache_kwargs: Additional arguments for KV cache construction.
        temperature: Temperature for log-prob computation. Values > 1 make
            the distribution more uniform. Default 1.0 (no scaling).
        compute_entropy: If True, also compute Shannon entropy at each
            completion position.

    Returns:
        Tuple of (log_probs, entropies):
            - log_probs: shape ``(batch_size, logits_to_keep)``
            - entropies: shape ``(batch_size, logits_to_keep)`` or None

    Example::

        from keys_values.logprobs import chunked_per_token_logps
        from keys_values.model import GPT

        model = GPT(config)
        model.load_state_dict(...)

        # input_ids is (batch, prompt_len + completion_len)
        logps, ent = chunked_per_token_logps(
            gpt_model=model,
            input_ids=input_ids,
            logits_to_keep=completion_len,
            cache_name="h2o-torch-quantized8",
            cache_length=16384,
            chunk_size=1024,
            compute_entropy=True,
        )
    """
    batch_size, seq_length = input_ids.shape
    device, config = input_ids.device, gpt_model.config
    dtype = next(gpt_model.parameters()).dtype
    completion_start = seq_length - logits_to_keep

    caches_created = _ensure_kv_caches(
        gpt_model, cache_name, batch_size, cache_length, dtype, cache_kwargs or {}
    )
    gpt_model.reset()

    chunk_sizes = create_chunk_sizes(
        gpt_model=gpt_model,
        seq_length=seq_length,
        chunk_size=chunk_size,
        randomize_chunk_sizes=False,
    )
    chunks_per_cell = _partition_into_cells(chunk_sizes, cache_length, config)
    cells = get_chunks_for_cells(chunks_per_cell, chunk_sizes)

    log_probs = torch.zeros(batch_size, logits_to_keep, device=device, dtype=dtype)
    entropies = (
        torch.zeros(batch_size, logits_to_keep, device=device, dtype=dtype)
        if compute_entropy
        else None
    )

    blocks = [
        DefaultCellBlocks(model=gpt_model, first_layer_idx=i, num_layers=1)
        for i in range(config.n_layer)
    ]
    wte, scale = gpt_model.transformer.wte, config.n_embd**0.5

    with torch.no_grad():
        for cell in cells:
            start, end = cell.input_range

            embeddings = wte(input_ids[:, start:end])
            if config.scale_embeddings:
                embeddings *= scale

            for block in blocks:
                embeddings = torch.cat(
                    [
                        block.forward(
                            x=embeddings[:, rs:re, :],
                            idx=input_ids[:, (start + rs) : (start + re)],
                        )
                        for rs, re in cell.chunk_ranges
                    ],
                    dim=1,
                )

            for rs, re in cell.chunk_ranges:
                abs_s, abs_e = start + rs, start + re
                p_lo = max(abs_s, completion_start - 1)
                p_hi = min(abs_e, seq_length - 1)
                if p_lo >= p_hi:
                    continue

                chunk_logits = gpt_model.lm_head(
                    embeddings[:, (p_lo - start) : (p_hi - start), :]
                )
                if temperature != 1.0:
                    chunk_logits /= temperature

                targets = input_ids[:, (p_lo + 1) : (p_hi + 1)]
                out_slice = slice(
                    p_lo + 1 - completion_start, p_hi + 1 - completion_start
                )
                log_probs[:, out_slice] = _selective_log_softmax(chunk_logits, targets)

                if entropies is not None:
                    entropies[:, out_slice] = _entropy(chunk_logits)

            del embeddings

    write_back_cache_buffers(gpt_model)
    if caches_created:
        deallocate_kv_cache_buffers_of_model(gpt_model)
        gpt_model.assign_kv_caches([None] * config.n_layer)

    return log_probs, entropies


def _ensure_kv_caches(gpt_model, name, batch_size, length, dtype, kwargs) -> bool:
    """Assign KV caches if not already present. Returns True if we created them."""
    if (existing := gpt_model.get_kv_caches())[0] is not None:
        return False
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            gpt_model=gpt_model,
            name=name,
            max_batch_size=batch_size,
            cache_length=length,
            dtype=dtype,
            cache_kwargs=kwargs,
        )
    )
    return True


def _partition_into_cells(chunk_sizes, cache_length, config):
    """Partition chunks into cells whose total length ≈ cache_length."""
    max_cell_len = int(
        2 * config.n_query_groups * config.head_size / config.n_embd * cache_length
    )
    cells, cur_len, cur_n = [], 0, 0
    for cs in chunk_sizes:
        if cur_n and cur_len + cs > max_cell_len:
            cells.append(cur_n)
            cur_len, cur_n = cs, 1
        else:
            cur_len += cs
            cur_n += 1
    if cur_n:
        cells.append(cur_n)
    return cells


def _selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Log-softmax + gather — mirrors TRL's implementation."""
    if logits.dtype in (torch.float32, torch.float64):
        selected = torch.gather(logits, -1, index.unsqueeze(-1)).squeeze(-1)
        return selected - torch.logsumexp(logits, dim=-1)
    return torch.gather(F.log_softmax(logits, dim=-1), -1, index.unsqueeze(-1)).squeeze(
        -1
    )


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy in nats from logits, shape (..., vocab) → (...)."""
    p = F.log_softmax(logits, dim=-1)
    return -(p.exp() * p).sum(-1)
