# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Debug script to investigate FlashInfer vs eager attention weight differences

import math
import random
import torch
from torch.linalg import vector_norm

from keys_values.attention import DefaultKeysAndValues
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.test_utils import (
    compute_attn_weights,
    random_keys_values,
    random_tensor,
)
from keys_values.flashinfer_wrapper import FlashInferSDPA, get_flashinfer_sdpa
from keys_values import flashinfer_ops


def compute_attn_weights_with_flashinfer(
    query: torch.Tensor,
    k_and_v: DefaultKeysAndValues,
    scale_factor: float,
    input_pos: int = 0,
) -> torch.Tensor:
    """Compute attention weights using FlashInfer wrapper."""
    wrapper = get_flashinfer_sdpa()
    keys = k_and_v.keys()
    values = k_and_v.values()
    
    kv_len = keys.shape[2]
    q_len = query.shape[2]
    
    # For decode phase (q_len=1), the query position should be at the end of the KV cache
    # to allow attending to all previous positions with causal masking
    # input_pos should be kv_len - 1 for the query to see all KV positions
    effective_input_pos = input_pos if input_pos > 0 else kv_len - q_len
    
    print(f"  [FlashInfer] q_len={q_len}, kv_len={kv_len}, input_pos={input_pos}, effective_input_pos={effective_input_pos}")
    
    _, attn_weights = wrapper.scaled_dot_product_attention(
        query=query,
        key=keys,
        value=values,
        scale_factor=scale_factor,
        return_attn_weights=True,
        token_positions=None,
        input_pos=effective_input_pos,
        sliding_window_size=None,
    )
    return attn_weights


def main():
    if not flashinfer_ops.is_available():
        print("FlashInfer not available")
        return
    
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    # Simple test case
    batch_size = 2
    n_query_groups = 4
    n_head = 4
    head_size = 64
    kv_len = 8
    q_len = 1  # Single query token (decode phase)
    
    params = KVCacheParams(
        max_batch_size=batch_size,
        n_query_groups=n_query_groups,
        cache_length=32,
        head_size=head_size,
        n_head=n_head,
        device=device,
        dtype=dtype,
    )
    
    # Generate test data
    keys, values = random_keys_values(params, num=kv_len)
    query = random_tensor(params, num=q_len, is_query=True)
    
    k_and_v = DefaultKeysAndValues(keys, values)
    scale_factor = 1.0 / math.sqrt(head_size)
    
    print(f"Query shape: {query.shape}")
    print(f"Keys shape: {keys.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Scale factor: {scale_factor}")
    print()
    
    # Compute with eager implementation
    eager_weights = compute_attn_weights(query, k_and_v)
    print(f"Eager weights shape: {eager_weights.shape}")
    print(f"Eager weights dtype: {eager_weights.dtype}")
    print(f"Eager weights sum: {eager_weights.sum()}")
    print(f"Eager weights min: {eager_weights.min()}, max: {eager_weights.max()}")
    print()
    
    # Compute with FlashInfer
    # For decode phase, input_pos should be kv_len - 1 so query can see all KV positions
    flashinfer_weights = compute_attn_weights_with_flashinfer(query, k_and_v, scale_factor, input_pos=kv_len - 1)
    print(f"FlashInfer weights shape: {flashinfer_weights.shape}")
    print(f"FlashInfer weights dtype: {flashinfer_weights.dtype}")
    print(f"FlashInfer weights sum: {flashinfer_weights.sum()}")
    print(f"FlashInfer weights min: {flashinfer_weights.min()}, max: {flashinfer_weights.max()}")
    print()
    
    # Compare
    print("=== Comparison ===")
    print(f"Eager weights[0,0,:]: {eager_weights[0,0,:]}")
    print(f"FlashInfer weights[0,0,:]: {flashinfer_weights[0,0,:]}")
    print()
    
    # Check if shapes match
    if eager_weights.shape != flashinfer_weights.shape:
        print(f"SHAPE MISMATCH: eager={eager_weights.shape}, flashinfer={flashinfer_weights.shape}")
    else:
        diff = torch.abs(eager_weights - flashinfer_weights)
        print(f"Max diff: {diff.max()}")
        print(f"Mean diff: {diff.mean()}")
        print(f"Diff[0,0,:]: {diff[0,0,:]}")
    
    # Check the eager implementation details
    print("\n=== Eager Implementation Details ===")
    from keys_values.attention_utils import build_mask_cache
    kwargs = dict(dtype=query.dtype, device=query.device)
    mask = build_mask_cache(
        max_seq_length=q_len,
        sliding_window_size=None,
        **kwargs,
    )
    print(f"Mask shape before padding: {mask.shape}")
    if q_len < kv_len:
        _pad_zeros = torch.zeros((1, 1), **kwargs).expand(q_len, kv_len - q_len)
        mask = torch.cat((mask, _pad_zeros), dim=-1)
    print(f"Mask shape after padding: {mask.shape}")
    print(f"Mask: {mask}")


if __name__ == "__main__":
    main()
