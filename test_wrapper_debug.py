# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

"""Debug script to isolate segfault in wrapper's _flashinfer_sdpa_standard."""

import torch
import math


def test_wrapper_standard_path():
    """Test the exact path that verify_backend_equivalence uses."""
    print("=" * 60)
    print("Testing wrapper's _flashinfer_sdpa_standard path")
    print("=" * 60)
    
    from keys_values.flashinfer_wrapper import FlashInferSDPA
    
    wrapper = FlashInferSDPA()
    print(f"FlashInfer available: {wrapper.available}")
    
    if not wrapper.available:
        print("SKIPPED: FlashInfer not available")
        return
    
    # Same dimensions as test_basic_equivalence
    batch_size, n_head, n_query_groups = 2, 8, 2
    q_len, kv_len, head_size = 16, 16, 64
    scale_factor = 1.0 / math.sqrt(head_size)
    
    # Create tensors in keys_values format
    query = torch.randn(batch_size, n_head, q_len, head_size, dtype=torch.float16, device='cuda')
    key = torch.randn(batch_size, n_query_groups, kv_len, head_size, dtype=torch.float16, device='cuda')
    value = torch.randn(batch_size, n_query_groups, kv_len, head_size, dtype=torch.float16, device='cuda')
    
    print(f"\nInput shapes (keys_values format):")
    print(f"  query: {query.shape}")
    print(f"  key: {key.shape}")
    print(f"  value: {value.shape}")
    
    # Step 1: Test the transformation
    print("\nStep 1: Testing tensor transformation...")
    query_transformed = query.transpose(1, 2).contiguous()
    key_transformed = key.transpose(1, 2).contiguous()
    value_transformed = value.transpose(1, 2).contiguous()
    
    print(f"  query_transformed: {query_transformed.shape}")
    print(f"  key_transformed: {key_transformed.shape}")
    print(f"  value_transformed: {value_transformed.shape}")
    
    # Step 2: Call kernel directly with transformed tensors
    print("\nStep 2: Calling kernel directly with transformed tensors...")
    torch.cuda.synchronize()
    
    from keys_values import flashinfer_ops
    
    try:
        output, weights = flashinfer_ops.sdpa_prefill(
            query=query_transformed,
            key=key_transformed,
            value=value_transformed,
            scale=scale_factor,
            causal=True,
            return_weights=True,
        )
        torch.cuda.synchronize()
        print(f"  SUCCESS: output={output.shape}, weights={weights.shape}")
    except Exception as e:
        print(f"  FAILED: {e}")
        return
    
    # Step 3: Call kernel with input_pos parameter (like wrapper does)
    print("\nStep 3: Calling kernel with input_pos parameter...")
    torch.cuda.synchronize()
    
    input_pos_tensor = torch.tensor([0, 0], device='cuda', dtype=torch.int32)
    print(f"  input_pos_tensor: {input_pos_tensor.shape}, dtype={input_pos_tensor.dtype}")
    
    try:
        output2, weights2 = flashinfer_ops.sdpa_prefill(
            query=query_transformed,
            key=key_transformed,
            value=value_transformed,
            scale=scale_factor,
            token_positions=None,
            input_pos=input_pos_tensor,
            sliding_window_size=-1,
            causal=True,
            return_weights=True,
        )
        torch.cuda.synchronize()
        print(f"  SUCCESS: output={output2.shape}, weights={weights2.shape}")
    except Exception as e:
        print(f"  FAILED: {e}")
        return
    
    # Step 4: Call wrapper's _flashinfer_sdpa_standard
    print("\nStep 4: Calling wrapper._flashinfer_sdpa_standard...")
    torch.cuda.synchronize()
    
    try:
        output3, weights3 = wrapper._flashinfer_sdpa_standard(
            query, key, value, scale_factor,
            return_attn_weights=True,
        )
        torch.cuda.synchronize()
        print(f"  SUCCESS: output={output3.shape}, weights={weights3.shape}")
    except Exception as e:
        print(f"  FAILED: {e}")
        return
    
    # Step 5: Call wrapper._flashinfer_sdpa (the dispatcher)
    print("\nStep 5: Calling wrapper._flashinfer_sdpa...")
    torch.cuda.synchronize()
    
    try:
        output4, weights4 = wrapper._flashinfer_sdpa(
            query, key, value, scale_factor,
            return_attn_weights=True,
        )
        torch.cuda.synchronize()
        print(f"  SUCCESS: output={output4.shape}, weights={weights4.shape}")
    except Exception as e:
        print(f"  FAILED: {e}")
        return
    
    print("\nAll steps passed!")


if __name__ == "__main__":
    test_wrapper_standard_path()
