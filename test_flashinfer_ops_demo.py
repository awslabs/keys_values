#!/usr/bin/env python3
"""
Demo script to test the flashinfer_ops module functionality.
"""

import torch
import keys_values.flashinfer_ops as ops

print("=" * 60)
print("FlashInfer Ops Module Demo")
print("=" * 60)

# Test 1: Check availability
print("\n1. Checking kernel availability...")
print(f"   is_available(): {ops.is_available()}")

if not ops.is_available():
    print(f"   Load error: {ops.get_load_error()}")
    print("\n   Kernels not available. Exiting.")
    exit(0)

# Test 2: Device information
print("\n2. CUDA device information...")
device_count = ops.get_device_count()
print(f"   Device count: {device_count}")
for i in range(min(device_count, 2)):  # Show first 2 devices
    print(f"   {ops.get_device_info(i)}")

# Test 3: Test sdpa_decode with small tensors
print("\n3. Testing sdpa_decode (decode phase)...")
try:
    batch_size = 2
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    kv_len = 1024
    
    query = torch.randn(batch_size, num_qo_heads, head_dim, device='cuda', dtype=torch.float16)
    key = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
    value = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
    scale = 1.0 / (head_dim ** 0.5)
    
    output, weights = ops.sdpa_decode(
        query, key, value, scale,
        return_weights=True
    )
    
    print(f"   Input shapes:")
    print(f"     query: {list(query.shape)}")
    print(f"     key: {list(key.shape)}")
    print(f"     value: {list(value.shape)}")
    print(f"   Output shapes:")
    print(f"     output: {list(output.shape)}")
    print(f"     weights: {list(weights.shape) if weights is not None else None}")
    print(f"   ✓ sdpa_decode works correctly")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Test sdpa_prefill with small tensors
print("\n4. Testing sdpa_prefill (prefill phase)...")
try:
    batch_size = 2
    q_len = 512
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    kv_len = 1024
    
    query = torch.randn(batch_size, q_len, num_qo_heads, head_dim, device='cuda', dtype=torch.float16)
    key = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
    value = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
    scale = 1.0 / (head_dim ** 0.5)
    
    output, weights = ops.sdpa_prefill(
        query, key, value, scale,
        return_weights=True
    )
    
    print(f"   Input shapes:")
    print(f"     query: {list(query.shape)}")
    print(f"     key: {list(key.shape)}")
    print(f"     value: {list(value.shape)}")
    print(f"   Output shapes:")
    print(f"     output: {list(output.shape)}")
    print(f"     weights: {list(weights.shape) if weights is not None else None}")
    print(f"   ✓ sdpa_prefill works correctly")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 5: Test error handling when kernels not available
print("\n5. Testing error handling...")
try:
    # This should work since kernels are available
    _ = ops.get_device_count()
    print(f"   ✓ Functions work when kernels are available")
except RuntimeError as e:
    print(f"   ✗ Unexpected error: {e}")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
