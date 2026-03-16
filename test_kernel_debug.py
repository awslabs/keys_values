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
Debug script to isolate where the segfault occurs in FlashInfer kernels.
"""

import torch
import math


def test_import():
    """Test 1: Can we import the module?"""
    print("Test 1: Import module...")
    try:
        from keys_values import flashinfer_ops
        print(f"  SUCCESS: Module imported")
        print(f"  is_available(): {flashinfer_ops.is_available()}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_decode_minimal():
    """Test 2: Minimal decode kernel call."""
    print("\nTest 2: Minimal decode kernel (single token)...")
    try:
        from keys_values import flashinfer_ops
        
        # Smallest possible inputs
        batch_size = 1
        num_qo_heads = 1
        num_kv_heads = 1
        head_dim = 32
        kv_len = 4
        
        query = torch.randn(batch_size, num_qo_heads, head_dim, device='cuda', dtype=torch.float16)
        key = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
        value = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
        scale = 1.0 / math.sqrt(head_dim)
        
        print(f"  query shape: {query.shape}")
        print(f"  key shape: {key.shape}")
        print(f"  value shape: {value.shape}")
        
        # Synchronize before kernel call
        torch.cuda.synchronize()
        print("  Calling sdpa_decode...")
        
        output, weights = flashinfer_ops.sdpa_decode(
            query=query,
            key=key,
            value=value,
            scale=scale,
            causal=True,
            return_weights=False,
        )
        
        torch.cuda.synchronize()
        print(f"  SUCCESS: output shape = {output.shape}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_decode_with_weights():
    """Test 3: Decode kernel with attention weights."""
    print("\nTest 3: Decode kernel with attention weights...")
    try:
        from keys_values import flashinfer_ops
        
        batch_size = 1
        num_qo_heads = 1
        num_kv_heads = 1
        head_dim = 32
        kv_len = 4
        
        query = torch.randn(batch_size, num_qo_heads, head_dim, device='cuda', dtype=torch.float16)
        key = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
        value = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
        scale = 1.0 / math.sqrt(head_dim)
        
        torch.cuda.synchronize()
        print("  Calling sdpa_decode with return_weights=True...")
        
        output, weights = flashinfer_ops.sdpa_decode(
            query=query,
            key=key,
            value=value,
            scale=scale,
            causal=True,
            return_weights=True,
        )
        
        torch.cuda.synchronize()
        print(f"  SUCCESS: output shape = {output.shape}, weights shape = {weights.shape}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_prefill_minimal():
    """Test 4: Minimal prefill kernel call."""
    print("\nTest 4: Minimal prefill kernel...")
    try:
        from keys_values import flashinfer_ops
        
        # Smallest possible inputs
        batch_size = 1
        q_len = 2
        kv_len = 2
        num_qo_heads = 1
        num_kv_heads = 1
        head_dim = 32
        
        query = torch.randn(batch_size, q_len, num_qo_heads, head_dim, device='cuda', dtype=torch.float16)
        key = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
        value = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
        scale = 1.0 / math.sqrt(head_dim)
        
        print(f"  query shape: {query.shape}")
        print(f"  key shape: {key.shape}")
        print(f"  value shape: {value.shape}")
        
        torch.cuda.synchronize()
        print("  Calling sdpa_prefill...")
        
        output, weights = flashinfer_ops.sdpa_prefill(
            query=query,
            key=key,
            value=value,
            scale=scale,
            causal=True,
            return_weights=False,
        )
        
        torch.cuda.synchronize()
        print(f"  SUCCESS: output shape = {output.shape}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_prefill_with_weights():
    """Test 5: Prefill kernel with attention weights."""
    print("\nTest 5: Prefill kernel with attention weights...")
    try:
        from keys_values import flashinfer_ops
        
        batch_size = 1
        q_len = 2
        kv_len = 2
        num_qo_heads = 1
        num_kv_heads = 1
        head_dim = 32
        
        query = torch.randn(batch_size, q_len, num_qo_heads, head_dim, device='cuda', dtype=torch.float16)
        key = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
        value = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
        scale = 1.0 / math.sqrt(head_dim)
        
        torch.cuda.synchronize()
        print("  Calling sdpa_prefill with return_weights=True...")
        
        output, weights = flashinfer_ops.sdpa_prefill(
            query=query,
            key=key,
            value=value,
            scale=scale,
            causal=True,
            return_weights=True,
        )
        
        torch.cuda.synchronize()
        print(f"  SUCCESS: output shape = {output.shape}, weights shape = {weights.shape}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_prefill_larger():
    """Test 6: Prefill kernel with larger dimensions (like the failing test)."""
    print("\nTest 6: Prefill kernel with larger dimensions...")
    try:
        from keys_values import flashinfer_ops
        
        # Dimensions from the failing test
        batch_size = 2
        q_len = 16
        kv_len = 16
        num_qo_heads = 8
        num_kv_heads = 2
        head_dim = 64
        
        query = torch.randn(batch_size, q_len, num_qo_heads, head_dim, device='cuda', dtype=torch.float16)
        key = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
        value = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float16)
        scale = 1.0 / math.sqrt(head_dim)
        
        print(f"  query shape: {query.shape}")
        print(f"  key shape: {key.shape}")
        print(f"  value shape: {value.shape}")
        
        torch.cuda.synchronize()
        print("  Calling sdpa_prefill with return_weights=True...")
        
        output, weights = flashinfer_ops.sdpa_prefill(
            query=query,
            key=key,
            value=value,
            scale=scale,
            causal=True,
            return_weights=True,
        )
        
        torch.cuda.synchronize()
        print(f"  SUCCESS: output shape = {output.shape}, weights shape = {weights.shape}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_prefill_float32():
    """Test 7: Prefill kernel with float32 (might be more stable)."""
    print("\nTest 7: Prefill kernel with float32...")
    try:
        from keys_values import flashinfer_ops
        
        batch_size = 1
        q_len = 4
        kv_len = 4
        num_qo_heads = 2
        num_kv_heads = 2
        head_dim = 32
        
        query = torch.randn(batch_size, q_len, num_qo_heads, head_dim, device='cuda', dtype=torch.float32)
        key = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float32)
        value = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, device='cuda', dtype=torch.float32)
        scale = 1.0 / math.sqrt(head_dim)
        
        torch.cuda.synchronize()
        print("  Calling sdpa_prefill with float32...")
        
        output, weights = flashinfer_ops.sdpa_prefill(
            query=query,
            key=key,
            value=value,
            scale=scale,
            causal=True,
            return_weights=True,
        )
        
        torch.cuda.synchronize()
        print(f"  SUCCESS: output shape = {output.shape}, weights shape = {weights.shape}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def run_tests():
    """Run all debug tests sequentially."""
    print("=" * 60)
    print("FlashInfer Kernel Debug Tests")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    tests = [
        ("Import", test_import),
        ("Decode Minimal", test_decode_minimal),
        ("Decode with Weights", test_decode_with_weights),
        ("Prefill Minimal", test_prefill_minimal),
        ("Prefill with Weights", test_prefill_with_weights),
        ("Prefill Larger", test_prefill_larger),
        ("Prefill Float32", test_prefill_float32),
    ]
    
    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n{name}: CRASHED with {type(e).__name__}: {e}")
            results[name] = False
            # If we get a segfault, the process will terminate here
            # So we won't see subsequent tests
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    run_tests()
