# FlashInfer Optimization TODOs

## Completed

(None yet)

## Future Work

### TODO: Batch Decode Kernel Optimization (Option 1)

**Priority**: High
**Complexity**: Low
**Potential Speedup**: 2-3x for decode with batch sizes > 1

#### Background

Currently `launch_batch_decode_attention` in `sdpa_decode.cu` launches one kernel per batch item in a sequential loop (lines 462-501), causing:
1. Device-to-host sync for `input_pos` (blocking)
2. Sequential kernel launches (poor GPU utilization)
3. Decode throughput doesn't scale with batch size

#### Proposed Fix

Add `batched_tiled_decode_attention_kernel` with batch dimension in grid:
- Grid: `dim3(num_qo_heads, batch_size)`
- Read `input_pos` from device memory inside kernel
- Single kernel launch processes all batch items in parallel

#### Expected Results

Based on prototype testing (Mistral-7B, ctx=2048):
- bs=8 decode: 0.23x → 0.67x vs eager (2.9x improvement)
- Decode scales with batch size (1080 → 3664 tok/s)

---

### TODO: Port FlashInfer Paged KV Cache System (Option 2)

**Priority**: Medium
**Complexity**: High
**Potential Speedup**: 2-5x for decode with large batch sizes

#### Background

The original FlashInfer library uses a paged KV cache system (`paged_kv_t`) that enables:
1. Efficient memory management for variable-length sequences
2. True batched decode kernel (`BatchDecodeWithPagedKVCacheDispatched`)
3. Better memory locality and reduced fragmentation

Currently, keys_values uses contiguous KV tensors which required a custom implementation that doesn't leverage FlashInfer's optimized batch decode infrastructure.

#### What Would Need to Be Ported

1. **Paged KV Cache Data Structure** (`paged_kv_t`)
   - Location: `flashinfer/include/flashinfer/attention/cascade.cuh`
   - Manages KV cache as pages/blocks instead of contiguous tensors
   - Supports variable sequence lengths per batch item

2. **Batch Decode Kernel** (`BatchDecodeWithPagedKVCacheKernel`)
   - Location: `flashinfer/include/flashinfer/attention/decode.cuh:613`
   - Uses 2D grid: `dim3 nblks(padded_batch_size, num_kv_heads)`
   - Processes all batch items in parallel

3. **Scheduler Infrastructure** (`scheduler.cuh`)
   - Work estimation and load balancing
   - Partition-KV for very long sequences
   - State merging utilities

4. **Python Wrapper Updates**
   - Page table management
   - Indptr/indices tensor handling
   - Memory pool integration

#### Files to Reference

```
flashinfer/include/flashinfer/attention/
├── decode.cuh                    # BatchDecodeWithPagedKVCacheDispatched
├── scheduler.cuh                 # DecodePlan, work estimation
├── cascade.cuh                   # paged_kv_t structure
└── default_decode_params.cuh     # Parameter structures

flashinfer/csrc/
├── batch_decode.cu               # Python bindings
└── batch_decode_kernel_inst.jinja # Kernel instantiation
```

#### Integration Considerations

1. **Backward Compatibility**: Must maintain support for contiguous KV tensors
2. **Memory Management**: Need to integrate with existing KV cache eviction policies (H2O)
3. **Token Positions**: Paged KV cache needs to work with per-head token position tracking
4. **Testing**: Extensive numerical equivalence testing required

#### Estimated Effort

- Initial port: 2-3 weeks
- Integration with sparse attention: 1-2 weeks
- Testing and optimization: 1-2 weeks

---

### TODO: Attention Weights in FlashInfer Decode Path

**Priority**: Low
**Complexity**: Medium

Currently, when `return_attn_weights=True`, the code falls back to the tiled reference implementation because FlashInfer's native decode kernel doesn't return attention weights.

A future optimization could:
1. Modify FlashInfer decode to optionally compute and return weights
2. Or use a two-pass approach: FlashInfer for output, separate kernel for weights

---

### TODO: Serialization/Deserialization of Attention State (Requirement 6)

**Priority**: Medium
**Complexity**: Medium

From requirements.md - not yet implemented:
- Serialize token_positions, cache metadata for resumable inference
- Support distributed inference scenarios
- Device transfer handling

---

## Performance Reference

### Current State (sequential kernel launches)

Benchmark on Mistral-7B, context=2048:

| Batch | FlashInfer Decode | Eager Decode | Ratio |
|-------|-------------------|--------------|-------|
| 1     | 1278 tok/s        | 2157 tok/s   | 0.59x |
| 2     | 1433 tok/s        | 4035 tok/s   | 0.36x |
| 4     | 1266 tok/s        | 5146 tok/s   | 0.25x |
| 8     | 1299 tok/s        | 5674 tok/s   | 0.23x |

FlashInfer decode throughput is constant regardless of batch size due to sequential kernel launches.





Notes:
BS: 2
Context length: 32k 
Chunk size: 2048 - 4096

Slower the chunk size better it works.

Check for compression?
attention masking should be done correctly. 

We need to know hwo they are ordered in the real sequence. If we dont have we have to sort the key and value buffer first. Currently, we sort first and then pass it in. Sorting is expensive. 

Forward : 