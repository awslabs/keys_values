/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Prefill SDPA kernel implementation.
 * 
 * This file provides two implementations:
 * 1. Real FlashInfer kernels for standard causal attention (token_positions=None)
 * 2. Tiled reference implementation for sparse attention with token_positions
 */

#include "sdpa_prefill.cuh"
#include <algorithm>
#include <vector>

// Include FlashInfer headers for real kernel dispatch
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>

namespace keys_values {
namespace kernels {

// ============================================================================
// FlashInfer kernel dispatch for standard causal attention
// ============================================================================

/**
 * @brief Dispatch to real FlashInfer prefill kernel
 * 
 * This function calls FlashInfer's SinglePrefillWithKVCacheDispatched with
 * the appropriate template parameters based on head dimension.
 */
template <uint32_t HEAD_DIM, typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t dispatch_flashinfer_prefill_impl(
    DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o, float* lse,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t qo_len, uint32_t kv_len,
    uint32_t q_stride_n, uint32_t q_stride_h,
    uint32_t kv_stride_n, uint32_t kv_stride_h,
    int32_t window_left,
    float sm_scale,
    cudaStream_t stream) {
    
    using namespace flashinfer;
    
    // Create FlashInfer params using the correct constructor
    flashinfer::SinglePrefillParams<DTypeQ, DTypeKV, DTypeO> fi_params(
        q, k, v,
        nullptr,  // maybe_custom_mask
        o, lse,
        nullptr,  // maybe_alibi_slopes
        num_qo_heads, num_kv_heads,
        qo_len, kv_len,
        q_stride_n, q_stride_h,
        kv_stride_n, kv_stride_h,
        HEAD_DIM,
        window_left,
        0.0f,  // logits_soft_cap
        sm_scale,
        1.0f,  // rope_scale
        10000.0f  // rope_theta
    );

    // Use DefaultAttention variant with no sliding window, no soft cap, no alibi
    constexpr bool use_sliding_window = false;
    using AttentionVariant = DefaultAttention<false, use_sliding_window, false, false>;

    // Dispatch to FlashInfer kernel with causal mask
    return SinglePrefillWithKVCacheDispatched<
        HEAD_DIM, HEAD_DIM, PosEncodingMode::kNone, false, MaskMode::kCausal, AttentionVariant>(
        fi_params, nullptr, stream);
}

/**
 * @brief Non-causal dispatch to FlashInfer prefill kernel (MaskMode::kNone)
 */
template <uint32_t HEAD_DIM, typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t dispatch_flashinfer_prefill_noncausal_impl(
    DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o, float* lse,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t qo_len, uint32_t kv_len,
    uint32_t q_stride_n, uint32_t q_stride_h,
    uint32_t kv_stride_n, uint32_t kv_stride_h,
    float sm_scale,
    cudaStream_t stream) {

    using namespace flashinfer;

    flashinfer::SinglePrefillParams<DTypeQ, DTypeKV, DTypeO> fi_params(
        q, k, v,
        nullptr,  // maybe_custom_mask
        o, lse,
        nullptr,  // maybe_alibi_slopes
        num_qo_heads, num_kv_heads,
        qo_len, kv_len,
        q_stride_n, q_stride_h,
        kv_stride_n, kv_stride_h,
        HEAD_DIM,
        -1,  // window_left (no window for non-causal)
        0.0f,  // logits_soft_cap
        sm_scale,
        1.0f,  // rope_scale
        10000.0f  // rope_theta
    );

    using AttentionVariant = DefaultAttention<false, false, false, false>;

    return SinglePrefillWithKVCacheDispatched<
        HEAD_DIM, HEAD_DIM, PosEncodingMode::kNone, false, MaskMode::kNone, AttentionVariant>(
        fi_params, nullptr, stream);
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t dispatch_flashinfer_prefill_noncausal(
    DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o, float* lse,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t qo_len, uint32_t kv_len,
    uint32_t q_stride_n, uint32_t q_stride_h,
    uint32_t kv_stride_n, uint32_t kv_stride_h,
    uint32_t head_dim,
    float sm_scale,
    cudaStream_t stream) {

    if constexpr (sizeof(DTypeQ) != 2 || sizeof(DTypeKV) != 2) {
        return cudaErrorNotSupported;
    } else {
        if (head_dim == 64) {
            return dispatch_flashinfer_prefill_noncausal_impl<64>(
                q, k, v, o, lse, num_qo_heads, num_kv_heads, qo_len, kv_len,
                q_stride_n, q_stride_h, kv_stride_n, kv_stride_h, sm_scale, stream);
        } else if (head_dim == 128) {
            return dispatch_flashinfer_prefill_noncausal_impl<128>(
                q, k, v, o, lse, num_qo_heads, num_kv_heads, qo_len, kv_len,
                q_stride_n, q_stride_h, kv_stride_n, kv_stride_h, sm_scale, stream);
        } else if (head_dim == 256) {
            return dispatch_flashinfer_prefill_noncausal_impl<256>(
                q, k, v, o, lse, num_qo_heads, num_kv_heads, qo_len, kv_len,
                q_stride_n, q_stride_h, kv_stride_n, kv_stride_h, sm_scale, stream);
        } else {
            return cudaErrorNotSupported;
        }
    }
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t dispatch_flashinfer_prefill(
    DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o, float* lse,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t qo_len, uint32_t kv_len,
    uint32_t q_stride_n, uint32_t q_stride_h,
    uint32_t kv_stride_n, uint32_t kv_stride_h,
    uint32_t head_dim,
    int32_t window_left,
    float sm_scale,
    cudaStream_t stream) {

    // FlashInfer only supports 16-bit types (half, bfloat16), not float32
    // Use compile-time check to skip FlashInfer for unsupported types
    if constexpr (sizeof(DTypeQ) != 2 || sizeof(DTypeKV) != 2) {
        return cudaErrorNotSupported;
    } else {
        // Dispatch based on head dimension (compile-time constant required)
        if (head_dim == 64) {
            return dispatch_flashinfer_prefill_impl<64>(
                q, k, v, o, lse, num_qo_heads, num_kv_heads, qo_len, kv_len,
                q_stride_n, q_stride_h, kv_stride_n, kv_stride_h, window_left, sm_scale, stream);
        } else if (head_dim == 128) {
            return dispatch_flashinfer_prefill_impl<128>(
                q, k, v, o, lse, num_qo_heads, num_kv_heads, qo_len, kv_len,
                q_stride_n, q_stride_h, kv_stride_n, kv_stride_h, window_left, sm_scale, stream);
        } else if (head_dim == 256) {
            return dispatch_flashinfer_prefill_impl<256>(
                q, k, v, o, lse, num_qo_heads, num_kv_heads, qo_len, kv_len,
                q_stride_n, q_stride_h, kv_stride_n, kv_stride_h, window_left, sm_scale, stream);
        } else {
            // Unsupported head dimension - return error to trigger fallback
            return cudaErrorNotSupported;
        }
    }
}

// ============================================================================
// Attention weight accumulation kernels
// ============================================================================

__global__ void prefill_accumulate_attention_weights_kernel(
    const float* __restrict__ attn_weights_per_query,
    float* __restrict__ attn_weights_sum,
    uint32_t q_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t kv_len) {
    
    const uint32_t kv_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t kv_head_idx = blockIdx.y;
    
    if (kv_idx >= kv_len || kv_head_idx >= num_kv_heads) {
        return;
    }
    
    const uint32_t q_per_kv = num_qo_heads / num_kv_heads;
    const uint32_t qo_head_start = kv_head_idx * q_per_kv;
    const uint32_t qo_head_end = qo_head_start + q_per_kv;
    
    float sum = 0.0f;
    
    for (uint32_t q_idx = 0; q_idx < q_len; ++q_idx) {
        for (uint32_t qo_head_idx = qo_head_start; qo_head_idx < qo_head_end; ++qo_head_idx) {
            sum += attn_weights_per_query[
                q_idx * num_qo_heads * kv_len + qo_head_idx * kv_len + kv_idx];
        }
    }
    
    attn_weights_sum[kv_head_idx * kv_len + kv_idx] = sum;
}


cudaError_t launch_prefill_accumulate_attention_weights(
    const float* attn_weights_per_query,
    float* attn_weights_sum,
    uint32_t q_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t kv_len,
    cudaStream_t stream) {
    
    const uint32_t block_size = 256;
    const uint32_t grid_x = (kv_len + block_size - 1) / block_size;
    
    dim3 grid(grid_x, num_kv_heads);
    dim3 block(block_size);
    
    prefill_accumulate_attention_weights_kernel<<<grid, block, 0, stream>>>(
        attn_weights_per_query, attn_weights_sum,
        q_len, num_qo_heads, num_kv_heads, kv_len);
    
    return cudaGetLastError();
}

__global__ void batch_prefill_accumulate_attention_weights_kernel(
    const float* __restrict__ attn_weights_per_query,
    float* __restrict__ attn_weights_sum,
    uint32_t batch_size,
    uint32_t q_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t kv_len) {
    
    const uint32_t kv_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t kv_head_idx = blockIdx.y;
    const uint32_t batch_idx = blockIdx.z;
    
    if (kv_idx >= kv_len || kv_head_idx >= num_kv_heads || batch_idx >= batch_size) {
        return;
    }
    
    const uint32_t q_per_kv = num_qo_heads / num_kv_heads;
    const uint32_t qo_head_start = kv_head_idx * q_per_kv;
    const uint32_t qo_head_end = qo_head_start + q_per_kv;
    
    const uint32_t batch_offset_in = batch_idx * q_len * num_qo_heads * kv_len;
    const uint32_t batch_offset_out = batch_idx * num_kv_heads * kv_len;
    
    float sum = 0.0f;
    
    for (uint32_t q_idx = 0; q_idx < q_len; ++q_idx) {
        for (uint32_t qo_head_idx = qo_head_start; qo_head_idx < qo_head_end; ++qo_head_idx) {
            sum += attn_weights_per_query[
                batch_offset_in + q_idx * num_qo_heads * kv_len + qo_head_idx * kv_len + kv_idx];
        }
    }
    
    attn_weights_sum[batch_offset_out + kv_head_idx * kv_len + kv_idx] = sum;
}

cudaError_t launch_batch_prefill_accumulate_attention_weights(
    const float* attn_weights_per_query,
    float* attn_weights_sum,
    uint32_t batch_size,
    uint32_t q_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t kv_len,
    cudaStream_t stream) {
    
    const uint32_t block_size = 256;
    const uint32_t grid_x = (kv_len + block_size - 1) / block_size;
    
    dim3 grid(grid_x, num_kv_heads, batch_size);
    dim3 block(block_size);
    
    batch_prefill_accumulate_attention_weights_kernel<<<grid, block, 0, stream>>>(
        attn_weights_per_query, attn_weights_sum,
        batch_size, q_len, num_qo_heads, num_kv_heads, kv_len);
    
    return cudaGetLastError();
}


// ============================================================================
// Tiled reference implementation for sparse attention with token_positions
// ============================================================================

constexpr uint32_t TILE_SIZE_KV = 256;

template<typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}

template<>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template<>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val) {
    return static_cast<T>(val);
}

template<>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) {
    return __float2bfloat16(val);
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
__global__ void tiled_prefill_attention_kernel(
    const DTypeQ* __restrict__ q,
    const DTypeKV* __restrict__ k,
    const DTypeKV* __restrict__ v,
    DTypeO* __restrict__ o,
    float* __restrict__ lse,
    float* __restrict__ attn_weights,
    const int32_t* __restrict__ token_positions,
    uint32_t token_positions_stride_h,
    uint32_t q_len,
    uint32_t kv_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t q_stride_n,
    uint32_t q_stride_h,
    uint32_t kv_stride_n,
    uint32_t kv_stride_h,
    float sm_scale,
    int32_t input_pos,
    int32_t sliding_window_size,
    bool causal,
    bool return_weights) {
    
    const uint32_t q_idx = blockIdx.x;
    const uint32_t qo_head_idx = blockIdx.y;
    const uint32_t kv_head_idx = qo_head_idx / (num_qo_heads / num_kv_heads);
    const uint32_t tid = threadIdx.x;
    
    if (q_idx >= q_len || qo_head_idx >= num_qo_heads) {
        return;
    }
    
    extern __shared__ float smem[];
    float* scores = smem;
    float* reduction_buffer = smem + TILE_SIZE_KV;
    float* o_shared = smem + TILE_SIZE_KV + blockDim.x;
    
    int32_t query_pos = input_pos + q_idx;
    
    const int32_t* head_token_positions = token_positions;
    if (token_positions != nullptr && token_positions_stride_h > 0) {
        head_token_positions = token_positions + kv_head_idx * token_positions_stride_h;
    }
    
    for (uint32_t d_idx = tid; d_idx < head_dim; d_idx += blockDim.x) {
        o_shared[d_idx] = 0.0f;
    }
    __syncthreads();
    
    __shared__ float m_shared;
    __shared__ float d_shared;
    if (tid == 0) {
        m_shared = -INFINITY;
        d_shared = 0.0f;
    }
    __syncthreads();
    
    const uint32_t num_tiles = (kv_len + TILE_SIZE_KV - 1) / TILE_SIZE_KV;

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const uint32_t tile_start = tile_idx * TILE_SIZE_KV;
        const uint32_t tile_end = min(tile_start + TILE_SIZE_KV, kv_len);
        const uint32_t tile_size = tile_end - tile_start;
        
        float tile_max = -INFINITY;
        for (uint32_t kv_offset = tid; kv_offset < tile_size; kv_offset += blockDim.x) {
            const uint32_t kv_idx = tile_start + kv_offset;
            
            bool valid = true;
            if (causal) {
                if (head_token_positions != nullptr) {
                    int32_t key_pos = head_token_positions[kv_idx];
                    valid = (key_pos <= query_pos);
                    if (valid && sliding_window_size > 0) {
                        valid = (query_pos - key_pos < sliding_window_size);
                    }
                } else {
                    valid = (kv_idx <= q_idx);
                    if (valid && sliding_window_size > 0) {
                        valid = (q_idx - kv_idx < static_cast<uint32_t>(sliding_window_size));
                    }
                }
            }
            
            float score = -INFINITY;
            if (valid) {
                score = 0.0f;
                for (uint32_t d_idx = 0; d_idx < head_dim; ++d_idx) {
                    float q_val = to_float(q[q_idx * q_stride_n + qo_head_idx * q_stride_h + d_idx]);
                    float k_val = to_float(k[kv_idx * kv_stride_n + kv_head_idx * kv_stride_h + d_idx]);
                    score += q_val * k_val;
                }
                score *= sm_scale;
            }
            scores[kv_offset] = score;
            tile_max = fmaxf(tile_max, score);
        }
        
        __syncthreads();
        
        reduction_buffer[tid] = tile_max;
        __syncthreads();
        
        for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                reduction_buffer[tid] = fmaxf(reduction_buffer[tid], reduction_buffer[tid + s]);
            }
            __syncthreads();
        }
        tile_max = reduction_buffer[0];
        
        float m_prev = m_shared;
        float m_new = fmaxf(m_prev, tile_max);
        float scale_prev = (m_prev > -INFINITY) ? expf(m_prev - m_new) : 0.0f;
        
        for (uint32_t d_idx = tid; d_idx < head_dim; d_idx += blockDim.x) {
            o_shared[d_idx] *= scale_prev;
        }
        
        if (tid == 0) {
            d_shared *= scale_prev;
            m_shared = m_new;
        }
        __syncthreads();
        
        float tile_sum = 0.0f;
        for (uint32_t kv_offset = tid; kv_offset < tile_size; kv_offset += blockDim.x) {
            float exp_score = expf(scores[kv_offset] - m_shared);
            scores[kv_offset] = exp_score;
            tile_sum += exp_score;
        }
        
        __syncthreads();
        
        reduction_buffer[tid] = tile_sum;
        __syncthreads();
        
        for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                reduction_buffer[tid] += reduction_buffer[tid + s];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            d_shared += reduction_buffer[0];
        }
        __syncthreads();
        
        for (uint32_t d_idx = tid; d_idx < head_dim; d_idx += blockDim.x) {
            float v_sum = 0.0f;
            for (uint32_t kv_offset = 0; kv_offset < tile_size; ++kv_offset) {
                const uint32_t kv_idx = tile_start + kv_offset;
                float v_val = to_float(v[kv_idx * kv_stride_n + kv_head_idx * kv_stride_h + d_idx]);
                v_sum += scores[kv_offset] * v_val;
            }
            o_shared[d_idx] += v_sum;
        }
        
        __syncthreads();
    }
    
    float inv_d = (d_shared > 0.0f) ? 1.0f / d_shared : 0.0f;
    
    for (uint32_t d_idx = tid; d_idx < head_dim; d_idx += blockDim.x) {
        float out_val = o_shared[d_idx] * inv_d;
        o[q_idx * num_qo_heads * head_dim + qo_head_idx * head_dim + d_idx] = from_float<DTypeO>(out_val);
    }

    // Write LSE if requested (log2 scale to match FlashInfer convention)
    if (lse != nullptr && tid == 0) {
        float lse_val = m_shared + logf(d_shared);  // natural log
        lse_val *= 1.4426950408889634f;  // convert to log2: log2(e) = 1/ln(2)
        lse[q_idx * num_qo_heads + qo_head_idx] = lse_val;
    }

    if (return_weights && attn_weights != nullptr) {
        for (uint32_t kv_idx = tid; kv_idx < kv_len; kv_idx += blockDim.x) {
            bool valid = true;
            if (causal) {
                if (head_token_positions != nullptr) {
                    int32_t key_pos = head_token_positions[kv_idx];
                    valid = (key_pos <= query_pos);
                    if (valid && sliding_window_size > 0) {
                        valid = (query_pos - key_pos < sliding_window_size);
                    }
                } else {
                    valid = (kv_idx <= q_idx);
                    if (valid && sliding_window_size > 0) {
                        valid = (q_idx - kv_idx < static_cast<uint32_t>(sliding_window_size));
                    }
                }
            }
            
            float weight = 0.0f;
            if (valid) {
                float score = 0.0f;
                for (uint32_t d_idx = 0; d_idx < head_dim; ++d_idx) {
                    float q_val = to_float(q[q_idx * q_stride_n + qo_head_idx * q_stride_h + d_idx]);
                    float k_val = to_float(k[kv_idx * kv_stride_n + kv_head_idx * kv_stride_h + d_idx]);
                    score += q_val * k_val;
                }
                score *= sm_scale;
                weight = expf(score - m_shared) * inv_d;
            }
            
            attn_weights[q_idx * num_qo_heads * kv_len + qo_head_idx * kv_len + kv_idx] = weight;
        }
    }
}

// ============================================================================
// Fused tiled prefill kernel — accumulates weights during tiling loop
// ============================================================================

/**
 * @brief Fused prefill kernel that computes O and accumulated attention weights
 *        in a single pass, using score caching to avoid re-reading K.
 *
 * Output weights have shape [num_kv_heads, kv_len], representing:
 *   W[kv_head, k] = Σ_{q, h∈group} softmax(Q[q,h]·K[kv_head,k])
 *
 * Strategy: During the main tiling loop, raw scores (Q·K * sm_scale) are
 * cached to a temporary global memory buffer (one float per KV position).
 * After the loop, a lightweight normalization pass reads the cached scores
 * and computes exp(score - m_final) / d_final, then atomicAdds to the
 * output weight buffer.
 *
 * Memory traffic for weight computation:
 *   - Cache write during loop: kv_len * 4 bytes (1 float per position)
 *   - Cache read + normalize:  kv_len * 4 bytes
 *   - Total: 8 bytes per KV position
 *   - vs re-reading K: kv_len * head_dim * sizeof(dtype) = kv_len * 256 bytes
 *   - Overhead ratio: 8 / 256 = ~3% for head_dim=128, bf16
 *
 * Requires: score_cache buffer of size [q_len * num_qo_heads * kv_len] floats
 *           (allocated by caller). This is smaller than the old per-query weight
 *           intermediate when kv_len < head_dim (typical), and the cache is only
 *           written/read once (vs the old approach which reads K twice).
 *
 * WAIT — that's the same large buffer. Instead, use a DIFFERENT approach:
 * write raw scores to a per-thread-block section of the cache. Each block
 * handles (q_idx, qo_head_idx), needs kv_len floats. Total = q_len * num_qo_heads * kv_len.
 *
 * BETTER APPROACH: Use per-tile score caching in shared memory + immediate
 * weight write. After computing exp scores for a tile (which are in shared
 * memory as scores[]), we already have the unnormalized exp values. We write
 * these to global memory immediately. After ALL tiles, we do a lightweight
 * normalization pass that reads these cached exp values and corrects them.
 *
 * But the exp values use the current m_shared which changes per tile. So we
 * also store per-tile m values in shared memory (tiny: num_tiles floats).
 *
 * SIMPLEST CORRECT APPROACH: Cache the raw score (before exp) per KV position
 * to global memory during the tiling loop. Cost: one float WRITE per position.
 * After the loop: one float READ + exp + atomicAdd per position. NO K re-read.
 */
template <typename DTypeQ, typename DTypeKV, typename DTypeO>
__global__ void fused_prefill_attention_kernel(
    const DTypeQ* __restrict__ q,
    const DTypeKV* __restrict__ k,
    const DTypeKV* __restrict__ v,
    DTypeO* __restrict__ o,
    float* __restrict__ attn_weights_sum,  // [num_kv_heads, kv_len], pre-zeroed
    float* __restrict__ score_cache,       // [q_len, num_qo_heads, kv_len] temp buffer
    const int32_t* __restrict__ token_positions,
    uint32_t token_positions_stride_h,
    uint32_t q_len,
    uint32_t kv_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t q_stride_n,
    uint32_t q_stride_h,
    uint32_t kv_stride_n,
    uint32_t kv_stride_h,
    float sm_scale,
    int32_t input_pos,
    int32_t sliding_window_size,
    bool causal) {

    const uint32_t q_idx = blockIdx.x;
    const uint32_t qo_head_idx = blockIdx.y;
    const uint32_t kv_head_idx = qo_head_idx / (num_qo_heads / num_kv_heads);
    const uint32_t tid = threadIdx.x;

    if (q_idx >= q_len || qo_head_idx >= num_qo_heads) {
        return;
    }

    extern __shared__ float smem[];
    float* scores = smem;
    float* reduction_buffer = smem + TILE_SIZE_KV;
    float* o_shared = smem + TILE_SIZE_KV + blockDim.x;

    int32_t query_pos = input_pos + q_idx;

    const int32_t* head_token_positions = token_positions;
    if (token_positions != nullptr && token_positions_stride_h > 0) {
        head_token_positions = token_positions + kv_head_idx * token_positions_stride_h;
    }

    // Per-block score cache offset
    float* my_score_cache = score_cache ?
        score_cache + (q_idx * num_qo_heads + qo_head_idx) * kv_len : nullptr;

    for (uint32_t d_idx = tid; d_idx < head_dim; d_idx += blockDim.x) {
        o_shared[d_idx] = 0.0f;
    }
    __syncthreads();

    __shared__ float m_shared;
    __shared__ float d_shared;
    if (tid == 0) {
        m_shared = -INFINITY;
        d_shared = 0.0f;
    }
    __syncthreads();

    const uint32_t num_tiles = (kv_len + TILE_SIZE_KV - 1) / TILE_SIZE_KV;

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const uint32_t tile_start = tile_idx * TILE_SIZE_KV;
        const uint32_t tile_end = min(tile_start + TILE_SIZE_KV, kv_len);
        const uint32_t tile_size = tile_end - tile_start;

        // Step 1: Compute Q·K scores for this tile
        float tile_max = -INFINITY;
        for (uint32_t kv_offset = tid; kv_offset < tile_size; kv_offset += blockDim.x) {
            const uint32_t kv_idx = tile_start + kv_offset;

            bool valid = true;
            if (causal) {
                if (head_token_positions != nullptr) {
                    int32_t key_pos = head_token_positions[kv_idx];
                    valid = (key_pos <= query_pos);
                    if (valid && sliding_window_size > 0) {
                        valid = (query_pos - key_pos < sliding_window_size);
                    }
                } else {
                    valid = (kv_idx <= q_idx);
                    if (valid && sliding_window_size > 0) {
                        valid = (q_idx - kv_idx < static_cast<uint32_t>(sliding_window_size));
                    }
                }
            }

            float score = -INFINITY;
            if (valid) {
                score = 0.0f;
                for (uint32_t d_idx = 0; d_idx < head_dim; ++d_idx) {
                    float q_val = to_float(q[q_idx * q_stride_n + qo_head_idx * q_stride_h + d_idx]);
                    float k_val = to_float(k[kv_idx * kv_stride_n + kv_head_idx * kv_stride_h + d_idx]);
                    score += q_val * k_val;
                }
                score *= sm_scale;
            }
            scores[kv_offset] = score;
            tile_max = fmaxf(tile_max, score);

            // Cache raw score to global memory (one float write per KV position)
            if (my_score_cache) {
                my_score_cache[kv_idx] = score;
            }
        }

        __syncthreads();

        // Step 2: Block-level max reduction
        reduction_buffer[tid] = tile_max;
        __syncthreads();

        for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                reduction_buffer[tid] = fmaxf(reduction_buffer[tid], reduction_buffer[tid + s]);
            }
            __syncthreads();
        }
        tile_max = reduction_buffer[0];

        // Step 3: Online softmax rescale
        float m_prev = m_shared;
        float m_new = fmaxf(m_prev, tile_max);
        float scale_prev = (m_prev > -INFINITY) ? expf(m_prev - m_new) : 0.0f;

        for (uint32_t d_idx = tid; d_idx < head_dim; d_idx += blockDim.x) {
            o_shared[d_idx] *= scale_prev;
        }

        if (tid == 0) {
            d_shared *= scale_prev;
            m_shared = m_new;
        }
        __syncthreads();

        // Step 4: Compute exp scores and sum
        float tile_sum = 0.0f;
        for (uint32_t kv_offset = tid; kv_offset < tile_size; kv_offset += blockDim.x) {
            float exp_score = expf(scores[kv_offset] - m_shared);
            scores[kv_offset] = exp_score;
            tile_sum += exp_score;
        }

        __syncthreads();

        reduction_buffer[tid] = tile_sum;
        __syncthreads();

        for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                reduction_buffer[tid] += reduction_buffer[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            d_shared += reduction_buffer[0];
        }
        __syncthreads();

        // Step 5: Accumulate weighted V
        for (uint32_t d_idx = tid; d_idx < head_dim; d_idx += blockDim.x) {
            float v_sum = 0.0f;
            for (uint32_t kv_offset = 0; kv_offset < tile_size; ++kv_offset) {
                const uint32_t kv_idx = tile_start + kv_offset;
                float v_val = to_float(v[kv_idx * kv_stride_n + kv_head_idx * kv_stride_h + d_idx]);
                v_sum += scores[kv_offset] * v_val;
            }
            o_shared[d_idx] += v_sum;
        }

        __syncthreads();
    }

    // Normalize O and write output
    float inv_d = (d_shared > 0.0f) ? 1.0f / d_shared : 0.0f;

    for (uint32_t d_idx = tid; d_idx < head_dim; d_idx += blockDim.x) {
        float out_val = o_shared[d_idx] * inv_d;
        o[q_idx * num_qo_heads * head_dim + qo_head_idx * head_dim + d_idx] = from_float<DTypeO>(out_val);
    }

    // Weight normalization pass: compute final attention weights and atomicAdd
    // to the per-kv-head output buffer.
    if (attn_weights_sum != nullptr) {
        if (my_score_cache != nullptr) {
            // Fast path: read cached raw scores (no K re-read)
            // Cost: kv_len * 4 bytes read + kv_len atomicAdds
            for (uint32_t kv_idx = tid; kv_idx < kv_len; kv_idx += blockDim.x) {
                float cached_score = my_score_cache[kv_idx];
                if (cached_score > -INFINITY) {
                    float weight = expf(cached_score - m_shared) * inv_d;
                    atomicAdd(&attn_weights_sum[kv_head_idx * kv_len + kv_idx], weight);
                }
            }
        } else {
            // Fallback: recompute Q·K from global memory (K likely L2-cached)
            // Used when score_cache is too large to allocate
            for (uint32_t kv_idx = tid; kv_idx < kv_len; kv_idx += blockDim.x) {
                bool valid = true;
                if (causal) {
                    if (head_token_positions != nullptr) {
                        int32_t key_pos = head_token_positions[kv_idx];
                        valid = (key_pos <= query_pos);
                        if (valid && sliding_window_size > 0) {
                            valid = (query_pos - key_pos < sliding_window_size);
                        }
                    } else {
                        valid = (kv_idx <= q_idx);
                        if (valid && sliding_window_size > 0) {
                            valid = (q_idx - kv_idx < static_cast<uint32_t>(sliding_window_size));
                        }
                    }
                }

                if (valid) {
                    float score = 0.0f;
                    for (uint32_t d_idx = 0; d_idx < head_dim; ++d_idx) {
                        float q_val = to_float(q[q_idx * q_stride_n + qo_head_idx * q_stride_h + d_idx]);
                        float k_val = to_float(k[kv_idx * kv_stride_n + kv_head_idx * kv_stride_h + d_idx]);
                        score += q_val * k_val;
                    }
                    score *= sm_scale;
                    float weight = expf(score - m_shared) * inv_d;
                    atomicAdd(&attn_weights_sum[kv_head_idx * kv_len + kv_idx], weight);
                }
            }
        }
    }
}

// ============================================================================
// Fused prefill kernel v2: O + weights in one pass, NO intermediate storage.
//
// Phase 1: Standard tiled attention → O + LSE (m_shared, d_shared).
// Phase 2: Re-tile over KV with Q in shared memory → atomicAdd weights.
//   - Q loaded into smem once (reusing scores/reduction_buffer space)
//   - K re-read from HBM per tile (L2 cache helps)
//   - No score_cache, no large intermediate buffer
// ============================================================================

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
__global__ void fused_prefill_attention_kernel_v2(
    const DTypeQ* __restrict__ q,
    const DTypeKV* __restrict__ k,
    const DTypeKV* __restrict__ v,
    DTypeO* __restrict__ o,
    float* __restrict__ attn_weights_sum,  // [num_kv_heads, kv_len], pre-zeroed
    const int32_t* __restrict__ token_positions,
    uint32_t token_positions_stride_h,
    uint32_t q_len,
    uint32_t kv_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t q_stride_n,
    uint32_t q_stride_h,
    uint32_t kv_stride_n,
    uint32_t kv_stride_h,
    float sm_scale,
    int32_t input_pos,
    int32_t sliding_window_size,
    bool causal) {

    const uint32_t q_idx = blockIdx.x;
    const uint32_t qo_head_idx = blockIdx.y;
    const uint32_t kv_head_idx = qo_head_idx / (num_qo_heads / num_kv_heads);
    const uint32_t tid = threadIdx.x;

    if (q_idx >= q_len || qo_head_idx >= num_qo_heads) {
        return;
    }

    extern __shared__ float smem[];
    float* scores = smem;
    float* reduction_buffer = smem + TILE_SIZE_KV;
    float* o_shared = smem + TILE_SIZE_KV + blockDim.x;

    int32_t query_pos = input_pos + q_idx;

    const int32_t* head_token_positions = token_positions;
    if (token_positions != nullptr && token_positions_stride_h > 0) {
        head_token_positions = token_positions + kv_head_idx * token_positions_stride_h;
    }

    for (uint32_t d_idx = tid; d_idx < head_dim; d_idx += blockDim.x) {
        o_shared[d_idx] = 0.0f;
    }
    __syncthreads();

    __shared__ float m_shared;
    __shared__ float d_shared;
    if (tid == 0) {
        m_shared = -INFINITY;
        d_shared = 0.0f;
    }
    __syncthreads();

    const uint32_t num_tiles = (kv_len + TILE_SIZE_KV - 1) / TILE_SIZE_KV;

    // ========================================================================
    // Phase 1: Tiled attention → O + LSE (identical to standard tiled kernel)
    // ========================================================================
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const uint32_t tile_start = tile_idx * TILE_SIZE_KV;
        const uint32_t tile_end = min(tile_start + TILE_SIZE_KV, kv_len);
        const uint32_t tile_size = tile_end - tile_start;

        // Step 1: Compute Q·K scores for this tile
        float tile_max = -INFINITY;
        for (uint32_t kv_offset = tid; kv_offset < tile_size; kv_offset += blockDim.x) {
            const uint32_t kv_idx = tile_start + kv_offset;

            bool valid = true;
            if (causal) {
                if (head_token_positions != nullptr) {
                    int32_t key_pos = head_token_positions[kv_idx];
                    valid = (key_pos <= query_pos);
                    if (valid && sliding_window_size > 0) {
                        valid = (query_pos - key_pos < sliding_window_size);
                    }
                } else {
                    valid = (kv_idx <= q_idx);
                    if (valid && sliding_window_size > 0) {
                        valid = (q_idx - kv_idx < static_cast<uint32_t>(sliding_window_size));
                    }
                }
            }

            float score = -INFINITY;
            if (valid) {
                score = 0.0f;
                for (uint32_t d_idx = 0; d_idx < head_dim; ++d_idx) {
                    float q_val = to_float(q[q_idx * q_stride_n + qo_head_idx * q_stride_h + d_idx]);
                    float k_val = to_float(k[kv_idx * kv_stride_n + kv_head_idx * kv_stride_h + d_idx]);
                    score += q_val * k_val;
                }
                score *= sm_scale;
            }
            scores[kv_offset] = score;
            tile_max = fmaxf(tile_max, score);
        }

        __syncthreads();

        // Step 2: Block-level max reduction
        reduction_buffer[tid] = tile_max;
        __syncthreads();

        for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                reduction_buffer[tid] = fmaxf(reduction_buffer[tid], reduction_buffer[tid + s]);
            }
            __syncthreads();
        }
        tile_max = reduction_buffer[0];

        // Step 3: Online softmax rescale
        float m_prev = m_shared;
        float m_new = fmaxf(m_prev, tile_max);
        float scale_prev = (m_prev > -INFINITY) ? expf(m_prev - m_new) : 0.0f;

        for (uint32_t d_idx = tid; d_idx < head_dim; d_idx += blockDim.x) {
            o_shared[d_idx] *= scale_prev;
        }

        if (tid == 0) {
            d_shared *= scale_prev;
            m_shared = m_new;
        }
        __syncthreads();

        // Step 4: Compute exp scores and sum
        float tile_sum = 0.0f;
        for (uint32_t kv_offset = tid; kv_offset < tile_size; kv_offset += blockDim.x) {
            float exp_score = expf(scores[kv_offset] - m_shared);
            scores[kv_offset] = exp_score;
            tile_sum += exp_score;
        }

        __syncthreads();

        reduction_buffer[tid] = tile_sum;
        __syncthreads();

        for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                reduction_buffer[tid] += reduction_buffer[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            d_shared += reduction_buffer[0];
        }
        __syncthreads();

        // Step 5: Accumulate weighted V
        for (uint32_t d_idx = tid; d_idx < head_dim; d_idx += blockDim.x) {
            float v_sum = 0.0f;
            for (uint32_t kv_offset = 0; kv_offset < tile_size; ++kv_offset) {
                const uint32_t kv_idx = tile_start + kv_offset;
                float v_val = to_float(v[kv_idx * kv_stride_n + kv_head_idx * kv_stride_h + d_idx]);
                v_sum += scores[kv_offset] * v_val;
            }
            o_shared[d_idx] += v_sum;
        }

        __syncthreads();
    }

    // Normalize O and write output
    float inv_d = (d_shared > 0.0f) ? 1.0f / d_shared : 0.0f;

    for (uint32_t d_idx = tid; d_idx < head_dim; d_idx += blockDim.x) {
        float out_val = o_shared[d_idx] * inv_d;
        o[q_idx * num_qo_heads * head_dim + qo_head_idx * head_dim + d_idx] = from_float<DTypeO>(out_val);
    }

    // ========================================================================
    // Phase 2: Compute attention weights via K re-read with Q in shared memory
    //
    // After Phase 1: m_shared = m_final, d_shared = d_final, inv_d computed.
    // Load Q into shared memory, then use warp-level dot products with
    // coalesced K reads for efficient bandwidth utilization.
    //
    // Each warp collaboratively computes one Q·K dot product:
    // - All 32 threads in a warp read consecutive K elements (coalesced)
    // - Warp shuffle reduction sums partial products
    // - Lane 0 does the atomicAdd
    // ========================================================================
    if (attn_weights_sum != nullptr) {
        __syncthreads();

        // Reuse smem: load Q row into the first head_dim floats
        float* q_shared = smem;
        for (uint32_t d_idx = tid; d_idx < head_dim; d_idx += blockDim.x) {
            q_shared[d_idx] = to_float(q[q_idx * q_stride_n + qo_head_idx * q_stride_h + d_idx]);
        }
        __syncthreads();

        const uint32_t WARP_SIZE = 32;
        const uint32_t warp_id = tid / WARP_SIZE;
        const uint32_t lane_id = tid % WARP_SIZE;
        const uint32_t num_warps = blockDim.x / WARP_SIZE;

        // Each warp handles one KV position at a time, striding by num_warps
        for (uint32_t kv_idx = warp_id; kv_idx < kv_len; kv_idx += num_warps) {
            bool valid = true;
            if (causal) {
                if (head_token_positions != nullptr) {
                    int32_t key_pos = head_token_positions[kv_idx];
                    valid = (key_pos <= query_pos);
                    if (valid && sliding_window_size > 0) {
                        valid = (query_pos - key_pos < sliding_window_size);
                    }
                } else {
                    valid = (kv_idx <= q_idx);
                    if (valid && sliding_window_size > 0) {
                        valid = (q_idx - kv_idx < static_cast<uint32_t>(sliding_window_size));
                    }
                }
            }

            if (valid) {
                // Warp-level dot product: each lane handles head_dim/32 elements
                // Coalesced reads: consecutive lanes read consecutive K elements
                float partial_sum = 0.0f;
                const uint32_t k_base = kv_idx * kv_stride_n + kv_head_idx * kv_stride_h;
                for (uint32_t d_idx = lane_id; d_idx < head_dim; d_idx += WARP_SIZE) {
                    partial_sum += q_shared[d_idx] * to_float(k[k_base + d_idx]);
                }

                // Warp shuffle reduction to sum partial products
                for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                    partial_sum += __shfl_down_sync(0xFFFFFFFF, partial_sum, offset);
                }

                // Lane 0 has the full dot product
                if (lane_id == 0) {
                    float score = partial_sum * sm_scale;
                    float weight = expf(score - m_shared) * inv_d;
                    atomicAdd(&attn_weights_sum[kv_head_idx * kv_len + kv_idx], weight);
                }
            }
        }
    }
}


// ============================================================================
// Launch functions
// ============================================================================

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t launch_single_prefill_attention(
    SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>& params,
    cudaStream_t stream) {
    
    if (params.q == nullptr || params.k == nullptr || 
        params.v == nullptr || params.o == nullptr) {
        return cudaErrorInvalidValue;
    }
    
    if (params.q_len == 0 || params.kv_len == 0 || 
        params.num_qo_heads == 0 || params.num_kv_heads == 0) {
        return cudaErrorInvalidValue;
    }
    
    // Try to use real FlashInfer kernels when:
    // 1. token_positions is not provided (standard causal attention)
    // 2. attention weights are not requested (FlashInfer doesn't return per-query weights)
    // 3. head_dim is supported (64, 128, or 256)
    bool can_use_flashinfer = (params.token_positions == nullptr) && 
                               (!params.return_attn_weights) &&
                               (params.head_dim == 64 || params.head_dim == 128 || params.head_dim == 256);
    
    if (can_use_flashinfer && params.causal) {
        int32_t window_left = params.sliding_window_size > 0 ? params.sliding_window_size : -1;

        cudaError_t err = dispatch_flashinfer_prefill<DTypeQ, DTypeKV, DTypeO>(
            const_cast<DTypeQ*>(params.q),
            const_cast<DTypeKV*>(params.k),
            const_cast<DTypeKV*>(params.v),
            params.o, params.lse,
            params.num_qo_heads, params.num_kv_heads,
            params.q_len, params.kv_len,
            params.q_stride_n, params.q_stride_h,
            params.kv_stride_n, params.kv_stride_h,
            params.head_dim,
            window_left,
            params.sm_scale,
            stream);
        
        if (err == cudaSuccess) {
            return err;
        }
        // Fall through to tiled implementation on error
    }

    if (can_use_flashinfer && !params.causal) {
        cudaError_t err = dispatch_flashinfer_prefill_noncausal<DTypeQ, DTypeKV, DTypeO>(
            const_cast<DTypeQ*>(params.q),
            const_cast<DTypeKV*>(params.k),
            const_cast<DTypeKV*>(params.v),
            params.o, params.lse,
            params.num_qo_heads, params.num_kv_heads,
            params.q_len, params.kv_len,
            params.q_stride_n, params.q_stride_h,
            params.kv_stride_n, params.kv_stride_h,
            params.head_dim,
            params.sm_scale,
            stream);

        if (err == cudaSuccess) {
            return err;
        }
        // Fall through to tiled implementation on error
    }

    // Use tiled implementation for:
    // - Sparse attention with token_positions
    // - When attention weights are requested
    // - Unsupported head dimensions
    // - FlashInfer dispatch errors
    const uint32_t block_size = 256;
    const uint32_t smem_size = (TILE_SIZE_KV + block_size + params.head_dim) * sizeof(float);

    dim3 grid(params.q_len, params.num_qo_heads);
    dim3 block(block_size);
    
    tiled_prefill_attention_kernel<DTypeQ, DTypeKV, DTypeO><<<grid, block, smem_size, stream>>>(
        params.q, params.k, params.v, params.o, params.lse, params.attn_weights,
        params.token_positions,
        params.token_positions_stride_h,
        params.q_len, params.kv_len, params.num_qo_heads, params.num_kv_heads, params.head_dim,
        params.q_stride_n, params.q_stride_h, params.kv_stride_n, params.kv_stride_h,
        params.sm_scale, params.input_pos, params.sliding_window_size,
        params.causal, params.return_attn_weights);
    
    return cudaGetLastError();
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t launch_batch_prefill_attention(
    BatchPrefillParams<DTypeQ, DTypeKV, DTypeO>& params,
    cudaStream_t stream) {
    
    if (params.q == nullptr || params.k == nullptr || 
        params.v == nullptr || params.o == nullptr) {
        return cudaErrorInvalidValue;
    }
    
    if (params.batch_size == 0 || params.q_len == 0 || params.kv_len == 0 || 
        params.num_qo_heads == 0 || params.num_kv_heads == 0) {
        return cudaErrorInvalidValue;
    }
    
    std::vector<int32_t> input_pos_host(params.batch_size, 0);
    if (params.input_pos != nullptr) {
        cudaError_t copy_err = cudaMemcpyAsync(
            input_pos_host.data(),
            params.input_pos,
            params.batch_size * sizeof(int32_t),
            cudaMemcpyDeviceToHost,
            stream);
        if (copy_err != cudaSuccess) {
            return copy_err;
        }
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            return sync_err;
        }
    }
    
    for (uint32_t b = 0; b < params.batch_size; ++b) {
        SinglePrefillParams<DTypeQ, DTypeKV, DTypeO> single_params;
        single_params.q = params.q + b * params.q_stride_b;
        single_params.k = params.k + b * params.kv_stride_b;
        single_params.v = params.v + b * params.kv_stride_b;
        single_params.o = params.o + b * params.q_len * params.num_qo_heads * params.head_dim;
        single_params.lse = params.lse ? 
            params.lse + b * params.q_len * params.num_qo_heads : nullptr;
        single_params.attn_weights = params.attn_weights ? 
            params.attn_weights + b * params.q_len * params.num_qo_heads * params.kv_len : nullptr;
        
        single_params.q_len = params.q_len;
        single_params.kv_len = params.kv_len;
        single_params.num_qo_heads = params.num_qo_heads;
        single_params.num_kv_heads = params.num_kv_heads;
        single_params.head_dim = params.head_dim;
        single_params.q_stride_n = params.q_stride_n;
        single_params.q_stride_h = params.q_stride_h;
        single_params.kv_stride_n = params.kv_stride_n;
        single_params.kv_stride_h = params.kv_stride_h;
        single_params.sm_scale = params.sm_scale;
        
        if (params.token_positions != nullptr) {
            single_params.token_positions = params.token_positions + b * params.token_positions_stride_b;
            single_params.token_positions_stride_h = params.token_positions_stride_h;
        } else {
            single_params.token_positions = nullptr;
            single_params.token_positions_stride_h = 0;
        }
        
        single_params.input_pos = input_pos_host[b];
        single_params.sliding_window_size = params.sliding_window_size;
        single_params.return_attn_weights = params.return_attn_weights;
        single_params.causal = params.causal;
        
        cudaError_t err = launch_single_prefill_attention(single_params, stream);
        if (err != cudaSuccess) {
            return err;
        }
    }
    
    return cudaSuccess;
}


// ============================================================================
// Fused prefill launch functions
// ============================================================================

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t launch_single_fused_prefill_attention(
    SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>& params,
    float* attn_weights_sum,  // [num_kv_heads, kv_len], pre-zeroed
    float* score_cache,       // [q_len, num_qo_heads, kv_len] temp buffer
    cudaStream_t stream) {

    if (params.q == nullptr || params.k == nullptr ||
        params.v == nullptr || params.o == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (params.q_len == 0 || params.kv_len == 0 ||
        params.num_qo_heads == 0 || params.num_kv_heads == 0) {
        return cudaErrorInvalidValue;
    }

    // Try FlashInfer fast path when weights are NOT needed
    // (caller should only use fused kernel when weights ARE needed,
    //  but handle this gracefully)
    if (attn_weights_sum == nullptr) {
        return launch_single_prefill_attention(params, stream);
    }

    const uint32_t block_size = 256;
    const uint32_t smem_size = (TILE_SIZE_KV + block_size + params.head_dim) * sizeof(float);

    dim3 grid(params.q_len, params.num_qo_heads);
    dim3 block(block_size);

    fused_prefill_attention_kernel<DTypeQ, DTypeKV, DTypeO><<<grid, block, smem_size, stream>>>(
        params.q, params.k, params.v, params.o, attn_weights_sum,
        score_cache,
        params.token_positions,
        params.token_positions_stride_h,
        params.q_len, params.kv_len, params.num_qo_heads, params.num_kv_heads, params.head_dim,
        params.q_stride_n, params.q_stride_h, params.kv_stride_n, params.kv_stride_h,
        params.sm_scale, params.input_pos, params.sliding_window_size,
        params.causal);

    return cudaGetLastError();
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t launch_batch_fused_prefill_attention(
    BatchPrefillParams<DTypeQ, DTypeKV, DTypeO>& params,
    float* attn_weights_sum,  // [batch_size, num_kv_heads, kv_len], pre-zeroed
    float* score_cache,       // [batch_size, q_len, num_qo_heads, kv_len] temp buffer
    cudaStream_t stream) {

    if (params.q == nullptr || params.k == nullptr ||
        params.v == nullptr || params.o == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (params.batch_size == 0 || params.q_len == 0 || params.kv_len == 0 ||
        params.num_qo_heads == 0 || params.num_kv_heads == 0) {
        return cudaErrorInvalidValue;
    }

    // Copy input_pos to host (same as non-fused path)
    std::vector<int32_t> input_pos_host(params.batch_size, 0);
    if (params.input_pos != nullptr) {
        cudaError_t copy_err = cudaMemcpyAsync(
            input_pos_host.data(),
            params.input_pos,
            params.batch_size * sizeof(int32_t),
            cudaMemcpyDeviceToHost,
            stream);
        if (copy_err != cudaSuccess) {
            return copy_err;
        }
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            return sync_err;
        }
    }

    const uint32_t weights_per_batch = params.num_kv_heads * params.kv_len;
    const uint32_t score_cache_per_batch = params.q_len * params.num_qo_heads * params.kv_len;

    for (uint32_t b = 0; b < params.batch_size; ++b) {
        SinglePrefillParams<DTypeQ, DTypeKV, DTypeO> single_params;
        single_params.q = params.q + b * params.q_stride_b;
        single_params.k = params.k + b * params.kv_stride_b;
        single_params.v = params.v + b * params.kv_stride_b;
        single_params.o = params.o + b * params.q_len * params.num_qo_heads * params.head_dim;
        single_params.lse = params.lse ?
            params.lse + b * params.q_len * params.num_qo_heads : nullptr;

        single_params.q_len = params.q_len;
        single_params.kv_len = params.kv_len;
        single_params.num_qo_heads = params.num_qo_heads;
        single_params.num_kv_heads = params.num_kv_heads;
        single_params.head_dim = params.head_dim;
        single_params.q_stride_n = params.q_stride_n;
        single_params.q_stride_h = params.q_stride_h;
        single_params.kv_stride_n = params.kv_stride_n;
        single_params.kv_stride_h = params.kv_stride_h;
        single_params.sm_scale = params.sm_scale;

        if (params.token_positions != nullptr) {
            single_params.token_positions = params.token_positions + b * params.token_positions_stride_b;
            single_params.token_positions_stride_h = params.token_positions_stride_h;
        } else {
            single_params.token_positions = nullptr;
            single_params.token_positions_stride_h = 0;
        }

        single_params.input_pos = input_pos_host[b];
        single_params.sliding_window_size = params.sliding_window_size;
        single_params.return_attn_weights = false;  // handled by fused kernel
        single_params.causal = params.causal;

        float* batch_weights = attn_weights_sum ? attn_weights_sum + b * weights_per_batch : nullptr;
        float* batch_score_cache = score_cache ? score_cache + b * score_cache_per_batch : nullptr;

        cudaError_t err = launch_single_fused_prefill_attention(single_params, batch_weights, batch_score_cache, stream);
        if (err != cudaSuccess) {
            return err;
        }
    }

    return cudaSuccess;
}


// ============================================================================
// Fused prefill v2 launch functions (no score_cache)
// ============================================================================

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t launch_single_fused_prefill_attention_v2(
    SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>& params,
    float* attn_weights_sum,  // [num_kv_heads, kv_len], pre-zeroed
    cudaStream_t stream) {

    if (params.q == nullptr || params.k == nullptr ||
        params.v == nullptr || params.o == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (params.q_len == 0 || params.kv_len == 0 ||
        params.num_qo_heads == 0 || params.num_kv_heads == 0) {
        return cudaErrorInvalidValue;
    }

    if (attn_weights_sum == nullptr) {
        return launch_single_prefill_attention(params, stream);
    }

    const uint32_t block_size = 256;
    const uint32_t smem_size = (TILE_SIZE_KV + block_size + params.head_dim) * sizeof(float);

    dim3 grid(params.q_len, params.num_qo_heads);
    dim3 block(block_size);

    fused_prefill_attention_kernel_v2<DTypeQ, DTypeKV, DTypeO><<<grid, block, smem_size, stream>>>(
        params.q, params.k, params.v, params.o, attn_weights_sum,
        params.token_positions,
        params.token_positions_stride_h,
        params.q_len, params.kv_len, params.num_qo_heads, params.num_kv_heads, params.head_dim,
        params.q_stride_n, params.q_stride_h, params.kv_stride_n, params.kv_stride_h,
        params.sm_scale, params.input_pos, params.sliding_window_size,
        params.causal);

    return cudaGetLastError();
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t launch_batch_fused_prefill_attention_v2(
    BatchPrefillParams<DTypeQ, DTypeKV, DTypeO>& params,
    float* attn_weights_sum,  // [batch_size, num_kv_heads, kv_len], pre-zeroed
    cudaStream_t stream) {

    if (params.q == nullptr || params.k == nullptr ||
        params.v == nullptr || params.o == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (params.batch_size == 0 || params.q_len == 0 || params.kv_len == 0 ||
        params.num_qo_heads == 0 || params.num_kv_heads == 0) {
        return cudaErrorInvalidValue;
    }

    std::vector<int32_t> input_pos_host(params.batch_size, 0);
    if (params.input_pos != nullptr) {
        cudaError_t copy_err = cudaMemcpyAsync(
            input_pos_host.data(),
            params.input_pos,
            params.batch_size * sizeof(int32_t),
            cudaMemcpyDeviceToHost,
            stream);
        if (copy_err != cudaSuccess) return copy_err;
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) return sync_err;
    }

    const uint32_t weights_per_batch = params.num_kv_heads * params.kv_len;

    for (uint32_t b = 0; b < params.batch_size; ++b) {
        SinglePrefillParams<DTypeQ, DTypeKV, DTypeO> single_params;
        single_params.q = params.q + b * params.q_stride_b;
        single_params.k = params.k + b * params.kv_stride_b;
        single_params.v = params.v + b * params.kv_stride_b;
        single_params.o = params.o + b * params.q_len * params.num_qo_heads * params.head_dim;
        single_params.lse = params.lse ?
            params.lse + b * params.q_len * params.num_qo_heads : nullptr;

        single_params.q_len = params.q_len;
        single_params.kv_len = params.kv_len;
        single_params.num_qo_heads = params.num_qo_heads;
        single_params.num_kv_heads = params.num_kv_heads;
        single_params.head_dim = params.head_dim;
        single_params.q_stride_n = params.q_stride_n;
        single_params.q_stride_h = params.q_stride_h;
        single_params.kv_stride_n = params.kv_stride_n;
        single_params.kv_stride_h = params.kv_stride_h;
        single_params.sm_scale = params.sm_scale;

        if (params.token_positions != nullptr) {
            single_params.token_positions = params.token_positions + b * params.token_positions_stride_b;
            single_params.token_positions_stride_h = params.token_positions_stride_h;
        } else {
            single_params.token_positions = nullptr;
            single_params.token_positions_stride_h = 0;
        }

        single_params.input_pos = input_pos_host[b];
        single_params.sliding_window_size = params.sliding_window_size;
        single_params.return_attn_weights = false;
        single_params.causal = params.causal;

        float* batch_weights = attn_weights_sum ? attn_weights_sum + b * weights_per_batch : nullptr;

        cudaError_t err = launch_single_fused_prefill_attention_v2(single_params, batch_weights, stream);
        if (err != cudaSuccess) return err;
    }

    return cudaSuccess;
}


// Explicit template instantiations
template cudaError_t launch_single_prefill_attention<half, half, half>(
    SinglePrefillParams<half, half, half>& params, cudaStream_t stream);
template cudaError_t launch_single_prefill_attention<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
    SinglePrefillParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>& params, cudaStream_t stream);
template cudaError_t launch_single_prefill_attention<float, float, float>(
    SinglePrefillParams<float, float, float>& params, cudaStream_t stream);

template cudaError_t launch_batch_prefill_attention<half, half, half>(
    BatchPrefillParams<half, half, half>& params, cudaStream_t stream);
template cudaError_t launch_batch_prefill_attention<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
    BatchPrefillParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>& params, cudaStream_t stream);
template cudaError_t launch_batch_prefill_attention<float, float, float>(
    BatchPrefillParams<float, float, float>& params, cudaStream_t stream);

template cudaError_t launch_single_fused_prefill_attention<half, half, half>(
    SinglePrefillParams<half, half, half>& params, float* attn_weights_sum, float* score_cache, cudaStream_t stream);
template cudaError_t launch_single_fused_prefill_attention<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
    SinglePrefillParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>& params, float* attn_weights_sum, float* score_cache, cudaStream_t stream);
template cudaError_t launch_single_fused_prefill_attention<float, float, float>(
    SinglePrefillParams<float, float, float>& params, float* attn_weights_sum, float* score_cache, cudaStream_t stream);

template cudaError_t launch_batch_fused_prefill_attention<half, half, half>(
    BatchPrefillParams<half, half, half>& params, float* attn_weights_sum, float* score_cache, cudaStream_t stream);
template cudaError_t launch_batch_fused_prefill_attention<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
    BatchPrefillParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>& params, float* attn_weights_sum, float* score_cache, cudaStream_t stream);
template cudaError_t launch_batch_fused_prefill_attention<float, float, float>(
    BatchPrefillParams<float, float, float>& params, float* attn_weights_sum, float* score_cache, cudaStream_t stream);

template cudaError_t launch_single_fused_prefill_attention_v2<half, half, half>(
    SinglePrefillParams<half, half, half>& params, float* attn_weights_sum, cudaStream_t stream);
template cudaError_t launch_single_fused_prefill_attention_v2<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
    SinglePrefillParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>& params, float* attn_weights_sum, cudaStream_t stream);
template cudaError_t launch_single_fused_prefill_attention_v2<float, float, float>(
    SinglePrefillParams<float, float, float>& params, float* attn_weights_sum, cudaStream_t stream);

template cudaError_t launch_batch_fused_prefill_attention_v2<half, half, half>(
    BatchPrefillParams<half, half, half>& params, float* attn_weights_sum, cudaStream_t stream);
template cudaError_t launch_batch_fused_prefill_attention_v2<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
    BatchPrefillParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>& params, float* attn_weights_sum, cudaStream_t stream);
template cudaError_t launch_batch_fused_prefill_attention_v2<float, float, float>(
    BatchPrefillParams<float, float, float>& params, float* attn_weights_sum, cudaStream_t stream);

}  // namespace kernels
}  // namespace keys_values
