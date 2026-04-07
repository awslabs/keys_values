/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Decode SDPA kernel implementation.
 *
 * This file provides two implementations:
 * 1. Real FlashInfer kernels for standard causal attention (token_positions=None)
 * 2. Tiled reference implementation for sparse attention with token_positions
 */

#include "sdpa_decode.cuh"
#include <algorithm>
#include <vector>

// Include FlashInfer headers for real kernel dispatch
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/variants.cuh>

// Explicit includes for FlashInfer primitives used in optimized tiled kernel
#include <flashinfer/vec_dtypes.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/attention/state.cuh>

namespace keys_values {
namespace kernels {

// ============================================================================
// FlashInfer kernel dispatch for standard causal attention
// ============================================================================

/**
 * @brief Dispatch to real FlashInfer decode kernel
 *
 * This function calls FlashInfer's SingleDecodeWithKVCacheDispatched with
 * the appropriate template parameters based on head dimension.
 */
template <uint32_t HEAD_DIM, typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t dispatch_flashinfer_decode_impl(
    DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o, float* lse,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t kv_len,
    uint32_t q_stride_h,
    uint32_t kv_stride_n, uint32_t kv_stride_h,
    int32_t window_left,
    float sm_scale,
    cudaStream_t stream) {

    using namespace flashinfer;

    // Create FlashInfer params using the correct constructor
    // SingleDecodeParams(q, k, v, o, maybe_alibi_slopes, seq_len, num_qo_heads, num_kv_heads,
    //                    kv_layout, head_dim, window_left, logits_soft_cap, sm_scale, rope_scale, rope_theta)
    flashinfer::SingleDecodeParams<DTypeQ, DTypeKV, DTypeO> fi_params(
        q, k, v, o,
        nullptr,  // maybe_alibi_slopes
        kv_len,
        num_qo_heads, num_kv_heads,
        QKVLayout::kNHD,  // KV layout
        HEAD_DIM,
        window_left,
        0.0f,  // logits_soft_cap
        sm_scale,
        1.0f,  // rope_scale
        10000.0f  // rope_theta
    );
    fi_params.lse = lse;

    // Use DefaultAttention variant (no custom mask, no sliding window in variant, no soft cap, no alibi)
    // Sliding window is handled via window_left parameter
    using AttentionVariant = DefaultAttention<false, false, false, false>;

    // Dispatch to FlashInfer kernel
    return SingleDecodeWithKVCacheDispatched<HEAD_DIM, PosEncodingMode::kNone, AttentionVariant>(
        fi_params, nullptr, stream);
}

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t dispatch_flashinfer_decode(
    DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o, float* lse,
    uint32_t num_qo_heads, uint32_t num_kv_heads,
    uint32_t kv_len,
    uint32_t q_stride_h,
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
            return dispatch_flashinfer_decode_impl<64>(
                q, k, v, o, lse, num_qo_heads, num_kv_heads, kv_len,
                q_stride_h, kv_stride_n, kv_stride_h, window_left, sm_scale, stream);
        } else if (head_dim == 128) {
            return dispatch_flashinfer_decode_impl<128>(
                q, k, v, o, lse, num_qo_heads, num_kv_heads, kv_len,
                q_stride_h, kv_stride_n, kv_stride_h, window_left, sm_scale, stream);
        } else if (head_dim == 256) {
            return dispatch_flashinfer_decode_impl<256>(
                q, k, v, o, lse, num_qo_heads, num_kv_heads, kv_len,
                q_stride_h, kv_stride_n, kv_stride_h, window_left, sm_scale, stream);
        } else {
            // Unsupported head dimension - return error to trigger fallback
            return cudaErrorNotSupported;
        }
    }
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


/**
 * @brief Tiled decode attention kernel using online softmax algorithm
 */
template <typename DTypeQ, typename DTypeKV, typename DTypeO>
__global__ void tiled_decode_attention_kernel(
    const DTypeQ* __restrict__ q,
    const DTypeKV* __restrict__ k,
    const DTypeKV* __restrict__ v,
    DTypeO* __restrict__ o,
    float* __restrict__ attn_weights,
    const int32_t* __restrict__ token_positions,
    uint32_t token_positions_stride_h,
    uint32_t kv_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t q_stride_h,
    uint32_t kv_stride_n,
    uint32_t kv_stride_h,
    float sm_scale,
    int32_t input_pos,
    int32_t sliding_window_size,
    bool causal,
    bool return_weights) {

    const uint32_t qo_head_idx = blockIdx.x;
    const uint32_t kv_head_idx = qo_head_idx / (num_qo_heads / num_kv_heads);
    const uint32_t tid = threadIdx.x;

    if (qo_head_idx >= num_qo_heads) {
        return;
    }

    extern __shared__ float smem[];
    float* scores = smem;
    float* reduction_buffer = smem + TILE_SIZE_KV;
    float* o_shared = smem + TILE_SIZE_KV + blockDim.x;

    int32_t query_pos = input_pos;

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
                    valid = (static_cast<int32_t>(kv_idx) <= query_pos);
                    if (valid && sliding_window_size > 0) {
                        valid = (query_pos - static_cast<int32_t>(kv_idx) < sliding_window_size);
                    }
                }
            }

            float score = -INFINITY;
            if (valid) {
                score = 0.0f;
                for (uint32_t d_idx = 0; d_idx < head_dim; ++d_idx) {
                    float q_val = to_float(q[qo_head_idx * q_stride_h + d_idx]);
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
        o[qo_head_idx * head_dim + d_idx] = from_float<DTypeO>(out_val);
    }

    // Store attention weights if requested
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
                    valid = (static_cast<int32_t>(kv_idx) <= query_pos);
                    if (valid && sliding_window_size > 0) {
                        valid = (query_pos - static_cast<int32_t>(kv_idx) < sliding_window_size);
                    }
                }
            }

            float weight = 0.0f;
            if (valid) {
                float score = 0.0f;
                for (uint32_t d_idx = 0; d_idx < head_dim; ++d_idx) {
                    float q_val = to_float(q[qo_head_idx * q_stride_h + d_idx]);
                    float k_val = to_float(k[kv_idx * kv_stride_n + kv_head_idx * kv_stride_h + d_idx]);
                    score += q_val * k_val;
                }
                score *= sm_scale;
                weight = expf(score - m_shared) * inv_d;
            }

            // For decode, aggregate across query heads that share the same KV head
            atomicAdd(&attn_weights[kv_head_idx * kv_len + kv_idx], weight);
        }
    }
}

// ============================================================================
// Batched tiled decode attention kernel
// ============================================================================

/**
 * @brief Batched tiled decode attention kernel using online softmax algorithm
 *
 * Processes all batch items in a single kernel launch using a 2D grid:
 *   grid(batch_size, num_qo_heads), block(256)
 *
 * Each block handles one (batch_item, qo_head) pair. input_pos is read
 * directly from device memory, eliminating the need for device-to-host
 * copy and cudaStreamSynchronize.
 */
template <typename DTypeQ, typename DTypeKV, typename DTypeO>
__global__ void batched_tiled_decode_attention_kernel(
    const DTypeQ* __restrict__ q,
    const DTypeKV* __restrict__ k,
    const DTypeKV* __restrict__ v,
    DTypeO* __restrict__ o,
    float* __restrict__ attn_weights,
    const int32_t* __restrict__ token_positions,
    uint32_t token_positions_stride_b,
    uint32_t token_positions_stride_h,
    uint32_t kv_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t q_stride_b,
    uint32_t q_stride_h,
    uint32_t kv_stride_b,
    uint32_t kv_stride_n,
    uint32_t kv_stride_h,
    float sm_scale,
    const int32_t* __restrict__ input_pos,
    int32_t sliding_window_size,
    bool causal,
    bool return_weights) {

    const uint32_t batch_idx = blockIdx.x;
    const uint32_t qo_head_idx = blockIdx.y;
    const uint32_t kv_head_idx = qo_head_idx / (num_qo_heads / num_kv_heads);
    const uint32_t tid = threadIdx.x;

    if (qo_head_idx >= num_qo_heads) {
        return;
    }

    // Compute per-batch pointer offsets
    const DTypeQ* batch_q = q + batch_idx * q_stride_b;
    const DTypeKV* batch_k = k + batch_idx * kv_stride_b;
    const DTypeKV* batch_v = v + batch_idx * kv_stride_b;
    DTypeO* batch_o = o + batch_idx * num_qo_heads * head_dim;
    float* batch_attn_weights = attn_weights ?
        attn_weights + batch_idx * num_kv_heads * kv_len : nullptr;

    const int32_t* batch_token_positions = nullptr;
    if (token_positions != nullptr) {
        batch_token_positions = token_positions + batch_idx * token_positions_stride_b;
    }

    extern __shared__ float smem[];
    float* scores = smem;
    float* reduction_buffer = smem + TILE_SIZE_KV;
    float* o_shared = smem + TILE_SIZE_KV + blockDim.x;

    int32_t query_pos = (input_pos != nullptr) ? input_pos[batch_idx] : 0;

    const int32_t* head_token_positions = batch_token_positions;
    if (batch_token_positions != nullptr && token_positions_stride_h > 0) {
        head_token_positions = batch_token_positions + kv_head_idx * token_positions_stride_h;
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
                    valid = (static_cast<int32_t>(kv_idx) <= query_pos);
                    if (valid && sliding_window_size > 0) {
                        valid = (query_pos - static_cast<int32_t>(kv_idx) < sliding_window_size);
                    }
                }
            }

            float score = -INFINITY;
            if (valid) {
                score = 0.0f;
                for (uint32_t d_idx = 0; d_idx < head_dim; ++d_idx) {
                    float q_val = to_float(batch_q[qo_head_idx * q_stride_h + d_idx]);
                    float k_val = to_float(batch_k[kv_idx * kv_stride_n + kv_head_idx * kv_stride_h + d_idx]);
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
                float v_val = to_float(batch_v[kv_idx * kv_stride_n + kv_head_idx * kv_stride_h + d_idx]);
                v_sum += scores[kv_offset] * v_val;
            }
            o_shared[d_idx] += v_sum;
        }

        __syncthreads();
    }

    float inv_d = (d_shared > 0.0f) ? 1.0f / d_shared : 0.0f;

    for (uint32_t d_idx = tid; d_idx < head_dim; d_idx += blockDim.x) {
        float out_val = o_shared[d_idx] * inv_d;
        batch_o[qo_head_idx * head_dim + d_idx] = from_float<DTypeO>(out_val);
    }

    // Store attention weights if requested
    if (return_weights && batch_attn_weights != nullptr) {
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
                    valid = (static_cast<int32_t>(kv_idx) <= query_pos);
                    if (valid && sliding_window_size > 0) {
                        valid = (query_pos - static_cast<int32_t>(kv_idx) < sliding_window_size);
                    }
                }
            }

            float weight = 0.0f;
            if (valid) {
                float score = 0.0f;
                for (uint32_t d_idx = 0; d_idx < head_dim; ++d_idx) {
                    float q_val = to_float(batch_q[qo_head_idx * q_stride_h + d_idx]);
                    float k_val = to_float(batch_k[kv_idx * kv_stride_n + kv_head_idx * kv_stride_h + d_idx]);
                    score += q_val * k_val;
                }
                score *= sm_scale;
                weight = expf(score - m_shared) * inv_d;
            }

            // For decode, aggregate across query heads that share the same KV head
            atomicAdd(&batch_attn_weights[kv_head_idx * kv_len + kv_idx], weight);
        }
    }
}

// ============================================================================
// Optimized batched decode kernel using FlashInfer primitives
// ============================================================================

/**
 * @brief Optimized batched decode attention kernel following FlashInfer's
 * thread model with vectorized loads, warp reductions, and online softmax.
 *
 * Grid:  dim3(batch_size, num_qo_heads)
 * Block: dim3(BDX, 1, BDZ)
 *
 * BDX threads span head_dim (each owns VEC_SIZE elements, BDX * VEC_SIZE = HEAD_DIM).
 * BDZ threads tile over KV positions within each tile.
 *
 * Each block handles one (batch, qo_head) pair. Multiple blocks sharing
 * the same KV head (GQA) rely on L2 cache for K/V data sharing rather
 * than shared memory GQA, maximizing GPU occupancy (num_qo_heads blocks
 * per batch item instead of num_kv_heads).
 *
 * Uses a single shared memory buffer for K then V loading (halves smem).
 */
template <uint32_t HEAD_DIM, uint32_t VEC_SIZE, uint32_t BDX, uint32_t BDZ,
          uint32_t TILE_PER_TZ,
          typename DTypeQ, typename DTypeKV, typename DTypeO>
__global__ void optimized_batched_decode_kernel(
    const DTypeQ* __restrict__ q,
    const DTypeKV* __restrict__ k,
    const DTypeKV* __restrict__ v,
    DTypeO* __restrict__ o,
    float* __restrict__ attn_weights,
    float* __restrict__ logits_tmp,
    const int32_t* __restrict__ token_positions,
    uint32_t token_positions_stride_b,
    uint32_t token_positions_stride_h,
    uint32_t kv_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t q_stride_b,
    uint32_t q_stride_h,
    uint32_t kv_stride_b,
    uint32_t kv_stride_n,
    uint32_t kv_stride_h,
    float sm_scale,
    const int32_t* __restrict__ input_pos,
    int32_t sliding_window_size,
    bool causal,
    bool return_weights) {

    using namespace flashinfer;
    using namespace flashinfer::math;

    constexpr uint32_t TILE_SIZE = BDZ * TILE_PER_TZ;

    const uint32_t batch_idx = blockIdx.x;
    const uint32_t qo_head_idx = blockIdx.y;
    const uint32_t kv_head_idx = qo_head_idx / (num_qo_heads / num_kv_heads);
    const uint32_t tx = threadIdx.x;  // head dim (0..BDX-1)
    const uint32_t tz = threadIdx.z;  // KV tile chunk (0..BDZ-1)

    if (qo_head_idx >= num_qo_heads) return;

    // Single shared memory buffer for K then V (halves smem requirement)
    extern __shared__ uint8_t smem_raw[];
    DTypeKV* kv_smem = (DTypeKV*)smem_raw;

    // Per-batch pointer offsets
    const DTypeQ* batch_q = q + batch_idx * q_stride_b;
    const DTypeKV* batch_k = k + batch_idx * kv_stride_b;
    const DTypeKV* batch_v = v + batch_idx * kv_stride_b;
    DTypeO* batch_o = o + batch_idx * num_qo_heads * HEAD_DIM;
    float* batch_attn_weights = attn_weights ?
        attn_weights + batch_idx * num_kv_heads * kv_len : nullptr;

    // ---- Phase 0: Load Q into registers (done once) ----
    vec_t<float, VEC_SIZE> q_vec;
    q_vec.cast_load(batch_q + qo_head_idx * q_stride_h + tx * VEC_SIZE);

    int32_t query_pos = (input_pos != nullptr) ? input_pos[batch_idx] : 0;

    const int32_t* batch_token_positions = nullptr;
    if (token_positions != nullptr) {
        batch_token_positions = token_positions + batch_idx * token_positions_stride_b;
    }
    const int32_t* head_token_positions = batch_token_positions;
    if (batch_token_positions != nullptr && token_positions_stride_h > 0) {
        head_token_positions = batch_token_positions + kv_head_idx * token_positions_stride_h;
    }

    float sm_scale_log2 = sm_scale * log2e;

    // ---- Phase 1: Tiled attention with online softmax ----
    state_t<VEC_SIZE> st;
    const uint32_t num_tiles = (kv_len + TILE_SIZE - 1) / TILE_SIZE;

    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        const uint32_t tile_start = tile_idx * TILE_SIZE;
        float s[TILE_PER_TZ];

        // -- Step 1: Cooperative K load into shared memory --
        // Each tz loads its TILE_PER_TZ rows of K
        #pragma unroll
        for (uint32_t j = 0; j < TILE_PER_TZ; ++j) {
            const uint32_t smem_row = tz * TILE_PER_TZ + j;
            const uint32_t kv_idx = tile_start + smem_row;
            if (kv_idx < kv_len) {
                vec_t<DTypeKV, VEC_SIZE> k_load;
                k_load.load(batch_k + kv_idx * kv_stride_n +
                            kv_head_idx * kv_stride_h + tx * VEC_SIZE);
                k_load.store(kv_smem + smem_row * HEAD_DIM + tx * VEC_SIZE);
            } else {
                vec_t<DTypeKV, VEC_SIZE> zero;
                zero.fill(DTypeKV(0));
                zero.store(kv_smem + smem_row * HEAD_DIM + tx * VEC_SIZE);
            }
        }
        __syncthreads();

        // -- Step 2: Compute Q·K scores for TILE_PER_TZ positions --
        float m_prev = st.m;
        #pragma unroll
        for (uint32_t j = 0; j < TILE_PER_TZ; ++j) {
            const uint32_t smem_row = tz * TILE_PER_TZ + j;
            const uint32_t kv_idx = tile_start + smem_row;

            vec_t<float, VEC_SIZE> k_vec;
            k_vec.cast_load(kv_smem + smem_row * HEAD_DIM + tx * VEC_SIZE);

            s[j] = 0.f;
            #pragma unroll
            for (uint32_t i = 0; i < VEC_SIZE; ++i) {
                s[j] += q_vec[i] * k_vec[i];
            }

            // Warp reduction across BDX threads (butterfly shuffle)
            #pragma unroll
            for (uint32_t offset = BDX / 2; offset > 0; offset /= 2) {
                s[j] += shfl_xor_sync(s[j], offset);
            }

            // Causal mask
            bool valid = (kv_idx < kv_len);
            if (valid && causal) {
                if (head_token_positions != nullptr) {
                    int32_t key_pos = head_token_positions[kv_idx];
                    valid = (key_pos <= query_pos);
                    if (valid && sliding_window_size > 0) {
                        valid = (query_pos - key_pos < sliding_window_size);
                    }
                } else {
                    valid = (static_cast<int32_t>(kv_idx) <= query_pos);
                    if (valid && sliding_window_size > 0) {
                        valid = (query_pos - static_cast<int32_t>(kv_idx) < sliding_window_size);
                    }
                }
            }

            s[j] = valid ? s[j] * sm_scale_log2 : -inf;

            // Cache logit for attention weights pass (avoid Q·K recomputation)
            if (return_weights && logits_tmp != nullptr && tx == 0 && kv_idx < kv_len) {
                logits_tmp[batch_idx * num_qo_heads * kv_len +
                           qo_head_idx * kv_len + kv_idx] = s[j];
            }

            st.m = max(st.m, s[j]);
        }

        // -- Step 3: Online softmax rescale --
        float o_scale = ptx_exp2(m_prev - st.m);
        st.d *= o_scale;
        #pragma unroll
        for (uint32_t j = 0; j < TILE_PER_TZ; ++j) {
            s[j] = ptx_exp2(s[j] - st.m);
            st.d += s[j];
        }
        #pragma unroll
        for (uint32_t i = 0; i < VEC_SIZE; ++i) {
            st.o[i] *= o_scale;
        }

        // -- Step 4: Load V into shared memory (reusing K buffer) --
        __syncthreads();  // ensure all K reads from smem are done
        #pragma unroll
        for (uint32_t j = 0; j < TILE_PER_TZ; ++j) {
            const uint32_t smem_row = tz * TILE_PER_TZ + j;
            const uint32_t kv_idx = tile_start + smem_row;
            if (kv_idx < kv_len) {
                vec_t<DTypeKV, VEC_SIZE> v_load;
                v_load.load(batch_v + kv_idx * kv_stride_n +
                            kv_head_idx * kv_stride_h + tx * VEC_SIZE);
                v_load.store(kv_smem + smem_row * HEAD_DIM + tx * VEC_SIZE);
            } else {
                vec_t<DTypeKV, VEC_SIZE> zero;
                zero.fill(DTypeKV(0));
                zero.store(kv_smem + smem_row * HEAD_DIM + tx * VEC_SIZE);
            }
        }
        __syncthreads();

        // -- Step 5: Accumulate weighted V --
        #pragma unroll
        for (uint32_t j = 0; j < TILE_PER_TZ; ++j) {
            const uint32_t smem_row = tz * TILE_PER_TZ + j;
            vec_t<float, VEC_SIZE> v_vec;
            v_vec.cast_load(kv_smem + smem_row * HEAD_DIM + tx * VEC_SIZE);
            #pragma unroll
            for (uint32_t i = 0; i < VEC_SIZE; ++i) {
                st.o[i] += s[j] * v_vec[i];
            }
        }

        __syncthreads();  // ensure smem can be reused next tile
    }

    // ---- Phase 2: Cross-BDZ sync (merge partial states) ----
    if constexpr (BDZ > 1) {
        // Reuse kv_smem as sync buffers (main loop is done)
        float* smem_sync = (float*)smem_raw;
        float* smem_md = smem_sync + BDZ * HEAD_DIM;

        st.o.store(smem_sync + tz * HEAD_DIM + tx * VEC_SIZE);
        smem_md[tz * 2] = st.m;
        smem_md[tz * 2 + 1] = st.d;
        __syncthreads();

        st.init();
        #pragma unroll
        for (uint32_t z = 0; z < BDZ; ++z) {
            float mz = smem_md[z * 2];
            float dz = smem_md[z * 2 + 1];
            vec_t<float, VEC_SIZE> oz;
            oz.load(smem_sync + z * HEAD_DIM + tx * VEC_SIZE);
            st.merge(oz, mz, dz);
        }
    }

    // ---- Phase 3: Normalize and write output ----
    st.normalize();
    if (tz == 0) {
        st.o.cast_store(batch_o + qo_head_idx * HEAD_DIM + tx * VEC_SIZE);
    }

    // ---- Phase 4: Attention weights ----
    if (return_weights && batch_attn_weights != nullptr) {
        float m_final = st.m;
        float d_rcp = ptx_rcp(st.d);

        if (logits_tmp != nullptr) {
            // Fast path: read cached logits (no Q·K recomputation needed).
            // Only tx=0 participates — logits are scalar per KV position.
            if (tx == 0) {
                const float* my_logits = logits_tmp +
                    batch_idx * num_qo_heads * kv_len +
                    qo_head_idx * kv_len;
                for (uint32_t kv_idx = tz; kv_idx < kv_len; kv_idx += BDZ) {
                    float logit = my_logits[kv_idx];
                    float weight = ptx_exp2(logit - m_final) * d_rcp;
                    atomicAdd(&batch_attn_weights[kv_head_idx * kv_len + kv_idx], weight);
                }
            }
        } else {
            // Slow path: recompute Q·K from global memory (Q still in registers).
            for (uint32_t kv_idx = tz; kv_idx < kv_len; kv_idx += BDZ) {
                vec_t<float, VEC_SIZE> k_vec;
                k_vec.cast_load(batch_k + kv_idx * kv_stride_n +
                                kv_head_idx * kv_stride_h + tx * VEC_SIZE);

                float score = 0.f;
                #pragma unroll
                for (uint32_t i = 0; i < VEC_SIZE; ++i) {
                    score += q_vec[i] * k_vec[i];
                }
                #pragma unroll
                for (uint32_t offset = BDX / 2; offset > 0; offset /= 2) {
                    score += shfl_xor_sync(score, offset);
                }

                if (tx == 0) {
                    bool valid = true;
                    if (causal) {
                        if (head_token_positions != nullptr) {
                            int32_t key_pos = head_token_positions[kv_idx];
                            valid = (key_pos <= query_pos);
                            if (valid && sliding_window_size > 0) {
                                valid = (query_pos - key_pos < sliding_window_size);
                            }
                        } else {
                            valid = (static_cast<int32_t>(kv_idx) <= query_pos);
                            if (valid && sliding_window_size > 0) {
                                valid = (query_pos - static_cast<int32_t>(kv_idx) < sliding_window_size);
                            }
                        }
                    }

                    float weight = 0.f;
                    if (valid) {
                        weight = ptx_exp2(score * sm_scale_log2 - m_final) * d_rcp;
                    }
                    atomicAdd(&batch_attn_weights[kv_head_idx * kv_len + kv_idx], weight);
                }
            }
        }
    }
}

// ============================================================================
// Dispatch helpers for optimized decode kernel
// ============================================================================

template <uint32_t HEAD_DIM, uint32_t VEC_SIZE, uint32_t BDX, uint32_t BDZ,
          uint32_t TILE_PER_TZ, typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t launch_optimized_decode(
    BatchDecodeParams<DTypeQ, DTypeKV, DTypeO>& params,
    cudaStream_t stream) {

    constexpr uint32_t TILE_SIZE = BDZ * TILE_PER_TZ;

    // Single K/V buffer (loaded sequentially: K then V)
    uint32_t kv_smem_size = TILE_SIZE * HEAD_DIM * sizeof(DTypeKV);
    // Sync smem for cross-BDZ merge (reuses kv_smem after main loop)
    uint32_t sync_smem_size = BDZ * HEAD_DIM * sizeof(float) +
                               BDZ * 2 * sizeof(float);
    uint32_t smem_size = max(kv_smem_size, sync_smem_size);

    dim3 grid(params.batch_size, params.num_qo_heads);
    dim3 block(BDX, 1, BDZ);

    // Initialize attention weights to zero if requested
    if (params.return_attn_weights && params.attn_weights != nullptr) {
        cudaMemsetAsync(params.attn_weights, 0,
            params.batch_size * params.num_kv_heads * params.kv_len * sizeof(float),
            stream);
    }

    optimized_batched_decode_kernel<HEAD_DIM, VEC_SIZE, BDX, BDZ, TILE_PER_TZ,
                                    DTypeQ, DTypeKV, DTypeO>
        <<<grid, block, smem_size, stream>>>(
        params.q, params.k, params.v, params.o, params.attn_weights,
        params.logits_tmp, params.token_positions,
        params.token_positions_stride_b,
        params.token_positions_stride_h,
        params.kv_len, params.num_qo_heads, params.num_kv_heads,
        params.q_stride_b, params.q_stride_h,
        params.kv_stride_b, params.kv_stride_n, params.kv_stride_h,
        params.sm_scale, params.input_pos, params.sliding_window_size,
        params.causal, params.return_attn_weights);

    return cudaGetLastError();
}

/**
 * @brief Dispatch optimized decode kernel based on head_dim.
 *
 * Grid is (batch_size, num_qo_heads) — each block handles one QO head.
 * GQA heads sharing the same KV head rely on L2 cache.
 * All configs use 128 threads per block and TILE_PER_TZ=16.
 *
 * Returns cudaErrorNotSupported if the configuration is not supported,
 * allowing fallback to the generic batched tiled kernel.
 */
template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t dispatch_optimized_batched_decode(
    BatchDecodeParams<DTypeQ, DTypeKV, DTypeO>& params,
    cudaStream_t stream) {

    if (params.head_dim == 64) {
        // BDX=8, BDZ=16: 128 threads. TILE_SIZE=256, smem=32KB
        return launch_optimized_decode<64, 8, 8, 16, 16>(params, stream);
    } else if (params.head_dim == 128) {
        // BDX=16, BDZ=8: 128 threads. TILE_SIZE=128, smem=32KB
        return launch_optimized_decode<128, 8, 16, 8, 16>(params, stream);
    } else if (params.head_dim == 256) {
        // BDX=32, BDZ=4: 128 threads. TILE_SIZE=64, smem=32KB
        return launch_optimized_decode<256, 8, 32, 4, 16>(params, stream);
    }

    return cudaErrorNotSupported;
}

// ============================================================================
// Launch functions
// ============================================================================

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t launch_single_decode_attention(
    SingleDecodeParams<DTypeQ, DTypeKV, DTypeO>& params,
    cudaStream_t stream) {

    if (params.q == nullptr || params.k == nullptr ||
        params.v == nullptr || params.o == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (params.kv_len == 0 || params.num_qo_heads == 0 || params.num_kv_heads == 0) {
        return cudaErrorInvalidValue;
    }

    // Try to use real FlashInfer kernels when:
    // 1. token_positions is not provided (standard causal attention)
    // 2. attention weights are not requested (FlashInfer doesn't return weights)
    // 3. head_dim is supported (64, 128, or 256)
    bool can_use_flashinfer = (params.token_positions == nullptr) &&
                               (!params.return_attn_weights) &&
                               (params.head_dim == 64 || params.head_dim == 128 || params.head_dim == 256);

    if (can_use_flashinfer && params.causal) {
        int32_t window_left = params.sliding_window_size > 0 ? params.sliding_window_size : -1;

        cudaError_t err = dispatch_flashinfer_decode<DTypeQ, DTypeKV, DTypeO>(
            const_cast<DTypeQ*>(params.q),
            const_cast<DTypeKV*>(params.k),
            const_cast<DTypeKV*>(params.v),
            params.o, params.lse,
            params.num_qo_heads, params.num_kv_heads,
            params.kv_len,
            params.q_stride_h,
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

    // Initialize attention weights to zero if requested
    if (params.return_attn_weights && params.attn_weights != nullptr) {
        cudaMemsetAsync(params.attn_weights, 0,
            params.num_kv_heads * params.kv_len * sizeof(float), stream);
    }

    const uint32_t block_size = 256;
    const uint32_t smem_size = (TILE_SIZE_KV + block_size + params.head_dim) * sizeof(float);

    dim3 grid(params.num_qo_heads);
    dim3 block(block_size);

    tiled_decode_attention_kernel<DTypeQ, DTypeKV, DTypeO><<<grid, block, smem_size, stream>>>(
        params.q, params.k, params.v, params.o, params.attn_weights,
        params.token_positions,
        params.token_positions_stride_h,
        params.kv_len, params.num_qo_heads, params.num_kv_heads, params.head_dim,
        params.q_stride_h, params.kv_stride_n, params.kv_stride_h,
        params.sm_scale, params.input_pos, params.sliding_window_size,
        params.causal, params.return_attn_weights);

    return cudaGetLastError();
}


template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t launch_batch_decode_attention(
    BatchDecodeParams<DTypeQ, DTypeKV, DTypeO>& params,
    cudaStream_t stream) {

    if (params.q == nullptr || params.k == nullptr ||
        params.v == nullptr || params.o == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (params.batch_size == 0 || params.kv_len == 0 ||
        params.num_qo_heads == 0 || params.num_kv_heads == 0) {
        return cudaErrorInvalidValue;
    }

    // Check if FlashInfer kernels can be used (no token_positions, no weights,
    // supported head_dim, causal). FlashInfer's SingleDecode doesn't support
    // batching, so we still loop per batch item but avoid the sync since
    // FlashInfer doesn't use input_pos.
    bool can_use_flashinfer = (params.token_positions == nullptr) &&
                               (!params.return_attn_weights) &&
                               (params.head_dim == 64 || params.head_dim == 128 || params.head_dim == 256) &&
                               params.causal;

    if (can_use_flashinfer) {
        // Path A: FlashInfer per-batch loop (no sync needed, launches are async)
        for (uint32_t b = 0; b < params.batch_size; ++b) {
            SingleDecodeParams<DTypeQ, DTypeKV, DTypeO> single_params;
            single_params.q = params.q + b * params.q_stride_b;
            single_params.k = params.k + b * params.kv_stride_b;
            single_params.v = params.v + b * params.kv_stride_b;
            single_params.o = params.o + b * params.num_qo_heads * params.head_dim;
            single_params.lse = params.lse ?
                params.lse + b * params.num_qo_heads : nullptr;
            single_params.attn_weights = nullptr;

            single_params.kv_len = params.kv_len;
            single_params.num_qo_heads = params.num_qo_heads;
            single_params.num_kv_heads = params.num_kv_heads;
            single_params.head_dim = params.head_dim;
            single_params.q_stride_h = params.q_stride_h;
            single_params.kv_stride_n = params.kv_stride_n;
            single_params.kv_stride_h = params.kv_stride_h;
            single_params.sm_scale = params.sm_scale;

            single_params.token_positions = nullptr;
            single_params.token_positions_stride_h = 0;
            single_params.input_pos = 0;  // Unused by FlashInfer
            single_params.sliding_window_size = params.sliding_window_size;
            single_params.return_attn_weights = false;
            single_params.causal = true;

            cudaError_t err = launch_single_decode_attention(single_params, stream);
            if (err != cudaSuccess) {
                return err;
            }
        }
        return cudaSuccess;
    }

    // Path B: Try optimized kernel first, fall back to generic batched tiled kernel.
    // The optimized kernel uses FlashInfer primitives (vec_t, shfl_xor_sync, ptx_exp2)
    // and is supported for head_dim in {64, 128, 256} with 16-bit types.

    // Check if num_qo_heads is evenly divisible by num_kv_heads (required for GQA dispatch)
    bool can_use_optimized = (params.num_qo_heads % params.num_kv_heads == 0) &&
                              (params.head_dim == 64 || params.head_dim == 128 || params.head_dim == 256) &&
                              (sizeof(DTypeQ) == 2);  // half or bfloat16

    if (can_use_optimized) {
        cudaError_t err = dispatch_optimized_batched_decode(params, stream);
        if (err == cudaSuccess) {
            return err;
        }
        // Fall through to generic kernel on unsupported config
    }

    // Fallback: Generic batched tiled kernel (handles all configurations)
    if (params.return_attn_weights && params.attn_weights != nullptr) {
        cudaMemsetAsync(params.attn_weights, 0,
            params.batch_size * params.num_kv_heads * params.kv_len * sizeof(float),
            stream);
    }

    const uint32_t block_size = 256;
    const uint32_t smem_size = (TILE_SIZE_KV + block_size + params.head_dim) * sizeof(float);

    dim3 grid(params.batch_size, params.num_qo_heads);
    dim3 block(block_size);

    batched_tiled_decode_attention_kernel<DTypeQ, DTypeKV, DTypeO><<<grid, block, smem_size, stream>>>(
        params.q, params.k, params.v, params.o, params.attn_weights,
        params.token_positions,
        params.token_positions_stride_b,
        params.token_positions_stride_h,
        params.kv_len, params.num_qo_heads, params.num_kv_heads, params.head_dim,
        params.q_stride_b, params.q_stride_h,
        params.kv_stride_b, params.kv_stride_n, params.kv_stride_h,
        params.sm_scale, params.input_pos, params.sliding_window_size,
        params.causal, params.return_attn_weights);

    return cudaGetLastError();
}

// Explicit template instantiations for common types
template cudaError_t launch_single_decode_attention<half, half, half>(
    SingleDecodeParams<half, half, half>& params, cudaStream_t stream);
template cudaError_t launch_single_decode_attention<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
    SingleDecodeParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>& params, cudaStream_t stream);
template cudaError_t launch_single_decode_attention<float, float, float>(
    SingleDecodeParams<float, float, float>& params, cudaStream_t stream);

template cudaError_t launch_batch_decode_attention<half, half, half>(
    BatchDecodeParams<half, half, half>& params, cudaStream_t stream);
template cudaError_t launch_batch_decode_attention<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>(
    BatchDecodeParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>& params, cudaStream_t stream);
template cudaError_t launch_batch_decode_attention<float, float, float>(
    BatchDecodeParams<float, float, float>& params, cudaStream_t stream);

}  // namespace kernels
}  // namespace keys_values
