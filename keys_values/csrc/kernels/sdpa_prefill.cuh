/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Header file for prefill SDPA kernel declarations.
 */

#ifndef KEYS_VALUES_SDPA_PREFILL_CUH_
#define KEYS_VALUES_SDPA_PREFILL_CUH_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace keys_values {
namespace kernels {

/**
 * @brief Parameters for single prefill SDPA kernel
 * 
 * Token positions support heterogeneous orderings per head:
 * - If token_positions_stride_h is 0: token_positions has shape [kv_len]
 * - If token_positions_stride_h > 0: token_positions has shape [num_kv_heads, kv_len]
 */
template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_ = int32_t>
struct SinglePrefillParams {
    using DTypeQ = DTypeQ_;
    using DTypeKV = DTypeKV_;
    using DTypeO = DTypeO_;
    using IdType = IdType_;
    
    const DTypeQ* q;
    const DTypeKV* k;
    const DTypeKV* v;
    DTypeO* o;
    float* lse;
    float* attn_weights;
    
    uint32_t q_len;
    uint32_t kv_len;
    uint32_t num_qo_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    
    uint32_t q_stride_n;
    uint32_t q_stride_h;
    uint32_t kv_stride_n;
    uint32_t kv_stride_h;
    
    float sm_scale;
    const int32_t* token_positions;
    uint32_t token_positions_stride_h;  // Stride for per-head token positions (0 if shared)
    int32_t input_pos;
    int32_t sliding_window_size;
    bool return_attn_weights;
    bool causal;
};

/**
 * @brief Parameters for batch prefill SDPA kernel
 * 
 * Token positions support heterogeneous orderings per batch and head:
 * - If token_positions_stride_h is 0: token_positions has shape [batch_size, kv_len]
 * - If token_positions_stride_h > 0: token_positions has shape [batch_size, num_kv_heads, kv_len]
 */
template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_ = int32_t>
struct BatchPrefillParams {
    using DTypeQ = DTypeQ_;
    using DTypeKV = DTypeKV_;
    using DTypeO = DTypeO_;
    using IdType = IdType_;
    
    const DTypeQ* q;
    const DTypeKV* k;
    const DTypeKV* v;
    DTypeO* o;
    float* lse;
    float* attn_weights;
    
    uint32_t batch_size;
    uint32_t q_len;
    uint32_t kv_len;
    uint32_t num_qo_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    
    uint32_t q_stride_b;
    uint32_t q_stride_n;
    uint32_t q_stride_h;
    uint32_t kv_stride_b;
    uint32_t kv_stride_n;
    uint32_t kv_stride_h;
    
    float sm_scale;
    const int32_t* token_positions;
    uint32_t token_positions_stride_b;  // Stride for batch dimension
    uint32_t token_positions_stride_h;  // Stride for per-head token positions (0 if shared)
    const int32_t* input_pos;
    int32_t sliding_window_size;
    bool return_attn_weights;
    bool causal;
};

// Kernel launch function declarations
template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t launch_single_prefill_attention(
    SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>& params,
    cudaStream_t stream);

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t launch_batch_prefill_attention(
    BatchPrefillParams<DTypeQ, DTypeKV, DTypeO>& params,
    cudaStream_t stream);

// Weight accumulation function declarations
cudaError_t launch_prefill_accumulate_attention_weights(
    const float* attn_weights_per_query,
    float* attn_weights_sum,
    uint32_t q_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t kv_len,
    cudaStream_t stream);

cudaError_t launch_batch_prefill_accumulate_attention_weights(
    const float* attn_weights_per_query,
    float* attn_weights_sum,
    uint32_t batch_size,
    uint32_t q_len,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t kv_len,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace keys_values

#endif  // KEYS_VALUES_SDPA_PREFILL_CUH_
