/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Header file for decode SDPA kernel declarations.
 */

#ifndef KEYS_VALUES_SDPA_DECODE_CUH_
#define KEYS_VALUES_SDPA_DECODE_CUH_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace keys_values {
namespace kernels {

/**
 * @brief Parameters for single decode SDPA kernel
 * 
 * Token positions support heterogeneous orderings per head:
 * - If token_positions_per_head is false: token_positions has shape [kv_len]
 * - If token_positions_per_head is true: token_positions has shape [num_kv_heads, kv_len]
 */
template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_ = int32_t>
struct SingleDecodeParams {
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
    
    uint32_t kv_len;
    uint32_t num_qo_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    
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
 * @brief Parameters for batch decode SDPA kernel
 * 
 * Token positions support heterogeneous orderings per batch and head:
 * - If token_positions_stride_h is 0: token_positions has shape [batch_size, kv_len]
 * - If token_positions_stride_h > 0: token_positions has shape [batch_size, num_kv_heads, kv_len]
 */
template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_ = int32_t>
struct BatchDecodeParams {
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
    uint32_t kv_len;
    uint32_t num_qo_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    
    uint32_t q_stride_b;
    uint32_t q_stride_h;
    uint32_t kv_stride_b;
    uint32_t kv_stride_n;
    uint32_t kv_stride_h;
    
    float sm_scale;
    const int32_t* token_positions;
    uint32_t token_positions_stride_b;  // Stride for batch dimension
    uint32_t token_positions_stride_h;  // Stride for per-head token positions (0 if shared across heads)
    const int32_t* input_pos;
    int32_t sliding_window_size;
    bool return_attn_weights;
    bool causal;

    // Temporary buffer for caching pre-softmax logits during Phase 1.
    // Used to avoid Q·K recomputation in the attention weights pass.
    // Shape: [batch_size, num_qo_heads, kv_len], dtype float32.
    // Set to nullptr when return_attn_weights is false or buffer not available.
    float* logits_tmp;
};

// Kernel launch function declarations
template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t launch_single_decode_attention(
    SingleDecodeParams<DTypeQ, DTypeKV, DTypeO>& params,
    cudaStream_t stream);

template <typename DTypeQ, typename DTypeKV, typename DTypeO>
cudaError_t launch_batch_decode_attention(
    BatchDecodeParams<DTypeQ, DTypeKV, DTypeO>& params,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace keys_values

#endif  // KEYS_VALUES_SDPA_DECODE_CUH_
