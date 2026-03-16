/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PyTorch C++ bindings for vendored FlashInfer CUDA kernels.
 * Exposes sdpa_decode and sdpa_prefill functions to Python.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// PyBind11 includes for Python bindings
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <stdexcept>
#include <string>
#include <optional>

namespace py = pybind11;

// Include kernel declarations
#include "kernels/sdpa_decode.cuh"
#include "kernels/sdpa_prefill.cuh"

namespace keys_values {
namespace bindings {

/**
 * @brief Validate tensor is on CUDA device
 */
void check_cuda(const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
}

/**
 * @brief Validate tensor is contiguous
 */
void check_contiguous(const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

/**
 * @brief Validate tensor dtype matches expected
 */
void check_dtype(const torch::Tensor& tensor, at::ScalarType expected, const std::string& name) {
    TORCH_CHECK(tensor.scalar_type() == expected, 
        name, " has dtype ", tensor.scalar_type(), " but expected ", expected);
}

/**
 * @brief Validate tensor has expected number of dimensions
 */
void check_dims(const torch::Tensor& tensor, int64_t expected_dims, const std::string& name) {
    TORCH_CHECK(tensor.dim() == expected_dims,
        name, " has ", tensor.dim(), " dimensions but expected ", expected_dims);
}

/**
 * @brief Get CUDA stream from PyTorch
 */
cudaStream_t get_cuda_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}

/**
 * @brief Scaled Dot Product Attention for decode phase (single query token)
 * 
 * @param query Query tensor: [batch_size, num_qo_heads, head_dim] or [num_qo_heads, head_dim]
 * @param key Key tensor: [batch_size, kv_len, num_kv_heads, head_dim] or [kv_len, num_kv_heads, head_dim]
 * @param value Value tensor: same shape as key
 * @param scale Softmax scale factor (typically 1/sqrt(head_dim))
 * @param token_positions Optional token positions for causal masking: [batch_size, kv_len] or [kv_len]
 * @param input_pos Current query position(s): scalar or [batch_size]
 * @param sliding_window_size Sliding window size (-1 for no window)
 * @param causal Whether to apply causal masking
 * @param return_weights Whether to return attention weights
 * 
 * @return Tuple of (output, attention_weights) where attention_weights is None if not requested
 */
std::tuple<torch::Tensor, std::optional<torch::Tensor>> sdpa_decode(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    double scale,
    const std::optional<torch::Tensor>& token_positions,
    const std::optional<torch::Tensor>& input_pos,
    int64_t sliding_window_size,
    bool causal,
    bool return_weights) {
    
    // Validate inputs
    check_cuda(query, "query");
    check_cuda(key, "key");
    check_cuda(value, "value");
    check_contiguous(query, "query");
    check_contiguous(key, "key");
    check_contiguous(value, "value");
    
    // Determine if batched or single
    bool batched = (query.dim() == 3);
    
    if (batched) {
        check_dims(query, 3, "query");
        check_dims(key, 4, "key");
        check_dims(value, 4, "value");
    } else {
        check_dims(query, 2, "query");
        check_dims(key, 3, "key");
        check_dims(value, 3, "value");
    }
    
    // Validate dtypes match
    TORCH_CHECK(query.scalar_type() == key.scalar_type(),
        "query and key must have same dtype");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(),
        "key and value must have same dtype");
    
    // Extract dimensions
    int64_t batch_size = batched ? query.size(0) : 1;
    int64_t num_qo_heads = batched ? query.size(1) : query.size(0);
    int64_t head_dim = batched ? query.size(2) : query.size(1);
    int64_t kv_len = batched ? key.size(1) : key.size(0);
    int64_t num_kv_heads = batched ? key.size(2) : key.size(1);
    
    // Validate head dimensions
    TORCH_CHECK(num_qo_heads % num_kv_heads == 0,
        "num_qo_heads (", num_qo_heads, ") must be divisible by num_kv_heads (", num_kv_heads, ")");
    
    // Validate head_dim matches
    int64_t key_head_dim = batched ? key.size(3) : key.size(2);
    int64_t value_head_dim = batched ? value.size(3) : value.size(2);
    TORCH_CHECK(head_dim == key_head_dim, "query and key head_dim must match");
    TORCH_CHECK(head_dim == value_head_dim, "query and value head_dim must match");
    
    // Validate token_positions if provided
    // Supports two formats:
    // - 2D: [batch_size, kv_len] - shared across all heads
    // - 3D: [batch_size, num_kv_heads, kv_len] - per-head token positions
    bool token_positions_per_head = false;
    if (token_positions.has_value()) {
        check_cuda(token_positions.value(), "token_positions");
        check_contiguous(token_positions.value(), "token_positions");
        if (batched) {
            TORCH_CHECK(token_positions.value().dim() == 2 || token_positions.value().dim() == 3,
                "token_positions must be 2D or 3D for batched input");
            TORCH_CHECK(token_positions.value().size(0) == batch_size,
                "token_positions batch size must match");
            if (token_positions.value().dim() == 3) {
                TORCH_CHECK(token_positions.value().size(1) == num_kv_heads,
                    "token_positions num_kv_heads must match");
                TORCH_CHECK(token_positions.value().size(2) == kv_len,
                    "token_positions kv_len must match");
                token_positions_per_head = true;
            } else {
                TORCH_CHECK(token_positions.value().size(1) == kv_len,
                    "token_positions kv_len must match");
            }
        } else {
            TORCH_CHECK(token_positions.value().dim() == 1 || token_positions.value().dim() == 2,
                "token_positions must be 1D or 2D for non-batched input");
            if (token_positions.value().dim() == 2) {
                TORCH_CHECK(token_positions.value().size(0) == num_kv_heads,
                    "token_positions num_kv_heads must match");
                TORCH_CHECK(token_positions.value().size(1) == kv_len,
                    "token_positions kv_len must match");
                token_positions_per_head = true;
            } else {
                TORCH_CHECK(token_positions.value().size(0) == kv_len,
                    "token_positions length must match kv_len");
            }
        }
    }
    
    // Set device guard
    c10::cuda::CUDAGuard device_guard(query.device());
    cudaStream_t stream = get_cuda_stream();
    
    // Create output tensor
    auto output_options = query.options();
    torch::Tensor output;
    if (batched) {
        output = torch::empty({batch_size, num_qo_heads, head_dim}, output_options);
    } else {
        output = torch::empty({num_qo_heads, head_dim}, output_options);
    }
    
    // Create attention weights tensor if requested
    std::optional<torch::Tensor> attn_weights = std::nullopt;
    if (return_weights) {
        auto weights_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(query.device());
        if (batched) {
            attn_weights = torch::zeros({batch_size, num_kv_heads, kv_len}, weights_options);
        } else {
            attn_weights = torch::zeros({num_kv_heads, kv_len}, weights_options);
        }
    }
    
    // Dispatch based on dtype
    auto dtype = query.scalar_type();
    cudaError_t err = cudaSuccess;
    
    if (batched) {
        // Batched decode
        if (dtype == at::ScalarType::Half) {
            kernels::BatchDecodeParams<half, half, half> params;
            params.q = reinterpret_cast<const half*>(query.data_ptr());
            params.k = reinterpret_cast<const half*>(key.data_ptr());
            params.v = reinterpret_cast<const half*>(value.data_ptr());
            params.o = reinterpret_cast<half*>(output.data_ptr());
            params.lse = nullptr;
            params.attn_weights = return_weights ? 
                attn_weights.value().data_ptr<float>() : nullptr;
            
            params.batch_size = static_cast<uint32_t>(batch_size);
            params.kv_len = static_cast<uint32_t>(kv_len);
            params.num_qo_heads = static_cast<uint32_t>(num_qo_heads);
            params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
            params.head_dim = static_cast<uint32_t>(head_dim);
            
            params.q_stride_b = static_cast<uint32_t>(query.stride(0));
            params.q_stride_h = static_cast<uint32_t>(query.stride(1));
            params.kv_stride_b = static_cast<uint32_t>(key.stride(0));
            params.kv_stride_n = static_cast<uint32_t>(key.stride(1));
            params.kv_stride_h = static_cast<uint32_t>(key.stride(2));
            
            params.sm_scale = static_cast<float>(scale);
            params.token_positions = token_positions.has_value() ?
                token_positions.value().data_ptr<int32_t>() : nullptr;
            // Set token_positions strides for per-head support
            if (token_positions.has_value() && token_positions_per_head) {
                params.token_positions_stride_b = static_cast<uint32_t>(token_positions.value().stride(0));
                params.token_positions_stride_h = static_cast<uint32_t>(token_positions.value().stride(1));
            } else if (token_positions.has_value()) {
                params.token_positions_stride_b = static_cast<uint32_t>(token_positions.value().stride(0));
                params.token_positions_stride_h = 0;  // Shared across heads
            } else {
                params.token_positions_stride_b = 0;
                params.token_positions_stride_h = 0;
            }
            params.input_pos = input_pos.has_value() ?
                input_pos.value().data_ptr<int32_t>() : nullptr;
            params.sliding_window_size = static_cast<int32_t>(sliding_window_size);
            params.return_attn_weights = return_weights;
            params.causal = causal;

            // Allocate logits cache for optimized attention weights computation
            torch::Tensor logits_tmp_tensor;
            if (return_weights) {
                logits_tmp_tensor = torch::empty(
                    {batch_size, num_qo_heads, kv_len},
                    torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));
                params.logits_tmp = logits_tmp_tensor.data_ptr<float>();
            } else {
                params.logits_tmp = nullptr;
            }

            err = kernels::launch_batch_decode_attention(params, stream);
        } else if (dtype == at::ScalarType::BFloat16) {
            kernels::BatchDecodeParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16> params;
            params.q = reinterpret_cast<const __nv_bfloat16*>(query.data_ptr());
            params.k = reinterpret_cast<const __nv_bfloat16*>(key.data_ptr());
            params.v = reinterpret_cast<const __nv_bfloat16*>(value.data_ptr());
            params.o = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
            params.lse = nullptr;
            params.attn_weights = return_weights ? 
                attn_weights.value().data_ptr<float>() : nullptr;
            
            params.batch_size = static_cast<uint32_t>(batch_size);
            params.kv_len = static_cast<uint32_t>(kv_len);
            params.num_qo_heads = static_cast<uint32_t>(num_qo_heads);
            params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
            params.head_dim = static_cast<uint32_t>(head_dim);
            
            params.q_stride_b = static_cast<uint32_t>(query.stride(0));
            params.q_stride_h = static_cast<uint32_t>(query.stride(1));
            params.kv_stride_b = static_cast<uint32_t>(key.stride(0));
            params.kv_stride_n = static_cast<uint32_t>(key.stride(1));
            params.kv_stride_h = static_cast<uint32_t>(key.stride(2));
            
            params.sm_scale = static_cast<float>(scale);
            params.token_positions = token_positions.has_value() ?
                token_positions.value().data_ptr<int32_t>() : nullptr;
            // Set token_positions strides for per-head support
            if (token_positions.has_value() && token_positions_per_head) {
                params.token_positions_stride_b = static_cast<uint32_t>(token_positions.value().stride(0));
                params.token_positions_stride_h = static_cast<uint32_t>(token_positions.value().stride(1));
            } else if (token_positions.has_value()) {
                params.token_positions_stride_b = static_cast<uint32_t>(token_positions.value().stride(0));
                params.token_positions_stride_h = 0;  // Shared across heads
            } else {
                params.token_positions_stride_b = 0;
                params.token_positions_stride_h = 0;
            }
            params.input_pos = input_pos.has_value() ?
                input_pos.value().data_ptr<int32_t>() : nullptr;
            params.sliding_window_size = static_cast<int32_t>(sliding_window_size);
            params.return_attn_weights = return_weights;
            params.causal = causal;

            // Allocate logits cache for optimized attention weights computation
            torch::Tensor logits_tmp_tensor_bf16;
            if (return_weights) {
                logits_tmp_tensor_bf16 = torch::empty(
                    {batch_size, num_qo_heads, kv_len},
                    torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));
                params.logits_tmp = logits_tmp_tensor_bf16.data_ptr<float>();
            } else {
                params.logits_tmp = nullptr;
            }

            err = kernels::launch_batch_decode_attention(params, stream);
        } else if (dtype == at::ScalarType::Float) {
            kernels::BatchDecodeParams<float, float, float> params;
            params.q = query.data_ptr<float>();
            params.k = key.data_ptr<float>();
            params.v = value.data_ptr<float>();
            params.o = output.data_ptr<float>();
            params.lse = nullptr;
            params.attn_weights = return_weights ? 
                attn_weights.value().data_ptr<float>() : nullptr;
            
            params.batch_size = static_cast<uint32_t>(batch_size);
            params.kv_len = static_cast<uint32_t>(kv_len);
            params.num_qo_heads = static_cast<uint32_t>(num_qo_heads);
            params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
            params.head_dim = static_cast<uint32_t>(head_dim);
            
            params.q_stride_b = static_cast<uint32_t>(query.stride(0));
            params.q_stride_h = static_cast<uint32_t>(query.stride(1));
            params.kv_stride_b = static_cast<uint32_t>(key.stride(0));
            params.kv_stride_n = static_cast<uint32_t>(key.stride(1));
            params.kv_stride_h = static_cast<uint32_t>(key.stride(2));
            
            params.sm_scale = static_cast<float>(scale);
            params.token_positions = token_positions.has_value() ?
                token_positions.value().data_ptr<int32_t>() : nullptr;
            // Set token_positions strides for per-head support
            if (token_positions.has_value() && token_positions_per_head) {
                params.token_positions_stride_b = static_cast<uint32_t>(token_positions.value().stride(0));
                params.token_positions_stride_h = static_cast<uint32_t>(token_positions.value().stride(1));
            } else if (token_positions.has_value()) {
                params.token_positions_stride_b = static_cast<uint32_t>(token_positions.value().stride(0));
                params.token_positions_stride_h = 0;  // Shared across heads
            } else {
                params.token_positions_stride_b = 0;
                params.token_positions_stride_h = 0;
            }
            params.input_pos = input_pos.has_value() ?
                input_pos.value().data_ptr<int32_t>() : nullptr;
            params.sliding_window_size = static_cast<int32_t>(sliding_window_size);
            params.return_attn_weights = return_weights;
            params.causal = causal;

            // Float32 uses fallback kernel which doesn't support logits caching
            params.logits_tmp = nullptr;

            err = kernels::launch_batch_decode_attention(params, stream);
        } else {
            TORCH_CHECK(false, "Unsupported dtype: ", dtype);
        }
    } else {
        // Single decode
        // Get input_pos value
        int32_t input_pos_val = 0;
        if (input_pos.has_value()) {
            TORCH_CHECK(input_pos.value().numel() == 1, 
                "input_pos must be scalar for non-batched input");
            input_pos_val = input_pos.value().item<int32_t>();
        }
        
        if (dtype == at::ScalarType::Half) {
            kernels::SingleDecodeParams<half, half, half> params;
            params.q = reinterpret_cast<const half*>(query.data_ptr());
            params.k = reinterpret_cast<const half*>(key.data_ptr());
            params.v = reinterpret_cast<const half*>(value.data_ptr());
            params.o = reinterpret_cast<half*>(output.data_ptr());
            params.lse = nullptr;
            params.attn_weights = return_weights ? 
                attn_weights.value().data_ptr<float>() : nullptr;
            
            params.kv_len = static_cast<uint32_t>(kv_len);
            params.num_qo_heads = static_cast<uint32_t>(num_qo_heads);
            params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
            params.head_dim = static_cast<uint32_t>(head_dim);
            
            params.q_stride_h = static_cast<uint32_t>(query.stride(0));
            params.kv_stride_n = static_cast<uint32_t>(key.stride(0));
            params.kv_stride_h = static_cast<uint32_t>(key.stride(1));
            
            params.sm_scale = static_cast<float>(scale);
            params.token_positions = token_positions.has_value() ?
                token_positions.value().data_ptr<int32_t>() : nullptr;
            // Set token_positions stride for per-head support (non-batched)
            if (token_positions.has_value() && token_positions_per_head) {
                params.token_positions_stride_h = static_cast<uint32_t>(token_positions.value().stride(0));
            } else {
                params.token_positions_stride_h = 0;  // Shared across heads
            }
            params.input_pos = input_pos_val;
            params.sliding_window_size = static_cast<int32_t>(sliding_window_size);
            params.return_attn_weights = return_weights;
            params.causal = causal;
            
            err = kernels::launch_single_decode_attention(params, stream);
        } else if (dtype == at::ScalarType::BFloat16) {
            kernels::SingleDecodeParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16> params;
            params.q = reinterpret_cast<const __nv_bfloat16*>(query.data_ptr());
            params.k = reinterpret_cast<const __nv_bfloat16*>(key.data_ptr());
            params.v = reinterpret_cast<const __nv_bfloat16*>(value.data_ptr());
            params.o = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
            params.lse = nullptr;
            params.attn_weights = return_weights ? 
                attn_weights.value().data_ptr<float>() : nullptr;
            
            params.kv_len = static_cast<uint32_t>(kv_len);
            params.num_qo_heads = static_cast<uint32_t>(num_qo_heads);
            params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
            params.head_dim = static_cast<uint32_t>(head_dim);
            
            params.q_stride_h = static_cast<uint32_t>(query.stride(0));
            params.kv_stride_n = static_cast<uint32_t>(key.stride(0));
            params.kv_stride_h = static_cast<uint32_t>(key.stride(1));
            
            params.sm_scale = static_cast<float>(scale);
            params.token_positions = token_positions.has_value() ?
                token_positions.value().data_ptr<int32_t>() : nullptr;
            // Set token_positions stride for per-head support (non-batched)
            if (token_positions.has_value() && token_positions_per_head) {
                params.token_positions_stride_h = static_cast<uint32_t>(token_positions.value().stride(0));
            } else {
                params.token_positions_stride_h = 0;  // Shared across heads
            }
            params.input_pos = input_pos_val;
            params.sliding_window_size = static_cast<int32_t>(sliding_window_size);
            params.return_attn_weights = return_weights;
            params.causal = causal;
            
            err = kernels::launch_single_decode_attention(params, stream);
        } else if (dtype == at::ScalarType::Float) {
            kernels::SingleDecodeParams<float, float, float> params;
            params.q = query.data_ptr<float>();
            params.k = key.data_ptr<float>();
            params.v = value.data_ptr<float>();
            params.o = output.data_ptr<float>();
            params.lse = nullptr;
            params.attn_weights = return_weights ? 
                attn_weights.value().data_ptr<float>() : nullptr;
            
            params.kv_len = static_cast<uint32_t>(kv_len);
            params.num_qo_heads = static_cast<uint32_t>(num_qo_heads);
            params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
            params.head_dim = static_cast<uint32_t>(head_dim);
            
            params.q_stride_h = static_cast<uint32_t>(query.stride(0));
            params.kv_stride_n = static_cast<uint32_t>(key.stride(0));
            params.kv_stride_h = static_cast<uint32_t>(key.stride(1));
            
            params.sm_scale = static_cast<float>(scale);
            params.token_positions = token_positions.has_value() ?
                token_positions.value().data_ptr<int32_t>() : nullptr;
            // Set token_positions stride for per-head support (non-batched)
            if (token_positions.has_value() && token_positions_per_head) {
                params.token_positions_stride_h = static_cast<uint32_t>(token_positions.value().stride(0));
            } else {
                params.token_positions_stride_h = 0;  // Shared across heads
            }
            params.input_pos = input_pos_val;
            params.sliding_window_size = static_cast<int32_t>(sliding_window_size);
            params.return_attn_weights = return_weights;
            params.causal = causal;
            
            err = kernels::launch_single_decode_attention(params, stream);
        } else {
            TORCH_CHECK(false, "Unsupported dtype: ", dtype);
        }
    }
    
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    return std::make_tuple(output, attn_weights);
}


/**
 * @brief Scaled Dot Product Attention for prefill phase (multiple query tokens)
 * 
 * @param query Query tensor: [batch_size, q_len, num_qo_heads, head_dim] or [q_len, num_qo_heads, head_dim]
 * @param key Key tensor: [batch_size, kv_len, num_kv_heads, head_dim] or [kv_len, num_kv_heads, head_dim]
 * @param value Value tensor: same shape as key
 * @param scale Softmax scale factor (typically 1/sqrt(head_dim))
 * @param token_positions Optional token positions for causal masking: [batch_size, kv_len] or [kv_len]
 * @param input_pos Starting query position(s): scalar or [batch_size]
 * @param sliding_window_size Sliding window size (-1 for no window)
 * @param causal Whether to apply causal masking
 * @param return_weights Whether to return attention weights (summed over query axis)
 * @param return_lse Whether to return log-sum-exp values per query position (for online weight computation)
 *
 * @return Tuple of (output, attention_weights, lse) where:
 *         - attention_weights is None if not requested or if return_lse=True
 *         - lse has shape [batch_size, q_len, num_qo_heads] or [q_len, num_qo_heads] if return_lse=True, else None
 *         If return_weights=True, weights have shape [batch_size, num_kv_heads, kv_len] or [num_kv_heads, kv_len]
 */
std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>> sdpa_prefill(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    double scale,
    const std::optional<torch::Tensor>& token_positions,
    const std::optional<torch::Tensor>& input_pos,
    int64_t sliding_window_size,
    bool causal,
    bool return_weights,
    bool return_lse) {
    
    // Validate inputs
    check_cuda(query, "query");
    check_cuda(key, "key");
    check_cuda(value, "value");
    check_contiguous(query, "query");
    check_contiguous(key, "key");
    check_contiguous(value, "value");
    
    // Determine if batched or single
    bool batched = (query.dim() == 4);
    
    if (batched) {
        check_dims(query, 4, "query");
        check_dims(key, 4, "key");
        check_dims(value, 4, "value");
    } else {
        check_dims(query, 3, "query");
        check_dims(key, 3, "key");
        check_dims(value, 3, "value");
    }
    
    // Validate dtypes match
    TORCH_CHECK(query.scalar_type() == key.scalar_type(),
        "query and key must have same dtype");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(),
        "key and value must have same dtype");
    
    // Extract dimensions
    int64_t batch_size = batched ? query.size(0) : 1;
    int64_t q_len = batched ? query.size(1) : query.size(0);
    int64_t num_qo_heads = batched ? query.size(2) : query.size(1);
    int64_t head_dim = batched ? query.size(3) : query.size(2);
    int64_t kv_len = batched ? key.size(1) : key.size(0);
    int64_t num_kv_heads = batched ? key.size(2) : key.size(1);
    
    // Validate head dimensions
    TORCH_CHECK(num_qo_heads % num_kv_heads == 0,
        "num_qo_heads (", num_qo_heads, ") must be divisible by num_kv_heads (", num_kv_heads, ")");
    
    // Validate head_dim matches
    int64_t key_head_dim = batched ? key.size(3) : key.size(2);
    int64_t value_head_dim = batched ? value.size(3) : value.size(2);
    TORCH_CHECK(head_dim == key_head_dim, "query and key head_dim must match");
    TORCH_CHECK(head_dim == value_head_dim, "query and value head_dim must match");
    
    // Validate token_positions if provided
    // Supports two formats:
    // - 2D: [batch_size, kv_len] - shared across all heads
    // - 3D: [batch_size, num_kv_heads, kv_len] - per-head token positions
    bool token_positions_per_head = false;
    if (token_positions.has_value()) {
        check_cuda(token_positions.value(), "token_positions");
        check_contiguous(token_positions.value(), "token_positions");
        if (batched) {
            TORCH_CHECK(token_positions.value().dim() == 2 || token_positions.value().dim() == 3,
                "token_positions must be 2D or 3D for batched input");
            TORCH_CHECK(token_positions.value().size(0) == batch_size,
                "token_positions batch size must match");
            if (token_positions.value().dim() == 3) {
                TORCH_CHECK(token_positions.value().size(1) == num_kv_heads,
                    "token_positions num_kv_heads must match");
                TORCH_CHECK(token_positions.value().size(2) == kv_len,
                    "token_positions kv_len must match");
                token_positions_per_head = true;
            } else {
                TORCH_CHECK(token_positions.value().size(1) == kv_len,
                    "token_positions kv_len must match");
            }
        } else {
            TORCH_CHECK(token_positions.value().dim() == 1 || token_positions.value().dim() == 2,
                "token_positions must be 1D or 2D for non-batched input");
            if (token_positions.value().dim() == 2) {
                TORCH_CHECK(token_positions.value().size(0) == num_kv_heads,
                    "token_positions num_kv_heads must match");
                TORCH_CHECK(token_positions.value().size(1) == kv_len,
                    "token_positions kv_len must match");
                token_positions_per_head = true;
            } else {
                TORCH_CHECK(token_positions.value().size(0) == kv_len,
                    "token_positions length must match kv_len");
            }
        }
    }
    
    // Set device guard
    c10::cuda::CUDAGuard device_guard(query.device());
    cudaStream_t stream = get_cuda_stream();
    
    // Create output tensor
    auto output_options = query.options();
    torch::Tensor output;
    if (batched) {
        output = torch::empty({batch_size, q_len, num_qo_heads, head_dim}, output_options);
    } else {
        output = torch::empty({q_len, num_qo_heads, head_dim}, output_options);
    }
    
    // Create intermediate attention weights tensor for per-query weights
    // Final output will be summed over query axis
    // When return_lse=true, we skip weight allocation (caller computes weights from LSE)
    torch::Tensor attn_weights_per_query;
    std::optional<torch::Tensor> attn_weights_sum = std::nullopt;

    if (return_weights && !return_lse) {
        auto weights_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(query.device());

        // Intermediate: [batch_size, q_len, num_qo_heads, kv_len] or [q_len, num_qo_heads, kv_len]
        if (batched) {
            attn_weights_per_query = torch::zeros(
                {batch_size, q_len, num_qo_heads, kv_len}, weights_options);
            attn_weights_sum = torch::zeros(
                {batch_size, num_kv_heads, kv_len}, weights_options);
        } else {
            attn_weights_per_query = torch::zeros(
                {q_len, num_qo_heads, kv_len}, weights_options);
            attn_weights_sum = torch::zeros(
                {num_kv_heads, kv_len}, weights_options);
        }
    }

    // Create LSE tensor if requested
    // LSE shape: [batch_size, q_len, num_qo_heads] or [q_len, num_qo_heads], dtype float32
    std::optional<torch::Tensor> lse_output = std::nullopt;
    torch::Tensor lse_tensor;

    if (return_lse) {
        auto lse_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(query.device());

        if (batched) {
            lse_tensor = torch::empty({batch_size, q_len, num_qo_heads}, lse_options);
        } else {
            lse_tensor = torch::empty({q_len, num_qo_heads}, lse_options);
        }
        lse_output = lse_tensor;
    }
    
    // Dispatch based on dtype
    auto dtype = query.scalar_type();
    cudaError_t err = cudaSuccess;
    
    if (batched) {
        // Batched prefill
        if (dtype == at::ScalarType::Half) {
            kernels::BatchPrefillParams<half, half, half> params;
            params.q = reinterpret_cast<const half*>(query.data_ptr());
            params.k = reinterpret_cast<const half*>(key.data_ptr());
            params.v = reinterpret_cast<const half*>(value.data_ptr());
            params.o = reinterpret_cast<half*>(output.data_ptr());
            params.lse = return_lse ? lse_tensor.data_ptr<float>() : nullptr;
            params.attn_weights = (return_weights && !return_lse) ?
                attn_weights_per_query.data_ptr<float>() : nullptr;

            params.batch_size = static_cast<uint32_t>(batch_size);
            params.q_len = static_cast<uint32_t>(q_len);
            params.kv_len = static_cast<uint32_t>(kv_len);
            params.num_qo_heads = static_cast<uint32_t>(num_qo_heads);
            params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
            params.head_dim = static_cast<uint32_t>(head_dim);

            params.q_stride_b = static_cast<uint32_t>(query.stride(0));
            params.q_stride_n = static_cast<uint32_t>(query.stride(1));
            params.q_stride_h = static_cast<uint32_t>(query.stride(2));
            params.kv_stride_b = static_cast<uint32_t>(key.stride(0));
            params.kv_stride_n = static_cast<uint32_t>(key.stride(1));
            params.kv_stride_h = static_cast<uint32_t>(key.stride(2));

            params.sm_scale = static_cast<float>(scale);
            params.token_positions = token_positions.has_value() ?
                token_positions.value().data_ptr<int32_t>() : nullptr;
            // Set token_positions strides for per-head support
            if (token_positions.has_value() && token_positions_per_head) {
                params.token_positions_stride_b = static_cast<uint32_t>(token_positions.value().stride(0));
                params.token_positions_stride_h = static_cast<uint32_t>(token_positions.value().stride(1));
            } else if (token_positions.has_value()) {
                params.token_positions_stride_b = static_cast<uint32_t>(token_positions.value().stride(0));
                params.token_positions_stride_h = 0;  // Shared across heads
            } else {
                params.token_positions_stride_b = 0;
                params.token_positions_stride_h = 0;
            }
            params.input_pos = input_pos.has_value() ?
                input_pos.value().data_ptr<int32_t>() : nullptr;
            params.sliding_window_size = static_cast<int32_t>(sliding_window_size);
            params.return_attn_weights = return_weights && !return_lse;
            params.causal = causal;

            err = kernels::launch_batch_prefill_attention(params, stream);
        } else if (dtype == at::ScalarType::BFloat16) {
            kernels::BatchPrefillParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16> params;
            params.q = reinterpret_cast<const __nv_bfloat16*>(query.data_ptr());
            params.k = reinterpret_cast<const __nv_bfloat16*>(key.data_ptr());
            params.v = reinterpret_cast<const __nv_bfloat16*>(value.data_ptr());
            params.o = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
            params.lse = return_lse ? lse_tensor.data_ptr<float>() : nullptr;
            params.attn_weights = (return_weights && !return_lse) ?
                attn_weights_per_query.data_ptr<float>() : nullptr;
            
            params.batch_size = static_cast<uint32_t>(batch_size);
            params.q_len = static_cast<uint32_t>(q_len);
            params.kv_len = static_cast<uint32_t>(kv_len);
            params.num_qo_heads = static_cast<uint32_t>(num_qo_heads);
            params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
            params.head_dim = static_cast<uint32_t>(head_dim);
            
            params.q_stride_b = static_cast<uint32_t>(query.stride(0));
            params.q_stride_n = static_cast<uint32_t>(query.stride(1));
            params.q_stride_h = static_cast<uint32_t>(query.stride(2));
            params.kv_stride_b = static_cast<uint32_t>(key.stride(0));
            params.kv_stride_n = static_cast<uint32_t>(key.stride(1));
            params.kv_stride_h = static_cast<uint32_t>(key.stride(2));
            
            params.sm_scale = static_cast<float>(scale);
            params.token_positions = token_positions.has_value() ?
                token_positions.value().data_ptr<int32_t>() : nullptr;
            // Set token_positions strides for per-head support
            if (token_positions.has_value() && token_positions_per_head) {
                params.token_positions_stride_b = static_cast<uint32_t>(token_positions.value().stride(0));
                params.token_positions_stride_h = static_cast<uint32_t>(token_positions.value().stride(1));
            } else if (token_positions.has_value()) {
                params.token_positions_stride_b = static_cast<uint32_t>(token_positions.value().stride(0));
                params.token_positions_stride_h = 0;  // Shared across heads
            } else {
                params.token_positions_stride_b = 0;
                params.token_positions_stride_h = 0;
            }
            params.input_pos = input_pos.has_value() ?
                input_pos.value().data_ptr<int32_t>() : nullptr;
            params.sliding_window_size = static_cast<int32_t>(sliding_window_size);
            params.return_attn_weights = return_weights && !return_lse;
            params.causal = causal;

            err = kernels::launch_batch_prefill_attention(params, stream);
        } else if (dtype == at::ScalarType::Float) {
            kernels::BatchPrefillParams<float, float, float> params;
            params.q = query.data_ptr<float>();
            params.k = key.data_ptr<float>();
            params.v = value.data_ptr<float>();
            params.o = output.data_ptr<float>();
            params.lse = return_lse ? lse_tensor.data_ptr<float>() : nullptr;
            params.attn_weights = (return_weights && !return_lse) ?
                attn_weights_per_query.data_ptr<float>() : nullptr;
            
            params.batch_size = static_cast<uint32_t>(batch_size);
            params.q_len = static_cast<uint32_t>(q_len);
            params.kv_len = static_cast<uint32_t>(kv_len);
            params.num_qo_heads = static_cast<uint32_t>(num_qo_heads);
            params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
            params.head_dim = static_cast<uint32_t>(head_dim);
            
            params.q_stride_b = static_cast<uint32_t>(query.stride(0));
            params.q_stride_n = static_cast<uint32_t>(query.stride(1));
            params.q_stride_h = static_cast<uint32_t>(query.stride(2));
            params.kv_stride_b = static_cast<uint32_t>(key.stride(0));
            params.kv_stride_n = static_cast<uint32_t>(key.stride(1));
            params.kv_stride_h = static_cast<uint32_t>(key.stride(2));
            
            params.sm_scale = static_cast<float>(scale);
            params.token_positions = token_positions.has_value() ?
                token_positions.value().data_ptr<int32_t>() : nullptr;
            // Set token_positions strides for per-head support
            if (token_positions.has_value() && token_positions_per_head) {
                params.token_positions_stride_b = static_cast<uint32_t>(token_positions.value().stride(0));
                params.token_positions_stride_h = static_cast<uint32_t>(token_positions.value().stride(1));
            } else if (token_positions.has_value()) {
                params.token_positions_stride_b = static_cast<uint32_t>(token_positions.value().stride(0));
                params.token_positions_stride_h = 0;  // Shared across heads
            } else {
                params.token_positions_stride_b = 0;
                params.token_positions_stride_h = 0;
            }
            params.input_pos = input_pos.has_value() ?
                input_pos.value().data_ptr<int32_t>() : nullptr;
            params.sliding_window_size = static_cast<int32_t>(sliding_window_size);
            params.return_attn_weights = return_weights && !return_lse;
            params.causal = causal;

            err = kernels::launch_batch_prefill_attention(params, stream);
        } else {
            TORCH_CHECK(false, "Unsupported dtype: ", dtype);
        }

        // Accumulate attention weights over query axis
        if (return_weights && !return_lse && err == cudaSuccess) {
            err = kernels::launch_batch_prefill_accumulate_attention_weights(
                attn_weights_per_query.data_ptr<float>(),
                attn_weights_sum.value().data_ptr<float>(),
                static_cast<uint32_t>(batch_size),
                static_cast<uint32_t>(q_len),
                static_cast<uint32_t>(num_qo_heads),
                static_cast<uint32_t>(num_kv_heads),
                static_cast<uint32_t>(kv_len),
                stream);
        }
    } else {
        // Single prefill
        // Get input_pos value
        int32_t input_pos_val = 0;
        if (input_pos.has_value()) {
            TORCH_CHECK(input_pos.value().numel() == 1, 
                "input_pos must be scalar for non-batched input");
            input_pos_val = input_pos.value().item<int32_t>();
        }
        
        if (dtype == at::ScalarType::Half) {
            kernels::SinglePrefillParams<half, half, half> params;
            params.q = reinterpret_cast<const half*>(query.data_ptr());
            params.k = reinterpret_cast<const half*>(key.data_ptr());
            params.v = reinterpret_cast<const half*>(value.data_ptr());
            params.o = reinterpret_cast<half*>(output.data_ptr());
            params.lse = return_lse ? lse_tensor.data_ptr<float>() : nullptr;
            params.attn_weights = (return_weights && !return_lse) ?
                attn_weights_per_query.data_ptr<float>() : nullptr;

            params.q_len = static_cast<uint32_t>(q_len);
            params.kv_len = static_cast<uint32_t>(kv_len);
            params.num_qo_heads = static_cast<uint32_t>(num_qo_heads);
            params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
            params.head_dim = static_cast<uint32_t>(head_dim);

            params.q_stride_n = static_cast<uint32_t>(query.stride(0));
            params.q_stride_h = static_cast<uint32_t>(query.stride(1));
            params.kv_stride_n = static_cast<uint32_t>(key.stride(0));
            params.kv_stride_h = static_cast<uint32_t>(key.stride(1));

            params.sm_scale = static_cast<float>(scale);
            params.token_positions = token_positions.has_value() ?
                token_positions.value().data_ptr<int32_t>() : nullptr;
            // Set token_positions stride for per-head support (non-batched)
            if (token_positions.has_value() && token_positions_per_head) {
                params.token_positions_stride_h = static_cast<uint32_t>(token_positions.value().stride(0));
            } else {
                params.token_positions_stride_h = 0;  // Shared across heads
            }
            params.input_pos = input_pos_val;
            params.sliding_window_size = static_cast<int32_t>(sliding_window_size);
            params.return_attn_weights = return_weights && !return_lse;
            params.causal = causal;

            err = kernels::launch_single_prefill_attention(params, stream);
        } else if (dtype == at::ScalarType::BFloat16) {
            kernels::SinglePrefillParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16> params;
            params.q = reinterpret_cast<const __nv_bfloat16*>(query.data_ptr());
            params.k = reinterpret_cast<const __nv_bfloat16*>(key.data_ptr());
            params.v = reinterpret_cast<const __nv_bfloat16*>(value.data_ptr());
            params.o = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
            params.lse = return_lse ? lse_tensor.data_ptr<float>() : nullptr;
            params.attn_weights = (return_weights && !return_lse) ?
                attn_weights_per_query.data_ptr<float>() : nullptr;

            params.q_len = static_cast<uint32_t>(q_len);
            params.kv_len = static_cast<uint32_t>(kv_len);
            params.num_qo_heads = static_cast<uint32_t>(num_qo_heads);
            params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
            params.head_dim = static_cast<uint32_t>(head_dim);

            params.q_stride_n = static_cast<uint32_t>(query.stride(0));
            params.q_stride_h = static_cast<uint32_t>(query.stride(1));
            params.kv_stride_n = static_cast<uint32_t>(key.stride(0));
            params.kv_stride_h = static_cast<uint32_t>(key.stride(1));

            params.sm_scale = static_cast<float>(scale);
            params.token_positions = token_positions.has_value() ?
                token_positions.value().data_ptr<int32_t>() : nullptr;
            // Set token_positions stride for per-head support (non-batched)
            if (token_positions.has_value() && token_positions_per_head) {
                params.token_positions_stride_h = static_cast<uint32_t>(token_positions.value().stride(0));
            } else {
                params.token_positions_stride_h = 0;  // Shared across heads
            }
            params.input_pos = input_pos_val;
            params.sliding_window_size = static_cast<int32_t>(sliding_window_size);
            params.return_attn_weights = return_weights && !return_lse;
            params.causal = causal;

            err = kernels::launch_single_prefill_attention(params, stream);
        } else if (dtype == at::ScalarType::Float) {
            kernels::SinglePrefillParams<float, float, float> params;
            params.q = query.data_ptr<float>();
            params.k = key.data_ptr<float>();
            params.v = value.data_ptr<float>();
            params.o = output.data_ptr<float>();
            params.lse = return_lse ? lse_tensor.data_ptr<float>() : nullptr;
            params.attn_weights = (return_weights && !return_lse) ?
                attn_weights_per_query.data_ptr<float>() : nullptr;
            
            params.q_len = static_cast<uint32_t>(q_len);
            params.kv_len = static_cast<uint32_t>(kv_len);
            params.num_qo_heads = static_cast<uint32_t>(num_qo_heads);
            params.num_kv_heads = static_cast<uint32_t>(num_kv_heads);
            params.head_dim = static_cast<uint32_t>(head_dim);
            
            params.q_stride_n = static_cast<uint32_t>(query.stride(0));
            params.q_stride_h = static_cast<uint32_t>(query.stride(1));
            params.kv_stride_n = static_cast<uint32_t>(key.stride(0));
            params.kv_stride_h = static_cast<uint32_t>(key.stride(1));
            
            params.sm_scale = static_cast<float>(scale);
            params.token_positions = token_positions.has_value() ?
                token_positions.value().data_ptr<int32_t>() : nullptr;
            // Set token_positions stride for per-head support (non-batched)
            if (token_positions.has_value() && token_positions_per_head) {
                params.token_positions_stride_h = static_cast<uint32_t>(token_positions.value().stride(0));
            } else {
                params.token_positions_stride_h = 0;  // Shared across heads
            }
            params.input_pos = input_pos_val;
            params.sliding_window_size = static_cast<int32_t>(sliding_window_size);
            params.return_attn_weights = return_weights && !return_lse;
            params.causal = causal;

            err = kernels::launch_single_prefill_attention(params, stream);
        } else {
            TORCH_CHECK(false, "Unsupported dtype: ", dtype);
        }

        // Accumulate attention weights over query axis
        if (return_weights && !return_lse && err == cudaSuccess) {
            err = kernels::launch_prefill_accumulate_attention_weights(
                attn_weights_per_query.data_ptr<float>(),
                attn_weights_sum.value().data_ptr<float>(),
                static_cast<uint32_t>(q_len),
                static_cast<uint32_t>(num_qo_heads),
                static_cast<uint32_t>(num_kv_heads),
                static_cast<uint32_t>(kv_len),
                stream);
        }
    }

    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return std::make_tuple(output, attn_weights_sum, lse_output);
}

/**
 * @brief Check if CUDA is available for the vendored kernels
 * 
 * @return true if CUDA is available and kernels can be used
 */
bool is_cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

/**
 * @brief Get the number of available CUDA devices
 * 
 * @return Number of CUDA devices, or 0 if CUDA is not available
 */
int get_cuda_device_count() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        return 0;
    }
    return device_count;
}

/**
 * @brief Get CUDA device properties as a string for debugging
 * 
 * @param device_id CUDA device ID
 * @return String with device properties
 */
std::string get_device_info(int device_id) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return "Error getting device properties: " + std::string(cudaGetErrorString(err));
    }
    
    std::string info = "Device " + std::to_string(device_id) + ": " + prop.name;
    info += " (SM " + std::to_string(prop.major) + "." + std::to_string(prop.minor) + ")";
    info += ", " + std::to_string(prop.totalGlobalMem / (1024 * 1024)) + " MB";
    return info;
}

}  // namespace bindings
}  // namespace keys_values

// PyTorch extension module registration
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Vendored FlashInfer CUDA kernels for sparse SDPA";
    
    // Utility functions
    m.def("is_available", &keys_values::bindings::is_cuda_available,
        "Check if CUDA is available for vendored kernels");
    
    m.def("get_device_count", &keys_values::bindings::get_cuda_device_count,
        "Get the number of available CUDA devices");
    
    m.def("get_device_info", &keys_values::bindings::get_device_info,
        "Get CUDA device properties as a string",
        py::arg("device_id") = 0);
    
    // SDPA functions
    m.def("sdpa_decode", &keys_values::bindings::sdpa_decode,
        R"doc(
        Scaled Dot Product Attention for decode phase (single query token).
        
        Args:
            query: Query tensor [batch_size, num_qo_heads, head_dim] or [num_qo_heads, head_dim]
            key: Key tensor [batch_size, kv_len, num_kv_heads, head_dim] or [kv_len, num_kv_heads, head_dim]
            value: Value tensor, same shape as key
            scale: Softmax scale factor (typically 1/sqrt(head_dim))
            token_positions: Optional token positions for causal masking [batch_size, kv_len] or [kv_len]
            input_pos: Current query position(s), scalar or [batch_size]
            sliding_window_size: Sliding window size (-1 for no window)
            causal: Whether to apply causal masking (default: True)
            return_weights: Whether to return attention weights (default: False)
        
        Returns:
            Tuple of (output, attention_weights) where attention_weights is None if not requested.
            If return_weights=True, weights have shape [batch_size, num_kv_heads, kv_len] or [num_kv_heads, kv_len]
        )doc",
        py::arg("query"),
        py::arg("key"),
        py::arg("value"),
        py::arg("scale"),
        py::arg("token_positions") = py::none(),
        py::arg("input_pos") = py::none(),
        py::arg("sliding_window_size") = -1,
        py::arg("causal") = true,
        py::arg("return_weights") = false);
    
    m.def("sdpa_prefill", &keys_values::bindings::sdpa_prefill,
        R"doc(
        Scaled Dot Product Attention for prefill phase (multiple query tokens).

        Args:
            query: Query tensor [batch_size, q_len, num_qo_heads, head_dim] or [q_len, num_qo_heads, head_dim]
            key: Key tensor [batch_size, kv_len, num_kv_heads, head_dim] or [kv_len, num_kv_heads, head_dim]
            value: Value tensor, same shape as key
            scale: Softmax scale factor (typically 1/sqrt(head_dim))
            token_positions: Optional token positions for causal masking [batch_size, kv_len] or [kv_len]
            input_pos: Starting query position(s), scalar or [batch_size]
            sliding_window_size: Sliding window size (-1 for no window)
            causal: Whether to apply causal masking (default: True)
            return_weights: Whether to return attention weights summed over query axis (default: False)
            return_lse: Whether to return log-sum-exp values per query position (default: False).
                When True, uses the fast FlashInfer kernel and returns LSE instead of computing
                per-query attention weights. The caller can compute weights from Q, K, and LSE.

        Returns:
            Tuple of (output, attention_weights, lse):
            - output: Attention output with same shape as query
            - attention_weights: If return_weights=True and return_lse=False, tensor with shape
              [batch_size, num_kv_heads, kv_len] or [num_kv_heads, kv_len]. None otherwise.
            - lse: If return_lse=True, tensor with shape [batch_size, q_len, num_qo_heads] or
              [q_len, num_qo_heads], dtype float32. Values are log-sum-exp in log base 2. None otherwise.
        )doc",
        py::arg("query"),
        py::arg("key"),
        py::arg("value"),
        py::arg("scale"),
        py::arg("token_positions") = py::none(),
        py::arg("input_pos") = py::none(),
        py::arg("sliding_window_size") = -1,
        py::arg("causal") = true,
        py::arg("return_weights") = false,
        py::arg("return_lse") = false);
}
