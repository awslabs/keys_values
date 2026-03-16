# Requirements Document: FlashInfer Sparse SDPA Integration

## Introduction

This feature extracts and vendors FlashInfer's optimized CUDA kernel source code directly into the keys_values directory, eliminating any runtime dependency on the FlashInfer package. The primary goal is to enable efficient inference with sparse attention patterns using self-contained CUDA kernels that are compiled as part of keys_values. By copying the relevant kernel source files (CUDA/C++ headers and implementations) into keys_values, we achieve true isolation where the FlashInfer package is not required at runtime. The integration will support returning attention weights summed over the query axis, enabling H2O and other cache policies that depend on attention weight information.

## Glossary

- **SDPA**: Scaled Dot Product Attention - the core attention computation in transformers
- **FlashInfer**: A library providing optimized CUDA kernels for attention operations (source code is vendored, not used as runtime dependency)
- **Vendored Kernels**: CUDA kernel source code copied from FlashInfer into keys_values/csrc/ and compiled as part of keys_values package
- **KV Cache**: Key-Value cache storing previously computed keys and values during inference
- **Sparse Attention**: Attention mechanism that maintains a fixed-size KV cache by selectively evicting tokens based on a cache policy
- **H2O Cache Policy**: A sparse attention policy that scores cache slots by attention weight usage
- **Chunk Processing**: Processing long sequences in smaller chunks to manage GPU memory
- **Token Positions**: Tensor tracking which token positions occupy which slots in the KV cache
- **Attention Weights**: The softmax-normalized scores computed during attention, shape `(batch_size, n_head, kv_len)`
- **Query Groups**: Number of key/value heads (supports Grouped Query Attention where n_query_groups < n_head)
- **cpp_extension**: PyTorch utility for compiling and loading custom C++/CUDA extensions

## Requirements

### Requirement 1: FlashInfer Kernel Source Extraction and Vendoring

**User Story:** As an inference engineer, I want FlashInfer's CUDA kernel source code extracted and vendored directly into the keys_values directory, so that I can use optimized kernels without any runtime dependency on the FlashInfer package.

#### Acceptance Criteria

1. WHEN the keys_values module is built THEN the system SHALL compile vendored CUDA kernel source files from a `keys_values/csrc/` directory without requiring FlashInfer package installation
2. WHEN a query tensor with length smaller than key/value length is provided THEN the vendored kernels SHALL support chunk processing variant of SDPA
3. WHEN vendored kernels are used THEN the system SHALL maintain numerical equivalence with the original FlashInfer implementation for all supported configurations
4. WHEN vendored kernels fail to compile or load THEN the system SHALL gracefully fall back to the existing eager SDPA implementation without errors
5. WHEN the vendored kernels are used THEN they SHALL be compiled via PyTorch's cpp_extension or a similar mechanism during package installation

### Requirement 2: Attention Weights Return

**User Story:** As a sparse attention researcher, I want to retrieve attention weights summed over the query axis from SDPA computation, so that I can implement H2O and other cache policies that depend on attention weight information.

#### Acceptance Criteria

1. WHEN `return_attn_weights=True` is specified in SDPA call THEN the system SHALL return attention weights summed over the query axis with shape `(batch_size, n_query_groups, kv_len)`
2. WHEN attention weights are returned THEN they SHALL be in float32 dtype regardless of query dtype for numerical stability
3. WHEN Grouped Query Attention is used (n_query_groups < n_head) THEN the system SHALL correctly aggregate attention weights across query groups
4. WHEN attention weights are requested THEN the computation SHALL not require materializing the full attention matrix in memory

### Requirement 3: Sparse Attention Mask Support

**User Story:** As a sparse attention implementer, I want to specify attention masks using token position indices rather than dense mask matrices, so that I can efficiently implement cache policies without excessive memory overhead.

#### Acceptance Criteria

1. WHEN token_positions tensor is provided THEN the system SHALL apply causal masking based on token positions rather than requiring a dense mask matrix
2. WHEN token_positions is provided THEN the system SHALL correctly handle cases where different batch positions and heads have different token orderings in the KV cache
3. WHEN computing attention with token_positions THEN the system SHALL support sliding window attention constraints in addition to causal masking
4. WHEN token_positions is used THEN the system SHALL avoid materializing the full dense attention mask

### Requirement 4: Chunk Processing for Long Contexts

**User Story:** As an inference system designer, I want to process long prompts in chunks while maintaining a fixed-size KV cache, so that I can support long-context inference without excessive GPU memory usage.

#### Acceptance Criteria

1. WHEN a query sequence is longer than available GPU memory allows THEN the system SHALL split the query into chunks and process them sequentially
2. WHEN processing chunks THEN the system SHALL correctly apply causal masking such that each chunk only attends to appropriate positions
3. WHEN chunk processing is used with sparse attention THEN the system SHALL allow the cache policy to determine which KV cache slots are overwritten for each chunk
4. WHEN multiple chunks are processed THEN the system SHALL accumulate attention weights correctly across chunks when `return_attn_weights=True`

### Requirement 5: Python Interface and Fallback Mechanism

**User Story:** As a system administrator, I want a clean Python interface that loads the vendored CUDA kernels and provides fallback behavior, so that I can use optimized kernels on supported hardware while maintaining compatibility elsewhere.

#### Acceptance Criteria

1. WHEN the Python interface module is imported THEN it SHALL attempt to load the compiled vendored CUDA kernels and expose them through a unified SDPA interface
2. WHEN a requested SDPA configuration is not supported by the vendored kernels THEN the interface SHALL automatically fall back to the existing eager SDPA implementation
3. WHEN vendored kernels are not compiled or CUDA is unavailable THEN the interface SHALL provide a clear warning message and gracefully fall back to eager implementation
4. WHEN the interface is used THEN it SHALL select the most efficient backend based on input dimensions and configuration without exposing backend details to callers

### Requirement 6: Serialization and Deserialization of Attention State

**User Story:** As a checkpoint management system, I want to serialize and deserialize attention computation state including token positions and cache metadata, so that I can support resumable inference and distributed inference scenarios.

#### Acceptance Criteria

1. WHEN attention state is serialized THEN the system SHALL preserve token_positions, cache metadata, and all necessary information to resume computation
2. WHEN serialized attention state is deserialized THEN the system SHALL restore the exact same computation state without loss of information
3. WHEN attention state is serialized and deserialized THEN the system SHALL produce identical results to continuous computation without serialization
4. WHEN attention state is transferred between devices THEN the system SHALL correctly handle device-specific tensor properties

