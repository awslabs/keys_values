# Vendored FlashInfer CUDA Kernels

This directory contains CUDA kernel source files vendored from the [FlashInfer](https://github.com/flashinfer-ai/flashinfer) library.

## Purpose

These vendored kernels enable optimized sparse attention computation without requiring the FlashInfer package as a runtime dependency. The kernels are compiled as part of the keys_values package installation using PyTorch's `cpp_extension` mechanism.

## Directory Structure

```
csrc/
├── flashinfer/                    # Vendored FlashInfer headers
│   ├── attention/                 # Attention kernel headers
│   │   ├── blackwell/             # SM100+ (Blackwell) specific kernels
│   │   ├── hopper/                # SM90+ (Hopper) specific kernels
│   │   ├── decode.cuh             # Decode attention kernels
│   │   ├── prefill.cuh            # Prefill attention kernels
│   │   ├── cascade.cuh            # Cascade attention utilities
│   │   ├── mask.cuh               # Attention masking utilities
│   │   ├── state.cuh              # Attention state management
│   │   ├── variants.cuh           # Attention variants
│   │   └── ...
│   ├── cp_async.cuh               # Async copy utilities
│   ├── math.cuh                   # Math utilities
│   ├── mma.cuh                    # Matrix multiply-accumulate utilities
│   ├── page.cuh                   # Paged KV cache utilities
│   ├── pos_enc.cuh                # Positional encoding utilities
│   ├── utils.cuh                  # General utilities
│   ├── vec_dtypes.cuh             # Vector data types
│   └── ...
├── kernels/                       # Implementation files for PyTorch bindings
│   ├── sdpa_decode.cu             # Decode SDPA implementation
│   └── sdpa_prefill.cu            # Prefill SDPA implementation
└── bindings.cpp                   # PyBind11/PyTorch bindings

```

## External Dependencies

The vendored kernels require the following external dependencies:

1. **CUDA Toolkit** (>= 11.8): Required for CUDA compilation
2. **CUTLASS** (>= 3.0): Required for some attention variants (hopper/blackwell)
3. **spdlog**: Required for logging (optional, can be disabled)

## Build Requirements

- CUDA Toolkit with nvcc compiler
- PyTorch with CUDA support
- C++17 compatible compiler

## Source

These files are vendored from FlashInfer version corresponding to the commit in the `flashinfer/` submodule.

## License

The vendored FlashInfer code is licensed under the Apache License 2.0.
See the original FlashInfer repository for full license details.
