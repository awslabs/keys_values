# Building keys_values with CUDA Extension

## Overview

The `keys_values` package now includes a build system for compiling vendored FlashInfer CUDA kernels as a PyTorch extension. The build system supports conditional compilation - if CUDA is unavailable, the package builds without the extension and falls back to eager implementations.

## What Was Configured

### 1. Build System Files

#### `setup.py`
- Main build script using PyTorch's `cpp_extension` framework
- Detects CUDA availability (CUDA_HOME and torch.cuda.is_available())
- Conditionally compiles CUDA extension only when CUDA is available
- Configures include paths for vendored headers
- Sets appropriate compiler flags for C++ and NVCC
- Targets multiple GPU architectures (SM 80, 86, 89, 90)

#### `pyproject.toml`
- Updated with `[build-system]` section specifying build requirements
- Declares dependency on PyTorch >= 2.0.0 for building
- Properly structured `[project]` section with dependencies
- Compatible with modern Python packaging standards (PEP 517/518)

### 2. Source Files to Compile

The build system compiles the following files:
- `keys_values/csrc/bindings.cpp` - PyTorch/PyBind11 bindings
- `keys_values/csrc/kernels/sdpa_decode.cu` - Decode attention kernel
- `keys_values/csrc/kernels/sdpa_prefill.cu` - Prefill attention kernel

### 3. Include Paths

The build system configures the following include directory:
- `keys_values/csrc/` - Root directory containing:
  - `flashinfer/` - Vendored FlashInfer headers
  - `kernels/` - Kernel implementation files

### 4. Compiler Flags

#### C++ Compiler (cxx)
- `-O3` - Maximum optimization
- `-std=c++17` - C++17 standard support

#### NVCC (CUDA Compiler)
- `-O3` - Maximum optimization
- `-std=c++17` - C++17 standard
- `--expt-relaxed-constexpr` - Relaxed constexpr evaluation
- `--expt-extended-lambda` - Extended lambda support
- `-use_fast_math` - Fast math operations
- `-Xcompiler=-fPIC` - Position-independent code
- `-Xcompiler=-Wno-float-conversion` - Suppress float conversion warnings
- Architecture-specific flags for SM 80, 86, 89, 90

### 5. Conditional Compilation Logic

The build system implements three-level fallback:

1. **CUDA_HOME not found** → Skip extension, warn user
2. **torch.cuda.is_available() returns False** → Skip extension, warn user
3. **PyTorch not installed** → Skip extension, warn user

In all cases, the package still installs successfully and uses eager fallback implementations.

## Installation

### Standard Installation
```bash
pip install -e .
```

### Development Build (in-place)
```bash
python setup.py build_ext --inplace
```

### Clean Build
```bash
python setup.py clean --all
python setup.py build_ext --inplace
```

## Testing the Configuration

### Test CUDA Detection and Source Files
```bash
python test_build.py
```

Expected output on CUDA-enabled system:
```
✓ CUDA_HOME found: /usr/local/cuda
✓ PyTorch CUDA available: 12.8
✓ CUDA device count: 8
✓ keys_values/csrc/bindings.cpp
✓ keys_values/csrc/kernels/sdpa_decode.cu
✓ keys_values/csrc/kernels/sdpa_prefill.cu
✓ All checks passed - CUDA extension should build successfully
```

### Validate Configuration Files
```bash
python validate_config.py
```

Expected output:
```
✓ pyproject.toml is valid
✓ setup.py exists and contains required components
✓ All configuration files are valid
```

## Verifying the Build

After building, verify the extension is available:

```python
try:
    from keys_values import _flashinfer_ops
    print("✓ CUDA extension loaded successfully")
    print(f"  Available functions: {dir(_flashinfer_ops)}")
except ImportError as e:
    print(f"⚠ CUDA extension not available: {e}")
    print("  Package will use eager fallback implementations")
```

## Build Output

The compiled extension will be created as:
- `keys_values/_flashinfer_ops.cpython-<version>-<platform>.so` (Linux)
- `keys_values/_flashinfer_ops.cpython-<version>-<platform>.dylib` (macOS)

## Supported Platforms

- **Linux**: Primary target, fully supported
- **Windows**: Should work with appropriate CUDA toolkit
- **macOS**: No CUDA support, will use eager fallback

## Supported GPU Architectures

- **Ampere (SM 80)**: NVIDIA A100
- **Ampere (SM 86)**: NVIDIA RTX 30xx series
- **Ada Lovelace (SM 89)**: NVIDIA RTX 40xx series
- **Hopper (SM 90)**: NVIDIA H100

## Requirements Satisfied

This build configuration satisfies the following requirements from the spec:

- **Requirement 1.1**: Compiles vendored CUDA kernel source files from `keys_values/csrc/` without requiring FlashInfer package installation
- **Requirement 1.5**: Kernels are compiled via PyTorch's cpp_extension during package installation
- **Requirement 1.4**: Gracefully falls back to eager SDPA implementation when kernels fail to compile or load

## Next Steps

After this task, the next step is:
- **Task 5**: Checkpoint - Verify CUDA extension compiles successfully

This will involve actually running the build and confirming the extension loads properly.
