# Building Vendored FlashInfer CUDA Kernels

## Overview

The `keys_values` package includes vendored CUDA kernels from FlashInfer for optimized attention computation. These kernels are compiled as a PyTorch extension during package installation.

## Build Requirements

### Required
- Python >= 3.8
- PyTorch >= 2.0.0 with CUDA support
- CUDA Toolkit >= 11.8 (CUDA 12.x recommended)
- C++ compiler with C++17 support
- NVCC (NVIDIA CUDA Compiler)

### Optional
- Ninja build system (for faster compilation)

## Build Process

### Standard Installation

The CUDA extension is built automatically during package installation:

```bash
pip install -e .
```

### Development Build

For development, you can build the extension in-place:

```bash
python setup.py build_ext --inplace
```

### Conditional Compilation

The build system automatically detects CUDA availability:

- **CUDA Available**: Compiles vendored kernels as `keys_values._flashinfer_ops`
- **CUDA Unavailable**: Skips extension compilation, package falls back to eager implementations

### Testing Build Configuration

Before building, you can test if your system is properly configured:

```bash
python test_build.py
```

This will check:
- CUDA toolkit installation (CUDA_HOME)
- PyTorch CUDA availability
- Required source files
- Include directories

## Supported GPU Architectures

The build configuration targets the following compute capabilities:

- **SM 80**: NVIDIA A100 (Ampere)
- **SM 86**: NVIDIA RTX 30xx (Ampere)
- **SM 89**: NVIDIA RTX 40xx (Ada Lovelace)
- **SM 90**: NVIDIA H100 (Hopper)

## Build Flags

### C++ Compiler Flags
- `-O3`: Maximum optimization
- `-std=c++17`: C++17 standard

### NVCC Flags
- `-O3`: Maximum optimization
- `--expt-relaxed-constexpr`: Enable relaxed constexpr
- `--expt-extended-lambda`: Enable extended lambda support
- `-use_fast_math`: Use fast math operations
- `-Xcompiler=-fPIC`: Position-independent code

## Troubleshooting

### CUDA Not Found

If you see warnings about CUDA not being found:

```
CUDA toolkit not found. Building without vendored FlashInfer kernels.
```

**Solution**: Install CUDA toolkit and set `CUDA_HOME` environment variable:

```bash
export CUDA_HOME=/usr/local/cuda
```

### PyTorch CUDA Not Available

If PyTorch is installed without CUDA support:

```
CUDA runtime not available. Building without vendored FlashInfer kernels.
```

**Solution**: Install PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Compilation Errors

If you encounter compilation errors:

1. Check CUDA toolkit version compatibility with PyTorch
2. Ensure C++ compiler supports C++17
3. Check that NVCC is in your PATH
4. Try building with verbose output:

```bash
VERBOSE=1 pip install -e .
```

## Fallback Behavior

If the CUDA extension fails to build or is unavailable at runtime:

- The package will still install successfully
- All functionality remains available through eager implementations
- A warning will be logged when attempting to use vendored kernels
- Performance will be reduced compared to CUDA-accelerated version

## Verifying Installation

After installation, verify the extension is available:

```python
from keys_values import flashinfer_ops

if flashinfer_ops.is_available():
    print("✓ Vendored FlashInfer kernels available")
else:
    print("⚠ Using eager fallback implementations")
```

## Directory Structure

```
keys_values/csrc/
├── flashinfer/              # Vendored FlashInfer headers
│   ├── attention/           # Attention kernel headers
│   │   ├── decode.cuh
│   │   ├── prefill.cuh
│   │   └── ...
│   ├── utils.cuh
│   └── ...
├── kernels/                 # Kernel implementations
│   ├── sdpa_decode.cu       # Decode attention wrapper
│   └── sdpa_prefill.cu      # Prefill attention wrapper
└── bindings.cpp             # PyTorch/PyBind11 bindings
```

## License

The vendored FlashInfer code is licensed under Apache 2.0. See `flashinfer/LICENSE` for details.
