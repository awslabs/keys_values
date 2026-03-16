#!/usr/bin/env python3
"""
Build script for keys_values package with optional CUDA extension.

This script compiles vendored FlashInfer CUDA kernels as a PyTorch extension.
If CUDA is unavailable, the package is built without the extension and falls
back to eager implementations at runtime.
"""

import os
import sys
import warnings
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


def get_extensions():
    """
    Build CUDA extension if CUDA is available, otherwise return empty list.
    
    Returns:
        List of extensions to build (empty if CUDA unavailable)
    """
    # Check if CUDA is available
    if CUDA_HOME is None:
        warnings.warn(
            "CUDA toolkit not found. Building without vendored FlashInfer kernels. "
            "The package will fall back to eager SDPA implementations."
        )
        return []
    
    try:
        import torch
        if not torch.cuda.is_available():
            warnings.warn(
                "CUDA runtime not available. Building without vendored FlashInfer kernels. "
                "The package will fall back to eager SDPA implementations."
            )
            return []
    except ImportError:
        warnings.warn(
            "PyTorch not found. Cannot build CUDA extension. "
            "Install PyTorch first if you want to use vendored FlashInfer kernels."
        )
        return []
    
    # Define source files for the CUDA extension
    csrc_dir = Path("keys_values/csrc")
    sources = [
        str(csrc_dir / "bindings.cpp"),
        str(csrc_dir / "kernels" / "sdpa_decode.cu"),
        str(csrc_dir / "kernels" / "sdpa_prefill.cu"),
    ]
    
    # Define include directories
    include_dirs = [
        str(csrc_dir),  # For including "kernels/..." and "flashinfer/..."
    ]
    
    # Compiler flags
    extra_compile_args = {
        "cxx": [
            "-O3",
            "-std=c++17",
        ],
        "nvcc": [
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-use_fast_math",
            "-Xcompiler=-fPIC",
            "-Xcompiler=-Wno-float-conversion",
            # Enable half-precision operators required by FlashInfer kernels
            # PyTorch's cpp_extension disables these by default, but FlashInfer needs them
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
    }
    
    # Add architecture-specific flags
    # Support common GPU architectures (Volta, Turing, Ampere, Hopper, Ada)
    # SM70: Volta (V100)
    # SM75: Turing (T4, RTX 20xx)
    # SM80: Ampere (A100, A10)
    # SM86: Ampere (RTX 30xx)
    # SM89: Ada Lovelace (RTX 40xx, L4)
    # SM90: Hopper (H100)
    compute_capabilities = ["70", "75", "80", "86", "89", "90"]
    for cc in compute_capabilities:
        extra_compile_args["nvcc"].append(f"-gencode=arch=compute_{cc},code=sm_{cc}")
    
    # Create the CUDA extension
    ext_modules = [
        CUDAExtension(
            name="keys_values._flashinfer_ops",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            language="c++",
        )
    ]
    
    return ext_modules


def main():
    """Main setup function."""
    # Get extensions (empty list if CUDA unavailable)
    ext_modules = get_extensions()
    
    # Build configuration
    cmdclass = {}
    if ext_modules:
        cmdclass["build_ext"] = BuildExtension.with_options(
            no_python_abi_suffix=True,
            use_ninja=True,  # Use ninja for faster builds if available
        )
    
    # Read long description from README
    readme_path = Path("README.md")
    long_description = ""
    if readme_path.exists():
        long_description = readme_path.read_text(encoding="utf-8")
    
    # Setup configuration
    setup(
        name="keys_values",
        version="0.1.dev1",
        description="Efficient Inference, Fine-tuning and Key-value Caching on top of LitGPT",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Matthias Seeger",
        author_email="mseeger@gmail.com",
        url="https://github.com/awslabs/keys_values",
        license="Apache-2.0",
        packages=find_packages(include=["keys_values", "keys_values.*"]),
        package_data={
            "keys_values": ["LICENSE", "README.md"],
        },
        entry_points={
            "console_scripts": [
                "keys_values=keys_values.__main__:main",
            ],
        },
        install_requires=[
            "torchao",
            "filelock",
        ],
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        python_requires=">=3.8",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
