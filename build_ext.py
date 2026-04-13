#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Build the FlashInfer CUDA extension for vendored kernels.

Usage:
    python build_ext.py

Prerequisites:
    - PyTorch with CUDA support
    - pip install flashinfer-python (for headers)
    - CUDA toolkit
"""

import os
import sys
from pathlib import Path

# Disable CUDA version check to work around mismatch between system nvcc
# and the PyTorch build
import torch.utils.cpp_extension as ext

ext._check_cuda_version = lambda *args, **kwargs: None

from torch.utils.cpp_extension import CUDAExtension, BuildExtension  # noqa: E402
from setuptools import setup  # noqa: E402


def get_flashinfer_include_dir():
    """Get FlashInfer header include directory from installed package."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("flashinfer")
        if spec and spec.origin:
            include_dir = Path(spec.origin).parent / "data" / "include"
            if include_dir.exists():
                return str(include_dir)
    except (ImportError, AttributeError):
        pass
    raise RuntimeError(
        "FlashInfer package not found. Install with: pip install flashinfer-python"
    )


def main():
    csrc_dir = Path(__file__).parent / "keys_values" / "csrc"
    kernels_dir = csrc_dir / "kernels"

    sources = [
        str(csrc_dir / "bindings.cpp"),
        str(kernels_dir / "sdpa_decode.cu"),
        str(kernels_dir / "sdpa_prefill.cu"),
    ]

    include_dirs = [
        str(csrc_dir),
        get_flashinfer_include_dir(),
    ]

    ext_modules = [
        CUDAExtension(
            name="keys_values._flashinfer_ops",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-Xcompiler=-fPIC",
                    "-Xcompiler=-Wno-float-conversion",
                    # Undo PyTorch's half/bfloat16 conversion restrictions
                    # (FlashInfer headers require these operators)
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-gencode=arch=compute_80,code=sm_80",
                ],
            },
        )
    ]

    # Use setuptools to build the extension inplace
    sys.argv = [sys.argv[0], "build_ext", "--inplace"]
    setup(
        name="keys_values_flashinfer",
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
    )


if __name__ == "__main__":
    main()
