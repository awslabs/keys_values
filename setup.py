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

import os
from pathlib import Path

from setuptools import setup


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


def get_extensions():
    if not os.environ.get("FORCE_CUDA_EXT"):
        return []

    from torch.utils.cpp_extension import CUDAExtension

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

    return [
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


def main():
    from torch.utils.cpp_extension import BuildExtension

    setup(
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension},
    )


if __name__ == "__main__":
    main()
