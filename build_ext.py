#!/usr/bin/env python3
"""
Build helper that compiles the FlashInfer CUDA extension.

Usage:
    python build_ext.py build_ext --inplace

This script works around CUDA version mismatch between system nvcc and
the PyTorch build by disabling the version check, and sets FORCE_CUDA_EXT
so that setup.py actually builds the extension.
"""
import os

os.environ["FORCE_CUDA_EXT"] = "1"

import torch.utils.cpp_extension as ext  # noqa: E402

ext._check_cuda_version = lambda *args, **kwargs: None

import runpy  # noqa: E402

runpy.run_path("setup.py", run_name="__main__")
