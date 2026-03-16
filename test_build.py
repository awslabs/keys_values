#!/usr/bin/env python3
"""
Test script to verify CUDA extension build configuration.

This script checks if the build system is properly configured and can
detect CUDA availability.
"""

import sys
import warnings


def test_cuda_detection():
    """Test if CUDA is properly detected."""
    print("Testing CUDA detection...")
    
    # Check CUDA_HOME
    from torch.utils.cpp_extension import CUDA_HOME
    if CUDA_HOME is None:
        print("❌ CUDA_HOME not found - extension will not be built")
        return False
    else:
        print(f"✓ CUDA_HOME found: {CUDA_HOME}")
    
    # Check PyTorch CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ PyTorch CUDA available: {torch.version.cuda}")
            print(f"✓ CUDA device count: {torch.cuda.device_count()}")
        else:
            print("❌ PyTorch CUDA not available - extension will not be built")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    return True


def test_source_files():
    """Test if all required source files exist."""
    print("\nTesting source files...")
    
    from pathlib import Path
    
    required_files = [
        "keys_values/csrc/bindings.cpp",
        "keys_values/csrc/kernels/sdpa_decode.cu",
        "keys_values/csrc/kernels/sdpa_prefill.cu",
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} not found")
            all_exist = False
    
    return all_exist


def test_include_dirs():
    """Test if include directories exist."""
    print("\nTesting include directories...")
    
    from pathlib import Path
    
    include_dirs = [
        "keys_values/csrc",
        "keys_values/csrc/flashinfer",
        "keys_values/csrc/kernels",
    ]
    
    all_exist = True
    for dir_path in include_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print(f"✓ {dir_path}")
        else:
            print(f"❌ {dir_path} not found")
            all_exist = False
    
    return all_exist


def main():
    """Run all tests."""
    print("=" * 60)
    print("CUDA Extension Build Configuration Test")
    print("=" * 60)
    
    cuda_ok = test_cuda_detection()
    sources_ok = test_source_files()
    includes_ok = test_include_dirs()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    if cuda_ok and sources_ok and includes_ok:
        print("✓ All checks passed - CUDA extension should build successfully")
        return 0
    elif sources_ok and includes_ok:
        print("⚠ CUDA not available - package will build without extension")
        print("  (This is expected on systems without CUDA)")
        return 0
    else:
        print("❌ Some checks failed - please fix the issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
