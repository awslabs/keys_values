#!/usr/bin/env python3
"""
Test script to verify CUDA extension compilation.
This simulates what would happen during pip install.
"""

import sys
import warnings

def check_cuda_availability():
    """Check if CUDA is available for compilation."""
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
        
        print(f"CUDA_HOME: {CUDA_HOME}")
        
        if CUDA_HOME is None:
            print("\n❌ CUDA toolkit not found. Extension will not be built.")
            print("   The package will fall back to eager SDPA implementations.")
            return False
        
        if not torch.cuda.is_available():
            print("\n⚠️  CUDA runtime not available. Extension may not work at runtime.")
            print("   But compilation should proceed if CUDA toolkit is installed.")
        
        print("\n✅ CUDA is available for compilation")
        return True
        
    except ImportError as e:
        print(f"\n❌ PyTorch not found: {e}")
        print("   Install PyTorch first to build CUDA extension.")
        return False

def check_source_files():
    """Check if all required source files exist."""
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
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist

def main():
    print("=" * 70)
    print("CUDA Extension Compilation Check")
    print("=" * 70)
    print()
    
    print("Checking CUDA availability...")
    print("-" * 70)
    cuda_ok = check_cuda_availability()
    print()
    
    print("Checking source files...")
    print("-" * 70)
    files_ok = check_source_files()
    print()
    
    print("=" * 70)
    if cuda_ok and files_ok:
        print("✅ All checks passed! Extension should compile successfully.")
        print()
        print("To build the extension, run:")
        print("  python3 -m pip install -e . --no-build-isolation")
        return 0
    else:
        print("⚠️  Some checks failed. See details above.")
        if not cuda_ok:
            print()
            print("Without CUDA, the package will still install but will use")
            print("eager SDPA implementations instead of optimized kernels.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
