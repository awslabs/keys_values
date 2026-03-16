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
Benchmark script for comparing FlashInfer vs eager SDPA for long context inference.

This script benchmarks:
- Prefill latency for long prompts (2K, 4K, 8K, 16K, 32K, 64K, 128K tokens)
- Decode latency with large KV cache sizes
- Memory usage for both implementations
- Throughput metrics (tokens/second) for prefill and decode

The vendored FlashInfer kernels use a tiled algorithm that processes KV in
fixed-size chunks (TILE_SIZE_KV=256). This enables support for arbitrarily
long context lengths without shared memory limitations.

Usage:
    python benchmark_long_context.py [--output results.csv] [--json results.json]
    python benchmark_long_context.py --context-lengths 2048 4096 8192 16384 32768 65536 131072
    python benchmark_long_context.py --batch-sizes 1 2 4
    python benchmark_long_context.py --backends flashinfer eager pytorch
"""

import argparse
import csv
import gc
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import torch

from keys_values.flashinfer_wrapper import FlashInferSDPA


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    model_name: str
    batch_size: int
    context_length: int
    n_head: int
    n_query_groups: int
    head_size: int
    dtype: torch.dtype
    backend: str  # 'flashinfer', 'eager', 'pytorch'


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: BenchmarkConfig
    # Prefill metrics
    prefill_latency_ms: float
    prefill_throughput_tokens_per_sec: float
    prefill_memory_mb: float
    # Decode metrics
    decode_latency_ms: float
    decode_throughput_tokens_per_sec: float
    decode_memory_mb: float
    # Status
    success: bool
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON/CSV export."""
        return {
            'model_name': self.config.model_name,
            'batch_size': self.config.batch_size,
            'context_length': self.config.context_length,
            'n_head': self.config.n_head,
            'n_query_groups': self.config.n_query_groups,
            'head_size': self.config.head_size,
            'dtype': str(self.config.dtype),
            'backend': self.config.backend,
            'prefill_latency_ms': self.prefill_latency_ms,
            'prefill_throughput_tokens_per_sec': self.prefill_throughput_tokens_per_sec,
            'prefill_memory_mb': self.prefill_memory_mb,
            'decode_latency_ms': self.decode_latency_ms,
            'decode_throughput_tokens_per_sec': self.decode_throughput_tokens_per_sec,
            'decode_memory_mb': self.decode_memory_mb,
            'success': self.success,
            'error_message': self.error_message,
        }


# Model configurations (realistic architectures)
MODEL_CONFIGS = {
    'Llama-3.2-1B': {
        'n_head': 32,
        'n_query_groups': 8,
        'head_size': 64,
    },
    'Mistral-7B': {
        'n_head': 32,
        'n_query_groups': 8,
        'head_size': 128,
    },
    'Llama-3-8B': {
        'n_head': 32,
        'n_query_groups': 8,
        'head_size': 128,
    },
    'Qwen2-7B': {
        'n_head': 28,
        'n_query_groups': 4,
        'head_size': 128,
    },
    'Qwen3-4B': {
        'n_head': 32,
        'n_query_groups': 8,
        'head_size': 128,
    },
}

# Default context lengths to test
# Extended to include 64K and 128K tokens now that tiled kernels are implemented
DEFAULT_CONTEXT_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

# Note: The vendored FlashInfer kernels now use a tiled algorithm that processes
# KV in fixed-size chunks (TILE_SIZE_KV=256). This enables support for arbitrarily
# long context lengths without shared memory limitations. The shared memory usage
# is now O(TILE_SIZE_KV + head_dim) instead of O(kv_len).

# Default batch sizes to test
DEFAULT_BATCH_SIZES = [1, 2, 4]

# Default backends to test
DEFAULT_BACKENDS = ['flashinfer', 'eager', 'pytorch']

# Number of warmup iterations
WARMUP_ITERATIONS = 3

# Number of benchmark iterations
BENCHMARK_ITERATIONS = 10


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def get_peak_gpu_memory_mb() -> float:
    """Get peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_gpu_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


def create_test_tensors(
    batch_size: int,
    n_head: int,
    n_query_groups: int,
    q_len: int,
    kv_len: int,
    head_size: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Create test tensors for SDPA benchmarking."""
    query = torch.randn(batch_size, n_head, q_len, head_size, dtype=dtype, device=device)
    key = torch.randn(batch_size, n_query_groups, kv_len, head_size, dtype=dtype, device=device)
    value = torch.randn(batch_size, n_query_groups, kv_len, head_size, dtype=dtype, device=device)
    scale_factor = 1.0 / math.sqrt(head_size)
    return query, key, value, scale_factor



def benchmark_flashinfer_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: float,
    wrapper: FlashInferSDPA,
    return_attn_weights: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run SDPA using FlashInfer wrapper."""
    return wrapper.scaled_dot_product_attention(
        query, key, value, scale_factor,
        return_attn_weights=return_attn_weights,
    )


def benchmark_eager_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: float,
    wrapper: FlashInferSDPA,
    return_attn_weights: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run SDPA using eager fallback implementation."""
    return wrapper._fallback_sdpa(
        query, key, value, scale_factor,
        return_attn_weights=return_attn_weights,
    )


def benchmark_pytorch_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale_factor: float,
    return_attn_weights: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run SDPA using PyTorch's native implementation."""
    batch_size, n_head, q_len, head_size = query.shape
    _, n_query_groups, kv_len, _ = key.shape
    
    # Expand key and value for GQA
    if n_head > n_query_groups:
        q_per_kv = n_head // n_query_groups
        key = key.unsqueeze(2).expand(-1, -1, q_per_kv, -1, -1)
        key = key.reshape(batch_size, n_head, kv_len, head_size)
        value = value.unsqueeze(2).expand(-1, -1, q_per_kv, -1, -1)
        value = value.reshape(batch_size, n_head, kv_len, head_size)
    
    # Use PyTorch's scaled_dot_product_attention
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value,
        scale=scale_factor,
        is_causal=(q_len == kv_len),
    )
    
    # PyTorch SDPA doesn't return attention weights efficiently
    # We return None for weights
    return output, None


def run_prefill_benchmark(
    config: BenchmarkConfig,
    wrapper: FlashInferSDPA,
    warmup_iterations: int = WARMUP_ITERATIONS,
    benchmark_iterations: int = BENCHMARK_ITERATIONS,
) -> Tuple[float, float, float]:
    """
    Run prefill benchmark (q_len == kv_len).
    
    Returns:
        Tuple of (latency_ms, throughput_tokens_per_sec, memory_mb)
    """
    device = "cuda"
    q_len = config.context_length
    kv_len = config.context_length
    
    # Create tensors
    query, key, value, scale_factor = create_test_tensors(
        config.batch_size, config.n_head, config.n_query_groups,
        q_len, kv_len, config.head_size, config.dtype, device
    )
    
    # Select backend function
    if config.backend == 'flashinfer':
        sdpa_fn = lambda: benchmark_flashinfer_sdpa(query, key, value, scale_factor, wrapper)
    elif config.backend == 'eager':
        sdpa_fn = lambda: benchmark_eager_sdpa(query, key, value, scale_factor, wrapper)
    elif config.backend == 'pytorch':
        sdpa_fn = lambda: benchmark_pytorch_sdpa(query, key, value, scale_factor)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")
    
    # Warmup
    for _ in range(warmup_iterations):
        _ = sdpa_fn()
        torch.cuda.synchronize()
    
    # Reset memory stats
    reset_gpu_memory_stats()
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for _ in range(benchmark_iterations):
        _ = sdpa_fn()
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    # Calculate metrics
    total_time_ms = (end_time - start_time) * 1000
    latency_ms = total_time_ms / benchmark_iterations
    total_tokens = config.batch_size * q_len
    throughput = (total_tokens / latency_ms) * 1000  # tokens/second
    memory_mb = get_peak_gpu_memory_mb()
    
    # Cleanup
    del query, key, value
    torch.cuda.empty_cache()
    gc.collect()
    
    return latency_ms, throughput, memory_mb


def run_decode_benchmark(
    config: BenchmarkConfig,
    wrapper: FlashInferSDPA,
    warmup_iterations: int = WARMUP_ITERATIONS,
    benchmark_iterations: int = BENCHMARK_ITERATIONS,
) -> Tuple[float, float, float]:
    """
    Run decode benchmark (q_len = 1, kv_len = context_length).
    
    Returns:
        Tuple of (latency_ms, throughput_tokens_per_sec, memory_mb)
    """
    device = "cuda"
    q_len = 1  # Single token decode
    kv_len = config.context_length
    
    # Create tensors
    query, key, value, scale_factor = create_test_tensors(
        config.batch_size, config.n_head, config.n_query_groups,
        q_len, kv_len, config.head_size, config.dtype, device
    )
    
    # Create token positions for decode phase
    token_positions = torch.arange(kv_len, device=device).unsqueeze(0).unsqueeze(0)
    token_positions = token_positions.expand(config.batch_size, config.n_query_groups, -1)
    
    # Select backend function
    if config.backend == 'flashinfer':
        sdpa_fn = lambda: wrapper.scaled_dot_product_attention(
            query, key, value, scale_factor,
            return_attn_weights=True,
            token_positions=token_positions,
            input_pos=kv_len - 1,
        )
    elif config.backend == 'eager':
        sdpa_fn = lambda: wrapper._fallback_sdpa(
            query, key, value, scale_factor,
            return_attn_weights=True,
            token_positions=token_positions,
            input_pos=kv_len - 1,
        )
    elif config.backend == 'pytorch':
        # PyTorch SDPA doesn't support token_positions, use simple decode
        sdpa_fn = lambda: benchmark_pytorch_sdpa(query, key, value, scale_factor)
    else:
        raise ValueError(f"Unknown backend: {config.backend}")
    
    # Warmup
    for _ in range(warmup_iterations):
        _ = sdpa_fn()
        torch.cuda.synchronize()
    
    # Reset memory stats
    reset_gpu_memory_stats()
    
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    for _ in range(benchmark_iterations):
        _ = sdpa_fn()
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    # Calculate metrics
    total_time_ms = (end_time - start_time) * 1000
    latency_ms = total_time_ms / benchmark_iterations
    total_tokens = config.batch_size * q_len
    throughput = (total_tokens / latency_ms) * 1000  # tokens/second
    memory_mb = get_peak_gpu_memory_mb()
    
    # Cleanup
    del query, key, value, token_positions
    torch.cuda.empty_cache()
    gc.collect()
    
    return latency_ms, throughput, memory_mb



def run_single_benchmark(config: BenchmarkConfig, wrapper: FlashInferSDPA) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    import logging
    # Suppress FlashInfer fallback warnings during benchmarking
    logging.getLogger('keys_values.flashinfer_wrapper').setLevel(logging.ERROR)
    
    try:
        # Run prefill benchmark
        prefill_latency, prefill_throughput, prefill_memory = run_prefill_benchmark(
            config, wrapper
        )
        
        # Run decode benchmark
        decode_latency, decode_throughput, decode_memory = run_decode_benchmark(
            config, wrapper
        )
        
        return BenchmarkResult(
            config=config,
            prefill_latency_ms=prefill_latency,
            prefill_throughput_tokens_per_sec=prefill_throughput,
            prefill_memory_mb=prefill_memory,
            decode_latency_ms=decode_latency,
            decode_throughput_tokens_per_sec=decode_throughput,
            decode_memory_mb=decode_memory,
            success=True,
        )
    except torch.cuda.OutOfMemoryError as e:
        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        return BenchmarkResult(
            config=config,
            prefill_latency_ms=0.0,
            prefill_throughput_tokens_per_sec=0.0,
            prefill_memory_mb=0.0,
            decode_latency_ms=0.0,
            decode_throughput_tokens_per_sec=0.0,
            decode_memory_mb=0.0,
            success=False,
            error_message=f"OOM: {str(e)[:100]}",
        )
    except Exception as e:
        return BenchmarkResult(
            config=config,
            prefill_latency_ms=0.0,
            prefill_throughput_tokens_per_sec=0.0,
            prefill_memory_mb=0.0,
            decode_latency_ms=0.0,
            decode_throughput_tokens_per_sec=0.0,
            decode_memory_mb=0.0,
            success=False,
            error_message=str(e)[:100],
        )


def run_all_benchmarks(
    model_names: List[str],
    context_lengths: List[int],
    batch_sizes: List[int],
    backends: List[str],
    dtype: torch.dtype = torch.float16,
) -> List[BenchmarkResult]:
    """Run all benchmark configurations."""
    results = []
    wrapper = FlashInferSDPA()
    
    print(f"\nFlashInfer available: {wrapper.available}")
    print(f"Running benchmarks with {len(model_names)} models, "
          f"{len(context_lengths)} context lengths, "
          f"{len(batch_sizes)} batch sizes, "
          f"{len(backends)} backends")
    print("=" * 80)
    
    total_configs = len(model_names) * len(context_lengths) * len(batch_sizes) * len(backends)
    current_config = 0
    
    for model_name in model_names:
        model_config = MODEL_CONFIGS.get(model_name)
        if model_config is None:
            print(f"WARNING: Unknown model {model_name}, skipping")
            continue
        
        for context_length in context_lengths:
            for batch_size in batch_sizes:
                for backend in backends:
                    current_config += 1
                    
                    # Skip FlashInfer if not available
                    if backend == 'flashinfer' and not wrapper.available:
                        print(f"[{current_config}/{total_configs}] SKIP {model_name} "
                              f"ctx={context_length} bs={batch_size} {backend} "
                              f"(FlashInfer not available)")
                        continue
                    
                    config = BenchmarkConfig(
                        model_name=model_name,
                        batch_size=batch_size,
                        context_length=context_length,
                        n_head=model_config['n_head'],
                        n_query_groups=model_config['n_query_groups'],
                        head_size=model_config['head_size'],
                        dtype=dtype,
                        backend=backend,
                    )
                    
                    print(f"[{current_config}/{total_configs}] Running {model_name} "
                          f"ctx={context_length} bs={batch_size} {backend}...", end=" ")
                    
                    result = run_single_benchmark(config, wrapper)
                    results.append(result)
                    
                    if result.success:
                        print(f"prefill={result.prefill_latency_ms:.2f}ms "
                              f"({result.prefill_throughput_tokens_per_sec:.0f} tok/s), "
                              f"decode={result.decode_latency_ms:.2f}ms "
                              f"({result.decode_throughput_tokens_per_sec:.0f} tok/s)")
                    else:
                        print(f"FAILED: {result.error_message[:50]}...")
    
    return results


def save_results_csv(results: List[BenchmarkResult], filename: str):
    """Save benchmark results to CSV file."""
    if not results:
        print("No results to save")
        return
    
    fieldnames = list(results[0].to_dict().keys())
    
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())
    
    print(f"Results saved to {filename}")


def save_results_json(results: List[BenchmarkResult], filename: str):
    """Save benchmark results to JSON file."""
    data = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
            'pytorch_version': torch.__version__,
            'total_benchmarks': len(results),
            'successful_benchmarks': sum(1 for r in results if r.success),
        },
        'results': [r.to_dict() for r in results],
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {filename}")


def print_summary(results: List[BenchmarkResult]):
    """Print summary of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"\nTotal benchmarks: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if not successful:
        print("\nNo successful benchmarks to summarize")
        return
    
    # Group by backend for comparison
    backends = set(r.config.backend for r in successful)
    
    print("\n" + "-" * 80)
    print("PREFILL PERFORMANCE (tokens/second)")
    print("-" * 80)
    print(f"{'Model':<20} {'Context':<10} {'Batch':<8}", end="")
    for backend in sorted(backends):
        print(f"{backend:<15}", end="")
    print()
    
    # Group results by model, context, batch
    grouped = {}
    for r in successful:
        key = (r.config.model_name, r.config.context_length, r.config.batch_size)
        if key not in grouped:
            grouped[key] = {}
        grouped[key][r.config.backend] = r
    
    for (model, ctx, batch), backend_results in sorted(grouped.items()):
        print(f"{model:<20} {ctx:<10} {batch:<8}", end="")
        for backend in sorted(backends):
            if backend in backend_results:
                throughput = backend_results[backend].prefill_throughput_tokens_per_sec
                print(f"{throughput:<15.0f}", end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()
    
    print("\n" + "-" * 80)
    print("DECODE PERFORMANCE (tokens/second)")
    print("-" * 80)
    print(f"{'Model':<20} {'Context':<10} {'Batch':<8}", end="")
    for backend in sorted(backends):
        print(f"{backend:<15}", end="")
    print()
    
    for (model, ctx, batch), backend_results in sorted(grouped.items()):
        print(f"{model:<20} {ctx:<10} {batch:<8}", end="")
        for backend in sorted(backends):
            if backend in backend_results:
                throughput = backend_results[backend].decode_throughput_tokens_per_sec
                print(f"{throughput:<15.0f}", end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()
    
    # Print speedup comparison if FlashInfer is available
    if 'flashinfer' in backends and 'eager' in backends:
        print("\n" + "-" * 80)
        print("FLASHINFER vs EAGER SPEEDUP")
        print("-" * 80)
        print(f"{'Model':<20} {'Context':<10} {'Batch':<8} {'Prefill':<15} {'Decode':<15}")
        
        for (model, ctx, batch), backend_results in sorted(grouped.items()):
            if 'flashinfer' in backend_results and 'eager' in backend_results:
                fi = backend_results['flashinfer']
                eager = backend_results['eager']
                
                prefill_speedup = fi.prefill_throughput_tokens_per_sec / eager.prefill_throughput_tokens_per_sec
                decode_speedup = fi.decode_throughput_tokens_per_sec / eager.decode_throughput_tokens_per_sec
                
                print(f"{model:<20} {ctx:<10} {batch:<8} {prefill_speedup:<15.2f}x {decode_speedup:<15.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark FlashInfer vs eager SDPA for long context inference'
    )
    parser.add_argument(
        '--models', nargs='+', default=list(MODEL_CONFIGS.keys()),
        help='Model configurations to benchmark'
    )
    parser.add_argument(
        '--context-lengths', nargs='+', type=int, default=DEFAULT_CONTEXT_LENGTHS,
        help='Context lengths to test'
    )
    parser.add_argument(
        '--batch-sizes', nargs='+', type=int, default=DEFAULT_BATCH_SIZES,
        help='Batch sizes to test'
    )
    parser.add_argument(
        '--backends', nargs='+', default=DEFAULT_BACKENDS,
        choices=['flashinfer', 'eager', 'pytorch'],
        help='Backends to benchmark'
    )
    parser.add_argument(
        '--dtype', default='float16',
        choices=['float16', 'bfloat16', 'float32'],
        help='Data type for tensors'
    )
    parser.add_argument(
        '--output', '-o', default=None,
        help='Output CSV file for results'
    )
    parser.add_argument(
        '--json', '-j', default=None,
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick benchmark with reduced configurations'
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        return 1
    
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Parse dtype
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    # Quick mode: reduced configurations
    if args.quick:
        args.models = ['Llama-3.2-1B']
        args.context_lengths = [2048, 4096]
        args.batch_sizes = [1]
    
    # Run benchmarks
    results = run_all_benchmarks(
        model_names=args.models,
        context_lengths=args.context_lengths,
        batch_sizes=args.batch_sizes,
        backends=args.backends,
        dtype=dtype,
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    if args.output:
        save_results_csv(results, args.output)
    
    if args.json:
        save_results_json(results, args.json)
    
    # Default output if no output specified
    if not args.output and not args.json:
        default_csv = 'benchmark_long_context_results.csv'
        default_json = 'benchmark_long_context_results.json'
        save_results_csv(results, default_csv)
        save_results_json(results, default_json)
    
    return 0


if __name__ == "__main__":
    exit(main())
