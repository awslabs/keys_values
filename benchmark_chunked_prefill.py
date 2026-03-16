"""
Comprehensive benchmark for chunked prefill / non-square attention (q_len < kv_len).

Measures:
- Phase 1 (FlashInfer prefill -> O + LSE) and Phase 2 (Triton kernel -> weights) separately
- Scaling across q_len, kv_len, and batch_size dimensions
- Comparison against eager fallback and PyTorch SDPA baseline
- Peak GPU memory usage
- Sliding window impact

Model config: Qwen3-4B (n_head=32, n_kv_heads=8, head_dim=128)
"""

import gc
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from keys_values.flashinfer_wrapper import FlashInferSDPA


@dataclass
class BenchResult:
    """Result of a single benchmark run."""
    q_len: int
    kv_len: int
    batch_size: int
    backend: str
    total_ms: float = 0.0
    phase1_ms: float = 0.0  # FlashInfer prefill (O + LSE)
    phase2_ms: float = 0.0  # Triton weight kernel
    peak_mem_mb: float = 0.0
    has_weights: bool = False
    oom: bool = False
    error: str = ""


def _cleanup():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()


def _run_flashinfer_phased(
    query, key, value, scale_factor, input_pos,
    sliding_window_size=None, warmup=3, iterations=10,
):
    """Run two-phase FlashInfer with per-phase timing."""
    from keys_values import flashinfer_ops
    from keys_values.triton_kernels import compute_weights_from_lse_triton

    batch_size, n_head, q_len, head_size = query.shape

    # Prepare Phase 1 inputs
    query_t = query.transpose(1, 2).contiguous()
    key_t = key.transpose(1, 2).contiguous()
    value_t = value.transpose(1, 2).contiguous()
    input_pos_tensor = torch.tensor(
        [input_pos] * batch_size, device=query.device, dtype=torch.int32
    )
    window_size = sliding_window_size if sliding_window_size is not None else -1

    # Warmup both phases together
    for _ in range(warmup):
        out_t, _, lse = flashinfer_ops.sdpa_prefill(
            query=query_t, key=key_t, value=value_t, scale=scale_factor,
            token_positions=None, input_pos=input_pos_tensor,
            sliding_window_size=window_size, causal=True,
            return_weights=False, return_lse=True,
        )
        _ = compute_weights_from_lse_triton(
            query, key, lse, scale_factor, input_pos, sliding_window_size
        )
        torch.cuda.synchronize()

    # Benchmark Phase 1
    torch.cuda.synchronize()
    p1_start = time.perf_counter()
    for _ in range(iterations):
        out_t, _, lse = flashinfer_ops.sdpa_prefill(
            query=query_t, key=key_t, value=value_t, scale=scale_factor,
            token_positions=None, input_pos=input_pos_tensor,
            sliding_window_size=window_size, causal=True,
            return_weights=False, return_lse=True,
        )
        torch.cuda.synchronize()
    p1_end = time.perf_counter()

    # Benchmark Phase 2
    torch.cuda.synchronize()
    p2_start = time.perf_counter()
    for _ in range(iterations):
        weights = compute_weights_from_lse_triton(
            query, key, lse, scale_factor, input_pos, sliding_window_size
        )
        torch.cuda.synchronize()
    p2_end = time.perf_counter()

    # Benchmark total (both phases together)
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    for _ in range(iterations):
        out_t, _, lse = flashinfer_ops.sdpa_prefill(
            query=query_t, key=key_t, value=value_t, scale=scale_factor,
            token_positions=None, input_pos=input_pos_tensor,
            sliding_window_size=window_size, causal=True,
            return_weights=False, return_lse=True,
        )
        weights = compute_weights_from_lse_triton(
            query, key, lse, scale_factor, input_pos, sliding_window_size
        )
        torch.cuda.synchronize()
    total_end = time.perf_counter()

    phase1_ms = (p1_end - p1_start) * 1000 / iterations
    phase2_ms = (p2_end - p2_start) * 1000 / iterations
    total_ms = (total_end - total_start) * 1000 / iterations

    return total_ms, phase1_ms, phase2_ms, True


def _run_eager(query, key, value, scale_factor, input_pos, wrapper, warmup=3, iterations=10):
    """Run eager fallback."""
    def run():
        return wrapper._fallback_sdpa(
            query, key, value, scale_factor,
            return_attn_weights=True, input_pos=input_pos,
        )

    for _ in range(warmup):
        _ = run()
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        out, weights = run()
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000 / iterations, weights is not None


def _run_pytorch(query, key, value, scale_factor, n_query_groups, warmup=3, iterations=10):
    """Run native PyTorch SDPA (no weights)."""
    batch_size, n_head, q_len, head_size = query.shape
    kv_len = key.shape[2]
    q_per_kv = n_head // n_query_groups

    key_expanded = key.unsqueeze(2).expand(-1, -1, q_per_kv, -1, -1)
    key_expanded = key_expanded.reshape(batch_size, n_head, kv_len, head_size)
    value_expanded = value.unsqueeze(2).expand(-1, -1, q_per_kv, -1, -1)
    value_expanded = value_expanded.reshape(batch_size, n_head, kv_len, head_size)

    def run():
        return torch.nn.functional.scaled_dot_product_attention(
            query, key_expanded, value_expanded,
            scale=scale_factor, is_causal=False,
        )

    for _ in range(warmup):
        _ = run()
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = run()
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000 / iterations


def benchmark_config(
    n_head, n_query_groups, head_size, q_len, kv_len, batch_size,
    backends, sliding_window_size=None, warmup=3, iterations=10,
    dtype=torch.float16,
) -> list[BenchResult]:
    """Benchmark a single (q_len, kv_len, bs) configuration across backends."""
    device = "cuda"
    scale_factor = 1.0 / math.sqrt(head_size)
    input_pos = kv_len - q_len
    results = []

    # Allocate tensors
    query = torch.randn(batch_size, n_head, q_len, head_size, dtype=dtype, device=device)
    key = torch.randn(batch_size, n_query_groups, kv_len, head_size, dtype=dtype, device=device)
    value = torch.randn(batch_size, n_query_groups, kv_len, head_size, dtype=dtype, device=device)

    wrapper = FlashInferSDPA()
    import logging
    logging.getLogger('keys_values.flashinfer_wrapper').setLevel(logging.ERROR)

    for backend in backends:
        _cleanup()
        torch.cuda.reset_peak_memory_stats()

        r = BenchResult(q_len=q_len, kv_len=kv_len, batch_size=batch_size, backend=backend)
        try:
            if backend == 'flashinfer':
                total, p1, p2, has_w = _run_flashinfer_phased(
                    query, key, value, scale_factor, input_pos,
                    sliding_window_size=sliding_window_size,
                    warmup=warmup, iterations=iterations,
                )
                r.total_ms = total
                r.phase1_ms = p1
                r.phase2_ms = p2
                r.has_weights = has_w

            elif backend == 'eager':
                total, has_w = _run_eager(
                    query, key, value, scale_factor, input_pos, wrapper,
                    warmup=warmup, iterations=iterations,
                )
                r.total_ms = total
                r.has_weights = has_w

            elif backend == 'pytorch':
                total = _run_pytorch(
                    query, key, value, scale_factor, n_query_groups,
                    warmup=warmup, iterations=iterations,
                )
                r.total_ms = total
                r.has_weights = False

            r.peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        except torch.cuda.OutOfMemoryError:
            _cleanup()
            r.oom = True
        except Exception as e:
            r.error = str(e)[:120]

        results.append(r)

    del query, key, value
    _cleanup()
    return results


def _fmt_ms(val, oom=False, error=""):
    if oom:
        return "OOM"
    if error:
        return "ERR"
    return f"{val:.2f}"


def _fmt_ratio(a, b):
    """Compute b/a ratio (speedup of a over b). Returns 'N/A' if either is missing."""
    if a <= 0 or b <= 0:
        return "N/A"
    return f"{b/a:.2f}x"


def print_section_header(title):
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")


def run_scaling_suite(
    title, configs, n_head, n_query_groups, head_size, backends,
    varying_param, warmup=3, iterations=10, dtype=torch.float16,
):
    """Run a suite of benchmarks and print a formatted table.

    Args:
        configs: list of (q_len, kv_len, batch_size) tuples
        varying_param: name of the varying column ('q_len', 'kv_len', 'batch_size')
    """
    print_section_header(title)

    # Header
    header_col = {'q_len': 'q_len', 'kv_len': 'kv_len', 'batch_size': 'bs'}[varying_param]
    print(f"\n  {header_col:>8}  {'total':>8}  {'phase1':>8}  {'phase2':>8}  "
          f"{'eager':>8}  {'pytorch':>8}  {'fi/eager':>10}  {'fi/pt':>10}  "
          f"{'fi_mem':>10}  {'ea_mem':>10}")
    print(f"  {'':-<8}  {'':-<8}  {'':-<8}  {'':-<8}  "
          f"{'':-<8}  {'':-<8}  {'':-<10}  {'':-<10}  "
          f"{'':-<10}  {'':-<10}")

    all_results = []
    for q_len, kv_len, bs in configs:
        results = benchmark_config(
            n_head, n_query_groups, head_size, q_len, kv_len, bs,
            backends, warmup=warmup, iterations=iterations, dtype=dtype,
        )
        result_map = {r.backend: r for r in results}
        all_results.append(result_map)

        fi = result_map.get('flashinfer')
        ea = result_map.get('eager')
        pt = result_map.get('pytorch')

        vary_val = {'q_len': q_len, 'kv_len': kv_len, 'batch_size': bs}[varying_param]

        fi_total = _fmt_ms(fi.total_ms, fi.oom, fi.error) if fi else "N/A"
        fi_p1 = _fmt_ms(fi.phase1_ms, fi.oom, fi.error) if fi else "N/A"
        fi_p2 = _fmt_ms(fi.phase2_ms, fi.oom, fi.error) if fi else "N/A"
        ea_total = _fmt_ms(ea.total_ms, ea.oom, ea.error) if ea else "N/A"
        pt_total = _fmt_ms(pt.total_ms, pt.oom, pt.error) if pt else "N/A"

        fi_ok = fi and not fi.oom and not fi.error
        ea_ok = ea and not ea.oom and not ea.error
        pt_ok = pt and not pt.oom and not pt.error

        ratio_ea = _fmt_ratio(fi.total_ms, ea.total_ms) if fi_ok and ea_ok else "N/A"
        ratio_pt = _fmt_ratio(fi.total_ms, pt.total_ms) if fi_ok and pt_ok else "N/A"

        fi_mem = f"{fi.peak_mem_mb:.0f}" if fi_ok else "N/A"
        ea_mem = f"{ea.peak_mem_mb:.0f}" if ea_ok else "N/A"

        print(f"  {vary_val:>8}  {fi_total:>8}  {fi_p1:>8}  {fi_p2:>8}  "
              f"{ea_total:>8}  {pt_total:>8}  {ratio_ea:>10}  {ratio_pt:>10}  "
              f"{fi_mem:>10}  {ea_mem:>10}")

    return all_results


def run_sliding_window_suite(
    n_head, n_query_groups, head_size, q_len, kv_len, batch_size,
    window_sizes, warmup=3, iterations=10, dtype=torch.float16,
):
    """Benchmark sliding window impact on Triton kernel."""
    print_section_header(
        f"Sliding Window Impact (q={q_len}, kv={kv_len}, bs={batch_size})"
    )

    print(f"\n  {'window':>8}  {'total':>8}  {'phase1':>8}  {'phase2':>8}")
    print(f"  {'':-<8}  {'':-<8}  {'':-<8}  {'':-<8}")

    for sw in window_sizes:
        _cleanup()
        scale_factor = 1.0 / math.sqrt(head_size)
        input_pos = kv_len - q_len
        device = "cuda"

        query = torch.randn(batch_size, n_head, q_len, head_size, dtype=dtype, device=device)
        key = torch.randn(batch_size, n_query_groups, kv_len, head_size, dtype=dtype, device=device)
        value = torch.randn(batch_size, n_query_groups, kv_len, head_size, dtype=dtype, device=device)

        try:
            total, p1, p2, _ = _run_flashinfer_phased(
                query, key, value, scale_factor, input_pos,
                sliding_window_size=sw,
                warmup=warmup, iterations=iterations,
            )
            sw_label = str(sw) if sw is not None else "None"
            print(f"  {sw_label:>8}  {total:>8.2f}  {p1:>8.2f}  {p2:>8.2f}")
        except Exception as e:
            sw_label = str(sw) if sw is not None else "None"
            print(f"  {sw_label:>8}  ERROR: {str(e)[:60]}")

        del query, key, value
        _cleanup()


def run_full_model_suite(
    n_head, n_query_groups, head_size, n_layers,
    configs, warmup=2, iterations=5, dtype=torch.float16,
):
    """Simulate full-model attention cost across all decoder layers.

    For each config, allocates per-layer Q/K/V tensors and runs all layers
    sequentially (as in a real forward pass). Measures total wall-clock time
    for all layers combined, capturing memory pressure and L2 cache effects.
    """
    print_section_header(
        f"Full Model Simulation ({n_layers} layers, Qwen3-4B)"
    )
    print(f"  Note: Simulates attention-only cost across all {n_layers} decoder layers.")
    print(f"  Does NOT include FFN, LayerNorm, embeddings, or sampling.\n")

    print(f"  {'q_len':>6} {'kv_len':>7} {'bs':>4}  "
          f"{'fi_total':>10} {'fi_p1':>10} {'fi_p2':>10}  "
          f"{'eager':>10} {'pytorch':>10}  "
          f"{'fi/eager':>10} {'fi/pt':>10}  "
          f"{'fi_mem':>10} {'ea_mem':>10}")
    print(f"  {'':->6} {'':->7} {'':->4}  "
          f"{'':->10} {'':->10} {'':->10}  "
          f"{'':->10} {'':->10}  "
          f"{'':->10} {'':->10}  "
          f"{'':->10} {'':->10}")

    from keys_values import flashinfer_ops
    from keys_values.triton_kernels import compute_weights_from_lse_triton

    device = "cuda"
    scale_factor = 1.0 / math.sqrt(head_size)
    wrapper = FlashInferSDPA()
    import logging
    logging.getLogger('keys_values.flashinfer_wrapper').setLevel(logging.ERROR)

    for q_len, kv_len, batch_size in configs:
        input_pos = kv_len - q_len

        # Allocate per-layer tensors (simulates distinct KV caches per layer)
        layer_data = []
        try:
            for _ in range(n_layers):
                q = torch.randn(batch_size, n_head, q_len, head_size, dtype=dtype, device=device)
                k = torch.randn(batch_size, n_query_groups, kv_len, head_size, dtype=dtype, device=device)
                v = torch.randn(batch_size, n_query_groups, kv_len, head_size, dtype=dtype, device=device)
                layer_data.append((q, k, v))
        except torch.cuda.OutOfMemoryError:
            _cleanup()
            print(f"  {q_len:>6} {kv_len:>7} {batch_size:>4}  OOM allocating {n_layers} layers")
            continue

        # Precompute transformed tensors for flashinfer
        layer_fi = []
        for q, k, v in layer_data:
            qt = q.transpose(1, 2).contiguous()
            kt = k.transpose(1, 2).contiguous()
            vt = v.transpose(1, 2).contiguous()
            layer_fi.append((q, k, qt, kt, vt))
        input_pos_tensor = torch.tensor(
            [input_pos] * batch_size, device=device, dtype=torch.int32
        )

        # Precompute expanded K/V for pytorch
        q_per_kv = n_head // n_query_groups
        layer_pt = []
        try:
            for q, k, v in layer_data:
                ke = k.unsqueeze(2).expand(-1, -1, q_per_kv, -1, -1).reshape(
                    batch_size, n_head, kv_len, head_size
                )
                ve = v.unsqueeze(2).expand(-1, -1, q_per_kv, -1, -1).reshape(
                    batch_size, n_head, kv_len, head_size
                )
                layer_pt.append((q, ke, ve))
        except torch.cuda.OutOfMemoryError:
            layer_pt = []  # Skip pytorch if OOM on expand

        # ── Benchmark: FlashInfer (phased) ──
        fi_total_ms = fi_p1_ms = fi_p2_ms = 0.0
        fi_mem = 0.0
        fi_ok = True
        try:
            # Warmup
            for _ in range(warmup):
                for q, k, qt, kt, vt in layer_fi:
                    out_t, _, lse = flashinfer_ops.sdpa_prefill(
                        query=qt, key=kt, value=vt, scale=scale_factor,
                        token_positions=None, input_pos=input_pos_tensor,
                        sliding_window_size=-1, causal=True,
                        return_weights=False, return_lse=True,
                    )
                    _ = compute_weights_from_lse_triton(
                        q, k, lse, scale_factor, input_pos, None
                    )
                torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()

            # Phase 1 only
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iterations):
                for q, k, qt, kt, vt in layer_fi:
                    out_t, _, lse = flashinfer_ops.sdpa_prefill(
                        query=qt, key=kt, value=vt, scale=scale_factor,
                        token_positions=None, input_pos=input_pos_tensor,
                        sliding_window_size=-1, causal=True,
                        return_weights=False, return_lse=True,
                    )
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            # Phase 2 only (use last lse from phase 1)
            # Run all layers' phase 2
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            for _ in range(iterations):
                for q, k, qt, kt, vt in layer_fi:
                    # Need lse for each layer — run phase 1 to get it
                    out_t, _, lse = flashinfer_ops.sdpa_prefill(
                        query=qt, key=kt, value=vt, scale=scale_factor,
                        token_positions=None, input_pos=input_pos_tensor,
                        sliding_window_size=-1, causal=True,
                        return_weights=False, return_lse=True,
                    )
                    _ = compute_weights_from_lse_triton(
                        q, k, lse, scale_factor, input_pos, None
                    )
                torch.cuda.synchronize()
            t3 = time.perf_counter()

            fi_p1_ms = (t1 - t0) * 1000 / iterations
            fi_total_ms = (t3 - t2) * 1000 / iterations
            fi_p2_ms = fi_total_ms - fi_p1_ms
            fi_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except torch.cuda.OutOfMemoryError:
            _cleanup()
            fi_ok = False
        except Exception as e:
            fi_ok = False

        # ── Benchmark: Eager ──
        ea_ms = 0.0
        ea_mem = 0.0
        ea_ok = True
        try:
            _cleanup()
            torch.cuda.reset_peak_memory_stats()
            # Warmup
            for _ in range(warmup):
                for q, k, v in layer_data:
                    wrapper._fallback_sdpa(
                        q, k, v, scale_factor,
                        return_attn_weights=True, input_pos=input_pos,
                    )
                torch.cuda.synchronize()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iterations):
                for q, k, v in layer_data:
                    wrapper._fallback_sdpa(
                        q, k, v, scale_factor,
                        return_attn_weights=True, input_pos=input_pos,
                    )
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            ea_ms = (t1 - t0) * 1000 / iterations
            ea_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except torch.cuda.OutOfMemoryError:
            _cleanup()
            ea_ok = False
        except Exception:
            ea_ok = False

        # ── Benchmark: PyTorch SDPA ──
        pt_ms = 0.0
        pt_ok = bool(layer_pt)
        if pt_ok:
            try:
                _cleanup()
                # Warmup
                for _ in range(warmup):
                    for q, ke, ve in layer_pt:
                        torch.nn.functional.scaled_dot_product_attention(
                            q, ke, ve, scale=scale_factor, is_causal=False,
                        )
                    torch.cuda.synchronize()

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(iterations):
                    for q, ke, ve in layer_pt:
                        torch.nn.functional.scaled_dot_product_attention(
                            q, ke, ve, scale=scale_factor, is_causal=False,
                        )
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                pt_ms = (t1 - t0) * 1000 / iterations
            except torch.cuda.OutOfMemoryError:
                _cleanup()
                pt_ok = False
            except Exception:
                pt_ok = False

        # Print row
        fi_s = f"{fi_total_ms:.1f}ms" if fi_ok else "OOM"
        p1_s = f"{fi_p1_ms:.1f}ms" if fi_ok else ""
        p2_s = f"{fi_p2_ms:.1f}ms" if fi_ok else ""
        ea_s = f"{ea_ms:.1f}ms" if ea_ok else "OOM"
        pt_s = f"{pt_ms:.1f}ms" if pt_ok else "OOM"
        ratio_ea = f"{ea_ms/fi_total_ms:.2f}x" if fi_ok and ea_ok and fi_total_ms > 0 else "N/A"
        ratio_pt = f"{pt_ms/fi_total_ms:.2f}x" if fi_ok and pt_ok and fi_total_ms > 0 else "N/A"
        fi_m = f"{fi_mem:.0f}" if fi_ok else "N/A"
        ea_m = f"{ea_mem:.0f}" if ea_ok else "N/A"

        print(f"  {q_len:>6} {kv_len:>7} {batch_size:>4}  "
              f"{fi_s:>10} {p1_s:>10} {p2_s:>10}  "
              f"{ea_s:>10} {pt_s:>10}  "
              f"{ratio_ea:>10} {ratio_pt:>10}  "
              f"{fi_m:>10} {ea_m:>10}")

        # Cleanup between configs
        del layer_data, layer_fi, layer_pt
        _cleanup()


def main():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.0f} GB")
    print(f"PyTorch Version: {torch.__version__}")

    # Qwen3-4B config
    N_HEAD = 32
    N_KV_HEADS = 8
    HEAD_DIM = 128
    N_LAYERS = 36
    BACKENDS = ['flashinfer', 'eager', 'pytorch']

    # ─── Suite 1: q_len scaling (fixed kv_len=32768, bs=2) ───
    q_len_configs = [(q, 32768, 2) for q in [128, 256, 512, 1024, 2048, 4096]]
    run_scaling_suite(
        "Suite 1: q_len Scaling (kv_len=32768, bs=2)",
        q_len_configs, N_HEAD, N_KV_HEADS, HEAD_DIM, BACKENDS,
        varying_param='q_len',
    )

    # ─── Suite 2: kv_len scaling (fixed q_len=2048, bs=2) ───
    kv_len_configs = [(2048, kv, 2) for kv in [4096, 8192, 16384, 32768, 65536, 131072]]
    run_scaling_suite(
        "Suite 2: kv_len Scaling (q_len=2048, bs=2)",
        kv_len_configs, N_HEAD, N_KV_HEADS, HEAD_DIM, BACKENDS,
        varying_param='kv_len',
    )

    # ─── Suite 3: batch size scaling (fixed q_len=2048, kv_len=32768) ───
    bs_configs = [(2048, 32768, bs) for bs in [1, 2, 4, 8]]
    run_scaling_suite(
        "Suite 3: Batch Size Scaling (q_len=2048, kv_len=32768)",
        bs_configs, N_HEAD, N_KV_HEADS, HEAD_DIM, BACKENDS,
        varying_param='batch_size',
    )

    # ─── Suite 4: Sliding window impact ───
    run_sliding_window_suite(
        N_HEAD, N_KV_HEADS, HEAD_DIM,
        q_len=2048, kv_len=32768, batch_size=2,
        window_sizes=[None, 512, 1024, 4096, 16384],
    )

    # ─── Suite 5: Full model simulation (36 layers) ───
    full_model_configs = [
        (2048, 8192, 1),
        (2048, 8192, 2),
        (2048, 32768, 1),
        (2048, 32768, 2),
        (2048, 65536, 1),
    ]
    run_full_model_suite(
        N_HEAD, N_KV_HEADS, HEAD_DIM, N_LAYERS,
        full_model_configs, warmup=2, iterations=5,
    )


if __name__ == "__main__":
    main()
