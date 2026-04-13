"""Forward-pass-only benchmark for H2O KV cache.

Runs the same training setup but patches loss.backward() to be a no-op,
so iter time ≈ forward pass time only.

Usage:
  cd ${KEYS_VALUES_PATH}
  CUDA_VISIBLE_DEVICES="0" \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  python scripts/benchmark_forward_only.py
"""

import torch


def patch_backward():
    """Monkey-patch LossValue.backward to be a no-op.

    With cpu_offload=True, the training loop calls loss.backward()
    where loss is a LossValue (custom tensor subclass). We replace
    backward with a no-op so iter time = forward time only.
    """
    from keys_values.kvcache.gradient.main import LossValue

    original_backward = LossValue.backward

    def noop_backward(self, *args, **kwargs):
        # Skip the actual gradient computation but run cleanup,
        # otherwise the model is in a bad state for the next forward.
        model = self._model
        model.clear()  # Resets status, replay logs, checkpoints, etc.

    LossValue.backward = noop_backward
    print("[benchmark] Patched LossValue.backward -> no-op (forward-only mode)")


def main():
    import sys
    import os

    print("=" * 60)
    print("FORWARD-ONLY BENCHMARK (backward is no-op)")
    print("=" * 60)

    import keys_values

    print(f"keys_values: {keys_values.__file__}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        from keys_values.flashinfer_wrapper import FlashInferSDPA

        wrapper = FlashInferSDPA()
        print(f"FlashInfer available: {wrapper.available}")
        backend = "FlashInfer + Triton"
    except ImportError:
        print("FlashInfer: not available (FlexAttention baseline)")
        backend = "Double FlexAttention"

    print(f"Backend: {backend}")
    print("=" * 60)

    # Patch backward before anything runs
    patch_backward()

    # Set up CLI args (same as training, but no eval to save time)
    out_dir = os.environ.get("BENCH_OUT_DIR", "/tmp/bench_forward_only")
    os.makedirs(out_dir, exist_ok=True)
    if "KEYSVALS_LOG_DIR" not in os.environ:
        os.environ["KEYSVALS_LOG_DIR"] = f"{out_dir}/logs"

    sys.argv = [
        "benchmark_forward_only.py",
        "finetune_long_lora",
        "Qwen/Qwen3-4B-Instruct-2507",
        "--out_dir",
        out_dir,
        "--precision",
        "bf16-true",
        "--verbose",
        "some",
        "--data",
        "Helmet",
        "--data.dataset_key",
        "nq",
        "--data.max_length",
        "64k",
        "--data.trainloader_longest_first",
        "True",
        "--train.save_interval",
        "100",
        "--train.micro_batch_size",
        "2",
        "--train.global_batch_size",
        "2",
        "--train.max_steps",
        "6",
        "--eval.interval",
        "100",
        "--eval.initial_validation",
        "False",
        "--eval.final_validation",
        "False",
        "--attention_forward_temp_size_gb",
        "2",
        "--kv_cache.cache_length",
        "32768",
        "--kv_cache.chunk_size",
        "2048",
        "--kv_cache.name",
        "h2o-torch-quantized8",
        "--kv_cache.cpu_offload",
        "True",
        "--grad.layers_per_cell",
        "1",
    ]

    # Run through the normal CLI - with backward patched out,
    # the reported iter times will be forward-only
    from keys_values.__main__ import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
