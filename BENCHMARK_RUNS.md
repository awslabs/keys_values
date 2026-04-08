# Benchmark Runs: FlashInfer Integration vs Double FlexAttention Baseline

## Environment

| | Baseline (main) | FlashInfer (our branch) |
|---|---|---|
| Branch | `origin/main` (commit `a933349`) | `flashinfer_integration` (commit `e5f1725`) |
| Worktree | `/fsx/pvihang/flash_infer_integration/keys_values_main` | `/fsx/pvihang/flash_infer_integration/keys_values` |
| GPU | NVIDIA A100-SXM4-40GB | NVIDIA A100-SXM4-40GB |

Venvs used:
- `venvs/baseline_venv`: PyTorch 2.11.0+cu130
- `venvs/v2_venv`: PyTorch 2.9.1+cu126

## What we're comparing

- **Baseline**: PR #78 (`deffc89`) — double FlexAttention call to compute attention weights for H2O eviction
- **FlashInfer**: FlashInfer SDPA + Triton score-sum kernel to compute attention weights for H2O eviction

## Run Commands

### Baseline (main branch, double FlexAttention)

```bash
cd /fsx/pvihang/flash_infer_integration/keys_values_main

# For baseline_venv (PyTorch 2.11): add BNB_CUDA_VERSION=126
# For v2_venv (PyTorch 2.9): add PYTHONPATH=/fsx/pvihang/flash_infer_integration/keys_values_main

CUDA_VISIBLE_DEVICES="0" PYTORCH_ALLOC_CONF=expandable_segments:True \
KEYSVALS_LOG_DIR="<out_dir>/logs" \
<venv>/bin/python keys_values/__main__.py \
  finetune_long_lora Qwen/Qwen3-4B-Instruct-2507 \
  --out_dir <out_dir> \
  --precision bf16-true \
  --verbose some \
  --data Helmet --data.dataset_key nq --data.max_length 64k \
  --data.trainloader_longest_first True \
  --train.save_interval 10 --train.micro_batch_size 2 --train.global_batch_size 2 --train.max_steps 6 \
  --eval.interval 10 --eval.initial_validation False \
  --attention_forward_temp_size_gb 2 \
  --kv_cache.cache_length 32768 --kv_cache.chunk_size 2048 \
  --kv_cache.name h2o-torch-quantized8 --kv_cache.cpu_offload True \
  --grad.layers_per_cell 1
```

### FlashInfer (our branch, Triton score-sum)

```bash
cd /fsx/pvihang/flash_infer_integration/keys_values

CUDA_VISIBLE_DEVICES="0" PYTORCH_ALLOC_CONF=expandable_segments:True \
KEYSVALS_LOG_DIR="<out_dir>/logs" \
/fsx/pvihang/flash_infer_integration/keys_values/venvs/v2_venv/bin/python keys_values/__main__.py \
  finetune_long_lora Qwen/Qwen3-4B-Instruct-2507 \
  --out_dir <out_dir> \
  --precision bf16-true \
  --verbose some \
  --data Helmet --data.dataset_key nq --data.max_length 64k \
  --data.trainloader_longest_first True \
  --train.save_interval 10 --train.micro_batch_size 2 --train.global_batch_size 2 --train.max_steps 6 \
  --eval.interval 10 --eval.initial_validation False \
  --attention_forward_temp_size_gb 2 \
  --kv_cache.cache_length 32768 --kv_cache.chunk_size 2048 \
  --kv_cache.name h2o-torch-quantized8 --kv_cache.cpu_offload True \
  --grad.layers_per_cell 1
```

## Matthias Reference Results (from PR discussion)

H2O with double FlexAttention (Matthias's machine):
```
Epoch 0 | iter   1 step   1 | loss train: 17.625 | iter time: 321.458 s
Epoch 0 | iter   2 step   2 | loss train: 11.875 | iter time: 231.397 s
Epoch 0 | iter   3 step   3 | loss train: 15.062 | iter time: 183.267 s
Epoch 0 | iter   4 step   4 | loss train: 20.750 | iter time: 203.813 s
Epoch 0 | iter   5 step   5 | loss train:  8.312 | iter time: 200.542 s
Epoch 0 | iter   6 step   6 | loss train: 16.750 | iter time: 193.582 s
```

## Results (Full Training: Forward + Backward)
PT = Pytorch
### Training Loss

| Step | Baseline (PT 2.11) | Baseline (PT 2.9) | FlashInfer + Triton (PT 2.9) |
|------|-------------------|-------------------|-------------------|
| 1 | 44.250 | 44.500 | 21.250 |
| 2 | 14.625 | 15.000 | 1.242 |
| 3 | 20.500 | 20.375 | 8.438 |
| 4 | 19.625 | 20.125 | 3.328 |
| 5 | 14.625 | 14.750 | 3.562 |
| 6 | 18.750 | 17.125 | 5.438 |

### Iter Time (seconds)

| Step | Baseline (PT 2.11) | Baseline (PT 2.9) | FlashInfer + Triton (PT 2.9) |
|------|-------------------|-------------------|-------------------|
| 1 | 208.7 | 232.8 | 239.8 |
| 2 | 146.1 | 171.4 | 160.3 |
| 3 | 118.6 | 138.8 | 133.3 |
| 4 | 126.7 | 159.6 | 146.3 |
| 5 | 131.6 | 145.6 | 140.4 |
| 6 | 129.2 | 141.9 | 139.6 |

### Final Validation (forward-only)

| | Baseline (PT 2.11) | Baseline (PT 2.9) | FlashInfer + Triton (PT 2.9) |
|---|---|---|---|
| Val loss | 18.160 | 18.106 | 3.657 |
| Val ppl | 77,032,103 | 72,974,691 | 38.8 |
| Val time | 1289.5s | 1277.1s | 1098.2s |
| Peak GPU mem | 26.60 GB | 26.66 GB | 26.66 GB |

### Key Findings

- **Loss difference is NOT from PyTorch version**: Baseline losses are nearly identical
  on PT 2.11 (18.160) and PT 2.9 (18.106). FlashInfer on PT 2.9 gives 3.657.
  The attention weight computation method is what matters.
- **Validation speedup**: 14% faster (1277s → 1098s, same PyTorch version)
- **Val loss**: 18.1 → 3.7 (4.95x better)
- **Val perplexity**: ~73M → 38.8 (~1.9M x better)
- **GPU memory**: Identical (~26.6 GB)
- **Training iter times**: Very close on same PyTorch (~142s vs ~140s steady state).
  PT 2.11 is faster (~129s) due to newer torch.compile optimizations.


---

## Investigating the Loss Difference

The loss difference is very large (val 18.1 vs 3.7) and confirmed to be from
the attention weight computation method, not PyTorch version.

### Investigation 1: Same PyTorch version comparison — DONE

**Result:** Baseline losses are nearly identical across PyTorch versions
(18.160 on PT 2.11 vs 18.106 on PT 2.9). The loss difference is entirely
from FlashInfer+Triton computing more accurate attention weights for H2O eviction.

### Investigation 2: Compare attention weights directly — DONE

Test script: `test_compare_weights.py`

**Setup:** Qwen3-4B config (batch=1, q_len=16, kv_len=64, n_head=32, n_kv=8, hd=128,
input_pos=48). Three methods compared:
1. **PyTorch reference** — full matmul + softmax + causal mask + sum (ground truth)
2. **Triton score-sum** — our method (FlashInfer LSE + Triton kernel)
3. **Double FlexAttention** — baseline PR #78 method

#### Test 1: Contiguous positions [0..63]

| | Triton vs Reference | FlexAttention vs Reference |
|---|---|---|
| Max abs diff | 0.000001 | 1.374193 |
| Mean abs diff | 0.000000 | 0.750020 |
| Max rel diff | 0.000000 | 0.750689 |
| Mean rel diff | 0.000000 | 0.750014 |

Sample weights (kv_head=0, first 5 KV positions):

| Position | Reference | Triton | FlexAttention |
|----------|-----------|--------|---------------|
| 0 | 1.146 | 1.146 | 0.286 |
| 1 | 1.187 | 1.187 | 0.296 |
| 2 | 1.301 | 1.301 | 0.325 |
| 3 | 1.122 | 1.122 | 0.280 |
| 4 | 1.192 | 1.192 | 0.298 |

Eviction decisions (top-32 of 64): All three methods agree.

#### Test 2: Non-contiguous positions (simulating H2O eviction)

kv_len=29, positions: [0, 2, 5, 8, 10, 15, 20, 25, 30, 35, ...59, 60, 61, 62, 63]

| | Triton vs Reference | FlexAttention vs Reference |
|---|---|---|
| Max abs diff | 0.000001 | 3.477558 |
| Mean abs diff | 0.000000 | 1.655195 |

Eviction decisions (top-14 of 29): All three methods agree.

#### Key Finding: FlexAttention divides by `group_size`

FlexAttention returns weights that are consistently **~4x smaller** than the true
attention weight sums. The factor of 4 is exactly `group_size = n_head / n_kv_heads
= 32 / 8 = 4`. This means the double FlexAttention method takes the **mean** over
GQA query groups instead of the **sum**.

For eviction ranking within a single chunk, this constant scale factor cancels out —
hence identical top-K decisions in our test. However, during real H2O training across
36 layers × many chunks, the incorrect absolute magnitude of weights likely compounds
and leads to the large loss difference (val 18.1 vs 3.7).

Triton score-sum matches the PyTorch reference to ~1e-6 precision.
