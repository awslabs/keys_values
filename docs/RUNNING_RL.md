# Running the GRPO (RLVR) experiments on your own GPUs

Install the `rl-longproc` branch and run the GRPO arms we compare (dense cache
vs. bounded H2O cache) on a p4d/p4de (8 x A100, 40 or 80 GB, sm_80, Ubuntu).
One run uses one GPU. This is what `scripts/provision_worker.sh` and the job
snippets from `scripts/queue_rlvr_jobs.sh` do, minus our S3 job queue. The
driver is `examples/grpo_rlvr.py`; `-h` is the reference for every flag here.

## 1. Install

```bash
git clone https://github.com/mmjerge/keys_values.git ~/keys_values
cd ~/keys_values && git checkout rl-longproc
python3 -m venv ~/venv && source ~/venv/bin/activate   # Python 3.12 on Ubuntu 24.04
pip install --upgrade pip
pip install -r scripts/requirements-worker.txt          # frozen known-good stack (torch 2.13.0, CUDA 13 wheels)
pip install -e .
pip install bitsandbytes==0.49.1 math-verify==0.8.0    # paged AdamW 8-bit; symbolic answer verifier
```

The torch pin is a CUDA 13 build: `nvidia-smi` must show `CUDA Version: 13.x`
or newer (the current Deep Learning Base AMI for Ubuntu 24.04 does).
`math-verify` is optional in the code but not in practice: without it the
reward falls back to string matching and the driver prints
`math_verify: NO (string match only)` at start-up. Check for `math_verify: yes`.

### FlashInfer kernels (required for the H2O arm to be fast)

The H2O cache needs per-slot attention weights during decode. Without the
vendored FlashInfer extension it falls back to eager SDPA, which is several
times slower. `build_ext.py` already targets sm_80 and sm_90, so on A100 no
patching is needed (our L40S workers had to add sm_89):

```bash
cd ~/keys_values && source ~/venv/bin/activate
pip install flashinfer-python           # headers only; the kernels are vendored in keys_values/csrc
unset LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH   # nvcc from the system CUDA toolkit
python build_ext.py > ~/flashinfer_build.log 2>&1
python -c "import torch, keys_values._flashinfer_ops; print('flashinfer OK')"
```

The `.so` is gitignored and survives `git pull`; rebuild if anything under
`keys_values/csrc/` changes. Build once per machine, before launching runs
(concurrent builds corrupt the `.so`). If the import fails the driver still
runs, with a loud warning and eager decode; do not use such a run for timing.

### Environment for every run

```bash
unset LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`LD_LIBRARY_PATH`: the AMI puts the system CUDA libs ahead of torch's bundled
ones, and cuDNN SDPA then aborts with `cublasLtGetVersion` (exit 134). Torch
finds its own libraries without it. `expandable_segments` reduces allocator
fragmentation in the long chunked backward; the 32k runs were tuned with it on.

### Model and data

Qwen3 checkpoints and the datasets (Polaris-53K, AIME 2024/2025) are public
on the Hugging Face Hub; no token is needed. For a gated repo pass
`--access-token <HF token>` or set `HF_TOKEN` in the environment (never in a
script you share). Pre-download the model once per machine so eight parallel
runs do not race on the same files:

```bash
HF_HUB_DISABLE_XET=1 litgpt download Qwen/Qwen3-1.7B
```

## 2. One run = one GPU

The driver is single-device. Pin it with `CUDA_VISIBLE_DEVICES=<N>`; eight
independent processes on a p4d is the intended way to use the machine, e.g.

```bash
for g in 0 1 2 3 4 5 6 7; do
  OUT=~/runs/<name>_gpu$g; mkdir -p $OUT
  CUDA_VISIBLE_DEVICES=$g nohup python examples/grpo_rlvr.py ... --seed $g --out-dir $OUT > $OUT/job.log 2>&1 &
done
```

Use `tmux` or `nohup` as above; a 32k run takes days (see section 3). Never
start two runs on one GPU: both OOM at model load.

Output goes to `--out-dir`:

- `job.log`: the redirected stdout/stderr: start-up lines (`math_verify`,
  attention backend, `polaris: difficulty in [1, 2]/8 keeps 12415/53291`),
  one `step` line per optimizer update, and `[eval @ ...]` lines.
- `history.json`: the same step and eval records as JSON, rewritten after
  every step, so a killed run keeps its history up to the last step.
- `final.pt`: bf16 state dict of the trained policy (about 3.4 GB for 1.7B).
  `--save-intermediate` also writes `step<N>.pt` at each eval; `--eval-only`
  writes `eval_base.json` instead.

A step line looks like

```
step   70 | reward 0.812 | 3494.3s | len 11263 spread 0.75 overlong_frac 0.00 raw 0.812 | dev peak 25.0G | grad flexattention
```

- `reward`: mean reward of this update's rollouts (`--prompts-per-update`
  prompts x `--group-size` completions). With the default
  `--overlong-penalty 0` it equals `raw`.
- `<seconds>s`: wall time of the step (all rollouts + gradient pass).
- `len`: mean completion length in tokens.
- `spread`: fraction of groups whose rewards are not all equal. Only those
  groups give GRPO a gradient; `spread 0.00` means a wasted step.
- `overlong_frac`: fraction of rollouts that used the full `--max-new-tokens`
  budget without emitting EOS (truncated mid-reasoning).
- `raw`: correctness-only mean reward, comparable across runs with and
  without `--overlong-penalty`.
- `dev peak`: peak device memory of the gradient phase in GiB
  (`grad_peak_device_mib` in `history.json`). Use it to judge headroom.
- `grad`: SDPA mode of the gradient cells: `flexattention` under
  `--attn auto`, `qpadded-pytorch` under `--attn eager`. Absent on the dense
  arm (`--dense-baseline` does not use the chunked cells).

An eval line, one per eval set:

```
[eval @ step 100] aime24 avg@4 = 0.367 (n=30) | mean len 13412 | truncated 0.50 | 17409s
```

`avg@k` is mean accuracy over `k` samples per problem at temperature 0.6 /
top-p 0.95; `truncated` is the fraction of samples that ran out of budget
inside `<think>`. Eval is slow: one AIME set (n=30) at a 32k cutoff takes
several hours, which is why the recipe below evaluates every 100 steps and
skips the step-0 eval.

## 3. The current recipe ("aa3z32k") and its arms

Qwen3-1.7B, Polaris problems with difficulty 1/8 to 2/8 (the hard end the
reference model rarely solves, excluding the never-solved 0/8), 32k generation
cutoff, groups of 8, 4 prompts per update, lr 5e-6, paged 8-bit AdamW, chunk
1024, 200 steps, eval AIME24 + AIME25 (n=30, avg@4) at step 100 and at the end.

```bash
COMMON="--device cuda --model Qwen/Qwen3-1.7B \
  --train-dataset polaris --difficulty-min 1 --difficulty-max 2 \
  --max-new-tokens 32768 --chunk-size 1024 \
  --steps 200 --group-size 8 --prompts-per-update 4 --adv-mode grpo \
  --lr 5e-6 --optimizer paged_adamw8bit \
  --eval-sets aime24,aime25 --n-eval 30 --eval-samples 4 --final-eval-samples 4 \
  --eval-every 100 --skip-initial-eval"
```

Dense baseline (cache sized to prompt + generation; 1024 covers the prompt):

```bash
python examples/grpo_rlvr.py $COMMON \
  --kv-cache-name dense-default --cache-length 33792 --dense-baseline \
  --seed 1 --out-dir ~/runs/aa3z32k_dense_s1
```

H2O with K = 16384 slots (half the cutoff):

```bash
python examples/grpo_rlvr.py $COMMON \
  --kv-cache-name h2o-default --cache-length 16384 --grad-micro-batch 2 \
  --seed 1 --out-dir ~/runs/aa3z32k_h2o16384_s1
```

H2O with K = 32768 (bounded cache as large as the cutoff; eviction only
happens once prompt + generation exceed K):

```bash
python examples/grpo_rlvr.py $COMMON \
  --kv-cache-name h2o-default --cache-length 32768 --grad-micro-batch 1 \
  --seed 1 --out-dir ~/runs/aa3z32k_h2o32768_s1
```

Repeat with `--seed 2` (and more) for paired comparisons; seeds also shuffle
the training problems, so arms with the same seed see the same problem order.

Observed cost on our L40S (44 GiB usable): dense 32k peaks at ~32 GiB and a
step takes 1.5 to 4 h (growing with completion length); H2O K=16384 with
`--grad-micro-batch 2` peaks at ~22 GiB and takes 1.5 to 2.5 h per step; H2O
K=32768 with `--grad-micro-batch 1` peaks at ~22.5 GiB. 200 steps plus two
eval passes is one to two weeks per run; A100s should be somewhat faster.

### Memory levers

- `--grad-micro-batch M`: sequences per chunked backward pass (H2O arm only;
  the dense arm already does one sequence at a time). Every memory term of the
  gradient phase scales with the batch, so M=2 or M=1 divides the peak by
  about G/M. Exact (same gradients and loss as one pass), only slower. This is
  what made K=16384 and K=32768 fit a 44 GiB GPU for 1.7B.
- `--chunk-size C`: query chunk of prefill and of the gradient cells. Under
  `--attn auto` the cells run FlexAttention with C-long queries and C only
  trades compile shapes against per-chunk overhead. Under `--attn eager` the
  cells zero-pad the query to K and keep a K x K problem per chunk (a K^2/C
  memory term), so a smaller C there makes things *worse*. Keep 1024.
- `--backward-tmp-gb` does not help: it bounds the temporary array of the
  eager attention implementation, which the gradient cells do not use.
- `--layers-per-cell` is already at its minimum (1).

### `--attn` semantics

- `auto` (default): FlashInfer decode if the extension is built (else eager,
  with a warning); FlexAttention in the gradient cells.
- `eager`: eager decode and query-padded PyTorch SDPA in the cells, the
  pre-Flex path kept as the A/B baseline. Slower, and far more memory at
  K >= 16384.
- `flex`: FlexAttention everywhere including decode, which recompiles per
  `kv_len`; experiments only.
- `--disable-flashinfer` forces eager decode even if the extension is built
  (and, with `--attn auto`, the eager cells too).

### Larger cutoffs and models (80 GB)

- Cutoff 49152 (48k) with K=24576, or 65536 (64k) with K=32768: change
  `--max-new-tokens` and `--cache-length`; keep `--grad-micro-batch 1`. For
  the dense arm at 64k use `--cache-length 66560`. The dense KV cache alone at
  64k x batch 8 is about 60 GB for 1.7B, so dense 64k is 80 GB only and tight.
- Qwen/Qwen3-4B and Qwen/Qwen3-8B: change `--model` only. Start with
  `--grad-micro-batch 1` and watch `dev peak` on the first steps. `final.pt`
  grows to ~8 GB (4B) / ~16 GB (8B). Step and eval time grow with cutoff and
  model size; plan for multi-week runs at 64k.

## 4. Runs we want

| priority | run | GPU memory |
|---|---|---|
| 1 | 1.7B, 32k cutoff, dense, seeds 1 and 2 | 40 GB tight (~32 GiB peak); 80 GB preferred |
| 1 | 1.7B, 32k cutoff, H2O K=16384 `--grad-micro-batch 2`, seeds 1 and 2 | 40 GB ok (~22 GiB) |
| 2 | 1.7B, 32k cutoff, H2O K=32768 `--grad-micro-batch 1`, seeds 1 and 2 | 40 GB ok (~22.5 GiB) |
| 2 | Qwen3-4B, 32k cutoff, dense vs. H2O K=16384 `--grad-micro-batch 1`, seed 1 | 80 GB |
| 3 | 1.7B, 64k cutoff: dense (`--cache-length 66560`) vs. H2O K=32768 | 80 GB |

Do not rerun 1.7B H2O K=8192 at the 32k cutoff: it is known to fail. The
seed-0 runs of the first two rows are ours; start your seeds at 1.

## 5. Sending results back

Only `history.json` and `job.log` are needed (skip `final.pt` unless asked):

```bash
tar czf aa3z32k_dense_s1.tgz -C ~/runs aa3z32k_dense_s1 --exclude='*.pt'
```

Share the tarball, or `aws s3 cp` it to a bucket we will give you. A partial
run is still useful: `history.json` is rewritten after every step.

## 6. Known issues

- **Upstream issue #159** (FlashInfer decode without attention weights
  crashes for GQA group sizes outside {1,2,3,4,6,8}): hits the *dense* arm
  for Qwen2.5-7B-Instruct (group 7) and Qwen2.5-14B/32B (group 5). Qwen3
  1.7B/4B/8B have group 2 or 4 and are unaffected. H2O is never affected.
- **Upstream issue #176**: `test/kvcache/test_h2o.py::test_token_pos_and_pos_log`
  fails deterministically in the CUDA fp16 case on `main`. Not caused by the
  RL code; our runs are bf16.
- `test/attention/test_flex_attention.py::test_comparison_with_attn_weights`
  has two half-precision cases that miss the 3e-4 tolerance on GPU. Harmless;
  the rest of the file passes and a 2-step `--attn auto` vs `--attn eager`
  run gives the same rewards and lengths.
- **Ubuntu `unattended-upgrades` + `needrestart` kills long jobs.** The daily
  upgrade runs `needrestart`, which restarts services whose libraries changed;
  we lost multi-day runs this way while training ran under a systemd unit.
  If you run under systemd, exclude the unit:
  `echo '$nrconf{override_rc}{qr(^<unit-name>)} = 0;' | sudo tee /etc/needrestart/conf.d/<unit-name>.conf`.
  A plain `tmux`/`nohup` process is not restarted by needrestart; still
  consider disabling unattended upgrades on a box dedicated to week-long runs.
- **No checkpoint/resume.** A killed run is lost; only `history.json` up to
  the last step survives. `--save-intermediate` saves policy weights at each
  eval, and `--checkpoint <file> --eval-only` can score them, but training
  cannot be resumed from them.
