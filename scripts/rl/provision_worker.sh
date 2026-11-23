#!/bin/bash
# Provision a keys_values RL worker (run ON the instance, as ubuntu).
#
# Assumes: Deep Learning Base AMI (Ubuntu 24.04), IAM instance profile
# `kv-worker` attached (S3 read: canonical splits; S3 write: results).
#
#   curl -sL https://raw.githubusercontent.com/mmjerge/keys_values/experiments/scripts/rl/provision_worker.sh | bash -s -- <branch>
#
# Idempotent: safe to re-run.
set -euo pipefail

BRANCH="${1:-experiments}"
REPO_URL="https://github.com/mmjerge/keys_values.git"
MODEL="${KV_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

echo "=== [1/5] repo @ ${BRANCH} ==="
if [ ! -d "$HOME/keys_values/.git" ]; then
    git clone "$REPO_URL" "$HOME/keys_values"
fi
cd "$HOME/keys_values"
git fetch origin -q && git checkout -q "$BRANCH" && git pull -q

echo "=== [2/5] venv + deps ==="
if [ ! -d "$HOME/venv" ]; then
    python3 -m venv "$HOME/venv"
fi
source "$HOME/venv/bin/activate"
pip install -q --upgrade pip
# The package's own install_requires is minimal; the full runtime stack is
# frozen from the known-good reference worker (grpo-bench).
pip install -q -r scripts/rl/requirements-worker.txt
pip install -q -e .
pip install -q awscli

# FlashInfer CUDA extension: the H2O decode path needs attention weights, and
# without this it falls back to eager SDPA (far slower). build_ext.py targets
# sm_80/sm_90; add the host GPU's arch (L40S = 8.9) so the kernels are native.
echo "=== [2b/5] FlashInfer extension ==="
if ! python -c "import torch, keys_values._flashinfer_ops" 2>/dev/null; then
    pip install -q flashinfer-python
    CC_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    unset LD_LIBRARY_PATH
    export PATH=/usr/local/cuda/bin:$PATH
    # build_ext.py locates csrc/ relative to itself, so patch in place and restore.
    sed -i "s/-gencode=arch=compute_90,code=sm_90/-gencode=arch=compute_${CC_ARCH},code=sm_${CC_ARCH}/" build_ext.py
    python build_ext.py > "$HOME/flashinfer_build.log" 2>&1 \
        && python -c "import torch, keys_values._flashinfer_ops; print('flashinfer OK')" \
        || echo "flashinfer build FAILED, see ~/flashinfer_build.log"
    git checkout -q build_ext.py
    # The .so is gitignored, so the per-job 'git clean -fd' in worker_loop.sh keeps it.
fi

echo "=== [3/5] canonical HELMET splits from S3 ==="
mkdir -p "$HOME/.cache/huggingface/helmet/longtrain"
aws s3 sync s3://keys-values-helmet-canonical/longtrain/ \
    "$HOME/.cache/huggingface/helmet/longtrain/" --region us-east-2 --only-show-errors
echo "splits: $(ls $HOME/.cache/huggingface/helmet/longtrain/)"

echo "=== [4/5] model checkpoint (${MODEL}) ==="
export HF_HUB_DISABLE_XET=1
litgpt download "$MODEL" || true  # already-downloaded is fine

echo "=== [5/5] smoke test ==="
python - <<'EOF'
import torch
from keys_values.config import Config
from keys_values.model import GPT
assert torch.cuda.is_available(), "no CUDA device"
print("GPU:", torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
print("keys_values imports OK")
EOF

echo "PROVISION_COMPLETE"
