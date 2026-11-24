#!/bin/bash
# Generate + queue the standard RL job matrix (task x seed) for the worker
# fleet. Encodes the known-good 7B recipe on 48GB GPUs:
#   - paged 8-bit AdamW (plain AdamW's states are ~30GB for 7B -> OOM)
#   - group 4 x 2 accumulated prompts (same 8 rollouts/update as group 8,
#     half the peak batch memory)
#   - expandable_segments (recovers ~5GB lost to allocator fragmentation)
#   - EM-anchored em_f1 reward + RLOO advantages (campaign-validated: real
#     gains, no answer-style drift, high signal rate)
#
# Usage:
#   ./scripts/rl/queue_jobs.sh <max_length> <task1,task2,...> <seed1,seed2,...> [prefix]
# Example (the 32k flagship):
#   ./scripts/rl/queue_jobs.sh 32k nq,hotpot_qa 0,1 flag32k
set -euo pipefail

MAXLEN="${1:?max_length, e.g. 32k}"
TASKS="${2:?comma-separated dataset keys}"
SEEDS="${3:?comma-separated seeds}"
PREFIX="${4:-run${MAXLEN}}"
BUCKET="s3://keys-values-rl-results"
REGION="us-east-2"
MODEL="${KV_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CACHE_LEN="${KV_CACHE_LEN:-8192}"
STEPS="${KV_STEPS:-200}"

TMP=$(mktemp -d)
for task in ${TASKS//,/ }; do
  for seed in ${SEEDS//,/ }; do
    NAME="${PREFIX}_${task}_s${seed}"
    cat > "$TMP/$NAME.sh" <<EOF
cd ~/keys_values
git pull -q
source ~/venv/bin/activate
pip install -q bitsandbytes==0.49.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python examples/rl/grpo_helmet.py --device cuda --model ${MODEL} \\
    --dataset-key ${task} --max-length ${MAXLEN} \\
    --kv-cache-name h2o-torch-quantized8 --cache-length ${CACHE_LEN} \\
    --group-size 4 --prompts-per-update 2 --reward em_f1 --adv-mode rloo \\
    --optimizer paged_adamw8bit \\
    --steps ${STEPS} --lr 5e-6 --eval-every 100 --n-eval 8 --seed ${seed} \\
    --chunk-size 1024 --disable-flashinfer --out-dir \$OUT
python examples/rl/grpo_helmet_crosseval.py --device cuda --model ${MODEL} \\
    --dataset-key ${task} --max-length ${MAXLEN} \\
    --checkpoints base rloo_h2o=\$OUT/final.pt \\
    --h2o-cache-length ${CACHE_LEN} --n-eval 50 --disable-flashinfer \\
    --out \$OUT/crosseval.json
EOF
    aws s3 cp "$TMP/$NAME.sh" "$BUCKET/queue/pending/$NAME.sh" \
        --region $REGION --only-show-errors
    echo "queued $NAME"
  done
done
rm -rf "$TMP"
echo "pending now: $(aws s3 ls $BUCKET/queue/pending/ --region $REGION | wc -l) jobs"
