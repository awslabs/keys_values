#!/bin/bash
# keys_values RL worker loop: claim jobs from an S3 queue, run, upload results.
#
# Queue layout in s3://keys-values-rl-results/:
#   queue/pending/<job>.sh     one shell snippet per job (the command to run)
#   queue/claimed/<job>.sh     moved here (with instance-id suffix) on claim
#   queue/failed/<job>.sh      parked here if the job exits non-zero
#   runs/<job>/                stdout log + any files the job leaves in $OUT
#
# Three protections, each added after it cost us GPU hours:
#
# 1. One loop per instance (flock). Two loops on one box run two trainings on
#    one GPU, which OOMs both at model-load time.
# 2. Never start on a busy GPU. Belt-and-braces for (1): also covers a loop
#    started before this guard existed, which cannot hold the lock.
# 3. Resolve double claims. Claiming is copy+delete and S3 has no atomic
#    move, so two workers can claim one job. That is not harmless: they race
#    on the result upload and duplicate hours of compute. After claiming we
#    look for competing markers and back off unless we own the smallest
#    instance id.
#
# Usage:  ./worker_loop.sh            # run until queue empty, then exit
#         KV_STOP_WHEN_DONE=1 ./worker_loop.sh   # ...then stop this instance
set -uo pipefail

# --- protection 1: single loop per instance ---------------------------------
# Multi-GPU boxes: run one loop per GPU with KV_GPU=<index>. Each loop pins
# CUDA_VISIBLE_DEVICES to its GPU, takes its own lock, and claims jobs under
# its own id (<instance>-gpu<N>), so eight loops share one queue safely.
# Unset KV_GPU = single-GPU box, behaviour unchanged.
if [ -n "${KV_GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$KV_GPU"
    LOCK="$HOME/.kv_worker_gpu${KV_GPU}.lock"
else
    LOCK="$HOME/.kv_worker.lock"
fi
if [ "${KV_WORKER_LOCKED:-0}" != "1" ]; then
    export KV_WORKER_LOCKED=1
    exec flock -n "$LOCK" "$0" "$@" || {
        echo "another worker loop already holds the lock on this instance"
        exit 0
    }
fi

BUCKET="s3://keys-values-rl-results"
REGION="us-east-2"
IID=$(TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 60") && \
      curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
        http://169.254.169.254/latest/meta-data/instance-id)
# Claim id: per GPU slot on multi-GPU boxes so markers never collide.
[ -n "${KV_GPU:-}" ] && IID="${IID}-gpu${KV_GPU}"
# nvidia-smi ignores CUDA_VISIBLE_DEVICES; scope every query to our GPU.
SMI_ID=${KV_GPU:+--id=$KV_GPU}
cd "$HOME/keys_values"
source "$HOME/venv/bin/activate"

while true; do
    JOB=$(aws s3 ls "$BUCKET/queue/pending/" --region $REGION 2>/dev/null \
          | awk '{print $4}' | grep '\.sh$' | head -1)
    if [ -z "$JOB" ]; then
        echo "queue empty"
        break
    fi
    NAME="${JOB%.sh}"
    echo "claiming $NAME"
    # claim: copy to claimed/ then delete from pending/
    aws s3 cp "$BUCKET/queue/pending/$JOB" "$BUCKET/queue/claimed/${NAME}.${IID}.sh" \
        --region $REGION --only-show-errors || continue
    aws s3 rm "$BUCKET/queue/pending/$JOB" --region $REGION --only-show-errors

    # --- protection 3: resolve double claims -------------------------------
    # Wait out the window in which a competitor's marker may not be visible
    # yet, then let the smallest instance id win.
    sleep $(( (RANDOM % 5) + 8 ))
    OWNERS=$(aws s3 ls "$BUCKET/queue/claimed/" --region $REGION 2>/dev/null \
             | awk '{print $4}' | grep "^${NAME}\." \
             | sed "s/^${NAME}\.//; s/\.sh$//" | sort)
    WINNER=$(echo "$OWNERS" | head -1)
    if [ -n "$WINNER" ] && [ "$WINNER" != "$IID" ]; then
        echo "double claim on $NAME: $WINNER owns it, dropping our marker"
        aws s3 rm "$BUCKET/queue/claimed/${NAME}.${IID}.sh" \
            --region $REGION --only-show-errors
        continue
    fi

    aws s3 cp "$BUCKET/queue/claimed/${NAME}.${IID}.sh" "/tmp/${JOB}" \
        --region $REGION --only-show-errors

    # --- protection 2: never start on a busy GPU ---------------------------
    GPU_USED=$(nvidia-smi $SMI_ID --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "${GPU_USED:-}" ] && [ "$GPU_USED" -gt 2000 ]; then
        echo "GPU busy (${GPU_USED} MiB used): returning $NAME to pending and exiting"
        aws s3 cp "$BUCKET/queue/claimed/${NAME}.${IID}.sh" \
            "$BUCKET/queue/pending/${JOB}" --region $REGION --only-show-errors
        aws s3 rm "$BUCKET/queue/claimed/${NAME}.${IID}.sh" \
            --region $REGION --only-show-errors
        break
    fi

    # Force a clean checkout before every job. A diagnostic job that reverted
    # a file with 'git checkout <commit> -- <file>' left it STAGED as the old
    # version; 'git pull' in later jobs preserved it, and a day of runs on
    # this box silently used pre-fix code. Never trust the tree between jobs.
    (cd "$HOME/keys_values" && git reset -q --hard HEAD && git clean -qfd -e repos/ -e runs/)
    # The FlashInfer .so is gitignored and survives the clean above, so a
    # job that pulls a changed kernel source would otherwise run the OLD
    # binary. Rebuild when any csrc file is newer than the .so (or no .so).
    # Serialize across GPU loops on one box: eight concurrent rebuilds of the
    # same .so would corrupt it. Re-check staleness after taking the lock.
    (cd "$HOME/keys_values" && exec 9>"$HOME/.kv_build.lock" && flock 9 && {
        SO=$(ls keys_values/_flashinfer_ops*.so 2>/dev/null | head -1)
        if [ -z "$SO" ] || [ -n "$(find keys_values/csrc -newer "$SO" -type f | head -1)" ]; then
            echo "FlashInfer kernels missing or stale: rebuilding"
            CC_ARCH=$(nvidia-smi $SMI_ID --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
            ( unset LD_LIBRARY_PATH; export PATH=/usr/local/cuda/bin:$PATH
              pip install -q flashinfer-python
              sed -i "s/-gencode=arch=compute_90,code=sm_90/-gencode=arch=compute_${CC_ARCH},code=sm_${CC_ARCH}/" build_ext.py
              rm -f keys_values/_flashinfer_ops*.so; rm -rf build
              python build_ext.py > "$HOME/flashinfer_build.log" 2>&1 \
                  && python -c "import torch, keys_values._flashinfer_ops" && echo "FlashInfer OK" \
                  || echo "FlashInfer build FAILED (see ~/flashinfer_build.log); jobs will run eager" )
            git checkout -q build_ext.py
        fi
    })

    export OUT="$HOME/runs/$NAME"
    mkdir -p "$OUT"
    echo "=== running $NAME on $IID ($(date -u +%FT%TZ)) ==="
    # GPU utilization sampling: one CSV per job, synced to S3 with the rest
    # of $OUT. 10s cadence keeps a multi-day job under ~3 MB. The summary
    # line in job.log makes utilization regressions (issue #152 class)
    # visible at a glance instead of requiring live nvidia-smi eyeballing.
    nvidia-smi $SMI_ID --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
        --format=csv,noheader -l 10 > "$OUT/gpu_util.csv" 2>/dev/null &
    SMI_PID=$!
    bash "/tmp/${JOB}" > "$OUT/job.log" 2>&1
    STATUS=$?
    kill "$SMI_PID" 2>/dev/null
    wait "$SMI_PID" 2>/dev/null
    if [ -s "$OUT/gpu_util.csv" ]; then
        awk -F', ' '{gsub(/ %/,"",$2); s+=$2; if ($2+0>m) m=$2; n++}
            END {if (n) printf "gpu_util: mean=%.0f%% max=%d%% samples=%d\n", s/n, m, n}' \
            "$OUT/gpu_util.csv" >> "$OUT/job.log"
    fi
    echo "exit=$STATUS" >> "$OUT/job.log"

    aws s3 sync "$OUT" "$BUCKET/runs/$NAME/" --region $REGION --only-show-errors
    # Local checkpoints are 15 GB each and are now in S3: drop them, or the
    # disk fills after ~10 jobs and every later job dies on import.
    if [ $? -eq 0 ]; then
        rm -f "$OUT"/*.pt
    fi
    if [ $STATUS -ne 0 ]; then
        # Park failed jobs visibly instead of silently draining the queue;
        # requeue after diagnosis with: aws s3 mv .../failed/X.sh .../pending/X.sh
        aws s3 mv "$BUCKET/queue/claimed/${NAME}.${IID}.sh" \
            "$BUCKET/queue/failed/${JOB}" --region $REGION --only-show-errors
        echo "$NAME FAILED (exit=$STATUS), parked in queue/failed/, log at $BUCKET/runs/$NAME/job.log"
    else
        echo "$NAME done, results at $BUCKET/runs/$NAME/"
    fi
done

if [ "${KV_STOP_WHEN_DONE:-0}" = "1" ]; then
    echo "stopping instance $IID"
    aws ec2 stop-instances --instance-ids "$IID" \
        --region "$(curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT \
          http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" \
          http://169.254.169.254/latest/meta-data/placement/region)"
fi
