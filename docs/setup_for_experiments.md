# Setup for Learning How To Forget Experiments

Here, we detail the setup we used for running experiments for the report
[Learning how to Forget: Fine-tuning for Long-Context Sparse Attention](https://arxiv.org/abs/2608.19920).

Our experiments were run on a number of AWS EC2 instances:

* Type: `p4d.24xlarge`
* AMI: `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)`

We used CUDA 12.8.

## Install Code

This is done more or less as detailed in [README.md](../README.md). For the
following, we assume that CUDA 12.8 is used.

```bash
mkdir git
mkdir virtenvs
cd git
git clone https://github.com/awslabs/keys_values.git
cd ../virtenvs
python3 -m venv keys_values
. keys_values/bin/activate
pip install --upgrade pip
pip install torch==2.10.0 torchmetrics==1.8.2 torchvision==0.25.0
echo "torch==2.10.0" >constraints.txt
pip install flashinfer-python==0.6.7 -c constraints.txt
rm constraints.txt
pip install 'litgpt[all,test,extra]'
cd ../git/keys_values
pip install -e .
python build_ext.py
```

At this point, you are ready to run training or evaluation runs. One
complication is that for the `Helmet` and `RULER` datasets we mainly use,
concrete datasets are generated and split at random with the first call and
loaded afterwards. When using different instances (or EFS volumes), we made
sure to copy the respective files written by these data generators. This is
described next.

## Setup Datasets

When experiments are run on different instances, the concrete dataset files
need to be copied across (unless you keep them on a single EFS volume
mounted from all instances). Here, we detail which files need to be copied,
and which symlinks need to be set.

### Helmet

There are (1) dataset files, separate for each task and sequence length, and
(2) a single metadata file.

The metadata file is `helmet_metadata.json`. It needs to be in the directory
specified as `--data.metadata_dir`. For us, this was always
`home/ubuntu/out/finetune/data`.

The dataset files are in subdirectories, for example:
```text
banking77_128k  hotpot_qa_32k           infinite_bench_qa_128k  nlu_128k  nq_64k       ruler_mk_uuid_128k  trivia_qa_128k
clinc150_128k   hotpot_qa_64k           json_kv_128k            nq_128k   pop_qa_128k  trec_coarse_128k    trivia_qa_64k
hotpot_qa_128k  infinite_bench_mc_128k  ms_macro_128k           nq_32k    pop_qa_64k   trec_fine_128k
```

They look like (for example):
```text
clinc150_128k/
   dataset_dict.json
   development/
      data-00000-of-00002.arrow
      data-00001-of-00002.arrow
      dataset_info.json
      state.json
   evaluation/
      data-00000-of-00001.arrow
      dataset_info.json
      state.json
```

Copy these files where you have enough space, for example an EFS volume or
`/opt/dlami/nvme/`. Then, create a symlink so they can be read from
`~/.cache/huggingface/helmet/longtrain`, so that:
```bash
$ ls ~/.cache/huggingface/helmet/longtrain/

banking77_128k  hotpot_qa_32k           infinite_bench_qa_128k  nlu_128k  nq_64k       ruler_mk_uuid_128k  trivia_qa_128k
clinc150_128k   hotpot_qa_64k           json_kv_128k            nq_128k   pop_qa_128k  trec_coarse_128k    trivia_qa_64k
hotpot_qa_128k  infinite_bench_mc_128k  ms_macro_128k           nq_32k    pop_qa_64k   trec_fine_128k
```

If you start a training or evaluation run, you must not see any attempts at
downloading, generating, or tokenizing data, or at splitting it. Look for
outputs such as:
```bash
[...]
Loaded cached datasets (development, evaluation) from /home/ubuntu/.cache/huggingface/helmet/longtrain/clinc150_128k

Transforming HELMET 'clinc150' (128k) ...
Kept 2600 of 2600 dev records
Kept 500 of 500 eval records
Split development set into training (2340) and validation (260)
Sampled train_target_choice (2344)
Sampled val_target_choice (264)
Sampled test_target_choice (504)
[...]
```

### RULER

There are (1) dataset files, separate for each task and sequence length, and
(2) a single metadata file.

The metadata file is `ruler_metadata.json`. It needs to be in the directory
specified as `--data.metadata_dir`. For us, this was always
`home/ubuntu/out/finetune/data`.

The dataset files are in subdirectories, for example:
```text
cwe_256k_500  fwe_256k_500  niah_single_1_256k_500  vt_256k_500
```

They look like (for example):
```text
niah_single_1_256k_500/
   dataset_dict.json
   development/
      data-00000-of-00001.arrow
      dataset_info.json
      state.json
   evaluation/
      data-00000-of-00001.arrow
      dataset_info.json
      state.json
```

Copy these files where you have enough space, for example an EFS volume or
`/opt/dlami/nvme/`. Then, create a symlink so they can be read from
`~/.cache/huggingface/ruler/longtrain`, so that:
```bash
$ ls ~/.cache/huggingface/ruler/longtrain/

cwe_256k_500  fwe_256k_500  niah_single_1_256k_500  vt_256k_500
```

If you start a training or evaluation run, you must not see any attempts at
downloading, generating, or tokenizing data, or at splitting it. Look for
outputs such as:
```bash
[...]
Loaded cached datasets (development, evaluation) from /home/ubuntu/.cache/huggingface/ruler/longtrain/niah_single_1_256k_500

Transforming RULER 'niah_single_1' (256k) ...
Kept 400 of 400 dev records
Kept 100 of 100 eval records
Split development set into training (360) and validation (40)
Sampled train_target_choice (360)
Sampled val_target_choice (40)
Sampled test_target_choice (104)
[...]
```

## Training Runs

In general, we ran a training job on 4 devices, either 0,1,2,3 or 4,5,6,7, so
that two jobs can be run in parallel on the same instance.

NOTE: For RULER training jobs, this can lead to out of CPU memory crashes. This
is not resolved yet. The working solution is to not run two of them in parallel
on the same instance.

### Helmet

Run this with the venv `keys_values` active and in `~/.git/keys_values`. We
used a screen.

Command:
```bash
export DATASET_KEY="clinc150"; \
export DATASET_SIZE="128k"; \
export EVAL_STEPS=10; \
export MAX_EPOCHS=5; \
export KV_LOGIC="smart-lastrec"; \
export KVL_SHORT="slr"; \
export KV_NORM="False"; \
export CHUNK_SIZE=8192; \
export CHUNK_MULTIPLIER=1; \
export BASE_DIR="/home/ubuntu/out/finetune/neurips_exp/lora/qwen3_4b"; \
CUDA_VISIBLE_DEVICES="0,1,2,3" \
PYTORCH_ALLOC_CONF=expandable_segments:True \
KEYSVALS_LOG_DIR="${BASE_DIR}/helmet_${DATASET_KEY}_${DATASET_SIZE}/${KVL_SHORT}_4gpu_cs${CHUNK_SIZE}_lr5/logs" \
python3 keys_values/__main__.py finetune_long_lora \
    Qwen/Qwen3-4B-Instruct-2507 \
    --out_dir ${BASE_DIR}/helmet_${DATASET_KEY}_${DATASET_SIZE}/${KVL_SHORT}_4gpu_cs${CHUNK_SIZE}_lr5 \
    --precision bf16-true \
    --verbose some \
    --devices 4 \
    --data Helmet \
        --data.dataset_key ${DATASET_KEY} \
        --data.max_length ${DATASET_SIZE} \
        --data.metadata_dir /home/ubuntu/out/finetune/data \
        --data.trainloader_longest_first True \
    --train.save_interval ${EVAL_STEPS} \
        --train.micro_batch_size 2 \
        --train.epochs ${MAX_EPOCHS} \
        --train.average_loss_per_batch True \
    --eval.interval ${EVAL_STEPS} \
        --eval.initial_validation True \
        --eval.use_sample_metric False \
    --kv_cache.cache_length 32768 \
        --kv_cache.chunk_size ${CHUNK_SIZE} \
        --kv_cache.name ${KV_LOGIC}-torch-quantized8 \
        --kv_cache.normalize_scores ${KV_NORM} \
    --grad.layers_per_cell 1 \
        --grad.layercp_qname default \
        --grad.cachecp_qname torch-quantized8 \
        --grad.chunks_per_cell_multiplier ${CHUNK_MULTIPLIER} \
    --optimizer.name AdamW \
        --optimizer.learning_rate 0.0005
```

The KV cache policies as named in the paper are obtained by setting tuples
`(KVL_SHORT, KV_LOGIC, KV_NORM)` to:

* `lr`: `(lr, lastrec, False)`
* `slr`: `(slr, smart-lastrec, False)`
* `h2o`: `(h2o, h2o, False)`
* `h2onorm`: `(h2onorm, h2o, True)`
* `h2oorig`: `(h2oorig, h2o-orig, False)`

Further notes:

* `DATASET_KEY` selects the task from Helmet
* Set `CUDA_VISIBLE_DEVICES` to "0,1,2,3" or "4,5,6,7", whatever is free
* `CHUNK_SIZE` is the chunk size (called `S` in the paper)
* `CHUNK_MULTIPLIER` is called `alpha` in the paper
* `EVALS_STEPS` states after how many update steps a validation evaluation is
  run. We used 10, or 5 for small datasets

### RULER

Run this with the venv `keys_values` active and in `~/.git/keys_values`. We
used a screen.

Command:
```bash
export DATASET_KEY="fwe"; \
export DATASET_SIZE="256k"; \
export EVAL_STEPS=10; \
export MAX_EPOCHS=5; \
export KV_LOGIC="smart-lastrec"; \
export KVL_SHORT="slr"; \
export KV_NORM="False"; \
export CHUNK_SIZE=8192; \
export CHUNK_MULTIPLIER=1; \
export BASE_DIR="/home/ubuntu/out/finetune/neurips_exp/lora/qwen3_4b"; \
CUDA_VISIBLE_DEVICES="0,1,2,3" \
PYTORCH_ALLOC_CONF=expandable_segments:True \
KEYSVALS_LOG_DIR="${BASE_DIR}/ruler_${DATASET_KEY}_${DATASET_SIZE}/${KVL_SHORT}_4gpu_cs${CHUNK_SIZE}_lr5/logs" \
python3 keys_values/__main__.py finetune_long_lora \
    Qwen/Qwen3-4B-Instruct-2507 \
    --out_dir ${BASE_DIR}/ruler_${DATASET_KEY}_${DATASET_SIZE}/${KVL_SHORT}_4gpu_cs${CHUNK_SIZE}_lr5 \
    --precision bf16-true \
    --verbose some \
    --devices 4 \
    --data Ruler \
        --data.dataset_key ${DATASET_KEY} \
        --data.max_length ${DATASET_SIZE} \
        --data.num_samples 500 \
        --data.metadata_dir /home/ubuntu/out/finetune/data \
        --data.trainloader_longest_first True \
    --train.save_interval ${EVAL_STEPS} \
        --train.micro_batch_size 2 \
        --train.epochs ${MAX_EPOCHS} \
        --train.average_loss_per_batch True \
    --eval.interval ${EVAL_STEPS} \
        --eval.initial_validation True \
        --eval.use_sample_metric False \
    --kv_cache.cache_length 32768 \
        --kv_cache.chunk_size ${CHUNK_SIZE} \
        --kv_cache.name ${KV_LOGIC}-torch-quantized8 \
        --kv_cache.normalize_scores ${KV_NORM} \
    --grad.layers_per_cell 1 \
        --grad.layercp_qname default \
        --grad.cachecp_qname torch-quantized8 \
        --grad.chunks_per_cell_multiplier ${CHUNK_MULTIPLIER} \
        --grad.checkpoint_temp_dir /opt/dlami/nvme/swapspace \
    --optimizer.name AdamW \
        --optimizer.learning_rate 0.0005
```

Extra notes:

* `--grad.checkpoint_temp_dir` ensures that checkpoints are written to disk
  when CPU memory becomes full. Pick a fast disk. Note that this feature
  may not work properly right now, because we still see instances crashing
  with CPU out of memory
