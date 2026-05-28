# Qwen3 LoRA to LitGPT Workflow

This document describes how to convert a Hugging Face PEFT LoRA adapter trained on Qwen3 4B into a LitGPT checkpoint.

The recommended workflow is:

1. Merge the LoRA adapter into the original Hugging Face base model
2. Save the merged Hugging Face full model
3. Convert the merged model to LitGPT format

This workflow is used because Hugging Face PEFT LoRA adapters and LitGPT LoRA adapters use different parameter names and module layouts. Direct adapter conversion is not a simple file format conversion. It may require model-specific key mapping and QKV layout handling. Merging first avoids these issues.

## Files

Recommended directory layout:

```text
project
  lora_adapter
    adapter_config.json
    adapter_model.safetensors
  merged
```

The adapter directory should contain `adapter_config.json` and either `adapter_model.safetensors` or `adapter_model.bin`.

## Step 1 Create Environment

Create a separate environment to avoid package conflicts.

```bash
python3 -m venv lora_to_litgpt
. lora_to_litgpt/bin/activate
```

Install dependencies.

```bash
pip install -U torch transformers accelerate peft safetensors litgpt requests
```

Use a CUDA-enabled PyTorch build if you want GPU support.

## Step 2 Merge LoRA Adapter

Run the merge script.

NOTE: Run this script in a common base directory. Otherwise, the Hugging Face
checkpoint is downloaded again every time.

```bash
python merge_qwen3_lora.py \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter-dir <data-dir>/lora_adapter \
  --output-dir <data-dir>/merged
```

The base model must be exactly the same model used during LoRA training.

## Step 3 Check Merged Model

The merged model directory should look similar to this:

```text
merged
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
```

Depending on the save settings and model size, the weight files may also be sharded, for example:

```text
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
```

## Step 4 Convert to LitGPT

Run:

```bash
litgpt convert_to_litgpt <data-dir>/merged --model_name Qwen3-4B
```

## Step 5 Test LitGPT Checkpoint

Run:

```bash
litgpt chat --checkpoint_dir <data-dir>/merged
```

Or:

```bash
litgpt generate --checkpoint_dir <data-dir>/merged --prompt "Hello"
```

## Step 6 Copy Extra Config Files

The checkpoints lack some config files which are written with checkpoints during
training. These files depend mostly on the model and the dataset, while their
`kvcache` and `sdpa` args can be overwritten in the evaluation script.

```bash
cp model_config.yaml <data-dir>/merged/.
cp generation_config.json <data-dir>/merged/.
cp <data>/hyperparameters.yaml <data-dir>/merged/.
```

Here, <data> denotes the dataset.

## Common Issues

If the merged model gives poor answers, the most likely reason is that the adapter was merged into the wrong base model. Use the exact base model used during LoRA training.

If LitGPT conversion fails, upgrade LitGPT and check that the merged Hugging Face model directory contains `config.json`, tokenizer files, and model weight files.

## Notes

This workflow creates a full merged model. It does not preserve the LoRA adapter as a separate LitGPT adapter.

This is the most reliable path for inference, evaluation, and deployment in LitGPT.
