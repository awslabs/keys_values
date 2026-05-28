#!/bin/bash

data_size="64k"
base_dir="./baseline_$data_size"

datasets=("hotpot_qa_$data_size" "nq_$data_size" "pop_qa_$data_size" "trivia_qa_$data_size")

for dataset in "${datasets[@]}"; do
    echo "\n[Processing $dataset ...]"
    mypath=$base_dir/$dataset
    mkdir -p $mypath/lora_adapter
    mv $mypath/* $mypath/lora_adapter/.
    python merge_qwen3_lora.py --base-model Qwen/Qwen3-4B-Instruct-2507 --adapter-dir $mypath/lora_adapter --output-dir $mypath/merged
    litgpt convert_to_litgpt $mypath/merged --model_name Qwen3-4B
    # Copy default files
    cp $dataset/hyperparameters.yaml $mypath/merged/.
    cp model_config.yaml $mypath/merged/.
    cp generation_config.json $mypath/merged/.
done
