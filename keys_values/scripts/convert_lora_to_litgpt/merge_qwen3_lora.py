"""
Merge a PEFT LoRA adapter into a Hugging Face base model.

Example:
    python merge_lora.py \
        --base-model Qwen/Qwen3-4B \
        --adapter-dir ./your_lora_adapter \
        --output-dir ./merged_model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT LoRA adapter into a Hugging Face base model."
    )

    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_model = args.base_model
    adapter_dir = Path(args.adapter_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    dtype = torch.bfloat16

    print(f"Loading tokenizer from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        use_fast=True,
    )

    print(f"Loading base model from {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
    )

    print(f"Loading LoRA adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(
        model,
        str(adapter_dir),
        torch_dtype=dtype,
    )

    print("Merging LoRA adapter")
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(output_dir)

    print("Done")


if __name__ == "__main__":
    main()