# Original Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# Modification Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This script merges the LoRA weights with the base model"""
from pathlib import Path
from pprint import pprint
from typing import Optional

import lightning as L
import torch

from litgpt.lora import lora_filter, merge_lora_weights, Config
from litgpt.utils import check_valid_checkpoint_dir, extend_checkpoint_dir
from litgpt.scripts.merge_lora import load_lora_metadata

from keys_values.lora import GPT


def merge_lora(
    checkpoint_dir: Path,
    pretrained_checkpoint_dir: Optional[Path] = None,
    precision: Optional[str] = None,
    lora_fname: str = "lit_model.pth.lora",
    pretrained_fname: str = "lit_model.pth",
) -> None:
    """Merges the LoRA weights with the base model.

    See ``litgpt finetune lora``.

    Creates a new ``lit_model.pth`` file by merging the LoRA weights (``lit_model.pth.lora``)
    with the original checkpoint weights.

    Arguments:
        checkpoint_dir: Path to the checkpoint directory with trained LoRA weights, which is the output of
            ``litgpt finetune lora``.
        pretrained_checkpoint_dir: Optional path to the checkpoint directory with the weights of the base model
            corresponding to the LoRA checkpoint. By default, this will automatically be inferred from the metadata
            in the given `checkpoint_dir` directory. Only set this if the base model's checkpoint directory
            has moved or was renamed.
        precision: Optional precision setting to instantiate the model weights in. By default, this will
            automatically be inferred from the metadata in the given ``checkpoint_dir`` directory.
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    if pretrained_checkpoint_dir is not None:
        pretrained_checkpoint_dir = extend_checkpoint_dir(pretrained_checkpoint_dir)
    pprint(locals())

    check_valid_checkpoint_dir(checkpoint_dir, model_filename=lora_fname)
    if pretrained_checkpoint_dir is not None:
        check_valid_checkpoint_dir(pretrained_checkpoint_dir)
    if (checkpoint_dir / pretrained_fname).is_file():
        print("LoRA weights have already been merged in this checkpoint.")
        return

    lora_params, meta_pretrained_checkpoint_dir, lora_precision = load_lora_metadata(
        checkpoint_dir
    )
    precision = precision if precision is not None else lora_precision

    if pretrained_checkpoint_dir is None:
        pretrained_checkpoint_dir = meta_pretrained_checkpoint_dir
        pretrained_checkpoint_dir = extend_checkpoint_dir(pretrained_checkpoint_dir)

    fabric = L.Fabric(devices=1, precision=precision, accelerator="cpu")
    config = Config.from_file(checkpoint_dir / "model_config.yaml", **lora_params)

    with fabric.init_module(), torch.device("meta"):
        model = GPT(config)
        # we don't care about these to perform merging
        model.cos = None
        model.sin = None

    lora_path = checkpoint_dir / lora_fname
    pretrained_checkpoint = torch.load(
        str(pretrained_checkpoint_dir / pretrained_fname), mmap=True
    )
    lora_checkpoint = torch.load(str(lora_path), mmap=True)
    lora_checkpoint = lora_checkpoint.get("model", lora_checkpoint)

    # Merge LoRA weights into the base model
    pretrained_checkpoint.update(lora_checkpoint)
    model.load_state_dict(pretrained_checkpoint, assign=True)
    # since LoRA finetuning only saves the LoRA weights, we treat the lora weights dtype as the expected dtype
    lora_dtype = next(iter(lora_checkpoint.values())).dtype
    model.to(dtype=lora_dtype, device="cpu")
    merge_lora_weights(model)

    # Remove LoRA parameters and the LoRA linear substring
    state_dict = {
        k.replace("linear.", ""): v
        for k, v in model.state_dict().items()
        if not lora_filter(k, v)
    }
    save_path = checkpoint_dir / pretrained_fname
    torch.save(state_dict, save_path)

    fabric.print(f"Saved merged weights to {str(save_path)!r}")
