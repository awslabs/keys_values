# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Literal, Dict, Any

import lightning as L
import torch

from litgpt.args import TrainArgs
from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer
from litgpt.utils import choose_logger as _choose_logger, instantiate_torch_optimizer

from keys_values.data.base import (
    LIT_MODEL_FNAME,
    HEAD_MODEL_FNAME,
    LORA_WEIGHTS_FNAME,
)
from keys_values.data.dataloader import MyDataLoader
from keys_values.finetune.args import EvalArgs, KVCacheArgs, OptimizerArgs, GradientArgs
from keys_values.head_model import HeadModel
from keys_values.kvcache.gradient.annotation import NodeAnnotation
from keys_values.kvcache.gradient.autograd_hooks import MayMatchTwiceType
from keys_values.long_context import GPTAndHeadModel
from keys_values.model import GPT
from keys_values.utils import flush_io_streams


def debug_print_param_names(model: GPT):
    rows = ["", "Names of model (GPT)", ""]
    rows.extend([name for name, _ in model.named_parameters()])
    for i, block in enumerate(model.transformer.h):
        rows.extend(["", f"Names of block {i} (Block)", ""])
        rows.extend([name for name, _ in block.named_parameters()])
    for pname, block in [
        ("lm_head (Linear)", model.lm_head),
        ("wte (Embedding)", model.transformer.wte),
    ]:
        rows.extend(["", f"Names of {pname}", ""])
        rows.extend([name for name, _ in block.named_parameters()])
    print("\n".join(rows))


MAX_PRINT_HEAD = 256

MAX_PRINT_TAIL = 128


def print_but_limit_size(
    fabric: L.Fabric,
    text: str,
):
    text_length = len(text)
    if text_length <= MAX_PRINT_HEAD + MAX_PRINT_TAIL:
        print_message("\n" + text, fabric)
    else:
        print_message(
            "\n" + text[:MAX_PRINT_HEAD] + "\n\n[...]\n\n" + text[(-MAX_PRINT_TAIL):],
            fabric,
        )


def get_lr_scheduler(
    optimizer,
    train_args: TrainArgs,
    max_steps: int,
):
    if train_args.lr_warmup_fraction is None:
        if train_args.lr_warmup_steps is None:
            raise ValueError(
                "Either train.lr_warmup_fraction or train_args.lr_warmup_steps must be given"
            )
        warmup_steps = min(train_args.lr_warmup_steps, max_steps)
    else:
        if not (0 <= train_args.lr_warmup_fraction <= 1):
            raise ValueError(
                f"train_args.lr_warmup_fraction = {train_args.lr_warmup_fraction}, must be in [0, 1]"
            )
        if train_args.lr_warmup_steps is not None:
            print(
                f"train.lr_warmup_fraction = {train_args.lr_warmup_fraction}, train_args.lr_warmup_steps = {train_args.lr_warmup_steps}. Using the former."
            )
        warmup_steps = train_args.lr_warmup_fraction * max_steps
    # Linear warmup followed by cosine annealing
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(max_steps - warmup_steps)
    )
    if warmup_steps <= 0:
        return scheduler2
    # Note: The first LR (for `step=0`) is being used. Must not be 0
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / warmup_steps
    )
    if warmup_steps >= max_steps:
        return scheduler1
    else:
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [scheduler1, scheduler2],
            milestones=[warmup_steps],
        )


def get_dataloaders(
    data: DataModule,
    tokenizer: Tokenizer,
    head_model: str,
    train: TrainArgs,
    eval: EvalArgs,
    fabric: Optional[L.Fabric] = None,
) -> Tuple[MyDataLoader, MyDataLoader]:
    num_devices = 1 if fabric is None else fabric.world_size
    rank = 0 if fabric is None else fabric.local_rank
    data.connect(
        tokenizer=tokenizer,
        batch_size=train.micro_batch_size,
        num_devices=num_devices,
        rank=rank,
        max_seq_length=train.max_seq_length,
        head_model=head_model,
        val_batch_size=eval.micro_batch_size,
    )
    if fabric is not None:
        with fabric.rank_zero_first():
            data.prepare_data()
    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    return train_dataloader, val_dataloader


def validate_args(train: TrainArgs, eval: EvalArgs) -> None:
    issues = []
    unsupported = [(train, ["max_tokens", "max_norm", "tie_embeddings"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(
                    f"{__file__} doesn't support the {name!r} argument. This is set in {args}"
                )
    required = [(train, ["epochs"]), (eval, ["max_new_tokens"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(
                    f"{__file__} requires the {name!r} argument. This is set in {args}"
                )
    if not train.epochs and not train.max_steps:
        issues.append(
            f"{__file__} requires either epochs or max_steps to be set. This is set in {train}"
        )
    if issues:
        raise ValueError("\n".join(issues))


def is_lora_model(model: GPTAndHeadModel) -> bool:
    from keys_values.lora import GPT as GPTLoRA

    return isinstance(model, GPTLoRA)


def save_model_checkpoint(
    fabric: L.Fabric,
    model: GPTAndHeadModel,
    file_dir: Path,
) -> None:
    from litgpt.lora import lora_filter

    if is_lora_model(model):
        file_path = file_dir / LORA_WEIGHTS_FNAME
        save_kwargs = dict(filter={"model": lora_filter})
    else:
        file_path = file_dir / LIT_MODEL_FNAME
        save_kwargs = dict()
    file_dir.mkdir(parents=True, exist_ok=True)
    print_message(
        f"\nSaving model weights to {str(file_path)!r}",
        fabric,
    )
    fabric.save(file_path, state={"model": model.gpt_model}, **save_kwargs)
    if model.head_model.state_dict():
        file_path = file_dir / HEAD_MODEL_FNAME
        print_message(
            f"Saving head model weights to {str(file_path)!r}",
            fabric,
        )
        fabric.save(file_path, state={"model": model.head_model})


def choose_logger(
    logger_name: Literal["csv", "tensorboard", "wandb", "mlflow"],
    out_dir: Path,
    name: str,
    use_fabric: bool = True,
    log_interval: int = 1,
    log_args: Optional[Dict] = None,
    resume: Optional[bool] = None,
    **kwargs: Any,
):
    if use_fabric:
        return _choose_logger(logger_name, out_dir, name, log_interval, **kwargs)
    else:
        if logger_name == "csv":
            from lightning.pytorch.loggers.csv_logs import CSVLogger

            return CSVLogger(
                out_dir,
                name=name,
                flush_logs_every_n_steps=log_interval,
                **kwargs,
            )
        if logger_name == "tensorboard":
            from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

            return TensorBoardLogger(
                out_dir,
                name=name,
                **kwargs,
            )
        if logger_name == "wandb":
            from lightning.pytorch.loggers.wandb import WandbLogger

            if log_args is None:
                log_args = dict()
            project = log_args.get("project", name)
            run = log_args.get("run", os.environ.get("WANDB_RUN_NAME"))
            group = log_args.get("group", os.environ.get("WANDB_RUN_GROUP"))
            return WandbLogger(
                project=project,
                name=run,
                group=group,
                resume=resume,
                **kwargs,
            )
        if logger_name == "mlflow":
            from lightning.pytorch.loggers.mlflow import MLFlowLogger

            if log_args is None:
                log_args = dict()
            experiment_name = log_args.get("experiment_name", name)
            tracking_uri = log_args.get("tracking_uri")
            return MLFlowLogger(
                experiment_name=experiment_name,
                tracking_uri=tracking_uri,
                save_dir=str(out_dir),
                **kwargs,
            )
        raise ValueError(
            f"`logger_name={logger_name}` is not a valid option. Choose from 'csv', 'tensorboard', 'wandb', 'mlflow'."
        )


def adapt_requires_grad(
    gpt_model: GPT,
    head_model: HeadModel,
):
    """
    If `head_model.needs_logits() == False`, we mark weights related to
    `gpt_model.lm_head` with `requires_grad=False`. Also works if `gpt_model`
    is a LoRA model.

    Args:
        gpt_model (GPT): GPT model
        head_model (HeadModel): Head model

    """
    from keys_values.optimize.model_factory import BlockComponentName

    if not head_model.needs_logits():
        prefix = BlockComponentName.lm_head()
        for name, param in gpt_model.named_parameters():
            if name.startswith(prefix):
                param.requires_grad_(False)


def print_with_rank_and_timestamp(
    msg: str,
    rank: int,
    start_newline: bool = True,
    flush_streams: bool = True,
):
    time_format = "%Y-%m-%d %H:%M:%S"
    time_stamp = datetime.now().strftime(time_format)
    prefix = ("\n" if start_newline else "") + f"[rank {rank} | {time_stamp}]: "
    print(prefix + msg)
    if flush_streams:
        flush_io_streams()


def print_message(msg: str, fabric: Optional[L.Fabric] = None):
    if fabric is not None:
        fabric.print(msg)
    else:
        print(msg)


def check_kv_cache(kv_cache: KVCacheArgs):
    if kv_cache.name.startswith("dense"):
        raise ValueError(
            "kv_cache must be given for long-context inference, and "
            "kv_cache.name must not be dense-*"
        )


def create_optimizer(
    optim_args: OptimizerArgs,
    gpt_model: GPT,
    gpt_param_prefixes: Tuple[str, ...],
):
    parameters = [
        param
        for name, param in gpt_model.named_parameters()
        if name.startswith(gpt_param_prefixes)
    ]
    return instantiate_torch_optimizer(
        optim_args.name,
        parameters,
        **optim_args.optimizer_kwargs(),
    )


def may_match_twice_flex_attention_sdpa(annotation: NodeAnnotation) -> bool:
    """
    With `flex_attention` and the new training replay cache, the "ext-*"
    annotations match twice. The same holds for the new training replay cache
    with zero-padded query SDPA.

    """
    return annotation.is_ext


def may_match_twice_fused_eager_sdpa(annotation: NodeAnnotation) -> bool:
    """
    With the old training replay cache using our special fused eager SDPA, the
    "scatter-*" annotations for chunk index 1 match twice.

    """
    return annotation.is_scatter and annotation.chunk_idx == 1


def may_match_twice_factory(
    grad: GradientArgs,
    gpt_model: GPT,
) -> MayMatchTwiceType:
    """
    Helper for :func:`wrap_gpt_model`. Selects the best `may_match_twice`
    predicate for :class:`CellComputationAutogradHooks`.

    Args:
        grad: Arguments passed to :func:`wrap_gpt_model`.
        gpt_model: GPT model passed to :func:`wrap_gpt_model`.

    Returns:
        `may_match_twice` predicate

    """
    if grad.use_old_cache:
        return may_match_twice_fused_eager_sdpa
    else:
        return may_match_twice_flex_attention_sdpa
