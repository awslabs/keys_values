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
from contextlib import AbstractContextManager, ExitStack
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Optional, Tuple, Literal, Dict, Any, Union, Callable, Mapping, List
import yaml

from lightning.fabric.connector import _convert_precision_to_unified_args
from lightning.fabric.loggers import Logger
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.fabric.plugins.precision import (
    Precision,
    HalfPrecision,
    DoublePrecision,
    TransformerEnginePrecision,
    MixedPrecision,
)
from lightning.fabric.utilities.apply_func import convert_tensors_to_scalars
from lightning.fabric.utilities.init import _EmptyInit
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from lightning.fabric.wrappers import _unwrap_objects
from tokenizers import Tokenizer as HFTokenizer
import torch

from keys_values.kvcache.smart_lastrec import SmartInitialInformation
from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer
from litgpt.utils import instantiate_torch_optimizer

from keys_values.data.constants import (
    LIT_MODEL_FNAME,
    HEAD_MODEL_FNAME,
    LORA_WEIGHTS_FNAME,
)
from keys_values.data.dataloader import MyDataLoader
from keys_values.data.trainstate import DataTrainState
from keys_values.distributed.fabric import Fabric
from keys_values.finetune.args import (
    TrainArgs,
    EvalArgs,
    KVCacheArgs,
    OptimizerArgs,
    GradientArgs,
)
from keys_values.head_model import HeadModel
from keys_values.kvcache.gradient.annotation import NodeAnnotation
from keys_values.kvcache.gradient.autograd_hooks import MayMatchTwiceType
from keys_values.long_context import GPTAndHeadModel
from keys_values.model import GPT
from keys_values.parser_config import (
    save_hyperparameters,
    HYPERPARAMETERS_FILENAME,
)
from keys_values.utils import flush_io_streams


def debug_print_param_names(model: GPT):
    rows = ["", "Names of model (GPT)", ""]
    rows.extend([name for name, _ in model.named_parameters()])
    for i, block in enumerate(model._get_layer_blocks()):
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


def print_but_limit_size(text: str):
    text_length = len(text)
    if text_length <= MAX_PRINT_HEAD + MAX_PRINT_TAIL:
        Fabric.print("\n" + text)
    else:
        Fabric.print(
            "\n" + text[:MAX_PRINT_HEAD] + "\n\n[...]\n\n" + text[(-MAX_PRINT_TAIL):],
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
    training_state: Optional[DataTrainState] = None,
) -> Tuple[MyDataLoader, MyDataLoader]:
    data.connect(
        tokenizer=tokenizer,
        batch_size=train.micro_batch_size,
        num_devices=Fabric.world_size(),
        rank=Fabric.rank(),
        max_seq_length=train.max_seq_length,
        head_model=head_model,
        val_batch_size=eval.micro_batch_size,
        training_state=training_state,
    )

    # Everybody needs to wait until `data.prepare_data()` finished on rank 0
    if Fabric.rank() == 0:
        data.prepare_data()
    Fabric.barrier()

    data.setup()
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    return train_dataloader, val_dataloader


def validate_args(train: TrainArgs, eval: EvalArgs) -> None:
    issues = []
    unsupported = [(train, ["max_tokens", "tie_embeddings"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(
                    f"{__file__} doesn't support the {name!r} argument. This is set in {args}"
                )
    required = [(train, ["epochs"])]
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

    return isinstance(model.gpt_model, GPTLoRA)


# From `lightning.fabric.strategies.strategy`
def _apply_filter(
    key: str,
    filter: dict[str, Callable[[str, Any], bool]],
    source_dict: object,
    target_dict: dict[str, Any],
) -> None:
    # filter out if necessary
    if key in filter and isinstance(source_dict, dict):
        filter_fn = filter[key]
        for k, v in source_dict.items():
            if filter_fn(k, v):
                # save the state
                target_dict.setdefault(key, {})
                target_dict[key][k] = v
    else:
        # save the state
        target_dict[key] = source_dict


# From `lightning.fabric.strategies.strategy.Strategy`
def _convert_stateful_objects_in_state(
    state: dict[str, Union[torch.nn.Module, torch.optim.Optimizer, Any]],
    filter: Optional[dict[str, Callable[[str, Any], bool]]] = None,
) -> dict[str, Any]:
    converted_state: dict[str, Any] = {}
    for key, obj in state.items():
        if hasattr(obj, "state_dict"):
            converted = obj.state_dict()
        else:
            converted = obj
        _apply_filter(key, filter, converted, converted_state)
    return converted_state


# From `lightning.Fabric.save`
def save_checkpoint(
    path: Union[str, Path],
    state: dict[str, Union[torch.nn.Module, torch.optim.Optimizer, Any]],
    storage_options: Optional[Any] = None,
    filter: Optional[dict[str, Callable[[str, Any], bool]]] = None,
):
    """
    All processes are synced by `Fabric.barrier()`.

    """
    if filter is not None:
        if not isinstance(filter, dict):
            raise TypeError(f"Filter should be a dictionary, given {filter!r}")
        if not set(filter).issubset(state):
            raise ValueError(
                f"The filter keys {filter.keys() - state} are not present in the state keys {set(state)}."
            )
        for k, v in filter.items():
            if not callable(v):
                raise TypeError(
                    f"Expected `save_checkpoint(filter=...)` for key {k!r} to be a callable, given {v!r}"
                )
    state = _convert_stateful_objects_in_state(
        _unwrap_objects(state),
        filter=(filter or {}),
    )
    if Fabric.rank() == 0:
        checkpoint_io = TorchCheckpointIO()
        checkpoint_io.save_checkpoint(
            checkpoint=state, path=path, storage_options=storage_options
        )
    Fabric.barrier()


def save_model_checkpoint(
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
    Fabric.print(f"\nSaving model weights to {str(file_path)!r}")
    save_checkpoint(file_path, state={"model": model.gpt_model}, **save_kwargs)
    if model.head_model.state_dict():
        file_path = file_dir / HEAD_MODEL_FNAME
        Fabric.print(f"Saving head model weights to {str(file_path)!r}")
        save_checkpoint(file_path, state={"model": model.head_model})


# From `litgpt.utils.load_checkpoint`
def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    strict: bool = True,
) -> None:
    state_dict = lazy_load(checkpoint_path)
    state_dict = state_dict.get("model", state_dict)
    model.load_state_dict(state_dict, strict=strict)


def load_model_checkpoint(
    model: GPTAndHeadModel,
    checkpoint_dir: Path,
    resume_dir: Optional[Path] = None,
) -> None:
    """
    Loads weights of `model` from a model checkpoint. Depends on whether the
    model is of LoRA type or not:

    * Normal type (no LoRA): If `resume_dir` is given, weights are loaded from
        there. Otherwise, weights are loaded from `checkpoint_dir`.
    * LoRA: Base model weights are loaded from `checkpoint_dir`. Afterwards, if
        `resume_dir` is given, LoRA weights are loaded from there. Otherwise,
        LoRA weights are reset / initialized at random.

    Args:
        fabric: Fabric instance
        model: Model to load weights into
        checkpoint_dir: Base model checkpoint loaded from there
        resume_dir: Optional. See above.

    """
    is_lora = is_lora_model(model)
    file_path = checkpoint_dir / LIT_MODEL_FNAME
    if not is_lora and resume_dir is not None:
        file_path = resume_dir / LIT_MODEL_FNAME
    Fabric.print(f"Loading model checkpoint: {file_path}")
    load_checkpoint(model.gpt_model, file_path, strict=not is_lora)
    if is_lora:
        if resume_dir is not None:
            file_path = resume_dir / LORA_WEIGHTS_FNAME
            Fabric.print("Loading LoRA weights checkpoint")
            load_checkpoint(model.gpt_model, file_path, strict=False)
        else:
            Fabric.print("Reset/initialize LoRA weights")
            model.gpt_model.reset_lora_parameters()
    # If there are head model weights, load them as well. Otherwise, we use
    # random initialization (or the head model may not have weights)
    if resume_dir is not None:
        file_path = resume_dir / HEAD_MODEL_FNAME
        if file_path.exists():
            load_checkpoint(model.head_model, file_path, strict=True)


def choose_logger(
    logger_name: Literal["csv", "tensorboard", "wandb", "mlflow"],
    out_dir: Path,
    name: str,
    log_interval: int = 1,
    log_args: Optional[Dict] = None,
    resume: Optional[bool] = None,
    **kwargs: Any,
) -> Logger:
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
    from keys_values.distributed.model_factory import BlockComponentName

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
    return may_match_twice_flex_attention_sdpa


def fix_dtype_of_score_buffers(gpt_model: GPT):
    """
    This function is needed to undo the annoying property of `fabric.setup`
    to change the score buffer dtypes.

    """
    from keys_values.kvcache.attn_weights import AttnWeightsKVCache

    for cache in gpt_model.get_kv_caches():
        if isinstance(cache, AttnWeightsKVCache):
            cache.fix_dtype_of_score_buffers()


def _get_smart_lastrec_info(
    data: DataModule,
    tokenizer: HFTokenizer,
) -> SmartInitialInformation:
    from keys_values.data.longbench_v2 import LongBenchV2
    from keys_values.data.helmet import Helmet

    if isinstance(data, LongBenchV2) or isinstance(data, Helmet):
        return data.smart_lastrec_info(tokenizer)
    else:
        raise ValueError(
            f"data of type {type(data)} does not provide SmartInitialInformation. Implement `data.smart_lastrec_info`"
        )


def adjust_cache_kwargs(
    kv_cache: KVCacheArgs,
    data: DataModule,
    tokenizer: Union[HFTokenizer, Tokenizer],
):
    """
    Called before :func:`wrap_gpt_model`. Sets fields in
    `kv_cache.cache_kwargs`, needed to create KV cache of type `kv_cache.name`.

    Note: Other adjustments of `kv_cache.cache_kwargs` are made in
    :func:`get_mha_and_cache_kwargs`, but they pertain to multi-head attention,
    not to the KV cache type.

    """
    if kv_cache.name.startswith("smart-lastrec"):
        cache_kwargs = kv_cache.cache_kwargs
        if cache_kwargs is None:
            cache_kwargs = dict()
        if isinstance(tokenizer, Tokenizer):
            tokenizer = tokenizer.processor
        assert isinstance(
            tokenizer, HFTokenizer
        ), f"type(tokenizer) = {type(tokenizer)} not supported"
        cache_kwargs["tokenizer"] = tokenizer
        smart_lastrec_info = None
        for name in (
            "end_initial_regex",
            "max_initial_fraction",
            "include_end_string",
        ):
            if name not in cache_kwargs:
                print(f"adjust_cache_kwargs: Setting {name} from data module")
                if smart_lastrec_info is None:
                    smart_lastrec_info = _get_smart_lastrec_info(data, tokenizer)
                cache_kwargs[name] = getattr(smart_lastrec_info, name)
        kv_cache.cache_kwargs = cache_kwargs


def copy_config_files(
    source_dir: Path,
    out_dir: Path,
    include_tokenizer_files: bool = False,
) -> None:
    """Copies the specified configuration and tokenizer files into the output directory."""

    all_files = ("config.json", "generation_config.json", "model_config.yaml")
    if include_tokenizer_files:
        all_files += ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")

    for file_name in all_files:
        src_path = source_dir / file_name
        if src_path.exists():
            shutil.copy(src_path, out_dir)


_GENERATION_CONFIG_KEYS = (
    "temperature",
    "top_k",
    "top_p",
)


def load_generation_config(
    checkpoint_dir: Path,
    eval_args: EvalArgs,
) -> EvalArgs:
    path = checkpoint_dir / "generation_config.json"
    generation_config = None
    if path.exists():
        with open(path, "r") as fp:
            generation_config = {
                k: v for k, v in json.load(fp).items() if k in _GENERATION_CONFIG_KEYS
            }
        if not generation_config:
            generation_config = None
        else:
            print(f"Loaded generation config from {path}")
    if generation_config is not None:
        sample_kwargs = (
            generation_config
            if eval_args.sample_metric_kwargs is None
            else eval_args.sample_metric_kwargs.update(generation_config)
        )
        eval_args = replace(eval_args, sample_metric_kwargs=sample_kwargs)
    return eval_args


# From `lightning.fabric.connector._check_and_init_precision`
def _check_and_init_precision(precision: str) -> Precision:
    if precision in ("16-true", "bf16-true"):
        return HalfPrecision(precision)
    if precision == "32-true":
        return Precision()
    if precision == "64-true":
        return DoublePrecision()
    if precision == "transformer-engine":
        return TransformerEnginePrecision(weights_dtype=torch.bfloat16)
    if precision == "transformer-engine-float16":
        return TransformerEnginePrecision(weights_dtype=torch.float16)
    if precision in ("16-mixed", "bf16-mixed"):
        Fabric.print(
            "Using 16-bit Automatic Mixed Precision (AMP)"
            if precision == "16-mixed"
            else "Using bfloat16 Automatic Mixed Precision (AMP)"
        )
        return MixedPrecision(precision=precision, device="cuda")
    raise RuntimeError(f"precision={precision} not supported")


# From `lightning.fabric.Fabric.__init__`,
# `lightning.fabric.connector._Connector`.
def get_fabric_precision(precision: Optional[str] = None) -> Precision:
    precision_input = _convert_precision_to_unified_args(precision)
    if precision_input is None:
        precision_input = "32-true"
    return _check_and_init_precision(precision_input)


# From `lightning.fabric.fabric.Fabric.init_module`,
# `lightning.fabric.strategies.strategy.Strategy.module_init_context`.
def init_module(
    empty_init: bool = False,
    precision: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> AbstractContextManager:
    """
    This context manager can be used to speed up model creation. By default,
    PyTorch creates modules on CPU, using `float32` as type.

    Args:
        empty_init: If `True`, parameters are not initialized. Use this only
            if a checkpoint is loaded afterwards, which initializes all parameters.
        precision: Parameters are created for this precision and dtype. Defaults
            to "32-true" (which is `float32`).
        device: PyTorch creates all parameters on this device directly. Defaults
            to `Fabric.device()`.

    """
    if device is None:
        device = Fabric.device()
    precision_module_ctx = get_fabric_precision(precision).module_init_context()
    stack = ExitStack()
    stack.enter_context(device)
    stack.enter_context(_EmptyInit(enabled=empty_init))
    stack.enter_context(precision_module_ctx)
    return stack


# From `lightning.fabric.fabric.Fabric.log_dict
def fabric_log_dict(
    loggers: List[Logger],
    metrics: Mapping[str, Any],
    step: Optional[int] = None,
):
    if loggers:
        metrics = convert_tensors_to_scalars(metrics)
        for logger in loggers:
            logger.log_metrics(metrics=metrics, step=step)

def get_hyperparameters_from_original_setup(
    original_setup: Callable,
    checkpoint_dir: Path,
) -> Dict[str, Any]:
    # Extract hyperparameters (needed for storing checkpoints)
    _hp_path = checkpoint_dir / "__TEMP_31415927__" / HYPERPARAMETERS_FILENAME
    _hp_path.parent.mkdir(parents=True, exist_ok=True)
    save_hyperparameters(
        original_setup,
        checkpoint_dir=_hp_path.parent,
    )
    hyperparameters = yaml.safe_load(_hp_path.open())
    _hp_path.unlink()
    _hp_path.parent.rmdir()
    return hyperparameters
