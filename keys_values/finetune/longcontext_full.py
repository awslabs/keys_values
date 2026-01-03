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
import csv
import dataclasses
import random
import sys

from datetime import datetime
import math
import os
import time
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union, Any, List

import lightning as L
import torch
from lightning.fabric.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torchmetrics import RunningMean
from torch.nn.attention import SDPBackend

from litgpt.args import EvalArgs as _EvalArgs, TrainArgs
from litgpt.data import DataModule
from litgpt.generate.base import generate
from litgpt.config import Config
from litgpt.prompts import save_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    CycleIterator,
    auto_download_checkpoint,
    check_nvlink_connectivity,
    check_valid_checkpoint_dir,
    choose_logger as _choose_logger,
    copy_config_files,
    create_finetuning_performance_report,
    find_resume_path,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    load_checkpoint,
    num_parameters,
    parse_devices,
    select_sft_generate_example,
)

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.attention_utils import DEFAULT_TMP_ARRAY_LIMIT_GB
from keys_values.data import LongBenchV2, INPUT_IDS_NAME
from keys_values.finetune.args import OptimizerArgs
from keys_values.finetune.batch_transform import (
    BatchTransformFactory, BatchTransform,
)
from keys_values.finetune.utils import LIT_MODEL_FNAME, HEAD_MODEL_FNAME
from keys_values.head_model import HeadModel, CrossEntropyOnLogits
from keys_values.head_model_factory import HeadModelFactory
from keys_values.kvcache.factory import (
    KVCacheFactory,
    deallocate_kv_cache_buffers_of_model,
    cleanup_cache_kwargs,
    split_name,
)
from keys_values.kvcache.gradient.gpu_memory import RecordGPUMemory
from keys_values.kvcache.gradient.main import (
    LongContextGradientModel,
    NaiveGPTAndHeadModel,
)
from keys_values.kvcache.utils import (
    fabric_precision_to_dtype,
    VerbosityLevels,
    message_memory_all_devices,
    log_memory_all_devices,
)
from keys_values.long_context import (
    KVCacheArgs,
    GPTAndHeadModel,
    LongContextInferenceModel,
)
from keys_values.model import GPT
from keys_values.parser_config import save_hyperparameters
from keys_values.pos_encoding import position_encoding_factory


DEFAULT_OUT_DIR = "out/finetune/longcontext_full"

DEBUG_NUM_SELECTED_PARAMS = 20


@dataclass
class EvalArgs(_EvalArgs):
    """
    If `micro_batch_size` is not given, `train.micro_batch_size` is used.

    """
    micro_batch_size: Optional[int] = None


def setup(
    checkpoint_dir: Path,
    out_dir: Path = Path(DEFAULT_OUT_DIR),
    precision: Optional[str] = None,
    devices: Union[int, str] = 1,
    num_nodes: int = 1,
    resume: Union[bool, Literal["auto"], Path] = False,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=16,
        micro_batch_size=1,
        lr_warmup_steps=None,
        lr_warmup_fraction=0.15,
        epochs=5,
        max_seq_length=None,
    ),
    eval: EvalArgs = EvalArgs(
        interval=600,
        max_new_tokens=100,
        max_iters=100,
        initial_validation=None,  # Default set below
        final_validation=True,
    ),
    optimizer: Optional[OptimizerArgs] = None,
    logger_name: Literal["wandb", "tensorboard", "csv", "mlflow"] = "csv",
    seed: int = 1337,
    access_token: Optional[str] = None,
    kv_cache: KVCacheArgs = KVCacheArgs(
        name="h2o-torch-quantized8",
        cache_length=8192,
        layers_per_cell=1,
        chunk_size=256,
        cache_kwargs={
            "replay_log_blocksize": 1024,
            "allocate_buffers": False,
            "max_num_ranges": 4,
        },
        randomize_chunk_sizes=False,
        chunks_per_cell_multiplier=1.0,
        single_tokens_for_targets=False,
        verbose=VerbosityLevels.SOME.value,
        attention_forward_temp_size_gb=4,
        attention_backward_temp_size_gb=2,
        use_new_cache=False,
        max_match_trials_pack_arg=8,
        layer_checkpoint_chunk_size=None,
    ),
    head_model: str = CrossEntropyOnLogits.NAME,
    head_model_kwargs: Optional[Dict[str, Any]] = None,
    yarn_rope: bool = True,
    record_gpu_memory_snapshots: Optional[int] = None,
    record_gpu_memory_kind: int = 0,
    record_gpu_memory_period: int = 0,
    debug_check_updates: bool = False,
    profile_grad_times: int = 0,
) -> None:
    """Finetune a model.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to load for finetuning.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        devices: How many devices/GPUs to use
        num_nodes: How many nodes the code is being run on.
        resume: Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
            from the latest checkpoint in ``out_dir``. An error will be raised if no checkpoint is found. Passing
            ``'auto'`` will resume from the latest checkpoint but not error if no checkpoint exists.
        data: Data-related arguments. If not provided, the default is
            ``keys_values.data.LongBenchV2``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
            Note: We modified the defaults from `train.lr_warmup_steps=100` to
            `train.lr_warmup_fraction=0.15`, so the linear warm-up is the first
            15% of all steps.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: Selects optimizer and its parameters, see
            ``keys_values.finetune.args.OptimizerArgs`` for details. Defaults to
            "AdamW" with default parameters.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
        kv_cache: Configuration for the KV caches. Defaults to H2O with PyTorch
            8-bit quantization and cache length 8192. Should be increased if GPU
            memory is sufficient. Also consider increasing layers per cell.
        kv_cache: Configuration for the KV caches
        head_model: Name of the head model to use, see
            :class:`HeadModelFactory`. Defaults to "next_token_prediction"
        head_model_kwargs: Extra keyword arguments to pass to the head model
            factory.
        yarn_rope: Should YaRN be used to adjust RoPE (position encoding) to the
            sequence length for each batch? Defaults to `True`. If not, RoPE is
            determined by the model configuration, and is static (no dependence
            on sequence length).
        record_gpu_memory_snapshots: If given, we record GPU memory traces in
            snapshots. This argument is the `max_entries` parameter, a good
            value is 50000 or 100000.
        record_gpu_memory_kind: There are different GPU memory recording
            strategies, selected by this argument:
            - 0: One snapshot file per update step, recording during all
                computations.
            - 1: Only record gradient computations (after initial forward). For
                each update, we store one snapshot file per row of cells being
                processed.
            Defaults to 0.
        record_gpu_memory_period: Only if `record_gpu_memory_snapshots` is used.
            Snapshot files are written once per update step. Files are overwritten
            on this period, in that those for step `step` are written to
            directory `f"iteration{step % record_gpu_memory_period}"`.
            If this is 0, files are not overwritten, we use `f"iteration{step}"`.
            Defaults to 0.

    """
    checkpoint_dir = auto_download_checkpoint(model_name=checkpoint_dir, access_token=access_token)
    pprint(locals())
    data = LongBenchV2() if data is None else data
    if isinstance(data, LongBenchV2) and data.metadata_dir is None:
        data.metadata_dir = str(out_dir / "data")
        print(f"Setting LongBenchV2.metadata_dir to {data.metadata_dir}")
    out_dir = init_out_dir(out_dir)
    data.metadata_dir = str(init_out_dir(data.metadata_dir))
    if head_model_kwargs is None:
        head_model_kwargs = dict()
    devices = parse_devices(devices)
    if eval.initial_validation is None:
        # Run initial evaluation in multi-device setup, but not with a
        # single device
        eval.initial_validation = devices > 1
    if optimizer is None:
        optimizer = OptimizerArgs(name="AdamW")
        print("Choosing optimizer AdamW with default learning rate. We highly recommend to at least tune optimizer.learning_rate")
    else:
        print(str(optimizer))

    check_kv_cache(kv_cache)
    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")

    precision = precision or get_default_supported_precision(training=True)
    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"finetune-{config.name}",
        use_fabric=True,
        resume=bool(resume),
        log_interval=train.log_interval,
    )

    if devices * num_nodes > 1:
        # FSDP without cpu off load
        #strategy = FSDPStrategy(
        #    auto_wrap_policy={Block},
        #    activation_checkpointing_policy={Block},
        #    state_dict_type="full",
        #    limit_all_gathers=True,
        #    cpu_offload=False,
        #)
        # Baseline DDP strategy (just data parallelism)
        # - static_graph=True: Optimizes communication by assuming the
        #   computation graph is the same every iteration (no dynamic control
        #   flow, like in MoE). Without this flag, our code does not work,
        #   because it calls `autograd.backwards` several times internally.
        # - broadcast_buffers=False: Avoids broadcasting model buffers from rank
        #   0 to others at every forward pass, which can save bandwidth and avoid
        #   conflicting with our own buffer manipulation.
        # Note: Strictly speaking, the computation graph during training
        # changes with each new batch, since the graph depends on the sequence
        # length of the batch. However, DDP still seems to work.
        strategy = DDPStrategy(static_graph=True, broadcast_buffers=False)
    else:
        strategy = "auto"

    fabric = L.Fabric(
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        loggers=logger,
    )

    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    if record_gpu_memory_snapshots is not None:
        record_gpu_memory_snapshots = RecordGPUMemory(
            max_entries=record_gpu_memory_snapshots,
        )
    fabric.launch(
        main,
        devices=devices,
        resume=resume,
        seed=seed,
        config=config,
        data=data,
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        train=train,
        eval=eval,
        optimizer=optimizer,
        kv_cache=kv_cache,
        num_nodes=num_nodes,
        head_model_name=head_model,
        head_model_kwargs=head_model_kwargs,
        yarn_rope=yarn_rope,
        record_gpu_memory_snapshots=record_gpu_memory_snapshots,
        record_gpu_memory_kind=record_gpu_memory_kind,
        record_gpu_memory_period=record_gpu_memory_period,
        debug_check_updates=debug_check_updates,
        profile_grad_times=profile_grad_times,
    )


def main(
    fabric: L.Fabric,
    devices: int,
    resume: Union[bool, Literal["auto"], Path],
    seed: int,
    config: Config,
    data: DataModule,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: OptimizerArgs,
    kv_cache: KVCacheArgs,
    num_nodes: int,
    head_model_name: str,
    head_model_kwargs: Dict[str, Any],
    yarn_rope: bool,
    record_gpu_memory_snapshots: Optional[RecordGPUMemory],
    record_gpu_memory_kind: int,
    record_gpu_memory_period: int,
    debug_check_updates: bool,
    profile_grad_times: int,
) -> None:
    validate_args(train, eval)

    tokenizer = Tokenizer(checkpoint_dir)
    train_dataloader, val_dataloader = get_dataloaders(
        data=data,
        tokenizer=tokenizer,
        head_model=head_model_name,
        train=train,
        eval=eval,
        fabric=fabric,
    )
    ignore_index = getattr(data, "ignore_index", -100)
    batch_transform = BatchTransformFactory.from_head_model(
        head_model=head_model_name,
        pad_id=0,
        eos_id=tokenizer.eos_id,
        ignore_index=ignore_index,
    )
    steps_per_epoch = len(train_dataloader) // train.gradient_accumulation_iters(devices, num_nodes)
    lr_max_steps = min(train.epochs * steps_per_epoch, (train.max_steps or float("inf")))

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        # Order of preference for SDPA kernels
        sdpa_kernels = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.MATH,
        ]
        limit_gb = kv_cache.attention_forward_temp_size_gb
        if limit_gb is None:
            limit_gb = DEFAULT_TMP_ARRAY_LIMIT_GB
        print_message(
            f"Setting limit attention_forward_temp_size_gb to {limit_gb} GB",
            fabric,
        )
        tmp_array_limit_forward = TemporaryArrayLimit(
            init_val=limit_gb,
            name="attention_forward_temp_size_gb",
        )
        mha_kwargs = dict(
            sdpa_kernels=sdpa_kernels,
            tmp_array_limit_gb=tmp_array_limit_forward,
            pos_encoding=position_encoding_factory(config, do_yarn=yarn_rope),
        )
        if "sdpa_kernels" not in kv_cache.cache_kwargs:
            kv_cache.cache_kwargs["sdpa_kernels"] = sdpa_kernels
        kv_cache.cache_kwargs["tmp_array_limit_gb"] = tmp_array_limit_forward
        kv_cache.cache_kwargs["pos_encoding"] = mha_kwargs["pos_encoding"]
        gpt_model = GPT(config, **mha_kwargs)
        head_model = HeadModelFactory.create(
            name=head_model_name,
            config=config,
            data=data,
            **head_model_kwargs,
        )
        batch_size = train.micro_batch_size
        if eval.micro_batch_size is not None:
            batch_size = max(batch_size, eval.micro_batch_size)
        model = wrap_gpt_model(
            gpt_model=gpt_model,
            head_model=head_model,
            kv_cache=kv_cache,
            max_batch_size=batch_size,
            dtype=fabric_precision_to_dtype(fabric._precision.precision),
            profile_grad_times=profile_grad_times > 0,
            fabric=fabric,
        )

    num_trainable_params = num_parameters(model, requires_grad=True)
    print_message(
        f"\nNumber of trainable parameters: {num_trainable_params:,}",
        fabric,
    )

    model = fabric.setup(model)

    optimizer = instantiate_torch_optimizer(
        optimizer.name, model.parameters(), **optimizer.optimizer_kwargs(),
    )
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_lr_scheduler(optimizer, train_args=train, max_steps=lr_max_steps)
    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "iter_num": 0,
        "step_count": 0,
    }

    resume = find_resume_path(resume, out_dir)
    if resume:
        print_message(f"Resuming training from {resume}", fabric)
        fabric.load(resume, state)
    else:
        file_path = checkpoint_dir / LIT_MODEL_FNAME
        load_checkpoint(fabric, model.gpt_model, file_path, strict=True)
        # If there are head model weights, load them as well. Otherwise, we use
        # random initialization (or the head model may not have weights)
        file_path = checkpoint_dir / HEAD_MODEL_FNAME
        if file_path.exists():
            load_checkpoint(fabric, model.head_model, file_path, strict=True)

    if profile_grad_times > 0:
        thresh = kv_cache.max_match_trials_pack_arg
        name = "new" if kv_cache.use_new_cache else "old"
        profile_grad_params = {
            "path": Path(out_dir) / f"profile_grad_times_{name}_{thresh}.csv",
            "use_new_cache": kv_cache.use_new_cache,
            "max_match_trials_pack_arg": thresh,
            "profile_grad_times": profile_grad_times,
            "cache_name": kv_cache.name,
        }
    else:
        profile_grad_params = None
    train_time = time.perf_counter()
    token_counts = fit(
        fabric=fabric,
        state=state,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        batch_transform=batch_transform,
        devices=devices,
        resume=resume,
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        train=train,
        eval=eval,
        data=data,
        num_nodes=num_nodes,
        record_gpu_memory_snapshots=record_gpu_memory_snapshots,
        record_gpu_memory_kind=record_gpu_memory_kind,
        record_gpu_memory_period=record_gpu_memory_period,
        debug_check_updates=debug_check_updates,
        num_trainable_params=num_trainable_params,
        profile_grad_params=profile_grad_params,
    )
    training_time = time.perf_counter() - train_time
    output = create_finetuning_performance_report(training_time, token_counts, fabric.device.type)
    print_message(output, fabric)

    # Final evaluation
    if eval.final_validation:
        print_with_rank_and_timestamp("Starting validation evaluations.", fabric.global_rank)
        print_message("\nFinal validation evaluation ...", fabric)
        metrics = validate_and_all_reduce(
            model=model,
            val_dataloader=val_dataloader,
            eval=dataclasses.replace(eval, max_iters=len(val_dataloader)),
            batch_transform=batch_transform,
            log_metrics=False,
            fabric=fabric,
        )
        fabric.log_dict(metrics, step=state["iter_num"])
        print_message(
            f"Final evaluation | val loss: {metrics['val_loss']:.3f} | val ppl: {metrics['val_ppl']:.3f} | val_time_in_ms: {metrics['val_time_in_ms']:.3f}",
            fabric,
        )
        flush_io_streams()

    # Save the final checkpoint at the end of training
    save_dir = out_dir / "final"
    save_model_checkpoint(fabric, model, save_dir)
    if fabric.global_rank == 0:
        # Copy checkpoint files from original checkpoint dir
        copy_config_files(checkpoint_dir, save_dir)
        save_hyperparameters(setup, save_dir)
        if hasattr(data, "prompt_style"):
            save_prompt_style(data.prompt_style, save_dir)


def print_message(msg: str, fabric: Optional[L.Fabric] = None):
    if fabric is not None:
        fabric.print(msg)
    else:
        print(msg)


# TODO: Support caches of different lengths, maybe even different types
def wrap_gpt_model(
    gpt_model: GPT,
    head_model: HeadModel,
    kv_cache: KVCacheArgs,
    max_batch_size: int,
    dtype: torch.dtype,
    model_for_training: bool = True,
    profile_grad_times: bool = False,
    cpu_offload_device: Optional[torch.device] = None,
    fabric: Optional[L.Fabric] = None,
) -> LongContextGradientModel:
    print_message(
        "Assigning KV caches to layers of model:\n"
        f"name:           {kv_cache.name}\n"
        f"cache_length:   {kv_cache.cache_length}\n"
        f"max_batch_size: {max_batch_size}",
        fabric,
    )
    gpt_model.clear_kv_caches()
    cache_kwargs = cleanup_cache_kwargs(
        split_name(kv_cache.name)[0], kv_cache.cache_kwargs,
    )
    tmp_array_limit_gb = cache_kwargs.get("tmp_array_limit_gb")
    if tmp_array_limit_gb is not None:
        del cache_kwargs["tmp_array_limit_gb"]
    kv_caches = KVCacheFactory.create(
        gpt_model=gpt_model,
        name=kv_cache.name,
        max_batch_size=max_batch_size,
        dtype=dtype,
        cache_length=kv_cache.cache_length,
        cache_kwargs=cache_kwargs,
    )
    gpt_model.assign_kv_caches(kv_caches)
    common_kwargs = dict(
        gpt_model=gpt_model,
        head_model=head_model,
        chunk_size=kv_cache.chunk_size,
        randomize_chunk_sizes=kv_cache.randomize_chunk_sizes,
        chunks_per_cell_multiplier=kv_cache.chunks_per_cell_multiplier,
        single_tokens_for_targets=kv_cache.single_tokens_for_targets,
        verbose=kv_cache.verbosity_level,
        tmp_array_limit_gb=tmp_array_limit_gb,
    )
    if model_for_training:
        # Temp array size limit can be different for backward and forward
        limit_gb = kv_cache.attention_backward_temp_size_gb
        if limit_gb is None:
            limit_gb = kv_cache.attention_forward_temp_size_gb
            if limit_gb is None:
                limit_gb = DEFAULT_TMP_ARRAY_LIMIT_GB
        print_message(
            f"Setting limit attention_backward_temp_size_gb to {limit_gb} GB",
            fabric,
        )
        backward_tmp_array_limit_gb = TemporaryArrayLimit(
            init_val=limit_gb,
            name="attention_backward_temp_size_gb",
        )
        train_cache_kwargs = {
            "sdpa_kernels": cache_kwargs["sdpa_kernels"],
            "use_new_cache": kv_cache.use_new_cache,
        }
        if kv_cache.max_match_trials_pack_arg is not None:
            autograd_hooks_kwargs = dict(
                max_match_trials_pack_arg=kv_cache.max_match_trials_pack_arg,
            )
        else:
            autograd_hooks_kwargs = None
        if cpu_offload_device is not None:
            common_kwargs["head_model"] = head_model.to(device=cpu_offload_device)
        layer_checkpoint_chunk_size = kv_cache.layer_checkpoint_chunk_size
        if layer_checkpoint_chunk_size is None:
            # Default value for chunk size if not given
            layer_checkpoint_chunk_size = kv_cache.cache_length
            add_msg = " (default)"
        else:
            add_msg = ""
        if kv_cache.qname != "default":
            print_message(
                f"Using layer_checkpoint_chunk_size = {layer_checkpoint_chunk_size}" + add_msg,
                fabric,
            )
        model = LongContextGradientModel(
            **common_kwargs,
            layers_per_cell=kv_cache.layers_per_cell,
            qname=kv_cache.qname,
            cache_kwargs=cache_kwargs,
            train_cache_kwargs=train_cache_kwargs,
            backward_tmp_array_limit_gb=backward_tmp_array_limit_gb,
            autograd_hooks_kwargs=autograd_hooks_kwargs,
            profile_steps=profile_grad_times,
            offload_device=cpu_offload_device,
            layer_checkpoint_chunk_size=layer_checkpoint_chunk_size,
        )
    else:
        model = LongContextInferenceModel(**common_kwargs)
    return model


def create_baseline_model(
    gpt_model: GPT,
    config: Config,
    head_model_name: str,
    data: DataModule,
    head_model_kwargs: Dict[str, Any],
) -> NaiveGPTAndHeadModel:
    head_model = HeadModelFactory.create(
        name=head_model_name,
        config=config,
        data=data,
        **head_model_kwargs,
    )
    return NaiveGPTAndHeadModel(
        gpt_model=gpt_model,
        head_model=head_model,
    )


def fit(
    fabric: L.Fabric,
    state: Dict[str, Any],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    batch_transform: BatchTransform,
    devices: int,
    resume: Union[bool, Literal["auto"], Path],
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    data: DataModule,
    num_nodes: int,
    record_gpu_memory_snapshots: Optional[RecordGPUMemory],
    record_gpu_memory_kind: int,
    record_gpu_memory_period: int,
    debug_check_updates: bool,
    num_trainable_params: int,
    profile_grad_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    tokenizer = Tokenizer(checkpoint_dir)

    # Initial evaluation
    token_counts = {
        "raw_tokens": torch.tensor(0, device=fabric.device, dtype=torch.long),
        "raw_tokens_plus_prompt_template": torch.tensor(0, device=fabric.device, dtype=torch.long),
        "raw_tokens_plus_prompt_template_and_padding": torch.tensor(0, device=fabric.device, dtype=torch.long),
    }
    val_loss = "n/a"
    if eval.initial_validation:
        print_with_rank_and_timestamp("Starting validation evaluations.", fabric.global_rank)
        print_message("\nInitial validation evaluation ...", fabric)
        metrics = validate_and_all_reduce(
            model=model,
            val_dataloader=val_dataloader,
            eval=dataclasses.replace(eval, max_iters=len(val_dataloader)),
            batch_transform=batch_transform,
            fabric=fabric,
        )
        val_loss = f"{metrics['val_loss']:.3f}"
        print_message(
            f"Initial evaluation | val loss: {val_loss} | val ppl: {metrics['val_ppl']:.3f} | val_time_in_ms: {metrics['val_time_in_ms']:.3f}",
            fabric,
        )
        flush_io_streams()
    else:
        print_message("Verifying settings ...", fabric)
        with torch.no_grad():
            validate(
                model,
                val_dataloader,
                dataclasses.replace(eval, max_iters=2),
                batch_transform,
            )  # sanity check

    initial_iter = state["iter_num"]
    max_steps = train.max_steps or float("inf")
    train_iterator = CycleIterator(train_dataloader)

    # resume data loader state by fast-forwarding through all seen batches
    if resume:
        resume_t0 = time.perf_counter()
        for resume_iter in range(initial_iter):
            next(train_iterator)
            if resume_iter % 1000 == 0:
                print_message(
                    f"Resuming dataset: {resume_iter} / {initial_iter}",
                    fabric,
                )
        fabric.barrier()
        print_message(
            f"Resuming data loader finished. Took {time.perf_counter() - resume_t0:.1f}"
            " seconds to reach iteration {initial_iter}.",
            fabric,
        )

    running_loss = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=False).to(
        fabric.device
    )
    fabric.barrier()
    print_message(
        "\nGPU memory before training starts:\n" + message_memory_all_devices(),
        fabric,
    )

    while state["step_count"] < max_steps:
        state["iter_num"] += 1
        iter_t0 = time.perf_counter()
        batch = batch_transform(next(train_iterator))
        if train_iterator.epoch >= train.epochs:
            break

        if record_gpu_memory_snapshots is not None:
            run_no = state['iter_num'] - 1
            if record_gpu_memory_period >= 1:
                run_no = run_no % record_gpu_memory_period
            name = "snapshot.pickle" if record_gpu_memory_kind == 0 else "snapshot_initial.pickle"
            path = out_dir / "gpu_memory_snapshots" / f"iteration{run_no}" / name
            record_gpu_memory_snapshots = RecordGPUMemory(
                path=str(path),
                max_entries=record_gpu_memory_snapshots.max_entries,
            )
            record_gpu_memory_snapshots.start_recording()

        is_accumulating = state["iter_num"] % train.gradient_accumulation_iters(devices, num_nodes) != 0
        print_with_rank_and_timestamp("Starting gradient computation.", fabric.global_rank)
        if debug_check_updates:
            debug_names = debug_random_trainable_param_names(
                model, DEBUG_NUM_SELECTED_PARAMS, num_trainable_params,
            )
            debug_orig_params = debug_clone_selected_params(model, debug_names)
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            loss = model(
                input_ids=batch[INPUT_IDS_NAME],
                targets=batch["targets"],
                scale_factor=1.0 / train.gradient_accumulation_iters(devices, num_nodes),
                record_gpu_memory_snapshots=record_gpu_memory_snapshots,
                record_gpu_memory_kind=record_gpu_memory_kind if record_gpu_memory_snapshots is not None else None,
            )
            fabric.backward(loss)

        running_loss.update(loss.detach())
        flush_io_streams()
        if profile_grad_params is not None:
            records = model.profile_records()
            skip_names = ("path", "profile_grad_times")
            fixed_col_names = [
                name
                for name in profile_grad_params.keys()
                if name not in skip_names
            ]
            prefix = [profile_grad_params[name] for name in fixed_col_names]
            var_col_names = list(records[0].keys())
            with profile_grad_params["path"].open("w") as fp:
                writer = csv.writer(fp, delimiter=",")
                writer.writerow(fixed_col_names + var_col_names)
                for record in records:
                    row = prefix + [record[name] for name in var_col_names]
                    writer.writerow(row)
            num_steps = profile_grad_params["profile_grad_times"]
            if len(records) >= num_steps:
                print(f"Done {num_steps} updates. Stopping.")
                exit(0)

        if record_gpu_memory_snapshots is not None:
            # Stop recording and store snapshot. For kind 0, this is the single
            # snapshot for the iteration. For kind 1, this is the final snapshot.
            record_gpu_memory_snapshots.store_current_snapshot()
            record_gpu_memory_snapshots.stop_recording()

        if not is_accumulating:
            print_with_rank_and_timestamp("Waiting for optimizer to update.", fabric.global_rank)
            if debug_check_updates:
                norm = debug_sum_gradient_norms(model)
                print_with_rank_and_timestamp(
                    f"Gradient average norm before update: {norm}",
                    fabric.global_rank,
                    start_newline=False,
                )
            optimizer.step()
            print_message("Optimizer update done.", fabric)
            if debug_check_updates:
                if fabric.global_rank == 0:
                    norm = debug_sum_gradient_norms(model)
                    print_with_rank_and_timestamp(
                        f"Gradient average norm after update: {norm}",
                        fabric.global_rank,
                        start_newline=False,
                    )
                num_changed = debug_compare_selected_params(model, debug_orig_params)
                msg = f"{num_changed} of {DEBUG_NUM_SELECTED_PARAMS} parameters changed"
                print_with_rank_and_timestamp(
                    msg, fabric.global_rank, start_newline=False,
                )
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            state["step_count"] += 1

        del loss
        torch.cuda.empty_cache()
        print_message(
            f"\nGPU memory at training step {state['iter_num'] - 1}:\n"
            + message_memory_all_devices() + "\n",
            fabric,
        )

        token_counts["raw_tokens"] += batch["token_counts"]["raw"].sum().item()
        token_counts["raw_tokens_plus_prompt_template"] += (
            batch["token_counts"]["raw_plus_prompt_template"].sum().item()
        )
        num_tokens = batch[INPUT_IDS_NAME].numel()
        token_counts["raw_tokens_plus_prompt_template_and_padding"] += num_tokens

        if state["iter_num"] % train.log_interval == 0:
            loss = running_loss.compute().item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "tokens": token_counts["raw_tokens_plus_prompt_template"],
                "total_tokens": token_counts["raw_tokens_plus_prompt_template"] * fabric.world_size,
                "learning_rate": scheduler.get_last_lr()[0],
                **log_memory_all_devices(),
            }
            if isinstance(val_loss, torch.Tensor):
                val_loss = f"{val_loss:.3f}"
            print_message(
                f"Epoch {metrics['epoch']} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" val: {val_loss} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} s"
                f"{' (step)' if not is_accumulating else ''}",
                fabric,
            )
            fabric.log_dict(metrics, step=state["iter_num"])

        if not is_accumulating and state["step_count"] % eval.interval == 0:
            print_with_rank_and_timestamp("Starting validation evaluations.", fabric.global_rank)
            print_message("\nPeriodic validation evaluation ...", fabric)
            generate_example_kwargs = dict(
                tokenizer=tokenizer,
                data=data,
            )
            # TODO: Fix bug in generation!
            metrics = validate_and_all_reduce(
                model=model,
                val_dataloader=val_dataloader,
                eval=eval,
                batch_transform=batch_transform,
                # generate_example_kwargs=generate_example_kwargs,
                log_metrics=False,
                fabric=fabric,
            )
            fabric.log_dict(metrics, step=state["iter_num"])
            print_with_rank_and_timestamp("Finished validation evaluations.", fabric.global_rank)
            val_loss = f"{metrics['val_loss']:.3f}"
            print_message(
                f"Epoch {train_iterator.epoch} | iter {state['iter_num']} | val loss: {val_loss} | val ppl: {metrics['val_ppl']:.3f} | val_time_in_ms: {metrics['val_time_in_ms']:.3f}",
                fabric,
            )
            flush_io_streams()
            fabric.barrier()

        if train.save_interval is not None and not is_accumulating and state["step_count"] % train.save_interval == 0:
            checkpoint_file = out_dir / f"step-{state['step_count']:06d}" / LIT_MODEL_FNAME
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            print_message(
                f"Saving checkpoint to {str(checkpoint_file.parent)!r}",
                fabric,
            )
            fabric.save(checkpoint_file, state)
            if fabric.global_rank == 0:
                copy_config_files(checkpoint_dir, checkpoint_file.parent)
                save_hyperparameters(setup, checkpoint_file.parent)
                if hasattr(data, "prompt_style"):
                    save_prompt_style(data.prompt_style, checkpoint_file.parent)

    return {
        key: fabric.all_reduce(token_counts[key], reduce_op="sum").item()
        for key in token_counts.keys()
    }


def check_kv_cache(kv_cache: KVCacheArgs):
    if kv_cache.name.startswith("dense"):
        raise ValueError(
            "kv_cache must be given for long-context inference, and "
            "kv_cache.name must not be dense-*"
        )


def validate_and_all_reduce(
    model: GPTAndHeadModel,
    val_dataloader: DataLoader,
    eval: EvalArgs,
    batch_transform: BatchTransform,
    generate_example_kwargs: Optional[Dict[str, Any]] = None,
    log_metrics: bool = True,
    fabric: Optional[L.Fabric] = None,
) -> Dict[str, float]:
    with torch.no_grad():
        deallocate_kv_cache_buffers_of_model(model.gpt_model)
        time_start = time.perf_counter()
        val_loss = validate(model, val_dataloader, eval, batch_transform)
        if generate_example_kwargs is not None:
            generate_example(
                fabric=fabric,
                model=model,
                eval=eval,
                **generate_example_kwargs,
            )
        val_time = time.perf_counter() - time_start
        # Validation can have larger batch size than training. Deallocate
        # buffers not to waste memory
        deallocate_kv_cache_buffers_of_model(model.gpt_model)

    if fabric is not None:
        val_loss_tensor = val_loss.clone().to(fabric.device)
        val_time_tensor = torch.tensor(
            val_time, device=fabric.device, dtype=torch.float32,
        )
        fabric.all_reduce(val_loss_tensor, reduce_op="mean")
        fabric.all_reduce(val_time_tensor, reduce_op="mean")
        val_time = val_time_tensor.item()
    else:
        val_loss_tensor = val_loss.clone()
    val_loss = val_loss_tensor.item()
    metrics = {
        "val_loss": val_loss,
        "val_ppl": math.exp(val_loss),
        "val_time_in_ms": val_time * 1000,
    }
    if fabric is not None and log_metrics:
        fabric.log_dict(metrics)
    return metrics


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(
    model: GPTAndHeadModel,
    val_dataloader: DataLoader,
    eval: EvalArgs,
    batch_transform: BatchTransform,
) -> torch.Tensor:
    model.eval()
    losses = torch.zeros(min(len(val_dataloader), eval.max_iters))
    for k, batch in enumerate(val_dataloader):
        if k >= eval.max_iters:
            break
        batch = batch_transform(batch)
        losses[k] = model(batch[INPUT_IDS_NAME], batch["targets"]).mean()

    val_loss = losses.mean()
    model.train()
    return val_loss


@torch.no_grad()
def generate_example(
    fabric: L.Fabric,
    model: GPTAndHeadModel,
    tokenizer: Tokenizer,
    eval: EvalArgs,
    data: DataModule,
):
    instruction = select_sft_generate_example(eval, data)
    print_message("\n[Instruction]:", fabric)
    print_but_limit_size(fabric, instruction)
    if hasattr(data, "prompt_style"):
        prompt = data.prompt_style.apply(instruction)
    else:
        prompt = instruction
    encoded = tokenizer.encode(prompt, device=fabric.device)
    gpt_model = model.gpt_model
    if not gpt_model.are_kv_caches_assigned():
        raise IndexError("model.gpt_model must have KV caches assigned")
    model.eval()

    max_returned_tokens = len(encoded) + eval.max_new_tokens

    if max_returned_tokens < gpt_model.max_seq_length:
        output = generate(
            model=gpt_model,
            prompt=encoded,
            max_returned_tokens=max_returned_tokens,
            temperature=0.8,
            include_prompt=False,
            eos_id=tokenizer.eos_id,
        )
        model.train()
        output = tokenizer.decode(output)
        print_message("\n[Generated Output (without prompt)]:", fabric)
        print_but_limit_size(fabric, output)
    else:
        print_message(
            f"Length of encoded instruction ({len(encoded)}) and eval.max_new_tokens ({eval.max_new_tokens}) "
            f"exceeds model.max_seq_length ({gpt_model.max_seq_length}) used for training. Skipping example generation for efficiency. "
            f"The model's supported context size (post-training) is {gpt_model.config.block_size}.",
            fabric,
        )


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
            raise ValueError("Either train.lr_warmup_fraction or train_args.lr_warmup_steps must be given")
        warmup_steps = min(train_args.lr_warmup_steps, max_steps)
    else:
        if not (0 <= train_args.lr_warmup_fraction <= 1):
            raise ValueError(f"train_args.lr_warmup_fraction = {train_args.lr_warmup_fraction}, must be in [0, 1]")
        if train_args.lr_warmup_steps is not None:
            print(f"train.lr_warmup_fraction = {train_args.lr_warmup_fraction}, train_args.lr_warmup_steps = {train_args.lr_warmup_steps}. Using the former.")
        warmup_steps = train_args.lr_warmup_fraction * max_steps
    # Linear warmup followed by cosine annealing
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(max_steps - warmup_steps))
    if warmup_steps <= 0:
        return scheduler2
    # Note: The first LR (for `step=0`) is being used. Must not be 0
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (step + 1) / warmup_steps)
    if warmup_steps >= max_steps:
        return scheduler1
    else:
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, [scheduler1, scheduler2], milestones=[warmup_steps],
        )


def get_dataloaders(
    data: DataModule,
    tokenizer: Tokenizer,
    head_model: str,
    train: TrainArgs,
    eval: EvalArgs,
    fabric: Optional[L.Fabric] = None,
) -> Tuple[DataLoader, DataLoader]:
    data.connect(
        tokenizer=tokenizer,
        batch_size=train.micro_batch_size,
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
    if fabric is not None:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(
            train_dataloader, val_dataloader,
        )
    return train_dataloader, val_dataloader


def validate_args(train: TrainArgs, eval: EvalArgs) -> None:
    issues = []
    unsupported = [(train, ["max_tokens", "max_norm", "tie_embeddings"])]
    for args, names in unsupported:
        for name in names:
            if getattr(args, name) is not None:
                issues.append(f"{__file__} doesn't support the {name!r} argument. This is set in {args}")
    required = [(train, ["epochs"]), (eval, ["max_new_tokens"])]
    for args, names in required:
        for name in names:
            if getattr(args, name) is None:
                issues.append(f"{__file__} requires the {name!r} argument. This is set in {args}")
    if not train.epochs and not train.max_steps:
        issues.append(f"{__file__} requires either epochs or max_steps to be set. This is set in {train}")
    if issues:
        raise ValueError("\n".join(issues))


def save_model_checkpoint(
    fabric: L.Fabric,
    model: GPTAndHeadModel,
    file_dir: Path,
) -> None:
    file_dir.mkdir(parents=True, exist_ok=True)
    file_path = file_dir / LIT_MODEL_FNAME
    print_message(
        f"\nSaving model weights to {str(file_path)!r}",
        fabric,
    )
    fabric.save(file_path, state={"model": model.gpt_model})
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
                out_dir, name=name, **kwargs,
            )
        if logger_name == "wandb":
            from lightning.pytorch.loggers.wandb import WandbLogger

            if log_args is None:
                log_args = dict()
            project = log_args.get("project", name)
            run = log_args.get("run", os.environ.get("WANDB_RUN_NAME"))
            group = log_args.get("group", os.environ.get("WANDB_RUN_GROUP"))
            return WandbLogger(
                project=project, name=run, group=group, resume=resume, **kwargs,
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
        raise ValueError(f"`logger_name={logger_name}` is not a valid option. Choose from 'csv', 'tensorboard', 'wandb', 'mlflow'.")


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


def flush_io_streams():
    sys.stdout.flush()
    sys.stderr.flush()


def debug_sum_gradient_norms(model: torch.nn.Module) -> float:
    sum = 0
    num = 0
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            sum = param.grad.norm() + sum
            num += 1
    return sum.item() / num


def debug_random_trainable_param_names(
    model: torch.nn.Module,
    num_to_select: int,
    num_trainable_params: int,
) -> List[str]:
    positions = set(random.sample(range(num_trainable_params), num_to_select))
    return [
        name
        for i, name in enumerate(
            name
            for name, param in model.named_parameters()
            if param.requires_grad
        )
        if i in positions
    ]


def debug_clone_selected_params(
    model: torch.nn.Module,
    names: List[str],
) -> Dict[str, torch.Tensor]:
    state_dict = model.state_dict()
    return {
        name: state_dict[name].detach().clone()
        for name in names
    }


def debug_compare_selected_params(
    model: torch.nn.Module,
    orig_params: Dict[str, torch.Tensor],
) -> int:
    state_dict = model.state_dict()
    return sum(
        not torch.allclose(state_dict[name], value)
        for name, value in orig_params.items()
    )
