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

import math
import os
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, Literal, Optional, Union, Any

import lightning as L
import torch
from lightning.fabric.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torchmetrics import RunningMean

from keys_values.utils import flush_io_streams
from litgpt.args import TrainArgs
from litgpt.data import DataModule
from litgpt.config import Config
from litgpt.prompts import save_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    CycleIterator,
    auto_download_checkpoint,
    check_nvlink_connectivity,
    check_valid_checkpoint_dir,
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
from keys_values.attention_utils import (
    DEFAULT_TMP_ARRAY_LIMIT_GB,
    SDPA_KERNELS_BEST_ORDERING,
)
from keys_values.data import LongBenchV2, INPUT_IDS_NAME
from keys_values.finetune.args import (
    EvalArgs,
    GradientArgs,
    KVCacheArgs,
    OptimizerArgs,
)
from keys_values.finetune.batch_transform import (
    BatchTransformFactory,
    BatchTransform,
)
from keys_values.finetune.utils import (
    LIT_MODEL_FNAME,
    HEAD_MODEL_FNAME,
    print_but_limit_size,
    get_lr_scheduler,
    get_dataloaders,
    validate_args,
    save_model_checkpoint,
    choose_logger,
    adapt_requires_grad,
    print_with_rank_and_timestamp,
    print_message,
    check_kv_cache,
)
from keys_values.generate.base import generate
from keys_values.gpu_memory import RecordGPUMemory
from keys_values.head_model import HeadModel, CrossEntropyOnLogits
from keys_values.head_model_factory import HeadModelFactory
from keys_values.kvcache.factory import (
    KVCacheFactory,
    deallocate_kv_cache_buffers_of_model,
    cleanup_cache_kwargs,
    split_name,
)
from keys_values.kvcache.gradient.main import (
    LongContextGradientModel,
    NaiveGPTAndHeadModel,
)
from keys_values.kvcache.utils import (
    fabric_precision_to_dtype,
    log_memory_all_devices,
    message_memory_all_devices,
    VerbosityLevels,
)
from keys_values.long_context import (
    GPTAndHeadModel,
    LongContextInferenceModel,
)
from keys_values.model import GPT
from keys_values.optimize.grad_accumulate import CPUOffloadAccumulateGradients
from keys_values.parser_config import save_hyperparameters
from keys_values.pos_encoding import position_encoding_factory


DEFAULT_OUT_DIR = "out/finetune/longcontext_full"


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
        cache_length=16384,
        chunk_size=1024,
        cache_kwargs={
            "replay_log_blocksize": 1024,
            "allocate_buffers": False,
            "max_num_ranges": 4,
        },
        randomize_chunk_sizes=False,
        allocate_buffers=False,
    ),
    grad: GradientArgs = GradientArgs(
        layers_per_cell=1,
        chunks_per_cell_multiplier=1.0,
        single_tokens_for_targets=False,
        use_new_cache=False,
        max_match_trials_pack_arg=8,
        layer_checkpoint_chunk_size=None,
    ),
    head_model: str = CrossEntropyOnLogits.NAME,
    head_model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: Optional[str] = None,
    attention_forward_temp_size_gb: Optional[float] = None,
    attention_backward_temp_size_gb: Optional[float] = None,
    yarn_rope: bool = True,
    record_gpu_memory_snapshots: Optional[int] = None,
    record_gpu_memory_kind: int = 0,
    record_gpu_memory_period: int = 0,
    generate_with_eval: bool = False,
    profile_grad_times: int = 0,
    profile_parts: Optional[str] = None,
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
        eval: Evaluation-related arguments. See
            ``keys_values.finetune.args.EvalArgs`` for details.
        optimizer: Selects optimizer and its parameters, see
            ``keys_values.finetune.args.OptimizerArgs`` for details. Defaults to
            "AdamW" with default parameters.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
        kv_cache: Configuration for the KV caches. See
            ``keys_values.finetune.args.KVCacheArgs`` for details. Defaults to
            H2O with PyTorch 8-bit quantization. Make sure to adjust
            `kv_cache.cache_length`.
        grad: Configuration for gradient computation, see
            ``keys_values.finetune.args.GradientArgs`` for details. Adjust
            `grad.layers_per_cell` and `grad.chunks_per_cell_multiplier` given
            your GPU memory (defaults are smallest sensible values).
        head_model: Name of the head model to use, see
            :class:`HeadModelFactory`. Defaults to "next_token_prediction"
        head_model_kwargs: Extra keyword arguments to pass to the head model
            factory.
        verbose: Verbosity level for logging outputs.
        attention_forward_temp_size_gb: Size of GPU memory buffers (in GB) used
            in naive SDPA. At present, naive SDPA is used with KV caches which
            require attention weights (e.g., H2O).
        attention_backward_temp_size_gb: Size of GPU memory buffers (in GB) used
            in naive SDPA during backward computations. At present, naive SDPA
            is used in backward if `grad.use_new_cache == False`.
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
        profile_grad_times: If given, we profile complete gradient computation
            for this many steps, then stop. Results are written to CSV file.
        profile_parts: If given, we use `cProfile` to profile the first forward
            (if "forward") or first backward (if "backward") pass. Results are
            printed, then the program stops.

    """
    checkpoint_dir = auto_download_checkpoint(
        model_name=checkpoint_dir, access_token=access_token
    )
    pprint(locals())
    data = LongBenchV2() if data is None else data
    if isinstance(data, LongBenchV2) and data.metadata_dir is None:
        data.metadata_dir = str(out_dir / "data")
        print(f"Setting LongBenchV2.metadata_dir to {data.metadata_dir}")
    out_dir = init_out_dir(out_dir)
    data.metadata_dir = str(init_out_dir(Path(data.metadata_dir)))
    if head_model_kwargs is None:
        head_model_kwargs = dict()
    devices = parse_devices(devices)
    if eval.initial_validation is None:
        # Run initial evaluation in multi-device setup, but not with a
        # single device
        eval.initial_validation = devices > 1
    if optimizer is None:
        optimizer = OptimizerArgs(name="AdamW")
        print(
            "Choosing optimizer AdamW with default learning rate. We recommend to at least tune optimizer.learning_rate"
        )
    else:
        print(str(optimizer))
    if profile_parts is not None and profile_parts not in ("forward", "backward"):
        raise ValueError("profile_parts: Must be 'forward' or 'backward'")
    # Legacy arguments
    if verbose is None:
        if kv_cache.verbose is not None:
            verbose = kv_cache.verbose
            kv_cache.verbose = None
        else:
            verbose = VerbosityLevels.SOME.value
    verbose = VerbosityLevels(verbose)
    if attention_forward_temp_size_gb is None:
        if kv_cache.attention_forward_temp_size_gb is not None:
            attention_forward_temp_size_gb = kv_cache.attention_forward_temp_size_gb
            kv_cache.attention_forward_temp_size_gb = None
        else:
            attention_forward_temp_size_gb = 4
    if attention_backward_temp_size_gb is None:
        if kv_cache.attention_backward_temp_size_gb is not None:
            attention_backward_temp_size_gb = kv_cache.attention_backward_temp_size_gb
            kv_cache.attention_backward_temp_size_gb = None
        else:
            attention_backward_temp_size_gb = 2

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
        # strategy = FSDPStrategy(
        #    auto_wrap_policy={Block},
        #    activation_checkpointing_policy={Block},
        #    state_dict_type="full",
        #    limit_all_gathers=True,
        #    cpu_offload=False,
        # )
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
        grad=grad,
        num_nodes=num_nodes,
        head_model_name=head_model,
        head_model_kwargs=head_model_kwargs,
        verbose=verbose,
        attention_forward_temp_size_gb=attention_forward_temp_size_gb,
        attention_backward_temp_size_gb=attention_backward_temp_size_gb,
        yarn_rope=yarn_rope,
        record_gpu_memory_snapshots=record_gpu_memory_snapshots,
        record_gpu_memory_kind=record_gpu_memory_kind,
        record_gpu_memory_period=record_gpu_memory_period,
        generate_with_eval=generate_with_eval,
        profile_grad_times=profile_grad_times,
        profile_parts=profile_parts,
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
    grad: GradientArgs,
    num_nodes: int,
    head_model_name: str,
    head_model_kwargs: Dict[str, Any],
    verbose: VerbosityLevels,
    attention_forward_temp_size_gb: float,
    attention_backward_temp_size_gb: float,
    yarn_rope: bool,
    record_gpu_memory_snapshots: Optional[RecordGPUMemory],
    record_gpu_memory_kind: int,
    record_gpu_memory_period: int,
    generate_with_eval: bool,
    profile_grad_times: int,
    profile_parts: Optional[str],
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
    steps_per_epoch = len(train_dataloader) // train.gradient_accumulation_iters(
        devices, num_nodes
    )
    lr_max_steps = min(
        train.epochs * steps_per_epoch, (train.max_steps or float("inf"))
    )
    print(f"Number of optimizer steps per epoch: {lr_max_steps}")
    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        # Order of preference for SDPA kernels
        limit_gb = attention_forward_temp_size_gb
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
            sdpa_kernels=SDPA_KERNELS_BEST_ORDERING,
            tmp_array_limit_gb=tmp_array_limit_forward,
            pos_encoding=position_encoding_factory(config, do_yarn=yarn_rope),
        )
        if "sdpa_kernels" not in kv_cache.cache_kwargs:
            kv_cache.cache_kwargs["sdpa_kernels"] = SDPA_KERNELS_BEST_ORDERING
        kv_cache.cache_kwargs["tmp_array_limit_gb"] = tmp_array_limit_forward
        kv_cache.cache_kwargs["pos_encoding"] = mha_kwargs["pos_encoding"]
        gpt_model = GPT(config, **mha_kwargs)
        head_model = HeadModelFactory.create(
            name=head_model_name,
            config=config,
            data=data,
            **head_model_kwargs,
        )
        adapt_requires_grad(gpt_model, head_model)
        batch_size = train.micro_batch_size
        if eval.micro_batch_size is not None:
            batch_size = max(batch_size, eval.micro_batch_size)
        model = wrap_gpt_model(
            gpt_model=gpt_model,
            head_model=head_model,
            kv_cache=kv_cache,
            grad=grad,
            verbose=verbose,
            attention_backward_temp_size_gb=attention_backward_temp_size_gb,
            max_batch_size=batch_size,
            dtype=fabric_precision_to_dtype(fabric._precision.precision),
            profile_grad_times=profile_grad_times > 0,
            profile_parts=profile_parts,
            fabric=fabric,
        )

    num_trainable_params = num_parameters(model, requires_grad=True)
    print_message(
        f"\nNumber of trainable parameters: {num_trainable_params:,}",
        fabric,
    )

    model = fabric.setup(model)

    optimizer = instantiate_torch_optimizer(
        optimizer.name,
        model.parameters(),
        **optimizer.optimizer_kwargs(),
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
        thresh = grad.max_match_trials_pack_arg
        name = "new" if grad.use_new_cache else "old"
        profile_grad_params = {
            "path": Path(out_dir) / f"profile_grad_times_{name}_{thresh}.csv",
            "use_new_cache": grad.use_new_cache,
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
        generate_with_eval=generate_with_eval,
        profile_grad_params=profile_grad_params,
    )
    training_time = time.perf_counter() - train_time
    output = create_finetuning_performance_report(
        training_time, token_counts, fabric.device.type
    )
    print_message(output, fabric)

    # Final evaluation
    if eval.final_validation:
        print_with_rank_and_timestamp(
            "Starting validation evaluations.", fabric.global_rank
        )
        print_message("\nFinal validation evaluation ...", fabric)
        if generate_with_eval:
            generate_example_kwargs = dict(
                tokenizer=tokenizer,
                data=data,
            )
        else:
            generate_example_kwargs = None
        metrics = validate_and_all_reduce(
            model=model,
            val_dataloader=val_dataloader,
            eval=dataclasses.replace(eval, max_iters=len(val_dataloader)),
            batch_transform=batch_transform,
            log_metrics=False,
            fabric=fabric,
            generate_example_kwargs=generate_example_kwargs,
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


def wrap_gpt_model(
    gpt_model: GPT,
    head_model: HeadModel,
    kv_cache: KVCacheArgs,
    grad: Optional[GradientArgs],
    verbose: VerbosityLevels,
    attention_backward_temp_size_gb: Optional[float],
    max_batch_size: int,
    dtype: torch.dtype,
    profile_grad_times: bool = False,
    profile_parts: Optional[str] = None,
    cpu_offload_device: Optional[torch.device] = None,
    offload_num_devices: int = 1,
    fabric: Optional[L.Fabric] = None,
) -> Union[LongContextGradientModel, LongContextInferenceModel]:
    model_for_training = grad is not None
    print_message(
        "Assigning KV caches to layers of model:\n"
        f"name:           {kv_cache.name}\n"
        f"cache_length:   {kv_cache.cache_length}\n"
        f"max_batch_size: {max_batch_size}",
        fabric,
    )
    gpt_model.clear_kv_caches()
    cache_kwargs = dict() if kv_cache.cache_kwargs is None else kv_cache.cache_kwargs
    cache_kwargs = dict(
        cache_kwargs,
        max_chunk_size=kv_cache.maximum_chunk_size(),
    )
    cache_kwargs = cleanup_cache_kwargs(
        split_name(kv_cache.name)[0],
        cache_kwargs,
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
    multiplier = 1.0 if grad is None else grad.chunks_per_cell_multiplier
    common_kwargs = dict(
        gpt_model=gpt_model,
        head_model=head_model,
        chunk_size=kv_cache.chunk_size,
        randomize_chunk_sizes=kv_cache.randomize_chunk_sizes,
        chunks_per_cell_multiplier=multiplier,
        verbose=verbose,
        tmp_array_limit_gb=tmp_array_limit_gb,
    )
    if model_for_training:
        # Temp array size limit can be different for backward and forward
        limit_gb = attention_backward_temp_size_gb
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
            "use_new_cache": grad.use_new_cache,
        }
        if grad.max_match_trials_pack_arg is not None:
            autograd_hooks_kwargs = dict(
                max_match_trials_pack_arg=grad.max_match_trials_pack_arg,
            )
        else:
            autograd_hooks_kwargs = None
        if cpu_offload_device is not None:
            common_kwargs["head_model"] = head_model.to(device=cpu_offload_device)
            offload_grad_accum = CPUOffloadAccumulateGradients(
                group=list(range(offload_num_devices)),
                fabric=fabric,
            )
        else:
            offload_grad_accum = None
        layer_checkpoint_chunk_size = grad.layer_checkpoint_chunk_size
        if layer_checkpoint_chunk_size is None:
            # Default value for chunk size if not given
            layer_checkpoint_chunk_size = kv_cache.cache_length
            add_msg = " (default)"
        else:
            add_msg = ""
        if kv_cache.qname != "default":
            print_message(
                f"Using layer_checkpoint_chunk_size = {layer_checkpoint_chunk_size}"
                + add_msg,
                fabric,
            )
        model = LongContextGradientModel(
            **common_kwargs,
            layers_per_cell=grad.layers_per_cell,
            single_tokens_for_targets=grad.single_tokens_for_targets,
            qname=kv_cache.qname,
            cache_kwargs=cache_kwargs,
            train_cache_kwargs=train_cache_kwargs,
            backward_tmp_array_limit_gb=backward_tmp_array_limit_gb,
            autograd_hooks_kwargs=autograd_hooks_kwargs,
            profile_steps=profile_grad_times,
            offload_device=cpu_offload_device,
            offload_grad_accum=offload_grad_accum,
            layer_checkpoint_chunk_size=layer_checkpoint_chunk_size,
            debug_profile_forward=profile_parts == "forward",
            debug_profile_backward=profile_parts == "backward",
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
    generate_with_eval: bool,
    profile_grad_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]
    tokenizer = Tokenizer(checkpoint_dir)

    # Initial evaluation
    token_counts = {
        "raw_tokens": torch.tensor(0, device=fabric.device, dtype=torch.long),
        "raw_tokens_plus_prompt_template": torch.tensor(
            0, device=fabric.device, dtype=torch.long
        ),
        "raw_tokens_plus_prompt_template_and_padding": torch.tensor(
            0, device=fabric.device, dtype=torch.long
        ),
    }
    val_loss = "n/a"
    if eval.initial_validation:
        print_with_rank_and_timestamp(
            "Starting validation evaluations.", fabric.global_rank
        )
        print_message("\nInitial validation evaluation ...", fabric)
        if generate_with_eval:
            generate_example_kwargs = dict(
                tokenizer=tokenizer,
                data=data,
            )
        else:
            generate_example_kwargs = None
        metrics = validate_and_all_reduce(
            model=model,
            val_dataloader=val_dataloader,
            eval=dataclasses.replace(eval, max_iters=len(val_dataloader)),
            batch_transform=batch_transform,
            fabric=fabric,
            generate_example_kwargs=generate_example_kwargs,
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

    running_loss = RunningMean(
        window=train.gradient_accumulation_iters(devices, num_nodes),
        sync_on_compute=False,
    ).to(fabric.device)
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
            run_no = state["iter_num"] - 1
            if record_gpu_memory_period >= 1:
                run_no = run_no % record_gpu_memory_period
            name = (
                "snapshot.pickle"
                if record_gpu_memory_kind == 0
                else "snapshot_initial.pickle"
            )
            path = out_dir / "gpu_memory_snapshots" / f"iteration{run_no}" / name
            record_gpu_memory_snapshots = RecordGPUMemory(
                path=str(path),
                max_entries=record_gpu_memory_snapshots.max_entries,
            )
            record_gpu_memory_snapshots.start_recording()

        is_accumulating = (
            state["iter_num"] % train.gradient_accumulation_iters(devices, num_nodes)
            != 0
        )
        print_with_rank_and_timestamp(
            "Starting gradient computation.", fabric.global_rank
        )
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            loss = model(
                input_ids=batch[INPUT_IDS_NAME],
                targets=batch["targets"],
                scale_factor=1.0
                / train.gradient_accumulation_iters(devices, num_nodes),
                record_gpu_memory_snapshots=record_gpu_memory_snapshots,
                record_gpu_memory_kind=(
                    record_gpu_memory_kind
                    if record_gpu_memory_snapshots is not None
                    else None
                ),
            )
            fabric.backward(loss)

        running_loss.update(loss.detach())
        flush_io_streams()
        if profile_grad_params is not None:
            records = model.profile_records()
            skip_names = ("path", "profile_grad_times")
            fixed_col_names = [
                name for name in profile_grad_params.keys() if name not in skip_names
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
            print_with_rank_and_timestamp(
                "Waiting for optimizer to update.", fabric.global_rank
            )
            optimizer.step()
            print_message("Optimizer update done.", fabric)
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            state["step_count"] += 1

        del loss
        torch.cuda.empty_cache()
        print_message(
            f"\nGPU memory at training step {state['iter_num'] - 1}:\n"
            + message_memory_all_devices()
            + "\n",
            fabric,
        )

        token_counts["raw_tokens"] += batch["token_counts"]["raw"].sum().item()
        token_counts["raw_tokens_plus_prompt_template"] += (
            batch["token_counts"]["raw_plus_prompt_template"].sum().item()
        )
        num_tokens = batch[INPUT_IDS_NAME].numel()
        token_counts["raw_tokens_plus_prompt_template_and_padding"] += num_tokens

        if state["iter_num"] % train.log_interval == 0:
            loss = (
                running_loss.compute().item()
            )  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "tokens": token_counts["raw_tokens_plus_prompt_template"],
                "total_tokens": token_counts["raw_tokens_plus_prompt_template"]
                * fabric.world_size,
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
            print_with_rank_and_timestamp(
                "Starting validation evaluations.", fabric.global_rank
            )
            print_message("\nPeriodic validation evaluation ...", fabric)
            if generate_with_eval:
                generate_example_kwargs = dict(
                    tokenizer=tokenizer,
                    data=data,
                )
            else:
                generate_example_kwargs = None
            metrics = validate_and_all_reduce(
                model=model,
                val_dataloader=val_dataloader,
                eval=eval,
                batch_transform=batch_transform,
                generate_example_kwargs=generate_example_kwargs,
                log_metrics=False,
                fabric=fabric,
            )
            fabric.log_dict(metrics, step=state["iter_num"])
            print_with_rank_and_timestamp(
                "Finished validation evaluations.", fabric.global_rank
            )
            val_loss = f"{metrics['val_loss']:.3f}"
            print_message(
                f"Epoch {train_iterator.epoch} | iter {state['iter_num']} | val loss: {val_loss} | val ppl: {metrics['val_ppl']:.3f} | val_time_in_ms: {metrics['val_time_in_ms']:.3f}",
                fabric,
            )
            flush_io_streams()
            fabric.barrier()

        if (
            train.save_interval is not None
            and not is_accumulating
            and state["step_count"] % train.save_interval == 0
        ):
            checkpoint_file = (
                out_dir / f"step-{state['step_count']:06d}" / LIT_MODEL_FNAME
            )
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
            val_time,
            device=fabric.device,
            dtype=torch.float32,
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
            model=model,
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
