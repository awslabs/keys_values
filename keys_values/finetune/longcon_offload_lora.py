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
import gc
import os
from pathlib import Path
from pprint import pprint
import time
from typing import Dict, Literal, Optional, Any, Union

import lightning as L
from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.utilities import ThroughputMonitor
import torch
from torch.utils.data import DataLoader
from torchmetrics import RunningMean

from litgpt.args import TrainArgs
from litgpt.data import DataModule
from litgpt.lora import Config, mark_only_lora_as_trainable
from litgpt.prompts import save_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    CycleIterator,
    auto_download_checkpoint,
    check_valid_checkpoint_dir,
    copy_config_files,
    create_finetuning_performance_report,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    load_checkpoint,
    num_parameters,
    check_nvlink_connectivity,
    parse_devices,
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
    LoRAARgs,
    OptimizerArgs,
)
from keys_values.finetune.batch_transform import (
    BatchTransformFactory,
    BatchTransform,
)
from keys_values.finetune.longcontext_full import (
    wrap_gpt_model,
    validate,
    validate_and_all_reduce,
)
from keys_values.finetune.longcontext_lora import save_lora_checkpoint
from keys_values.finetune.utils import (
    HEAD_MODEL_FNAME,
    LIT_MODEL_FNAME,
    get_lr_scheduler,
    get_dataloaders,
    validate_args,
    choose_logger,
    print_with_rank_and_timestamp,
    print_message,
    check_kv_cache,
    adapt_requires_grad,
    create_optimizer,
)
from keys_values.gpu_memory import RecordGPUMemory
from keys_values.head_model import CrossEntropyOnLogits
from keys_values.head_model_factory import HeadModelFactory
from keys_values.kvcache.factory import deallocate_kv_cache_buffers_of_model
from keys_values.kvcache.utils import (
    fabric_precision_to_dtype,
    log_memory_all_devices,
    message_memory_all_devices,
    VerbosityLevels,
)
from keys_values.lora import GPT
from keys_values.optimize.model_factory import BlockComponentName
from keys_values.parser_config import save_hyperparameters
from keys_values.pos_encoding import position_encoding_factory
from keys_values.utils import flush_io_streams


DEFAULT_OUT_DIR = "out/finetune/longcontext_lora"


def setup(
    checkpoint_dir: Path,
    out_dir: Path = Path(DEFAULT_OUT_DIR),
    precision: Optional[str] = None,
    devices: Union[int, str] = 1,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=2,
        micro_batch_size=2,
        lr_warmup_steps=None,
        lr_warmup_fraction=0.15,
        epochs=5,
        max_seq_length=None,
    ),
    lora: LoRAARgs = LoRAARgs(
        r=8,
        alpha=16,
        dropout=0.05,
        query=True,
        key=False,
        value=True,
        projection=False,
        mlp=False,
        head=False,
    ),
    eval: EvalArgs = EvalArgs(
        interval=100,
        max_new_tokens=100,
        max_iters=100,
        initial_validation=None,
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
    """Finetune a model using the LoRA method, with CPU offloading

    Makes use of devices `range(devices)`. If `devices > 1`, we run distributed
    data parallel (DDP). We use `lightning.Fabric` here as well, but just to
    launch the processes. Model and optimizer are not wrapped, and we only make
    use of `fabric.all_reduce`.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to load for finetuning.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        devices: Number of GPU devices (ranks) to be used. We use ranks
            `range(devices)`.
        data: Data-related arguments. If not provided, the default is
            ``keys_values.data.LongBenchV2``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
            Note: We modified the defaults from `train.lr_warmup_steps=100` to
            `train.lr_warmup_fraction=0.15`, so the linear warm-up is the first
            15% of all steps.
        lora: Arguments for LoRA extension of model, see
            ``keys_values.finetune.args.LoRAArgs`` for details. Adjust the LoRA
            rank with `lora.r`.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
            Note: We modify the default for `eval.initial_validation` (whether
            evaluation is run before training starts). It is set if
            `devices > 1`, but off if `devices == 1`.
        optimizer: Selects optimizer and its parameters, see
            ``keys_values.finetune.args.OptimizerArgs`` for details. Defaults to
            "AdamW" with default parameters.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
        kv_cache: Configuration for the KV caches. Defaults to H2O with PyTorch
            8-bit quantization and cache length 8192. Should be increased if GPU
            memory is sufficient. Also consider increasing layers per cell.
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
            - 2: Special case
            - 3: One snapshot file during initial validation
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
    if not torch.cuda.is_available():
        raise ValueError("CUDA not available")
    checkpoint_dir = auto_download_checkpoint(
        model_name=checkpoint_dir,
        access_token=access_token,
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
    if not (1 <= devices <= torch.cuda.device_count()):
        raise ValueError(
            f"devices = {devices}, must be in [1, {torch.cuda.device_count()}]"
        )
    if eval.initial_validation is None:
        # Run initial evaluation in multi-device setup, but not with a
        # single device
        eval.initial_validation = devices > 1
    if optimizer is None:
        optimizer = OptimizerArgs(name="AdamW")
        print(
            "Choosing optimizer AdamW with default learning rate. We highly recommend to at least tune optimizer.learning_rate"
        )
    else:
        print(str(optimizer))
    global_batch_size = train.micro_batch_size * devices
    if train.global_batch_size != global_batch_size:
        print(f"train.global_batch_size not supported, set to {global_batch_size}")
        train.global_batch_size = global_batch_size
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
    config = Config.from_file(
        checkpoint_dir / "model_config.yaml",
        lora_r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        lora_query=lora.query,
        lora_key=lora.key,
        lora_value=lora.value,
        lora_projection=lora.projection,
        lora_mlp=lora.mlp,
        lora_head=lora.head,
    )

    precision = precision or get_default_supported_precision(training=True)
    # Currently not used:
    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"finetune-{config.name}",
        use_fabric=True,
        log_interval=train.log_interval,
    )

    if devices > 1:
        strategy = DDPStrategy(static_graph=True, broadcast_buffers=False)
    else:
        strategy = "auto"

    fabric = L.Fabric(
        devices=devices,
        num_nodes=1,
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
        devices,
        1,
    )
    lr_max_steps = min(
        train.epochs * steps_per_epoch, (train.max_steps or float("inf"))
    )
    print_message(
        f"Number of optimizer steps per epoch: {lr_max_steps}",
        fabric,
    )

    fabric.seed_everything(seed)
    cpu_offload_device = torch.device("cuda", fabric.local_rank)
    optim_device = torch.device("cpu")

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
        # We create the GPT model on the device, then copy. This is faster
        print_message("Creating model on CPU", fabric)
        with torch.device(cpu_offload_device):
            gpt_model = GPT(config, **mha_kwargs)
            head_model = HeadModelFactory.create(
                name=head_model_name,
                config=config,
                data=data,
                **head_model_kwargs,
            )
        gpt_model = gpt_model.to(optim_device)
        mark_only_lora_as_trainable(gpt_model)
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
            cpu_offload_device=cpu_offload_device,
            offload_num_devices=devices,
            fabric=fabric,
        )

    num_trainable_params = num_parameters(model, requires_grad=True)
    print_message(
        f"\nNumber of trainable parameters: {num_trainable_params:,}",
        fabric,
    )
    print_message(
        f"Number of non-trainable parameters: {num_parameters(model, requires_grad=False):,}",
        fabric,
    )

    # We use a optimizer on CPU for all parameters of `gpt_model`. If
    # `head_model` has parameters, we use another optimizer on GPU for them.
    # Note: We do not wrap model or optimizer with `fabric`, since our CPU
    # offloading deviates from their strategies.
    gpt_param_prefixes = tuple(
        BlockComponentName.h(layer_idx) for layer_idx in range(config.n_layer)
    ) + (
        BlockComponentName.wte(),
        BlockComponentName.ln_f(),
    )
    if head_model.needs_logits():
        gpt_param_prefixes += (BlockComponentName.lm_head(),)
    cpu_optimizer = create_optimizer(
        optim_args=optimizer,
        gpt_model=gpt_model,
        gpt_param_prefixes=gpt_param_prefixes,
    )
    cpu_scheduler = get_lr_scheduler(
        cpu_optimizer,
        train_args=train,
        max_steps=lr_max_steps,
    )
    state = {
        "model": model,
        "cpu_optimizer": cpu_optimizer,
        "cpu_scheduler": cpu_scheduler,
        "iter_num": 0,
        "step_count": 0,
    }
    head_model_params = list(head_model.parameters())
    if head_model_params:
        state["gpu_optimizer"] = instantiate_torch_optimizer(
            optimizer.name,
            head_model_params,
            **optimizer.optimizer_kwargs(),
        )
        state["gpu_scheduler"] = get_lr_scheduler(
            state["gpu_optimizer"],
            train_args=train,
            max_steps=lr_max_steps,
        )

    # strict=False because missing keys due to LoRA weights not contained in state dict
    print_message(f"Loading model checkpoint: {checkpoint_dir}", fabric)
    file_path = checkpoint_dir / LIT_MODEL_FNAME
    load_checkpoint(fabric, model.gpt_model, file_path, strict=False)
    # If there are head model weights, load them as well. Otherwise, we use
    # random initialization (or the head model may not have weights)
    file_path = checkpoint_dir / HEAD_MODEL_FNAME
    if file_path.exists():
        load_checkpoint(fabric, model.head_model, file_path, strict=True)

    if profile_grad_times > 0 and fabric.global_rank == 0:
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
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        train=train,
        eval=eval,
        data=data,
        record_gpu_memory_snapshots=record_gpu_memory_snapshots,
        record_gpu_memory_kind=record_gpu_memory_kind,
        record_gpu_memory_period=record_gpu_memory_period,
        generate_with_eval=generate_with_eval,
        profile_grad_params=profile_grad_params,
    )
    training_time = time.perf_counter() - train_time
    output = create_finetuning_performance_report(
        training_time,
        token_counts,
        fabric.device.type,
    )
    print_message(output, fabric)

    # Final evaluation
    if eval.final_validation:
        print_with_rank_and_timestamp(
            "Starting validation evaluations.", fabric.global_rank
        )
        print_message(
            f"\nFinal validation evaluation (batch_size = {val_dataloader.batch_size}) ...",
            fabric,
        )
        if generate_with_eval:
            generate_example_kwargs = dict(
                tokenizer=tokenizer,
                data=data,
            )
        else:
            generate_example_kwargs = None
        valid_model = model.copy_model_for_evaluation()
        metrics = validate_and_all_reduce(
            model=valid_model,
            val_dataloader=val_dataloader,
            eval=dataclasses.replace(eval, max_iters=len(val_dataloader)),
            batch_transform=batch_transform,
            log_metrics=False,
            generate_example_kwargs=generate_example_kwargs,
            fabric=fabric,
        )
        fabric.log_dict(metrics, step=state["iter_num"])
        print_message(
            f"Final evaluation | val loss: {metrics['val_loss']:.3f} | val ppl: {metrics['val_ppl']:.3f} | val_time_in_ms: {metrics['val_time_in_ms']:.3f}",
            fabric,
        )
        deallocate_kv_cache_buffers_of_model(valid_model.gpt_model)
        del valid_model
        flush_io_streams()

    # Save the final checkpoint at the end of training
    save_dir = out_dir / "final"
    save_lora_checkpoint(fabric, model, save_dir)
    if fabric.global_rank == 0:
        # Copy checkpoint files from original checkpoint dir
        copy_config_files(checkpoint_dir, save_dir)
        save_hyperparameters(setup, save_dir)
        if hasattr(data, "prompt_style"):
            save_prompt_style(data.prompt_style, save_dir)


def fit(
    fabric: L.Fabric,
    state: Dict[str, Any],
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    batch_transform: BatchTransform,
    devices: int,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    data: DataModule,
    record_gpu_memory_snapshots: Optional[RecordGPUMemory],
    record_gpu_memory_kind: int,
    record_gpu_memory_period: int,
    generate_with_eval: bool,
    profile_grad_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    model = state["model"]
    cpu_optimizer = state["cpu_optimizer"]
    cpu_scheduler = state["cpu_scheduler"]
    gpu_optimizer = state.get("gpu_optimizer")
    gpu_scheduler = state.get("gpu_scheduler")
    tokenizer = Tokenizer(checkpoint_dir)
    optim_device = torch.device("cpu")

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

    if record_gpu_memory_kind == 3:
        path = out_dir / "gpu_memory_snapshots" / "snapshot_validation.pickle"
        record_gpu_memory_snapshots = RecordGPUMemory(
            path=str(path),
            max_entries=record_gpu_memory_snapshots.max_entries,
            verbose=VerbosityLevels.MORE,
        )
        record_gpu_memory_snapshots.start_recording()

    val_loss = "n/a"
    valid_model = model.copy_model_for_evaluation()
    if record_gpu_memory_kind == 3:
        valid_model.set_record_gpu_memory(
            record_gpu_memory_snapshots,
            record_gpu_memory_kind,
        )
    if eval.initial_validation:
        print_with_rank_and_timestamp(
            "Starting validation evaluations.",
            fabric.global_rank,
        )
        print_message(
            f"\nInitial validation evaluation  (batch_size = {val_dataloader.batch_size}) ...",
            fabric,
        )
        if generate_with_eval:
            generate_example_kwargs = dict(
                tokenizer=tokenizer,
                data=data,
            )
        else:
            generate_example_kwargs = None
        metrics = validate_and_all_reduce(
            model=valid_model,
            val_dataloader=val_dataloader,
            eval=dataclasses.replace(eval, max_iters=len(val_dataloader)),
            batch_transform=batch_transform,
            generate_example_kwargs=generate_example_kwargs,
            fabric=fabric,
        )
        val_loss = f"{metrics['val_loss']:.3f}"
        print_message(
            f"Initial evaluation | val loss: {val_loss} | val ppl: {metrics['val_ppl']:.3f} | val_time_in_ms: {metrics['val_time_in_ms']:.3f}",
            fabric,
        )
    else:
        # Note: Even if `generate_with_eval == True`, we don't generate here
        print_message("Verifying settings ...", fabric)
        with torch.no_grad():
            validate(
                valid_model,
                val_dataloader,
                dataclasses.replace(eval, max_iters=1),
                batch_transform,
            )  # sanity check
    deallocate_kv_cache_buffers_of_model(valid_model.gpt_model)
    del valid_model
    flush_io_streams()

    if record_gpu_memory_kind == 3:
        if record_gpu_memory_snapshots.is_recording:
            record_gpu_memory_snapshots.store_current_snapshot()
            record_gpu_memory_snapshots.stop_recording()
        # Switch off from here on
        record_gpu_memory_snapshots = None
        record_gpu_memory_kind = 0

    max_steps = train.max_steps or float("inf")
    train_iterator = CycleIterator(train_dataloader)
    throughput = ThroughputMonitor(fabric, window_size=50)
    running_loss = RunningMean(
        window=train.gradient_accumulation_iters(devices, 1),
        sync_on_compute=False,
    ).to(optim_device)
    total_lengths = 0
    gc.collect()
    torch.cuda.empty_cache()
    print_message(
        "\nGPU memory before training starts:\n" + message_memory_all_devices(),
        fabric,
    )
    total_t0 = time.perf_counter()

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
            if record_gpu_memory_kind == 0:
                name = "snapshot.pickle"
                path = out_dir / "gpu_memory_snapshots" / f"iteration{run_no}" / name
                verbose = VerbosityLevels.MORE
            elif record_gpu_memory_kind == 1:
                name = "snapshot_initial.pickle"
                path = out_dir / "gpu_memory_snapshots" / f"iteration{run_no}" / name
                verbose = VerbosityLevels.NONE
            else:
                path = out_dir / "gpu_memory_snapshots" / "snapshot_forward.pickle"
                verbose = VerbosityLevels.MORE
            record_gpu_memory_snapshots = RecordGPUMemory(
                path=str(path),
                max_entries=record_gpu_memory_snapshots.max_entries,
                verbose=verbose,
            )
            record_gpu_memory_snapshots.start_recording()

        print_with_rank_and_timestamp(
            "Starting gradient computation.",
            fabric.global_rank,
        )

        # Note: We do not use `fabric.backward` here. If `devices > 1`,
        # gradient accumulation happens in `model.backward`, using
        # `fabric.all_reduce` explicitly.
        loss = model(
            input_ids=batch[INPUT_IDS_NAME],
            targets=batch["targets"],
            scale_factor=1.0 / train.gradient_accumulation_iters(devices, 1),
            record_gpu_memory_snapshots=record_gpu_memory_snapshots,
            record_gpu_memory_kind=(
                record_gpu_memory_kind
                if record_gpu_memory_snapshots is not None
                else None
            ),
        )
        loss.backward()

        running_loss.update(loss.detach().to(device=optim_device))
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

        if record_gpu_memory_snapshots is not None and record_gpu_memory_kind != 2:
            # Stop recording and store snapshot. For kind 0, this is the single
            # snapshot for the iteration. For kind 1, this is the final snapshot.
            record_gpu_memory_snapshots.store_current_snapshot()
            record_gpu_memory_snapshots.stop_recording()

        cpu_optimizer.step()
        cpu_optimizer.zero_grad(set_to_none=True)
        cpu_scheduler.step()
        if gpu_optimizer is not None:
            gpu_optimizer.step()
            gpu_optimizer.zero_grad(set_to_none=True)
            gpu_scheduler.step()
        print_message("Optimizer update done.", fabric)
        state["step_count"] += 1

        del loss
        gc.collect()
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

        total_lengths += num_tokens
        if state["iter_num"] % train.log_interval == 0:
            loss = running_loss.compute().item()
            t1 = time.perf_counter()
            throughput.update(
                time=t1 - total_t0,
                batches=state["iter_num"],
                samples=state["iter_num"] * train.micro_batch_size,
                lengths=total_lengths,
            )
            throughput.compute_and_log(step=state["iter_num"])
            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "tokens": token_counts["raw_tokens_plus_prompt_template"],
                "total_tokens": token_counts["raw_tokens_plus_prompt_template"]
                * fabric.world_size,
                "learning_rate": cpu_scheduler.get_last_lr()[0],
                **log_memory_all_devices(),
            }
            if isinstance(val_loss, torch.Tensor):
                val_loss = f"{val_loss:.3f}"
            print_message(
                f"Epoch {metrics['epoch']} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" val: {val_loss} |"
                f" iter time: {metrics['iter_time']:.2f} s",
                fabric,
            )
            fabric.log_dict(metrics, step=state["iter_num"])

        if state["step_count"] % eval.interval == 0:
            print_with_rank_and_timestamp(
                "Starting validation evaluations.",
                fabric.global_rank,
            )
            print_message(
                f"\nPeriodic validation evaluation  (batch_size = {val_dataloader.batch_size}) ...",
                fabric,
            )
            if generate_with_eval:
                generate_example_kwargs = dict(
                    tokenizer=tokenizer,
                    data=data,
                )
            else:
                generate_example_kwargs = None
            valid_model = model.copy_model_for_evaluation()
            metrics = validate_and_all_reduce(
                model=valid_model,
                val_dataloader=val_dataloader,
                eval=eval,
                batch_transform=batch_transform,
                generate_example_kwargs=generate_example_kwargs,
                log_metrics=False,
                fabric=fabric,
            )
            fabric.log_dict(metrics, step=state["iter_num"])
            print_with_rank_and_timestamp(
                "Finished validation evaluations.",
                fabric.global_rank,
            )
            flush_io_streams()
            val_loss = f"{metrics['val_loss']:.3f}"
            print_message(
                f"Epoch {train_iterator.epoch} | iter {state['iter_num']} | val loss: {val_loss} | val ppl: {metrics['val_ppl']:.3f} | val_time_in_ms: {metrics['val_time_in_ms']:.3f}",
                fabric,
            )
            deallocate_kv_cache_buffers_of_model(valid_model.gpt_model)
            del valid_model
            fabric.barrier()

        if (
            train.save_interval is not None
            and state["step_count"] % train.save_interval == 0
        ):
            interval_dir = out_dir / f"step-{state['step_count']:06d}"
            save_lora_checkpoint(fabric, model, interval_dir)
            if fabric.global_rank == 0:
                copy_config_files(checkpoint_dir, interval_dir)
                save_hyperparameters(setup, interval_dir)
                if hasattr(data, "prompt_style"):
                    save_prompt_style(data.prompt_style, interval_dir)

    return {
        key: fabric.all_reduce(token_counts[key], reduce_op="sum").item()
        for key in token_counts.keys()
    }
