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

import math
import os
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, Literal, Optional, Union, Any

import lightning as L
from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.utilities import ThroughputMonitor
import torch
from torchmetrics import RunningMean

from litgpt.args import TrainArgs
from litgpt.config import Config as ConfigFull
from litgpt.data import DataModule
from litgpt.lora import Config as ConfigLoRA, mark_only_lora_as_trainable
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
from keys_values.data import LongBenchV2, INPUT_IDS_NAME, MyDataLoader
from keys_values.flex_attention import FlexAttentionArgs
from keys_values.finetune.args import (
    EvalArgs,
    GradientArgs,
    KVCacheArgs,
    OptimizerArgs,
    SDPAArgs,
    LoRAARgs,
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
    create_optimizer,
    may_match_twice_factory,
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
from keys_values.long_context import (
    GPTAndHeadModel,
    LongContextInferenceModel,
)
from keys_values.lora import GPT as GPTLoRA
from keys_values.model import GPT as GPTFull
from keys_values.optimize.grad_accumulate import CPUOffloadAccumulateGradients
from keys_values.optimize.model_factory import BlockComponentName
from keys_values.parser_config import save_hyperparameters
from keys_values.pos_encoding import position_encoding_factory
from keys_values.utils import (
    flush_io_streams,
    VerbosityLevels,
    fabric_precision_to_dtype,
    message_memory_all_devices,
    log_memory_all_devices,
    check_for_nan_module_weights,
)


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
        use_old_cache=False,
        max_match_trials_pack_arg=8,
        layer_checkpoint_chunk_size=None,
    ),
    head_model: str = CrossEntropyOnLogits.NAME,
    head_model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: Optional[str] = None,
    attention_forward_temp_size_gb: Optional[float] = None,
    attention_backward_temp_size_gb: Optional[float] = None,
    yarn_rope: bool = True,
    sdpa: SDPAArgs = SDPAArgs(
        flex_attention=True,
        flex_extend_kv=False,
    ),
    record_gpu_memory_snapshots: Optional[int] = None,
    record_gpu_memory_kind: int = 0,
    record_gpu_memory_period: int = 0,
    generate_with_eval: bool = False,
    profile_grad_times: int = 0,
    profile_parts: Optional[str] = None,
    debug_dont_use_autograd_hooks: bool = False,
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
            is used in backward if `grad.use_old_cache == True`.
        yarn_rope: Should YaRN be used to adjust RoPE (position encoding) to the
            sequence length for each batch? Defaults to `True`. If not, RoPE is
            determined by the model configuration, and is static (no dependence
            on sequence length).
        sdpa: Configuration for scaled dot product attention (SDPA), the core
            of multi-head self attention, see
            ``keys_values.finetune.args.SDPAArgs`` for details. Set
            `sdpa.flex_attention` to `True` to activate PyTorch
            `flex_attention`. Otherwise, the zero-padded query SDPA kernel is
            used.
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
        generate_with_eval: If `True`, we test token generation with each
            evaluation
        profile_grad_times: If given, we profile complete gradient computation
            for this many steps, then stop. Results are written to CSV file.
        profile_parts: If given, we use `cProfile` to profile the first forward
            (if "forward") or first backward (if "backward") pass. Results are
            printed, then the program stops.

    """
    setup_internal(
        False,
        checkpoint_dir,
        out_dir,
        precision,
        devices,
        num_nodes,
        resume,
        data,
        train,
        None,
        eval,
        optimizer,
        logger_name,
        seed,
        access_token,
        kv_cache,
        grad,
        head_model,
        head_model_kwargs,
        verbose,
        attention_forward_temp_size_gb,
        attention_backward_temp_size_gb,
        yarn_rope,
        sdpa,
        record_gpu_memory_snapshots,
        record_gpu_memory_kind,
        record_gpu_memory_period,
        generate_with_eval,
        profile_grad_times,
        profile_parts,
        debug_dont_use_autograd_hooks,
    )


def setup_internal(
    do_cpu_offload: bool,
    checkpoint_dir: Path,
    out_dir: Path,
    precision: Optional[str],
    devices: Union[int, str],
    num_nodes: int,
    resume: Union[bool, Literal["auto"], Path],
    data: Optional[DataModule],
    train: TrainArgs,
    lora: Optional[LoRAARgs],
    eval: EvalArgs,
    optimizer: Optional[OptimizerArgs],
    logger_name: Literal["wandb", "tensorboard", "csv", "mlflow"],
    seed: int,
    access_token: Optional[str],
    kv_cache: KVCacheArgs,
    grad: GradientArgs,
    head_model: str,
    head_model_kwargs: Optional[Dict[str, Any]],
    verbose: Optional[str],
    attention_forward_temp_size_gb: Optional[float],
    attention_backward_temp_size_gb: Optional[float],
    yarn_rope: bool,
    sdpa: SDPAArgs,
    record_gpu_memory_snapshots: Optional[int],
    record_gpu_memory_kind: int,
    record_gpu_memory_period: int,
    generate_with_eval: bool,
    profile_grad_times: int,
    profile_parts: Optional[str],
    debug_dont_use_autograd_hooks: bool,
) -> None:
    if do_cpu_offload and not torch.cuda.is_available():
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
    if do_cpu_offload and not (1 <= devices <= torch.cuda.device_count()):
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
            "Choosing optimizer AdamW with default learning rate. We recommend to at least tune optimizer.learning_rate"
        )
    else:
        print(str(optimizer))
    if do_cpu_offload:
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
    if lora is None:
        config = ConfigFull.from_file(checkpoint_dir / "model_config.yaml")
    else:
        config = ConfigLoRA.from_file(
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
    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"finetune-{config.name}",
        use_fabric=True,
        resume=bool(resume),
        log_interval=train.log_interval,
    )

    if devices * num_nodes > 1:
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
        do_cpu_offload=do_cpu_offload,
        devices=devices,
        num_nodes=num_nodes,
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
        head_model_name=head_model,
        head_model_kwargs=head_model_kwargs,
        verbose=verbose,
        attention_forward_temp_size_gb=attention_forward_temp_size_gb,
        attention_backward_temp_size_gb=attention_backward_temp_size_gb,
        yarn_rope=yarn_rope,
        sdpa=sdpa,
        record_gpu_memory_snapshots=record_gpu_memory_snapshots,
        record_gpu_memory_kind=record_gpu_memory_kind,
        record_gpu_memory_period=record_gpu_memory_period,
        generate_with_eval=generate_with_eval,
        profile_grad_times=profile_grad_times,
        profile_parts=profile_parts,
        debug_dont_use_autograd_hooks=debug_dont_use_autograd_hooks,
    )


def main(
    fabric: L.Fabric,
    do_cpu_offload: bool,
    devices: int,
    num_nodes: int,
    resume: Union[bool, Literal["auto"], Path],
    seed: int,
    config: Union[ConfigFull, ConfigLoRA],
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
    sdpa: SDPAArgs,
    record_gpu_memory_snapshots: Optional[RecordGPUMemory],
    record_gpu_memory_kind: int,
    record_gpu_memory_period: int,
    generate_with_eval: bool,
    profile_grad_times: int,
    profile_parts: Optional[str],
    debug_dont_use_autograd_hooks: bool,
) -> None:
    validate_args(train, eval)
    is_lora = isinstance(config, ConfigLoRA)

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
    steps_per_epoch = len(train_dataloader)
    lr_max_steps = min(
        train.epochs * steps_per_epoch, (train.max_steps or float("inf"))
    )
    print_message(
        f"Number of optimizer steps per epoch: {lr_max_steps}",
        fabric,
    )
    fabric.seed_everything(seed)
    if do_cpu_offload:
        cpu_offload_device = torch.device("cuda", fabric.local_rank)
        optim_device = torch.device("cpu")
    else:
        cpu_offload_device = None
        optim_device = fabric.device

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
            tmp_array_limit_gb=tmp_array_limit_forward,
            pos_encoding=position_encoding_factory(config, do_yarn=yarn_rope),
        )
        if "sdpa_kernels" in kv_cache.cache_kwargs:
            mha_kwargs["sdpa_kernels"] = kv_cache.cache_kwargs["sdpa_kernels"]
        else:
            mha_kwargs["sdpa_kernels"] = SDPA_KERNELS_BEST_ORDERING
        if sdpa.flex_attention:
            # The block mask managers (for prefill, for chunks) are shared
            # among all multi-head attention blocks
            flexatt_args = FlexAttentionArgs(extend_kv=sdpa.flex_extend_kv)
            mha_kwargs["flexatt_args"] = flexatt_args
        kv_cache.cache_kwargs.update(mha_kwargs)
        dtype = fabric_precision_to_dtype(fabric._precision.precision)
        torch.set_default_dtype(dtype)
        if do_cpu_offload:
            # We create the GPT model on the device, then copy. This is faster
            with torch.device(cpu_offload_device):
                if not is_lora:
                    gpt_model = GPTFull(config, **mha_kwargs)
                else:
                    gpt_model = GPTLoRA(config, **mha_kwargs)
                head_model = HeadModelFactory.create(
                    name=head_model_name,
                    config=config,
                    data=data,
                    **head_model_kwargs,
                )
                gpt_model.apply(gpt_model._init_weights)
            gpt_model = gpt_model.to(optim_device)
            wrap_kwargs = dict(
                cpu_offload_device=cpu_offload_device,
                offload_num_devices=devices,
            )
        else:
            if not is_lora:
                gpt_model = GPTFull(config, **mha_kwargs)
            else:
                gpt_model = GPTLoRA(config, **mha_kwargs)
            head_model = HeadModelFactory.create(
                name=head_model_name,
                config=config,
                data=data,
                **head_model_kwargs,
            )
            gpt_model.apply(gpt_model._init_weights)
            wrap_kwargs = dict()
        if is_lora:
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
            dtype=dtype,
            profile_grad_times=profile_grad_times > 0,
            profile_parts=profile_parts,
            fabric=fabric,
            debug_dont_use_autograd_hooks=debug_dont_use_autograd_hooks,
            **wrap_kwargs,
        )

    num_trainable_params = num_parameters(model, requires_grad=True)
    print_message(
        f"\nNumber of trainable parameters: {num_trainable_params:,}",
        fabric,
    )
    if is_lora:
        print_message(
            f"Number of non-trainable parameters: {num_parameters(model, requires_grad=False):,}",
            fabric,
        )

    if do_cpu_offload:
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
    else:
        model = fabric.setup(model)
        optimizer = instantiate_torch_optimizer(
            optimizer.name,
            model.parameters(),
            **optimizer.optimizer_kwargs(),
        )
        optimizer = fabric.setup_optimizers(optimizer)
        scheduler = get_lr_scheduler(
            optimizer, train_args=train, max_steps=lr_max_steps
        )
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
        print_message(f"Loading model checkpoint: {checkpoint_dir}", fabric)
        file_path = checkpoint_dir / LIT_MODEL_FNAME
        # strict=False because missing keys due to LoRA weights not contained in state dict
        load_checkpoint(fabric, model.gpt_model, file_path, strict=not is_lora)
        # If there are head model weights, load them as well. Otherwise, we use
        # random initialization (or the head model may not have weights)
        file_path = checkpoint_dir / HEAD_MODEL_FNAME
        if file_path.exists():
            load_checkpoint(fabric, model.head_model, file_path, strict=True)
        check_for_nan_module_weights(model.gpt_model)

    if profile_grad_times > 0 and fabric.global_rank == 0:
        thresh = grad.max_match_trials_pack_arg
        name = "old" if grad.use_old_cache else "new"
        profile_grad_params = {
            "path": Path(out_dir) / f"profile_grad_times_{name}_{thresh}.csv",
            "use_old_cache": grad.use_old_cache,
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
        num_nodes=num_nodes,
        resume=resume,
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
            "Starting validation evaluations.",
            fabric.global_rank,
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
        if do_cpu_offload:
            valid_model = model.copy_model_for_evaluation()
        else:
            valid_model = model
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
            f"Final evaluation | val loss: {metrics['val_loss']:.3f} | val ppl: {metrics['val_ppl']:.3f} | val_time: {metrics['val_time']:.3f} s",
            fabric,
        )
        flush_io_streams()
        if do_cpu_offload:
            deallocate_kv_cache_buffers_of_model(valid_model.gpt_model)
            del valid_model

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
    gpt_model: Union[GPTFull, GPTLoRA],
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
    debug_dont_use_autograd_hooks: bool = False,
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
            "use_old_cache": grad.use_old_cache,
        }
        autograd_hooks_kwargs: Dict[str, Any] = dict(
            may_match_twice=may_match_twice_factory(grad, gpt_model),
        )
        if grad.max_match_trials_pack_arg is not None:
            autograd_hooks_kwargs["max_match_trials_pack_arg"] = (
                grad.max_match_trials_pack_arg
            )
        if cpu_offload_device is not None:
            common_kwargs["head_model"] = head_model.to(device=cpu_offload_device)
            offload_grad_accum = CPUOffloadAccumulateGradients(
                group=list(range(offload_num_devices)),
                fabric=fabric,
            )
            if offload_num_devices > 1:
                # Test connection: all-reduce with sum must work
                offload_grad_accum.test_all_reduce()
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
        # DEBUG
        # track_unmatched_annotations = lambda layer_idx, chunk_idx: layer_idx in (0, 35)
        # END DEBUG
        may_match_twice = may_match_twice_factory(grad, gpt_model)
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
            track_unmatched_annotations=None,
            debug_profile_forward=profile_parts == "forward",
            debug_profile_backward=profile_parts == "backward",
            debug_dont_use_autograd_hooks=debug_dont_use_autograd_hooks,
        )
    else:
        model = LongContextInferenceModel(**common_kwargs)
    return model


def create_baseline_model(
    gpt_model: Union[GPTFull, GPTLoRA],
    config: Union[ConfigFull, ConfigLoRA],
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
    train_dataloader: MyDataLoader,
    val_dataloader: MyDataLoader,
    batch_transform: BatchTransform,
    devices: int,
    num_nodes: int,
    resume: Union[bool, Literal["auto"], Path],
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
    do_cpu_offloading = "cpu_optimizer" in state
    model = state["model"]
    if not do_cpu_offloading:
        gpu_optimizer = state["optimizer"]
        gpu_scheduler = state["scheduler"]
        cpu_optimizer = None
        cpu_scheduler = None
        optim_device = fabric.device
    else:
        gpu_optimizer = state.get("gpu_optimizer")
        gpu_scheduler = state.get("gpu_scheduler")
        cpu_optimizer = state["cpu_optimizer"]
        cpu_scheduler = state["cpu_scheduler"]
        optim_device = torch.device("cpu")
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

    if record_gpu_memory_kind == 3:
        path = out_dir / "gpu_memory_snapshots" / "snapshot_validation.pickle"
        record_gpu_memory_snapshots = RecordGPUMemory(
            path=str(path),
            max_entries=record_gpu_memory_snapshots.max_entries,
            verbose=VerbosityLevels.MORE,
        )
        record_gpu_memory_snapshots.start_recording()

    val_loss = "n/a"
    if do_cpu_offloading:
        valid_model = model.copy_model_for_evaluation()
    else:
        valid_model = model
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
            f"Initial evaluation | val loss: {val_loss} | val ppl: {metrics['val_ppl']:.3f} | val_time: {metrics['val_time']:.3f} s",
            fabric,
        )
    else:
        print_message("Verifying settings ...", fabric)
        with torch.no_grad():
            validate(
                valid_model,
                val_dataloader,
                dataclasses.replace(eval, max_iters=1),
                batch_transform,
            )  # sanity check
    flush_io_streams()
    if do_cpu_offloading:
        deallocate_kv_cache_buffers_of_model(valid_model.gpt_model)
        del valid_model

    if record_gpu_memory_kind == 3:
        if record_gpu_memory_snapshots.is_recording:
            record_gpu_memory_snapshots.store_current_snapshot()
            record_gpu_memory_snapshots.stop_recording()
        # Switch off from here on
        record_gpu_memory_snapshots = None
        record_gpu_memory_kind = 0

    initial_iter = state["iter_num"]
    max_steps = train.max_steps or float("inf")
    train_iterator = CycleIterator(train_dataloader)
    throughput = ThroughputMonitor(fabric, window_size=50)

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
            f" seconds to reach iteration {initial_iter}.",
            fabric,
        )

    running_loss = RunningMean(
        window=train.gradient_accumulation_iters(devices, num_nodes),
        sync_on_compute=False,
    ).to(optim_device)
    fabric.barrier()
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

        is_accumulating = (not do_cpu_offloading) and state[
            "iter_num"
        ] % train.gradient_accumulation_iters(devices, num_nodes) != 0
        print_with_rank_and_timestamp(
            "Starting gradient computation.",
            fabric.global_rank,
        )

        model_kwargs = dict(
            input_ids=batch[INPUT_IDS_NAME],
            targets=batch["targets"],
            scale_factor=1.0 / train.gradient_accumulation_iters(devices, num_nodes),
            record_gpu_memory_snapshots=record_gpu_memory_snapshots,
            record_gpu_memory_kind=(
                record_gpu_memory_kind
                if record_gpu_memory_snapshots is not None
                else None
            ),
        )
        if not do_cpu_offloading:
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                loss = model(**model_kwargs)
                fabric.backward(loss)
        else:
            # Note: We do not use `fabric.backward` here. If `devices > 1`,
            # gradient accumulation happens in `model.backward`, using
            # `fabric.all_reduce` explicitly.
            loss = model(**model_kwargs)
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

        if not is_accumulating:
            if cpu_optimizer is not None:
                cpu_optimizer.step()
                cpu_optimizer.zero_grad(set_to_none=True)
                cpu_scheduler.step()
            if gpu_optimizer is not None:
                gpu_optimizer.step()
                gpu_optimizer.zero_grad(set_to_none=True)
                gpu_scheduler.step()
            print_message("Optimizer update done.", fabric)
            state["step_count"] += 1
            check_for_nan_module_weights(model.gpt_model)

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
            if gpu_scheduler is not None:
                learning_rate = gpu_scheduler.get_last_lr()[0]
            else:
                assert cpu_scheduler is not None
                learning_rate = cpu_scheduler.get_last_lr()[0]
            metrics = {
                "loss": loss,
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "tokens": token_counts["raw_tokens_plus_prompt_template"],
                "total_tokens": token_counts["raw_tokens_plus_prompt_template"]
                * fabric.world_size,
                "learning_rate": learning_rate,
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
            if do_cpu_offloading:
                valid_model = model.copy_model_for_evaluation()
            else:
                valid_model = model
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
            val_loss = f"{metrics['val_loss']:.3f}"
            print_message(
                f"Epoch {train_iterator.epoch} | iter {state['iter_num']} | val loss: {val_loss} | val ppl: {metrics['val_ppl']:.3f} | val_time: {metrics['val_time']:.3f} s",
                fabric,
            )
            flush_io_streams()
            if do_cpu_offloading:
                deallocate_kv_cache_buffers_of_model(valid_model.gpt_model)
                del valid_model
            fabric.barrier()

        if (
            train.save_interval is not None
            and not is_accumulating
            and state["step_count"] % train.save_interval == 0
        ):
            interval_dir = out_dir / f"step-{state['step_count']:06d}"
            save_model_checkpoint(fabric, model, interval_dir)
            if fabric.global_rank == 0:
                copy_config_files(checkpoint_dir, interval_dir)
                save_hyperparameters(setup, interval_dir)
                if hasattr(data, "prompt_style"):
                    save_prompt_style(data.prompt_style, interval_dir)

    return {
        key: fabric.all_reduce(token_counts[key], reduce_op="sum").item()
        for key in token_counts.keys()
    }


def validate_and_all_reduce(
    model: GPTAndHeadModel,
    val_dataloader: MyDataLoader,
    eval: EvalArgs,
    batch_transform: BatchTransform,
    generate_example_kwargs: Optional[Dict[str, Any]] = None,
    log_metrics: bool = True,
    fabric: Optional[L.Fabric] = None,
) -> Dict[str, float]:
    val_time = None
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
        "val_time": val_time,
    }
    if fabric is not None and log_metrics:
        fabric.log_dict(metrics)
    return metrics


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(
    model: GPTAndHeadModel,
    val_dataloader: MyDataLoader,
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
