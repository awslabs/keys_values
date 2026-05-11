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
from dataclasses import dataclass
import gc

import os
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, Literal, Optional, Union, Any, Tuple, List, Callable

import lightning as L
from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.utilities import ThroughputMonitor
import torch
from torchmetrics import RunningMean

from litgpt.data import DataModule
from litgpt.prompts import save_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    CycleIterator,
    auto_download_checkpoint,
    check_nvlink_connectivity,
    check_valid_checkpoint_dir,
    create_finetuning_performance_report,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    num_parameters,
    parse_devices,
    select_sft_generate_example,
)

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.attention.attention_utils import (
    DEFAULT_TMP_ARRAY_LIMIT_GB,
    SDPA_KERNELS_BEST_ORDERING,
)
from keys_values.config import Config as ConfigFull
from keys_values.data import Helmet, LongBenchV2, MyDataLoader
from keys_values.data.base import INPUT_IDS_NAME, TARGETS_STRINGS_NAME
from keys_values.evaluation.evaluator import SampleBasedMetricsEvaluator
from keys_values.attention.flashinfer_wrapper import get_flashinfer_sdpa
from keys_values.attention.flex_attention import FlexAttentionArgs, choose_q_lens
from keys_values.finetune.args import (
    TrainArgs,
    EvalArgs,
    GradientArgs,
    KVCacheArgs,
    OptimizerArgs,
    SDPAArgs,
    LoRAArgs,
)
from keys_values.finetune.batch_transform import (
    BatchTransformFactory,
    BatchTransform,
)
from keys_values.finetune.resume_state import (
    TrainingStateManager,
    load_training_state,
    restore_dataset_from_training_state,
    restore_from_training_state,
    TRAINSTATE_ITERATOR_FNAME,
)
from keys_values.finetune.longcontext_full import (
    create_gpt_model,
    wrap_gpt_model,
    get_mha_and_cache_kwargs,
    validate,
)
from keys_values.finetune.utils import (
    print_but_limit_size,
    get_lr_scheduler,
    get_dataloaders,
    validate_args,
    save_model_checkpoint,
    load_model_checkpoint,
    choose_logger,
    adapt_requires_grad,
    print_with_rank_and_timestamp,
    print_message,
    check_kv_cache,
    create_optimizer,
    may_match_twice_factory,
    adjust_cache_kwargs,
    copy_config_files,
)
from keys_values.fused import (
    set_fused_swiglu_enabled,
    set_fused_rmsnorm_enabled,
)
from keys_values.generate.base import generate
from keys_values.gpu_memory import RecordGPUMemory
from keys_values.head_model import HeadModel, CrossEntropyOnLogits
from keys_values.head_model_factory import HeadModelFactory
from keys_values.kvcache.consts import split_name
from keys_values.kvcache.factory import (
    KVCacheFactory,
    deallocate_kv_cache_buffers_of_model,
    cleanup_cache_kwargs,
)
from keys_values.kvcache.gradient.main import (
    LongContextGradientModel,
    NaiveGPTAndHeadModel,
)
from keys_values.kvcache.offloading import KVCacheOffloader
from keys_values.long_context import (
    GPTAndHeadModel,
    LongContextInferenceModel,
)
from keys_values.lora import (
    GPT as GPTLoRA,
    Config as ConfigLoRA,
    mark_only_lora_as_trainable,
)
from keys_values.model import GPT as GPTFull
from keys_values.optimize.grad_accumulate import CPUOffloadAccumulateGradients
from keys_values.optimize.model_factory import BlockComponentName
from keys_values.parser_config import save_hyperparameters
from keys_values.pos_encoding import (
    position_encoding_factory,
    set_fused_rope_enabled,
)
from keys_values.tools.size_log import (
    SizeWeightsGradientsLog,
    SizeLogMapper,
    SizeLogMapperRule,
    StoreWeightsRule,
    get_match_for_store_rule,
)
from keys_values.utils import (
    flush_io_streams,
    VerbosityLevels,
    fabric_precision_to_dtype,
    message_memory_all_devices,
    log_memory_all_devices,
    check_for_nan_module_weights,
)

DEFAULT_OUT_DIR = "out/finetune/autotune_full"


def setup(
    checkpoint_dir: Path,
    out_dir: Path = Path(DEFAULT_OUT_DIR),
    precision: Optional[str] = None,
    devices: Union[int, str] = 1,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=50,
        log_interval=1,
        global_batch_size=None,
        micro_batch_size=2,
        lr_warmup_steps=None,
        lr_warmup_fraction=0.15,
        epochs=5,
        max_seq_length=None,
        intermed_save_interval=None,
        intermed_save_num=None,
        max_grad_norm=1.0,
        average_loss_per_batch=True,
    ),
    eval: EvalArgs = EvalArgs(
        interval=600,
        max_new_tokens=100,
        max_iters=100,
        initial_validation=None,  # Default set below
        final_validation=True,
        micro_batch_size=None,
        use_sample_metric=False,
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
            "max_num_ranges": 4,
        },
        randomize_chunk_sizes=False,
        allocate_buffers=False,
    ),
    grad: GradientArgs = GradientArgs(
        layers_per_cell=1,
        chunks_per_cell_multiplier=1.0,
        layercp_qname=None,
        cachecp_qname=None,
        single_tokens_for_targets=False,
        use_old_cache=False,
        max_match_trials_pack_arg=8,
        layercp_pin_memory=False,
        cachecp_pin_memory=False,
    ),
    head_model: str = CrossEntropyOnLogits.NAME,
    head_model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: Optional[str] = None,
    attention_forward_temp_size_gb: Optional[float] = None,
    attention_backward_temp_size_gb: Optional[float] = None,
    oom_error_recovery: bool = False,
    yarn_rope: bool = True,
    sdpa: SDPAArgs = SDPAArgs(
        flex_attention=True,
        flex_extend_kv=False,
        flex_num_q_lens=4,
    ),
) -> None:
    """Run automatic tuning over a number of parameters.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to
            load for finetuning. In general, this will be the Hugging Face
            model name. Use `resume` to restart fine-tuning from a checkpoint
            stored along the way.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        devices: How many devices/GPUs to user
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
        oom_error_recovery: If `True`, we try to recover from device out of
            memory errors by lowering `attention_forward_temp_size_gb`,
            `attention_backward_temp_size_gb` and trying again.
            NOTE: This feature does not properly work at the moment!
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

    """
    setup_internal(
        False,
        setup,
        checkpoint_dir,
        out_dir,
        precision,
        devices,
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
        oom_error_recovery,
        yarn_rope,
        sdpa,
    )


def setup_internal(
    do_cpu_offload: bool,
    original_setup: Callable,
    checkpoint_dir: Path,
    out_dir: Path,
    precision: Optional[str],
    devices: Union[int, str],
    data: Optional[DataModule],
    train: TrainArgs,
    lora: Optional[LoRAArgs],
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
    oom_error_recovery: bool,
    yarn_rope: bool,
    sdpa: SDPAArgs,
) -> None:
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
    if isinstance(data, Helmet) and data.metadata_dir is None:
        data.metadata_dir = str(out_dir / "data")
        print(f"Setting Helmet.metadata_dir to {data.metadata_dir}")
    if not isinstance(data, Helmet) and eval.use_sample_metric:
        raise ValueError(
            "use_sample_metric=True currently supported only for Helmet datasets"
        )
    out_dir = init_out_dir(out_dir)
    if data.metadata_dir is not None:
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
            "Choosing optimizer AdamW with default learning rate. We recommend to at least tune optimizer.learning_rate"
        )
    else:
        print(str(optimizer))
    if train.max_grad_norm is not None:
        print(f"Using gradient clipping with max_grad_norm = {train.max_grad_norm}")
    global_batch_size = train.micro_batch_size * devices
    if train.global_batch_size != global_batch_size:
        print(f"train.global_batch_size not supported, set to {global_batch_size}")
        train.global_batch_size = global_batch_size
    if oom_error_recovery:
        print(
            "Warning: Device out of memory error recovery does not properly "
            "work at the moment."
        )
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
            lora_kind=lora.kind,
        )

    precision = precision or get_default_supported_precision(training=True)
    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"finetune-{config.name}",
        use_fabric=True,
        resume=False,
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

    fabric.launch(
        main,
        do_cpu_offload=do_cpu_offload,
        original_setup=original_setup,
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
        oom_error_recovery=oom_error_recovery,
        yarn_rope=yarn_rope,
        sdpa=sdpa,
    )


# TODO:
# - Worker which runs one job after the next
# - For now: Iterate over jobs sampled at random
def main(
    fabric: L.Fabric,
    do_cpu_offload: bool,
    original_setup: Callable,
    devices: int,
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
    oom_error_recovery: bool,
    yarn_rope: bool,
    sdpa: SDPAArgs,
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
        f"\nNumber of optimizer steps per epoch: {lr_max_steps}",
        fabric,
    )
    fabric.seed_everything(seed)
    if do_cpu_offload:
        cpu_offload_device = torch.device("cuda", fabric.local_rank)
        optim_device = torch.device("cpu")
    else:
        cpu_offload_device = None
        optim_device = fabric.device
    # Enable/disable fused operators
    set_fused_rope_enabled(sdpa.fused_rope)
    set_fused_rmsnorm_enabled(sdpa.fused_rmsnorm)
    set_fused_swiglu_enabled(sdpa.fused_swiglu)

    if fabric.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        # Updates `kv_cache.cache_kwargs` from other args:
        kv_cache = kv_cache.update_cache_kwargs()
        # Set `mha_kwargs`, update kv_cache.cache_kwargs` with that as well:
        mha_kwargs = get_mha_and_cache_kwargs(
            attention_forward_temp_size_gb,
            config,
            kv_cache,
            sdpa,
            yarn_rope,
            fabric,
            devices,
        )
        # Depending on the cache type `kv_cache.name`, the arguments
        # `kv_cache.cache_kwargs` are adjusted
        adjust_cache_kwargs(kv_cache, data, tokenizer)
        dtype = fabric_precision_to_dtype(fabric._precision.precision)
        torch.set_default_dtype(dtype)
        if do_cpu_offload:
            # We create the GPT model on the device, then copy. This is faster
            with torch.device(cpu_offload_device):
                gpt_model = create_gpt_model(config, **mha_kwargs)
                head_model = HeadModelFactory.create(
                    name=head_model_name,
                    config=config,
                    data=data,
                    **head_model_kwargs,
                )
            gpt_model = gpt_model.to(optim_device)
            wrap_kwargs = dict(
                cpu_offload_device=cpu_offload_device,
                offload_num_devices=devices,
            )
        else:
            gpt_model = create_gpt_model(config, **mha_kwargs)
            head_model = HeadModelFactory.create(
                name=head_model_name,
                config=config,
                data=data,
                **head_model_kwargs,
            )
            wrap_kwargs = dict()
        adapt_requires_grad(gpt_model, head_model)
        batch_size = train.micro_batch_size
        if eval.micro_batch_size is not None:
            batch_size = max(batch_size, eval.micro_batch_size)
        model, cache_offloader = wrap_gpt_model(
            gpt_model=gpt_model,
            head_model=head_model,
            kv_cache=kv_cache,
            grad=grad,
            verbose=verbose,
            attention_backward_temp_size_gb=attention_backward_temp_size_gb,
            max_batch_size=batch_size,
            dtype=dtype,
            average_loss_per_batch=train.average_loss_per_batch,
            profile_grad_times=profile_grad_times > 0,
            profile_parts=profile_parts,
            fabric=fabric,
            debug_dont_use_autograd_hooks=debug_dont_use_autograd_hooks,
            oom_error_recovery=oom_error_recovery,
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
            "cache_offloader": cache_offloader,
            "cpu_optimizer": cpu_optimizer,
            "cpu_scheduler": cpu_scheduler,
            "iter_num": 0,
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
        # Note: We do not wrap `model` or `optimizer` in `fabric`, since we do
        # not use their abstraction (which creates endless trouble with DDP,
        # such as autograd graphs not being deallocated)
        optimizer = instantiate_torch_optimizer(
            optimizer.name,
            model.parameters(),
            **optimizer.optimizer_kwargs(),
        )
        scheduler = get_lr_scheduler(
            optimizer, train_args=train, max_steps=lr_max_steps
        )
        state = {
            "model": model,
            "cache_offloader": cache_offloader,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "iter_num": 0,
        }

    if eval.use_sample_metric:
        assert isinstance(data, Helmet)
        evaluator = SampleBasedMetricsEvaluator(
            metrics=[
                SampleBasedMetricsEvaluator.metric_for_helmet_task(data.dataset_key)
            ],
            max_generated_tokens=eval.sample_metric_max_generated_tokens,
            tokenizer=tokenizer,
            sample_kwargs=eval.sample_metric_kwargs,
        )
        print(f"Evaluation metric: {evaluator.metrics[0]}")
    else:
        print("Evaluation metric: eval_loss (same as training loss)")
        evaluator = None

    load_model_checkpoint(fabric, model, checkpoint_dir)
    check_for_nan_module_weights(model.gpt_model)

    train_time = time.perf_counter()
    token_counts = fit(
        fabric=fabric,
        original_setup=original_setup,
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
        evaluator=evaluator,
        tokenizer=tokenizer,
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
        if do_cpu_offload:
            valid_model = model.copy_model_for_evaluation()
        else:
            valid_model = model
        metrics = validate_and_all_reduce(
            model=valid_model,
            evaluator=evaluator,
            val_dataloader=val_dataloader,
            eval=dataclasses.replace(eval, max_iters=len(val_dataloader)),
            batch_transform=batch_transform,
            log_metrics=False,
            fabric=fabric,
        )
        fabric.log_dict(metrics, step=state["iter_num"])
        print_message(
            f"Final evaluation            | "
            + string_for_val_metrics(metrics, evaluator)
            + f" | val_time: {metrics['val_time']:.3f} s",
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
        save_hyperparameters(original_setup, save_dir)
        if hasattr(data, "prompt_style"):
            save_prompt_style(data.prompt_style, save_dir)


def eval_autotune_metrics(
    fabric: L.Fabric,
    original_setup: Callable,
    data: DataModule,
    train_iterator: CycleIterator,
    val_dataloader: MyDataLoader,
    num_train_steps: int,
    num_valid_steps: int,
    num_warmup_steps: int,
    batch_transform: BatchTransform,
    kv_cache: KVCacheArgs,
    grad: GradientArgs,
    devices: int,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    optimizer: OptimizerArgs,
    do_cpu_offload: bool,
    config: Union[ConfigFull, ConfigLoRA],
    head_model_name: str,
    head_model_kwargs: Dict[str, Any],
    verbose: VerbosityLevels,
    attention_forward_temp_size_gb: float,
    attention_backward_temp_size_gb: float,
    yarn_rope: bool,
    sdpa: SDPAArgs,
    tokenizer: Tokenizer,
    do_cpu_offloading: bool,
    cpu_offload_device: torch.device,
    optim_device: torch.device,
) -> Dict[str, Any]:
    # Create model
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        # Updates `kv_cache.cache_kwargs` from other args:
        kv_cache = kv_cache.update_cache_kwargs()
        # Set `mha_kwargs`, update kv_cache.cache_kwargs` with that as well:
        mha_kwargs = get_mha_and_cache_kwargs(
            attention_forward_temp_size_gb,
            config,
            kv_cache,
            sdpa,
            yarn_rope,
            fabric,
            devices,
        )
        # Depending on the cache type `kv_cache.name`, the arguments
        # `kv_cache.cache_kwargs` are adjusted
        adjust_cache_kwargs(kv_cache, data, tokenizer)
        dtype = fabric_precision_to_dtype(fabric._precision.precision)
        torch.set_default_dtype(dtype)
        if do_cpu_offload:
            # We create the GPT model on the device, then copy. This is faster
            with torch.device(cpu_offload_device):
                gpt_model = create_gpt_model(config, **mha_kwargs)
                head_model = HeadModelFactory.create(
                    name=head_model_name,
                    config=config,
                    data=data,
                    **head_model_kwargs,
                )
            gpt_model = gpt_model.to(optim_device)
            wrap_kwargs = dict(
                cpu_offload_device=cpu_offload_device,
                offload_num_devices=devices,
            )
        else:
            gpt_model = create_gpt_model(config, **mha_kwargs)
            head_model = HeadModelFactory.create(
                name=head_model_name,
                config=config,
                data=data,
                **head_model_kwargs,
            )
            wrap_kwargs = dict()
        adapt_requires_grad(gpt_model, head_model)
        batch_size = train.micro_batch_size
        if eval.micro_batch_size is not None:
            batch_size = max(batch_size, eval.micro_batch_size)
        model, cache_offloader = wrap_gpt_model(
            gpt_model=gpt_model,
            head_model=head_model,
            kv_cache=kv_cache,
            grad=grad,
            verbose=verbose,
            attention_backward_temp_size_gb=attention_backward_temp_size_gb,
            max_batch_size=batch_size,
            dtype=dtype,
            average_loss_per_batch=train.average_loss_per_batch,
            fabric=fabric,
            **wrap_kwargs,
        )
    # Create optimizer(s)
    cpu_optimizer = None
    cpu_scheduler = None
    gpu_optimizer = None
    gpu_scheduler = None
    if do_cpu_offloading:
        # We use a optimizer on CPU for all parameters of `gpt_model`. If
        # `head_model` has parameters, we use another optimizer on GPU for them.
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
            max_steps=100,
        )
        head_model_params = list(head_model.parameters())
        if head_model_params:
            gpu_optimizer = instantiate_torch_optimizer(
                optimizer.name,
                head_model_params,
                **optimizer.optimizer_kwargs(),
            )
            gpu_scheduler = get_lr_scheduler(
                gpu_optimizer,
                train_args=train,
                max_steps=100,
            )
    else:
        # Note: We do not wrap `model` or `optimizer` in `fabric`, since we do
        # not use their abstraction (which creates endless trouble with DDP,
        # such as autograd graphs not being deallocated)
        gpu_optimizer = instantiate_torch_optimizer(
            optimizer.name,
            model.parameters(),
            **optimizer.optimizer_kwargs(),
        )
        gpu_scheduler = get_lr_scheduler(
            gpu_optimizer, train_args=train, max_steps=100
        )

    result_metrics = {
        "out_of_memory": False,
        "runtime_exception": False,
    }

    try:

        # Run some evaluation steps
        if do_cpu_offloading:
            valid_model = model.copy_model_for_evaluation()
        else:
            valid_model = model
        # Warmup
        if num_warmup_steps > 0:
            validate(
                model=valid_model,
                val_dataloader=val_dataloader,
                eval=dataclasses.replace(eval, max_iters=num_warmup_steps),
                batch_transform=batch_transform,
            )
        # Validation (timed)
        timer_t0 = time.perf_counter()
        avg_loss, _ = validate(
            model=valid_model,
            val_dataloader=val_dataloader,
            eval=dataclasses.replace(eval, max_iters=num_valid_steps),
            batch_transform=batch_transform,
        )
        time_eval = time.perf_counter() - timer_t0
        result_metrics["time_eval"] = time_eval
        print(f"Validation: val_loss = {avg_loss:.2f}, time = {time_eval:.3f} secs")
        flush_io_streams()
        if do_cpu_offloading:
            deallocate_kv_cache_buffers_of_model(valid_model.gpt_model)
            del valid_model

        # Run some training steps
        gc.collect()
        torch.cuda.empty_cache()
        sum_loss = 0
        time_train = 0
        for iter_num in range(num_train_steps):
            timer_t0 = time.perf_counter()
            batch = batch_transform(next(train_iterator))
            # Compute loss and gradients
            # We do not use `fabric.backward`. For CPU offloading, loss and
            # gradient accumulation happens in `loss.backward` already. Otherwise,
            # we run an explicit all_reduce.
            loss = model(
                input_ids=batch[INPUT_IDS_NAME],
                targets=batch["targets"],
                scale_factor=1.0,
            )
            loss.backward()
            sum_loss += loss.item()
            time_train += (time.perf_counter() - timer_t0)
            flush_io_streams()

            if train.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train.max_grad_norm,
                )
            if cpu_optimizer is not None:
                cpu_optimizer.step()
                cpu_optimizer.zero_grad(set_to_none=True)
                cpu_scheduler.step()
            if gpu_optimizer is not None:
                gpu_optimizer.step()
                gpu_optimizer.zero_grad(set_to_none=True)
                gpu_scheduler.step()
            check_for_nan_module_weights(model.gpt_model)

            del loss
            gc.collect()
            torch.cuda.empty_cache()

        result_metrics["time_train"] = time_train
        print(f"Training: train_loss = {(sum_loss / num_train_steps):.2f}, time = {time_train:.3f} secs")

    except RuntimeError as ex:
        if "out of memory" in str(ex):
            result_metrics["out_of_memory"] = True
        else:
            result_metrics.update(
                {
                    "runtime_exception": True,
                    "exception_message": str(ex),
                }
            )

    return result_metrics
