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
import dataclasses
from datetime import datetime
import gc
import os
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, Literal, Optional, Union, Any, Callable, Tuple, Set

import lightning as L
from lightning.fabric.strategies import DDPStrategy
import torch

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    CycleIterator,
    auto_download_checkpoint,
    check_nvlink_connectivity,
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    init_out_dir,
    instantiate_torch_optimizer,
    parse_devices,
)

from keys_values.autotune.optuna_study import (
    OptunaStudyConfig,
    ask_optuna_trial,
    create_optuna_study,
    open_optuna_study,
    tell_optuna_trial,
    require_optuna,
    MAX_SUGGESTION_TRIES,
    OptunaArgs,
)
from keys_values.config import Config as ConfigFull
from keys_values.data import Helmet, LongBenchV2, MyDataLoader
from keys_values.data.base import INPUT_IDS_NAME
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
from keys_values.finetune.longcontext_full import (
    create_gpt_model,
    wrap_gpt_model,
    get_mha_and_cache_kwargs,
    validate,
)
from keys_values.finetune.utils import (
    get_lr_scheduler,
    get_dataloaders,
    validate_args,
    load_model_checkpoint,
    choose_logger,
    adapt_requires_grad,
    check_kv_cache,
    create_optimizer,
    adjust_cache_kwargs,
)
from keys_values.finetune.resume_state import get_iterator
from keys_values.fused import (
    set_fused_swiglu_enabled,
    set_fused_rmsnorm_enabled,
)
from keys_values.head_model import CrossEntropyOnLogits
from keys_values.head_model_factory import HeadModelFactory
from keys_values.kvcache.consts import split_name
from keys_values.kvcache.factory import deallocate_kv_cache_buffers_of_model
from keys_values.lora import Config as ConfigLoRA
from keys_values.optimize.model_factory import BlockComponentName
from keys_values.pos_encoding import set_fused_rope_enabled
from keys_values.utils import (
    flush_io_streams,
    VerbosityLevels,
    fabric_precision_to_dtype,
    check_for_nan_module_weights,
    randchoice_torch,
)

DEFAULT_OUT_DIR = "out/finetune/autotune_full"

DEFAULT_NUM_TRAIN_STEPS = 4

DEFAULT_NUM_VALID_STEPS = 6

DEFAULT_NUM_WARMUP_STEPS = 1


VARIABLE_CHOICES = {
    "kv_cache:buffer_name": (
        "default",
        "torch-quantized8",
        "torch-quantized4",
    ),
    "kv_cache:cache_length": (32768, 34816, 36864, 38912, 40960),
    "kv_cache:chunk_size": (512, 1024, 2048, 4096),
    "kv_cache:cpu_offload": (True, False),
    "grad:layers_per_cell": (1, 2),
    "grad:chunks_per_cell_multiplier": (0.5, 0.75, 1, 1.25, 1.5, 1.75),
    "grad:layercp_qname": (
        "default",
        "torch-quantized8",
    ),
    "grad:cachecp_qname": (
        "default",
        "torch-quantized8",
    ),
}


def setup(
    checkpoint_dir: Path,
    out_dir: Path = Path(DEFAULT_OUT_DIR),
    precision: Optional[str] = None,
    devices: Union[int, str] = 1,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        micro_batch_size=2,
        max_grad_norm=1.0,
        average_loss_per_batch=True,
    ),
    eval: EvalArgs = EvalArgs(
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
    yarn_rope: bool = True,
    sdpa: SDPAArgs = SDPAArgs(
        flex_attention=True,
        flex_extend_kv=False,
        flex_num_q_lens=4,
    ),
    time_budget: Optional[int] = None,
    max_num_evals: Optional[int] = None,
    num_train_steps: Optional[int] = None,
    num_valid_steps: Optional[int] = None,
    num_warmup_steps: Optional[int] = None,
    optuna: Optional[OptunaArgs] = None,
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
        time_budget: Maximum time (in seconds) a worker (device) may run
            evaluations. The total time may be longer, since an evaluation is
            started whenever elapsed time is below budget.
        max_num_evals: Maximum number of evaluations a worker may run.
        num_train_steps: Number of training steps run per evaluation.
        num_valid_steps: Number of validation steps run per evaluation.
        num_warmup_steps: Number of warmup steps run per evaluation. A warmup
            step is a validation step not counted in the validation time. We
            run warmup, then validation, then training.
        optuna: Configuration for Optuna-guided search. When provided,
            ``main`` will ask Optuna for trial configurations. The sampler is
            selected by ``optuna.name`, from "nsgaii", "tpe", "random". For
            additional arguments, see :class:`OptunaArgs`. If ``None``, random
            search is used (not using Optuna).

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
        yarn_rope,
        sdpa,
        time_budget,
        max_num_evals,
        num_train_steps,
        num_valid_steps,
        num_warmup_steps,
        optuna,
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
    yarn_rope: bool,
    sdpa: SDPAArgs,
    time_budget: Optional[int],
    max_num_evals: Optional[int],
    num_train_steps: Optional[int],
    num_valid_steps: Optional[int],
    num_warmup_steps: Optional[int],
    optuna: Optional[OptunaArgs],
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
    if train.epochs is None:
        train.epochs = 1  # Not used, but is checked below
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
    if time_budget is not None and time_budget <= 0:
        raise ValueError(f"time_budget must be positive, got {time_budget}")
    if max_num_evals is not None and max_num_evals <= 0:
        raise ValueError(f"max_num_evals must be positive, got {max_num_evals}")
    if num_train_steps is None:
        num_train_steps = DEFAULT_NUM_TRAIN_STEPS
    elif num_train_steps <= 0:
        raise ValueError(f"num_train_steps must be positive, got {num_train_steps}")
    if num_valid_steps is None:
        num_valid_steps = DEFAULT_NUM_VALID_STEPS
    elif num_valid_steps <= 0:
        raise ValueError(f"num_valid_steps must be positive, got {num_valid_steps}")
    if num_warmup_steps is None:
        num_warmup_steps = DEFAULT_NUM_WARMUP_STEPS
    elif num_warmup_steps < 0:
        raise ValueError(
            f"num_warmup_steps must be nonnegative, got {num_warmup_steps}"
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
        devices=devices,
        global_seed=seed,
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
        time_budget=time_budget,
        max_num_evals=max_num_evals,
        num_train_steps=num_train_steps,
        num_valid_steps=num_valid_steps,
        num_warmup_steps=num_warmup_steps,
        optuna=optuna,
    )


def main(
    fabric: L.Fabric,
    do_cpu_offload: bool,
    devices: int,
    global_seed: int,
    config: Union[ConfigFull, ConfigLoRA],
    data: DataModule,
    checkpoint_dir: Path,
    out_dir: Path,
    time_budget: Optional[int],
    max_num_evals: Optional[int],
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
    num_train_steps: int,
    num_valid_steps: int,
    num_warmup_steps: int,
    optuna: Optional[OptunaArgs],
) -> None:
    validate_args(train, eval)
    tokenizer = Tokenizer(checkpoint_dir)
    ignore_index = getattr(data, "ignore_index", -100)
    batch_transform = BatchTransformFactory.from_head_model(
        head_model=head_model_name,
        pad_id=0,
        eos_id=tokenizer.eos_id,
        ignore_index=ignore_index,
    )
    # Ensure that PRNG seed depends on rank
    seed = global_seed + fabric.global_rank
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

    # Create objects to be shared across ranks. Do this on rank 0:
    # - Optuna study (also pick the name)
    # - Data training state
    # Note: Earlier code did this in `setup_internal`. But somehow,
    # `setup_internal` is run on each rank independently, so this did not work.
    with fabric.rank_zero_first():
        if fabric.global_rank == 0:
            if optuna is not None:
                require_optuna()
                time_format = "%Y%m%d_%H:%M:%S"
                time_stamp = datetime.now().strftime(time_format)
                study_name = optuna.name + "_" + time_stamp
                optuna_config = OptunaStudyConfig(
                    storage_path=str(out_dir / study_name),
                    study_name=study_name,
                    sampler_args=optuna,
                )
                study = create_optuna_study(optuna_config, seed=seed)
                print(f"Created Optuna study: {study_name}")
                with open(str(out_dir / "optuna_study_name.txt"), "w") as fp:
                    fp.write(study_name + "\n")
            # We load and split the data here, in order to obtain the data training
            # state. All evaluations on all devices need to use the same data
            # training state, so that the same training and validation batches are used.
            print(
                "Sample the training state for the data. Must be the same across all evaluations."
            )
            tokenizer = Tokenizer(checkpoint_dir)
            train_dataloader, _ = get_dataloaders(
                data=data,
                tokenizer=tokenizer,
                head_model=head_model_name,
                train=train,
                eval=eval,
            )
            train_state = {
                "data_state": data.training_state.state_dict(),
                "train_iterator": iter(train_dataloader).state_dict(),
            }
            torch.save(train_state, str(out_dir / "data_train_state.pth"))
        else:
            if optuna is not None:
                name_path = out_dir / "optuna_study_name.txt"
                if not name_path.exists():
                    raise FileNotFoundError(f"{name_path} does not exist. Should have been written by rank 0")
                with open(str(name_path), "r") as fp:
                    study_name = fp.readline().strip()
                optuna_config = OptunaStudyConfig(
                    storage_path=str(out_dir / study_name),
                    study_name=study_name,
                    sampler_args=optuna,
                )
                study = open_optuna_study(optuna_config, seed=seed)
                print(f"Loaded Optuna study: {study_name}")
            name_path = out_dir / "data_train_state.pth"
            if not name_path.exists():
                raise FileNotFoundError(f"{name_path} does not exist. Should have been written by rank 0")
            train_state = torch.load(str(name_path))
            if "data_state" not in train_state or "train_iterator" not in train_state:
                raise ValueError(f"train_state loaded from {name_path} has keys {train_state.keys()}: Missing `data_state`, `train_iterator`")

    if optuna is not None:
        print(
            "Using Optuna for auto-tuning:\n"
            f"Study:        {optuna_config.study_name}\n"
            f"Sampler:      {optuna.name}\n"
            f"Sampler args: {optuna.kwargs_not_none()}\n"
            f"Storage path: {optuna_config.storage_path}"
        )
    else:
        optuna_config = None
        print("Using independent random search for auto-tuning.")

    # Resolve variable choices: use optuna_config override if given, else default.
    variable_choices = VARIABLE_CHOICES
    if optuna_config is not None and optuna_config.variable_choices is not None:
        variable_choices = optuna_config.variable_choices

    fixed_kwargs = dict(
        fabric=fabric,
        data=data,
        num_train_steps=num_train_steps,
        num_valid_steps=num_valid_steps,
        num_warmup_steps=num_warmup_steps,
        batch_transform=batch_transform,
        devices=devices,
        checkpoint_dir=checkpoint_dir,
        train=train,
        eval=eval,
        optimizer=optimizer,
        do_cpu_offload=do_cpu_offload,
        config=config,
        head_model_name=head_model_name,
        head_model_kwargs=head_model_kwargs,
        verbose=verbose,
        attention_forward_temp_size_gb=attention_forward_temp_size_gb,
        attention_backward_temp_size_gb=attention_backward_temp_size_gb,
        yarn_rope=yarn_rope,
        sdpa=sdpa,
        tokenizer=tokenizer,
        cpu_offload_device=cpu_offload_device,
        optim_device=optim_device,
    )
    total_time_t0 = time.perf_counter()
    if max_num_evals is None:
        max_num_evals = 1_000_000
    configs_already_sampled: Set[str] = set()
    for num_evals in range(max_num_evals):
        if (
            time_budget is not None
            and time.perf_counter() >= total_time_t0 + time_budget
        ):
            print("Time budget exceeded, exiting.")
            break

        if study is not None:
            # Optuna-guided search: each rank independently asks for its own trial.
            variables, trial = ask_optuna_trial(study, variable_choices)
        else:
            # Fallback: uniform random search.
            variables = sample_configuration(
                configs_already_sampled,
                variable_choices,
            )
            trial = None

        this_kv_cache, this_grad = update_args(kv_cache, grad, variables)
        # Data access must be the same for each configuration
        data.load_training_state(train_state["data_state"])
        train_dataloader, val_dataloader = get_dataloaders(
            data=data,
            tokenizer=tokenizer,
            head_model=head_model_name,
            train=train,
            eval=eval,
            training_state=data.training_state,
        )
        train_iterator = CycleIterator(train_dataloader)
        inner_iter = get_iterator(train_iterator)
        inner_iter.load_state_dict(train_state["train_iterator"])
        train_iterator._iterator = inner_iter
        # Run evaluation for this configuration
        results = eval_autotune_metrics(
            **fixed_kwargs,
            train_iterator=train_iterator,
            val_dataloader=val_dataloader,
            kv_cache=this_kv_cache,
            grad=this_grad,
        )
        print(
            f"\nTrial {num_evals}. Time spent: {time.perf_counter() - total_time_t0}\n"
            f"Evaluated {variables}:\n"
            f"Results: {results}"
        )
        if trial is not None:
            tell_optuna_trial(study, trial, results)


def config_fingerprint(variables: Dict[str, Any]) -> str:
    parts = [str(variables[k]) for k in sorted(variables.keys())]
    return "|".join(parts)


def sample_configuration(
    configs_already_sampled: Set[str],
    variable_choices: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if variable_choices is None:
        variable_choices = VARIABLE_CHOICES
    for iter in range(MAX_SUGGESTION_TRIES):
        variables = {
            name: randchoice_torch(choices)
            for name, choices in variable_choices.items()
        }
        # Check for non-permitted combinations
        if (
            variables.get("kv_cache:cpu_offload")
            and variables.get("kv_cache:buffer_name") == "default"
        ):
            continue
        fingerprint = config_fingerprint(variables)
        if fingerprint in configs_already_sampled:
            if iter < MAX_SUGGESTION_TRIES - 1:
                continue
            # If the configuration space is small, this may happen
            print(
                "Did not manage to sample novel configuration. Will choose one which was already chosen before"
            )
        configs_already_sampled.add(fingerprint)
        return variables
    raise RuntimeError(
        f"Did not manage to sample valid and novel configuration after {MAX_SUGGESTION_TRIES} trials."
    )


def update_args(
    kv_cache: KVCacheArgs,
    grad: GradientArgs,
    variables: Dict[str, Any],
) -> Tuple[KVCacheArgs, GradientArgs]:
    extra_args = [
        {
            k[len(prefix) :]: v
            for k, v in variables.items()
            if k.startswith(prefix) and not k.endswith("buffer_name")
        }
        for prefix in ("kv_cache:", "grad:")
    ]
    name = "kv_cache:buffer_name"
    if name in variables:
        cache_name = split_name(kv_cache.name)[0]
        extra_args[0]["name"] = cache_name + "-" + variables[name]
    return tuple(
        dataclasses.replace(original, **kwargs)
        for original, kwargs in zip((kv_cache, grad), extra_args)
    )


def eval_autotune_metrics(
    fabric: L.Fabric,
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
    load_model_checkpoint(fabric, model, checkpoint_dir)
    check_for_nan_module_weights(model.gpt_model)

    # Create optimizer(s)
    cpu_optimizer = None
    cpu_scheduler = None
    gpu_optimizer = None
    gpu_scheduler = None
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
        gpu_scheduler = get_lr_scheduler(gpu_optimizer, train_args=train, max_steps=100)

    result_metrics: Dict[str, Any] = {
        "out_of_memory": False,
        "runtime_exception": False,
    }

    try:

        # Run some evaluation steps
        if do_cpu_offload:
            valid_model = model.copy_model_for_evaluation()
        else:
            valid_model = model
        # Warmup
        if num_warmup_steps > 0:
            print(f"Warm-up for validation: {num_warmup_steps} steps")
            validate(
                model=valid_model,
                val_dataloader=val_dataloader,
                eval=dataclasses.replace(eval, max_iters=num_warmup_steps),
                batch_transform=batch_transform,
            )
        # Validation (timed)
        timer_t0 = time.perf_counter()
        print(f"Validation: {num_valid_steps} steps")
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
        if do_cpu_offload:
            deallocate_kv_cache_buffers_of_model(valid_model.gpt_model)
            del valid_model

        # Run some training steps
        gc.collect()
        torch.cuda.empty_cache()
        sum_loss = 0
        time_train = 0
        print(f"Training: {num_train_steps} steps")
        for iter_num in range(num_train_steps):
            timer_t0 = time.perf_counter()
            batch = batch_transform(next(train_iterator))
            # Compute loss and gradients
            # Note: There is no accumulation of gradients between devices
            loss = model(
                input_ids=batch[INPUT_IDS_NAME],
                targets=batch["targets"],
                scale_factor=1.0,
            )
            loss.backward()
            sum_loss += loss.item()
            time_train += time.perf_counter() - timer_t0
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
        print(
            f"Training: train_loss = {(sum_loss / num_train_steps):.2f}, time = {time_train:.3f} secs"
        )

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
