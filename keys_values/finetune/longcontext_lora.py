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
import os
import time
import warnings
from pathlib import Path
from pprint import pprint
from typing import Dict, Literal, Optional, Union, Any

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.utilities import ThroughputMonitor
from lightning_utilities.core.imports import RequirementCache
from torch.utils.data import DataLoader
from torchmetrics import RunningMean
from torch.nn.attention import SDPBackend

from litgpt.args import TrainArgs
from litgpt.data import DataModule
from litgpt.lora import Config, mark_only_lora_as_trainable, lora_filter
from litgpt.prompts import save_prompt_style
from litgpt.scripts.merge_lora import merge_lora
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    CycleIterator,
    auto_download_checkpoint,
    check_nvlink_connectivity,
    check_valid_checkpoint_dir,
    copy_config_files,
    create_finetuning_performance_report,
    get_default_supported_precision,
    init_out_dir,
    instantiate_bnb_optimizer,
    instantiate_torch_optimizer,
    load_checkpoint,
    num_parameters,
    parse_devices,
)

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.attention_utils import DEFAULT_TMP_ARRAY_LIMIT_GB
from keys_values.data import LongBenchV2, INPUT_IDS_NAME
from keys_values.finetune.args import OptimizerArgs, LoRAARgs
from keys_values.finetune.batch_transform import (
    BatchTransformFactory, BatchTransform,
)
from keys_values.finetune.longcontext_full import (
    wrap_gpt_model,
    get_lr_scheduler,
    get_dataloaders,
    validate_args,
    check_kv_cache,
    validate,
    validate_and_all_reduce,
    print_with_rank_and_timestamp,
    flush_io_streams,
    debug_sum_gradient_norms,
    debug_random_trainable_param_names,
    debug_clone_selected_params,
    debug_compare_selected_params,
    DEBUG_NUM_SELECTED_PARAMS,
    choose_logger,
    EvalArgs,
)
from keys_values.finetune.utils import (
    LORA_WEIGHTS_FNAME,
    HEAD_MODEL_FNAME,
    LIT_MODEL_FNAME,
)
from keys_values.head_model import CrossEntropyOnLogits
from keys_values.head_model_factory import HeadModelFactory
from keys_values.kvcache.gradient.gpu_memory import RecordGPUMemory
from keys_values.kvcache.utils import (
    VerbosityLevels,
    message_memory_all_devices,
    log_memory_all_devices,
)
from keys_values.long_context import KVCacheArgs, GPTAndHeadModel
from keys_values.lora import GPT
from keys_values.parser_config import save_hyperparameters


DEFAULT_OUT_DIR = "out/finetune/longcontext_lora"


def setup(
    checkpoint_dir: Path,
    out_dir: Path = Path(DEFAULT_OUT_DIR),
    precision: Optional[str] = None,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"]] = None,
    devices: Union[int, str] = 1,
    num_nodes: int = 1,
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
    lora: LoRAARgs = LoRAARgs(
        r = 8,
        alpha = 16,
        dropout = 0.05,
        query = True,
        key = False,
        value = True,
        projection = False,
        mlp = False,
        head = False,
    ),
    eval: EvalArgs = EvalArgs(
        interval=100,
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
        single_tokens_for_targets=False,
        verbose=VerbosityLevels.SOME.value,
        attention_forward_temp_size_gb=4,
        attention_backward_temp_size_gb=2,
        use_new_cache=False,
    ),
    head_model: str = CrossEntropyOnLogits.NAME,
    head_model_kwargs: Optional[Dict[str, Any]] = None,
    record_gpu_memory_snapshots: Optional[int] = None,
    record_gpu_memory_kind: int = 0,
    record_gpu_memory_period: int = 0,
    debug_check_updates: bool = False,
    generate_with_eval: bool = False,
) -> None:
    """Finetune a model using the LoRA method.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to load for finetuning.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        quantize: If set, quantize the model with this algorithm. See ``tutorials/quantize.md`` for more information.
        devices: How many devices/GPUs to use.
        num_nodes: How many nodes the code is being run on.
        data: Data-related arguments. If not provided, the default is
            ``keys_values.data.LongBenchV2``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
            Note: We modified the defaults from `train.lr_warmup_steps=100` to
            `train.lr_warmup_fraction=0.15`, so the linear warm-up is the first
            15% of all steps.
        lora: Arguments for LoRA extension of model:
            - r: The LoRA rank.
            - alpha: The LoRA alpha.
            - dropout: The LoRA dropout value.
            - query: Whether to apply LoRA to the query weights in attention.
            - key: Whether to apply LoRA to the key weights in attention.
            - value: Whether to apply LoRA to the value weights in attention.
            - projection: Whether to apply LoRA to the output projection in the attention block.
            - mlp: Whether to apply LoRA to the weights of the MLP in the attention block.
            - head: Whether to apply LoRA to output head in GPT.
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
        head_model: Name of the head model to use, see
            :class:`HeadModelFactory`. Defaults to "next_token_prediction"
        head_model_kwargs: Extra keyword arguments to pass to the head model
            factory.
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
            - 2: Special case (DEBUG)
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
    logger = choose_logger(
        logger_name,
        out_dir,
        name=f"finetune-{config.name}",
        use_fabric=True,
        log_interval=train.log_interval,
    )

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        if RequirementCache("bitsandbytes != 0.42.0"):
            warnings.warn(
                "LitGPT only supports bitsandbytes v0.42.0. This may result in errors when using quantization."
            )
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    if devices * num_nodes > 1:
        if quantize:
            raise NotImplementedError(
                "Quantization is currently not supported for multi-GPU training. Please set devices=1 and num_nodes=1"
                " when using the --quantize flag."
            )
        # FSDP without cpu off load
        #strategy = FSDPStrategy(
        #    auto_wrap_policy={torch.nn.Linear},
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
        plugins=plugins,
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
        num_nodes=num_nodes,
        head_model_name=head_model,
        head_model_kwargs=head_model_kwargs,
        record_gpu_memory_snapshots=record_gpu_memory_snapshots,
        record_gpu_memory_kind=record_gpu_memory_kind,
        record_gpu_memory_period=record_gpu_memory_period,
        debug_check_updates=debug_check_updates,
        generate_with_eval=generate_with_eval,
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
    num_nodes: int,
    head_model_name: str,
    head_model_kwargs: Dict[str, Any],
    record_gpu_memory_snapshots: Optional[RecordGPUMemory],
    record_gpu_memory_kind: int,
    record_gpu_memory_period: int,
    debug_check_updates: bool,
    generate_with_eval: bool,
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
    fabric.print(f"Number of optimizer steps per epoch: {lr_max_steps}")

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
        mha_kwargs = {"sdpa_kernels": sdpa_kernels}
        if "sdpa_kernels" not in kv_cache.cache_kwargs:
            kv_cache.cache_kwargs["sdpa_kernels"] = sdpa_kernels
        limit_gb = kv_cache.attention_forward_temp_size_gb
        if limit_gb is None:
            limit_gb = DEFAULT_TMP_ARRAY_LIMIT_GB
        fabric.print(f"Setting limit attention_forward_temp_size_gb to {limit_gb} GB")
        tmp_array_limit_forward = TemporaryArrayLimit(
            init_val=limit_gb,
            name="attention_forward_temp_size_gb",
        )
        mha_kwargs["tmp_array_limit_gb"] = tmp_array_limit_forward
        kv_cache.cache_kwargs["tmp_array_limit_gb"] = tmp_array_limit_forward
        gpt_model = GPT(config, **mha_kwargs)
        if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
            from bitsandbytes.nn import StableEmbedding

            old_embedding = gpt_model.transformer.wte
            new_wte = StableEmbedding(
                old_embedding.num_embeddings, old_embedding.embedding_dim,
            )
            with torch.no_grad():
                new_wte.weight.copy_(old_embedding.weight)
            gpt_model.transformer.wte = new_wte.to(
                device=old_embedding.weight.device,
                dtype=old_embedding.weight.dtype,
            )
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
            fabric=fabric,
            gpt_model=gpt_model,
            head_model=head_model,
            kv_cache=kv_cache,
            max_batch_size=batch_size,
        )
    mark_only_lora_as_trainable(model.gpt_model)

    num_trainable_params = num_parameters(model, requires_grad=True)
    fabric.print(f"\nNumber of trainable parameters: {num_trainable_params:,}")
    fabric.print(f"Number of non-trainable parameters: {num_parameters(model, requires_grad=False):,}")

    model = fabric.setup_module(model)

    if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
        optimizer = instantiate_bnb_optimizer(optimizer.name, model.parameters())
    else:
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

    # strict=False because missing keys due to LoRA weights not contained in state dict
    file_path = checkpoint_dir / LIT_MODEL_FNAME
    load_checkpoint(fabric, model.gpt_model, file_path, strict=False)
    # If there are head model weights, load them as well. Otherwise, we use
    # random initialization (or the head model may not have weights)
    file_path = checkpoint_dir / HEAD_MODEL_FNAME
    if file_path.exists():
        load_checkpoint(fabric, model.head_model, file_path, strict=True)

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
        num_nodes=num_nodes,
        record_gpu_memory_snapshots=record_gpu_memory_snapshots,
        record_gpu_memory_kind=record_gpu_memory_kind,
        record_gpu_memory_period=record_gpu_memory_period,
        debug_check_updates=debug_check_updates,
        num_trainable_params=num_trainable_params,
        generate_with_eval=generate_with_eval,
    )
    training_time = time.perf_counter() - train_time
    output = create_finetuning_performance_report(training_time, token_counts, fabric.device.type)
    fabric.print(output)

    # Final evaluation
    if eval.final_validation:
        print_with_rank_and_timestamp("Starting validation evaluations.", fabric.global_rank)
        fabric.print(f"\nFinal validation evaluation (batch_size = {val_dataloader.batch_size}) ...")
        if generate_with_eval:
            generate_example_kwargs = dict(
                tokenizer=tokenizer,
                data=data,
            )
        else:
            generate_example_kwargs = None
        metrics = validate_and_all_reduce(
            fabric=fabric,
            model=model,
            val_dataloader=val_dataloader,
            eval=dataclasses.replace(eval, max_iters=len(val_dataloader)),
            batch_transform=batch_transform,
            log_metrics=False,
            generate_example_kwargs=generate_example_kwargs,
        )
        fabric.log_dict(metrics, step=state["iter_num"])
        fabric.print(
            f"Final evaluation | val loss: {metrics['val_loss']:.3f} | val ppl: {metrics['val_ppl']:.3f} | val_time_in_ms: {metrics['val_time_in_ms']:.3f}"
        )
        flush_io_streams()

    # Save the final LoRA checkpoint at the end of training
    save_dir = out_dir / "final"
    save_lora_checkpoint(fabric, model, save_dir)
    if fabric.global_rank == 0:
        # Copy checkpoint files from original checkpoint dir
        copy_config_files(checkpoint_dir, save_dir)
        save_hyperparameters(setup, save_dir)
        if hasattr(data, "prompt_style"):
            save_prompt_style(data.prompt_style, save_dir)
        merge_lora(
            checkpoint_dir=save_dir,
            lora_fname=LORA_WEIGHTS_FNAME,
            pretrained_fname=LIT_MODEL_FNAME,
        )


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
    num_nodes: int,
    record_gpu_memory_snapshots: Optional[RecordGPUMemory],
    record_gpu_memory_kind: int,
    record_gpu_memory_period: int,
    debug_check_updates: bool,
    num_trainable_params: int,
    generate_with_eval: bool,
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
        fabric.print(f"\nInitial validation evaluation  (batch_size = {val_dataloader.batch_size}) ...")
        if generate_with_eval:
            generate_example_kwargs = dict(
                tokenizer=tokenizer,
                data=data,
            )
        else:
            generate_example_kwargs = None
        metrics = validate_and_all_reduce(
            fabric=fabric,
            model=model,
            val_dataloader=val_dataloader,
            eval=dataclasses.replace(eval, max_iters=len(val_dataloader)),
            batch_transform=batch_transform,
            generate_example_kwargs=generate_example_kwargs,
        )
        val_loss = f"{metrics['val_loss']:.3f}"
        fabric.print(
            f"Initial evaluation | val loss: {val_loss} | val ppl: {metrics['val_ppl']:.3f} | val_time_in_ms: {metrics['val_time_in_ms']:.3f}"
        )
        flush_io_streams()
    else:
        # Note: Even if `generate_with_eval == True`, we don't generate here
        fabric.print("Verifying settings ...")
        with torch.no_grad():
            validate(
                model,
                val_dataloader,
                dataclasses.replace(eval, max_iters=1),
                batch_transform,
            )  # sanity check

    max_steps = train.max_steps or float("inf")
    train_iterator = CycleIterator(train_dataloader)
    throughput = ThroughputMonitor(fabric, window_size=50)
    running_loss = RunningMean(window=train.gradient_accumulation_iters(devices, num_nodes), sync_on_compute=False).to(
        fabric.device
    )
    total_lengths = 0
    fabric.print(
        "\nGPU memory before training starts:\n" + message_memory_all_devices()
    )
    total_t0 = time.perf_counter()

    while state["step_count"] < max_steps:
        state["iter_num"] += 1
        iter_t0 = time.perf_counter()
        batch = batch_transform(next(train_iterator))
        if train_iterator.epoch >= train.epochs:
            break

        if record_gpu_memory_snapshots is not None:
            if not (0 <= record_gpu_memory_kind <= 2):
                raise ValueError(f"record_gpu_memory_kind = {record_gpu_memory_kind} is not valid")
            run_no = state['iter_num'] - 1
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
            if record_gpu_memory_kind != 2:
                record_gpu_memory_snapshots.start_recording()

        is_accumulating = state["iter_num"] % train.gradient_accumulation_iters(devices, num_nodes) != 0
        print_with_rank_and_timestamp("Starting gradient computation.", fabric.global_rank)
        if debug_check_updates:
            debug_names = debug_random_trainable_param_names(
                model, DEBUG_NUM_SELECTED_PARAMS, num_trainable_params,
            )
            debug_orig_params = debug_clone_selected_params(model, debug_names)
        time_grad_t0 = time.perf_counter()
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            loss = model(
                input_ids=batch[INPUT_IDS_NAME],
                targets=batch["targets"],
                scale_factor=1.0 / train.gradient_accumulation_iters(devices, num_nodes),
                record_gpu_memory_snapshots=None if record_gpu_memory_kind == 0 else record_gpu_memory_snapshots,
                record_gpu_memory_kind=None if record_gpu_memory_kind == 0 else record_gpu_memory_kind - 1,
            )
            fabric.backward(loss)

        running_loss.update(loss.detach())
        time_grad_in_secs = (time.perf_counter() - time_grad_t0) * 1000000
        print_with_rank_and_timestamp(f"Finished gradient computation [{time_grad_in_secs:.2} secs]", fabric.global_rank)
        flush_io_streams()

        if record_gpu_memory_snapshots is not None and record_gpu_memory_kind != 2:
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
            fabric.print("Optimizer update done.")
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
        fabric.print(
            f"\nGPU memory at training step {state['iter_num'] - 1}:\n"
            + message_memory_all_devices() + "\n"
        )

        token_counts["raw_tokens"] += batch["token_counts"]["raw"].sum().item()
        token_counts["raw_tokens_plus_prompt_template"] += (
            batch["token_counts"]["raw_plus_prompt_template"].sum().item()
        )
        num_tokens = batch[INPUT_IDS_NAME].numel()
        token_counts["raw_tokens_plus_prompt_template_and_padding"] += num_tokens

        total_lengths += num_tokens
        if state["iter_num"] % train.log_interval == 0:
            loss = running_loss.compute().item()  # expensive device-to-host synchronization
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
                "total_tokens": token_counts["raw_tokens_plus_prompt_template"] * fabric.world_size,
                "learning_rate": scheduler.get_last_lr()[0],
                **log_memory_all_devices(),
            }
            if isinstance(val_loss, torch.Tensor):
                val_loss = f"{val_loss:.3f}"
            fabric.print(
                f"Epoch {metrics['epoch']} | iter {metrics['iter']} step {metrics['step']} |"
                f" loss train: {metrics['loss']:.3f},"
                f" val: {val_loss} |"
                f" iter time: {metrics['iter_time'] * 1000:.2f} ms"
                f"{' (step)' if not is_accumulating else ''}"
            )
            fabric.log_dict(metrics, step=state["iter_num"])

        if not is_accumulating and state["step_count"] % eval.interval == 0:
            print_with_rank_and_timestamp("Starting validation evaluations.", fabric.global_rank)
            fabric.print(f"\nPeriodic validation evaluation  (batch_size = {val_dataloader.batch_size}) ...")
            if generate_with_eval:
                generate_example_kwargs = dict(
                    tokenizer=tokenizer,
                    data=data,
                )
            else:
                generate_example_kwargs = None
            # TODO: Fix bug in generation!
            metrics = validate_and_all_reduce(
                fabric=fabric,
                model=model,
                val_dataloader=val_dataloader,
                eval=eval,
                batch_transform=batch_transform,
                generate_example_kwargs=generate_example_kwargs,
                log_metrics=False,
            )
            fabric.log_dict(metrics, step=state["iter_num"])
            print_with_rank_and_timestamp("Finished validation evaluations.", fabric.global_rank)
            flush_io_streams()
            val_loss = f"{metrics['val_loss']:.3f}"
            fabric.print(
                f"Epoch {train_iterator.epoch} | iter {state['iter_num']} | val loss: {val_loss} | val ppl: {metrics['val_ppl']:.3f} | val_time_in_ms: {metrics['val_time_in_ms']:.3f}"
            )
            fabric.barrier()

        if train.save_interval is not None and not is_accumulating and state["step_count"] % train.save_interval == 0:
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


def save_lora_checkpoint(
    fabric: L.Fabric,
    model: GPTAndHeadModel,
    file_dir: Path,
) -> None:
    file_dir.mkdir(parents=True, exist_ok=True)
    file_path = file_dir / LORA_WEIGHTS_FNAME
    fabric.print(f"\nSaving LoRA weights to {str(file_path)!r}")
    fabric.save(
        file_path,
        state={"model": model.gpt_model},
        filter={"model": lora_filter},
    )
    if model.head_model.state_dict():
        file_path = file_dir / HEAD_MODEL_FNAME
        fabric.print(f"Saving head model weights to {str(file_path)!r}")
        fabric.save(file_path, state={"model": model.head_model})


