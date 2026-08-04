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
from pathlib import Path
from pprint import pprint
import re
import time
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
    load_checkpoint,
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
from keys_values.data import Helmet, LongBenchV2, MyDataLoader, INPUT_IDS_NAME
from keys_values.data.constants import (
    ORIG_IDX_NAME,
    TASK_NAME,
    TARGETS_STRINGS_NAME,
    LIT_MODEL_FNAME,
    HEAD_MODEL_FNAME,
    LORA_WEIGHTS_FNAME,
    LORA_WEIGHTS_FNAME_OLD,
)
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
from keys_values.finetune.longcontext_eval_ext import (
    load_configuration,
    cleanup_kvcache_kwargs,
    cleanup_longbench_v2_kwargs,
    ModelConfiguration,
)
from keys_values.finetune.longcontext_full import (
    wrap_gpt_model,
    get_mha_and_cache_kwargs,
    create_gpt_model,
)
from keys_values.finetune.resume_state import (
    TrainingStateManager,
    load_training_state,
    restore_dataset_from_training_state,
    restore_from_training_state,
    TRAINSTATE_ITERATOR_FNAME,
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
    load_generation_config,
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


def get_checkpoint_path(
    out_dir: Path,
    index: int
) -> Path:
    return out_dir / ("final" if index == -1 else f"step-{index:06d}")


def get_checkpoints_to_evaluate(
    out_dir: Path,
    top_k: int,
    delta: int,
    include_final: bool = True,
) -> Tuple[List[int], List[Tuple[int, float]]]:
    """
    Determine checkpoint indexes for which to recompute validation losses. We
    extract all old loss values, identify the `top_k` argmins, and use all
    indexes at distance less or equal to `delta` from these. Here, "final" is
    represented by -1. Initial evaluation values are not used.

    """
    # Extract validation loss values from log files: As in
    # `keys_values/scripts/extract_valid_loss_from_logs.py`
    log_path = out_dir / "logs"

    # Collect log files: gpu0.log and resume{N}/gpu0.log for positive integer N
    log_files = []
    direct = log_path / "gpu0.log"
    if direct.exists():
        log_files.append(direct)
    for resume_dir in sorted(log_path.glob("resume*")):
        if resume_dir.is_dir():
            suffix = resume_dir.name[len("resume") :]
            if suffix.isdigit() and int(suffix) > 0:
                f = resume_dir / "gpu0.log"
                if f.exists():
                    log_files.append(f)

    if not log_files:
        print(f"{out_dir}: No logs")
        return [], []

    # Parse iter -> val_loss from all log files (last occurrence wins on collision)
    iter_pattern = re.compile(r"iter\s+(\d+)\s+\|.*val_loss:\s*([\d.]+)")
    final_pattern = re.compile(r"Final evaluation\s+\|.*val_loss:\s*([\d.]+)")
    records: Dict[int, float] = {}
    for log_file in log_files:
        with open(log_file) as fh:
            for line in fh:
                m = iter_pattern.search(line)
                if m:
                    records[int(m.group(1))] = float(m.group(2))
                    continue
                m = final_pattern.search(line)
                if m:
                    records[-1] = float(m.group(1))

    if not records:
        print(f"{out_dir}: No logs")
        return [], []

    pairs = sorted(records.items())
    if pairs[0][0] == -1:
        final_entry = pairs.pop(0)
    else:
        final_entry = None
    if len(pairs) == 1:
        raise ValueError(f"Found one val_loss entry only: {pairs}. Need at least 2")
    step = pairs[1][0] - pairs[0][0]
    max_index = pairs[-1][0]
    if final_entry is not None:
        final_index = max_index + step
        max_index = final_index
        pairs.append((final_index, final_entry[1]))
    else:
        final_index = None
    best_entries = sorted(pairs, key=lambda x: x[1])[:top_k]
    best_indexes = [x[0] for x in best_entries]
    new_indexes = {
        min(max(x + y * step, 0), max_index)
        for x in best_indexes
        for y in range(-delta, delta + 1)
    }
    result = [-1 if x == final_index else x for x in new_indexes]
    if include_final and final_entry is not None and -1 not in result:
        result.append(-1)
    best_entries = [(-1 if a == final_index else a, b) for a, b in best_entries]

    # Validate
    for index in result:
        path = get_checkpoint_path(out_dir, index)
        if not path.exists():
            raise FileNotFoundError(f"{path}: Checkpoint does not exist")
    return result, best_entries


def setup(
    out_dir: Path,
    top_k: int = 5,
    delta: int = 2,
    devices: Union[int, str] = 1,
    seed: int = 1337,
    access_token: Optional[str] = None,
    verbose: Optional[str] = None,
) -> None:
    """Recompute validation loss values for a certain number of checkpoints.

    When loading checkpoints, this script behaves like the evaluation script
    (loading configs and args from the checkpoints) and like the training
    script with `resume=True` (loading the training state for the train/valid
    split).

    Arguments:
        out_dir: Directory where checkpoints and logs have been saved during
            training, and where results are written to
        top_k: See :func:`get_checkpoints_to_evaluate`
        delta: See :func:`get_checkpoints_to_evaluate`
        devices: How many devices/GPUs to user
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
        verbose: Verbosity level for logging outputs.

    """
    # For LoRA without CPU offloading:
    setup_internal(
        setup,
        "lora",
        out_dir,
        top_k,
        delta,
        devices,
        seed,
        access_token,
        verbose,
    )


def setup_internal(
    original_setup: Callable,
    model_type: str,
    out_dir: Path,
    top_k: int,
    delta: int,
    devices: Union[int, str],
    seed: int,
    access_token: Optional[str],
    verbose: Optional[str],
) -> None:
    if not torch.cuda.is_available():
        raise ValueError("CUDA not available")
    pprint(locals())
    out_dir = init_out_dir(out_dir)
    devices = parse_devices(devices)
    if not (1 <= devices <= torch.cuda.device_count()):
        raise ValueError(
            f"devices = {devices}, must be in [1, {torch.cuda.device_count()}]"
        )
    verbose = VerbosityLevels(verbose)

    # Determine checkpoint indices where to compute validation loss
    checkpoint_indexes, old_topk_entries = get_checkpoints_to_evaluate(
        out_dir, top_k, delta,
    )
    if not checkpoint_indexes:
        raise ValueError(f"Did not find checkpoints for out_dir = {out_dir}")
    # Need to obtain `precision` from hyperparameters of first setup
    task_path = get_checkpoint_path(out_dir, checkpoint_indexes[0])
    _, hyp_pars = load_configuration(
        task_path=task_path,
        model_type=model_type,
    )
    precision = hyp_pars["precision"] or get_default_supported_precision(training=True)
    if devices > 1:
        strategy = DDPStrategy(static_graph=True, broadcast_buffers=False)
    else:
        strategy = "auto"
    fabric = L.Fabric(
        devices=devices,
        num_nodes=1,
        strategy=strategy,
        precision=precision,
    )
    if torch.cuda.is_available() and devices > 1:
        check_nvlink_connectivity(fabric)

    fabric.launch(
        main,
        original_setup=original_setup,
        model_type=model_type,
        devices=devices,
        checkpoint_indexes=checkpoint_indexes,
        old_topk_entries=old_topk_entries,
        seed=seed,
        out_dir=out_dir,
        verbose=verbose,
        access_token=access_token,
    )


def main(
    fabric: L.Fabric,
    original_setup: Callable,
    model_type: str,
    devices: int,
    checkpoint_indexes: List[int],
    old_topk_entries: List[Tuple[int, float]],
    seed: int,
    out_dir: Path,
    verbose: Optional[str],
    access_token: Optional[str],
) -> None:
    fabric.seed_everything(seed)
    is_lora = model_type == "lora"
    # Load configuration from first checkpoint (the same for all)
    task_path = get_checkpoint_path(out_dir, checkpoint_indexes[0])
    # Copied from `keys_values.finetune.longcontext_eval_ext.main`:
    model_config, hyp_pars = load_configuration(
        task_path=task_path,
        model_type=model_type,
    )
    model_name = hyp_pars["checkpoint_dir"].split("/")[-1]
    # Base model checkpoint
    # - For LoRA, most model weights are loaded from there
    # - Tokenizer or generation params are loaded from there if they are
    #   not part of the checkpoint
    checkpoint_dir = auto_download_checkpoint(
        model_name=hyp_pars["checkpoint_dir"],
        access_token=access_token,
    )
    check_valid_checkpoint_dir(checkpoint_dir)
    train = TrainArgs(**hyp_pars["train"])
    evals = EvalArgs(**hyp_pars["evals"])
    batch_size = evals.micro_batch_size
    if batch_size is None:
        batch_size = 2
    kv_cache = KVCacheArgs(**cleanup_kvcache_kwargs(hyp_pars["kv_cache"]))
    if kv_cache.cache_kwargs is None:
        kv_cache.cache_kwargs = dict()
    check_kv_cache(kv_cache)
    sdpa = hyp_pars.get(
        "sdpa",
        dict(
            flex_attention=True,
            flex_extend_kv=False,
        ),
    )
    sdpa = SDPAArgs(**sdpa)
    if verbose is None:
        verbose = hyp_pars.get("verbose")
        if verbose is None:
            verbose = kv_cache.verbose
            if verbose is None:
                verbose = VerbosityLevels.SOME.value
    verbose = VerbosityLevels(verbose)
    attention_forward_temp_size_gb = hyp_pars.get(
        "attention_forward_temp_size_gb"
    )
    if attention_forward_temp_size_gb is None:
        attention_forward_temp_size_gb = kv_cache.attention_forward_temp_size_gb
        if attention_forward_temp_size_gb is None:
            attention_forward_temp_size_gb = DEFAULT_TMP_ARRAY_LIMIT_GB
    yarn_rope = hyp_pars.get("yarn_rope")
    if yarn_rope is None:
        yarn_rope = True
    # If the checkpoint contains generation_config.json, load sample args.
    eval_args = load_generation_config(checkpoint_dir, EvalArgs())
    if not (checkpoint_dir / "generation_config.json").exists():
        # Load from base model checkpoint
        eval_args = load_generation_config(checkpoint_dir, EvalArgs())

    # Dataset
    data_class_path = hyp_pars["data"]["class_path"]
    data_init_args = hyp_pars["data"]["init_args"]
    if data_class_path.endswith("data.LongBenchV2"):
        data = LongBenchV2(**cleanup_longbench_v2_kwargs(data_init_args))
        if data.metadata_dir is None:
            data.metadata_dir = str(out_dir / "data")
            print(f"Setting LongBenchV2.metadata_dir to {data.metadata_dir}")
        if data.test_set_tag is None:
            data.test_set_tag = "rest"
            print(f"Setting LongBenchV2.test_set_tag to {data.test_set_tag}")
    elif data_class_path.endswith("data.Helmet"):
        data = Helmet(**data_init_args)
        if data.metadata_dir is None:
            data.metadata_dir = str(out_dir / "data")
            print(f"Setting Helmet.metadata_dir to {data.metadata_dir}")
    else:
        raise ValueError(f"Data class path {data_class_path} is not supported")

    # Enable/disable fused operators
    set_fused_rope_enabled(sdpa.fused_rope)
    set_fused_rmsnorm_enabled(sdpa.fused_rmsnorm)
    set_fused_swiglu_enabled(sdpa.fused_swiglu)

    # Create model
    if torch.cuda.is_available():
        device = torch.device("cuda", fabric.local_rank)
    else:
        device = torch.device("cpu")
    tokenizer = Tokenizer(checkpoint_dir)
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        # Updates `kv_cache.cache_kwargs` from other args:
        kv_cache = kv_cache.update_cache_kwargs()
        # Set `mha_kwargs`, update kv_cache.cache_kwargs` with that as well:
        mha_kwargs = get_mha_and_cache_kwargs(
            attention_forward_temp_size_gb,
            model_config.config,
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
        with torch.device(device):
            gpt_model = create_gpt_model(model_config.config, **mha_kwargs)
            if isinstance(data, LongBenchV2):
                extra_kwargs = dict(tokenizer=tokenizer)
            else:
                extra_kwargs = dict()
            head_model = HeadModelFactory.create(
                name=model_config.head_model_name,
                config=model_config.config,
                data=data,
                **model_config.head_model_kwargs,
                **extra_kwargs,
            )
        adapt_requires_grad(gpt_model, head_model)
        model, _ = wrap_gpt_model(
            gpt_model=gpt_model,
            head_model=head_model,
            kv_cache=kv_cache,
            grad=None,
            verbose=verbose,
            attention_backward_temp_size_gb=None,
            max_batch_size=batch_size,
            dtype=dtype,
            average_loss_per_batch=False,
            fabric=fabric,
        )
    # Load base model
    file_path = checkpoint_dir / LIT_MODEL_FNAME
    load_checkpoint(fabric, model.gpt_model, file_path, strict=False)
    # If there are head model weights, load them as well. Otherwise, we use
    # random initialization (or the head model may not have weights)
    file_path = checkpoint_dir / HEAD_MODEL_FNAME
    if file_path.exists():
        load_checkpoint(fabric, model.head_model, file_path, strict=True)

    eval_for_setup(
        fabric,
        checkpoint_indexes,
        old_topk_entries,
        model,
        data,
        train,
        evals,
        tokenizer,
        out_dir,
        model_type,
        model_config,
        devices,
        batch_size,
        model_name,
        checkpoint_dir,
    )


def eval_for_setup(
    fabric: L.Fabric,
    checkpoint_indexes: List[int],
    old_topk_entries: List[Tuple[int, float]],
    model: LongContextInferenceModel,
    data: DataModule,
    train: TrainArgs,
    evals: EvalArgs,
    tokenizer: Tokenizer,
    out_dir: Path,
    model_type: str,
    model_config: ModelConfiguration,
    devices: int,
    batch_size: int,
    model_name: str,
    checkpoint_dir: Optional[Path],
) -> None:
    print_message(
        f"\nIterating over {len(checkpoint_indexes)} checkpoints:" +
        "\n".join([get_checkpoint_path(out_dir, i) for i in checkpoint_indexes]),
        fabric,
    )

    new_records: Dict[int, float] = dict()
    for cp_ind in checkpoint_indexes:
        cp_path = get_checkpoint_path(out_dir, cp_ind)
        print_message(f"\nComputing validation loss for {cp_path}", fabric)
        data_train_state = restore_dataset_from_training_state(data, cp_path)
        _, val_dataloader = get_dataloaders(
            data=data,
            tokenizer=tokenizer,
            head_model=model_config.head_model_name,
            train=train,
            eval=evals,
            fabric=fabric,
            training_state=data_train_state,
        )
        ignore_index = getattr(data, "ignore_index", -100)
        batch_transform = BatchTransformFactory.from_head_model(
            head_model=model_config.head_model_name,
            pad_id=0,
            eos_id=tokenizer.eos_id,
            ignore_index=ignore_index,
        )



    # HIER!
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

    if training_state_num is not None:
        training_state = TrainingStateVars(
            manager=TrainingStateManager(
                state=state,
                dataset=data,
            ),
            files=[],
            training_state_num=training_state_num,
            devices=devices,
        )
    else:
        training_state = None

    load_model_checkpoint(fabric, model, checkpoint_dir, resume_dir=resume_path)
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
        training_state=training_state,
        resume_path=resume_path,
        record_gpu_memory_snapshots=record_gpu_memory_snapshots,
        record_gpu_memory_kind=record_gpu_memory_kind,
        record_gpu_memory_period=record_gpu_memory_period,
        generate_with_eval=generate_with_eval,
        profile_grad_params=profile_grad_params,
        size_log_quantiles=size_log_quantiles,
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
            evaluator=evaluator,
            val_dataloader=val_dataloader,
            eval=dataclasses.replace(eval, max_iters=len(val_dataloader)),
            batch_transform=batch_transform,
            log_metrics=False,
            generate_example_kwargs=generate_example_kwargs,
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
    if training_state is not None:
        training_state.save_state(fabric, save_dir)
    if fabric.global_rank == 0:
        # Copy checkpoint files from original checkpoint dir
        copy_config_files(checkpoint_dir, save_dir)
        save_hyperparameters(original_setup, save_dir)
        if hasattr(data, "prompt_style"):
            save_prompt_style(data.prompt_style, save_dir)


def fit(
    fabric: L.Fabric,
    original_setup: Callable,
    state: Dict[str, Any],
    train_dataloader: MyDataLoader,
    val_dataloader: MyDataLoader,
    batch_transform: BatchTransform,
    devices: int,
    checkpoint_dir: Path,
    out_dir: Path,
    train: TrainArgs,
    eval: EvalArgs,
    data: DataModule,
    evaluator: Optional[SampleBasedMetricsEvaluator],
    tokenizer: Tokenizer,
    training_state: Optional[TrainingStateVars],
    resume_path: Optional[Path],
    record_gpu_memory_snapshots: Optional[RecordGPUMemory],
    record_gpu_memory_kind: int,
    record_gpu_memory_period: int,
    generate_with_eval: bool,
    profile_grad_params: Optional[Dict[str, Any]],
    size_log_quantiles: List[float],
) -> Dict[str, Any]:
    do_cpu_offloading = "cpu_optimizer" in state
    model = state["model"]
    if not do_cpu_offloading:
        gpu_optimizer = state["optimizer"]
        gpu_scheduler = state["scheduler"]
        cpu_optimizer = None
        cpu_scheduler = None
        optim_device = fabric.device
        grad_reducer = CPUOffloadAccumulateGradients(
            group=list(range(devices)),
            fabric=fabric,
        )
    else:
        gpu_optimizer = state.get("gpu_optimizer")
        gpu_scheduler = state.get("gpu_scheduler")
        cpu_optimizer = state["cpu_optimizer"]
        cpu_scheduler = state["cpu_scheduler"]
        optim_device = torch.device("cpu")
        grad_reducer = None
    if evaluator is None:
        eval_metric_name = "val_loss"
    else:
        eval_metric_name = evaluator.metrics[0]

    try:

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
        if resume_path is None:
            if record_gpu_memory_kind == 3:
                path = out_dir / "gpu_memory_snapshots" / "snapshot_validation.pickle"
                record_gpu_memory_snapshots = RecordGPUMemory(
                    path=str(path),
                    max_entries=record_gpu_memory_snapshots.max_entries,
                    verbose=VerbosityLevels.MORE,
                )
                record_gpu_memory_snapshots.start_recording()

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
                    evaluator=evaluator,
                    val_dataloader=val_dataloader,
                    eval=dataclasses.replace(eval, max_iters=len(val_dataloader)),
                    batch_transform=batch_transform,
                    generate_example_kwargs=generate_example_kwargs,
                    fabric=fabric,
                )
                val_loss = metrics[eval_metric_name]
                print_message(
                    f"Initial evaluation          | "
                    + string_for_val_metrics(metrics, evaluator)
                    + f" | val_time: {metrics['val_time']:.3f} s",
                    fabric,
                )
            else:
                print_message("Verifying settings ...", fabric)
                with torch.no_grad():
                    if evaluator is None:
                        validate(
                            valid_model,
                            val_dataloader,
                            dataclasses.replace(eval, max_iters=1),
                            batch_transform,
                        )
                    else:
                        validate_sample_metric(
                            valid_model,
                            evaluator,
                            val_dataloader,
                            dataclasses.replace(eval, max_iters=1),
                            batch_transform,
                        )
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

        # Prepare start of training loop
        max_steps = train.max_steps or float("inf")
        train_iterator = CycleIterator(train_dataloader)
        if resume_path is not None:
            # Restore from training state
            print_message(
                f"Resume training: Loading training state from {resume_path}",
                fabric,
            )
            train_state = load_training_state(resume_path, fabric.global_rank)
            restore_from_training_state(
                state=state,
                train_iterator=train_iterator,
                train_state=train_state,
                rank=fabric.global_rank,
                num_devices=devices,
            )
            print_message(
                f"Resume training: Continue from epoch {train_iterator.epoch}, iteration {state['iter_num']}",
                fabric,
            )
        if training_state is not None:
            training_state.manager.init_train_iterator(train_iterator)
        throughput = ThroughputMonitor(fabric, window_size=50)
        if size_log_quantiles is not None and fabric.global_rank == 0:
            print_message(
                f"Logging size distributions for weights and gradients: quantiles = {size_log_quantiles}",
                fabric,
            )
            config = model.gpt_model.config
            mapper = None
            store_weights_rules = None
            store_grads_rules = None
            if not isinstance(config, ConfigLoRA):
                # Rules to split qkv variables into q, k, v. Only for full
                # fine-tuning
                hs = config.head_size
                query_size = config.n_head * hs
                key_size = config.n_query_groups * hs
                rules = [
                    SizeLogMapperRule(
                        postfix="qkv.weight",
                        sizes_names=(
                            (query_size, "q.weight"),
                            (key_size, "k.weight"),
                            (key_size, "v.weight"),
                        ),
                        dim=0,
                    ),
                    SizeLogMapperRule(
                        postfix="qkv.bias",
                        sizes_names=(
                            (query_size, "q.bias"),
                            (key_size, "k.bias"),
                            (key_size, "v.bias"),
                        ),
                        dim=0,
                    ),
                ]
                mapper = SizeLogMapper(rules=rules)
                # We also store weights and gradients for q.bias, k.bias,
                # reshaping these vectors into matrices
                do_store_weights = False
                if do_store_weights:
                    store_weights_rules = [
                        StoreWeightsRule(
                            match=get_match_for_store_rule("attn.k.bias"),
                            name="attn_k_bias",
                            shape=(config.n_query_groups, hs),
                            num_layers=config.n_layer,
                        ),
                        StoreWeightsRule(
                            match=get_match_for_store_rule("attn.q.bias"),
                            name="attn_q_bias",
                            shape=(config.n_head, hs),
                            num_layers=config.n_layer,
                        ),
                    ]
                    if config.n_embd % hs == 0:
                        shape_norm1 = (config.n_embd // hs, hs)
                    else:
                        shape_norm1 = (1, config.n_embd)
                    store_grads_rules = [
                        StoreWeightsRule(
                            match=get_match_for_store_rule("attn.v.bias"),
                            name="attn_v_bias",
                            shape=(config.n_query_groups, hs),
                            num_layers=config.n_layer,
                        ),
                        StoreWeightsRule(
                            match=get_match_for_store_rule("attn.v.weight"),
                            name="attn_v_weight",
                            shape=(key_size, config.n_embd),
                            num_layers=config.n_layer,
                        ),
                        StoreWeightsRule(
                            match=get_match_for_store_rule("norm_1.weight"),
                            name="norm_1_weight",
                            shape=shape_norm1,
                            num_layers=config.n_layer,
                        ),
                        StoreWeightsRule(
                            match=get_match_for_store_rule("attn.q.bias"),
                            name="attn_q_bias",
                            shape=(config.n_head, hs),
                            num_layers=config.n_layer,
                        ),
                    ]
            size_logs = SizeWeightsGradientsLog(
                quantiles=size_log_quantiles,
                path=out_dir,
                mapper=mapper,
                store_weights_rules=store_weights_rules,
                store_grads_rules=store_grads_rules,
            )
        else:
            size_logs = None

        running_loss = RunningMean(window=1, sync_on_compute=False).to(optim_device)
        fabric.barrier()
        total_lengths = 0
        gc.collect()
        torch.cuda.empty_cache()
        print_message(
            "\nGPU memory before training starts:\n" + message_memory_all_devices(),
            fabric,
        )
        total_t0 = time.perf_counter()

        while state["iter_num"] < max_steps:
            state["iter_num"] += 1
            iter_t0 = time.perf_counter()
            batch = batch_transform(next(train_iterator))
            if train_iterator.epoch >= train.epochs:
                break

            loss_weight = 1.0
            if train.average_loss_per_batch and devices > 1:
                # Cater for token-averaging of loss values and gradients
                num_tokens_batch = model.head_model.num_target_entries(batch["targets"])
                if num_tokens_batch is not None:
                    num_tokens_batch = num_tokens_batch.sum()
                    avg_tokens_tensor = num_tokens_batch.to(
                        device=fabric.device
                    ).clone()
                    fabric.all_reduce(avg_tokens_tensor, reduce_op="mean")
                    loss_weight = num_tokens_batch.item() / avg_tokens_tensor.item()

            if record_gpu_memory_snapshots is not None:
                run_no = state["iter_num"] - 1
                if record_gpu_memory_period >= 1:
                    run_no = run_no % record_gpu_memory_period
                if record_gpu_memory_kind == 0:
                    name = "snapshot.pickle"
                    path = (
                        out_dir / "gpu_memory_snapshots" / f"iteration{run_no}" / name
                    )
                    verbose = VerbosityLevels.MORE
                elif record_gpu_memory_kind == 1:
                    name = "snapshot_initial.pickle"
                    path = (
                        out_dir / "gpu_memory_snapshots" / f"iteration{run_no}" / name
                    )
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

            # DEBUG
            # Compute loss and gradient naively for the current batch, to compare
            # with what is done below. Works only for short enough sequences
            # assert devices == 1, "DEBUG only for single device"
            # debug_gradient, debug_loss = debug_compute_loss_and_gradient(
            #    gpt_model=model.gpt_model,
            #    batch=batch,
            #    device=fabric.device,
            #    average_loss_per_batch=train.average_loss_per_batch,
            # )
            # model.gpt_model.reset()
            # END DEBUG
            print_with_rank_and_timestamp(
                "Starting gradient computation.",
                fabric.global_rank,
            )

            # Compute loss and gradients
            # We do not use `fabric.backward`. For CPU offloading, loss and
            # gradient accumulation happens in `loss.backward` already. Otherwise,
            # we run an explicit all_reduce.
            loss = model(
                input_ids=batch[INPUT_IDS_NAME],
                targets=batch["targets"],
                scale_factor=loss_weight,
                record_gpu_memory_snapshots=record_gpu_memory_snapshots,
                record_gpu_memory_kind=(
                    record_gpu_memory_kind
                    if record_gpu_memory_snapshots is not None
                    else None
                ),
            )
            loss.backward()

            if not do_cpu_offloading:
                module_pairs = [(model.gpt_model, None)]
                if model.head_model.parameters():
                    module_pairs.append((model.head_model, None))
                grad_reducer(
                    module_pairs=module_pairs,
                    mean_reduction=True,
                )
                fabric.all_reduce(loss, reduce_op="mean")

            running_loss.update(loss.detach().to(device=optim_device))
            flush_io_streams()
            if size_logs is not None:
                size_logs(model.gpt_model)
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

            # DEBUG
            # Compare loss and gradient to naively computed ones
            # real_loss = loss.item()
            # real_gradient = debug_get_gradient(model.gpt_model)
            # print(f"real_loss = {real_loss}, debug_loss = {debug_loss}")
            # for name, real_grad in real_gradient.items():
            #    print(name)
            #    debug_grad = debug_gradient.get(name)
            #    if debug_grad is None:
            #        raise IndexError(f"{name} is in real_gradient, but not in debug_gradient")
            #    torch.testing.assert_close(real_grad, debug_grad)
            # END DEBUG

            if record_gpu_memory_snapshots is not None and record_gpu_memory_kind != 2:
                # Stop recording and store snapshot. For kind 0, this is the single
                # snapshot for the iteration. For kind 1, this is the final snapshot.
                record_gpu_memory_snapshots.store_current_snapshot()
                record_gpu_memory_snapshots.stop_recording()

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
            print_message("Optimizer update done.", fabric)
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
                    "epoch": train_iterator.epoch,
                    "iter_time": t1 - iter_t0,
                    "tokens": token_counts["raw_tokens_plus_prompt_template"],
                    "total_tokens": token_counts["raw_tokens_plus_prompt_template"]
                    * fabric.world_size,
                    "learning_rate": learning_rate,
                    **log_memory_all_devices(),
                }
                if not isinstance(val_loss, str):
                    val_loss = f"{val_loss:.3f}"
                print_message(
                    f"\nEpoch {metrics['epoch']} | iter {metrics['iter']:3d} |"
                    f" loss train: {metrics['loss']:.3f},"
                    f" {eval_metric_name} valid: {val_loss} |"
                    f" iter time: {metrics['iter_time']:.3f} s",
                    fabric,
                )
                fabric.log_dict(metrics, step=state["iter_num"])

            if state["iter_num"] % eval.interval == 0:
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
                    evaluator=evaluator,
                    val_dataloader=val_dataloader,
                    eval=eval,
                    batch_transform=batch_transform,
                    generate_example_kwargs=generate_example_kwargs,
                    log_metrics=False,
                    fabric=fabric,
                )
                val_loss = metrics[eval_metric_name]
                fabric.log_dict(metrics, step=state["iter_num"])
                print_with_rank_and_timestamp(
                    "Finished validation evaluations.",
                    fabric.global_rank,
                )
                print_message(
                    f"Epoch {train_iterator.epoch} | iter {state['iter_num']:3d}          | "
                    + string_for_val_metrics(metrics, evaluator)
                    + f" | val_time: {metrics['val_time']:.3f} s",
                    fabric,
                )
                flush_io_streams()
                if do_cpu_offloading:
                    deallocate_kv_cache_buffers_of_model(valid_model.gpt_model)
                    del valid_model
                fabric.barrier()

            save_checkpoint_regular(
                fabric=fabric,
                model=model,
                out_dir=out_dir,
                checkpoint_dir=checkpoint_dir,
                step=state["iter_num"],
                train=train,
                data=data,
                original_setup=original_setup,
                training_state=training_state,
            )

    except torch._dynamo.exc.FailOnRecompileLimitHit as ex:
        # This error is thrown by FlexAttention if too many graphs have been
        # compiled. We print all the graphs maintained, and how often each
        # has been used.
        print_flex_attn_report(fabric, model)
        raise ex

    return {
        key: fabric.all_reduce(token_counts[key], reduce_op="sum").item()
        for key in token_counts.keys()
    }


def print_flex_attn_report(
    fabric: L.Fabric,
    model: NaiveGPTAndHeadModel,
):
    flexatt_args = model.gpt_model.mha.flexatt_args
    if flexatt_args is not None:
        print_with_rank_and_timestamp(
            "\n" + flexatt_args.report(),
            fabric.global_rank,
        )


def validate_and_all_reduce(
    model: GPTAndHeadModel,
    evaluator: Optional[SampleBasedMetricsEvaluator],
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
        # `avg_loss` is the average metric or loss value over all cases, and
        # `num_entries` the number of cases.
        if evaluator is None:
            avg_loss, num_entries = validate(
                model,
                val_dataloader,
                eval,
                batch_transform,
            )
            metric_name = "val_loss"
        else:
            avg_loss, num_entries = validate_sample_metric(
                model,
                evaluator,
                val_dataloader,
                eval,
                batch_transform,
            )
            metric_name = evaluator.metrics[0]
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
        sum_num_entries_tensor = torch.tensor(
            num_entries,
            device=fabric.device,
            dtype=torch.int64,
        )
        fabric.all_reduce(sum_num_entries_tensor, reduce_op="sum")
        weight = num_entries / sum_num_entries_tensor.item()
        val_loss_tensor = torch.tensor(
            avg_loss * weight,
            device=fabric.device,
            dtype=torch.float32,
        )
        fabric.all_reduce(val_loss_tensor, reduce_op="sum")
        avg_loss = val_loss_tensor.item()
        val_time_tensor = torch.tensor(
            val_time,
            device=fabric.device,
            dtype=torch.float32,
        )
        fabric.all_reduce(val_time_tensor, reduce_op="mean")
        val_time = val_time_tensor.item()

    metrics = {
        metric_name: avg_loss,
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
) -> Tuple[float, int]:
    model.eval()
    sum_loss = 0.0
    num_entries = 0
    for k, batch in enumerate(val_dataloader):
        if k >= eval.max_iters:
            break
        batch = batch_transform(batch)
        num_entries += 1
        sum_loss += model(batch[INPUT_IDS_NAME], batch["targets"]).mean().item()
    model.train()
    return sum_loss / num_entries, num_entries


@torch.no_grad()
def validate_sample_metric(
    model: GPTAndHeadModel,
    evaluator: SampleBasedMetricsEvaluator,
    val_dataloader: MyDataLoader,
    eval: EvalArgs,
    batch_transform: BatchTransform,
) -> Tuple[float, int]:
    model.eval()
    sum_metric_values = 0.0
    num_entries = 0
    for k, batch in enumerate(val_dataloader):
        if k >= eval.max_iters:
            break
        batch = batch_transform(batch)
        input_ids = batch[INPUT_IDS_NAME]
        raw_targets = batch[TARGETS_STRINGS_NAME]
        prompt_len = input_ids.shape[1] - batch["targets"].shape[1] + 1
        prompts = input_ids[:, :prompt_len]
        metric_vals = evaluator(model, prompts, raw_targets)[0]
        sum_metric_values += metric_vals[evaluator.metrics[0]].sum().item()
        num_entries += 1
    model.train()
    return sum_metric_values / num_entries, num_entries


def string_for_val_metrics(
    metrics: Dict[str, float],
    evaluator: Optional[SampleBasedMetricsEvaluator],
) -> str:
    if evaluator is None:
        return f"val_loss: {metrics['val_loss']:.3f}"
    else:
        name = evaluator.metrics[0]
        return f"{name}: {metrics[name]:.3f}"


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

    max_returned_tokens = eval.max_new_tokens
    if max_returned_tokens is None:
        max_returned_tokens = 50
    max_returned_tokens += len(encoded)

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


def do_save(step: int, train: TrainArgs, intermed: bool) -> bool:
    interval = train.intermed_save_interval if intermed else train.save_interval
    return interval is not None and step % interval == 0


def save_checkpoint_regular(
    fabric: L.Fabric,
    model: GPTAndHeadModel,
    out_dir: Path,
    checkpoint_dir: Path,
    step: int,
    train: TrainArgs,
    data: DataModule,
    original_setup: Callable,
    training_state: Optional[TrainingStateVars],
):
    save_intermed = do_save(step, train, intermed=True)
    if save_intermed or do_save(step, train, intermed=False):
        interval_dir = out_dir / f"step-{step:06d}"
        save_model_checkpoint(fabric, model, interval_dir)
        if training_state is not None:
            training_state.save_state(fabric, interval_dir)
        if fabric.global_rank == 0:
            copy_config_files(checkpoint_dir, interval_dir)
            save_hyperparameters(original_setup, interval_dir)
            if hasattr(data, "prompt_style"):
                save_prompt_style(data.prompt_style, interval_dir)
    if save_intermed:
        # Check whether previous intermediate checkpoint has to be removed
        rem_step = step - train.intermed_save_num * train.intermed_save_interval
        if rem_step > 0 and not do_save(rem_step, train, intermed=False):
            interval_dir = out_dir / f"step-{rem_step:06d}"
            if interval_dir.exists():
                print_message(
                    f"Removing intermediate checkpoint {interval_dir}",
                    fabric,
                )
                for root, dirs, files in interval_dir.walk(top_down=True):
                    for name in files:
                        (root / name).unlink()
                    for name in dirs:
                        (root / name).rmdir()
                if interval_dir.exists():
                    interval_dir.rmdir()


# DEBUG: Code for comparison of gradients and loss value against naive


def debug_loss_function(
    logits: torch.Tensor,
    targets: torch.Tensor,
    average_loss_per_batch: bool,
    ignore_index: int = -100,
) -> torch.Tensor:
    assert logits.ndim == 3 and targets.ndim == 2
    assert logits.shape[:2] == targets.shape
    vocab_size = logits.shape[-1]
    num_target_entries = targets.ne(ignore_index).to(dtype=torch.float32).sum(dim=-1)
    if average_loss_per_batch:
        num_target_entries = num_target_entries.mean()
    num_targets = targets.shape[-1]
    losses = (
        torch.nn.functional.cross_entropy(
            logits[:, (-num_targets):, :].reshape(-1, vocab_size),
            targets.reshape(-1),
            ignore_index=ignore_index,
            reduction="none",
        )
        .view(*logits.shape[:2])
        .sum(dim=-1)
        .to(dtype=torch.float32)
    )
    return losses / num_target_entries.to(dtype=torch.float32)


def debug_get_gradient(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.grad.data.to(device=torch.device("cpu"))
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def debug_compute_loss_and_gradient(
    gpt_model: Union[GPTFull, GPTLoRA],
    batch: Dict[str, Any],
    device: torch.device,
    average_loss_per_batch: bool,
    ignore_index: int = -100,
) -> Tuple[Dict[str, torch.Tensor], float]:
    input_ids = batch[INPUT_IDS_NAME].to(device=device)
    targets = batch["targets"].to(device=device)
    gpt_model.reset()
    gpt_model.max_seq_length = input_ids.shape[1]
    logits = gpt_model(input_ids)
    loss_value = debug_loss_function(
        logits,
        targets,
        average_loss_per_batch,
        ignore_index,
    ).mean()
    loss_value.backward()
    gradient = debug_get_gradient(gpt_model)
    gpt_model.zero_grad(set_to_none=True)
    loss_value = loss_value.item()
    return gradient, loss_value
