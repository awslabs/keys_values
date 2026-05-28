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
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Dict, Optional, Union, Any, List, Tuple
import yaml

import lightning as L
import torch
from lightning.fabric.strategies import DDPStrategy

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    auto_download_checkpoint,
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    parse_devices,
    load_checkpoint,
)

from keys_values.attention.attention_utils import DEFAULT_TMP_ARRAY_LIMIT_GB
from keys_values.config import Config as ConfigFull
from keys_values.data import LongBenchV2, Helmet, INPUT_IDS_NAME
from keys_values.evaluation.evaluator import (
    SampleBasedMetricsEvaluator,
    TargetType,
)
from keys_values.evaluation.tasks import (
    EvaluationTasks,
    EvaluationWithTasksHelper,
)
from keys_values.data.evaluation import (
    EvaluationDataLoader,
)
from keys_values.data.constants import (
    ORIG_IDX_NAME,
    TASK_NAME,
    TARGETS_STRINGS_NAME,
    LIT_MODEL_FNAME,
    HEAD_MODEL_FNAME,
    LORA_WEIGHTS_FNAME,
    LORA_WEIGHTS_FNAME_OLD,
)
from keys_values.finetune.args import KVCacheArgs, SDPAArgs, EvalArgs
from keys_values.finetune.batch_transform import BatchTransformFactory
from keys_values.finetune.longcontext_full import (
    wrap_gpt_model,
    get_mha_and_cache_kwargs,
    create_gpt_model,
)
from keys_values.finetune.utils import (
    check_kv_cache,
    adapt_requires_grad,
    print_with_rank_and_timestamp,
    adjust_cache_kwargs,
    load_generation_config,
)
from keys_values.fused import (
    set_fused_swiglu_enabled,
    set_fused_rmsnorm_enabled,
)
from keys_values.head_model_factory import HeadModelFactory
from keys_values.long_context import LongContextInferenceModel
from keys_values.lora import Config as ConfigLoRA
from keys_values.pos_encoding import set_fused_rope_enabled
from keys_values.utils import (
    flush_io_streams,
    VerbosityLevels,
    fabric_precision_to_dtype,
    remove_keys,
)

GENERATED_SAMPLES_FILENAME = "generated_samples_{}.yaml"


@dataclass
class ConfigFull_OLD(ConfigFull):
    start_of_layer_hook: Optional[callable] = None


@dataclass
class ConfigLoRA_OLD(ConfigLoRA):
    start_of_layer_hook: Optional[callable] = None


@dataclass(frozen=True)
class ModelConfiguration:
    config: Union[ConfigFull, ConfigLoRA]
    head_model_name: str
    head_model_kwargs: Dict[str, Any]


def cleanup_longbench_v2_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return remove_keys(
        kwargs,
        {"num_workers", "include_multiturn_conversations"},
    )


def cleanup_kvcache_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return remove_keys(
        kwargs,
        {"layers_per_cell", "single_tokens_for_targets"},
    )


def setup(
    setups_filename: str,
    devices: Union[int, str] = 1,
    seed: int = 1337,
    access_token: Optional[str] = None,
    batch_size: Optional[int] = None,
    verbose: Optional[str] = None,
    attention_forward_temp_size_gb: Optional[float] = None,
    lora_dropout: Optional[float] = None,
    use_sample_metric: bool = True,
    sample_metric_max_generated_tokens: int = 20,
    sample_metric_temperature: Optional[float] = None,
    sample_metric_top_k: Optional[int] = None,
    sample_metric_top_p: Optional[float] = None,
    num_store_generated_samples: Optional[int] = None,
    skip_eval: bool = False,
) -> None:
    """Evaluate a range of checkpoints for several models on a test set

    This is an advanced version of `longcontext_eval.py`. Instead of a single
    model and dataset, for which evaluation is run over different checkpoints,
    we run a nested loop here:

    * Outer loop over setups: A setup is a tuple of base model and dataset
    * Inner loop over tasks and test set batches for each setup: A task is a
        checkpoint. Either run over all checkpoints or a selected list

    The setups and tasks are read from a jobs file, whose name is passed as
    first argument.

    Arguments:
        setups_filename: Name of YAML file describing the setups and tasks per
            setup.
        devices: How many devices/GPUs to use.
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
        batch_size: Size for test set batches. Only if you like to overwrite
            the configuration stored with the checkpoints
        verbose: Verbosity level for logging outputs. Only if you like to
            overwrite the configuration stored with the checkpoints
        attention_forward_temp_size_gb: Size of GPU memory buffers (in GB) used
            in naive SDPA. Only if you like to overwrite the configuration
            stored with the checkpoints
        lora_dropout: If given and `model_type == "lora"`, this overwrites the
            `config.lora_dropout` values. Pass 0 to switch dropout off
        use_sample_metric: If `True` and the dataset has an associated
            sample-based metric, this is used. Otherwise, we use the same loss
            as used for training
        sample_metric_max_generated_tokens: Maximum number of tokens sampled
            for sample-based metric evaluation
        sample_metric_temperature: Parameter for token generation. Overrides
            what comes with the checkpoint.
        sample_metric_top_k: Parameter for token generation. Overrides
            what comes with the checkpoint.
        sample_metric_top_p: Parameter for token generation. Overrides
            what comes with the checkpoint.
        num_store_generated_samples: If given and positive, we write files
            containing the generated sequences along with SFT targets and raw
            targets. These files are written alongside metric files, using the
            same naming convention. They are written for the initial test set
            batches, until `num_store_generated_samples` cases are covered
            (rounded up to a multiple of `batch_size`). Must have
            `use_sample_metric == True`.
        skip_eval: If `True`, we skip evaluations and only write files related
            to `num_store_generated_samples`.

    """
    devices = parse_devices(devices)
    if torch.cuda.is_available():
        if not (1 <= devices <= torch.cuda.device_count()):
            raise ValueError(
                f"devices = {devices}, must be in [1, {torch.cuda.device_count()}]"
            )
    elif devices != 1:
        raise ValueError("CUDA is not available, can only do devices = 1")
    sample_metric_kwargs = dict()
    if sample_metric_temperature is not None:
        sample_metric_kwargs["temperature"] = sample_metric_temperature
    if sample_metric_top_k is not None:
        sample_metric_kwargs["top_k"] = sample_metric_top_k
    if sample_metric_top_p is not None:
        sample_metric_kwargs["top_p"] = sample_metric_top_p

    pprint(locals())

    # Load setups file
    setups = yaml.safe_load(Path(setups_filename).open())
    print(f"Loaded setups from {setups_filename}:")
    print("\n".join(str(x) for x in setups))

    setup_internal(
        setups,
        devices,
        seed,
        access_token,
        batch_size,
        verbose,
        attention_forward_temp_size_gb,
        lora_dropout,
        use_sample_metric,
        sample_metric_max_generated_tokens,
        sample_metric_kwargs,
        num_store_generated_samples,
        skip_eval,
    )


def setup_internal(
    setups: List[Dict[str, Any]],
    devices: Union[int, str],
    seed: int,
    access_token: Optional[str],
    batch_size: Optional[int],
    verbose: Optional[str],
    attention_forward_temp_size_gb: Optional[float],
    lora_dropout: Optional[float],
    use_sample_metric: bool,
    sample_metric_max_generated_tokens: int,
    sample_metric_kwargs: Dict[str, Any],
    num_store_generated_samples: Optional[int],
    skip_eval: bool,
) -> None:
    if num_store_generated_samples is None and skip_eval:
        raise ValueError(
            "If skip_eval is True, num_store_generated_samples must be given"
        )
    if num_store_generated_samples is not None:
        if num_store_generated_samples <= 0:
            raise ValueError(
                f"num_store_generated_samples = {num_store_generated_samples}, must be positive"
            )
        if not use_sample_metric:
            raise ValueError(
                f"num_store_generated_samples can only be used if use_sample_metric is True"
            )
    # Need to obtain `precision` from hyperparameters of first setup
    out_dir = Path(setups[0]["out_dir"])
    model_type = setups[0]["model_type"]
    checkpoint_dir = setups[0].get("checkpoint_dir")
    if checkpoint_dir is None:
        tasks = setups[0].get("eval_tasks")
        eval = EvaluationTasks(out_dir, model_type, tasks)
        if not eval.tasks:
            raise ValueError(
                f"No completed model checkpoints detected at {out_dir}. Are you "
                f"sure that model_type = {model_type} is correct?"
            )
        task_path = out_dir / eval.tasks[0]
    else:
        task_path = Path(checkpoint_dir)
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

    fabric.launch(
        main,
        seed=seed,
        setups=setups,
        batch_size=batch_size,
        devices=devices,
        verbose=verbose,
        attention_forward_temp_size_gb=attention_forward_temp_size_gb,
        use_sample_metric=use_sample_metric,
        sample_metric_max_generated_tokens=sample_metric_max_generated_tokens,
        sample_metric_kwargs=sample_metric_kwargs,
        lora_dropout=lora_dropout,
        access_token=access_token,
        num_store_generated_samples=num_store_generated_samples,
        skip_eval=skip_eval,
    )


def main(
    fabric: L.Fabric,
    seed: int,
    setups: List[Dict[str, Any]],
    batch_size: int,
    devices: int,
    verbose: Optional[str],
    attention_forward_temp_size_gb: Optional[float],
    use_sample_metric: bool,
    sample_metric_max_generated_tokens,
    sample_metric_kwargs: Dict[str, Any],
    lora_dropout: Optional[float],
    access_token: Optional[str],
    num_store_generated_samples: Optional[int],
    skip_eval: bool,
) -> None:
    fabric.seed_everything(seed)

    # Loop over setups
    _batch_size = batch_size
    data_class_path = None
    data_init_args = None
    data = None
    for setup_no, _setup in enumerate(setups):
        prefix = f"Setup {setup_no}: "
        out_dir = Path(_setup["out_dir"])
        model_type = _setup["model_type"]
        checkpoint_dir = _setup.get("checkpoint_dir")
        kv_cache = _setup.get("kv_cache")
        sdpa = _setup.get("sdpa")
        print(f"\n{prefix}out_dir = {out_dir}, model_type = {model_type}")
        if checkpoint_dir is None:
            tasks = _setup.get("eval_tasks")
            eval_tasks = EvaluationTasks(out_dir, model_type, tasks)
            if not eval_tasks.tasks:
                raise ValueError(
                    f"{prefix}No completed model checkpoints detected at {out_dir}. Are you "
                    f"sure that model_type = {model_type} is correct?"
                )
            print(
                "Detected model checkpoints to evaluate from:\n" + str(eval_tasks.tasks)
            )
            task_path = out_dir / eval_tasks.tasks[0]
        else:
            checkpoint_dir = Path(checkpoint_dir)
            if not checkpoint_dir.exists():
                raise ValueError(
                    f"{prefix}No completed model checkpoint detected at {checkpoint_dir}."
                )
            eval_tasks = None
            task_path = checkpoint_dir

        # Configuration from first task (must be the same over all tasks)
        model_config, hyp_pars = load_configuration(
            task_path=task_path,
            model_type=model_type,
        )
        model_name = hyp_pars["checkpoint_dir"].split("/")[-1]
        if model_type == "lora" and lora_dropout is not None:
            if lora_dropout < 0:
                raise ValueError(f"lora_dropout {lora_dropout}, must be non-negative")
            if model_config.config.lora_dropout != lora_dropout:
                print(
                    f"Changing config.lora_dropout from {model_config.config.lora_dropout} to {lora_dropout}"
                )
            model_config.config.lora_dropout = lora_dropout
        # Base model checkpoint
        # - For LoRA, most model weights are loaded from there
        # - Tokenizer or generation params are loaded from there if they are
        #   not part of the checkpoint
        base_checkpoint_dir = auto_download_checkpoint(
            model_name=hyp_pars["checkpoint_dir"],
            access_token=access_token,
        )
        if checkpoint_dir is None:
            checkpoint_dir = base_checkpoint_dir
        if _batch_size is None:
            batch_size = hyp_pars["evals"]["micro_batch_size"]
            if batch_size is None:
                batch_size = 2
        else:
            batch_size = _batch_size
        if kv_cache is None:
            kv_cache = cleanup_kvcache_kwargs(hyp_pars["kv_cache"])
        kv_cache = KVCacheArgs(**kv_cache)
        if kv_cache.cache_kwargs is None:
            kv_cache.cache_kwargs = dict()
        check_kv_cache(kv_cache)
        if sdpa is None:
            if "sdpa" in hyp_pars:
                sdpa = hyp_pars["sdpa"]
            else:
                sdpa = dict(
                    flex_attention=True,
                    flex_extend_kv=False,
                )
        sdpa = SDPAArgs(**sdpa)
        if sdpa.flashinfer_attention:
            print(
                "FlashInfer SDPA not currently available for token generation: Setting sdpa.flashinfer_attention = False"
            )
            sdpa.flashinfer_attention = False
        if verbose is None:
            verbose = hyp_pars.get("verbose")
            if verbose is None:
                verbose = kv_cache.verbose
                if verbose is None:
                    verbose = VerbosityLevels.SOME.value
        verbose = VerbosityLevels(verbose)
        if attention_forward_temp_size_gb is None:
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
        check_valid_checkpoint_dir(checkpoint_dir)
        # If the checkpoint contains generation_config.json, load sample args.
        eval_args = load_generation_config(checkpoint_dir, EvalArgs())
        if (
            eval_tasks is None
            and not (checkpoint_dir / "generation_config.json").exists()
        ):
            # Load from base model checkpoint
            eval_args = load_generation_config(base_checkpoint_dir, EvalArgs())
        if eval_args.sample_metric_kwargs is not None:
            sample_metric_kwargs = {
                **eval_args.sample_metric_kwargs,
                **sample_metric_kwargs,
            }
        print(
            "Token generation parameters:\n"
            f"temperature: {sample_metric_kwargs['temperature']}\n"
            f"top_k:       {sample_metric_kwargs['top_k']}\n"
            f"top_p:       {sample_metric_kwargs['top_p']}\n"
        )

        # Dataset
        _data_class_path = hyp_pars["data"]["class_path"]
        _data_init_args = hyp_pars["data"]["init_args"]
        if _data_class_path == data_class_path and _data_init_args == data_init_args:
            print(
                f"data_class_path = {_data_class_path}, data_init_args = {_data_init_args}\nSame as previous setup."
            )
        else:
            if _data_class_path.endswith("data.LongBenchV2"):
                data = LongBenchV2(**cleanup_longbench_v2_kwargs(_data_init_args))
                if data.metadata_dir is None:
                    data.metadata_dir = str(out_dir / "data")
                    print(f"Setting LongBenchV2.metadata_dir to {data.metadata_dir}")
                if data.test_set_tag is None:
                    data.test_set_tag = "rest"
                    print(f"Setting LongBenchV2.test_set_tag to {data.test_set_tag}")
                if use_sample_metric:
                    print(
                        "LongBenchV2 does not support a sample-based metric. Switching to loss used during training"
                    )
            elif _data_class_path.endswith("data.Helmet"):
                data = Helmet(**_data_init_args)
                if data.metadata_dir is None:
                    data.metadata_dir = str(out_dir / "data")
                    print(f"Setting Helmet.metadata_dir to {data.metadata_dir}")
            else:
                raise ValueError(f"Data class path {_data_class_path} is not supported")
            data_class_path = _data_class_path
            data_init_args = _data_init_args

        # Enable/disable fused operators
        set_fused_rope_enabled(sdpa.fused_rope)
        set_fused_rmsnorm_enabled(sdpa.fused_rmsnorm)
        set_fused_swiglu_enabled(sdpa.fused_swiglu)

        # Create model
        if torch.cuda.is_available():
            device = torch.device("cuda", fabric.local_rank)
        else:
            device = torch.device("cpu")
        try:
            tokenizer = Tokenizer(checkpoint_dir)
        except Exception as ex:
            if eval_tasks is None:
                # Load tokenizer from base model checkpoint
                tokenizer = Tokenizer(base_checkpoint_dir)
            else:
                raise ex
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
                head_model = HeadModelFactory.create(
                    name=model_config.head_model_name,
                    config=model_config.config,
                    data=data,
                    **model_config.head_model_kwargs,
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
        file_path = base_checkpoint_dir / LIT_MODEL_FNAME
        load_checkpoint(fabric, model.gpt_model, file_path, strict=False)
        # If there are head model weights, load them as well. Otherwise, we use
        # random initialization (or the head model may not have weights)
        file_path = base_checkpoint_dir / HEAD_MODEL_FNAME
        if file_path.exists():
            load_checkpoint(fabric, model.head_model, file_path, strict=True)

        # Evaluation over tasks and batches
        # `num_store_generated_batches` is the number of batches for which
        # generated samples are written out
        if num_store_generated_samples is not None:
            num_store_generated_batches = (
                num_store_generated_samples // batch_size
                + int(num_store_generated_samples % batch_size > 0)
            )
            if devices > 1:
                num_store_generated_batches = (
                    num_store_generated_batches // devices
                    + int(fabric.local_rank < num_store_generated_batches % devices)
                )
        else:
            num_store_generated_batches = None
        eval_for_setup(
            fabric,
            model,
            data,
            tokenizer,
            out_dir,
            model_type,
            model_config,
            devices,
            batch_size,
            eval_tasks.tasks if eval_tasks is not None else None,
            use_sample_metric,
            sample_metric_max_generated_tokens,
            sample_metric_kwargs,
            num_store_generated_batches,
            skip_eval,
            model_name,
            checkpoint_dir if eval_tasks is None else None,
        )


def eval_for_setup(
    fabric: L.Fabric,
    model: LongContextInferenceModel,
    data: DataModule,
    tokenizer: Tokenizer,
    out_dir: Path,
    model_type: str,
    model_config: ModelConfiguration,
    devices: int,
    batch_size: int,
    eval_tasks: Optional[List[str]],
    use_sample_metric: bool,
    sample_metric_max_generated_tokens,
    sample_metric_kwargs: Dict[str, Any],
    num_store_generated_batches: Optional[int],
    skip_eval: bool,
    model_name: str,
    checkpoint_dir: Optional[Path],
) -> None:
    # Test dataloader is over cross product of test dataset batches and
    # evaluation tasks
    test_dataloader = get_dataloader(
        data=data,
        tokenizer=tokenizer,
        eval_tasks=eval_tasks,
        head_model=model_config.head_model_name,
        batch_size=batch_size,
        devices=devices,
        fabric=fabric,
        model_name=model_name,
    )
    ignore_index = getattr(data, "ignore_index", -100)

    if use_sample_metric:
        assert isinstance(data, Helmet)
        evaluator = SampleBasedMetricsEvaluator(
            metrics=[
                SampleBasedMetricsEvaluator.metric_for_helmet_task(data.dataset_key)
            ],
            max_generated_tokens=sample_metric_max_generated_tokens,
            tokenizer=tokenizer,
            sample_kwargs=sample_metric_kwargs,
        )
        print(f"Evaluation metric: {evaluator.metrics[0]}")
    else:
        evaluator = None

    # Loop over test set batches
    eval_for_setup_internal(
        fabric,
        model,
        data,
        test_dataloader,
        evaluator,
        tokenizer,
        out_dir,
        model_type,
        model_config,
        devices,
        checkpoint_dir,
        num_store_generated_batches=num_store_generated_batches,
        skip_eval=skip_eval,
        ignore_index=ignore_index,
    )


def eval_for_setup_internal(
    fabric: L.Fabric,
    model: LongContextInferenceModel,
    data: DataModule,
    test_dataloader: EvaluationDataLoader,
    evaluator: Optional[SampleBasedMetricsEvaluator],
    tokenizer: Tokenizer,
    out_dir: Path,
    model_type: str,
    model_config: ModelConfiguration,
    devices: int,
    checkpoint_dir: Optional[Path],
    num_store_generated_batches: Optional[int],
    skip_eval: bool,
    ignore_index: int = -100,
) -> None:
    multiple_tasks = checkpoint_dir is None
    # Loop over test set batches
    # Note: `test_dataloader` returns the same batches on each rank. We use
    # a file lock to assign a batch to the first rank asking for a batch.
    # Others skip any batch that is locked or already done.
    batch_transform = BatchTransformFactory.from_head_model(
        head_model=model_config.head_model_name,
        pad_id=0,
        eos_id=tokenizer.eos_id,
        ignore_index=ignore_index,
    )
    if hasattr(data, "test_set_tag"):
        tag = data.test_set_tag
    else:
        tag = None
    # Note: If `skip_eval == True`, we assume that the eval metrics files are
    # already present, and we use the generated samples files for locking
    if skip_eval:
        fname = "eval/" + GENERATED_SAMPLES_FILENAME
    else:
        fname = None
    tasks_helper = EvaluationWithTasksHelper(
        out_dir,
        tag=tag,
        eval_metrics_filename=fname,
        multiple_tasks=multiple_tasks,
    )
    current_task = None
    test_dataiter = iter(test_dataloader)
    if devices > 1:
        # Ensure that lock for first batch is not checked at exactly the same
        # time by all devices
        time.sleep(0.05 * fabric.global_rank)
    batch_idx = 0  # Batch counter per task
    skip_until_next_task = False
    for batch in test_dataiter:
        if not batch:
            print("Empty batch: Continue")
            continue
        if skip_until_next_task:
            if not multiple_tasks or batch[TASK_NAME] == current_task:
                continue
            skip_until_next_task = False
        store_generated_batch = (
            num_store_generated_batches is not None
            and batch_idx < num_store_generated_batches
        )
        task = batch[TASK_NAME] if multiple_tasks else "THE_ONLY_ONE_31415927"
        if skip_eval and not store_generated_batch and task == current_task:
            print(
                f"Wrote out generated samples for {num_store_generated_batches} batches: Skipping remaining ones for this task"
            )
            skip_until_next_task = True
            continue
        orig_idxs = batch[ORIG_IDX_NAME]
        eval_metrics_path = tasks_helper.get_lock(batch)
        batch_name = f"{task}, {orig_idxs}" if multiple_tasks else str(orig_idxs)
        if eval_metrics_path is None:
            print(f"Batch {batch_name} already done or in progress: Skipping")
            continue
        try:
            print_with_rank_and_timestamp(
                f"Running inference for batch {batch_name}",
                fabric.global_rank,
            )
            if getattr(test_dataloader, "delay_tokenization", False):
                # Tokenization only happens here
                batch = test_dataiter.fetch_full(batch)
            batch = batch_transform(batch)
            if task != current_task:
                if multiple_tasks:
                    task_path = out_dir / task
                    part = " " + task
                else:
                    task_path = checkpoint_dir
                    part = ""
                print(f"New task{part}: Load model checkpoint from {task_path}")
                load_model_checkpoint(
                    model=model,
                    task_path=task_path,
                    model_type=model_type,
                    fabric=fabric,
                )
                current_task = task
                batch_idx = 0  # Reset

            t0 = time.perf_counter()
            # One entry per batch dimension:
            input_ids = batch[INPUT_IDS_NAME]
            targets = batch["targets"]
            if evaluator is None:
                with torch.no_grad():
                    metric_values = model(input_ids, targets)
                metric_name = "eval_loss"
                generated_samples = None
                raw_targets = None
            else:
                metric_name = evaluator.metrics[0]
                prompt_len = input_ids.shape[1] - targets.shape[1] + 1
                prompts = input_ids[:, :prompt_len]
                raw_targets = batch[TARGETS_STRINGS_NAME]
                metric_values, generated_samples = evaluator(
                    model,
                    prompts,
                    raw_targets,
                    return_samples=store_generated_batch,
                )
                metric_values = metric_values[metric_name]
            eval_time = time.perf_counter() - t0
            print_with_rank_and_timestamp(
                f"Batch {batch_name}: {metric_name} = {metric_values.mean().item():.3f}, eval_time = {eval_time * 1000:.2f} ms",
                fabric.global_rank,
            )
            flush_io_streams()
            if not skip_eval:
                print(f"Storing to {eval_metrics_path}")
                store_eval_metrics(
                    metric_name,
                    metric_values,
                    batch,
                    eval_metrics_path,
                    multiple_tasks,
                )
            if store_generated_batch:
                if skip_eval:
                    result_path = eval_metrics_path
                else:
                    eval_fname = eval_metrics_path.stem
                    suffix = eval_fname.split("_")[-1]
                    result_path = (
                        eval_metrics_path.parent
                        / GENERATED_SAMPLES_FILENAME.format(suffix)
                    )
                print(f"Storing generated samples to {result_path}")
                store_generated_samples(
                    metric_name=metric_name,
                    metric_values=metric_values,
                    batch=batch,
                    generated_samples=generated_samples,
                    targets=targets,
                    raw_targets=raw_targets,
                    tokenizer=tokenizer,
                    result_path=result_path,
                    ignore_index=ignore_index,
                )
            # Only count a batch if it was not skipped:
            batch_idx += 1

        except Exception as ex:
            print("Caught exception during evaluation:\n" + str(ex))
            eval_metrics_path.unlink(missing_ok=True)
            raise ex


def get_dataloader(
    data: DataModule,
    tokenizer: Tokenizer,
    eval_tasks: Optional[List[str]],
    head_model: str,
    batch_size: int,
    devices: int,
    fabric: Optional[L.Fabric],
    model_name: Optional[str] = None,
) -> EvaluationDataLoader:
    """
    Creates data loader for cross product of test dataset with evaluation
    tasks (if `eval_tasks` is given). Each evaluation task corresponds to a
    model checkpoint written during or at the end of fine-tuning. See
    :class:`EvaluationTasks` and :class:`EvaluationDataLoader` for more details.

    If `eval_tasks is None`, the data loader is for the test set only, since
    evaluation is run for a single checkpoint.

    Args:
        data: LongBenchV2 dataset
        tokenizer: Tokenizer
        eval_tasks: List of evaluation tasks
        head_model: Head model name
        batch_size: Size of test batches
        devices: Number of devices to use
        fabric: Fabric
        model_name: Sent to `data.connect`

    Returns:
        Data loader for cross product of test dataset with evaluation tasks

    """
    num_devices = 1 if fabric is None else fabric.world_size
    data.connect(
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_devices=num_devices,
        rank=None if fabric is None else fabric.local_rank,
        head_model=head_model,
        test_batch_size=batch_size,
        eval_tasks=eval_tasks,
        model_name=model_name,
    )
    if fabric is not None:
        with fabric.rank_zero_first():
            data.prepare_data()
    data.setup()
    test_dataloader = data.test_dataloader(num_devices=devices)
    return test_dataloader


def load_configuration(
    task_path: Path,
    model_type: str,
) -> Tuple[ModelConfiguration, Dict[str, Any]]:
    # Load hyperparameters
    hyp_pars = yaml.safe_load((task_path / "hyperparameters.yaml").open())
    # Model config
    if model_type == "full":
        try:
            config = ConfigFull.from_file(task_path / "model_config.yaml")
        except TypeError:
            config = ConfigFull_OLD.from_file(task_path / "model_config.yaml")
    else:
        lora = hyp_pars.get("lora")
        if lora is None:
            raise ValueError(
                f"{task_path / 'hyperparameters.yaml'} does not contain 'lora':\n{hyp_pars}"
            )
        kwargs = dict(
            lora_r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            lora_query=lora["query"],
            lora_key=lora["key"],
            lora_value=lora["value"],
            lora_projection=lora["projection"],
            lora_mlp=lora["mlp"],
            lora_head=lora["head"],
        )
        try:
            config = ConfigLoRA.from_file(
                task_path / "model_config.yaml",
                **kwargs,
            )
        except TypeError:
            config = ConfigLoRA_OLD.from_file(
                task_path / "model_config.yaml",
                **kwargs,
            )
    # Head model
    head_model_name = hyp_pars["head_model"]
    head_model_kwargs = hyp_pars.get("head_model_kwargs", dict())
    return (
        ModelConfiguration(
            config=config,
            head_model_name=head_model_name,
            head_model_kwargs=head_model_kwargs,
        ),
        hyp_pars,
    )


def load_model_checkpoint(
    model: LongContextInferenceModel,
    task_path: Path,
    model_type: str,
    fabric: L.Fabric,
):
    if model_type == "full":
        file_path = task_path / LIT_MODEL_FNAME
        strict = True
    else:
        # LoRA: Stored params are only part of the whole. Leave all other
        # parameters the same
        file_path = task_path / LORA_WEIGHTS_FNAME
        if not file_path.exists():
            file_path = task_path / LORA_WEIGHTS_FNAME_OLD
        strict = False
    load_checkpoint(fabric, model.gpt_model, file_path, strict=strict)
    # If there are head model weights, load them as well. Otherwise, we use
    # random initialization (or the head model may not have weights)
    file_path = task_path / HEAD_MODEL_FNAME
    if file_path.exists():
        load_checkpoint(fabric, model.head_model, file_path, strict=True)


def store_eval_metrics(
    metric_name: str,
    metric_values: torch.Tensor,
    batch: dict[str, Any],
    eval_metrics_path: Path,
    multiple_tasks: bool = True,
):
    if multiple_tasks:
        fieldnames = ["idx", "task", metric_name]
        task = batch[TASK_NAME]
    else:
        fieldnames = ["idx", metric_name]
        task = None
    with eval_metrics_path.open("w") as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerow(fieldnames)
        for idx, loss in zip(batch[ORIG_IDX_NAME], metric_values):
            if multiple_tasks:
                row = [idx, task, loss.item()]
            else:
                row = [idx, loss.item()]
            writer.writerow(row)


def store_generated_samples(
    metric_name: str,
    metric_values: torch.Tensor,
    batch: dict[str, Any],
    generated_samples: List[str],
    targets: torch.Tensor,
    raw_targets: List[TargetType],
    tokenizer: Tokenizer,
    result_path: Path,
    ignore_index: int,
):
    entries = [
        {
            "idx": idx,
            metric_name: metric_val.item(),
            "output": output,
            "raw_target": raw_target,
            "sft_target": tokenizer.decode(target[target != ignore_index]),
        }
        for idx, metric_val, output, raw_target, target in zip(
            batch[ORIG_IDX_NAME],
            metric_values,
            generated_samples,
            raw_targets,
            targets,
        )
    ]
    with result_path.open("w") as fp:
        yaml.safe_dump(entries, fp)
