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
from dataclasses import dataclass
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, Literal, Optional, Union, Any, List, Tuple, Set

import lightning as L
import torch
from lightning.fabric.strategies import DDPStrategy
import yaml

from litgpt.data import DataModule
from litgpt.lora import mark_only_lora_as_trainable
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    auto_download_checkpoint,
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    parse_devices,
    load_checkpoint,
)

from keys_values.attention_utils import DEFAULT_TMP_ARRAY_LIMIT_GB
from keys_values.config import Config as ConfigFull
from keys_values.data import LongBenchV2, INPUT_IDS_NAME
from keys_values.data.base import (
    LIT_MODEL_FNAME,
    HEAD_MODEL_FNAME,
    LORA_WEIGHTS_FNAME,
    LORA_WEIGHTS_FNAME_OLD,
)
from keys_values.data.evaluation import (
    TASK_NAME,
    ORIG_IDX_NAME,
    EvaluationTasks,
    EvaluationWithTasksHelper,
    EvaluationDataLoader,
)
from keys_values.debug_utils import debug_store_or_compare_state, DebugIntermediates
from keys_values.finetune.args import KVCacheArgs, SDPAArgs
from keys_values.finetune.batch_transform import BatchTransformFactory
from keys_values.finetune.longcontext_full import (
    wrap_gpt_model,
    get_mha_and_cache_kwargs,
)
from keys_values.finetune.utils import (
    check_kv_cache,
    adapt_requires_grad,
    print_with_rank_and_timestamp,
)
from keys_values.head_model_factory import HeadModelFactory
from keys_values.long_context import (
    LongContextInferenceModel,
)
from keys_values.lora import GPT as GPTLoRA, Config as ConfigLoRA
from keys_values.model import GPT as GPTFull
from keys_values.utils import (
    flush_io_streams,
    VerbosityLevels,
    fabric_precision_to_dtype,
)


@dataclass(frozen=True)
class ModelConfiguration:
    config: Union[ConfigFull, ConfigLoRA]
    head_model_name: str
    head_model_kwargs: Dict[str, Any]


@dataclass
class ConfigFull_OLD(ConfigFull):
    start_of_layer_hook: Optional[callable] = None


@dataclass
class ConfigLoRA_OLD(ConfigLoRA):
    start_of_layer_hook: Optional[callable] = None


def remove_keys(
    kwargs: Dict[str, Any],
    names: Set[str],
) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k not in names}


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
    out_dir: Path,
    model_type: Literal["full", "lora"] = "lora",
    devices: Union[int, str] = 1,
    seed: int = 1337,
    access_token: Optional[str] = None,
    batch_size: Optional[int] = None,
    kv_cache: Optional[KVCacheArgs] = None,
    sdpa: Optional[SDPAArgs] = None,
    verbose: Optional[str] = None,
    attention_forward_temp_size_gb: Optional[float] = None,
) -> None:
    """Evaluate a range of model checkpoints on a test set

    The aim is to compute an evaluation metric on a test dataset on a number
    of checkpoints, typically those stored along a training run. Each such
    checkpoint is called a "task" here. We compute evaluation metric parts on
    batches, which is indexed by a test dataset batch and task. Each such
    batch gives rise to a result file. At the end, result files can be
    collected and the score values per task can be computed by reduction.

    This script can be run any number of times, and each run can use several
    devices. A run with multiple devices should behave the same as separate
    runs on each device. The different runs organize via file locks, so
    batches which are locked or already done, are simply skipped over. At
    present, all these runs must have access to the same file system, but this
    could be improved, e.g. by reading from S3 or using ECS.

    How things work:

    * Checkpoints are loaded starting from `out_dir`. We look for
        subdirectories "step-[0-9]{6}" and "final". If "final" is present,
        this becomes the first task. A task is represented by its path.
    * The test dataset is provided in the configuration (each checkpoint must
        have the same configuration). Batches of size `batch_size` are formed,
        by sorting sequences by tokenized length and starting from the
        shortest ones. We then iterate over dataset batches (inner) and tasks
        (outer).
    * We write result files for every batch, to
        `<task-path>/eval/eval/eval_metrics_<suffix>.csv`, see
        :class:`EvaluationWithTasksHelper`.

    Arguments:
        out_dir: Directory from where to load checkpoints. Checkpoints are
            looked for in subdirectories "step-[0-9]{6}" and "final".
        model_type: Either "full" or "lora".
        devices: How many devices/GPUs to use.
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
        batch_size: Size for test set batches. Only if you like to overwrite
            the configuration stored with the checkpoints
        kv_cache: Configuration for the KV caches. Only if you like to overwrite
            the configuration stored with the checkpoints
        sdpa: Configuration for SDPA kernel. Only if you like to overwrite the
            configuration stored with the checkpoints
        verbose: Verbosity level for logging outputs. Only if you like to
            overwrite the configuration stored with the checkpoints
        attention_forward_temp_size_gb: Size of GPU memory buffers (in GB) used
            in naive SDPA. Only if you like to overwrite the configuration
            stored with the checkpoints

    """
    devices = parse_devices(devices)
    if torch.cuda.is_available():
        if not (1 <= devices <= torch.cuda.device_count()):
            raise ValueError(
                f"devices = {devices}, must be in [1, {torch.cuda.device_count()}]"
            )
    elif devices != 1:
        raise ValueError("CUDA is not available, can only do devices = 1")
    # Collect evaluation tasks
    eval = EvaluationTasks(out_dir, model_type)
    if not eval.tasks:
        raise ValueError(
            f"No completed model checkpoints detected at {out_dir}. Are you "
            f"sure that model_type = {model_type} is correct?"
        )
    print("Detected model checkpoints to evaluate from:\n" + str(eval.tasks))
    # Configuration from first task (must be the same over all tasks)
    model_config, hyp_pars = load_configuration(
        task_path=out_dir / eval.tasks[0],
        model_type=model_type,
    )
    # Base model checkpoint
    checkpoint_dir = auto_download_checkpoint(
        model_name=hyp_pars["checkpoint_dir"],
        access_token=access_token,
    )
    pprint(locals())
    # Dataset
    if not hyp_pars["data"]["class_path"].endswith("data.LongBenchV2"):
        raise ValueError(
            f"Currently, this script supports --data LongBenchV2 only, but got {hyp_pars['data']['class_path']}"
        )
    data = LongBenchV2(**cleanup_longbench_v2_kwargs(hyp_pars["data"]["init_args"]))
    if data.metadata_dir is None:
        data.metadata_dir = str(out_dir / "data")
        print(f"Setting LongBenchV2.metadata_dir to {data.metadata_dir}")
    if data.test_set_tag is None:
        data.test_set_tag = "rest"
        print(f"Setting LongBenchV2.test_set_tag to {data.test_set_tag}")
    if batch_size is None:
        batch_size = hyp_pars["evals"]["micro_batch_size"]
        if batch_size is None:
            batch_size = 8
    if kv_cache is None:
        kv_cache = KVCacheArgs(**cleanup_kvcache_kwargs(hyp_pars["kv_cache"]))
    elif kv_cache.cache_kwargs is None:
        kv_cache.cache_kwargs = dict()
    if sdpa is None:
        if "sdpa" in hyp_pars:
            kwargs = hyp_pars["sdpa"]
        else:
            kwargs = dict(
                flex_attention=True,
                flex_extend_kv=False,
            )
        sdpa = SDPAArgs(**kwargs)
    check_kv_cache(kv_cache)
    check_valid_checkpoint_dir(checkpoint_dir)
    if verbose is None:
        verbose = hyp_pars.get("verbose")
        if verbose is None:
            verbose = kv_cache.verbose
            if verbose is None:
                verbose = VerbosityLevels.SOME.value
    verbose = VerbosityLevels(verbose)
    if attention_forward_temp_size_gb is None:
        attention_forward_temp_size_gb = hyp_pars.get("attention_forward_temp_size_gb")
        if attention_forward_temp_size_gb is None:
            attention_forward_temp_size_gb = kv_cache.attention_forward_temp_size_gb
            if attention_forward_temp_size_gb is None:
                attention_forward_temp_size_gb = DEFAULT_TMP_ARRAY_LIMIT_GB
    yarn_rope = hyp_pars.get("yarn_rope")
    if yarn_rope is None:
        yarn_rope = True

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
        data=data,
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        batch_size=batch_size,
        kv_cache=kv_cache,
        sdpa=sdpa,
        model_type=model_type,
        model_config=model_config,
        eval_tasks=eval.tasks,
        devices=devices,
        verbose=verbose,
        attention_forward_temp_size_gb=attention_forward_temp_size_gb,
        yarn_rope=yarn_rope,
    )


def main(
    fabric: L.Fabric,
    seed: int,
    data: DataModule,
    checkpoint_dir: Path,
    out_dir: Path,
    batch_size: int,
    kv_cache: KVCacheArgs,
    sdpa: SDPAArgs,
    model_type: str,
    model_config: ModelConfiguration,
    eval_tasks: List[str],
    devices: int,
    verbose: VerbosityLevels,
    attention_forward_temp_size_gb: float,
    yarn_rope: bool,
) -> None:
    is_lora = model_type == "lora"
    if torch.cuda.is_available():
        device = torch.device("cuda", fabric.local_rank)
    else:
        device = torch.device("cpu")
    tokenizer = Tokenizer(checkpoint_dir)
    # Test dataloader is over cross product of test dataset and evaluation
    # tasks
    test_dataloader = get_dataloader(
        data=data,
        tokenizer=tokenizer,
        eval_tasks=eval_tasks,
        head_model=model_config.head_model_name,
        batch_size=batch_size,
        devices=devices,
        fabric=fabric,
    )
    ignore_index = getattr(data, "ignore_index", -100)
    batch_transform = BatchTransformFactory.from_head_model(
        head_model=model_config.head_model_name,
        pad_id=0,
        eos_id=tokenizer.eos_id,
        ignore_index=ignore_index,
    )

    fabric.seed_everything(seed)  # same seed for every process to init model (FSDP)

    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        mha_kwargs = get_mha_and_cache_kwargs(
            attention_forward_temp_size_gb,
            model_config.config,
            kv_cache.cache_kwargs,
            sdpa,
            yarn_rope,
            fabric,
        )
        dtype = fabric_precision_to_dtype(fabric._precision.precision)
        torch.set_default_dtype(dtype)
        with torch.device(device):
            if not is_lora:
                gpt_model = GPTFull(model_config.config, **mha_kwargs)
            else:
                gpt_model = GPTLoRA(model_config.config, **mha_kwargs)
            head_model = HeadModelFactory.create(
                name=model_config.head_model_name,
                config=model_config.config,
                data=data,
                **model_config.head_model_kwargs,
            )
        if is_lora:
            mark_only_lora_as_trainable(gpt_model)
        adapt_requires_grad(gpt_model, head_model)
        # DEBUG:
        def debug_intermediates_predicate(
            kind, block_idx, start, end, rel_start, rel_end,
        ):
            return kind == "wte" or (
                kind == "block" and block_idx == 0 and start == 0
            )
        debug_intermediates = DebugIntermediates(
            predicate=debug_intermediates_predicate,
        )
        model = wrap_gpt_model(
            gpt_model=gpt_model,
            head_model=head_model,
            kv_cache=kv_cache,
            grad=None,
            verbose=verbose,
            attention_backward_temp_size_gb=None,
            max_batch_size=batch_size,
            dtype=dtype,
            fabric=fabric,
            model_kwargs=dict(debug_intermediates=debug_intermediates),  # DEBUG!
        )

    # Load base model
    file_path = checkpoint_dir / LIT_MODEL_FNAME
    load_checkpoint(fabric, model.gpt_model, file_path, strict=False)
    # If there are head model weights, load them as well. Otherwise, we use
    # random initialization (or the head model may not have weights)
    file_path = checkpoint_dir / HEAD_MODEL_FNAME
    if file_path.exists():
        load_checkpoint(fabric, model.head_model, file_path, strict=True)

    # Loop over test set batches
    # Note: `test_dataloader` returns the same batches on each rank. We use
    # a file lock to assign a batch to the first rank asking for a batch.
    # Others skip any batch that is locked or already done.
    tasks_helper = EvaluationWithTasksHelper(out_dir, data.test_set_tag)
    current_task = None
    test_dataiter = iter(test_dataloader)
    for batch in test_dataiter:
        if not batch:
            print("Empty batch: Continue")
            continue
        task = batch[TASK_NAME]
        orig_idxs = batch[ORIG_IDX_NAME]
        eval_metrics_path = tasks_helper.get_lock(batch)
        if eval_metrics_path is None:
            print(f"Batch {task}, {orig_idxs} already done or in progress: Skipping")
            continue
        try:
            print_with_rank_and_timestamp(
                f"Running inference for batch {task}, {orig_idxs}",
                fabric.global_rank,
            )
            if test_dataloader.delay_tokenization:
                # Tokenization only happens here
                batch = test_dataiter.fetch_full(batch)
            batch = batch_transform(batch)
            if task != current_task:
                task_path = out_dir / task
                print(f"New task {task}: Load model checkpoint from {task_path}")
                load_model_checkpoint(
                    model=model,
                    task_path=task_path,
                    model_type=model_type,
                    fabric=fabric,
                )
                current_task = task
            # DEBUG
            cpu_device = torch.device("cpu")
            gpt_state_dict = {
               k: v.to(cpu_device) for k, v in model.gpt_model.state_dict().items()
            }
            head_state_dict = {
               k: v.to(cpu_device) for k, v in model.head_model.state_dict().items()
            }
            # END DEBUG
            t0 = time.perf_counter()
            # One entry per batch dimension:
            with torch.no_grad():
                loss_values = model(batch[INPUT_IDS_NAME], batch["targets"])
            loss_value = loss_values.mean().item()
            eval_time = time.perf_counter() - t0
            print_with_rank_and_timestamp(
                f"Batch {task}, {orig_idxs}: loss = {loss_value:.3f}, "
                f"eval_time = {eval_time * 1000:.2f} ms",
                fabric.global_rank,
            )
            flush_io_streams()
            print(f"Storing to {eval_metrics_path}")
            store_eval_metrics(loss_values, batch, eval_metrics_path)
            # DEBUG
            if batch[ORIG_IDX_NAME][0] == 0 and model.debug_intermediates is not None:
               debug_intermediates = model.debug_intermediates.entries
            else:
               debug_intermediates = None
            debug_store_or_compare_state(
               batch,
               loss_values,
               gpt_state_dict,
               head_state_dict,
               eval_metrics_path,
               debug_intermediates=debug_intermediates,
            )
            # Stop after storing state including the one for [0,1,2,3]:
            # States with debug_intermediates are very large!
            if devices > 1 or batch[ORIG_IDX_NAME][0] == 0:
               print("DEBUG: Terminating!")
               exit(0)
            # END DEBUG
        except Exception as ex:
            print("Caught exception during evaluation:\n" + str(ex))
            eval_metrics_path.unlink(missing_ok=True)
            raise ex


def get_dataloader(
    data: DataModule,
    tokenizer: Tokenizer,
    eval_tasks: List[str],
    head_model: str,
    batch_size: int,
    devices: int,
    fabric: Optional[L.Fabric] = None,
) -> EvaluationDataLoader:
    """
    Creates data loader for cross product of test dataset with evaluation
    tasks. Each evaluation task corresponds to a model checkpoint written
    during or at the end of fine-tuning. See :class:`EvaluationTasks` and
    :class:`EvaluationDataLoader` for more details.

    Args:
        data: LongBenchV2 dataset
        tokenizer: Tokenizer
        eval_tasks: List of evaluation tasks
        head_model: Head model name
        batch_size: Size of test batches
        devices: Number of devices to use
        fabric: Fabric

    Returns:
        Data loader for cross product of test dataset with evaluation tasks

    """
    if not isinstance(data, LongBenchV2):
        raise ValueError("Only LongBenchV2 is currently supported")
    num_devices = 1 if fabric is None else fabric.world_size
    data.connect(
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_devices=num_devices,
        rank=None if fabric is None else fabric.local_rank,
        head_model=head_model,
        test_batch_size=batch_size,
        eval_tasks=eval_tasks,
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


def store_eval_metrics(
    loss_values: torch.Tensor,
    batch: dict[str, Any],
    eval_metrics_path: Path,
):
    fieldnames = ["idx", "task", "loss"]
    task = batch[TASK_NAME]
    with eval_metrics_path.open("w") as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerow(fieldnames)
        for idx, loss in zip(batch[ORIG_IDX_NAME], loss_values):
            writer.writerow([idx, task, loss.item()])
