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
from pathlib import Path
from pprint import pprint
import re
from typing import Dict, Optional, Union, Tuple, List
import yaml

import lightning as L
from lightning.fabric.strategies import DDPStrategy
import torch

from litgpt.data import DataModule
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    auto_download_checkpoint,
    load_checkpoint,
    check_nvlink_connectivity,
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    init_out_dir,
    parse_devices,
)

from keys_values.attention.attention_utils import DEFAULT_TMP_ARRAY_LIMIT_GB
from keys_values.data import Helmet, LongBenchV2
from keys_values.data.constants import (
    LIT_MODEL_FNAME,
    HEAD_MODEL_FNAME,
)
from keys_values.evaluation.longcontext_eval_ext import (
    load_configuration,
    cleanup_kvcache_kwargs,
    cleanup_longbench_v2_kwargs,
    ModelConfiguration,
    load_model_checkpoint,
)
from keys_values.finetune.args import (
    TrainArgs,
    EvalArgs,
    KVCacheArgs,
    SDPAArgs,
)
from keys_values.finetune.batch_transform import BatchTransformFactory
from keys_values.finetune.longcontext_full import (
    wrap_gpt_model,
    get_mha_and_cache_kwargs,
    create_gpt_model,
    validate_and_all_reduce,
    string_for_val_metrics,
)
from keys_values.finetune.resume_state import restore_dataset_from_training_state
from keys_values.finetune.utils import (
    get_dataloaders,
    adapt_requires_grad,
    print_message,
    check_kv_cache,
    adjust_cache_kwargs,
)
from keys_values.fused import (
    set_fused_swiglu_enabled,
    set_fused_rmsnorm_enabled,
)
from keys_values.head_model_factory import HeadModelFactory
from keys_values.long_context import LongContextInferenceModel
from keys_values.pos_encoding import set_fused_rope_enabled
from keys_values.utils import (
    flush_io_streams,
    VerbosityLevels,
    fabric_precision_to_dtype,
)

RESULT_FILENAME = "recomp_val_losses/eval_record.yaml"


def get_checkpoint_path(out_dir: Path, index: int) -> Path:
    return out_dir / ("final" if index == -1 else f"step-{index:06d}")


def get_checkpoints_to_evaluate(
    out_dir: Path,
    top_k: int,
    delta: int,
    include_final: bool = True,
) -> Tuple[List[int], List[Tuple[int, float]], int]:
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
    if max_index == final_index:
        max_index = -1

    # Validate
    for index in result:
        path = get_checkpoint_path(out_dir, index)
        if not path.exists():
            raise FileNotFoundError(f"{path}: Checkpoint does not exist")
    return result, best_entries, max_index


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

    # Determine checkpoint indices where to compute validation loss
    if not out_dir.exists():
        raise ValueError(f"{out_dir}: Directory does not exist")
    checkpoint_indexes, old_topk_entries, final_cp_index = get_checkpoints_to_evaluate(
        out_dir,
        top_k,
        delta,
    )
    if not checkpoint_indexes:
        raise ValueError(f"Did not find checkpoints for out_dir = {out_dir}")
    print(
        f"Best {top_k} iterations for old validation loss code:\n"
        + "\n".join(
            [
                f"{get_checkpoint_path(out_dir, ind).stem}: {val:.3f}"
                for ind, val in old_topk_entries
            ]
        ),
    )

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
        model_type=model_type,
        devices=devices,
        checkpoint_indexes=checkpoint_indexes,
        old_topk_entries=old_topk_entries,
        final_cp_index=final_cp_index,
        seed=seed,
        out_dir=out_dir,
        verbose=verbose,
        access_token=access_token,
    )


def main(
    fabric: L.Fabric,
    model_type: str,
    devices: int,
    checkpoint_indexes: List[int],
    old_topk_entries: List[Tuple[int, float]],
    final_cp_index: int,
    seed: int,
    out_dir: Path,
    verbose: Optional[str],
    access_token: Optional[str],
) -> None:
    fabric.seed_everything(seed)
    # Load configuration from first checkpoint (the same for all)
    task_path = get_checkpoint_path(out_dir, checkpoint_indexes[0])
    # Copied from `keys_values.finetune.longcontext_eval_ext.main`:
    model_config, hyp_pars = load_configuration(
        task_path=task_path,
        model_type=model_type,
    )
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
    evals = EvalArgs(**hyp_pars["eval"])
    batch_size = evals.micro_batch_size
    if batch_size is None:
        batch_size = 2
    kv_cache = KVCacheArgs(**cleanup_kvcache_kwargs(hyp_pars["kv_cache"]))
    if kv_cache.cache_kwargs is None:
        kv_cache.cache_kwargs = dict()
    check_kv_cache(kv_cache)
    sdpa = SDPAArgs(
        **hyp_pars.get(
            "sdpa",
            dict(
                flex_attention=True,
                flex_extend_kv=False,
            ),
        )
    )
    if verbose is None:
        verbose = hyp_pars.get("verbose")
        if verbose is None:
            verbose = kv_cache.verbose
            if verbose is None:
                verbose = VerbosityLevels.SOME.value
    verbose = VerbosityLevels(verbose)
    attention_forward_temp_size_gb = hyp_pars.get("attention_forward_temp_size_gb")
    if attention_forward_temp_size_gb is None:
        attention_forward_temp_size_gb = kv_cache.attention_forward_temp_size_gb
        if attention_forward_temp_size_gb is None:
            attention_forward_temp_size_gb = DEFAULT_TMP_ARRAY_LIMIT_GB
    yarn_rope = hyp_pars.get("yarn_rope")
    if yarn_rope is None:
        yarn_rope = True

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
        final_cp_index,
        model,
        data,
        train,
        evals,
        tokenizer,
        out_dir,
        model_type,
        model_config,
    )


def eval_for_setup(
    fabric: L.Fabric,
    checkpoint_indexes: List[int],
    old_topk_entries: List[Tuple[int, float]],
    final_cp_index: int,
    model: LongContextInferenceModel,
    data: DataModule,
    train: TrainArgs,
    evals: EvalArgs,
    tokenizer: Tokenizer,
    out_dir: Path,
    model_type: str,
    model_config: ModelConfiguration,
) -> None:
    print_message(
        f"\nIterating over {len(checkpoint_indexes)} checkpoints:\n"
        + "\n".join([get_checkpoint_path(out_dir, i).stem for i in checkpoint_indexes]),
        fabric,
    )
    # Training state can be obtained from the last checkpoint written, usually
    # removed for all other checkpoints. We only need the train/valid split,
    # which is the same across all checkpoints
    task_path = get_checkpoint_path(out_dir, final_cp_index)
    try:
        data_train_state = restore_dataset_from_training_state(data, task_path)
        print_message(f"Training state loaded from {task_path}", fabric)
    except FileNotFoundError:
        print_message(
            f"No training state found at {task_path}.\nContinue with new random split.",
            fabric,
        )
        data_train_state = None
    # Data loader for validation set: The train/valid split is obtained from
    # the training state stored alongside the checkpoint
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

    new_records: List[Tuple[int, float]] = []
    for cp_ind in checkpoint_indexes:
        cp_path = get_checkpoint_path(out_dir, cp_ind)
        print_message(f"\nComputing validation loss for {cp_path}", fabric)
        # Load checkpoint
        print_message("Loading checkpoint", fabric)
        load_model_checkpoint(
            model=model,
            task_path=cp_path,
            model_type=model_type,
            fabric=fabric,
        )
        # Compute validation loss
        print_message("Evaluation on validation set", fabric)
        metrics = validate_and_all_reduce(
            model=model,
            evaluator=None,
            val_dataloader=val_dataloader,
            eval=evals,
            batch_transform=batch_transform,
            log_metrics=False,
            fabric=fabric,
        )
        val_loss = metrics["val_loss"]
        new_records.append((cp_ind, val_loss))
        print_message(
            f"Checkpoint {cp_path.stem}: "
            + string_for_val_metrics(metrics, None)
            + f" | val_time: {metrics['val_time']:.3f} s",
            fabric,
        )
        flush_io_streams()
        fabric.barrier()

    # Print and store results
    top_k = len(old_topk_entries)
    new_topk_entries = sorted(new_records, key=lambda x: x[1])[:top_k]
    for name, entries in (("old", old_topk_entries), ("new", new_topk_entries)):
        print_message(
            f"\nBest {top_k} iterations for {name} validation loss code:\n"
            + "\n".join(
                [
                    f"{get_checkpoint_path(out_dir, ind).stem}: {val:.3f}"
                    for ind, val in entries
                ]
            ),
            fabric,
        )
    winn_ind = new_topk_entries[0][0]
    winn_task = get_checkpoint_path(out_dir, winn_ind).stem
    eval_entry = {
        "out_dir": str(out_dir),
        "model_type": model_type,
        "eval_tasks": [winn_task],
    }
    result_path = out_dir / RESULT_FILENAME
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as fp:
        yaml.safe_dump(eval_entry, fp, sort_keys=False)
