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
from typing import Dict, Literal, Optional, Union, Any, List, Tuple

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torch.nn.attention import SDPBackend
import yaml

from litgpt.config import Config as ConfigFull
from litgpt.data import DataModule
from litgpt.lora import Config as ConfigLoRA, mark_only_lora_as_trainable
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    auto_download_checkpoint,
    check_valid_checkpoint_dir,
    get_default_supported_precision,
    parse_devices,
    load_checkpoint,
)

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.attention_utils import DEFAULT_TMP_ARRAY_LIMIT_GB
from keys_values.data import LongBenchV2, INPUT_IDS_NAME
from keys_values.data.evaluation import (
    TASK_NAME,
    ORIG_IDX_NAME,
    EvaluationTasks,
    EvaluationWithTasksHelper,
)
from keys_values.finetune.batch_transform import BatchTransformFactory
from keys_values.finetune.longcontext_full import (
    wrap_gpt_model,
)
from keys_values.utils import flush_io_streams
from keys_values.finetune.utils import (
    LIT_MODEL_FNAME,
    HEAD_MODEL_FNAME,
    LORA_WEIGHTS_FNAME,
    LORA_WEIGHTS_FNAME_OLD,
    check_kv_cache,
)
from keys_values.head_model_factory import HeadModelFactory
from keys_values.kvcache.utils import VerbosityLevels
from keys_values.long_context import (
    LongContextInferenceModel,
)
from keys_values.finetune.args import KVCacheArgs
from keys_values.lora import GPT as GPTLoRA
from keys_values.model import GPT as GPTFull


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


def setup(
    out_dir: Path,
    model_type: Literal["full", "lora"] = "lora",
    devices: Union[int, str] = 1,
    num_nodes: int = 1,
    seed: int = 1337,
    access_token: Optional[str] = None,
    batch_size: Optional[int] = None,
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
    ),
) -> None:
    """Evaluate a range of model checkpoints on a test set

    Arguments:
        out_dir: Directory from where to load checkpoints. Checkpoints are
            looked for in subdirectories "step-[0-9]{6}" and "final".
        model_type: Either "full" or "lora".
        devices: How many devices/GPUs to use.
        num_nodes: How many nodes the code is being run on.
        seed: The random seed to use for reproducibility.
        access_token: Optional API token to access models with restrictions.
        batch_size: Size for test set batches
        kv_cache: Configuration for the KV caches. If not given, the configuration
            of the checkpoints is being used.

    """
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
    data = LongBenchV2(**hyp_pars["data"]["init_args"])
    if data.metadata_dir is None:
        data.metadata_dir = str(out_dir / "data")
        print(f"Setting LongBenchV2.metadata_dir to {data.metadata_dir}")
    if data.test_set_tag is None:
        data.test_set_tag = "rest"
        print(f"Setting LongBenchV2.test_set_tag to {data.test_set_tag}")
    devices = parse_devices(devices)
    if batch_size is None:
        batch_size = 8
    if kv_cache is None:
        kv_cache = KVCacheArgs(**hyp_pars["kv_cache"])
    check_kv_cache(kv_cache)
    check_valid_checkpoint_dir(checkpoint_dir)

    precision = hyp_pars["precision"] or get_default_supported_precision(training=True)
    if devices * num_nodes > 1:
        strategy = DDPStrategy(static_graph=True, broadcast_buffers=False)
    else:
        strategy = "auto"

    fabric = L.Fabric(
        devices=devices,
        num_nodes=num_nodes,
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
        model_type=model_type,
        model_config=model_config,
        eval_tasks=eval.tasks,
        devices=devices,
    )


def main(
    fabric: L.Fabric,
    seed: int,
    data: DataModule,
    checkpoint_dir: Path,
    out_dir: Path,
    batch_size: int,
    kv_cache: KVCacheArgs,
    model_type: str,
    model_config: ModelConfiguration,
    eval_tasks: List[str],
    devices: int,
) -> None:
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
        if model_type == "full":
            gpt_model = GPTFull(model_config.config, **mha_kwargs)
        else:
            gpt_model = GPTLoRA(model_config.config, **mha_kwargs)
            if isinstance(fabric.strategy.precision, BitsandbytesPrecision):
                from bitsandbytes.nn import StableEmbedding

                old_embedding = gpt_model.transformer.wte
                new_wte = StableEmbedding(
                    old_embedding.num_embeddings,
                    old_embedding.embedding_dim,
                )
                with torch.no_grad():
                    new_wte.weight.copy_(old_embedding.weight)
                gpt_model.transformer.wte = new_wte.to(
                    device=old_embedding.weight.device,
                    dtype=old_embedding.weight.dtype,
                )
        head_model = HeadModelFactory.create(
            name=model_config.head_model_name,
            config=model_config.config,
            data=data,
            **model_config.head_model_kwargs,
        )
        model = wrap_gpt_model(
            fabric=fabric,
            gpt_model=gpt_model,
            head_model=head_model,
            kv_cache=kv_cache,
            max_batch_size=batch_size,
            model_for_training=False,
        )
    if model_type == "lora":
        mark_only_lora_as_trainable(model.gpt_model)
    model = fabric.setup_module(model)

    # Load base model
    file_path = checkpoint_dir / LIT_MODEL_FNAME
    load_checkpoint(fabric, model.gpt_model, file_path, strict=False)
    # If there are head model weights, load them as well. Otherwise, we use
    # random initialization (or the head model may not have weights)
    file_path = checkpoint_dir / HEAD_MODEL_FNAME
    if file_path.exists():
        load_checkpoint(fabric, model.head_model, file_path, strict=True)
    # DEBUG:
    print("Check model weights for base model")
    debug_check_weights(model.gpt_model)

    # Loop over test set batches
    tasks_helper = EvaluationWithTasksHelper(out_dir, data.test_set_tag)
    current_task = None
    for batch in test_dataloader:
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
            print(f"Running inference for batch {task}, {orig_idxs}")
            batch = batch_transform(batch)
            if task != current_task:
                # DEBUG:
                if model_type == "lora":
                    lora_before = debug_lora_get_weights(model.gpt_model)
                task_path = out_dir / task
                print(f"New task {task}: Load model checkpoint from {task_path}")
                load_model_checkpoint(
                    model=model,
                    task_path=task_path,
                    model_type=model_type,
                    fabric=fabric,
                )
                current_task = task
                # DEBUG:
                print("Check model weights after loading checkpoint")
                debug_check_weights(model.gpt_model)
                # DEBUG:
                if model_type == "lora":
                    debug_lora_compare_weights(model.gpt_model, lora_before)
            # DEBUG
            print("*** longcontext_eval.main:")
            first = batch[INPUT_IDS_NAME][0, :]
            for second, label in zip(
                batch[INPUT_IDS_NAME][1:, :],
                batch["targets"],
            ):
                print(f"{label.item()}: {(first != second).sum().item()}")
                first = second
            # END DEBUG
            t0 = time.perf_counter()
            # One entry per batch dimension:
            loss_values = model(batch[INPUT_IDS_NAME], batch["targets"])
            loss_value = loss_values.mean().item()
            eval_time = time.perf_counter() - t0
            print(
                f"Batch {task}, {orig_idxs}: loss = {loss_value:.3f}, "
                f"eval_time = {eval_time * 1000:.2f} ms"
            )
            flush_io_streams()
            print(f"Storing to {eval_metrics_path}")
            store_eval_metrics(loss_values, batch, eval_metrics_path)
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
) -> DataLoader:
    """
    Creates data loader for cross product of test dataset with evaluation
    tasks. Each evaluation task corresponds to a model checkpoint written
    during or at the end of fine-tuning. See :class:`EvaluationTasks`.

    Args:
        data: LongBenchV2 dataset
        tokenizer: Tokenizer
        eval_tasks: List of evaluation tasks
        head_model: Head model name
        batch_size: Size of test batches
        fabric: Fabric

    Returns:
        Data loader for cross product of test dataset with evaluation tasks

    """
    if not isinstance(data, LongBenchV2):
        raise ValueError("Only LongBenchV2 is currently supported")
    data.connect(
        tokenizer=tokenizer,
        batch_size=batch_size,
        head_model=head_model,
        test_batch_size=batch_size,
        eval_tasks=eval_tasks,
    )
    if fabric is not None:
        with fabric.rank_zero_first():
            data.prepare_data()
    data.setup()
    test_dataloader = data.test_dataloader(num_devices=devices)
    if fabric is not None:
        test_dataloader = fabric.setup_dataloaders(test_dataloader)
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


def _is_zero(x: torch.Tensor) -> bool:
    return (x != 0).sum().item() == 0


def debug_check_weights(gpt_model: GPTFull):
    is_lora = isinstance(gpt_model, GPTLoRA)

    def weight(mod: torch.nn.Module) -> torch.Tensor:
        if is_lora:
            return mod.linear.weight
        else:
            return mod.weight

    if _is_zero(gpt_model.transformer.wte.weight):
        print("wte = 0")
    for block_idx, block in enumerate(gpt_model.transformer.h):
        if _is_zero(weight(block.attn.qkv)):
            print(f"attn.qkv[{block_idx}] = 0")
        if _is_zero(weight(block.attn.proj)):
            print(f"attn.proj[{block_idx}] = 0")
        if _is_zero(weight(block.mlp.fc_1)):
            print(f"mlp.fc_1[{block_idx}] = 0")
        if _is_zero(weight(block.mlp.fc_2)):
            print(f"mlp.fc_2[{block_idx}] = 0")
        if _is_zero(weight(block.mlp.proj)):
            print(f"mlp.proj[{block_idx}] = 0")
    if _is_zero(weight(gpt_model.lm_head)):
        print("lm_head = 0")


def debug_lora_get_weights(gpt_model: GPTLoRA) -> Dict[str, Any]:
    result = {
        "fixed": {
            "attn.qkv": gpt_model.transformer.h[6].attn.qkv.linear.weight,
            "attn.proj": gpt_model.transformer.h[10].attn.proj.linear.weight,
            "mlp.fc_1": gpt_model.transformer.h[2].mlp.fc_1.linear.weight,
            "mlp.fc_2": gpt_model.transformer.h[12].mlp.fc_2.linear.weight,
            "mlp.proj": gpt_model.transformer.h[5].mlp.proj.linear.weight,
        }
    }
    lora_result = {
        "attn.qkv_a": gpt_model.transformer.h[6].attn.qkv.lora_A,
        "attn.qkv_b": gpt_model.transformer.h[6].attn.qkv.lora_B,
    }
    config = gpt_model.config
    if config.lora_projection:
        lora_result.update(
            {
                "attn.proj_a": gpt_model.transformer.h[10].attn.proj.lora_A,
                "attn.proj_b": gpt_model.transformer.h[10].attn.proj.lora_B,
            }
        )
    if config.lora_mlp:
        lora_result.update(
            {
                "mlp.fc_1_a": gpt_model.transformer.h[2].mlp.fc_1.lora_A,
                "mlp.fc_1_b": gpt_model.transformer.h[2].mlp.fc_1.lora_B,
                "mlp.fc_2_a": gpt_model.transformer.h[12].mlp.fc_2.lora_A,
                "mlp.fc_2_b": gpt_model.transformer.h[12].mlp.fc_2.lora_B,
                "mlp.proj_a": gpt_model.transformer.h[5].mlp.proj.lora_A,
                "mlp.proj_b": gpt_model.transformer.h[5].mlp.proj.lora_B,
            }
        )
    result["lora"] = lora_result
    return result


def debug_lora_compare_weights(gpt_model: GPTLoRA, before: Dict[str, Any]):
    after = debug_lora_get_weights(gpt_model)
    for name, val1 in before["fixed"].items():
        val2 = after["fixed"][name]
        diff = (val1 != val2).sum().item()
        if diff != 0:
            print(f"Fixed: {name} different ({diff} / {val1.numel()})")
    for name, val1 in before["lora"].items():
        val2 = after["lora"][name]
        diff = (val1 == val2).sum().item()
        if diff == 0:
            print(f"LoRA: {name} the same")
