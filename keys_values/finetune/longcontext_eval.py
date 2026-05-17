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
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Literal, Optional, Union, Any

from keys_values.finetune.args import KVCacheArgs, SDPAArgs
from keys_values.finetune.longcontext_eval_ext import setup_internal
from keys_values.finetune.longcontext_full import setup_internal


def setup(
    out_dir: Path,
    model_type: Literal["full", "lora"] = "lora",
    devices: Union[int, str] = 1,
    seed: int = 1337,
    access_token: Optional[str] = None,
    batch_size: Optional[int] = None,
    tasks: Optional[str] = None,
    kv_cache: Optional[KVCacheArgs] = None,
    sdpa: Optional[SDPAArgs] = None,
    verbose: Optional[str] = None,
    attention_forward_temp_size_gb: Optional[float] = None,
    lora_dropout: Optional[float] = None,
    use_sample_metric: bool = True,
    sample_metric_max_generated_tokens: int = 20,
    sample_metric_kwargs: Optional[Dict[str, Any]] = None,
    num_store_generated_samples: Optional[int] = None,
    skip_eval: bool = False,
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
        If `tasks` is given, it is a comma-separated list of tasks (as string),
        for example "final,step-000010,step-000020". In this case, only the tasks
        provided there worked on.
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
        tasks: Comma-separated list of tasks (as string) for which to run
            evaluation, for example "final,step-000010,step-000020". If given,
            only these tasks (checkpoints) are evaluated for. Otherwise, we
            run over all tasks found under `out_dir`, starting with "final".
        kv_cache: Configuration for the KV caches. Only if you like to overwrite
            the configuration stored with the checkpoints
        sdpa: Configuration for SDPA kernel. Only if you like to overwrite the
            configuration stored with the checkpoints
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
        sample_metric_kwargs: Keyword arguments for token sampling (params
            can be "temperature", "top_k", "top_p")
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
    entry = {
        "out_dir": out_dir,
        "model_type": model_type,
        "eval_tasks": tasks,
    }
    if kv_cache is not None:
        entry["kv_cache"] = asdict(kv_cache)
    if sdpa is not None:
        entry["sdpa"] = asdict(sdpa)
    setup_internal(
        setups=[entry],
        devices=devices,
        seed=seed,
        access_token=access_token,
        batch_size=batch_size,
        verbose=verbose,
        attention_forward_temp_size_gb=attention_forward_temp_size_gb,
        lora_dropout=lora_dropout,
        use_sample_metric=use_sample_metric,
        sample_metric_max_generated_tokens=sample_metric_max_generated_tokens,
        sample_metric_kwargs=sample_metric_kwargs,
        num_store_generated_samples=num_store_generated_samples,
        skip_eval=skip_eval,
    )
