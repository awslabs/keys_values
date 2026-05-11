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
from typing import Dict, Literal, Optional, Union, Any

from litgpt.data import DataModule

from keys_values.autotune.optuna_study import OptunaArgs
from keys_values.autotune.autotune_full import setup_internal
from keys_values.finetune.args import (
    TrainArgs,
    EvalArgs,
    GradientArgs,
    KVCacheArgs,
    OptimizerArgs,
    SDPAArgs,
    LoRAArgs,
)
from keys_values.head_model import CrossEntropyOnLogits

DEFAULT_OUT_DIR = "out/finetune/autotune_lora"


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
    lora: LoRAArgs = LoRAArgs(
        r=8,
        alpha=16,
        dropout=0,
        query=True,
        key=False,
        value=True,
        projection=False,
        mlp=False,
        head=False,
        kind="default",
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
        lora: Arguments for LoRA extension of model, see
            ``keys_values.finetune.args.LoRAArgs`` for details. Adjust the LoRA
            rank with `lora.r`.
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
            started whenever elapsed time is below budget. One of
            `max_num_evals`, `time_budget` must be given.
        max_num_evals: Maximum number of evaluations a worker may run. One of
            `max_num_evals`, `time_budget` must be given.
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
        lora,
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
