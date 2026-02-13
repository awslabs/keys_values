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
from typing import Dict, Literal, Optional, Any, Union

from litgpt.args import TrainArgs
from litgpt.data import DataModule

from keys_values.finetune.args import (
    EvalArgs,
    GradientArgs,
    KVCacheArgs,
    OptimizerArgs,
    SDPAArgs,
)
from keys_values.finetune.longcontext_full import setup_internal
from keys_values.head_model import CrossEntropyOnLogits


DEFAULT_OUT_DIR = "out/finetune/longcon_offload_full"


def setup(
    checkpoint_dir: Path,
    out_dir: Path = Path(DEFAULT_OUT_DIR),
    precision: Optional[str] = None,
    devices: Union[int, str] = 1,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=2,
        micro_batch_size=2,
        lr_warmup_steps=None,
        lr_warmup_fraction=0.15,
        epochs=5,
        max_seq_length=None,
    ),
    eval: EvalArgs = EvalArgs(
        interval=100,
        max_new_tokens=100,
        max_iters=100,
        initial_validation=None,
        final_validation=True,
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
            "allocate_buffers": False,
            "max_num_ranges": 4,
        },
        randomize_chunk_sizes=False,
        allocate_buffers=False,
    ),
    grad: GradientArgs = GradientArgs(
        layers_per_cell=1,
        chunks_per_cell_multiplier=1.0,
        single_tokens_for_targets=False,
        use_new_cache=False,
        max_match_trials_pack_arg=8,
        layer_checkpoint_chunk_size=None,
    ),
    head_model: str = CrossEntropyOnLogits.NAME,
    head_model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: Optional[str] = None,
    attention_forward_temp_size_gb: Optional[float] = None,
    attention_backward_temp_size_gb: Optional[float] = None,
    yarn_rope: bool = True,
    sdpa: SDPAArgs = SDPAArgs(
        flex_attention=True,
        flex_extend_kv=True,
    ),
    record_gpu_memory_snapshots: Optional[int] = None,
    record_gpu_memory_kind: int = 0,
    record_gpu_memory_period: int = 0,
    generate_with_eval: bool = False,
    profile_grad_times: int = 0,
    profile_parts: Optional[str] = None,
    debug_dont_use_autograd_hooks: bool = False,
) -> None:
    """Finetune a model with CPU offloading

    Makes use of devices `range(devices)`. If `devices > 1`, we run distributed
    data parallel (DDP). We use `lightning.Fabric` here as well, but just to
    launch the processes. Model and optimizer are not wrapped, and we only make
    use of `fabric.all_reduce`.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to load for finetuning.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        devices: Number of GPU devices (ranks) to be used. We use ranks
            `range(devices)`.
        data: Data-related arguments. If not provided, the default is
            ``keys_values.data.LongBenchV2``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
            Note: We modified the defaults from `train.lr_warmup_steps=100` to
            `train.lr_warmup_fraction=0.15`, so the linear warm-up is the first
            15% of all steps.
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
            is used in backward if `grad.use_new_cache == False`.
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
            - 2: Special case
            - 3: One snapshot file during initial validation
            Defaults to 0.
        record_gpu_memory_period: Only if `record_gpu_memory_snapshots` is used.
            Snapshot files are written once per update step. Files are overwritten
            on this period, in that those for step `step` are written to
            directory `f"iteration{step % record_gpu_memory_period}"`.
            If this is 0, files are not overwritten, we use `f"iteration{step}"`.
            Defaults to 0.
        profile_grad_times: If given, we profile complete gradient computation
            for this many steps, then stop. Results are written to CSV file.
        profile_parts: If given, we use `cProfile` to profile the first forward
            (if "forward") or first backward (if "backward") pass. Results are
            printed, then the program stops.

    """
    setup_internal(
        True,
        checkpoint_dir,
        out_dir,
        precision,
        devices,
        1,
        False,
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
        record_gpu_memory_snapshots,
        record_gpu_memory_kind,
        record_gpu_memory_period,
        generate_with_eval,
        profile_grad_times,
        profile_parts,
        debug_dont_use_autograd_hooks,
    )
