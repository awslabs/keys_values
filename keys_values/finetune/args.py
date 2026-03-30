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
from dataclasses import dataclass
import math
from typing import Optional, Tuple, Dict, Any, List, Literal

from litgpt.args import EvalArgs as _EvalArgs

from keys_values.kvcache.factory import KVCacheFactory
from keys_values.kvcache.consts import split_name, SUPPORTED_QUANTIZERS
from keys_values.utils import VerbosityLevels


def _check_positive(value: Optional[float], name: str):
    if value is not None and value <= 0.0:
        raise ValueError(f"`{name}` must be positive, got {value}")


def _check_nonnegative(value: Optional[float], name: str):
    if value is not None and value < 0.0:
        raise ValueError(f"`{name}` must be nonnegative, got {value}")


def _check_int(value: Optional[int], name: str):
    if value is not None and value != int(value):
        raise ValueError(f"`{name}` must be an integer, got {value}")


def _set_attr(kwargs: Dict[str, Any], key: Optional[str], value: Optional[Any]):
    if key is not None and value is not None:
        kwargs[key] = value


def _append_line(lines: List[str], name: str, value: Optional[Any]):
    if value is not None:
        lines.append(f"  {name}: {value}")


@dataclass
class KVCacheArgs:
    """Command line arguments for key-value cache and long context inference
    Args:
        name: Name of KV cache, has form `{cache_name}-{buffer_name}`. At
            present, this is the same for all layers of a model
        cache_length: Number of slots of KV cache. At present, this is the
            same for all layers of a model
        chunk_size: Long sequence batches are processed in chunks. The first
            chunk has size `cache.max_prefill_length`. Subsequent chunks are
            of size `chunk_size`
        cache_kwargs: Additional keyword args passed to KV cache constructor
        randomize_chunk_sizes: If `True`, chunk sizes are randomized, with
            mean `chunk_size`
        allocate_buffers: If `True`, KV cache buffers are allocated with
            construction. This may be on the wrong device, or with a wrong
            dtype. The default is delayed allocation with first usage
        grace_period: Only for caches with score-based policies. If positive,
            this number of last recently inserted tokens are not evicted.
        init_grace_tokens: Only for `lastrec` cache policy. KV information for
            the first `init_grace_tokens` is not evicted.
        cpu_offload: If `True`, KV cache buffers are offloaded to CPU during the
            forward pass. At the moment, this is implemented only for quantized
            KV cache buffers.
    """

    name: str
    cache_length: int
    chunk_size: int = 16
    cache_kwargs: Optional[Dict[str, Any]] = None
    randomize_chunk_sizes: bool = False
    allocate_buffers: bool = False
    grace_period: int = 0
    init_grace_tokens: int = 0
    cpu_offload: bool = False
    # Legacy (these are global args now)
    verbose: Optional[str] = None
    attention_forward_temp_size_gb: Optional[float] = None
    attention_backward_temp_size_gb: Optional[float] = None

    def __post_init__(self):
        supported_names = KVCacheFactory.supported_names()
        assert (
            self.name in supported_names
        ), f"name = {self.name} not supported, must be in {supported_names}"
        _check_positive(self.cache_length, "cache_length")
        assert self.cache_length >= 1
        if not (0 <= self.grace_period < self.cache_length):
            raise ValueError(
                f"grace_period = {self.grace_period}, must be in [0, {self.cache_length}])"
            )
        if not (0 <= self.init_grace_tokens < self.cache_length):
            raise ValueError(
                f"init_grace_tokens = {self.init_grace_tokens}, must be in [0, {self.cache_length}])"
            )
        if self.cpu_offload and split_name(self.name)[1] == "default":
            raise NotImplementedError(
                "CPU offloading (--kv_cache.cpu_offload True) is currently not supported for non-quantized KV cache buffers"
            )
        # Deprecated
        if self.verbose is None:
            self.verbose = VerbosityLevels.SOME.value
        else:
            assert (
                self.verbose in VerbosityLevels
            ), f"verbose = {self.verbose} not supported, must be in {VerbosityLevels}"
            print("--kv_cache.verbose is deprecated, use --verbose instead")
        if self.attention_forward_temp_size_gb is not None:
            assert self.attention_forward_temp_size_gb > 0
            print(
                "--kv_cache.attention_forward_temp_size_gb is deprecated, use --attention_forward_temp_size_gb instead"
            )
        if self.attention_backward_temp_size_gb is not None:
            assert self.attention_backward_temp_size_gb > 0
            print(
                "--kv_cache.attention_backward_temp_size_gb is deprecated, use --attention_backward_temp_size_gb instead"
            )

    @property
    def verbosity_level(self) -> VerbosityLevels:
        return VerbosityLevels(self.verbose)

    @property
    def qname(self) -> str:
        return split_name(self.name)[1]

    def maximum_chunk_size(self) -> int:
        if not self.randomize_chunk_sizes:
            return self.chunk_size
        else:
            step = self.chunk_size // 2
            return self.chunk_size + step

    def needs_attn_weights(self) -> bool:
        return KVCacheFactory.needs_attn_weights(self.name)


@dataclass
class GradientArgs:
    """Command line arguments for gradient computation (fine-tuning)
    Args:
        layers_per_cell: Cells for gradient computation span this many layers
            (from the bottom). GPU memory scales linearly in this number.
            Decrease if you run OOM.
        chunks_per_cell_multiplier: Each cell contains a number of chunks. The
            length of a cell is the sum of lengths of its cells. We assign
            chunks to cells so that cell lengths are close to
            `int(factor * cache_length * chunks_per_cell_multiplier)`, but not
            larger. Here, `factor = 2 * n_query_groups * head_size / n_embd`.
            If `chunks_per_cell_multiplier == 1`, this means that embeddings for
            this cell are as large as KV cache buffers. GPU memory scales
            linearly in this number.
        layercp_qname: Name of buffer type to be used for layer input
            checkpointing. See :const:`SUPPORTED_QUANTIZERS`.
        cachecp_qname: Name of buffer type to be used for KV cache
            checkpointing. See :const:`SUPPORTED_QUANTIZERS`.
        single_tokens_for_targets: If `True`, the targets part of a sequence is
            processed token per token (i.e., with chunk size 1). This is slower,
            but more realistic, mirroring how inference looks like.
        use_old_cache: If `True`, we use
            :class:`TrainingAttnWeightsReplayCacheOld` instead of
            :class:`TrainingAttnWeightsReplayCache`. The old code uses the
            fused naive SDPA during backward, which is slower, but also needs
            less GPU memory.
        max_match_trials_pack_arg: Parameter controlling autograd saved tensors
            hook mechanism, see :class:`CellComputationAutogradHooks`.
            Arguments of :meth:`pack_hook` are matched against annotations. A
            pack argument is removed (and not packed) if it is not matched
            after this number of :meth:`pack_hook` calls. This avoids running
            up costs trying to match pack args over and over, which can be
            significant.
        layercp_pin_memory: If `True`, the CPU memory pages for layer input
            checkpoints are pinned. This can run faster, but also needs more
            real CPU memory.
        cachecp_pin_memory: If `True`, the CPU memory pages for KV cache
            checkpoints are pinned. This can run faster, but also needs more
            real CPU memory.
        debug_print_annotations: If `True`, debug logging during `backward`
            computations are written which allow to track annotations for
            `autograd` saved tensors hooks.
    """

    layers_per_cell: int = 1
    chunks_per_cell_multiplier: float = 1.0
    layercp_qname: Optional[str] = None
    cachecp_qname: Optional[str] = None
    single_tokens_for_targets: bool = False
    use_old_cache: bool = False
    max_match_trials_pack_arg: Optional[int] = None
    layercp_pin_memory: bool = True
    cachecp_pin_memory: bool = True
    debug_print_annotations: bool = False

    def __post_init__(self):
        _check_positive(self.layers_per_cell, "layers_per_cell")
        assert self.layers_per_cell >= 1
        _check_positive(self.chunks_per_cell_multiplier, "chunks_per_cell_multiplier")
        if (
            self.layercp_qname is not None
            and self.layercp_qname not in SUPPORTED_QUANTIZERS
        ):
            raise ValueError(
                f"layercp_qname = {self.layercp_qname} not supported, must be in {SUPPORTED_QUANTIZERS}"
            )
        if (
            self.cachecp_qname is not None
            and self.cachecp_qname not in SUPPORTED_QUANTIZERS
        ):
            raise ValueError(
                f"cachecp_qname = {self.cachecp_qname} not supported, must be in {SUPPORTED_QUANTIZERS}"
            )
        _check_int(self.max_match_trials_pack_arg, "max_match_trials_pack_arg")


HAS_LEARNING_RATE = {
    "Adam": "lr",
    "AdamW": "lr",
    "Adamax": "lr",
    "Adadelta": "lr",
    "RMSprop": "lr",
    "SGD": "lr",
}

SUPPORTED_OPTIMIZERS = list(HAS_LEARNING_RATE.keys())

HAS_WEIGHT_DECAY = {
    "Adam": "weight_decay",
    "AdamW": "weight_decay",
    "Adamax": "weight_decay",
    "Adadelta": "weight_decay",
    "RMSprop": "weight_decay",
    "SGD": "weight_decay",
}

HAS_EPS = {
    "Adam": "eps",
    "AdamW": "eps",
    "Adamax": "eps",
    "Adadelta": "eps",
    "RMSprop": "eps",
}

HAS_MOMENTUM = {
    "RMSprop": "momentum",
    "SGD": "momentum",
}

HAS_DAMPENING = {
    "SGD": "dampening",
}

HAS_BETAS = {
    "Adam": "betas",
    "AdamW": "betas",
    "Adamax": "betas",
}

HAS_RHO = {
    "Adadelta": "rho",
}

HAS_ALPHA = {
    "RMSprop": "alpha",
}


@dataclass
class OptimizerArgs:
    """Command line arguments for optimizer
    Args:
        name: Name of optimizer, one of :const:`SUPPORTED_OPTIMIZERS`
        learning_rate: Base learning rate
        weight_decay: Weight decay constant
        eps: Eps constant
        momentum: Momentum constant (if supported)
        dampening: Dampening constant for momentum (if supported)
        adam_betas: `(beta1, beta2)`, only for Adam optimizers
        adadelta_rho: Rho constant (Adadelta only)
        rmspprop_alpha: Alpha constant (RMSprop only)
    """

    name: Optional[str] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    eps: Optional[float] = None
    momentum: Optional[float] = None
    dampening: Optional[float] = None
    adam_betas: Optional[Tuple[float, float]] = None
    adadelta_rho: Optional[float] = None
    rmspprop_alpha: Optional[float] = None

    def __post_init__(self):
        if self.name is None:
            self.name = "AdamW"  # Default optimizer
        elif self.name not in SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"name = {self.name} not supported [{SUPPORTED_OPTIMIZERS}]"
            )
        _check_positive(self.learning_rate, "learning_rate")
        _check_nonnegative(self.weight_decay, "weight_decay")
        _check_positive(self.eps, "eps")
        _check_nonnegative(self.momentum, "momentum")
        _check_nonnegative(self.dampening, "dampening")
        if self.adam_betas is not None:
            if not isinstance(self.adam_betas, tuple) or len(self.adam_betas) != 2:
                raise ValueError(
                    f"adam_betas = {self.adam_betas} must be tuple of size 2"
                )
            if any(not 0 <= x < 1 for x in self.adam_betas):
                raise ValueError(
                    f"adam_betas = {self.adam_betas}, entries must be in [0, 1)"
                )
        if self.adadelta_rho is not None and not (0 <= self.adadelta_rho <= 1):
            raise ValueError(f"adadelta_rho = {self.adadelta_rho}, must be in [0, 1]")
        _check_nonnegative(self.rmspprop_alpha, "rmspprop_alpha")

    def optimizer_kwargs(self):
        kwargs = dict()
        _set_attr(kwargs, HAS_LEARNING_RATE.get(self.name), self.learning_rate)
        _set_attr(kwargs, HAS_WEIGHT_DECAY.get(self.name), self.weight_decay)
        _set_attr(kwargs, HAS_EPS.get(self.name), self.eps)
        _set_attr(kwargs, HAS_MOMENTUM.get(self.name), self.momentum)
        _set_attr(kwargs, HAS_DAMPENING.get(self.name), self.dampening)
        _set_attr(kwargs, HAS_BETAS.get(self.name), self.adam_betas)
        _set_attr(kwargs, HAS_RHO.get(self.name), self.adadelta_rho)
        _set_attr(kwargs, HAS_ALPHA.get(self.name), self.rmspprop_alpha)
        return kwargs

    def __str__(self) -> str:
        lines = [
            "OptimizerArgs:",
            f"  name: {self.name}",
        ]
        _append_line(lines, "learning_rate", self.learning_rate)
        _append_line(lines, "weight_decay", self.weight_decay)
        _append_line(lines, "eps", self.eps)
        _append_line(lines, "momentum", self.momentum)
        _append_line(lines, "dampening", self.dampening)
        _append_line(lines, "adam_betas", self.adam_betas)
        _append_line(lines, "adadelta_rho", self.adadelta_rho)
        _append_line(lines, "rmspprop_alpha", self.rmspprop_alpha)
        return "\n".join(lines)


@dataclass
class LoRAARgs:
    """Command line arguments for LoRA fine-tuning

    With `kind`, a number of variants of LoRA can be chosen:

    * "default": Standard LoRA as implemented in `LitGPT`.
    * "rms_norm": Modification suggested by Sebastian Raschka:
        https://github.com/rasbt/dora-from-scratch/blob/main/Using-LinearDoRA.ipynb
        He calls this DoRA, but the modification is simpler, runs faster, but
        may work less well.
    * "dora": DoRA, see :class:`keys_values.dora_utils.DoRALinear`. Note that
        `lora_dropout` is ignored for this variant

    Args:
        r: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout value (not if `kind == "dora"`)
        query: Whether to apply LoRA to the query weights in attention
        key: Whether to apply LoRA to the key weights in attention
        value: Whether to apply LoRA to the value weights in attention
        projection: Whether to apply LoRA to the output projection in the
            attention block.
        mlp: Whether to apply LoRA to the weights of the MLP in the attention
            block.
        head: Whether to apply LoRA to linear output weights in the head.
        kind: See above. Defaults to "default".
    """

    r: int = 8
    alpha: int = 16
    dropout: float = 0
    query: bool = True
    key: bool = False
    value: bool = True
    projection: bool = False
    mlp: bool = False
    head: bool = False
    kind: Literal["default", "rms_norm", "dora"] = "default"


@dataclass
class TrainArgs:
    """
    Modified training-related arguments in :class:`litgpt.args.TrainArgs`.

    Here, `global_batch_size` does not have a default value. If not given,
    it should be set to the product of `micro_batch_size` and the number of
    devices, unless sequential gradient averaging is desired.

    Storing intermediate checkpoints: Normal checkpoints are stored whenever
    `state["step_count"] % train.save_interval == 0`. If
    `intermed_save_interval` is given, we also store checkpoints whenever
    `state["step_count"] % train.intermed_save_interval == 0`. The value
    should be smaller than `save_interval` and can be 1. However, we make
    sure that no more than `intermed_save_num` intermediate checkpoints are
    stored (by removing the oldest one after a new one has been written).

    Args:
        intermed_save_interval: See above
        intermed_save_num: See above
    """

    save_interval: Optional[int] = 1000
    """Number of optimizer steps between saving checkpoints"""
    log_interval: int = 1
    """Number of iterations between logging calls"""
    global_batch_size: Optional[int] = None
    """Number of samples between optimizer steps across data-parallel ranks"""
    micro_batch_size: int = 4
    """Number of samples per data-parallel rank"""
    lr_warmup_steps: Optional[int] = 100
    """Number of iterations with learning rate warmup active"""
    lr_warmup_fraction: Optional[float] = None
    """The fraction of an epoch to use for learning rate warmup"""
    epochs: Optional[int] = None
    """Number of epochs to train on"""
    # TODO: `pretrain` is the only script using `max_tokens` explicitly. replace it with epoch_size*epochs?
    max_tokens: Optional[int] = None
    """Total number of tokens to train on"""
    max_steps: Optional[int] = None
    """Limits the number of optimizer steps to run"""
    max_time: Optional[float] = None
    """Limits the number of seconds to train for"""
    max_seq_length: Optional[int] = None
    """Limits the length of samples"""
    tie_embeddings: Optional[bool] = None
    """Whether to tie the embedding weights with the language modeling head weights"""
    intermed_save_interval: Optional[int] = None
    intermed_save_num: Optional[int] = None

    def __post_init__(self) -> None:
        if self.lr_warmup_fraction and self.lr_warmup_steps:
            raise ValueError(
                "Can't provide both `--train.lr_warmup_fraction` and `--train.lr_warmup_steps`. Choose one."
            )
        if self.lr_warmup_fraction and not (0 <= self.lr_warmup_fraction <= 1):
            raise ValueError("`--train.lr_warmup_fraction` must be between 0 and 1.")

        if (
            self.lr_warmup_steps
            and self.max_steps
            and (self.lr_warmup_steps >= self.max_steps)
        ):
            print(
                "`--train.lr_warmup_steps` should be less than `--train.max_steps`."
                f" Got {self.lr_warmup_steps} lr_warmup_steps and {self.max_steps} max_steps.",
            )

        if self.intermed_save_interval is not None:
            if self.intermed_save_interval <= 0:
                raise ValueError("intermed_save_interval must be positive")
            if (
                self.save_interval is not None
                and self.intermed_save_interval >= self.save_interval
            ):
                raise ValueError(
                    f"intermed_save_interval = {self.intermed_save_interval}, must be smaller than save_interval = {self.save_interval}"
                )
            if self.intermed_save_num is None or self.intermed_save_num <= 0:
                raise ValueError("intermed_save_num must be given and positive")
        elif self.intermed_save_num is not None:
            raise ValueError(
                "intermed_save_num only needed if intermed_save_interval is given"
            )

    def gradient_accumulation_iters(self, devices: int, num_nodes: int = 1) -> int:
        """Number of iterations between gradient synchronizations"""
        gradient_accumulation_iters = (
            self.batch_size(devices, num_nodes) // self.micro_batch_size
        )
        assert gradient_accumulation_iters > 0
        return gradient_accumulation_iters

    def batch_size(self, devices: int, num_nodes: int = 1) -> int:
        """Number of samples between optimizer steps per data-parallel rank"""
        batch_size = self.global_batch_size // (devices * num_nodes)
        assert batch_size > 0
        return batch_size

    def warmup_iters(
        self, devices: int, num_nodes: int, max_iters: int, train_dataloader
    ) -> int:
        """Number of iterations to warm up the learning rate."""
        if self.lr_warmup_fraction:
            return min(
                max_iters, math.ceil(self.lr_warmup_fraction * len(train_dataloader))
            )
        if self.lr_warmup_steps:
            return min(
                max_iters,
                self.lr_warmup_steps
                * self.gradient_accumulation_iters(devices, num_nodes),
            )
        return 0


@dataclass
class EvalArgs(_EvalArgs):
    """
    Extends arguments in :class:`litgpt.args.EvalArgs`.

    Args:
        micro_batch_size: If given, this overrides `train.micro_batch_size`
            for evaluation
    """

    micro_batch_size: Optional[int] = None


@dataclass
class SDPAArgs:
    """Command line arguments for gradient computation (fine-tuning)

    Use `flex_extend_kv=True` if you encounter errors like this:
    ```
    AssertionError: expected size 32==8, stride 4194304==4194304 at dim=1
    This error most often comes from a incorrect fake (aka meta) kernel for a custom op.
    ```

    Args:
        flex_attention: Should PyTorch `flex_attention` be used? If not,
            we use PyTorch SDPA with zero-padded queries. Defaults to
            `True`.
        flex_extend_kv: If `True`, we apply `repeat_interleave` to
            `key, value` to avoid the GQA case. This may be needed to get
            around bugs in `flex_attention`.
        flex_num_q_lens: If given, this is the number of `q_len` values for
            which different graphs are compiled. Zero-padding of the `query`
            argument is used then. If not given, each different `q_len` value
            gets its own graph (not recommended).
        reorder_sort_if_3d: For both SDPA variants, we (currently) reorder
            `key, value` tensors so that standard causal masking applies.
            If `token_positions` is inherently 3D (in that
            `token_positions[b, h, j]` depends on `b, h`), this can be done
            by sorting for each `b, h`, or in a different way (if this
            argument is `False`). In some comparisons, sorting ended up
            being faster overall.
        use_flex_for_attn_weights: If `False`, we do not use the FlexAttention
            baseline to compute SDPA with summed attention weights. This is
            slower.
    """

    flex_attention: bool = True
    flex_extend_kv: bool = True
    flex_num_q_lens: Optional[int] = 4
    reorder_sort_if_3d: bool = True
    use_flex_for_attn_weights: bool = True

    def __post_init__(self):
        if self.flex_num_q_lens is not None and self.flex_num_q_lens <= 0:
            raise ValueError("flex_num_q_lens must be positive")
