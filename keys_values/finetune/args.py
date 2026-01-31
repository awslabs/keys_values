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
from typing import Optional, Tuple, Dict, Any, List

from litgpt.args import EvalArgs as _EvalArgs

from keys_values.kvcache.factory import KVCacheFactory, split_name
from keys_values.kvcache.utils import VerbosityLevels


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

    """

    name: str
    cache_length: int
    chunk_size: int = 16
    cache_kwargs: Optional[Dict[str, Any]] = None
    randomize_chunk_sizes: bool = False
    allocate_buffers: bool = False
    grace_period: int = 0
    init_grace_tokens: int = 0
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
        single_tokens_for_targets: If `True`, the targets part of a sequence is
            processed token per token (i.e., with chunk size 1). This is slower,
            but more realistic, mirroring how inference looks like.
        use_new_cache: If `True`, we use
            :class:`TrainingAttnWeightsReplayCacheNew` instead of
            :class:`TrainingAttnWeightsReplayCache`. The new code uses a faster
            SDPA during backward as well, but at present needs more GPU memory.
        max_match_trials_pack_arg: Parameter controlling autograd saved tensors
            hook mechanism, see :class:`CellComputationAutogradHooks`.
            Arguments of :meth:`pack_hook` are matched against annotations. A
            pack argument is removed (and not packed) if it is not matched
            after this number of :meth:`pack_hook` calls. This avoids running
            up costs trying to match pack args over and over, which can be
            significant.
        layer_checkpoint_chunk_size: Only relevant if layer input checkpointing
            uses quantization. We quantize / dequantize checkpoints in chunks
            of this length (along sequence axis). Larger values require more
            GPU memory.

    """

    layers_per_cell: int = 1
    chunks_per_cell_multiplier: float = 1.0
    single_tokens_for_targets: bool = (False,)
    use_new_cache: bool = False
    max_match_trials_pack_arg: Optional[int] = None
    layer_checkpoint_chunk_size: Optional[int] = None

    def __post_init__(self):
        _check_positive(self.layers_per_cell, "layers_per_cell")
        assert self.layers_per_cell >= 1
        _check_positive(self.chunks_per_cell_multiplier, "chunks_per_cell_multiplier")
        _check_int(self.max_match_trials_pack_arg, "max_match_trials_pack_arg")
        _check_int(self.layer_checkpoint_chunk_size, "layer_checkpoint_chunk_size")


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

    Args:
        r: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout value
        query: Whether to apply LoRA to the query weights in attention
        key: Whether to apply LoRA to the key weights in attention
        value: Whether to apply LoRA to the value weights in attention
        projection: Whether to apply LoRA to the output projection in the
            attention block.
        mlp: Whether to apply LoRA to the weights of the MLP in the attention
            block.
        head: Whether to apply LoRA to linear output weights in the head.

    """

    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    query: bool = True
    key: bool = False
    value: bool = True
    projection: bool = False
    mlp: bool = False
    head: bool = False


@dataclass
class EvalArgs(_EvalArgs):
    """
    Extends arguments in :class:`litgpt.args.EvalArgs`.

    Args:
        micro_batch_size: If given, this overrides `train.micro_batch_size`
            for evaluation

    """

    micro_batch_size: Optional[int] = None
