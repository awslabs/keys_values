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
"""
Attention-backend selection for the RL drivers.

A GRPO step runs attention in two phases with different needs, on one ``GPT``
model:

* Rollout/decode (plus ``compute_logprobs`` and the no-grad replay forward of
  the gradient model): token-by-token decode with a growing ``kv_len``.
  Caches that need attention weights (H2O) have three SDPA implementations in
  :meth:`MultiHeadSelfAttention._sdpa_mode`: FlashInfer (vendored kernels,
  must be built with ``python build_ext.py``), the 2x FlexAttention baseline
  (only if a :class:`FlexAttentionArgs` is passed), and eager.
* Gradient cells (:class:`TrainingAttnWeightsReplayCache`, the chunked
  backward): C-long query chunks against a K-long replayed cache, with
  autograd. The cell picks its SDPA mode by one rule (``select_sdpa_mode``):
  FlexAttention iff its MHA has ``flexatt_args``, else query-padded PyTorch
  SDPA (``SDPA_IMPL_QPADDED_PYTORCH``: the C-long query is zero-padded to K,
  a K x K problem per chunk whose K-long output stays in the autograd graph;
  with K/C chunks per cell this is the K^2/C memory term and ~K/C x the
  attention FLOPs). Query-padded SDPA was a stop-gap until FlexAttention
  worked in the gradient pass.

Why decode does NOT use FlexAttention: the Flex manager compiles one kernel
per exact ``kv_len`` (``FlexAttnForPrefillManager._get_args``) and calls
``torch.compile`` on every miss. In token-by-token decoding ``kv_len``
changes every step until the cache is full, so every decode step recompiles
(observed: a 2-step smoke test did not finish in 25 minutes). The finetune
scripts never generate, which is why they can put ``flexatt_args`` on every
MHA. The right fix for a worker without FlashInfer is to build it
(``scripts/rl/provision_worker.sh``), not to pick a different kernel.

Per-phase mechanism: :func:`attention_mha_kwargs` configures the MHA objects
of the model and the caches (decode). :func:`gradient_train_cache_kwargs`
returns ``train_cache_kwargs`` for :class:`LongContextGradientModel`; its
``flexatt_args`` entry reaches :class:`TrainingAttnWeightsReplayCache`, which
then runs its chunk SDPA through a shallow copy of the shared ``kv_cache.mha``
with ``flexatt_args`` set (``gradient_mha``). The decode MHAs are never
mutated, so one model decodes with FlashInfer and takes gradients with Flex.

* ``auto``:  decode with FlashInfer if built (else eager, loud warning);
  gradient cells with FlexAttention on CUDA (query-padded on CPU).
* ``flex``:  FlexAttention everywhere, including decode (recompiles per
  ``kv_len``; experiments only). The cells use the MHA's own ``flexatt_args``.
* ``eager``: eager decode and query-padded PyTorch SDPA in the cells, i.e. the
  pre-Flex gradient path, kept as the A/B baseline.

Flex arguments for the cells (open point (b) of the task): ``q_lens =
choose_q_lens(chunk_size, num_q_lens)`` so the last, shorter chunk of a
sequence is padded to a bucket instead of compiling a graph per length;
``kv_lens=None`` because the ``ext-*`` annotations record the shape of the
key/value tensor the compiled graph saves and ``apply_ext_annotation`` undoes
reorder and GQA expansion but not ``kv_len`` padding, so a padded chunk would
leave its K/V unpacked in the graph. Without ``kv_lens`` the cells see
``kv_len = K`` once the cache is full and ``prefill + i*C`` while it fills,
so the compile count is bounded by about ``(K/C + 1) * 2`` graphs
(``q_len = C`` and the last-chunk bucket), times the number of distinct batch
sizes (two with ``grad_micro_batch``). ``extend_kv=False``: Flex handles GQA
natively (``enable_gqa``) and the first compiled graph per manager, the
dominant ``(q_len=C, kv_len=K)`` one, needs no ``repeat_interleave``; later
graphs fall back to expansion (issue #34 workaround in
``FlexAttnManager.__call__``), which is upstream behavior.

Open point (a): the no-grad inference forward that recomputes log-probs
(``compute_logprobs``, ``_inference_forward_pass``) stays on the decode
kernel. H2O needs attention weights there, where ``_sdpa_mode`` prefers
FlashInfer anyway, and these passes keep no K-long tensors (no autograd), so
they have no K^2/C term. Keeping them on the rollout kernel also keeps the
training-forward eviction decisions on the same kernel as the rollout.

Returned kwargs of :func:`attention_mha_kwargs` go to BOTH
``GPT(config, **mha_kwargs)`` and ``cache_kwargs`` of
:meth:`KVCacheFactory.create`: the caches build their own
``MultiHeadSelfAttention`` and the chunked-gradient cells reuse ``kv_cache.mha``.
"""

import warnings
from typing import Any, Dict

import torch

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.attention import flashinfer_ops
from keys_values.attention.base import (
    SDPA_IMPL_EAGER_BLOCKS,
    SDPA_IMPL_FLEXATTENTION,
    SDPA_IMPL_QPADDED_PYTORCH,
)
from keys_values.attention.flex_attention import FlexAttentionArgs, choose_q_lens
from keys_values.kvcache.base import DefaultKVCache
from keys_values.kvcache.gradient.train_attn_weights_replay import (
    gradient_mha,
    select_sdpa_mode,
)
from keys_values.model import GPT

ATTN_BACKENDS = ("auto", "flex", "eager")

# FlexAttention compiles a block-mask variant per (kv_len, q_len) shape; the
# stock dynamo limits are far too small. Same values as finetune's SDPAArgs.
DYNAMO_CACHE_SIZE_LIMIT = 32
DYNAMO_ACCUMULATED_CACHE_SIZE_LIMIT = 128

# Names of the SDPA modes a gradient cell can select (`select_sdpa_mode`)
SDPA_MODE_NAMES = {
    SDPA_IMPL_FLEXATTENTION: "flexattention",
    SDPA_IMPL_QPADDED_PYTORCH: "qpadded-pytorch",
    SDPA_IMPL_EAGER_BLOCKS: "eager-blocks",
}

EAGER_WARNING = (
    "FlashInfer extension is not built: KV cache '{name}' needs attention "
    "weights and will run the EAGER SDPA (several times slower in decode). "
    "Build it with `pip install flashinfer-python && python build_ext.py` "
    "(see scripts/rl/provision_worker.sh)."
)


def needs_attn_weights(kv_cache_name: str) -> bool:
    return kv_cache_name.startswith(("h2o", "qh2o"))


def _raise_dynamo_cache_limits() -> None:
    torch._dynamo.config.cache_size_limit = max(
        torch._dynamo.config.cache_size_limit, DYNAMO_CACHE_SIZE_LIMIT
    )
    torch._dynamo.config.accumulated_cache_size_limit = max(
        torch._dynamo.config.accumulated_cache_size_limit,
        DYNAMO_ACCUMULATED_CACHE_SIZE_LIMIT,
    )


def attention_mha_kwargs(
    backend: str,
    kv_cache_name: str,
    chunk_size: int,
    device: torch.device,
    attn_temp_gb: float = 0.0,
    num_q_lens: int = 4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Keyword arguments for :class:`MultiHeadSelfAttention` selecting the
    attention backend. Pass to ``GPT(config, **kwargs)`` and merge into
    ``cache_kwargs``.

    Args:
        backend: One of :data:`ATTN_BACKENDS`.
        kv_cache_name: Cache name; decides whether attention weights are needed.
        chunk_size: Prefill / gradient chunk size, anchors the Flex ``q_lens``.
        device: FlashInfer and Flex need CUDA; on CPU the result is eager.
        attn_temp_gb: If ``> 0``, bound eager attention-weight temporaries (GiB).
        num_q_lens: Number of anchor ``q_len`` values for Flex compilation.
        verbose: Print the chosen backend.
    """
    if backend not in ATTN_BACKENDS:
        raise ValueError(f"backend={backend!r}, must be one of {ATTN_BACKENDS}")
    weights = needs_attn_weights(kv_cache_name)
    kwargs: Dict[str, Any] = {}
    if attn_temp_gb > 0:
        kwargs["tmp_array_limit_gb"] = TemporaryArrayLimit(
            init_val=attn_temp_gb, name="attention_forward_temp_size_gb"
        )
    cuda = device.type == "cuda"
    if backend == "flex" and cuda:
        kwargs["use_flashinfer"] = False
        _raise_dynamo_cache_limits()
        kwargs["flexatt_args"] = FlexAttentionArgs(
            extend_kv=False,
            q_lens=choose_q_lens(chunk_size=chunk_size, num_q_lens=num_q_lens),
            forward_return_lse=weights,
        )
        chosen = "flex" + (" (2x, return_lse)" if weights else "")
        warnings.warn(
            "FlexAttention recompiles for every new kv_len; token-by-token "
            "decoding will be extremely slow. Use for gradient-pass experiments.",
            stacklevel=2,
        )
    else:
        # No flexatt_args, no use_eager_sdpa_always (the training replay
        # cache rejects the latter). _sdpa_mode then picks FlashInfer when
        # available and the call needs it, PyTorch SDPA for causal prefill,
        # and eager only for attention-weight calls without FlashInfer.
        if backend == "eager":
            kwargs["use_flashinfer"] = False
        flashinfer = cuda and flashinfer_ops._available and backend != "eager"
        if weights:
            chosen = "flashinfer" if flashinfer else "eager"
            if not flashinfer and backend == "auto" and cuda:
                warnings.warn(EAGER_WARNING.format(name=kv_cache_name), stacklevel=2)
        else:
            chosen = "pytorch-sdpa" + (" + flashinfer decode" if flashinfer else "")
    if verbose:
        print(
            f"attention backend: {chosen} (requested {backend}; "
            f"attention weights needed: {weights})",
            flush=True,
        )
    return kwargs


def gradient_train_cache_kwargs(
    backend: str,
    chunk_size: int,
    device: torch.device,
    num_q_lens: int = 4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    ``train_cache_kwargs`` for :class:`LongContextGradientModel` (via
    :func:`grpo_step`), selecting the SDPA mode of the gradient cells
    independently of the decode backend chosen by :func:`attention_mha_kwargs`.

    Args:
        backend: One of :data:`ATTN_BACKENDS`. ``auto`` gives the cells
            FlexAttention on CUDA. ``flex``: the MHAs already carry
            ``flexatt_args`` (the cells use those), nothing to add. ``eager``:
            query-padded PyTorch SDPA in the cells (pre-Flex path, baseline).
        chunk_size: Gradient chunk size, anchors the Flex ``q_lens``.
        device: Flex needs CUDA; on CPU the result is empty (query-padded).
        num_q_lens: Number of anchor ``q_len`` values for Flex compilation.
        verbose: Print the chosen mode.

    Returns:
        ``{"flexatt_args": FlexAttentionArgs(...)}`` or ``{}``.
    """
    if backend not in ATTN_BACKENDS:
        raise ValueError(f"backend={backend!r}, must be one of {ATTN_BACKENDS}")
    kwargs: Dict[str, Any] = {}
    if backend == "auto" and device.type == "cuda":
        _raise_dynamo_cache_limits()
        q_lens = choose_q_lens(chunk_size=chunk_size, num_q_lens=num_q_lens)
        kwargs["flexatt_args"] = FlexAttentionArgs(
            extend_kv=False,
            q_lens=q_lens,
            kv_lens=None,
            forward_return_lse=False,
        )
        chosen = f"flexattention (q_lens={q_lens}, kv_lens=None)"
    elif backend == "flex" and device.type == "cuda":
        chosen = "flexattention (flexatt_args of the model's MHA)"
    else:
        chosen = "query-padded PyTorch SDPA"
    if verbose:
        print(f"gradient-cell SDPA: {chosen} (requested {backend})", flush=True)
    return kwargs


def gradient_sdpa_mode_name(
    gpt_model: GPT,
    train_cache_kwargs: Dict[str, Any],
) -> str:
    """
    Name of the SDPA mode the gradient cells of `gpt_model` select, given
    `train_cache_kwargs` (see :func:`gradient_train_cache_kwargs`). Mirrors
    the choice in :class:`TrainingAttnWeightsReplayCache` for the first
    layer with a :class:`DefaultKVCache`.
    """
    kv_cache = next(
        (kvc for kvc in gpt_model.get_kv_caches() if isinstance(kvc, DefaultKVCache)),
        None,
    )
    if kv_cache is None:
        return "unknown"
    mha = gradient_mha(kv_cache.mha, train_cache_kwargs.get("flexatt_args"))
    mode = select_sdpa_mode(mha, mha._get_sliding_window_size(kv_cache.block_idx))
    return SDPA_MODE_NAMES.get(mode, str(mode))


def describe_backend(mha_kwargs: Dict[str, Any]) -> str:
    """Short label for logs/results, derived from the kwargs."""
    if "flexatt_args" in mha_kwargs:
        return "flex"
    if mha_kwargs.get("use_flashinfer", True) and flashinfer_ops._available:
        return "flashinfer"
    return "eager"
