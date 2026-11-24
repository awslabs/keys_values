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

FlexAttention in decode needs ``kv_lens`` padding: the Flex manager compiles
one graph per ``(q_len, kv_len)`` shape and calls ``torch.compile`` on every
miss. In token-by-token decoding ``kv_len`` changes every step until the
cache is full, so without ``kv_lens`` every decode step recompiles (observed:
a 2-step smoke test did not finish in 25 minutes; ``docs/DECODE_BENCH.md``
measured 7-10 tok/s). With ``kv_lens = choose_kv_lens(cache_length,
num_kv_lens)`` (upstream #160) K/V are left-padded to the smallest bucket
``>= kv_len`` and the graph is keyed on the bucket, so decode compiles at
most ``num_kv_lens`` graphs per ``q_len`` (two per bucket for H2O: the
forward with LSE and the fp32 reverse call of
``sdpa_flexatt_with_attn_weights``). The padding waste is up to
``cache_length / num_kv_lens`` slots per step while the cache fills and zero
once an H2O cache is full (``kv_len == K``). Prefill (``input_pos=0``) stays
on PyTorch SDPA either way. FlashInfer is still the preferred decode kernel
when it is built (``scripts/rl/provision_worker.sh``): the padded Flex 2x
path moves several times the attention memory traffic of the fused kernel.

Per-phase mechanism: :func:`attention_mha_kwargs` configures the MHA objects
of the model and the caches (decode). :func:`gradient_train_cache_kwargs`
returns ``train_cache_kwargs`` for :class:`LongContextGradientModel`; its
``flexatt_args`` entry reaches :class:`TrainingAttnWeightsReplayCache`, which
then runs its chunk SDPA through a shallow copy of the shared ``kv_cache.mha``
with ``flexatt_args`` set (``gradient_mha``). The decode MHAs are never
mutated, so one model decodes with FlashInfer (or padded Flex) and takes
gradients with the cells' own Flex args (``kv_lens=None``, see below).

* ``auto``:  decode with FlashInfer if built, else FlexAttention with
  ``kv_lens`` padding, else eager (CPU); gradient cells with FlexAttention
  on CUDA (query-padded on CPU).
* ``flex``:  FlexAttention everywhere, including decode, with ``kv_lens``
  padding. The cells get their own ``kv_lens=None`` args.
* ``eager``: eager decode and query-padded PyTorch SDPA in the cells, i.e. the
  pre-Flex gradient path, kept as the A/B baseline. ``--disable-flashinfer``
  maps ``auto`` to ``eager`` in the drivers.

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

from typing import Any, Dict

import torch

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.attention import flashinfer_ops
from keys_values.attention.base import (
    SDPA_IMPL_EAGER_BLOCKS,
    SDPA_IMPL_FLEXATTENTION,
    SDPA_IMPL_QPADDED_PYTORCH,
)
from keys_values.attention.flex_attention import (
    FlexAttentionArgs,
    choose_kv_lens,
    choose_q_lens,
)
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

FLEX_FALLBACK_NOTICE = (
    "FlashInfer extension is not built: decode runs FlexAttention with kv_lens "
    "padding (kv_lens={kv_lens}; first step compiles). Build FlashInfer with "
    "`pip install flashinfer-python==0.6.7 && python build_ext.py` "
    "(see scripts/rl/provision_worker.sh) for the faster decode kernel."
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
    cache_length: int,
    attn_temp_gb: float = 0.0,
    num_q_lens: int = 4,
    num_kv_lens: int = 4,
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
        cache_length: Largest ``kv_len`` of the decode caches, anchors the
            Flex ``kv_lens`` buckets (``choose_kv_lens``).
        attn_temp_gb: If ``> 0``, bound eager attention-weight temporaries (GiB).
        num_q_lens: Number of anchor ``q_len`` values for Flex compilation.
        num_kv_lens: Number of ``kv_len`` buckets for Flex decode; the
            compile count per ``q_len`` is bounded by it, the padding waste
            while the cache fills is ``cache_length / num_kv_lens`` slots.
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
    flex_fallback = backend == "auto" and cuda and not flashinfer_ops._available
    if (backend == "flex" and cuda) or flex_fallback:
        # Padded Flex decode. The gradient cells do not inherit these args:
        # `gradient_mha` gives them their own kv_lens=None copy (see
        # gradient_train_cache_kwargs).
        if backend == "flex":
            kwargs["use_flashinfer"] = False
        _raise_dynamo_cache_limits()
        kv_lens = choose_kv_lens(cache_length=cache_length, num_kv_lens=num_kv_lens)
        kwargs["flexatt_args"] = FlexAttentionArgs(
            extend_kv=False,
            q_lens=choose_q_lens(chunk_size=chunk_size, num_q_lens=num_q_lens),
            kv_lens=kv_lens,
            forward_return_lse=weights,
        )
        chosen = "flex (kv_lens padded)" + (" 2x return_lse" if weights else "")
        if flex_fallback and verbose:
            print(FLEX_FALLBACK_NOTICE.format(kv_lens=kv_lens), flush=True)
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
        backend: One of :data:`ATTN_BACKENDS`. ``auto`` and ``flex`` give the
            cells FlexAttention on CUDA with their own ``kv_lens=None`` args
            (the decode MHAs may carry ``kv_lens``-padded args, which the
            cells must not use, see the module docstring). ``eager``:
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
    if backend in ("auto", "flex") and device.type == "cuda":
        _raise_dynamo_cache_limits()
        q_lens = choose_q_lens(chunk_size=chunk_size, num_q_lens=num_q_lens)
        kwargs["flexatt_args"] = FlexAttentionArgs(
            extend_kv=False,
            q_lens=q_lens,
            kv_lens=None,
            forward_return_lse=False,
        )
        chosen = f"flexattention (q_lens={q_lens}, kv_lens=None)"
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
    """
    Short label for the decode backend, derived from the kwargs of
    :func:`attention_mha_kwargs`: ``flex`` (explicit ``--attn flex``, or the
    ``auto`` fallback when FlashInfer is not built; both carry ``kv_lens``),
    ``flashinfer``, or ``eager``.
    """
    if mha_kwargs.get("flexatt_args") is not None:
        return "flex"
    if mha_kwargs.get("use_flashinfer", True) and flashinfer_ops._available:
        return "flashinfer"
    return "eager"
