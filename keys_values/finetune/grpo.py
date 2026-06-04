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
GRPO fine-tuning with KeysAndValues long-context KV cache support.

Provides ``GRPOLongContextTrainer`` — a subclass of TRL's ``GRPOTrainer``
whose per-token log-probability computation uses KeysAndValues' chunked
KV-cache forward pass, bounding GPU memory for arbitrarily long sequences.

Usage::

    from keys_values.finetune.grpo import GRPOLongContextTrainer

    trainer = GRPOLongContextTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=my_reward_func,
        train_dataset=dataset,
        kv_cache_name="h2o-torch-quantized8",
        kv_cache_length=16384,
        kv_chunk_size=1024,
    )
    trainer.train()
"""
from __future__ import annotations

import torch
from trl.trainer.grpo_trainer import GRPOTrainer

from keys_values.logprobs import chunked_per_token_logps
from keys_values.model import GPT

_UNWRAP_ATTRS = ("gpt_model", "model", "base_model", "module")


class GRPOLongContextTrainer(GRPOTrainer):
    """``GRPOTrainer`` with KV-cache chunked log-prob computation.

    Overrides TRL's full-sequence forward with KeysAndValues' bounded-memory
    chunked path for sequences exceeding ``kv_cache_length``. Short sequences
    fall through to TRL's default — zero overhead.

    Parameters
    ----------
    kv_cache_name : str
        Cache policy, e.g. ``"h2o-torch-quantized8"``.
    kv_cache_length : int
        Slot count. Sequences longer than this trigger the chunked path.
    kv_chunk_size : int
        Chunk size for post-prefill processing.
    kv_cache_kwargs : dict | None
        Extra kwargs forwarded to ``KVCacheFactory.create``.
    """

    def __init__(
        self,
        *args,
        kv_cache_name: str = "h2o-torch-quantized8",
        kv_cache_length: int = 16384,
        kv_chunk_size: int = 1024,
        kv_cache_kwargs: dict | None = None,
        **kwargs,
    ):
        self.kv_cache_name = kv_cache_name
        self.kv_cache_length = kv_cache_length
        self.kv_chunk_size = kv_chunk_size
        self.kv_cache_kwargs = kv_cache_kwargs
        super().__init__(*args, **kwargs)

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        *,
        batch_size=None,
        compute_entropy=False,
        **kwargs,
    ):
        """Route long sequences through the chunked KV-cache path."""
        seq_len = input_ids.shape[1]

        if seq_len <= self.kv_cache_length:
            return super()._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=batch_size,
                compute_entropy=compute_entropy,
                **kwargs,
            )

        gpt = _unwrap(model)
        bs = batch_size or input_ids.size(0)

        results = [
            chunked_per_token_logps(
                gpt_model=gpt,
                input_ids=input_ids[i : i + bs],
                logits_to_keep=logits_to_keep,
                cache_name=self.kv_cache_name,
                cache_length=self.kv_cache_length,
                chunk_size=self.kv_chunk_size,
                cache_kwargs=self.kv_cache_kwargs,
                temperature=self.temperature,
                compute_entropy=compute_entropy,
            )
            for i in range(0, input_ids.size(0), bs)
        ]

        logps = torch.cat([r[0] for r in results])
        entropies = (
            torch.cat([r[1] for r in results if r[1] is not None])
            if results[0][1] is not None
            else None
        )
        return logps, entropies

def _unwrap(model) -> GPT:
    """Peel wrappers (DDP, PEFT, HF, ...) until we hit a ``GPT`` instance."""
    seen: set[int] = set()
    cur = model
    while not isinstance(cur, GPT):
        if id(cur) in seen:
            break
        seen.add(id(cur))
        nxt = next(
            (getattr(cur, a) for a in _UNWRAP_ATTRS if hasattr(cur, a)),
            None,
        )
        if nxt is None:
            break
        cur = nxt

    if not isinstance(cur, GPT):
        raise TypeError(
            f"Cannot locate a keys_values.model.GPT inside {type(model).__name__}. "
            "Ensure your model was loaded through keys_values."
        )
    return cur
