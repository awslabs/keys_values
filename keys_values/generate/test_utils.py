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
from typing import Optional

import torch

from keys_values.config import Config
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT
from keys_values.utils import VerbosityLevels


def _config(vocab_size: int) -> Config:
    return Config(
        block_size=128,
        vocab_size=vocab_size,
        padded_vocab_size=vocab_size,
        n_layer=2,
        n_head=4,
        n_embd=32,
        rotary_percentage=1.0,
    )


def _gpt(seed: int, vocab_size: int, noise: float = 0.0) -> GPT:
    """
    Random model in double precision, so that greedy decisions do not depend
    on how tokens are chunked. If `noise > 0`, parameters are perturbed,
    which is used to create draft models agreeing partially with the target.

    """
    torch.random.manual_seed(seed)
    gpt_model = GPT(_config(vocab_size)).to(dtype=torch.float64).eval()
    if noise > 0:
        gen = torch.Generator().manual_seed(seed + 9999)
        with torch.no_grad():
            for param in gpt_model.parameters():
                scale = noise * param.std().clamp_min(1e-3)
                param.add_(scale * torch.randn(param.shape, generator=gen).to(param))
    return gpt_model


def _inference_model(
    gpt_model: GPT,
    max_seq_length: int,
    cache_name: Optional[str] = None,
    batch_size: int = 1,
    chunk_size: int = 4,
    cache_length: Optional[int] = None,
) -> LongContextInferenceModel:
    if cache_name is None:
        cache_name = "dense-default"
    gpt_model.max_seq_length = max_seq_length
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            gpt_model=gpt_model,
            name=cache_name,
            max_batch_size=batch_size,
            cache_length=max_seq_length if cache_length is None else cache_length,
            dtype=torch.float64,
            cache_kwargs=(
                dict(init_grace_tokens=2)
                if cache_name.startswith("lastrec")
                else None
            ),
        )
    )
    return LongContextInferenceModel(
        gpt_model=gpt_model,
        head_model=None,
        chunk_size=chunk_size,
        verbose=VerbosityLevels.NONE,
    )
