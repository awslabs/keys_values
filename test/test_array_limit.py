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
import random

import torch

from litgpt.config import Config

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.finetune.longcontext_full import cleanup_cache_kwargs
from keys_values.head_model import CrossEntropyOnLogits
from keys_values.kvcache.factory import KVCacheFactory, split_name
from keys_values.kvcache.gradient.main import LongContextGradientModel
from keys_values.kvcache.utils import VerbosityLevels
from keys_values.long_context import KVCacheArgs
from keys_values.model import GPT, block_iterator


def test_tmp_array_limit_object():
    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)  # Set default dtype
    # Step 1: Forward limit
    # Create model and KV caches as in `longcontext_*` scripts
    config = Config.from_name(
        "Qwen2.5-0.5B",
        n_layer=3,
        padded_vocab_size=1024,
    )
    batch_size = 3
    tmp_array_limit_forward_gb = TemporaryArrayLimit(
        init_val=3, name="attention_forward_temp_size_gb",
    )
    gpt_model = GPT(config, tmp_array_limit_gb=tmp_array_limit_forward_gb)
    kv_cache_args = KVCacheArgs(
        name="h2o-torch-quantized8",
        cache_length=2048,
        layers_per_cell=1,
        chunk_size=256,
        cache_kwargs={
            "replay_log_blocksize": 1024,
            "allocate_buffers": False,
            "max_num_ranges": 4,
            "tmp_array_limit_gb": tmp_array_limit_forward_gb,
        },
        randomize_chunk_sizes=False,
        single_tokens_for_targets=False,
        verbose=VerbosityLevels.SOME.value,
    )
    cache_kwargs = cleanup_cache_kwargs(
        split_name(kv_cache_args.name)[0], kv_cache_args.cache_kwargs,
    )
    assert "tmp_array_limit_gb" in cache_kwargs
    kv_caches = KVCacheFactory.create(
        gpt_model=gpt_model,
        name=kv_cache_args.name,
        max_batch_size=batch_size,
        dtype=dtype,
        cache_length=kv_cache_args.cache_length,
        cache_kwargs=cache_kwargs,
    )
    gpt_model.assign_kv_caches(kv_caches)
    # Ensure that `tmp_array_limit_gb` is used in all relevant objects
    mha = gpt_model.mha
    if mha.tmp_array_limit_gb is None:
        raise ValueError("tmp_array_limit_gb is set, but model.mha.tmp_array_limit_gb is not")
    if not (mha.tmp_array_limit_gb is tmp_array_limit_forward_gb):
        raise ValueError("tmp_array_limit_gb and model.mha.tmp_array_limit_gb must be the same object")
    for block_idx, block in enumerate(block_iterator(gpt_model)):
        kv_cache = block.attn.kv_cache
        prefix = f"Block {block_idx} of model: "
        for obj, name in (
            (kv_cache.mha, "mha"),
            (kv_cache.kv_buffers.quantizer_k, "kv_buffers.quantizer_k"),
            (kv_cache.kv_buffers.quantizer_v, "kv_buffers.quantizer_v"),
        ):
            if obj.tmp_array_limit_gb is None:
                raise ValueError(prefix + f"tmp_array_limit_gb is set, but block.attn.kv_cache.{name}.tmp_array_limit_gb is not")
            if not (obj.tmp_array_limit_gb is tmp_array_limit_forward_gb):
                raise ValueError(prefix + f"tmp_array_limit_gb and block.attn.kv_cache.{name}.tmp_array_limit_gb must be the same object")
    # Step 2: Backward limit
    # Create wrapper model and run forward
    common_kwargs = dict(
        gpt_model=gpt_model,
        head_model=CrossEntropyOnLogits(config),
        chunk_size=kv_cache_args.chunk_size,
        randomize_chunk_sizes=kv_cache_args.randomize_chunk_sizes,
        single_tokens_for_targets=kv_cache_args.single_tokens_for_targets,
        verbose=kv_cache_args.verbosity_level,
        tmp_array_limit_gb=tmp_array_limit_forward_gb,
    )
    tmp_array_limit_backward_gb = TemporaryArrayLimit(
        init_val=2, name="attention_backward_temp_size_gb",
    )
    train_cache_kwargs = {"tmp_array_limit_gb": tmp_array_limit_backward_gb}
    cache_kwargs["tmp_array_limit_gb"] = tmp_array_limit_backward_gb
    model = LongContextGradientModel(
        **common_kwargs,
        layers_per_cell=kv_cache_args.layers_per_cell,
        qname=kv_cache_args.qname,
        cache_kwargs=cache_kwargs,
        train_cache_kwargs=train_cache_kwargs,
        layer_checkpoint_chunk_size=kv_cache_args.cache_length,
    )
    seq_length = 2 * kv_cache_args.cache_length
    token_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_length),
    )
    num_output_tokens = random.randint(4, int(seq_length * 0.75))
    targets = token_ids[:, (-num_output_tokens):]
    model.train()
    loss = model(token_ids, targets)
    model._create_members_for_backward()
    # Ensure that `tmp_array_limit_backward_gb` is used in all relevant objects
    # - Inference replay caches
    # - KV cache checkpointing
    # But for forward checkpointing: tmp_array_limit_gb !!
