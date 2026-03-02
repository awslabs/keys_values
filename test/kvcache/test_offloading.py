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
from typing import List, Tuple
from itertools import product

import torch
import pytest

from keys_values.config import Config

from keys_values.flex_attention import FlexAttentionArgs
from keys_values.head_model import CrossEntropyOnLogits
from keys_values.head_model_factory import HeadModelFactory
from keys_values.kvcache.factory import KVCacheFactory
from keys_values.kvcache.test_utils import cache_names_and_devices
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT
from keys_values.utils import randint_torch


def _filter_name(name):
    return name.startswith("lastrec") and not name.endswith("-default")

def args_compare_forward() -> Tuple[str, List[tuple]]:
    names_and_devices = cache_names_and_devices(filter_name=_filter_name)
    setups = [
        (128, 16, 256),
        (128, 32, 281),
    ]
    return (
        "name, device, cache_length, chunk_size, seq_length",
        [a + b for a, b in product(names_and_devices, setups)],
    )

# TODO: embeddings are empty. WHY??
@pytest.mark.parametrize(*args_compare_forward())
def test_compare_forward(name, device, cache_length, chunk_size, seq_length):
    seed = 31415927
    torch.random.manual_seed(seed)

    dtype = torch.float16
    torch.set_default_dtype(dtype)  # Set default dtype

    batch_size = 2
    n_layer = 4
    n_head = 8
    n_query_groups = 4
    head_size = 32
    vocab_size = 128

    # Create model and KV caches
    config = Config(
        n_layer=n_layer,
        n_head=n_head,
        n_query_groups=n_query_groups,
        n_embd=n_head * head_size,
        block_size=seq_length,
        vocab_size=vocab_size,
        rotary_percentage=1,
    )
    if device == torch.device("cpu"):
        mha_kwargs = dict()
    else:
        mha_kwargs = dict(flexatt_args=FlexAttentionArgs())
    with torch.device(device):
        gpt_model = GPT(config, **mha_kwargs)
        gpt_model.apply(gpt_model._init_weights)  # Initialization
    all_kv_caches = {
        "no": KVCacheFactory.create(
            gpt_model=gpt_model,
            name=name,
            max_batch_size=batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
            cache_kwargs=mha_kwargs,
        ),
    }
    all_kv_caches["yes"], cache_offloader = KVCacheFactory.create_cpu_offloading(
        gpt_model=gpt_model,
        name=name,
        max_batch_size=batch_size,
        cache_length=cache_length,
        dtype=dtype,
        device=device,
        cache_kwargs=mha_kwargs,
    )
    # Create data batches
    head_model_name = CrossEntropyOnLogits.NAME
    token_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_length),
        device=device,
    )
    num_output_tokens = randint_torch(4, int(seq_length * 0.75))
    input_ids = token_ids[:, :-1]
    targets = token_ids[:, (-num_output_tokens):]
    head_model = HeadModelFactory.create(name=head_model_name, config=config)
    # Run forward w/o offloading
    loss_values = dict()
    embeddings = dict(yes=[], no=[])
    for kind, kv_caches in all_kv_caches.items():
        print(kind)

        def layer_hook(x, block_idx):
            if block_idx > 0:
                embeddings[kind].append(x.clone())

        gpt_model.assign_kv_caches(kv_caches)
        gpt_model.set_start_of_layer_hook(layer_hook)
        model = LongContextInferenceModel(
            gpt_model=gpt_model,
            head_model=head_model,
            chunk_size=chunk_size,
            cache_offloader=cache_offloader if kind == "yes" else None,
        )
        with torch.no_grad():
            loss = model(input_ids, targets)
        loss_values[kind] = loss
    # Comparison
    for block_idx, (embd_no, embd_yes) in enumerate(
        zip(embeddings["no"], embeddings["yes"])
    ):
        print(f"Outputs of layer {block_idx}")
        torch.testing.assert_close(embd_no, embd_yes)
    print(f"Loss values: {loss_values["no"]} vs {loss_values["yes"]}")
    torch.testing.assert_close(loss_values["no"], loss_values["yes"])
    assert 1 == 0
