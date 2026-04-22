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
from typing import Optional, List, Dict, Tuple

import torch

from keys_values.config import Config
from keys_values.data.helmet import Helmet
from keys_values.data.longbench_v2 import (
    get_instruction_template as get_template_longbench_v2,
    LongBenchV2,
)
from keys_values.data.load_helmet_dev_eval import (
    get_instruction_template as get_template_helmet,
)
from keys_values.kvcache.smart_lastrec import SmartInitialLastRecentlyInsertedKVCache
from keys_values.kvcache.test_utils_advanced import (
    load_tokenizer,
    sequence_of_words,
)
from keys_values.utils import randint_torch


def sample_slot_values(
    slot_names: Tuple[str, ...], dataset_key: Optional[str],
) -> Dict[str, str]:
    values = []
    for slot_name in slot_names:
        if slot_name in ("choice_A", "choice_B", "choice_C", "choice_D", "options"):
            num_words = randint_torch(5, 15)
        elif slot_name in ("question", "query"):
            num_words = randint_torch(10, 30)
        elif slot_name == "type_needle_v":
            num_words = 1
        elif slot_name == "context":
            if dataset_key == "json_kv":
                num_words = 0
            else:
                num_words = randint_torch(50, 200)
        elif slot_name == "demos":
            if dataset_key in ("alce_asqa", "alce_qampari"):
                num_words = 0
            else:
                num_words = randint_torch(50, 200)
        else:
            num_words = randint_torch(2, 10)
        if num_words > 0:
            values.append(sequence_of_words(num_words))
        else:
            values.append("")
    return dict(zip(slot_names, values))


def test_smart_lastrec_set_init_length(
    data_name: str,
    dataset_key: Optional[str],
    head_model: Optional[str],
    cache_length: int,
    expected_init_length: List[int],
):
    seed = 31415927
    torch.random.manual_seed(seed)
    num_repeats = 5
    batch_size = len(expected_init_length)

    # Create tokenizer and dataset
    tokenizer = load_tokenizer(cache_dir="./test_tokenizer")
    if data_name == "longbench_v2":
        data = LongBenchV2()
    else:
        data = Helmet(dataset_key=dataset_key)
    smart_lastrec_info = data.smart_lastrec_info()

    # Create KV cache
    vocab_size = tokenizer.vocab_size + len(tokenizer.additional_special_tokens)
    config = Config(
        n_layer=1,
        n_head=8,
        n_query_groups=4,
        n_embd=8 * 64,
        block_size=2 * cache_length,
        vocab_size=vocab_size,
        rotary_percentage=1,
    )
    cache = SmartInitialLastRecentlyInsertedKVCache.from_config(
        config,
        max_batch_size=batch_size,
        cache_length=cache_length,
        block_idx=0,
        tokenizer=tokenizer,
        end_initial_regex=smart_lastrec_info.end_initial_regex,
        max_initial_fraction=smart_lastrec_info.max_initial_fraction,
        include_end_string=smart_lastrec_info.include_end_string,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )

    if data_name == "longbench_v2":
        assert head_model is not None
        instruction_template, slot_names = get_template_longbench_v2(head_model)
    else:
        assert dataset_key is not None
        instruction_template, slot_names = get_template_helmet(dataset_key)
    for _ in range(num_repeats):
        # Create input sequences
        sequences = []
        for _ in range(batch_size):
            slot_kwargs = sample_slot_values(slot_names, dataset_key)
            sequences.append(instruction_template.format(**slot_kwargs))
        # Tokenize and pad sequences. We do left padding, the number of
        # pad tokens need to be added to `expected_init_length`
        encoded_seqs = [
            tokenizer.encode(seq, return_tensors="pt") for seq in sequences
        ]
        kwargs = dict(dtype=encoded_seqs[0].dtype)
        max_length = max([x.numel() for x in encoded_seqs])
        num_left_pad = [max_length - x.numel() for x in encoded_seqs]
        input_ids = torch.cat(
            [
                torch.cat(
                    (torch.zeros(num_pad, **kwargs), x)
                ).unsqueeze(0)
                for x, num_pad in zip(encoded_seqs, num_left_pad)
            ],
            dim=0,
        )
        if max_length > cache_length:
            input_ids = input_ids[:, :cache_length]
        # Prefill call to determine `cache.init_length`
        # HIER!
