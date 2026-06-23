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
from dataclasses import replace
from functools import reduce
import operator
from typing import Optional, List, Dict, Tuple

import pytest
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
from keys_values.head_model import CrossEntropyOnLogits
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.smart_lastrec import SmartInitialLastRecentlyInsertedKVCache
from keys_values.kvcache.test_utils import random_keys_values, random_tensor
from keys_values.kvcache.test_utils_advanced import (
    load_tokenizer,
    sequence_of_words,
)
from keys_values.utils import randint_torch, encode


def sample_slot_values(
    slot_names: Tuple[str, ...],
    dataset_key: Optional[str],
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


@pytest.mark.parametrize(
    "data_name, dataset_key, head_model, cache_length, expected_init_length",
    [
        ("longbench_v2", None, CrossEntropyOnLogits.NAME, 4096, [15, 15, 15]),
        ("helmet", "nq", None, 4096, [30, 30, 30]),
        ("helmet", "trivia_qa", None, 4096, [30]),
        ("helmet", "hotpot_qa", None, 4096, [30, 30]),
        ("helmet", "pop_qa", None, 4096, [30, 30, 30, 30]),
        ("helmet", "alce_asqa", None, 4096, [145, 145, 145, 145, 145]),
        ("helmet", "alce_qampari", None, 4096, [145, 145, 145]),
        ("helmet", "ms_macro", None, 4096, [71, 71]),
        ("helmet", "trec_coarse", None, 4096, [31, 31, 31]),
        ("helmet", "trec_fine", None, 4096, [31]),
        ("helmet", "nlu", None, 4096, [31, 31, 31, 31]),
        ("helmet", "banking77", None, 4096, [31, 31, 31]),
        ("helmet", "clinc150", None, 4096, [31, 31, 31]),
        ("helmet", "narrative_qa", None, 4096, [40, 40, 40, 40]),
        ("helmet", "infinite_bench_qa", None, 4096, [28, 28]),
        ("helmet", "infinite_bench_mc", None, 4096, [56, 56, 56]),
        ("helmet", "infinite_bench_sum", None, 4096, [56]),
        ("helmet", "multi_lex_sum", None, 4096, [61, 61, 61, 61]),
        ("helmet", "json_kv", None, 4096, [16, 16, 16]),
        ("helmet", "ruler_mk_needle", None, 4096, [28, 28]),
        ("helmet", "ruler_mk_uuid", None, 4096, [28, 28]),
        ("helmet", "ruler_mv", None, 4096, [28, 28]),
    ],
)
def test_smart_lastrec_update_protected_ranges(
    data_name: str,
    dataset_key: Optional[str],
    head_model: Optional[str],
    cache_length: int,
    expected_init_length: List[int],
):
    seed = 31415927 + reduce(operator.mul, expected_init_length, 1)
    torch.random.manual_seed(seed)
    num_repeats = 1
    batch_size = len(expected_init_length)
    device = torch.device("cpu")
    dtype = torch.float16

    # Create tokenizer and dataset
    tokenizer = load_tokenizer()
    if data_name == "longbench_v2":
        data = LongBenchV2()
    else:
        data = Helmet(dataset_key=dataset_key)
    smart_lastrec_info = replace(
        data.smart_lastrec_info(tokenizer),
        max_initial_fraction=0.95,
    )

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
    params = KVCacheParams.from_config(
        config=config,
        max_batch_size=batch_size,
        cache_length=cache_length,
        dtype=dtype,
    )
    # Overwrite `max_initial_fraction`, because we use short cache lengths
    cache = SmartInitialLastRecentlyInsertedKVCache.from_config(
        config,
        max_batch_size=batch_size,
        cache_length=cache_length,
        block_idx=0,
        tokenizer=tokenizer,
        end_initial_regex=smart_lastrec_info.end_initial_regex,
        max_initial_fraction=smart_lastrec_info.max_initial_fraction,
        include_end_string=smart_lastrec_info.include_end_string,
        device=device,
        dtype=dtype,
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
            encode(tokenizer, seq, return_tensors="pt").squeeze(0) for seq in sequences
        ]
        # We also append "xyz" a random number of times on the left. Need to be
        # careful with BOS and EOS tokens
        bogus_tokens = encode(
            tokenizer, "xyz", return_tensors="pt", add_special_tokens=False,
        )
        random_left = [
            bogus_tokens.expand(num, -1).flatten()
            for num in torch.randint(0, 8, (batch_size,)).tolist()
        ]
        lengths = [x.numel() + y.numel() for x, y in zip(encoded_seqs, random_left)]
        max_length = max(lengths)
        num_left_pad = [max_length - l for l in lengths]
        kwargs = dict(dtype=encoded_seqs[0].dtype)
        input_ids = torch.cat(
            [
                torch.cat(
                    (torch.zeros(nlp, **kwargs), x[0:1], y, x[1:])
                ).unsqueeze(0)
                for x, y, nlp in zip(encoded_seqs, random_left, num_left_pad)
            ],
            dim=0,
        )
        if max_length > cache_length:
            input_ids = input_ids[:, :cache_length]
        # Prefill call to determine `cache.protected_*`
        cache.reset()
        num = input_ids.shape[1]
        key, value = random_keys_values(
            params,
            num=num,
            device=device,
        )
        query = random_tensor(
            params,
            num=num,
            is_query=True,
            device=device,
        )
        cache.forward(query, key, value, input_ids)
        max_init_length = max(
            min(
                num,
                int(cache_length * smart_lastrec_info.max_initial_fraction),
            ),
            1,
        )
        _expected_start = num_left_pad
        _expected_end = [
            nlp + min(rn.numel() + el, max_init_length)
            for el, nlp, rn in zip(expected_init_length, num_left_pad, random_left)
        ]
        print(
            f"cache.protected_start = {cache.protected_start}\ncache.protected_end = {cache.protected_end}"
        )
        for i, (tokens, pstart, pend) in enumerate(
            zip(input_ids, cache.protected_start, cache.protected_end)
        ):
            print(
                f"\n[{i}]: {tokenizer.decode(tokens[pstart:pend], skip_special_tokens=True)}"
            )
        assert _expected_start == cache.protected_start, (_expected_start, cache.protected_start)
        assert _expected_end == cache.protected_end, (_expected_end, cache.protected_end)


@pytest.mark.parametrize(
    "data_name, dataset_key, head_model, expected_init_length",
    [
        ("longbench_v2", None, CrossEntropyOnLogits.NAME, [15, 15, 15]),
        ("helmet", "nq", None, [30, 30, 30]),
        ("helmet", "trivia_qa", None, [30]),
        ("helmet", "hotpot_qa", None, [30, 30]),
        ("helmet", "pop_qa", None, [30, 30, 30, 30]),
        ("helmet", "alce_asqa", None, [145, 145, 145, 145, 145]),
        ("helmet", "alce_qampari", None, [145, 145, 145]),
        ("helmet", "ms_macro", None, [71, 71]),
        ("helmet", "trec_coarse", None, [31, 31, 31]),
        ("helmet", "trec_fine", None, [31]),
        ("helmet", "nlu", None, [31, 31, 31, 31]),
        ("helmet", "banking77", None, [31, 31, 31]),
        ("helmet", "clinc150", None, [31, 31, 31]),
        ("helmet", "narrative_qa", None, [40, 40, 40, 40]),
        ("helmet", "infinite_bench_qa", None, [28, 28]),
        ("helmet", "infinite_bench_mc", None, [56, 56, 56]),
        ("helmet", "infinite_bench_sum", None, [56]),
        ("helmet", "multi_lex_sum", None, [61, 61, 61, 61]),
        ("helmet", "json_kv", None, [16, 16, 16]),
        ("helmet", "ruler_mk_needle", None, [28, 28]),
        ("helmet", "ruler_mk_uuid", None, [28, 28]),
        ("helmet", "ruler_mv", None, [28, 28]),
    ],
)
def test_smart_lastrec_update_ranges_several_chunks(
    data_name: str,
    dataset_key: Optional[str],
    head_model: Optional[str],
    expected_init_length: List[int],
):
    seed = 31415927 + reduce(operator.mul, expected_init_length, 1)
    torch.random.manual_seed(seed)
    batch_size = len(expected_init_length)
    device = torch.device("cpu")
    dtype = torch.float16

    # Create tokenizer and dataset
    tokenizer = load_tokenizer()
    if data_name == "longbench_v2":
        data = LongBenchV2()
        assert head_model is not None
        instruction_template, slot_names = get_template_longbench_v2(head_model)
    else:
        data = Helmet(dataset_key=dataset_key)
        assert dataset_key is not None
        instruction_template, slot_names = get_template_helmet(dataset_key)
    smart_lastrec_info = replace(
        data.smart_lastrec_info(tokenizer),
        max_initial_fraction=0.99,
    )

    # Create input sequences
    sequences = []
    for _ in range(batch_size):
        slot_kwargs = sample_slot_values(slot_names, dataset_key)
        sequences.append(instruction_template.format(**slot_kwargs))
    # Tokenize and pad sequences. We do left padding, the number of
    # pad tokens need to be added to `expected_init_length`
    encoded_seqs = [
        encode(tokenizer, seq, return_tensors="pt").squeeze(0) for seq in sequences
    ]
    # We also append "xyz" a random number of times on the left. Need to be
    # careful with BOS and EOS tokens
    bogus_tokens = encode(
        tokenizer, "xyz", return_tensors="pt", add_special_tokens=False,
    )
    random_left = [
        bogus_tokens.expand(num, -1).flatten()
        for num in torch.randint(0, 8, (batch_size,)).tolist()
    ]
    lengths = [x.numel() + y.numel() for x, y in zip(encoded_seqs, random_left)]
    max_length = max(lengths)
    num_left_pad = [max_length - l for l in lengths]
    kwargs = dict(dtype=encoded_seqs[0].dtype)
    input_ids = torch.cat(
        [
            torch.cat(
                (torch.zeros(nlp, **kwargs), x[0:1], y, x[1:])
            ).unsqueeze(0)
            for x, y, nlp in zip(encoded_seqs, random_left, num_left_pad)
        ],
        dim=0,
    )
    # Determine cache length
    cache_length = max_length + 16
    chunk_size = max_length + 1
    # Prepare data so that some sequences move further down in chunks
    parts = []
    bogus_token = encode(
        tokenizer, "q", return_tensors="pt", add_special_tokens=False,
    )
    expected_start = []
    expected_end = []
    expected_token_pos = []
    for shift_chunks, tokens, nlp, el, rn in zip(
        torch.randint(0, 3, (batch_size,)).tolist(),
        input_ids,
        num_left_pad,
        expected_init_length,
        random_left,
    ):
        e_start = nlp
        e_end = nlp + rn.numel() + el
        if shift_chunks == 0:
            fill_sz = cache_length + chunk_size
            fill_right = bogus_token.expand(fill_sz, -1).flatten()
            new_tokens = torch.cat((tokens[:-1], fill_right, tokens[-1:]))
            expected_start.append(e_start)
            expected_end.append(e_end)
            expected_token_pos.append(list(range(e_start, e_end)))
        elif shift_chunks == 1:
            off = cache_length
            pad_left = torch.zeros(off, **kwargs)
            fill_sz = chunk_size
            fill_right = bogus_token.expand(fill_sz, -1).flatten()
            new_tokens = torch.cat(
                (pad_left, tokens[:-1], fill_right, tokens[-1:])
            )
            expected_start.append(e_start)
            expected_end.append(e_end)
            expected_token_pos.append(list(range(e_start + off, e_end + off)))
        else:
            assert shift_chunks == 2, (shift_chunks,)
            off = cache_length + chunk_size
            pad_left = torch.zeros(off, **kwargs)
            new_tokens = torch.cat((pad_left, tokens))
            expected_start.append((e_start + chunk_size) % cache_length)
            expected_end.append((e_end + chunk_size) % cache_length)
            expected_token_pos.append(list(range(e_start + off, e_end + off)))
        parts.append(new_tokens.unsqueeze(0))
    input_ids = torch.cat(parts, dim=0)

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
    params = KVCacheParams.from_config(
        config=config,
        max_batch_size=batch_size,
        cache_length=cache_length,
        dtype=dtype,
    )
    # Overwrite `max_initial_fraction`, because we use short cache lengths
    cache = SmartInitialLastRecentlyInsertedKVCache.from_config(
        config,
        max_batch_size=batch_size,
        cache_length=cache_length,
        block_idx=0,
        tokenizer=tokenizer,
        end_initial_regex=smart_lastrec_info.end_initial_regex,
        max_initial_fraction=smart_lastrec_info.max_initial_fraction,
        include_end_string=smart_lastrec_info.include_end_string,
        device=device,
        dtype=dtype,
    )

    # Prefill, then two chunks
    cache.reset()
    input_pos = 0
    fin_sz = input_ids.shape[-1] - cache_length - chunk_size
    for num in (cache_length, chunk_size, fin_sz):
        key, value = random_keys_values(
            params,
            num=num,
            device=device,
        )
        query = random_tensor(
            params,
            num=num,
            is_query=True,
            device=device,
        )
        print(f"input_pos = {input_pos}, num = {num}")
        cache.forward(
            query, key, value, input_ids[:, input_pos:(input_pos + num)],
        )
        input_pos += num
    # Compare
    print(
        f"cache.protected_start: {cache.protected_start}\ncache.protected_end:   {cache.protected_end}"
    )
    for i, (tokens, p_start, p_end, token_pos) in enumerate(
        zip(input_ids, cache.protected_start, cache.protected_end, cache.token_positions()[:, 0, :])
    ):
        if p_start < p_end:
            t_start = token_pos[p_start].item()
            t_end = token_pos[p_end - 1].item() + 1
            print(
                f"\n[{i}]: {tokenizer.decode(tokens[t_start:t_end], skip_special_tokens=True)}"
            )
    assert expected_start == cache.protected_start, (expected_start, cache.protected_start)
    assert expected_end == cache.protected_end, (expected_end, cache.protected_end)
    for p_start, p_end, token_pos, exp_tp in zip(
        cache.protected_start, cache.protected_end, cache.token_positions()[:, 0, :], expected_token_pos,
    ):
        if p_start < p_end:
            tp = token_pos[p_start:p_end].tolist()
        else:
            tp = token_pos[p_start:].tolist() + token_pos[:p_end].tolist()
        assert tp == exp_tp, (tp, exp_tp, p_start, p_end)
