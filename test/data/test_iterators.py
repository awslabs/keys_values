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
import math
from operator import itemgetter
from typing import List

import pytest
import torch

from keys_values.data.iterators import SimilarSequenceLengthIterator


def similar_sequence_length_should_be(
    sequence_lengths: List[int],
    micro_batch_size: int,
    num_devices: int,
    longest_first: bool,
    shortest_first: bool,
    seed: int,
) -> List[List[int]]:
    prng = torch.Generator().manual_seed(seed)
    len_sl = len(sequence_lengths)
    global_batch_size = micro_batch_size * num_devices
    len_dataset = math.ceil(len_sl / global_batch_size) * global_batch_size
    sort_ind = [i for i, _ in sorted(enumerate(sequence_lengths), key=itemgetter(1))]
    extra_ind = list(range(len_sl, len_dataset))
    assert len_sl + len(extra_ind) == len_dataset
    size_last = 0 if len_sl == len_dataset else global_batch_size - len_dataset + len_sl
    num_full = len_dataset // global_batch_size
    num_extra = 0
    if size_last > 0:
        num_full -= 1
        num_extra += 1
        if size_last < num_devices:
            num_full -= 1
            num_extra += 1
    result = []
    for start in range(0, num_full * global_batch_size, global_batch_size):
        for off in torch.randperm(num_devices, generator=prng):
            _start = start + off * micro_batch_size
            result.append(sort_ind[_start : (_start + micro_batch_size)])
    if num_extra > 0:
        size_rest = len_sl - num_full * global_batch_size
        num_mb = num_extra * num_devices
        mbs = math.ceil(size_rest / num_mb)
        assert mbs >= 1
        fix_me = num_mb * mbs - size_rest
        assert fix_me < num_mb
        assert fix_me == 0 or mbs > 1, (fix_me, mbs)
        sizes = [mbs] * (num_mb - fix_me) + [mbs - 1] * fix_me
        assert sum(sizes) == size_rest
        off1 = num_full * global_batch_size
        off2 = 0
        _result = [[] for _ in range(num_devices)]
        rind = torch.randperm(num_devices, generator=prng)
        if num_extra > 1:
            rind = torch.cat((rind, torch.randperm(num_devices, generator=prng)))
        for sz1, pos in zip(sizes, rind):
            sz2 = micro_batch_size - sz1
            _result[pos].append(
                sort_ind[off1 : off1 + sz1] + extra_ind[off2 : off2 + sz2]
            )
            off1 += sz1
            off2 += sz2
        assert off1 == len_sl
        assert off2 == len(extra_ind)
        result.extend([x[0] for x in _result])
        if num_extra > 1:
            result.extend([x[1] for x in _result])

    if longest_first:
        result = result[(-num_devices):] + result[:(-num_devices)]
    return result


@pytest.mark.parametrize(
    "len_sl, micro_batch_size, num_devices",
    [
        (15, 4, 1),
        (15, 4, 4),
        (329, 4, 4),
        (329, 4, 8),
        (15 * 32, 4, 8),
        (15 * 32 + 8, 4, 8),
        (15 * 32 + 11, 4, 8),
        (15 * 32 + 4, 4, 8),
        (15 * 32 + 1, 4, 8),
    ],
)
def test_similar_sequence_length_iterator(
    len_sl: int,
    micro_batch_size: int,
    num_devices: int,
):
    seed = 31415927
    torch.random.manual_seed(seed)

    global_batch_size = micro_batch_size * num_devices
    num_chunks = math.ceil(len_sl / global_batch_size)
    len_dataset = num_chunks * global_batch_size
    sequence_lengths = None
    done = False
    # Avoid ties, otherwise tests fail due to differences in sorting:
    for _ in range(100):
        sequence_lengths = torch.randint(16, 32768, size=(len_sl,)).tolist()
        if len(set(sequence_lengths)) == len_sl:
            done = True  # No ties
            break
    if not done:
        raise AssertionError("WTF !!!!")
    print(
        f"len_sl = {len_sl}, micro_batch_size = {micro_batch_size}, num_devices = {num_devices}"
    )
    for longest_first, shortest_first in (
        (False, False),
        (True, False),
        (False, True),
    ):
        print(f"longest_first = {longest_first}")
        should_be = similar_sequence_length_should_be(
            sequence_lengths=sequence_lengths,
            micro_batch_size=micro_batch_size,
            num_devices=num_devices,
            longest_first=longest_first,
            shortest_first=shortest_first,
            seed=seed,
        )
        iterator = [
            SimilarSequenceLengthIterator(
                sequence_lengths=sequence_lengths,
                micro_batch_size=micro_batch_size,
                seed=seed,
                num_devices=num_devices,
                rank=rank,
                shuffle=False,
                longest_first=longest_first,
                shortest_first=shortest_first,
            )
            for rank in range(num_devices)
        ]
        batches = [
            next(iterator[step % num_devices])
            for step in range(len_dataset // micro_batch_size)
        ]
        assert len(batches) == len(should_be)
        print("batches vs should_be:")
        print(
            "\n".join(
                f"{i:3d}: " + str(a) + " -- " + str(b)
                for i, (a, b) in enumerate(zip(batches, should_be))
            )
        )
        for i, (a, b) in enumerate(zip(batches, should_be)):
            assert a == b, (i, a, b)
