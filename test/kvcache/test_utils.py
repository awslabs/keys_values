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

from keys_values.kvcache.utils import smallest_covering_ranges


def test_smallest_covering_ranges():
    seed = 31415927
    random.seed(seed)
    torch.random.manual_seed(seed)

    num_repeats = 100
    cache_length = 2 ** 14
    for _ in range(num_repeats):
        num = random.randint(32, 1024)
        slots = torch.randperm(cache_length)[:num].tolist()
        max_num_ranges = random.randint(1, 8)
        ranges = smallest_covering_ranges(slots, max_num_ranges)
        assert 1 <= len(ranges) <= max_num_ranges
        # Test cover property
        cover_set = set(x for a, b in ranges for x in range(a, b))
        assert cover_set.issuperset(slots)
        if max_num_ranges == 1:
            a, b = ranges[0]
            assert min(slots) == a and max(slots) == b - 1
        else:
            num_covered = sum(b - a for a, b in ranges)
            a = ranges[0][0]
            b = ranges[-1][1]
            sum_size_holes = b - a - num_covered
            slots = sorted(slots)
            diffs = sorted(
                [b - a - 1 for a, b in zip(slots[:-1], slots[1:]) if b > a + 1],
                reverse=True,
            )[:(max_num_ranges - 1)]
            assert sum(diffs) == sum_size_holes
