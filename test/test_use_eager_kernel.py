# Original Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
# Modification Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import numpy as np
import pytest

from keys_values.use_eager_kernel import (
    linear_interpolation,
    DATA_KV_LEN,
    DefaultUseEagerKernel,
)

# DATA_KV_LEN = [4096, 6144, 8192, 12288, 16384, 24576, 32768]


@pytest.mark.parametrize(
    "data_q_len_thresh, kv_len, thresh",
    [
        ([1, 16, 16, 17, 17, 154, 344], 28672, 249),
        ([1, 16, 16, 17, 17, 154, 344], 128, 1),
        ([1, 16, 16, 17, 17, 154, 344], 40000, 419.921875),
        ([67, 128, 141, 257, 363, 357, 528], 15360, 336.5),
        ([67, 128, 141, 257, 363, 357, 528], 4224, 70.8125),
        ([67, 128, 141, 257, 363, 357, 528], 256, 5.125),
    ],
)
def test_linear_interpolation(
    data_q_len_thresh: List[int],
    kv_len: int,
    thresh: float,
):
    assert (
        abs(thresh - linear_interpolation(kv_len, DATA_KV_LEN, data_q_len_thresh))
        < 1e-9
    )


# (batch_size, n_head, n_query_groups, head_size)
@pytest.mark.parametrize(
    "fingerprint, nn_ind",
    [
        ((2, 14, 2, 64), 1),
        ((3, 16, 2, 128), 12),
        ((7, 27, 12, 128), 59),
        ((7, 29, 12, 128), 34),
        ((9, 54, 54, 132), 49),
        ((5, 54, 54, 132), 48),
    ],
)
def test_nearest_neighbor(fingerprint: Tuple[int, ...], nn_ind):
    default_factory = DefaultUseEagerKernel()
    _, ind = default_factory._nearest_neighbor.query(np.array(fingerprint))
    assert ind == nn_ind
