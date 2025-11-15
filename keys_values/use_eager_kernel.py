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
import csv
from functools import partial
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from scipy.spatial import KDTree

from litgpt.config import Config

from keys_values.attention import UseEagerPredicate


def linear_interpolation(
    kv_len: int,
    data_kv_len: List[int],
    data_q_len_thresh: List[int],
) -> float:
    right_x = data_kv_len[-1]
    if kv_len > right_x:
        return kv_len * (data_q_len_thresh[-1] / right_x)
    else:
        return np.interp(
            x=kv_len,
            xp=[0] + data_kv_len,
            fp=[1] + data_q_len_thresh,
        )


def default_use_eager_kernel_internal(
    kv_len: int,
    q_len: int,
    data_kv_len: List[int],
    data_q_len_thresh: List[int],
) -> bool:
    return q_len <= linear_interpolation(kv_len, data_kv_len, data_q_len_thresh)


DATA_KV_LEN = [4096, 6144, 8192, 12288, 16384, 24576, 32768]


def load_q_len_thresh_data() -> Tuple[np.ndarray, List[List[int]]]:
    path = Path(__file__).parent / "scripts" / "qlen_thresholds.csv"
    inputs = []
    q_len_thresh = []
    with path.open("r") as fp:
        reader = csv.DictReader(fp)
        curr_values = []
        for row in reader:
            curr_values.append(int(row["chunk_size"]))
            if int(row["cache_length"]) == DATA_KV_LEN[-1]:
                input = [
                    int(row["batch_size"]),
                    int(row["n_head"]),
                    int(row["n_query_groups"]),
                    int(row["head_size"]),
                ]
                if len(curr_values) != len(DATA_KV_LEN):
                    raise ValueError(f"Dataset in {path} corrupt: input={input}")
                q_len_thresh.append(curr_values)
                inputs.append(input)
                curr_values = []
    if curr_values:
        raise ValueError(f"Dataset in {path} corrupt: curr_values={curr_values}")
    return np.array(inputs), q_len_thresh


class DefaultUseEagerKernel:
    """
    Selects `use_eager_kernel` predicate to be used in
    :class:`MultiHeadSelfAttention`, see documentation there.

    The choice is made using profiling data obtained by running the
    `keys_values/scripts/profile_naive_vs_padded_sdpa.py` script on an EC2
    p4d.24xlarge instance (NVidia A100). The data was obtained for
    `dtype=torch.bfloat16`, `num_repeats=20`,
    `batch_sizes = [1, 2, 3, 4, 8]`, and a range of fingerprints
    `(h_head, n_query_groups, head_size)` from a range of transformer
    architectures of sizes 0.5B until 14B parameters.

    """
    def __init__(self):
        inputs, self._q_len_thresh = load_q_len_thresh_data()
        self._nearest_neighbor = KDTree(inputs)

    def __call__(
        self,
        n_head: int,
        n_query_groups: int,
        head_size: int,
        batch_size: int,
    ) -> UseEagerPredicate:
        """
        Args:
            n_head:: Number of heads in MHA
            n_query_groups: Number of query groups in MHA
            head_size: Size of K, V, Q vectors for each head
            batch_size: Batch size

        Returns:
            Predicate `use_eager_kernel(kv_len, q_len)`

        """
        assert batch_size >= 1
        assert n_head >= 1
        assert 1 <= n_query_groups <= n_head
        assert head_size >= 1
        # Nearest neighbor
        _, nn_ind = self._nearest_neighbor.query(
            np.array([batch_size, n_head, n_query_groups, head_size])
        )
        return partial(
            default_use_eager_kernel_internal,
            data_kv_len=DATA_KV_LEN,
            data_q_len_thresh=self._q_len_thresh[nn_ind],
        )


def transform_mha_kwargs(
    mha_kwargs: Dict[str, Any],
    config: Config,
    batch_size: int = 1,
    default_factory: Optional[DefaultUseEagerKernel] = None,
) -> Dict[str, Any]:
    if mha_kwargs.get("use_eager_kernel") is not None:
        return mha_kwargs
    if default_factory is None:
        default_factory = DefaultUseEagerKernel()
    use_eager_kernel = default_factory(
        config.n_head, config.n_query_groups, config.head_size, batch_size,
    )
    return dict(mha_kwargs, use_eager_kernel=use_eager_kernel)
