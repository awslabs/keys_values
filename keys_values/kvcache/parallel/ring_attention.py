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
from typing import Optional, List, Tuple

import torch

from keys_values.config import Config
from keys_values.kvcache.parallel.flex_for_ring import (
    sdpa_ring_flexatt_offdiag,
    sdpa_ring_flexatt_diag,
    RingOffdiagFlexAttentionArgs,
    RingDiagFlexAttentionArgs,
)


class RingAttentionDriver:
    """
    Implements the computations needed to drive RingAttention.

    If ranks in `range(num_devices)` are denoted by `r, s`, the KV cache
    buffers `keys, values` and the queries `queries` are split between them.
    The class represents computations for rank `rank_r`. Computation proceeds
    in `num_devices` steps. It is initiated by :meth:`reset`, passing the
    `query` part for this rank. Then, each step is done by calling
    :meth:`__call__`. In step `t`, `key, value` for
    `rank_s = (rank_r - t) % num_devices` is passed as input, the computation
    for cell `(rank_r, rank_s)` is done, and results are accumulated.

    Update and reordering of `key, value`:
    The KV arguments going into :meth:`__call__` must already have been
    updated by new information (as well as equalized before). Also, they must
    have been reordered so the new content is on the right end. This is also
    why `token_positions` is not needed here.
    """

    def __init__(
        self,
        rank_r: int,
        flexatt_args_diag: RingDiagFlexAttentionArgs,
        flexatt_args_offdiag: RingOffdiagFlexAttentionArgs,
    ):
        self.num_devices = flexatt_args_offdiag.num_devices
        if not (0 <= rank_r < self.num_devices):
            raise ValueError(f"rank_r={rank_r}, must be in [0, {self.num_devices})")
        self.rank_r = rank_r
        self.flexatt_args_diag = flexatt_args_diag
        self.flexatt_args_offdiag = flexatt_args_offdiag
        # Assigned with `reset`
        self.query = None
        self.scale = None
        self.input_pos = None
        self.num_new_tokens = None
        self.config = None
        self.batch_size = None
        self.kv_len = None
        self.steps_done = None
        self.q_len_per_rank = None
        self._accum_output = None
        self._accum_lse = None

    def reset(
        self,
        query: torch.Tensor,
        scale: Optional[float],
        input_pos: int,
        num_new_tokens: int,
        config: Config,
    ):
        if query.ndim != 4:
            raise ValueError(f"query.shape={query.shape}, must be 4D")
        batch_size, _, q_len, _ = query.shape
        shape = (
            batch_size,
            config.n_head,
            q_len,
            config.head_size,
        )
        if query.shape != shape:
            raise ValueError(f"query.shape={query.shape}, must be {shape}")
        if input_pos < 0:
            raise ValueError(f"input_pos={input_pos}, must be >= 0")
        if scale is not None and scale <= 0:
            raise ValueError(f"scale={scale}, must be positive")
        self.query = query
        self.scale = scale
        self.input_pos = input_pos
        self.num_new_tokens = num_new_tokens
        self.config = config
        self.batch_size = batch_size
        self.kv_len = None
        self.steps_done = 0
        self.q_len_per_rank = self._get_q_len_per_rank()
        self._accum_output = None
        self._accum_lse = None

    def _get_q_len_per_rank(self) -> List[int]:
        """
        Information for token at position `p` is sent to rank `p % num_devices`.

        Returns:
            Query length per rank, called `M(r)` in the technical report
        """
        ndevs = self.num_devices
        q_len_min = self.num_new_tokens // ndevs
        num_p1 = self.num_new_tokens - q_len_min * ndevs
        result = [q_len_min] * ndevs
        if num_p1 > 0:
            q_len_max = q_len_min + 1
            start = self.input_pos % ndevs
            end = min(start + num_p1, q_len_min)
            for i in range(start, end):
                result[i] = q_len_max
            num_rem = start + num_p1 - end
            if num_rem > 0:
                for i in range(num_rem):
                    result[i] = q_len_max
        return result

    def __call__(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        if self.steps_done is None:
            raise IndexError("Object not initialized. Call `reset` first")
        if self.steps_done >= self.num_devices:
            raise IndexError(
                "All steps of the round have been processed. Access results "
                "with `results`, or start a new round with `reset`"
            )
        rank_s = (self.rank_r - self.steps_done) % self.num_devices
        if key.ndim != 4:
            raise ValueError(f"key.shape={key.shape}, must be 4D")
        if self.steps_done == 0:
            self.kv_len = key.shape[2]
        shape = (
            self.batch_size,
            self.config.n_query_groups,
            self.kv_len,
            self.config.head_size,
        )
        if key.shape != shape or value.shape != shape:
            raise ValueError(
                f"key.shape={key.shape}, value.shape={value.shape}, must be {shape}"
            )
        new_output, new_lse = self._attention_for_cell(key, value, rank_s)
        self._accumulate(new_output, new_lse)
        self.steps_done += 1

    def _attention_for_cell(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        rank_s: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.steps_done == 0:
            # Diagonal case: `rank_r == rank_s`
            output, lse = sdpa_ring_flexatt_diag(
                flexatt_args=self.flexatt_args_diag,
                query=self.query,
                key=key,
                value=value,
                scale_factor=self.scale,
                input_pos=self.input_pos,
            )
        else:
            # Offdiagonal case: `rank_r != rank_s`
            output, lse = sdpa_ring_flexatt_offdiag(
                flexatt_args=self.flexatt_args_offdiag,
                rank_r=self.rank_r,
                rank_s=rank_s,
                query=self.query,
                key=key,
                value=value,
                scale_factor=self.scale,
                input_pos=self.input_pos,
                q_len_for_s=self.q_len_per_rank[rank_s],
            )
        return output, lse

    def _accumulate(self, new_output: torch.Tensor, new_lse: torch.Tensor):
        if self.steps_done == 0:
            self._accum_output = new_output
            self._accum_lse = new_lse
        else:
            # print(f"_accum_lse: ({self._accum_lse.dtype}, {self._accum_lse.shape}); new_lse: ({new_lse.dtype}; {new_lse.shape})")  # DEBUG
            new_accum_lse = torch.maximum(self._accum_lse, new_lse) + torch.log1p(
                torch.exp(-torch.abs(self._accum_lse - new_lse))
            )
            dtype = self._accum_output.dtype
            output_part1 = self._accum_output * torch.exp(
                self._accum_lse - new_accum_lse
            ).to(dtype=dtype).unsqueeze(-1)
            output_part2 = new_output * torch.exp(
                new_lse - new_accum_lse
            ).to(dtype=dtype).unsqueeze(-1)
            self._accum_output = output_part1 + output_part2
            self._accum_lse = new_accum_lse

    def results(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Accumulated outputs, accumulated log_sum_exp

        """
        if self._accum_output is None:
            raise IndexError("Accumulators are still empty")
        return self._accum_output, self._accum_lse
