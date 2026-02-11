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
from functools import partial
from typing import Optional, List

import torch

from keys_values.attention import (
    DefaultKeysAndValues,
)
from keys_values.kvcache.buffers import DefaultKVCacheBuffers
from keys_values.kvcache.gradient.accumulate import GradientAccumulator
from keys_values.kvcache.gradient.checkpoints import KVCacheBufferCheckpoints


def exchange_kv_cache_checkpoints(
    accumulator: GradientAccumulator,
    device: Optional[torch.device] = None,
):
    """
    Ensures that `accumulator._kv_cache_checkpoints` are of testing type
    :class:`KVCacheBufferTestingCheckpoints`. These do not quantize checkpoints,
    which simplifies gradient testing a lot.

    """

    def wrapped_create_checkpoints_and_buffers(
        orig_func,
        model_part,
    ):
        cache_buffers, checkpoints = orig_func(model_part)
        # Need to replace checkpoints
        chunk_numbers = checkpoints[0].chunk_numbers
        checkpoints = [
            KVCacheBufferTestingCheckpoints(
                chunk_numbers=chunk_numbers,
                device=device,
            )
            for _ in range(len(checkpoints))
        ]
        return cache_buffers, checkpoints

    accumulator._create_checkpoints_and_buffers = partial(
        wrapped_create_checkpoints_and_buffers,
        accumulator._create_checkpoints_and_buffers,
    )


class KVCacheBufferTestingCheckpoints(KVCacheBufferCheckpoints):
    """
    Checkpointing class used for testing. The checkpoints are not quantized,
    but the buffers are stored as they are. This is not recommended in
    practice, but simplifies gradient testing. Also, we do not reserve
    memory for checkpoints up front, but copy them as they come in.

    """

    def __init__(
        self,
        chunk_numbers: List[int],
        device: Optional[torch.device] = None,
    ):
        super().__init__(chunk_numbers)
        self._checkpoints: List[Optional[DefaultKeysAndValues]] = [None] * len(
            chunk_numbers
        )
        if device is None:
            device = torch.get_default_device()
        self.device = device

    def _set_checkpoint(
        self,
        pos: int,
        buffers: DefaultKVCacheBuffers,
    ) -> int:
        k_and_v = buffers.get_keys_values()
        self._checkpoints[pos] = DefaultKeysAndValues(
            keys=k_and_v.keys().to(device=self.device, copy=True),
            values=k_and_v.values().to(device=self.device, copy=True),
        )
        return pos

    def _get_checkpoint(
        self,
        pos: int,
        out: DefaultKVCacheBuffers,
    ):
        checkpoint = self._checkpoints[pos]
        if checkpoint is None:
            raise ValueError(
                f"checkpoint at pos={pos} is still empty. Use 'set_checkpoint'"
            )
        out.prefill_from_keys_values(checkpoint)
