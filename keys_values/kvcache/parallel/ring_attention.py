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
import contextlib
from typing import List, Optional, Any, Iterable

import torch

from keys_values.kvcache.parallel.ring_attention_utils import RingAttentionComputation


class DoubleBuffer:
    """
    Maintains two `torch.Tensor` objects, where in each cycle, one of them
    is read, the other is written. With :meth:`flip`, the cycle is switched, so
    that write becomes read and vice versa. An object is first written, then read
    after the flip, then overwritten after the next flip.

    If `single_only=True`, we only maintain one object. This is a dummy to be
    used if no double buffering is needed.
    """

    def __init__(self, single_only: bool = False):
        self._buffers: List[Optional[torch.Tensor]] = (
            [None] if single_only else [None, None]
        )
        self._write_pos = 0
        self._single_only = single_only

    def write(self, x: torch.Tensor):
        self._buffers[self._write_pos] = x

    def read(self) -> Optional[torch.Tensor]:
        return self._buffers[0 if self._single_only else 1 - self._write_pos]

    def flip(self):
        if not self._single_only:
            self._write_pos = 1 - self._write_pos


def stream_decorator(s: Optional[torch.cuda.Stream]) -> Any:
    return torch.cuda.stream(s) if s is not None else contextlib.nullcontext()


def main_stream_waits_for_events(events: List[torch.Event]):
    for event in events:
        torch.cuda.current_stream().wait_event(event)
    events.clear()


def streams_wait_for_event(
    streams: Iterable[torch.cuda.Stream],
    event: Optional[torch.Event],
):
    if event is not None:
        # Streams need to wait for GPU computation from previous
        # iteration to finish
        for s in streams:
            s.wait_event(event)
