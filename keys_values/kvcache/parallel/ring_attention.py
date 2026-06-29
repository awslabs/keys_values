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
from typing import List, Optional, Any, Iterable, Tuple

import torch
import torch.distributed as dist

from keys_values.config import Config
from keys_values.kvcache.parallel.ring_attention_utils import RingAttentionComputation


class DoubleBuffer:
    """
    Maintains two `torch.Tensor` objects, where in each cycle, one of them
    is read, the other is written to. With :meth:`flip`, the cycle is switched,
    so that write becomes read and vice versa.

    One tensor is passed at construction, the other is created (same device,
    dtype). The buffer passed starts in read.
    """

    def __init__(self, x: torch.Tensor):
        self._buffers = (x,)
        self._read_pos = 0

    def _are_we_ok(self):
        if self._buffers is None:
            raise IndexError("Don't use after `cleanup`")
        if len(self._buffers) < 2:
            raise IndexError("Call `set_other` first")

    def set_other(self, other: Optional[torch.Tensor]) -> torch.Tensor:
        if self._buffers is None:
            raise IndexError("Don't use after `cleanup`")
        if len(self._buffers) == 2:
            raise IndexError("`set_other` can be called only once")
        x = self._buffers[0]
        if other is None or other.shape != x.shape or other.dtype != x.dtype or other.device != x.device:
            other = torch.zeros_like(x)
        self._buffers = (x, other)
        return other

    def write(self) -> torch.Tensor:
        self._are_we_ok()
        return self._buffers[1 - self._read_pos]

    def read(self) -> torch.Tensor:
        self._are_we_ok()
        return self._buffers[self._read_pos]

    def flip(self):
        self._are_we_ok()
        self._read_pos = 1 - self._read_pos

    def cleanup(self):
        """
        At the end, the object with read status should be the one passed
        at construction. This is the case iff `_read_pos == 0`, i.e. if
        :meth:`flip` was called an even number of times. Otherwise, we copy.

        """
        self._are_we_ok()
        if self._read_pos == 1:
            self._buffers[0].copy_(self._buffers[1])
            self._read_pos = 0
        self._buffers = None


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


class RingAttentionDriver:
    """
    Implements RingAttention. This consists of a loop with `num_devices`
    iterations. In each iteration, each rank runs local computations and
    accumulation (done by an :class:`RingAttentionComputation` object),
    as well as peer-to-peer communication, sending keys and values to the
    next rank in the ring and receiving them from the previous rank. These
    three operations run in parallel, using different streams. Each rank
    uses a :class:`DoubleBuffer` for keys and values.

    Equalization, splitting per rank, and reordering must have been done on
    keys and values before :meth:`__call__` is called.

    If `retain_others == True`, the other arrays in the double buffers are
    not deleted, but are reused (the same driver is used for all layers, and
    across several updates).
    """

    def __init__(
        self,
        ring_att_comp: RingAttentionComputation,
        retain_others: bool = True,
    ):
        self.ring_att_comp = ring_att_comp
        self._retain_others = retain_others
        self._send_stream = torch.cuda.Stream()
        self._recv_stream = torch.cuda.Stream()
        self._other_keys = None
        self._other_values = None

    @property
    def num_devices(self) -> int:
        return self.ring_att_comp.num_devices

    @property
    def rank(self) -> int:
        return self.ring_att_comp.rank_r

    def __call__(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        scale: Optional[float],
        input_pos: int,
        num_new_tokens: int,
        config: Config,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs RingAttention computation on rank `rank`. This is a loop of
        `num_devices` iterations, where ranks are synchronized at the end
        of each iteration.

        `keys`, `values` are overwritten along the loop, but are restored at
        the end.

        Args:
            queries: Queries for this rank
            keys: Keys for this rank
            values: Values for this rank
            scale: Scale parameter for MHA
            input_pos: Position of first new tokens
            num_new_tokens: Total number of new tokens (over all ranks)
             config: Model configuration

        Returns:
            `(attn_outputs, attn_lse)`, attention outputs for `queries` and
            logsumexp values

        """
        self.ring_att_comp.reset(
            queries, scale, input_pos, num_new_tokens, config,
        )
        buff_keys = DoubleBuffer(keys)
        buff_values = DoubleBuffer(values)
        # If `retain_others == True`, we try to re-use the other arrays here
        self._other_keys = buff_keys.set_other(self._other_keys)
        self._other_values = buff_values.set_other(self._other_values)
        rank_send = (self.rank + 1) % self.num_devices
        rank_recv = (self.rank - 1) % self.num_devices

        # Main loop
        for iter in range(self.num_devices):
            # First try: Use `reqs` returned by `isend`, `irecv`, but no events
            # at end of transfer streams
            # events_for_wait: List[torch.Event] = []
            reqs = []
            # Send KV
            with stream_decorator(self._send_stream):
                reqs.append(dist.isend(tensor=buff_keys.read(), dst=rank_send))
                reqs.append(dist.isend(tensor=buff_values.read(), dst=rank_send))
                # events_for_wait.append(self._send_stream.record_event())
            # Receive KV
            with stream_decorator(self._recv_stream):
                reqs.append(dist.irecv(tensor=buff_keys.write(), src=rank_recv))
                reqs.append(dist.irecv(tensor=buff_values.write(), src=rank_recv))
                # events_for_wait.append(self._recv_stream.record_event())
            # Computation
            self.ring_att_comp(buff_keys.read(), buff_values.read())
            main_event = torch.cuda.current_stream().record_event()
            # Wait for sync
            # - Main stream waits for all transfers to be complete (`reqs`)
            # - Transfer streams wait for `main_event`
            # main_stream_waits_for_events(events_for_wait)
            for req in reqs:
                req.wait()
            streams_wait_for_event(
                (self._send_stream, self._recv_stream),
                main_event,
            )
            # At this point, all point-to-point transfers and computations have
            # finished. Now, flip the roles of buffers
            buff_keys.flip()
            buff_values.flip()

        # Finish
        buff_keys.cleanup()
        buff_values.cleanup()
        if not self._retain_others:
            # Deallocate: They'll not be reused
            del self._other_keys
            self._other_values = None
            del self._other_values
            self._other_values = None
        return self.ring_att_comp.results()
