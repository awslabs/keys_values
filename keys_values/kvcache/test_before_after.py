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
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import torch

from litgpt.config import Config

from keys_values.kvcache.attn_weights import AttnWeightsKVCache
from keys_values.kvcache.buffers import KVCacheBuffers
from keys_values.kvcache.h2o import H2OKVCache


@dataclass(frozen=True)
class ForwardInputOutput:
    # Inputs: Args to `forward`
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    token_idx: torch.Tensor
    input_pos: int
    # Inputs: State of cache before `forward` call
    cache_keys: torch.Tensor
    cache_values: torch.Tensor
    cache_token_pos: torch.Tensor
    # Outputs: Computed during `forward`
    attn_outputs: torch.Tensor
    attn_weights: torch.Tensor
    instant_scores: Optional[torch.Tensor]
    # Outputs: State of cache after `forward` call
    cache_keys_after: torch.Tensor
    cache_values_after: torch.Tensor
    cache_token_pos_after: torch.Tensor

    @staticmethod
    def from_file(path: str) -> "ForwardInputOutput":
        return ForwardInputOutput(**torch.load(path))

    def store(self, path: str):
        torch.save(asdict(self), path)

    def compare_replace(
        self,
        token_idx: torch.Tensor,
        input_pos: int,
        prefix: str,
    ) -> Dict[str, Any]:
        if self.input_pos != input_pos:
            raise ValueError(prefix + f": input_pos = {input_pos}; stored input_pos = {self.input_pos}")
        if not self.token_idx.equal(token_idx):
            raise ValueError(prefix + f": token_idx = {token_idx}; stored token_idx = {self.token_idx}")
        return dict(
            query=self.query,
            key=self.key,
            value=self.value,
            token_idx=token_idx,
            input_pos=input_pos,
        )


class TestAttnWeightsKVCacheMixin:
    """
    Enables before/after tests with subclasses of :class:`AttnWeightsKVCache`.
    We can (1) take snapshots of :meth:`forward` calls for the old code, then
    (2) compare snapshot outputs for the new code with stored ones. Here, for
    (2), we replace inputs with those from the snapshot. There may still be
    slow drift between old and new, this is not what we want to test here.

    """
    def start_storing(
        self,
        path_mask: str,
        num_steps: int,
    ):
        self._path_mask = path_mask
        self._num_steps = num_steps
        self._next_ind = 0
        self._debug_mode = "store"
        self._current_record = None
        self._debug_attn_weights = None
        self._debug_instant_scores = None

    def start_comparing(
        self,
        path_mask: str,
        num_steps: int,
    ):
        self._path_mask = path_mask
        self._num_steps = num_steps
        self._next_ind = 0
        self._debug_mode = "compare"
        self._current_record = None
        self._debug_attn_weights = None
        self._debug_instant_scores = None

    def _get_self(self) -> AttnWeightsKVCache:
        raise NotImplementedError

    @staticmethod
    def _has_instant_scores() -> bool:
        """
        Change this in subclasses where instant scores are different from
        attention weights.

        """
        return False

    def _get_cache_content(self) -> Dict[str, torch.Tensor]:
        cache = self._get_self()
        k_and_v = cache.kv_buffers.get_keys_values()
        return {
            "cache_keys": k_and_v.keys(),
            "cache_values": k_and_v.values(),
            "cache_token_pos": cache.token_pos.clone(),
        }

    def _set_cache_content(
        self,
        cache_keys: torch.Tensor,
        cache_values: torch.Tensor,
        cache_token_pos: torch.Tensor,
        **kwargs,
    ):
        cache = self._get_self()
        cache.token_pos.copy_(cache_token_pos)
        cache.kv_buffers.prefill(cache_keys, cache_values)

    def call_before_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        input_pos: int,
    ) -> Dict[str, Any]:
        forward_kwargs = dict(
            query=query,
            key=key,
            value=value,
            token_idx=token_idx,
            input_pos=input_pos,
        )
        if self._debug_mode is not None:
            if self._debug_mode == "store":
                # Collect inputs for record to be written
                self._current_record = {
                    **forward_kwargs,
                    **self._get_cache_content(),
                }
            else:
                # Load record and write its inputs to members here
                debug_path = self._path_mask.format(self._next_ind)
                state = ForwardInputOutput.from_file(debug_path)
                forward_kwargs = state.compare_replace(
                    token_idx=token_idx,
                    input_pos=input_pos,
                    prefix=f"Debug step {self._next_ind}",
                )
                self._current_record = asdict(state)
                self._set_cache_content(**self._current_record)
        return forward_kwargs

    def call_after_forward(self, attn_outputs: torch.Tensor):
        if self._debug_mode is not None:
            assert self._debug_attn_weights is not None
            if self._has_instant_scores():
                assert self._debug_instant_scores is not None
            # Collect outputs:
            outputs_here = {
                k + "_after": v
                for k, v in self._get_cache_content().items()
            }
            outputs_here["attn_outputs"] = attn_outputs
            outputs_here["attn_weights"] = self._debug_attn_weights
            outputs_here["instant_scores"] = self._debug_instant_scores
            if self._debug_mode == "store":
                self._current_record.update(**outputs_here)
                debug_path = self._path_mask.format(self._next_ind)
                ForwardInputOutput(**self._current_record).store(debug_path)
            else:
                # Compare loaded outputs with `outputs_here`
                names = (
                    "attn_outputs",
                    "attn_weights",
                    "instant_scores",
                    "cache_token_pos_after",
                    "cache_keys_after",
                    "cache_values_after",
                )
                if self._has_instant_scores():
                    names = names + ("instant_scores",)
                for name in names:
                    torch.testing.assert_close(
                        outputs_here[name],
                        self._current_record[name],
                    ), name

            self._debug_attn_weights = None
            self._debug_instant_scores = None
            self._next_ind += 1
            if self._next_ind >= self._num_steps:
                print(f"Did {self._next_ind} debug steps. Terminating.")
                exit(0)


class TestH2OKVCache(H2OKVCache, TestAttnWeightsKVCacheMixin):
    """
    For before/after testing, use this instead of :class:`H2OKVCache`.

    Run "before" with old code:
    - Use this instead of :class:`H2OKVCache`
    - Call :meth:`start_storing` when you want to start recording.
        The next `num_steps` calls of :meth:`forward` with
        `input_pos > 0` (not prefill) will be recorded. Then, the
        program exists.

    Run "after" with new code:
    - Use this instead of :class:`H2OKVCache`
    - Call :meth:`start_comparing` at the same place as in the old code for
        recording, with the same arguments
    - For the next `num_steps` calls of :meth:`forward` with
        `input_pos > 0`, outputs will be compared to what was recorded with the
        old code. As in a unit test, this stops when differences are too large.
        Otherwise, the program exits after `num_steps` calls.

    """
    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        grace_period: int = 0,
        replay_log_blocksize: Optional[int] = None,
        detach_attn_weights: bool = False,
        normalize_scores: bool = False,
        **base_kwargs,
    ):
        super().__init__(
            config=config,
            buffers=buffers,
            block_idx=block_idx,
            grace_period=grace_period,
            replay_log_blocksize=replay_log_blocksize,
            detach_attn_weights=detach_attn_weights,
            normalize_scores=normalize_scores,
            **base_kwargs,
        )

    def _get_self(self) -> AttnWeightsKVCache:
        return self

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        input_pos: int,
    ) -> torch.Tensor:
        _kwargs = dict(
            query=query,
            key=key,
            value=value,
            token_idx=token_idx,
            input_pos=input_pos,
        )
        if input_pos > 0:
            forward_kwargs = self.call_before_forward(**_kwargs)
            attn_outputs = super().forward(**forward_kwargs)
            self.call_after_forward(attn_outputs)
        else:
            attn_outputs = super().forward(**_kwargs)
        return attn_outputs

    def _update(self, *args, **kwargs):
        if len(args) >= 1:
            attn_weights = args[0]
        else:
            attn_weights = kwargs.get("attn_weights")
            if attn_weights is None:
                raise ValueError("Need to pass 'attn_weights' argument")
        self._debug_attn_weights = attn_weights.detach()
        super()._update(*args, **kwargs)
