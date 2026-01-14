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
from dataclasses import dataclass, asdict, replace
from typing import Dict, Any, Optional, Tuple, List

import torch

from litgpt.config import Config

from keys_values.kvcache.attn_weights import AttnWeightsKVCache
from keys_values.kvcache.buffers import KVCacheBuffers
from keys_values.kvcache.h2o import H2OKVCache


@dataclass(frozen=True)
class ForwardInput:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    token_idx: torch.Tensor
    next_positions: Optional[torch.Tensor]
    cache_keys: Optional[torch.Tensor]
    cache_values: Optional[torch.Tensor]
    cache_token_pos: Optional[torch.Tensor]
    attn_outputs_shape: Tuple[int, ...]
    cache_keys_after_shape: Tuple[int, ...]
    cache_values_after_shape: Tuple[int, ...]
    input_pos: int

    @staticmethod
    def from_inputs(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        input_pos: int,
        cache: AttnWeightsKVCache,
    ):
        if input_pos > 0:
            k_and_v = cache.kv_buffers.get_keys_values()
            kwargs = dict(
                cache_keys=k_and_v.keys(),
                cache_values=k_and_v.values(),
                cache_token_pos=cache.token_pos.clone(),
            )
        else:
            kwargs = dict(
                cache_keys=None,
                cache_values=None,
                cache_token_pos=None,
            )
        return ForwardInput(
            query=query,
            key=key,
            value=value,
            token_idx=token_idx,
            next_positions=cache._next_positions,
            attn_outputs_shape=(1,),
            cache_keys_after_shape=(1,),
            cache_values_after_shape=(1,),
            input_pos=input_pos,
            **kwargs,
        )


@dataclass(frozen=True)
class ForwardInputOutput:
    # Inputs: Args to `forward`
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    token_idx: torch.Tensor
    next_positions: torch.Tensor
    # Inputs: State of cache before `forward` call
    cache_keys: torch.Tensor
    cache_values: torch.Tensor
    cache_token_pos: torch.Tensor
    # Outputs: Computed during `forward`
    attn_outputs: torch.Tensor
    attn_weights: torch.Tensor
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
        next_positions: torch.Tensor,
        prefix: str,
    ) -> Tuple[Dict[str, Any], Optional[torch.Tensor]]:
        if not self.token_idx.equal(token_idx):
            raise ValueError(prefix + f": token_idx = {token_idx}; stored token_idx = {self.token_idx}")
        if next_positions.shape != self.next_positions.shape:
            raise ValueError(prefix + f": next_positions.shape = {next_positions.shape}; stored next_positions.shape = {self.next_positions.shape}")
        num_diff = (next_positions != self.next_positions).sum().item()
        if num_diff > 0:
            # Differences can arise, because small diffs between score entries
            # can lead to different sort orderings
            rel_diff = 100 * num_diff / next_positions.numel()
            print(prefix + f": next_positions: num_diff = {num_diff} ({rel_diff:.2f} %)")
            next_positions = self.next_positions
        else:
            next_positions = None
        return dict(
            query=self.query,
            key=self.key,
            value=self.value,
            token_idx=token_idx,
        ), next_positions


class TestLogInputsKVCacheMixin:
    """
    Enables logging the inputs to :meth:`forward` calls with subclasses of
    :class:`AttnWeightsKVCache`.

    """
    def start_logging(
        self,
        inputs: List[ForwardInput],
    ):
        self._inputs = inputs

    def _get_self(self) -> AttnWeightsKVCache:
        raise NotImplementedError

    def _get_cache_content(self) -> Dict[str, torch.Tensor]:
        cache = self._get_self()
        k_and_v = cache.kv_buffers.get_keys_values()
        return {
            "cache_keys": k_and_v.keys(),
            "cache_values": k_and_v.values(),
            "cache_token_pos": cache.token_pos.clone(),
        }

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
        )
        if hasattr(self, "_inputs"):
            self._entry = ForwardInput.from_inputs(
                **forward_kwargs, cache=self._get_self(), input_pos=input_pos,
            )
        return forward_kwargs

    def call_after_forward(self, attn_outputs: torch.Tensor):
        if hasattr(self, "_inputs"):
            k_and_v = self._get_self().kv_buffers.get_keys_values()  # just for shape
            entry = replace(
                self._entry,
                attn_outputs_shape=tuple(attn_outputs.shape),
                cache_keys_after_shape=tuple(k_and_v.keys().shape),
                cache_values_after_shape=tuple(k_and_v.values().shape),
            )
            self._inputs.append(entry)


class TestBeforeAfterKVCacheMixin:
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

    def start_comparing(
        self,
        path_mask: str,
        num_steps: int,
        tol_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._path_mask = path_mask
        self._num_steps = num_steps
        self._next_ind = 0
        self._debug_mode = "compare"
        self._current_record = None
        self._debug_attn_weights = None
        if tol_kwargs is None:
            tol_kwargs = dict()
        self._tol_kwargs = tol_kwargs

    def _get_self(self) -> AttnWeightsKVCache:
        raise NotImplementedError

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
    ) -> Dict[str, Any]:
        forward_kwargs = dict(
            query=query,
            key=key,
            value=value,
            token_idx=token_idx,
        )
        if self._debug_mode is not None:
            next_positions = self._get_self()._next_positions
            if self._debug_mode == "store":
                # Collect inputs for record to be written
                self._current_record = {
                    **forward_kwargs,
                    "next_positions": next_positions.clone(),
                    **self._get_cache_content(),
                }
            else:
                # Load record and write its inputs to members here
                debug_path = self._path_mask.format(self._next_ind)
                state = ForwardInputOutput.from_file(debug_path)
                forward_kwargs, next_positions = state.compare_replace(
                    token_idx=token_idx,
                    next_positions=next_positions,
                    prefix=f"Debug step {self._next_ind}",
                )
                if next_positions is not None:
                    self._get_self()._next_positions = next_positions
                self._current_record = asdict(state)
                self._set_cache_content(**self._current_record)
        return forward_kwargs

    def call_after_forward(self, attn_outputs: torch.Tensor):
        if self._debug_mode is not None:
            assert self._debug_attn_weights is not None
            # Collect outputs:
            outputs_here = {
                k + "_after": v
                for k, v in self._get_cache_content().items()
            }
            outputs_here["attn_outputs"] = attn_outputs
            outputs_here["attn_weights"] = self._debug_attn_weights
            if self._debug_mode == "store":
                self._current_record.update(**outputs_here)
                print(f"Store record {self._next_ind}")
                debug_path = self._path_mask.format(self._next_ind)
                ForwardInputOutput(**self._current_record).store(debug_path)
            else:
                # Need to sum attn_weights from old code!
                old_attn_weights = self._current_record["attn_weights"]
                self._current_record["attn_weights"] = old_attn_weights.to(
                    dtype=torch.float32
                ).sum(axis=2)
                # Compare loaded outputs with `outputs_here`
                print(f"Compare against record {self._next_ind}")
                names = (
                    "attn_weights",
                    "cache_token_pos_after",
                    "cache_keys_after",
                    "cache_values_after",
                    "attn_outputs",
                )
                for name in names:
                    print(f"Comparing {name}")
                    torch.testing.assert_close(
                        outputs_here[name],
                        self._current_record[name],
                        **self._tol_kwargs,
                    ), name

            self._debug_attn_weights = None
            self._next_ind += 1
            if self._next_ind >= self._num_steps:
                print(f"Did {self._next_ind} debug steps. Terminating.")
                exit(0)


class TestH2OKVCache(H2OKVCache, TestBeforeAfterKVCacheMixin):
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
    ) -> torch.Tensor:
        _kwargs = dict(
            query=query,
            key=key,
            value=value,
            token_idx=token_idx,
        )
        if self.input_pos > 0:
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
