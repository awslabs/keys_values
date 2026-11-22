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
from typing import List, Any, Dict, Optional, Tuple, Union

import torch

from keys_values.attention import KeysAndValues, MultiHeadSelfAttention
from keys_values.config import Config
from keys_values.kvcache.base import (
    KVCache,
    KVCacheParams,
    KVCacheReplayLog as KVCacheReplayLog_old,
)
from keys_values.kvcache.consts import split_name
from keys_values.kvcache.factory import KVCacheFactory as KVCacheFactory_old
from keys_values.kvcache.new import GenericKVCache
from keys_values.kvcache.new.basics import KVCacheReplayLog as KVCacheReplayLog_new
from keys_values.kvcache.new.factory import KVCacheFactory as KVCacheFactory_new
from keys_values.model import GPT

_EXCLUDE_FROM_COMPARISON = (MultiHeadSelfAttention,)

# Just while old and new code co-exist:
KVCacheReplayLog_oldornew = Union[KVCacheReplayLog_old, KVCacheReplayLog_new]


def _error_message(prefix: str, msg: str) -> str:
    return prefix + ":\n" + msg


def _compare(
    old_v: Any,
    new_v: Any,
    prefix: str,
):
    assert type(old_v) == type(
        new_v
    ), f"{prefix}: type(old_v) = {type(old_v)} != {type(new_v)} = type(new_v)"
    # May need to specialize to further types which (1) occur, and (2) for
    # which `==` comparison does not work
    if isinstance(old_v, _EXCLUDE_FROM_COMPARISON):
        return
    if isinstance(old_v, torch.Tensor):
        assert (
            old_v.shape == new_v.shape
        ), f"{prefix}: old_v.shape = {old_v.shape}, new_v.shape = {new_v.shape}"
        assert (
            old_v.dtype == new_v.dtype
        ), f"{prefix}: old_v.dtype = {old_v.dtype}, new_v.dtype = {new_v.dtype}"
        torch.testing.assert_close(
            old_v,
            new_v,
            msg=partial(_error_message, prefix=prefix + ": Tensor"),
        )
    elif isinstance(old_v, list) or isinstance(old_v, tuple):
        assert len(old_v) == len(
            new_v
        ), f"{prefix}: len(old_v) = {len(old_v)} != {len(new_v)} = len(new_v)"
        for pos, (ov, nv) in enumerate(zip(old_v, new_v)):
            _compare(ov, nv, f"{prefix}[{pos}]")
    elif isinstance(old_v, dict):
        old_keys = set(old_v.keys())
        new_keys = set(new_v.keys())
        assert old_keys == new_keys, (
            f"{prefix}: old_v, new_v dicts with different keys:\n"
            f"old_v.keys: {old_keys}\nnew_v.keys: {new_keys}"
        )
        for key, ov in old_v.items():
            nv = new_v[key]
            _compare(ov, nv, f"{prefix}[{key}]")
    elif isinstance(old_v, KeysAndValues):
        _compare(old_v.keys(), new_v.keys(), f"{prefix}.keys()")
        _compare(old_v.values(), new_v.values(), f"{prefix}.values()")
    else:
        assert old_v == new_v, f"{prefix}:\nold_v = {old_v}\nnew_v = {new_v}"


def _run_and_compare(
    old_cache: KVCache,
    new_cache: KVCache,
    name: str,
    args: Optional[Tuple[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    return_from_old: bool = True,
    do_compare: bool = True,
) -> Any:
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = dict()
    old_result = getattr(old_cache, name)(*args, **kwargs)
    new_result = getattr(new_cache, name)(*args, **kwargs)
    if do_compare:
        old_no_tuple = not isinstance(old_result, tuple)
        if old_no_tuple:
            old_result = (old_result,)
        new_no_tuple = not isinstance(new_result, tuple)
        if new_no_tuple:
            new_result = (new_result,)
        assert len(old_result) == len(
            new_result
        ), f"obj.{name}(...): len(old_result) = {len(old_result)} != {len(new_result)} = len(new_result)"
        for pos, (old_v, new_v) in enumerate(zip(old_result, new_result)):
            _compare(old_v, new_v, f"obj.{name}(...)[{pos}]")
    else:
        old_no_tuple = new_no_tuple = False
    if return_from_old:
        return old_result[0] if old_no_tuple else old_result
    else:
        return new_result[0] if new_no_tuple else new_result


def _compare_attribute(
    old_cache: KVCache,
    new_cache: KVCache,
    name: str,
    prefix: Optional[str] = None,
    return_from_old: bool = True,
    do_compare: bool = True,
) -> Any:
    old_v = getattr(old_cache, name)
    new_objs: List[Any] = [new_cache]
    if isinstance(new_cache, GenericKVCache):
        new_objs.extend([new_cache.policy, new_cache.driver])
    _done = False
    new_v = None
    for obj in new_objs:
        try:
            new_v = getattr(obj, name)
            _done = True
            break
        except AttributeError:
            pass
    prefix = "" if prefix is None else prefix + ": "
    _prefix = prefix + f"obj.{name}"
    if not _done:
        raise AttributeError(f"{_prefix} does not exist in new_cache or its components")
    if do_compare:
        _compare(old_v, new_v, _prefix)
    return old_v if return_from_old else new_v


class TestShadowKVCache(KVCache):
    """
    Helper for old/new comparison testing. Holds a pair of :class:`KVCache`
    objects, `old_cache` and `new_cache`. All calls are served from `old_cache`
    (if `return_from_old == True`; from `new_cache` otherwise). But for each
    call, both `old_cache`, `new_cache` are called with the same args, and the
    return values are compared. After each method call, we also compare all
    members listed in `state_names`.

    Important: Make sure that `old_cache` and `new_cache` work with complete
    replicas, not sharing underlying concepts. Otherwise, the comparison will
    not work.
    """

    def __init__(
        self,
        old_cache: KVCache,
        new_cache: KVCache,
        state_names: List[str],
        return_from_old: bool = True,
    ):
        super().__init__(
            config=old_cache.config,
            max_batch_size=old_cache.max_batch_size,
            cache_length=old_cache.cache_length,
            block_idx=old_cache.block_idx,
            dtype=old_cache.dtype,
            head_size=old_cache.head_size,
        )
        self.old_cache = old_cache
        self.new_cache = new_cache
        self._state_names = state_names.copy()
        self.return_from_old = return_from_old

    def _compare_state(self, prefix: str):
        for name in self._state_names:
            _compare_attribute(
                self.old_cache,
                self.new_cache,
                name,
                prefix,
                return_from_old=self.return_from_old,
            )

    @property
    def device(self) -> Optional[torch.device]:
        name = "device"
        return _compare_attribute(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
        )

    @property
    def dtype(self) -> Optional[torch.dtype]:
        name = "dtype"
        return _compare_attribute(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
        )

    @property
    def cache_length(self) -> int:
        name = "cache_length"
        return _compare_attribute(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
        )

    @property
    def n_query_groups(self) -> int:
        name = "n_query_groups"
        return _compare_attribute(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
        )

    @property
    def input_pos(self) -> int:
        name = "input_pos"
        return _compare_attribute(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
        )

    def forward(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        name = "forward"
        result = _run_and_compare(
            self.old_cache,
            self.new_cache,
            name,
            args,
            kwargs,
            return_from_old=self.return_from_old,
        )
        self._compare_state(f"After '{name}'")
        return result

    def get_keys_values(self) -> Optional[KeysAndValues]:
        name = "get_keys_values"
        result = _run_and_compare(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
        )
        self._compare_state(f"After '{name}'")
        return result

    def max_forward_length(self) -> int:
        name = "max_forward_length"
        result = _run_and_compare(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
        )
        self._compare_state(f"After '{name}'")
        return result

    @property
    def max_prefill_length(self) -> int:
        name = "max_prefill_length"
        return _compare_attribute(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
        )

    def get_params(self) -> KVCacheParams:
        name = "get_params"
        result = _run_and_compare(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
        )
        self._compare_state(f"After '{name}'")
        return result

    def token_positions(self) -> torch.Tensor:
        name = "token_positions"
        result = _run_and_compare(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
        )
        self._compare_state(f"After '{name}'")
        return result

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        name = "size_estimate"
        result = _run_and_compare(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
        )
        self._compare_state(f"After '{name}'")
        return result

    # TODO
    @classmethod
    def size_estimate_apriori(
        cls, params: KVCacheParams, **kwargs
    ) -> Tuple[int, Dict[str, int]]:
        raise NotImplementedError("Shadow size_estimate_apriori not implemented")

    def reset(self) -> None:
        name = "reset"
        result = _run_and_compare(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
        )
        # Skip comparison: For new implementation, `reset` is not called for
        # the policy
        # self._compare_state(f"After '{name}'")
        return result

    def set_seq_length(self, *args, **kwargs) -> None:
        name = "set_seq_length"
        result = _run_and_compare(
            self.old_cache,
            self.new_cache,
            name,
            args,
            kwargs,
            return_from_old=self.return_from_old,
        )
        self._compare_state(f"After '{name}'")
        return result

    def clone(self) -> "KVCache":
        name = "clone"
        # The returned objects are not equal, so do not compare them
        result = _run_and_compare(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
            do_compare=False,
        )
        self._compare_state(f"After '{name}'")
        return result

    def _base_kwargs_for_clone(self) -> Dict[str, Any]:
        name = "_base_kwargs_for_clone"
        result = _run_and_compare(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
        )
        self._compare_state(f"After '{name}'")
        return result

    @property
    def current_length(self) -> int:
        name = "current_length"
        return _compare_attribute(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
        )

    # kvcache.DefaultKVCache, kvcache.new.GenericKVCache

    @property
    def batch_size(self) -> Optional[int]:
        name = "batch_size"
        return _compare_attribute(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
        )

    def _forward_check_args(
        self,
        *args,
        **kwargs,
    ):
        name = "_forward_check_args"
        result = _run_and_compare(
            self.old_cache,
            self.new_cache,
            name,
            args,
            kwargs,
            return_from_old=self.return_from_old,
        )
        self._compare_state(f"After '{name}'")
        return result

    # Temporary attributes to run comparison testing
    # TODO: Remove them once comparison testing is done and old code
    # is removed.

    from keys_values.kvcache.buffers import KVCacheBuffers

    @property
    def mha(self) -> MultiHeadSelfAttention:
        name = "mha"
        return _compare_attribute(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
            do_compare=False,
        )

    @property
    def init_grace_tokens(self) -> int:
        name = "init_grace_tokens"
        return _compare_attribute(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
        )

    @property
    def scores(self) -> Optional[torch.Tensor]:
        name = "scores"
        return _compare_attribute(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
        )

    @property
    def kv_buffers(self) -> KVCacheBuffers:
        name = "kv_buffers"
        return _compare_attribute(
            self.old_cache,
            self.new_cache,
            name,
            return_from_old=self.return_from_old,
            do_compare=False,
        )

    # For new KVCache code, this method can be removed. There, a cache is
    # either recording or replaying at all times, or not at all, and the
    # switching beteen "record" and "play" works differently.
    def switch_replay_logging(self, status: bool) -> None:
        self.old_cache.switch_replay_logging(status)
        if not isinstance(self.new_cache, GenericKVCache):
            self.new_cache.switch_replay_logging(status)


def use_new_kvcache_code(cache: KVCache) -> bool:
    """
    Use this in code meant to work with new and old implementation of
    :class:`KVCache` subclasses (comparison tests, usage of
    :class:`TestShadowKVCache`).

    Args:
        cache: KV cache in question

    Returns:
        Using new code? If `cache` is of type :class:`TestShadowKVCache`,
        this depends on `cache.return_from_old`

    """
    return isinstance(cache, GenericKVCache) or (
        isinstance(cache, TestShadowKVCache) and not cache.return_from_old
    )


def use_new_replay_log(replay_log: KVCacheReplayLog_oldornew) -> bool:
    """
    Use this in code meant to work with new and old implementation of
    :class:`KVCache` subclasses (comparison tests, usage of
    :class:`TestShadowKVCache`).

    Args:
        replay_log: KV cache replay log in question

    Returns:
        Using new code? Depends on the concrete type of `replay_log`

    """
    return not isinstance(replay_log, KVCacheReplayLog_old)


def get_chunks_from_log(log: KVCacheReplayLog_oldornew) -> List[torch.Tensor]:
    """
     Use this in code meant to work with new and old implementation of
    :class:`KVCache` subclasses (comparison tests, usage of
    :class:`TestShadowKVCache`).

    Args:
        log: KV cache replay log (old or new type)

    Returns:
        List of token indexes, one for each chunk, from the log

    """
    if isinstance(log, KVCacheReplayLog_old):
        return log.token_chunks
    else:
        return [c.token_idx for c in log]


_GENERAL_STATE_NAMES = [
    "device",
    "dtype",
    "max_batch_size",
    "cache_length",
    "n_query_groups",
    "input_pos",
    "max_prefill_length",
    "batch_size",
    "current_length",
]

_SLR_STATE_NAMES = _GENERAL_STATE_NAMES + [
    "token_pos",
    "max_initial_fraction",
    "include_end_string",
    "range_is_prefix",
    "protected_start",
    "protected_end",
    "next_position",
    "_pad_id",
]

_ATTN_WEIGHTS_STATE_NAMES = _GENERAL_STATE_NAMES + [
    "token_pos",
    "_keep_initial_fraction",
    "grace_period",
    "_detach_attn_weights",
    "max_chunk_size",
    "_next_positions",
    "next_grace_pos",
    "prefill_length",
    "_use_initial_rule",
]

_H2O_STATE_NAMES = _ATTN_WEIGHTS_STATE_NAMES + [
    "scores",
    "normalize_scores",
]

_QH2O_STATE_NAMES = _H2O_STATE_NAMES + [
    "_scratch_blocksize",
    "combination_constant",
    "q_errors",
]

_STATE_NAMES = {
    "dense": _GENERAL_STATE_NAMES,
    "lastrec": _GENERAL_STATE_NAMES
    + ["token_pos", "init_grace_tokens", "next_position"],
    "smart-lastrec": _SLR_STATE_NAMES,
    "h2o": _H2O_STATE_NAMES,
    "h2o-vlen": _H2O_STATE_NAMES + ["v_norm"],
    "qh2o": _QH2O_STATE_NAMES,
    "qh2o-vlen": _QH2O_STATE_NAMES + ["v_norm"],
    "h2o-orig": _ATTN_WEIGHTS_STATE_NAMES + ["scores"],
}


class ShadowKVCacheFactory:
    """
    Same as :class:`KVCacheFactory`, but we also create KV cache objects from
    `keys_values.kvcache.new`, and return :class:`TestShadowKVCache` objects.
    """

    @staticmethod
    def create_single(
        name: str,
        config: Config,
        max_batch_size: int,
        cache_length: int,
        block_idx: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        return_from_old: bool = True,
    ) -> TestShadowKVCache:
        old_cache = KVCacheFactory_old.create_single(
            name,
            config,
            max_batch_size,
            cache_length,
            block_idx,
            device,
            dtype,
            None if cache_kwargs is None else cache_kwargs.copy(),
        )
        new_cache = KVCacheFactory_new.create_single(
            name,
            config,
            max_batch_size,
            cache_length,
            block_idx,
            device,
            dtype,
            None if cache_kwargs is None else cache_kwargs.copy(),
        )
        cname, _ = split_name(name)
        return TestShadowKVCache(
            old_cache=old_cache,
            new_cache=new_cache,
            state_names=_STATE_NAMES[cname],
            return_from_old=return_from_old,
        )

    @staticmethod
    def create(
        gpt_model: GPT,
        name: str,
        max_batch_size: int,
        cache_length: Union[int, List[int]],
        start: int = 0,
        end: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        return_from_old: bool = True,
    ) -> List[TestShadowKVCache]:
        old_caches = KVCacheFactory_old.create(
            gpt_model,
            name,
            max_batch_size,
            cache_length,
            start,
            end,
            device,
            dtype,
            None if cache_kwargs is None else cache_kwargs.copy(),
        )
        new_caches = KVCacheFactory_new.create(
            gpt_model,
            name,
            max_batch_size,
            cache_length,
            start,
            end,
            device,
            dtype,
            None if cache_kwargs is None else cache_kwargs.copy(),
        )
        cname, _ = split_name(name)
        state_names = _STATE_NAMES[cname]
        return [
            TestShadowKVCache(
                old_cache=old_cache,
                new_cache=new_cache,
                state_names=state_names,
                return_from_old=return_from_old,
            )
            for old_cache, new_cache in zip(old_caches, new_caches)
        ]
