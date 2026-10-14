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

from keys_values.attention import KeysAndValues
from keys_values.config import Config
from keys_values.kvcache.base import KVCache, KVCacheParams
from keys_values.kvcache.consts import split_name
from keys_values.kvcache.factory import (
    KVCacheFactory,
    SUPPORTED_CACHES,
    create_single_quantized_buffer,
    create_multiple_quantized_buffers,
)
from keys_values.kvcache.new import (
    GenericKVCache,
    KVCacheDriverWithBuffers,
    DenseKVCachePolicy,
    LastRecentlyInsertedKVCachePolicy,
    SmartInitialLastRecentlyInsertedKVCachePolicy,
    H2OKVCachePolicy,
    VLengthH2OKVCachePolicy,
    QuantizedH2OKVCachePolicy,
    QuantizedVLengthH2OKVCachePolicy,
    H2OOriginalKVCachePolicy,
)
from keys_values.kvcache.new.driver import DefaultKVCacheDriver
from keys_values.model import GPT


def _error_message(prefix: str, msg: str) -> str:
    return prefix + ":\n" + msg


def compare(
    old_v: Any,
    new_v: Any,
    prefix: str,
):
    assert type(old_v) == type(new_v), (
        f"{prefix}: type(old_v) = {type(old_v)} != {type(new_v)} = type(new_v)"
    )
    # TODO: More specializations for types!
    if isinstance(old_v, torch.Tensor):
        assert old_v.shape == new_v.shape, (
            f"{prefix}: old_v.shape = {old_v.shape}, new_v.shape = {new_v.shape}"
        )
        assert old_v.dtype == new_v.dtype, (
            f"{prefix}: old_v.dtype = {old_v.dtype}, new_v.dtype = {new_v.dtype}"
        )
        torch.testing.assert_close(
            old_v,
            new_v,
            msg=partial(_error_message, prefix=prefix + ": Tensor"),
        )
    elif isinstance(old_v, list) or isinstance(old_v, tuple):
        assert len(old_v) == len(new_v), (
            f"{prefix}: len(old_v) = {len(old_v)} != {len(new_v)} = len(new_v)"
        )
        for pos, (ov, nv) in enumerate(zip(old_v, new_v)):
            compare(ov, nv, f"{prefix}[{pos}]")
    elif isinstance(old_v, dict):
        old_keys = set(old_v.keys())
        new_keys = set(new_v.keys())
        assert old_keys == new_keys, (
            f"{prefix}: old_v, new_v dicts with different keys:\n"
            f"old_v.keys: {old_keys}\nnew_v.keys: {new_keys}"
        )
        for key, ov in old_v.items():
            nv = new_v[key]
            compare(ov, nv, f"{prefix}[{key}]")
    else:
        assert old_v == new_v, f"{prefix}:\nold_v = {old_v}\nnew_v = {new_v}"


def run_and_compare(
    old_cache: KVCache,
    new_cache: KVCache,
    name: str,
    args: Optional[Tuple[Any]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = dict()
    old_result = getattr(old_cache, name)(*args, **kwargs)
    new_result = getattr(new_cache, name)(*args, **kwargs)
    no_tuple = not isinstance(old_result, tuple)
    if no_tuple:
        old_result = (old_result,)
    if not isinstance(new_result, tuple):
        new_result = (new_result,)
    assert len(old_result) == len(new_result), (
        f"obj.{name}(...): len(old_result) = {len(old_result)} != {len(new_result)} = len(new_result)"
    )
    for pos, (old_v, new_v) in enumerate(zip(old_result, new_result)):
        compare(old_v, new_v, f"obj.{name}(...)[{pos}]")
    return old_result[0] if no_tuple else new_result


def compare_attribute(
    old_cache: KVCache,
    new_cache: KVCache,
    name: str,
    prefix: Optional[str] = None,
) -> Any:
    old_v = getattr(old_cache, name)
    new_objs = [new_cache]
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
        raise AttributeError(
            f"{_prefix} does not exist in new_cache or its components"
        )
    compare(old_v, new_v, _prefix)
    return old_v


class TestShadowKVCache(KVCache):
    """
    Helper for old/new comparison testing. Holds a pair of :class:`KVCache`
    objects, `old_cache` and `new_cache`. All calls are served from `new_cache`.
    But for each call, `old_cache` is called with the same args, and the
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

    def _compare_state(self, prefix: str):
        for name in self._state_names:
            compare_attribute(
                self.old_cache,
                self.new_cache,
                name,
                prefix,
            )

    @property
    def device(self) -> Optional[torch.device]:
        name = "device"
        return compare_attribute(self.old_cache, self.new_cache, name)

    @property
    def dtype(self) -> Optional[torch.dtype]:
        name = "dtype"
        return compare_attribute(self.old_cache, self.new_cache, name)

    @property
    def cache_length(self) -> int:
        name = "cache_length"
        return compare_attribute(self.old_cache, self.new_cache, name)

    @property
    def n_query_groups(self) -> int:
        name = "n_query_groups"
        return compare_attribute(self.old_cache, self.new_cache, name)

    @property
    def input_pos(self) -> int:
        name = "input_pos"
        return compare_attribute(self.old_cache, self.new_cache, name)

    def forward(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        name = "forward"
        result = run_and_compare(
            self.old_cache, self.new_cache, name, args, kwargs
        )
        self._compare_state(f"After {name}")
        return result

    def get_keys_values(self) -> Optional[KeysAndValues]:
        name = "get_keys_values"
        result = run_and_compare(
            self.old_cache, self.new_cache, name,
        )
        self._compare_state(f"After {name}")
        return result

    def max_forward_length(self) -> int:
        name = "max_forward_length"
        result = run_and_compare(
            self.old_cache, self.new_cache, name,
        )
        self._compare_state(f"After {name}")
        return result

    @property
    def max_prefill_length(self) -> int:
        name = "max_prefill_length"
        return compare_attribute(self.old_cache, self.new_cache, name)

    def get_params(self) -> KVCacheParams:
        name = "get_params"
        result = run_and_compare(
            self.old_cache, self.new_cache, name,
        )
        self._compare_state(f"After {name}")
        return result

    def token_positions(self) -> torch.Tensor:
        name = "token_positions"
        result = run_and_compare(
            self.old_cache, self.new_cache, name,
        )
        self._compare_state(f"After {name}")
        return result

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        name = "size_estimate"
        result = run_and_compare(
            self.old_cache, self.new_cache, name,
        )
        self._compare_state(f"After {name}")
        return result

    # TODO!!
    @classmethod
    def size_estimate_apriori(
        cls, params: KVCacheParams, **kwargs
    ) -> Tuple[int, Dict[str, int]]:
        raise NotImplementedError("Cannot shadow size_estimate_apriori")

    def reset(self) -> None:
        name = "reset"
        result = run_and_compare(
            self.old_cache, self.new_cache, name,
        )
        self._compare_state(f"After {name}")
        return result

    def set_seq_length(self, *args, **kwargs) -> None:
        name = "set_seq_length"
        result = run_and_compare(
            self.old_cache, self.new_cache, name, args, kwargs
        )
        self._compare_state(f"After {name}")
        return result

    def clone(self) -> "KVCache":
        name = "clone"
        result = run_and_compare(
            self.old_cache, self.new_cache, name,
        )
        self._compare_state(f"After {name}")
        return result

    def _base_kwargs_for_clone(self) -> Dict[str, Any]:
        name = "_base_kwargs_for_clone"
        result = run_and_compare(
            self.old_cache, self.new_cache, name,
        )
        self._compare_state(f"After {name}")
        return result

    # kvcache.DefaultKVCache, kvcache.new.GenericKVCache

    @property
    def batch_size(self) -> Optional[int]:
        name = "batch_size"
        result = run_and_compare(
            self.old_cache, self.new_cache, name,
        )
        self._compare_state(f"After {name}")
        return result

    def _forward_check_args(
        self, *args, **kwargs,
    ):
        name = "_forward_check_args"
        result = run_and_compare(
            self.old_cache, self.new_cache, name, args, kwargs
        )
        self._compare_state(f"After {name}")
        return result


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

_ATTN_WEIGHTS_STATE_NAMES = _GENERAL_STATE_NAMES + [
    "token_pos",
    "_keep_initial_fraction",
    "grace_period",
    "_detach_attn_weights",
    "max_chunk_size",
    "_next_positions",
    "next_grace_positions",
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
    "lastrec": _GENERAL_STATE_NAMES + ["token_pos", "init_grace_tokens", "next_position"],
    "smart-lastrec": _GENERAL_STATE_NAMES + ["token_pos", "max_initial_fraction", "include_end_string", "range_is_prefix", "protected_start", "protected_end", "next_position", "_pad_id"],
    "h2o": _H2O_STATE_NAMES,
    "h2o-vlen": _H2O_STATE_NAMES + ["v_norm"],
    "qh2o": _QH2O_STATE_NAMES
    "qh2o-vlen": _QH2O_STATE_NAMES + ["v_norm"],
    "h2o-orig": _ATTN_WEIGHTS_STATE_NAMES + ["scores"],
}

_SUPPORTED_POLICIES = {
    "dense": DenseKVCachePolicy,
    "lastrec": LastRecentlyInsertedKVCachePolicy,
    "smart-lastrec": SmartInitialLastRecentlyInsertedKVCachePolicy,
    "h2o": H2OKVCachePolicy,
    "h2o-vlen": VLengthH2OKVCachePolicy,
    "qh2o": QuantizedH2OKVCachePolicy,
    "qh2o-vlen": QuantizedVLengthH2OKVCachePolicy,
    "h2o-orig": H2OOriginalKVCachePolicy,
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
    ) -> TestShadowKVCache:
        old_cache = KVCacheFactory.create_single(
            name,
            config,
            max_batch_size,
            cache_length,
            block_idx,
            device,
            dtype,
            None if cache_kwargs is None else cache_kwargs.copy(),
        )

        params = KVCacheParams(
            max_batch_size=max_batch_size,
            n_query_groups=config.n_query_groups,
            cache_length=cache_length,
            head_size=config.head_size,
            n_head=config.n_head,
            dtype=dtype,
        )
        if cache_kwargs is None:
            cache_kwargs = dict()
        max_num_ranges = cache_kwargs.get("max_num_ranges")
        if max_num_ranges is not None:
            cache_kwargs.pop("max_num_ranges")
        cname, qname = split_name(name)
        if qname == "default":
            # HIER: Create default `buffers`
            from_config_kwargs = dict(
                config=config,
                max_batch_size=max_batch_size,
                cache_length=cache_length,
                block_idx=block_idx,
                device=device,
                dtype=dtype,
                **cache_kwargs,
            )
        else:
            buffers, cache_kwargs = create_single_quantized_buffer(
                params,
                cache_length,
                max_num_ranges,
                block_idx,
                qname,
                device,
                dtype,
                cache_kwargs,
            )
        # HIER: Policy creation kwargs
        policy = _SUPPORTED_POLICIES[cname](...)
        driver = KVCacheDriverWithBuffers(
            config=config,
            buffers=buffers,
            block_idx=block_idx,
        )
        new_cache = GenericKVCache(policy=policy, driver=driver)

        return TestShadowKVCache(
            old_cache=old_cache,
            new_cache=new_cache,
            state_names=_STATE_NAMES[cname],
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
    ) -> List[KVCache]:
        """
        By default, caches are created for all layers of the model. A subrange
        `range(start, end)` can be chosen, for example if not all layers should
        use the same `name`.

        Args:
            gpt_model: GPT model for which KV caches are to be created
            name: Determines cache and buffers type, must be in
                :const:`SUPPORTED_CACHES`
            max_batch_size: Maximum batch size for caches
            cache_length: Number of slots in caches. Can be different for
                different caches
            start: Caches are created for layers in `range(start, end)`
            end: Caches are created for layers in `range(start, end)`
            device: Device for cache objects. If not given, this is determined
                with first usage
            dtype: Data type for cache buffers (de-quantized). If not given,
                this is determined with first usage
            cache_kwargs: Additional keyword arguments for cache creation

        Returns:
            KV cache objects

        """
        config = gpt_model.config
        if end is None:
            end = config.n_layer
        if not (0 <= start < end <= config.n_layer):
            raise ValueError(
                f"start={start}, end={end}, must be 0 <= start < end <= {config.n_layer}"
            )
        num_layers = end - start
        if cache_kwargs is None:
            cache_kwargs = dict()
        if not isinstance(cache_length, list):
            cache_length = [cache_length] * num_layers
        elif len(cache_length) == 1:
            cache_length = [cache_length[0]] * num_layers
        elif len(cache_length) != num_layers:
            raise ValueError(
                f"len(cache_length) = {len(cache_length)}, must be 1 or {num_layers}"
            )
        if not all(x > 0 for x in cache_length):
            raise ValueError(
                f"cache_length = {cache_length}, must contain only positive integers"
            )

        cache_type = SUPPORTED_CACHES.get(name)
        if cache_type is not None:
            max_num_ranges = cache_kwargs.get("max_num_ranges")
            if max_num_ranges is not None:
                cache_kwargs.pop("max_num_ranges")
            cname, qname = split_name(name)
            if qname == "default":
                from_config_kwargs = dict(
                    config=config,
                    max_batch_size=max_batch_size,
                    dtype=dtype,
                    device=device,
                    **cache_kwargs,
                )
                kv_caches = [
                    cache_type.from_config(
                        **from_config_kwargs,
                        cache_length=c_len,
                        block_idx=i + start,
                    )
                    for i, c_len in enumerate(cache_length)
                ]
            else:
                quant_buffers, cache_kwargs = create_multiple_quantized_buffers(
                    config,
                    qname,
                    max_batch_size,
                    cache_length,
                    max_num_ranges,
                    start,
                    device,
                    dtype,
                    cache_kwargs,
                )
                kv_caches = [
                    cache_type(
                        config=config,
                        buffers=buffers,
                        block_idx=i + start,
                        **cache_kwargs,
                    )
                    for i, buffers in enumerate(quant_buffers)
                ]
            return kv_caches
        else:
            if name in ("qh2o-default", "qh2o-vlen-default"):
                raise ValueError(f"{name[:-8]} can only be used with quantized buffers")
            else:
                raise ValueError(f"name = {name} not supported")
