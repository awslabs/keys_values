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
from typing import List, Any, Dict, Optional, Tuple

import torch

from keys_values.attention import KeysAndValues
from keys_values.kvcache.base import KVCache, KVCacheParams


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
    new_v = getattr(new_cache, name)
    _prefix = f"obj.{name}"
    if prefix:
        _prefix = prefix + ": " + _prefix
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

    Note: We provide more than the base class methods and attributes here,
    so that comparisons become more fine-grained. But only methods present in
    both old and new cache types can be represented.

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

    # DefaultKVCache

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

    def _convert_or_check_device(
        self, *args, **kwargs,
    ):
        name = "_convert_or_check_device"
        result = run_and_compare(
            self.old_cache, self.new_cache, name, args, kwargs
        )
        self._compare_state(f"After {name}")
        return result

    # TODO: ???
    @classmethod
    def _parameter_names(cls) -> List[str]:
        """
        Returns:
            Names of `torch.Tensor` parameters which need to be checked for
            device or converted.

        """
        return []

    def _prefill(
        self, *args, **kwargs,
    ):
        name = "_prefill"
        result = run_and_compare(
            self.old_cache, self.new_cache, name, args, kwargs
        )
        self._compare_state(f"After {name}")
        return result

    def _forward(
        self, *args, **kwargs,
    ) -> KeysAndValues:
        name = "_forward"
        result = run_and_compare(
            self.old_cache, self.new_cache, name, args, kwargs
        )
        self._compare_state(f"After {name}")
        return result

    def _update(self, *args, **kwargs):
        name = "_update"
        result = run_and_compare(
            self.old_cache, self.new_cache, name, args, kwargs
        )
        self._compare_state(f"After {name}")
        return result

    def update_requires_attn_weights(self) -> bool:
        name = "update_requires_attn_weights"
        result = run_and_compare(
            self.old_cache, self.new_cache, name,
        )
        self._compare_state(f"After {name}")
        return result

    def _reset(self):
        name = "_reset"
        result = run_and_compare(
            self.old_cache, self.new_cache, name,
        )
        self._compare_state(f"After {name}")
        return result
