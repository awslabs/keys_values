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
from typing import List, Any, Dict, Optional, Tuple, Union

import torch

from keys_values.config import Config
from keys_values.constants import DEFAULT_PAD_ID
from keys_values.kvcache.base import (
    KVCache,
    KVCacheParams,
)
from keys_values.kvcache.consts import split_name
from keys_values.kvcache.factory import (
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
from keys_values.kvcache.qh2o import (
    DEFAULT_COMBINATION_CONSTANT,
    DEFAULT_SCRATCH_BLOCKSIZE,
)
from keys_values.model import GPT

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


def _extract_policy_kwargs(
    cname: str,
    cache_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    names_and_defaults: List[Tuple[str, Any, bool]] = [
        ("use_replay_logging", True, False)
    ]
    if cname == "lastrec":
        names_and_defaults.append(("init_grace_tokens", None, False))
    elif cname == "smart-lastrec":
        names_and_defaults.extend(
            [
                ("tokenizer", None, True),
                ("end_initial_regex", None, True),
                ("max_initial_fraction", None, True),
                ("max_batch_size", None, True),
                ("include_end_string", True, False),
                ("pad_id", DEFAULT_PAD_ID, False),
                ("range_is_prefix", True, False),
            ]
        )
    elif cname != "dense":
        if cname != "h2o-orig":
            names_and_defaults.extend(
                [
                    ("grace_period", 0, False),
                    ("detach_attn_weights", False, False),
                    ("keep_initial_fraction", None, False),
                    ("max_chunk_size", None, False),
                    ("normalize_scores", False, False),
                ]
            )
            if cname.startswith("qh2o"):
                names_and_defaults.extend(
                    [
                        ("normalize_scores", False, False),
                        ("combination_constant", DEFAULT_COMBINATION_CONSTANT, False),
                        ("scratch_blocksize", DEFAULT_SCRATCH_BLOCKSIZE, False),
                    ]
                )
            else:
                names_and_defaults.append(("max_batch_size", None, True))
        else:
            names_and_defaults.append(("max_batch_size", None, True))

    result = dict()
    for name, defval, mandatory in names_and_defaults:
        if name in cache_kwargs:
            result[name] = cache_kwargs[name]
        elif mandatory:
            raise ValueError(
                f"Policy parameter '{name}' is mandatory, but not in cache_kwargs"
            )
        else:
            result[name] = defval
    return result


# TODO: Static methods from old `KVCacheFactory` is missing here
class KVCacheFactory:
    """
    Replacement for :class:`keys_values.kvcache.KVCacheFactory`: we create KV
    cache objects from `keys_values.kvcache.new`.
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
    ) -> KVCache:
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
            buffers_kwargs = KVCacheDriverWithBuffers.extract_default_buffers_kwargs(
                cache_kwargs
            )
            buffers = KVCacheDriverWithBuffers.create_default_buffers(
                config=config,
                max_batch_size=max_batch_size,
                cache_length=cache_length,
                device=device,
                dtype=dtype,
                **buffers_kwargs,
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
        cache_kwargs["max_batch_size"] = max_batch_size
        if not cname.startswith("qh2o"):
            policy = _SUPPORTED_POLICIES[cname](
                config=config,
                cache_length=cache_length,
                **_extract_policy_kwargs(cname, cache_kwargs),
            )
        else:
            if qname == "default":
                raise ValueError(
                    f"name = {name} invalid. Policy {cname} needs quantized buffer"
                )
            policy = _SUPPORTED_POLICIES[cname](
                config=config,
                kv_buffers=buffers,
                **_extract_policy_kwargs(cname, cache_kwargs),
            )

        driver = KVCacheDriverWithBuffers(
            config=config,
            buffers=buffers,
            block_idx=block_idx,
        )
        return GenericKVCache(policy=policy, driver=driver)

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

        max_num_ranges = cache_kwargs.get("max_num_ranges")
        if max_num_ranges is not None:
            cache_kwargs.pop("max_num_ranges")
        cname, qname = split_name(name)
        if qname == "default":
            buffers_kwargs = KVCacheDriverWithBuffers.extract_default_buffers_kwargs(
                cache_kwargs
            )
            buffers_list = [
                KVCacheDriverWithBuffers.create_default_buffers(
                    config=config,
                    max_batch_size=max_batch_size,
                    cache_length=c_len,
                    device=device,
                    dtype=dtype,
                    **buffers_kwargs,
                )
                for c_len in cache_length
            ]
        else:
            buffers_list, cache_kwargs = create_multiple_quantized_buffers(
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
        kv_caches = []
        policy_kwargs = _extract_policy_kwargs(cname, cache_kwargs)
        for i, (buffers, c_len) in enumerate(zip(buffers_list, cache_length)):
            policy = _SUPPORTED_POLICIES[cname](
                config=config,
                cache_length=c_len,
                **policy_kwargs,
            )
            driver = KVCacheDriverWithBuffers(
                config=config,
                buffers=buffers,
                block_idx=i + start,
            )
            kv_caches.append(GenericKVCache(policy=policy, driver=driver))

        return kv_caches
