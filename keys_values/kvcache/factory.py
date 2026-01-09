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
from dataclasses import replace
from typing import List, Dict, Any, Optional, Union, Tuple

import torch

from litgpt.config import Config

from keys_values.kvcache.base import KVCache, KVCacheParams
from keys_values.kvcache.basics import (
    DenseKVCache,
    LastRecentlyInsertedKVCache,
    KVCacheWithBuffers,
)
from keys_values.kvcache.buffers import KVCacheBuffersParams, DefaultKVCacheBuffers
from keys_values.kvcache.h2o import H2OKVCache, VLengthH2OKVCache, H2OOriginalKVCache
from keys_values.kvcache.qh2o import QuantizedH2OKVCache, QuantizedVLengthH2OKVCache
from keys_values.kvcache.quant_buffers import (
    QuantizedKVCacheBuffers,
    DequantizedKVCacheBuffers,
)
from keys_values.kvcache.quantize import (
    Quantizer,
    TorchAOQuantizer,
    TorchBasicQuantizer,
    BitsAndBytesQuantizer,
)
from keys_values.model import GPT, block_iterator


_SUPPORTED_CACHES = (
    ("dense", DenseKVCache, True),
    ("lastrec", LastRecentlyInsertedKVCache, True),
    ("h2o", H2OKVCache, True),
    ("h2o-vlen", VLengthH2OKVCache, True),
    ("qh2o", QuantizedH2OKVCache, False),
    ("qh2o-vlen", QuantizedVLengthH2OKVCache, False),
    ("h2o-orig", H2OOriginalKVCache, True),
)

SUPPORTED_QUANTIZERS = {
    "default": None,
    "torch-quantized8": TorchBasicQuantizer,
    "ao-quantized4": TorchAOQuantizer,
    "ao-quantized8": TorchAOQuantizer,
    "bnb-quantized4": BitsAndBytesQuantizer,
    "bnb-quantized8": BitsAndBytesQuantizer,
}

SUPPORTED_CACHES = {
    f"{name}-{quant}": typ
    for quant in SUPPORTED_QUANTIZERS.keys()
    for name, typ, do_def in _SUPPORTED_CACHES
    if do_def or quant != "default"
}


def split_name(name: str) -> Tuple[str, str]:
    for qname in SUPPORTED_QUANTIZERS.keys():
        if name.endswith(qname):
            return name[:-(len(qname) + 1)], qname
    raise ValueError(f"Name {name} is not supported")


class KVCacheFactory:
    """
    Factory for KV caches for a GPT model. Creates a list of :class:`KVCache`
    objects, one for each layer of the model.

    Supported caches have names in :const:`SUPPORTED_CACHES`. The postfix
    determines the type of buffers:

    - "default": Normal storage, no quantization
    - "torch-quantized8": Default PyTorch 8-bit quantization
    - "ao-quantized4": Torch AO 4-bit quantization
    - "ao-quantized8": Torch AO 8-bit quantization
    - "bnb-quantized4": Bitsandbytes 4-bit quantization (does not work for
      every hardware)
    - "bnb-quantized8": Bitsandbytes 8-bit quantization (does not work for
      every hardware)

    The prefix determines the type of cache:

    - "dense": :class:`DenseKVCache`, store all KV tensors up to max size
    - "lastrec": :class:`LastRecentlyInsertedKVCache`, store the last recently
      inserted `cache_length` KV tensors
    - "h2o": :class:`H2OKVCache`, use (improved) H2O criterion for eviction
      decisions
    - "h2o-vlen": Variant of H2O, which takes length of V vectors into account
    - "qh2o": :class:`QuantizedH2OKVCache`, use (improved) q-H2O (or Q-Hitter)
      criterion for eviction decisions. Only with quantized buffers
    - "qh2o-vlen": Variant of q-H2O, which takes length of V vectors into
      account. Only with quantized buffers
    - "h2o-orig": :class:`H2OOriginalKVCache`. Corresponds to the original H2O
      publication. Not recommended, only for comparisons

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
        """
        Args:
            name: Determines cache and buffers type, must be in
                :const:`SUPPORTED_CACHES`
            config: Model configuration
            max_batch_size: Maximum batch size for caches
            cache_length: Number of slots in caches
            block_idx: Index of block (or layer) in model
            device: Device for cache objects. If not given, this is determined
                with first usage
            dtype: Data type for cache buffers (de-quantized). If not given,
                this is determined with first usage
            cache_kwargs: Additional keyword arguments for cache creation

        Returns:
            KV cache object

        """
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
                    cache_length=cache_length,
                    block_idx=block_idx,
                    device=device,
                    dtype=dtype,
                    **cache_kwargs,
                )
                result = cache_type.from_config(**from_config_kwargs)
            else:
                cache_params = KVCacheBuffersParams.from_params(params)
                if device is not None:
                    cache_params = replace(cache_params, device=device)
                allocate_buffers = cache_kwargs.get("allocate_buffers")
                if allocate_buffers is not None:
                    cache_kwargs.pop("allocate_buffers")
                else:
                    allocate_buffers = False
                dequant_buffers = DequantizedKVCacheBuffers(
                    params=cache_params,
                    cache_length=cache_length,
                    max_num_ranges=max_num_ranges,
                )
                quant_kwargs, quantizer_type, cache_kwargs = KVCacheFactory.get_quant_kwargs(
                    params, qname, cache_kwargs,
                )
                quant_kwargs["allocate_buffers"] = allocate_buffers
                quant_kwargs["device"] = device
                result = cache_type(
                    config=config,
                    buffers=QuantizedKVCacheBuffers(
                        quantizer_k=quantizer_type(**quant_kwargs),
                        quantizer_v=quantizer_type(**quant_kwargs),
                        dequant_buffers=dequant_buffers,
                        debug_label=f"block{block_idx}",
                    ),
                    block_idx=block_idx,
                    **cache_kwargs,
                )
        else:
            if name in ("qh2o-default", "qh2o-vlen-default"):
                raise ValueError(f"{name[:-8]} can only be used with quantized buffers")
            else:
                raise ValueError(f"name = {name} not supported")
        return result

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
            raise ValueError(f"start={start}, end={end}, must be 0 <= start < end <= {config.n_layer}")
        num_layers = end - start
        if cache_kwargs is None:
            cache_kwargs = dict()
        if not isinstance(cache_length, list):
            cache_length = [cache_length] * num_layers
        elif len(cache_length) == 1:
            cache_length = [cache_length[0]] * num_layers
        elif len(cache_length) != num_layers:
            raise ValueError(f"len(cache_length) = {len(cache_length)}, must be 1 or {num_layers}")
        if not all(x > 0 for x in cache_length):
            raise ValueError(f"cache_length = {cache_length}, must contain only positive integers")

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
                cache_params = KVCacheParams(
                    max_batch_size=max_batch_size,
                    n_query_groups=config.n_query_groups,
                    cache_length=42,  # will be replaced
                    head_size=config.head_size,
                    n_head=config.n_head,
                    dtype=dtype,
                )
                allocate_buffers = cache_kwargs.get("allocate_buffers")
                if allocate_buffers is not None:
                    cache_kwargs.pop("allocate_buffers")
                else:
                    allocate_buffers = False
                dequant_kwargs = dict(max_num_ranges=max_num_ranges)
                quant_buffers = create_quantized_kv_buffers(
                    qname=qname,
                    cache_lengths=cache_length,
                    cache_params=cache_params,
                    cache_kwargs=cache_kwargs,
                    dequant_kwargs=dequant_kwargs,
                    allocate_buffers=allocate_buffers,
                    device=device,
                    first_block_idx=start,
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

    @staticmethod
    def supported_names() -> List[str]:
        return list(SUPPORTED_CACHES.keys())

    @staticmethod
    def size_estimate(model_or_caches: Union[GPT, List[KVCache]]) -> Tuple[int, Dict[str, int]]:
        """
        Args:
            model_or_caches: GPT model or list of KV caches (one per layer).
                For a model, we use its KV caches

        Returns:
            num_bits_total, bits_by_part (unit is bit)

        """
        caches = KVCacheFactory._get_caches(model_or_caches)
        num_bits_total = 0
        bits_by_part = dict()
        for layer_no, cache in enumerate(caches):
            total_sz, dct_sz = cache.size_estimate()
            num_bits_total += total_sz
            for k, v in dct_sz.items():
                bits_by_part[f"layer{layer_no}_{k}"] = v
        buffers_first = caches[0].kv_buffers
        if isinstance(buffers_first, QuantizedKVCacheBuffers):
            total_sz, dct_sz = buffers_first.dequant_buffers.size_estimate()
            num_bits_total += total_sz
            for k, v in dct_sz.items():
                bits_by_part[f"dequant_" + k] = v
        return num_bits_total, bits_by_part

    @staticmethod
    def size_estimate_apriori(
        name: str,
        config: Config,
        max_batch_size: int,
        cache_length: int,
        dtype: torch.dtype,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, int]]:
        if dtype is None:
            raise ValueError("dtype must be specified")
        params = KVCacheParams(
            max_batch_size=max_batch_size,
            n_query_groups=config.n_query_groups,
            cache_length=cache_length,
            head_size=config.head_size,
            n_head=config.n_head,
            dtype=dtype,
        )
        n_layers = config.n_layer
        if cache_kwargs is None:
            cache_kwargs = dict()

        cache_type = SUPPORTED_CACHES.get(name)
        cname, _ = split_name(name)
        if cache_type is not None:
            kwargs = dict()
            quantized_buffer = False
            cname, qname = split_name(name)
            if qname == "default":
                kwargs["buffer_type"] = DefaultKVCacheBuffers
            else:
                kwargs["buffer_type"] = QuantizedKVCacheBuffers
                quantized_buffer = True
                quant_kwargs, quantizer_type, cache_kwargs = KVCacheFactory.get_quant_kwargs(
                    params, qname, cache_kwargs,
                )
                kwargs["quantizer_type"] = quantizer_type
                # Extra arguments going to the cache buffer
                kwargs.update(quant_kwargs)
            total_sz, dct_sz = cache_type.size_estimate_apriori(params, **kwargs)
            num_bits_total = total_sz * n_layers
            # Note: Only one entry for all layers, they are all the same
            bits_by_part = {
                "layer_" + k: v * n_layers for k, v in dct_sz.items()
            }
            if quantized_buffer:
                total_sz, dct_sz = DequantizedKVCacheBuffers.size_estimate_apriori(
                    params=KVCacheBuffersParams.from_params(params),
                    cache_length=cache_length,
                )
                num_bits_total += total_sz
                bits_by_part.update(
                    {
                        "dequant_" + k: v for k, v in dct_sz.items()
                    }
                )
            return num_bits_total, bits_by_part
        else:
            raise ValueError(f"name = {name} not supported")

    @staticmethod
    def _get_caches(model_or_caches: Union[GPT, List[KVCache]]) -> List[KVCache]:
        if isinstance(model_or_caches, GPT):
            caches = [block.attn.kv_cache for block in model_or_caches.transformer.h]
            if any(cache is None for cache in caches):
                raise IndexError("Some layers of model do not have a cache. Run 'assign_kv_cache' or 'set_kv_cache' first.")
        else:
            caches = model_or_caches
        return caches

    @staticmethod
    def get_quant_kwargs(
        params: KVCacheParams,
        qname: str,
        cache_kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], type[Quantizer], Dict[str, Any]]:
        quantizer_type = SUPPORTED_QUANTIZERS[qname]
        if quantizer_type is None:
            raise NotImplementedError(f"Quantizer type {qname} currently not supported")
        if cache_kwargs is None:
            cache_kwargs = dict()
        else:
            cache_kwargs = cache_kwargs.copy()
        try:
            blocks_over_heads = cache_kwargs.pop("blocks_over_heads")
        except KeyError:
            blocks_over_heads = False
        if blocks_over_heads is False and params.head_size < quantizer_type.minimum_blocksize():
            print(f"blocksize = {params.head_size} too small. Switching to blocks_over_heads = True.")
            blocks_over_heads = True
        shape = (
            params.max_batch_size,
            params.n_query_groups,
            params.cache_length,
            params.head_size,
        )
        quant_kwargs = {
            "shape": shape,
            "source_dtype": params.dtype,
            "blocks_over_heads": blocks_over_heads,
            "tmp_array_limit_gb": cache_kwargs.get("tmp_array_limit_gb"),
        }
        if qname.endswith("torch-quantized8"):
            quant_kwargs["target_dtype"] = torch.int8
        else:
            quant_kwargs["num_bits"] = int(qname[-1])
        return quant_kwargs, quantizer_type, cache_kwargs


def create_quantized_kv_buffers(
    qname: str,
    cache_lengths: List[int],
    cache_params: KVCacheParams,
    cache_kwargs: Optional[Dict[str, Any]] = None,
    dequant_kwargs: Optional[Dict[str, Any]] = None,
    allocate_buffers: bool = False,
    device: Optional[torch.device] = None,
    first_block_idx: Optional[int] = None,
) -> List[QuantizedKVCacheBuffers]:
    """
    Creates a list of :class:`QuantizedKVCacheBuffers` for cache lenghts in
    `cache_lenghts`. All buffers share one :class:`DequantizedKVCacheBuffers`
    object.

    """
    buffer_params = replace(
        KVCacheBuffersParams.from_params(cache_params),
        device=device,
    )
    max_cache_length = max(cache_lengths)
    if dequant_kwargs is None:
        dequant_kwargs = dict()
    dequant_buffers = DequantizedKVCacheBuffers(
        params=buffer_params,
        cache_length=max_cache_length,
        allocate_buffers=allocate_buffers,
        **dequant_kwargs,
    )
    quant_buffers = []
    for i, cache_length in enumerate(cache_lengths):
        _cache_params = replace(
            cache_params,
            cache_length=cache_length,
        )
        quant_kwargs, quantizer_type, cache_kwargs = KVCacheFactory.get_quant_kwargs(
            _cache_params, qname, cache_kwargs,
        )
        quant_kwargs["allocate_buffers"] = allocate_buffers
        quant_kwargs["device"] = device
        if first_block_idx is not None:
            kwargs = dict(debug_label=f"block{first_block_idx + i}")
        else:
            kwargs = dict()
        quant_buffers.append(
            QuantizedKVCacheBuffers(
                quantizer_k=quantizer_type(**quant_kwargs),
                quantizer_v=quantizer_type(**quant_kwargs),
                dequant_buffers=dequant_buffers,
                **kwargs,
            )
        )
    return quant_buffers


def deallocate_kv_cache_buffers(caches: List[KVCache]):
    """
    Deallocates buffers of KV caches in `caches`. Use this to free up GPU memory
    once the caches are not needed for the moment (buffers are reallocated on
    first usage), but also in preparation of cloning the caches on a different
    device (this works only if buffers are deallocated).

    """
    for cache in caches:
        if cache is not None and isinstance(cache, KVCacheWithBuffers):
            buffers = cache.kv_buffers
            buffers.deallocate()
            # Deallocate associated :class:`DequantizedKVCacheBuffers` buffers
            # as well. They may be shared by several entries in `caches`, but
            # calling :meth:`deallocate` is fine.
            if isinstance(buffers, QuantizedKVCacheBuffers):
                buffers.dequant_buffers.deallocate()


def deallocate_kv_cache_buffers_of_model(model: GPT):
    """
    Deallocates buffers of KV caches associated with `model`. Use this to free
    up GPU memory once the caches are not needed for the moment (buffers are
    reallocated on first usage), but also in preparation of cloning the caches
    on a different device (this works only if buffers are deallocated).

    """
    kv_caches = [block.attn.kv_cache for block in block_iterator(model)]
    deallocate_kv_cache_buffers(kv_caches)


REMOVE_KEYS = {
    "dense": ("replay_log_blocksize", "grace_period", "detach_attn_weights", "keep_initial_fraction", "normalize_scores", "combination_constant", "scratch_blocksize", "max_chunk_size"),
    "lastrec": ("replay_log_blocksize", "grace_period", "detach_attn_weights", "keep_initial_fraction", "normalize_scores" "combination_constant", "scratch_blocksize", "max_chunk_size"),
    "h2o": ("combination_constant", "scratch_blocksize"),
    "h2o-vlen": ("combination_constant", "scratch_blocksize"),
    "qh2o": (),
    "qh2o-vlen": (),
    "h2o-orig": ("normalize_scores", "combination_constant", "scratch_blocksize"),
}


def cleanup_cache_kwargs(
    cname: str, cache_kwargs: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if cache_kwargs is None:
        return dict()
    rem_keys = REMOVE_KEYS[cname]
    return {k: v for k, v in cache_kwargs.items() if k not in rem_keys}
