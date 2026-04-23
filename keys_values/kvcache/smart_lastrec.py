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
import copy
from dataclasses import dataclass
import re
from typing import Optional, Tuple, Dict, List, Any, Union

from tokenizers import Tokenizer
import torch

from keys_values.attention import KeysAndValues
from keys_values.kvcache.base import (
    KVCache,
    DefaultKVCacheReplayLog,
    KVCacheParams,
    KVCacheReplayLog,
)
from keys_values.kvcache.basics import KVCacheWithBuffers
from keys_values.kvcache.buffers import (
    DefaultKVCacheBuffers,
    KVCacheBuffers,
    KVCacheBuffersParams,
    positions_wrap_around,
)
from keys_values.config import Config
from keys_values.utils import bits_for_torch_dtype, bitsize_of


@dataclass(frozen=True)
class SmartInitialInformation:
    """
    Collects parameters of :class:`SmartInitialLastRecentlyInsertedKVCache`.
    The idea is that datasets can define default values for them, see for
    example
    :meth:`keys_values.data.longbench_v2.LongBenchV2.smart_lastrec_info`.

    Args:
        end_initial_regex: Regular expression for end of initial part sequence
        max_initial_fraction: Fraction of cache length initial parts can
            occupy at most
        include_end_string: Include end of init sequence in initial part?

    """
    end_initial_regex: Union[str, re.Pattern]
    max_initial_fraction: float
    include_end_string: bool = True

    def __post_init__(self):
        if self.max_initial_fraction == 0:
            raise ValueError("Use LastRecentlyInsertedKVCache for cache without initial parts")
        if not (0 < self.max_initial_fraction < 1):
            raise ValueError(f"max_initial_fraction = {self.max_initial_fraction}, must be in (0, 1)")


def end_initial_regex_from_string(
    s: str, tokenizer: Tokenizer,
) -> str:
    return re.escape(
        tokenizer.decode(tokenizer.encode(s), skip_special_tokens=True)
    )


class SmartInitialLastRecentlyInsertedKVCacheReplayLog(DefaultKVCacheReplayLog):
    """
    Replay log for :class:`SmartInitialLastRecentlyInsertedKVCache`.
    """

    def __init__(
        self,
        token_chunks: List[torch.Tensor],
        cache_length: int,
        n_query_groups: int,
        init_length: List[int],
    ):
        super().__init__(
            token_chunks,
            cache_length,
            max_prefill_length=cache_length,
            grace_period=0,
        )
        self.n_query_groups = n_query_groups
        self.init_length = init_length.copy()

    def extract_index(
        self,
        input_pos: int,
        num: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        seq_length = self.__len__()
        if num <= 0 or input_pos < 0 or input_pos > seq_length - num:
            raise ValueError(
                f"token_pos = {input_pos}, num = {num}, seq_length = {seq_length}: Out of range"
            )
        if input_pos < self.cache_length:
            raise ValueError(
                f"input_pos = {input_pos} must be >= {self.cache_length} = cache_length"
            )
        if device is None:
            device = torch.get_default_device()
        max_one_step = self.cache_length - max(self.init_length)
        if num <= max_one_step:
            offs_and_nums = ((0, num),)
        else:
            # Only happens if `num` is very large
            # In this case, we create two indexes and concatenate them. This may
            # not be exactly correct, because of duplicate entries. Very large
            # chunks should be avoided!
            offs_and_nums = ((0, max_one_step), (max_one_step, num - max_one_step),)
        positions = []
        for off, _num in offs_and_nums:
            _input_pos = input_pos + off
            position_parts = []
            for start in self.init_length:
                mod = self.cache_length - start
                current = (_input_pos - self.cache_length) % mod + start
                position_parts.append(
                    positions_wrap_around(
                        num=num,
                        current=current,
                        start=start,
                        end=self.cache_length,
                        batch_size=1,
                        n_query_groups=self.n_query_groups,
                        device=device,
                        return_tensor=True,
                        dtype=dtype,
                    )
                )
            positions.append(torch.cat(position_parts, dim=0))

        if len(positions) == 1:
            return positions[0]
        else:
            return torch.cat(positions, dim=-1)


class SmartInitialLastRecentlyInsertedKVCache(KVCacheWithBuffers):
    """
    Advanced variant of :class:`LastRecentlyInsertedKVCache`. Here, the initial
    "grace" part kept in the cache indefinitely can be of different length for
    each batch position and depends on the prompt content.

    A regular expression `end_initial_regex` is passed at construction.
    The length of initial parts (in tokens) per batch position is kept in
    `init_length`. It is determined in :meth:`prefill`. Namely,
    `init_length[b]` is the smallest of:

    * `end` (if `include_end_string == True`) or `start` (if
      `include_end_string == False`) if the first match of `end_initial_regex`
      in the string decoded from the token sequence `token_idx[b, :]` has span
      `(start, end)`.
    * `int(max_initial_fraction * cache_length)`
    * Prefill length `key.shape[2]`

    In other words, the initial parts are terminated by a certain string
    pattern described by `end_initial_regex`, and the pattern is included in the
    initial parts iff `include_end_string == True`. However, both
    `int(max_initial_fraction * cache_length)` and the prefill length limit the
    initial part lengths.

    ATTENTION: The regular expression `end_initial_regex` is matched against
    a string coming out of `tokenizer.decode(..., skip_special_tokens=True)`. It
    needs to account for the fact that in general, `decode(encode(text)) != text`,
    and differences may depend on the tokenizer (treatment of whitespace, newline,
    inserting of whitespace, etc.). If `end_initial_regex` is just a substring to
    be matched, we recommend to use :func:`end_initial_regex_from_string`, passing
    the tokenizer.
    """

    def __init__(
        self,
        config: Config,
        buffers: KVCacheBuffers,
        block_idx: int,
        tokenizer: Tokenizer,
        end_initial_regex: Union[str, re.Pattern],
        max_initial_fraction: float,
        include_end_string: bool = True,
        pad_id: int = 0,
        **base_kwargs,
    ):
        super().__init__(config, buffers, block_idx, **base_kwargs)
        self.tokenizer = tokenizer
        if not isinstance(end_initial_regex, re.Pattern):
            end_initial_regex = re.compile(end_initial_regex)
        self.end_initial_regex = end_initial_regex
        if max_initial_fraction == 0:
            raise ValueError("Use LastRecentlyInsertedKVCache for cache without initial parts")
        if not (0 < max_initial_fraction < 1):
            raise ValueError(f"max_initial_fraction = {max_initial_fraction}, must be in (0, 1)")
        self.max_initial_fraction = max_initial_fraction
        self.include_end_string = include_end_string
        # Lengths of initial parts (per batch dimension)
        self.init_length = None
        # Note: We could generate `token_pos` on the fly from `input_pos`,
        # `next_position` and `init_length`, but it is simpler just to maintain
        # it.
        device = self._default_device_for_new_params()
        self.register_buffer(
            "token_pos",
            torch.zeros(
                (buffers.max_batch_size, buffers.cache_length,),
                device=device,
                dtype=torch.int,
            ),
            persistent=False,
        )
        # Positions of first slot to overwrite with next :meth:`forward`
        # (per batch dimension)
        self.next_position = None
        self._replay_log = None
        self._pad_id = pad_id

    @staticmethod
    def from_config(
        config: Config,
        max_batch_size: int,
        cache_length: int,
        block_idx: int,
        tokenizer: Tokenizer,
        end_initial_regex: Union[str, re.Pattern],
        max_initial_fraction: float,
        include_end_string: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **base_kwargs,
    ) -> "SmartInitialLastRecentlyInsertedKVCache":
        """
        Creates KV cache with default buffers.

        Args:
            config: Model config
            max_batch_size: Maximum batch size supported
            cache_length: Number of slots (i.e., tokens) in cache
            block_idx: Block index
            tokenizer: Tokenizer
            end_initial_regex: Regular expression for end of initial part
                sequence
            max_initial_fraction: Fraction of cache length initial parts can
                occupy at most
            include_end_string: Include end of init sequence in initial part?
            device: Device for buffers. If not given, it is set with the
                first :meth:`forward` call, based on the input arguments.
            dtype: Data type for buffers. If not given, it is set with the
                first :meth:`forward` call, based on the input arguments.
            base_kwargs: Extra keyword arguments for cache and default buffer

        """
        buffers_kwargs = KVCacheWithBuffers.extract_default_buffers_kwargs(base_kwargs)
        buffers = KVCacheWithBuffers.create_default_buffers(
            config=config,
            max_batch_size=max_batch_size,
            cache_length=cache_length,
            device=device,
            dtype=dtype,
            **buffers_kwargs,
        )
        return SmartInitialLastRecentlyInsertedKVCache(
            config,
            buffers,
            block_idx,
            tokenizer,
            end_initial_regex,
            max_initial_fraction,
            include_end_string=include_end_string,
            **base_kwargs,
        )

    @classmethod
    def _parameter_names(cls) -> List[str]:
        return super()._parameter_names() + ["token_pos"]

    def max_forward_length(self) -> int:
        return self.cache_length

    def _forward_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> KeysAndValues:
        total_num = key.shape[2]
        max_one_step = self.cache_length - max(self.init_length)
        if total_num <= max_one_step:
            offs_and_nums = ((0, total_num),)
        else:
            # Only happens if `total_num` is very large
            offs_and_nums = ((0, max_one_step), (max_one_step, total_num - max_one_step),)
        tp_kwargs = dict(device=self.device, dtype=torch.int)
        k_and_v = None
        for off, num in offs_and_nums:
            # Determine `positions` for forward
            position_parts = [
                positions_wrap_around(
                    num=num,
                    current=np,
                    start=start,
                    end=self.cache_length,
                    batch_size=1,
                    n_query_groups=self.n_query_groups,
                    device=self.device,
                    return_tensor=True,
                )
                for start, np in zip(self.init_length, self.next_position)
            ]
            positions = torch.cat(position_parts, dim=0)
            k_and_v = self.kv_buffers.forward(
                positions=positions,
                key=key[:, :, off:(off + num), :],
                value=value[:, :, off:(off + num), :],
            )
            # Update `token_pos` and `next_positions`
            for bpos, (start, np) in enumerate(zip(self.init_length, self.next_position)):
                num1 = min(num, self.cache_length - np)
                diff = num - num1
                ntp = self.input_pos
                self.token_pos[bpos, np : (np + num1)] = torch.arange(
                    ntp, ntp + num1, **tp_kwargs,
                )
                np += num1
                if diff > 0:
                    self.token_pos[bpos, start : (start + diff)] = torch.arange(
                        ntp + num1, ntp + num, **tp_kwargs,
                    )
                    np = start + diff
                self.next_position[bpos] = np

        if self._replay_log is not None:
            if not isinstance(self._replay_log, DefaultKVCacheReplayLog):
                raise IndexError(
                    "Cannot switch on replay logging in the middle of inference run. Call 'prefill'."
                )
            self._replay_log.append_token_chunk(token_idx)
        return k_and_v

    def _set_init_length(self, token_idx: torch.Tensor):
        batch_size, max_init_length = token_idx.shape
        max_init_length = max(
            min(
                max_init_length,
                int(self.cache_length * self.max_initial_fraction),
            ),
            1,
        )
        self.init_length = []
        for tokens in token_idx:
            tokens = tokens.tolist()
            num_left_pad = next(i for i, t in enumerate(tokens) if t != self._pad_id)
            tokens = tokens[num_left_pad:]
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
            match = re.search(self.end_initial_regex, decoded)
            if match is not None:
                raw_length = match.end(0) if self.include_end_string else match.start(0)
                print(f"Found match:\n{decoded[:raw_length]}")  # DEBUG
                init_encoded = self.tokenizer.encode(decoded[:raw_length])
                new_length = len(init_encoded)
                try:
                    diff_pos = next(
                        i
                        for i, (t0, t1) in enumerate(zip(init_encoded, tokens[:new_length]))
                        if t0 != t1
                    )
                    # Can happen if `raw_length` hits in the middle of a token
                    if diff_pos < new_length - 2:
                        raise IndexError(
                            "Error mapping match raw position back to token position:"
                            "Too many tokens are different at the end:\n"
                            f"Raw initial part:\n{decoded[:raw_length]}\n"
                            f"Encoded initial part:\n{init_encoded}\n"
                            f"Corresponding initial token sequence:\n{tokens[:new_length]}"
                        )
                    else:
                        new_length = max(diff_pos, 1)
                except StopIteration:
                    pass
                print(f"new_length = {new_length}, before adding padding")  # DEBUG
                new_length = min(new_length + num_left_pad, max_init_length)
            else:
                new_length = max_init_length
            self.init_length.append(new_length)

    def _prefill_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        self._set_init_length(token_idx)
        init_length = key.shape[2]
        self.kv_buffers.prefill(key, value)
        if init_length < self.cache_length:
            self.next_position = [init_length] * self.batch_size
        else:
            assert init_length == self.cache_length, (init_length, self.cache_length)  # Sanity check
            self.next_position = self.init_length.copy()
        self.token_pos[:self.batch_size, :init_length] = torch.arange(
            init_length,
            dtype=self.token_pos.dtype,
            device=self.device,
        ).unsqueeze(0).expand(self.batch_size, -1)
        if self._replay_log is not None:
            self._replay_log = SmartInitialLastRecentlyInsertedKVCacheReplayLog(
                token_chunks=[token_idx],
                cache_length=self.cache_length,
                n_query_groups=self.n_query_groups,
                init_length=self.init_length,
            )

    def token_positions(self) -> torch.Tensor:
        return self.token_pos[
            : self.batch_size, :self.current_length
        ].unsqueeze(1).expand(-1, self.n_query_groups, -1)

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        tk_p = bitsize_of(self.token_pos)
        sz_total, dct_sz = self.kv_buffers.size_estimate()
        return sz_total + tk_p, {**dct_sz, "token_pos": tk_p}

    @classmethod
    def size_estimate_apriori(
        cls, params: KVCacheParams, **kwargs
    ) -> Tuple[int, Dict[str, int]]:
        """
        `cache_length` is required in `kwargs`. If `buffer_type` is given in
        `kwargs`, the size for this type is used, otherwise for the default
        type `DefaultKVCacheBuffers`.

        """
        buff_params = KVCacheBuffersParams.from_params(params)
        buffer_type = kwargs.get("buffer_type", DefaultKVCacheBuffers)
        sz_total, dct_sz = buffer_type.size_estimate_apriori(
            buff_params,
            cache_length=params.cache_length,
            **kwargs,
        )
        tk_p = params.max_batch_size * params.cache_length * bits_for_torch_dtype(torch.int)
        return sz_total + tk_p, {**dct_sz, "token_pos": tk_p}

    def switch_replay_logging(self, status: bool):
        if status:
            if self._replay_log is None:
                self._replay_log = []  # Dummy, created later
        else:
            self._replay_log = None

    @property
    def do_replay_logging(self) -> bool:
        return self._replay_log is not None

    def get_replay_log(self) -> Optional[KVCacheReplayLog]:
        return copy.copy(self._replay_log)

    def _base_kwargs_for_clone(self) -> Dict[str, Any]:
        return dict(
            super()._base_kwargs_for_clone(),
            tokenizer=self.tokenizer,
            end_initial_regex=self.end_initial_regex,
            max_initial_fraction=self.max_initial_fraction,
            include_end_string=self.include_end_string,
        )

    def clone(self) -> KVCache:
        return SmartInitialLastRecentlyInsertedKVCache(**self._base_kwargs_for_clone())
