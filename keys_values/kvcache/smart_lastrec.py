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

from tokenizers import Tokenizer as HFTokenizer
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
from keys_values.utils import bits_for_torch_dtype, bitsize_of, encode, index_to_3d


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
            raise ValueError(
                "Use LastRecentlyInsertedKVCache for cache without initial parts"
            )
        if not (0 < self.max_initial_fraction < 1):
            raise ValueError(
                f"max_initial_fraction = {self.max_initial_fraction}, must be in (0, 1)"
            )


def end_initial_regex_from_string(
    s: str,
    tokenizer: HFTokenizer,
    do_escape: bool = True,
) -> str:
    result = tokenizer.decode(
        encode(tokenizer, s),
        skip_special_tokens=True,
    )
    if do_escape:
        result = re.escape(result)
    return result


def update_next_position(
    cache_length: int,
    positions: torch.Tensor,
    protected_start: List[Optional[int]],
    protected_end: List[Optional[int]],
    next_position: List[int],
):
    assert positions.ndim == 2  # Sanity check
    for bpos, (pos_for_b, pstart, pend) in enumerate(
        zip(positions, protected_start, protected_end)
    ):
        np = pos_for_b[-1] + 1
        if pstart is None:
            np = np % cache_length
        elif np == cache_length:
            # If `pstart == 0`, must have `pend < cache_length`
            np = 0 if pstart > 0 else pend
        elif np == pstart:
            # If `pend == cache_length`, must have `pstart > 0`
            np = pend % cache_length
        next_position[bpos] = np


def positions_outside_protected_range(
    num: int,
    current: int,
    end: int,
    protected_start: int,
    protected_end: int,
    **index_kwargs,
) -> torch.Tensor:
    assert 0 <= protected_start < protected_end <= end
    assert 0 <= current < protected_start or protected_end <= current < end
    assert protected_end - protected_start < end
    parts = []
    remainder = num
    while remainder > 0:
        diff = protected_start - current
        if diff > 0:
            # Left of `protected_start`
            diff = min(diff, remainder)
            part = torch.arange(current, current + diff, **index_kwargs)
            remainder -= diff
            # Only used again if `remainder > 0`
            current = protected_end % end
        else:
            # Right of `protected_end`
            diff = min(end - current, remainder)
            part = torch.arange(current, current + diff, **index_kwargs)
            remainder -= diff
            # Only used again if `remainder > 0`
            current = 0 if protected_start > 0 else protected_end
        parts.append(part)
    return torch.cat(parts, dim=0)


def extract_index(
    num: int,
    cache_length: int,
    protected_start: List[Optional[int]],
    protected_end: List[Optional[int]],
    next_position: List[int],
    n_query_groups: int,
    **index_kwargs,
) -> List[Tuple[torch.Tensor, int, int]]:
    # Maximum number of slots which can be written in a single
    # step, without writing to some slot twice
    max_one_step = cache_length - max(
        b - a if a is not None else 0
        for a, b in zip(protected_start, protected_end)
    )
    if num <= max_one_step:
        offs_and_nums = ((0, num),)
    else:
        # Only happens if `num` is very large
        assert num <= 2 * max_one_step, f"Internal error: num = {num}, max_one_step = {max_one_step}"
        offs_and_nums = (
            (0, max_one_step),
            (max_one_step, num - max_one_step),
        )
        next_position = copy.copy(next_position)
    result = []
    for off, _num in offs_and_nums:
        # Determine `positions` for forward
        # If there is no range yet for a batch position, cache positions
        # are token positions module `cache_length`
        position_parts = [
            index_to_3d(
                positions_outside_protected_range(
                    num=_num,
                    current=np,
                    end=cache_length,
                    protected_start=pstart,
                    protected_end=pend,
                    **index_kwargs,
                ),
                1,
                n_query_groups,
            ) if pstart is not None else positions_wrap_around(
                num=_num,
                current=np,
                start=0,
                end=cache_length,
                batch_size=1,
                n_query_groups=n_query_groups,
                **index_kwargs,
            )
            for np, pstart, pend in zip(
                next_position,
                protected_start,
                protected_end,
            )
        ]
        positions = torch.cat(position_parts, dim=0)
        result.append((positions, off, _num))
        if len(offs_and_nums) > 1:
            update_next_position(
                cache_length=cache_length,
                positions=positions[:, 0, :],
                protected_start=protected_start,
                protected_end=protected_end,
                next_position=next_position,
            )
    return result


class SmartInitialLastRecentlyInsertedKVCacheReplayLog(DefaultKVCacheReplayLog):
    """
    Replay log for :class:`SmartInitialLastRecentlyInsertedKVCache`.

    `next_position_for_input_pos` maps `input_pos` values to `next_position`
    lists. Entries are passed with :meth:`update`. We need to store these,
    because we cannot easily compute `next_position` from `input_pos`: this
    depends on the chunk for which the protected range was set.
    """

    def __init__(
        self,
        token_chunks: List[torch.Tensor],
        cache_length: int,
        n_query_groups: int,
        tokenizer: HFTokenizer,
        end_initial_regex: Union[str, re.Pattern],
        max_initial_fraction: float,
        include_end_string: bool,
        pad_id: int,
        protected_start: List[Optional[int]],
        protected_end: List[Optional[int]],
    ):
        super().__init__(
            token_chunks,
            cache_length,
            max_prefill_length=cache_length,
            grace_period=0,
        )
        self.n_query_groups = n_query_groups
        self.tokenizer = tokenizer
        self.end_initial_regex = end_initial_regex
        self.max_initial_fraction = max_initial_fraction
        self.include_end_string = include_end_string
        self.pad_id = pad_id
        self.protected_start = None
        self.protected_end = None
        # Map of `input_pos` to `next_position`. This is needed because
        # we cannot easily compute this map. It depends on when each
        # protected range was fixed (this may happen during a chunk later
        # than the prefill one).
        self.next_position_for_input_pos = dict()
        self.update(
            protected_start=protected_start,
            protected_end=protected_end,
            input_pos=0,
            next_position=[0] * len(protected_start),
        )

    def update(
        self,
        protected_start: List[Optional[int]],
        protected_end: List[Optional[int]],
        input_pos: int,
        next_position: List[int],
    ):
        self.protected_start = protected_start.copy()
        self.protected_end = protected_end.copy()
        self.next_position_for_input_pos[input_pos] = next_position.copy()

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
        next_position = self.next_position_for_input_pos.get(input_pos)
        if next_position is None:
            raise IndexError(f"input_pos = {input_pos} not represented (keys are {list(self.next_position_for_input_pos.keys())})")
        indexes = extract_index(
            num=num,
            cache_length=self.cache_length,
            protected_start=self.protected_start,
            protected_end=self.protected_end,
            next_position=next_position,
            n_query_groups=self.n_query_groups,
            dtype=dtype,
            device=device,
        )
        if len(indexes) == 1:
            return indexes[0][0]
        else:
            return torch.cat([x[0] for x in indexes], dim=-1)


class SmartInitialLastRecentlyInsertedKVCache(KVCacheWithBuffers):
    """
    Advanced variant of :class:`LastRecentlyInsertedKVCache`. For each batch
    dimension, there is a protected range of cache positions, KV information
    for which remains in the cache, namely
    `range(protected_start[b], protected_end[b])` for each batch position `b`.

    A regular expression `end_initial_regex` is passed at construction.
    Protected ranges are determined in :meth:`prefill`, and possibly in
    subsequent :meth:`forward` calls. Namely, `protected_start[b]` is the cache
    position of the first non-padding token (cache position before protected
    range is token position modulo cache length). `protected_end[b]` is the
    smallest of:

    * `end` (if `include_end_string == True`) or `start` (if
      `include_end_string == False`) if the first match of `end_initial_regex`
      in the string decoded from the token sequence `token_idx[b, :]` has span
      `(start, end)`.
    * `protected_start[b] + int(max_initial_fraction * cache_length)`
    * Chunk length `key.shape[2]`

    In other words, protected ranges are terminated by a certain string pattern
    described by `end_initial_regex`, and the pattern is included in the
    initial parts iff `include_end_string == True`. However, both
    `int(max_initial_fraction * cache_length)` and the prefill length limit the
    length of the ranges. Also, the protected range must lie within a chunk
    and also not wrap around (for simplicity).

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
        tokenizer: HFTokenizer,
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
            raise ValueError(
                "Use LastRecentlyInsertedKVCache for cache without initial parts"
            )
        if not (0 < max_initial_fraction < 1):
            raise ValueError(
                f"max_initial_fraction = {max_initial_fraction}, must be in (0, 1)"
            )
        self.max_initial_fraction = max_initial_fraction
        self.include_end_string = include_end_string
        # Protected ranges
        self.protected_start = None
        self.protected_end = None
        # Note: We could generate `token_pos` on the fly from `input_pos`,
        # `next_position` and `protected_*`, but it is simpler just to maintain
        # it.
        device = self._default_device_for_new_params()
        self.register_buffer(
            "token_pos",
            torch.zeros(
                (
                    buffers.max_batch_size,
                    buffers.cache_length,
                ),
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
        tokenizer: HFTokenizer,
        end_initial_regex: Union[str, re.Pattern],
        max_initial_fraction: float,
        include_end_string: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **base_kwargs,
    ) -> "SmartInitialLastRecentlyInsertedKVCache":
        """
        Creates KV cache with default buffers.

        `tokenizer` is the Hugging Face object, not the LitGPT wrapper, pass
        `tokenizer.processor` if you have the latter.

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
        self._update_protected_ranges(token_idx)
        if self._replay_log is not None:
            if not isinstance(self._replay_log, SmartInitialLastRecentlyInsertedKVCacheReplayLog):
                raise IndexError(
                    "Cannot switch on replay logging in the middle of inference run. Call 'prefill'."
                )
            self._replay_log.update(
                protected_start=self.protected_start,
                protected_end=self.protected_end,
                input_pos=self.input_pos,
                next_position=self.next_position,
            )
            self._replay_log.append_token_chunk(token_idx)

        total_num = key.shape[2]
        index_kwargs = dict(device=self.device, dtype=torch.int)
        indexes = extract_index(
            num=total_num,
            cache_length=self.cache_length,
            protected_start=self.protected_start,
            protected_end=self.protected_end,
            next_position=self.next_position,
            n_query_groups=self.n_query_groups,
            **index_kwargs,
        )
        k_and_v = None
        ntp = self.input_pos
        for positions, off, num in indexes:
            k_and_v = self.kv_buffers.forward(
                positions=positions,
                key=key[:, :, off : (off + num), :],
                value=value[:, :, off : (off + num), :],
            )
            positions_2d = positions[:, 0, :]
            # Update `token_pos` and `next_positions`
            for bpos, (pos_for_b, pstart, pend) in enumerate(
                zip(positions_2d, self.protected_start, self.protected_end)
            ):
                self.token_pos[bpos, pos_for_b] = torch.arange(
                    ntp, ntp + num, **index_kwargs
                )
            update_next_position(
                cache_length=self.cache_length,
                positions=positions_2d,
                protected_start=self.protected_start,
                protected_end=self.protected_end,
                next_position=self.next_position,
            )
            ntp += num
        return k_and_v

    def _update_protected_ranges(
        self, token_idx: torch.Tensor,
    ):
        batch_size, max_init_length = token_idx.shape
        # New protected range cannot wrap around. Note that for batch
        # positions where the range is not yet determined, the current cache
        # position is `self.input_pos % self.cache_length`.
        max_init_length = min(
            max_init_length,
            self.cache_length - self.input_pos % self.cache_length,
        )
        max_init_length = max(
            min(
                max_init_length,
                int(self.cache_length * self.max_initial_fraction),
            ),
            1,
        )
        if self.protected_start is None:
            self.protected_start = [None] * batch_size
            self.protected_end = [None] * batch_size
        for bpos, (tokens, pstart, pend) in enumerate(
            zip(token_idx, self.protected_start, self.protected_end)
        ):
            if pstart is None:
                # Protected range not yet determined
                tokens = tokens.tolist()
                try:
                    num_left_pad = next(
                        i for i, t in enumerate(tokens) if t != self._pad_id
                    )
                except StopIteration:
                    # All tokens are padding
                    continue
                tokens = tokens[num_left_pad:]
                decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
                match = re.search(self.end_initial_regex, decoded)
                if match is not None:
                    raw_length = match.end(0) if self.include_end_string else match.start(0)
                    init_encoded = encode(self.tokenizer, decoded[:raw_length])
                    new_length = len(init_encoded)
                    try:
                        diff_pos = next(
                            i
                            for i, (t0, t1) in enumerate(
                                zip(init_encoded, tokens[:new_length])
                            )
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
                    new_length = min(new_length, max_init_length)
                else:
                    new_length = min(max_init_length, len(tokens))
                pstart = (self.input_pos + num_left_pad) % self.cache_length
                pend = pstart + new_length
                assert pend <= self.cache_length, (pstart, pend, self.cache_length)
                self.protected_start[bpos] = pstart
                self.protected_end[bpos] = pend

    def _prefill_internal(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        self._update_protected_ranges(token_idx)
        init_length = key.shape[2]
        self.kv_buffers.prefill(key, value)
        if init_length < self.cache_length:
            self.next_position = [init_length] * self.batch_size
        else:
            assert init_length == self.cache_length, (
                init_length,
                self.cache_length,
            )  # Sanity check
            self.next_position = [
                0 if pstart is None or pstart > 0 else pend
                for pstart, pend in zip(self.protected_start, self.protected_end)
            ]
        self.token_pos[: self.batch_size, :init_length] = (
            torch.arange(
                init_length,
                dtype=self.token_pos.dtype,
                device=self.device,
            )
            .unsqueeze(0)
            .expand(self.batch_size, -1)
        )
        if self._replay_log is not None:
            self._replay_log = SmartInitialLastRecentlyInsertedKVCacheReplayLog(
                token_chunks=[token_idx],
                cache_length=self.cache_length,
                n_query_groups=self.n_query_groups,
                tokenizer=self.tokenizer,
                end_initial_regex=self.end_initial_regex,
                max_initial_fraction=self.max_initial_fraction,
                include_end_string=self.include_end_string,
                pad_id=self._pad_id,
                protected_start=self.protected_start,
                protected_end=self.protected_end,
            )

    def token_positions(self) -> torch.Tensor:
        return (
            self.token_pos[: self.batch_size, : self.current_length]
            .unsqueeze(1)
            .expand(-1, self.n_query_groups, -1)
        )

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
        tk_p = (
            params.max_batch_size
            * params.cache_length
            * bits_for_torch_dtype(torch.int)
        )
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
