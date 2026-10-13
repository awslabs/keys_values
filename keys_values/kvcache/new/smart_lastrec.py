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
import re
from typing import Tuple, Dict, Any, List, Union, Optional

from tokenizers import Tokenizer as HFTokenizer
import torch

from keys_values.config import Config
from keys_values.constants import DEFAULT_PAD_ID
from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.new.basics import (
    PositionsNewEntriesType,
    KVCachePolicy,
    KVCacheReplayLog,
    KVCacheReplayLogStatus,
)
from keys_values.kvcache.new.policy import DefaultKVCachePolicy
from keys_values.kvcache.smart_lastrec import (
    update_next_position,
    extract_index,
)
from keys_values.utils import (
    bits_for_torch_dtype,
    bitsize_of,
    encode,
)


class SmartInitialLastRecentlyInsertedKVCachePolicy(DefaultKVCachePolicy):
    """
    Advanced variant of :class:`LastRecentlyInsertedKVCachePolicy`. For each
    batch dimension, there is a protected range of cache positions, KV
    information for which remains in the cache, namely
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

    The projected range may wrap around. If `protected_start[b] > protected_end[b]`,
    it is in fact `range(protected_start[b], cache_length)` plus
    `range(0, protected_end[b])`.

    If `range_is_prefix == True`, protected ranges are prefixes, so
    `protected_start[b] = 0` and they are always determined from the prefill
    chunk. In this case, protected ranges may contain left-padding tokens.

    TODO: This option is switched on by default, since we ran experiments with
    `range_is_prefix == False` and did not see improvements. To be revisited!

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
        cache_length: int,
        tokenizer: HFTokenizer,
        end_initial_regex: Union[str, re.Pattern],
        max_initial_fraction: float,
        max_batch_size: int,
        use_replay_logging: bool = True,
        replay_log: Optional[KVCacheReplayLog] = None,
        include_end_string: bool = True,
        pad_id: int = DEFAULT_PAD_ID,
        range_is_prefix: bool = True,
    ):
        super().__init__(config, cache_length, use_replay_logging, replay_log)
        self.tokenizer = tokenizer
        if not isinstance(end_initial_regex, re.Pattern):
            end_initial_regex = re.compile(end_initial_regex)
        self.end_initial_regex = end_initial_regex
        if max_initial_fraction == 0:
            raise ValueError(
                "Use LastRecentlyInsertedKVCachePolicy for cache without initial parts"
            )
        if not (0 < max_initial_fraction < 1):
            raise ValueError(
                f"max_initial_fraction = {max_initial_fraction}, must be in (0, 1)"
            )
        self.max_initial_fraction = max_initial_fraction
        self.include_end_string = include_end_string
        self.range_is_prefix = range_is_prefix
        # Protected ranges
        self.protected_start = None
        self.protected_end = None
        # Note: We could generate `token_pos` on the fly from `input_pos`,
        # `next_position` and `protected_*`, but it is simpler just to maintain
        # it.
        device = self._device_internal()
        self.register_buffer(
            "token_pos",
            torch.zeros(
                (max_batch_size, cache_length),
                device=device,
                dtype=torch.int,
            ),
            persistent=False,
        )
        # Positions of first slot to overwrite with next :meth:`forward`
        # (per batch dimension)
        self.next_position = None
        self._pad_id = pad_id
        # Tracked here as well:
        self._current_length = 0

    @classmethod
    def _parameter_names(cls) -> List[str]:
        return super()._parameter_names() + ["token_pos"]

    def token_positions(self) -> torch.Tensor:
        if self.batch_size is None:
            raise RuntimeError("batch_size is not set. Call `prefill` first.")
        return (
            self.token_pos[: self.batch_size, : self._current_length]
            .unsqueeze(1)
            .expand(-1, self.n_query_groups, -1)
        )

    def _forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        positions: Optional[PositionsNewEntriesType] = None,
    ) -> PositionsNewEntriesType:
        total_num = key.shape[2]
        index_kwargs = dict(device=self.device, dtype=torch.int)
        use_range = self._update_protected_ranges(token_idx)
        if positions is None:
            # If a protected range has just been set in `_update_protected_ranges`,
            # it must not be used in `extract_index`
            pstart_copy = [
                ps if ur else None for ps, ur in zip(self.protected_start, use_range)
            ]
            pend_copy = [
                pe if ur else None for pe, ur in zip(self.protected_end, use_range)
            ]
            positions = extract_index(
                num=total_num,
                cache_length=self.cache_length,
                protected_start=pstart_copy,
                protected_end=pend_copy,
                next_position=self.next_position,
                n_query_groups=self.n_query_groups,
                **index_kwargs,
            )
        ntp = self._input_pos
        for _positions, off, num in positions:
            positions_2d = _positions[:, 0, :]
            # Update `token_pos` and `next_positions`
            for bpos, pos_for_b in enumerate(positions_2d):
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
        self._current_length = min(self._input_pos + total_num, self.cache_length)
        return positions

    def _update_protected_ranges(
        self,
        token_idx: torch.Tensor,
    ) -> List[bool]:
        batch_size, max_range_length = token_idx.shape
        # Note that for batch positions where the range is not yet determined,
        # the current cache position is `self.input_pos % self.cache_length`.
        max_range_length = max(
            min(
                max_range_length,
                int(self.cache_length * self.max_initial_fraction),
            ),
            1,
        )
        if self.protected_start is None:
            self.protected_start = [None] * batch_size
            self.protected_end = [None] * batch_size
        use_range = [pstart is not None for pstart in self.protected_start]
        for bpos, (tokens, pstart, pend) in enumerate(
            zip(token_idx, self.protected_start, self.protected_end)
        ):
            if pstart is None:
                # Protected range not yet determined
                tokens = tokens.tolist()
                # Strip off left padding: Has to be done even if
                # `range_is_prefix == True`, since the pad tokens derail the
                # match for `end_initial_regex`. Have to add back `num_left_pad`
                # below in this case.
                try:
                    num_left_pad = next(
                        i for i, t in enumerate(tokens) if t != self._pad_id
                    )
                except StopIteration:
                    # All tokens are padding
                    if not self.range_is_prefix:
                        # Skip this chunk, search in the next one
                        continue
                    else:
                        # Protected range must be prefix, so must lie in first
                        # chunk
                        num_left_pad = -1
                if num_left_pad != -1:
                    tokens = tokens[num_left_pad:]
                    decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    match = re.search(self.end_initial_regex, decoded)
                else:
                    match = None
                if match is not None:
                    raw_length = (
                        match.end(0) if self.include_end_string else match.start(0)
                    )
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
                    if self.range_is_prefix:
                        new_length += num_left_pad
                else:
                    new_length = len(tokens)
                    if self.range_is_prefix:
                        new_length += num_left_pad
                new_length = min(new_length, max_range_length)
                if self.range_is_prefix:
                    pstart = 0
                else:
                    pstart = (self._input_pos + num_left_pad) % self.cache_length
                pend = pstart + new_length
                if pend > self.cache_length:
                    pend -= self.cache_length
                assert pstart != pend, (pstart, pend, self.cache_length)
                self.protected_start[bpos] = pstart
                self.protected_end[bpos] = pend
        return use_range

    def _prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        self.protected_start = None
        self.protected_end = None
        self._update_protected_ranges(token_idx)
        init_length = token_idx.shape[-1]
        _batch_size = token_idx.shape[0]
        if _batch_size > self.token_pos.shape[0]:
            raise ValueError(f"token_idx.shape = {token_idx.shape}: Batch size {_batch_size} > {self.token_pos.shape[0]} = max_batch_size")
        self._batch_size = _batch_size
        if init_length < self.cache_length:
            self.next_position = [init_length] * self.batch_size
        else:
            assert init_length == self.cache_length, (
                init_length,
                self.cache_length,
            )  # Sanity check
            self.next_position = [
                0 if pstart is None or 0 < pstart < pend else pend
                for pstart, pend in zip(self.protected_start, self.protected_end)
            ]
        self.token_pos[: self.batch_size, :init_length] = (
            torch.arange(
                init_length,
                dtype=self.token_pos.dtype,
                device=self._device_internal(),
            )
            .unsqueeze(0)
            .expand(self.batch_size, -1)
        )
        self._current_length = init_length

    def max_forward_length(self) -> int:
        return self.cache_length

    def _base_kwargs_for_clone(self) -> Dict[str, Any]:
        return dict(
            super()._base_kwargs_for_clone(),
            tokenizer=self.tokenizer,
            end_initial_regex=self.end_initial_regex,
            max_initial_fraction=self.max_initial_fraction,
            max_batch_size=self.token_pos.shape[0],
            include_end_string=self.include_end_string,
            pad_id=self._pad_id,
            range_is_prefix=self.range_is_prefix,
        )

    def clone(self) -> KVCachePolicy:
        return SmartInitialLastRecentlyInsertedKVCachePolicy(**self._base_kwargs_for_clone())

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        tk_p = bitsize_of(self.token_pos)
        return tk_p, {"token_pos": tk_p}

    @classmethod
    def size_estimate_apriori(
        cls, params: KVCacheParams, **kwargs
    ) -> Tuple[int, Dict[str, int]]:
        tk_p = (
            params.max_batch_size
            * params.cache_length
            * bits_for_torch_dtype(torch.int)
        )
        return tk_p, {"token_pos": tk_p}

    # Note: We refrain from doing more than setting `current_length = 0`.
    # The next valid call must be :meth:`prefill`, which does a proper
    # reset (this may need its args).
    def reset(self, replay_log_status: KVCacheReplayLogStatus):
        super().reset(replay_log_status)
        self._current_length = 0
