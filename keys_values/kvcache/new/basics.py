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
from dataclasses import dataclass
from typing import Union, List, Tuple, Optional, Dict, Literal

import torch

from keys_values.attention import KeysAndValues
from keys_values.config import Config
from keys_values.kvcache.base import KVCache, KVCacheParams


PositionsNewEntriesType = Union[torch.Tensor, List[Tuple[torch.Tensor, int, int]]]


@dataclass(frozen=True)
class KVCacheReplayLogEntry:
    token_idx: torch.Tensor
    positions: PositionsNewEntriesType


KVCacheReplayLog = List[KVCacheReplayLogEntry]


KVCacheReplayLogStatus = Literal["none", "record", "play"]


class KVCachePolicy(torch.nn.Module):
    """
    Base class for KV cache policies.

    A policy decides on where new `key`, `value` information is written to
    in a cache. These overwrite existing slots, i.e. the current information
    there is evicted.

    It also maintains `token_positions` for the cache. Some policies receive
    summed attention weights from scaled dot product attention (SDPA).
    """

    def __init__(self):
        super().__init__()
        self._device = None  # Unassigned so far

    @property
    def device(self) -> Optional[torch.device]:
        return self._device

    @property
    def cache_length(self) -> int:
        """
        Returns:
            Maximum cache length.

        """
        raise NotImplementedError

    @property
    def batch_size(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def n_query_groups(self) -> Optional[int]:
        raise NotImplementedError

    def token_positions(self) -> torch.Tensor:
        """
        See :meth:`keys_values.kvcache.base.KVCache.token_positions`.

        """
        raise NotImplementedError

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> PositionsNewEntriesType:
        """
        Given new information `key`, `value`, `query`, `token_idx`, returns
        slot positions where new `key`, `value` is to be stored. This can be
        an index with values in `range(cache_length)`, or a list with entries
        `(positions, off, num)`, where `positions` is an index, and the sum of
        `num` is equal to `key.shape[-2]`. See header comment of
        :class:`KVCacheDriver` for the semantics of slot positions, and how
        they are used to update the KV cache buffers.

        The arguments are the same as in
        :meth:`keys_values.kvcache.base.KVCache.forward`.

        """
        raise NotImplementedError

    def prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        """
        Called during prefill. Can be used to initiate replay log and token
        positions. Will also reset score buffers, if any.

        """
        raise NotImplementedError

    def max_forward_length(self) -> int:
        """
        Note: For some policies, this function returns different values
        depending on the state.

        Returns:
            Largest length of arguments for which next recent :meth:`forward`
            may be called. Does not restrict a prefill call.

        """
        raise NotImplementedError

    def requires_attn_weights(self) -> bool:
        """
        Note: Returned value may depend on inner state. For example,
        if `replay_log_status == "play"`, we usually return `False`.

        Returns:
            Do we require attention weights? If `True`, `pass_attn_weights`
            needs to be called passing the attention weights.

        """
        return False

    def pass_attn_weights(
        self,
        attn_weights: torch.Tensor,
        query_length: int,
    ):
        """
        Has to be called as part of each :meth:`forward` call if
        `requires_attn_weights` returns `True`.

        Args:
            attn_weights: Attention weights for the multi-head attention
                computation done just after the last recent :meth:`forward`
                call, summed over the query axis. Shape must be
                `(batch_size, n_query_groups, current_length)`, where
                `current_length <= cache_length` is the current cache length.
                Also, `dtype=float32`.
            query_length: Size of query axis

        """
        pass

    def set_seq_length(self, seq_length: int) -> None:
        pass

    def set_or_check_device(self, new_device: torch.device):
        """
        Called by :meth:`KVCache.forward` with `new_device=query.device`.
        Must be equal to `_device` unless latter is not yet set.

        """
        if self._device is None:
            self._convert_device(new_device)
            self._device = new_device
        elif new_device != self._device:
            raise ValueError(
                f"Arguments on device {new_device}, must be on {self._device}"
            )

    def _convert_device(self, new_device: torch.device):
        """
        There may be parameters which have been allocated on a default device
        before `_device` was set. If so, they need to be moved here.

        """
        for name in self._parameter_names():
            x = getattr(self, name, None)
            if x is not None:
                if not isinstance(x, torch.Tensor):
                    raise AssertionError(
                        f"Internal error in _convert_or_check_device: {name} must be of type torch.Tensor"
                    )
                if x.device != new_device:
                    setattr(self, name, x.to(new_device))

    @classmethod
    def _parameter_names(cls) -> List[str]:
        """
        Returns:
            Names of `torch.Tensor` parameters which need to be checked for
            device or converted.

        """
        return []

    def clone(self) -> "KVCachePolicy":
        raise NotImplementedError

    # TODO: Does this need args?
    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        raise NotImplementedError

    @classmethod
    def size_estimate_apriori(
        cls, params: KVCacheParams, **kwargs
    ) -> Tuple[int, Dict[str, int]]:
        raise NotImplementedError

    def replay_log_status(self) -> KVCacheReplayLogStatus:
        raise NotImplementedError

    def reset(self, replay_log_status: KVCacheReplayLogStatus):
        """
        Call this to switch replay log status between "record" and "play".
        This also resets everything, like :meth:`prefill`. This is done
        even if the replay log status stays the same. The replay log is
        reset to empty if `replay_log_status == "record"`, but is left
        unchanged if `replay_log_status == "play"`. This means we can use
        the same KV cache policy object to first record, then replay what
        was recorded.

        Note: In order to reuse a cache with the same replay log status,
        this method need not be called. :meth:`prefill` automatically resets.

        Args:
            replay_log_status: New replay log status. You can flip
                "record" to "play" and back, but "none" cannot be changed.
                Also, cannot change non-"none" to "none".

        """
        raise NotImplementedError

    def get_replay_log(self) -> Optional[KVCacheReplayLog]:
        """
        Returns:
            Cache replay log which is required for gradient computations.
            If `replay_log_status == "none"`, `None` is returned.

        """
        raise NotImplementedError


class KVCacheDriver(torch.nn.Module):
    """
    Base class for KV cache drivers.

    A driver maintains the KV buffers and performs the scaled dot product
    attention (SDPA) computation. K and V buffers have shape
    `(batch_size, n_query_groups, cache_length, head_size)`. For each
    `(b, h)`, up to `cache_length` slots can be stored. Once more than
    `cache_length` tokens are processed, slots are overwritten. Overwrite
    (or eviction) decisions are made by the KV cache logic.

    Note that KV buffers and SDPA are not separated here, but are dealt with
    together.

    Update of KV buffers based on `positions` in :meth:`forward`:

    We either have `positions == index` (single 3D index), or a list of
    entries `(index, off, num)`. In the latter case, we do the following
    for every entry in order, using `key[:, :, off:(off + num), :],
    `value[:, :, off:(off + num), :]` instead of `key, value` in each
    step.

    If `index.shape[-1] == num` (where `num = key.shape[2]` in the single
    entry case), then `key, value` are written to positions `index` (with
    `scatter_`) of the respective buffers. A special case arises if
    `index.shape[-1] > num`. Let `extra_pos = index[:, :, num:]`, and
    `key_cp, value_cp = kv_buffers.get_slots(extra_pos)` (extract by
    `gather`). Then, `torch.cat((key_cp, key), dim=-2)` and
    `torch.cat((value_cp, value), dim=-2)` are written to positions
    `index`. See :class:`AttnWeightsKVCachePolicy` for an example where
    this special case is needed.

    """
    def __init__(self, config: Config):
        super().__init__()
        self._config = config
        self._device = None  # Unassigned so far

    @property
    def config(self) -> Config:
        return self._config

    @property
    def max_batch_size(self) -> int:
        raise NotImplementedError

    @property
    def device(self) -> Optional[torch.device]:
        """
        A KV cache is created without association to a device. Its device
        is set based on the arguments of the first :meth:`forward` call. It
        cannot be changed afterwards (use :meth:`clone` to transport a
        cache to a different device).

        Returns:
            Device the KV cache buffers are kept on, and which arguments
            to :meth:`forward` need to be on. Returns `None` as long as the
            device is not fixed (upon first :meth:`forward` call).

        """
        return self._device

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """
        Returns:
            Data type of the KV cache buffers. This can be set at construction
            or is otherwise set with the first :meth:`forward` call. Cannot
            be changed afterwards.

        """
        raise NotImplementedError

    @property
    def cache_length(self) -> int:
        """
        Returns:
            Maximum cache length.

        """
        raise NotImplementedError

    @property
    def current_length(self) -> int:
        """
        Returns:
            Current cache length `<= cache_length`. This is equal to
            `cache_length` once `input_pos >= cache_length`.

        """
        raise NotImplementedError

    @property
    def batch_size(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def n_query_groups(self) -> Optional[int]:
        return self._config.n_query_groups

    @property
    def head_size(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def block_idx(self) -> int:
        raise NotImplementedError

    def prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Does the job of :meth:`forward` for `input_pos == 0` (i.e., the
        first call after :meth:`reset`). We must have `current_length == 0`.
        The length of inputs must be `<= cache_length`.

        Args:
            query: Initial queries
            key: Initial keys
            value: Initial values
            token_idx: Initial token indices

        Returns:
            Attention outputs

        """
        raise NotImplementedError

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
        positions: PositionsNewEntriesType,
        token_positions: torch.Tensor,
        input_pos: int,
        return_attn_weights: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward computation if `input_pos > 0`.
        First, the K and V buffers are updated for the new entries `key`,
        `value`, using overwrite positions `positions`, as determined by
        :meth:`KVCachePolicy.forward` called with the same arguments. See
        header comment on how this is to be done.

        Second, run SDPA with `query` and the extended K, V buffers, and
        `token_positions`. Here, `token_positions` has already been updated
        in the last recent :meth:`KVCachePolicy.forward` call. Returns the
        attention outputs.

        If `return_attn_weights == True`, the attention weights summed
        over the query axis are returned as well.

        Args:
            query: Queries for the new entries.
            key: Keys for the new entries.
            value: Values for the new entries.
            token_idx: Token indices for the new entries.
            positions: Positions for the new entries, coming from a call
                of :meth:`KVCachePolicy.forward`.
            token_positions: Token positions for KV buffer after update,
                obtained from :meth:`KVCachePolicy.token_positions`.
            input_pos: Input positions for the new entries. This is the number
                of tokens processed before the current update
            return_attn_weights: If `True`, the attention weights summed over
                the query axis are returned as well.

        Returns:
            `attn_output, attn_weights`, where `attn_weights` is `None` if
            attention weights are not returned. If given, we have
            `attn_weights.shape == (batch_size, n_query_groups, kv_len)`
            and `attn_weights.dtype == float32`, independent of `query.dtype`.
            Here, `kv_len == current_length` is the current cache length.

        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Resets the cache buffers so that `input_pos == 0` and
        `current_length == 0` afterwards. Needs to be called before the
        cache can be used for a new sequence. Has to be followed by a
        :meth:`prefill` call, before any :meth:`forward` call.

        """
        raise NotImplementedError

    def set_seq_length(self, seq_length: int) -> None:
        raise NotImplementedError

    def get_keys_values(self) -> Optional[KeysAndValues]:
        """
        Note: This method is not implemented in all subclasses (will raise
        `NotImplementedError` in such cases). Use for testing only.

        Returns:
            :class:`KeysAndValues` object, providing access to currently stored
            keys and values tensors. If the cache is empty or has not been
            initialized, `None` is returned.

        """
        raise NotImplementedError

    def set_or_check_device(self, new_device: torch.device):
        """
        Called by :meth:`KVCache.forward` with `new_device=query.device`.
        Must be equal to `_device` unless latter is not yet set.

        """
        if self._device is None:
            self._convert_device(new_device)
            self._device = new_device
        elif new_device != self._device:
            raise ValueError(
                f"Arguments on device {new_device}, must be on {self._device}"
            )

    def _convert_device(self, new_device: torch.device):
        """
        There may be parameters which have been allocated on a default device
        before `_device` was set. If so, they need to be moved here.

        """
        pass

    def clone(self) -> "KVCacheDriver":
        raise NotImplementedError

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        raise NotImplementedError

    @classmethod
    def size_estimate_apriori(
        cls, params: KVCacheParams, **kwargs
    ) -> Tuple[int, Dict[str, int]]:
        raise NotImplementedError


class GenericKVCache(KVCache):
    """
    Generic implementation for most of the :class:`KVCache` variants supported.
    The implementation is configured by a :class:`KVCachePolicy` object and a
    :class:`KVCacheDriver` object.

    """
    def __init__(
        self,
        policy: KVCachePolicy,
        driver: KVCacheDriver,
    ):
        self._check_compatible(policy, driver)
        super().__init__(
            driver.config,
            driver.max_batch_size,
            driver.cache_length,
            driver.block_idx,
            driver.dtype,
            driver.head_size,
        )
        self.policy = policy
        self.driver = driver
        self._input_pos = 0

    @staticmethod
    def _check_compatible(policy: KVCachePolicy, driver: KVCacheDriver):
        for name in ("cache_length", "n_query_groups"):
            val_p = getattr(policy, name)
            val_d = getattr(driver, name)
            if val_p != val_d:
                raise ValueError(
                    f"policy.{name} = {val_p} != {val_d} = driver.{name}"
                )

    def _check_batch_size(self):
        val_p = self.policy.cache_length
        val_d = self.driver.cache_length
        if val_p != val_d:
            raise ValueError(
                f"policy.batch_size = {val_p} != {val_d} = driver.batch_size"
            )

    @property
    def device(self) -> Optional[torch.device]:
        return self.driver.device

    @property
    def input_pos(self) -> int:
        return self._input_pos

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ) -> torch.Tensor:
        self._forward_check_args(query, key, value, token_idx)
        for_prefill = self.input_pos == 0
        num = query.shape[2]

        if for_prefill:
            # Prefill: `input_pos == 0`
            self.policy.prefill(query, key, value, token_idx)
            attn_outputs = self.driver.prefill(query, key, value, token_idx)
            attn_weights = None
            requires_attn_weights = False
        else:
            # Forward: `input_pos > 0`
            positions = self.policy.forward(query, key, value, token_idx)
            requires_attn_weights = self.policy.requires_attn_weights()
            attn_outputs, attn_weights = self.driver.forward(
                query,
                key,
                value,
                token_idx,
                positions,
                token_positions=self.policy.token_positions(),
                input_pos=self.input_pos,
                return_attn_weights=requires_attn_weights,
            )
        # Cache has been extended:
        self._input_pos += num
        if requires_attn_weights:
            # Pass summed attention weights to policy
            self.policy.pass_attn_weights(
                attn_weights=attn_weights,
                query_length=query.shape[2],
            )

        return attn_outputs

    def get_keys_values(self) -> Optional[KeysAndValues]:
        return self.driver.get_keys_values()

    def max_forward_length(self) -> int:
        return self.policy.max_forward_length()

    # TODO: Remove: Always equal to cache_length!
    @property
    def max_prefill_length(self) -> int:
        return self.cache_length

    def token_positions(self) -> torch.Tensor:
        return self.policy.token_positions()

    def size_estimate(self) -> Tuple[int, Dict[str, int]]:
        estim_p_total, estim_p_dct = self.policy.size_estimate()
        estim_d_total, estim_d_dct = self.driver.size_estimate()
        return (
            estim_p_total + estim_d_total,
            {**estim_p_dct, **estim_d_dct},
        )

    def reset(self) -> None:
        # Note: `self.policy` does not need to be reset. This is done when
        # `self.policy.prefill` is called
        self.driver.reset()
        self._input_pos = 0

    def set_seq_length(self, seq_length: int) -> None:
        self.policy.set_seq_length(seq_length)
        self.driver.set_seq_length(seq_length)

    def clone(self) -> "KVCache":
        return GenericKVCache(
            self.policy.clone(),
            self.driver.clone(),
        )

    def _forward_check_args(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_idx: torch.Tensor,
    ):
        """
        Apart from checking shapes of arguments, we also deal with `device`,
        calling `set_or_check_device(query.device)` for `driver` and `policy`.
        As long as `policy.device`, `driver.device` were not set, they become
        `query.device` (and respective parameters moved). Otherwise,
        `query.device` must be equal to their device. Arguments are not moved
        if on the wrong device.

        """
        for_prefill = self.input_pos == 0
        if query.ndim != 4:
            raise ValueError("query, key, value must be 4D tensors")
        batch_size, _, num, _ = query.shape
        if for_prefill:
            if not (1 <= batch_size <= self.max_batch_size):
                raise ValueError(
                    f"query.shape[0] = {batch_size}, must be in [1, {self.max_batch_size}]"
                )
            if not (1 <= num <= self.max_prefill_length):
                raise ValueError(
                    f"query.shape[2] = {num}, must be in [1, {self.max_prefill_length}]"
                )
        else:
            if batch_size != self.batch_size:
                raise ValueError(
                    f"query.shape[0] = {batch_size} != batch_size = {self.batch_size}"
                )
            if not (1 <= num <= self.max_forward_length()):
                raise ValueError(
                    f"query.shape[2] = {num}, must be in [1, {self.max_forward_length()}]"
                )
        q_shape = (batch_size, self.n_head, num, self.head_size)
        if query.shape != q_shape:
            raise ValueError(f"query.shape = {query.shape}, must be {q_shape}")
        k_shape = (batch_size, self.n_query_groups, num, self.head_size)
        if key.shape != k_shape:
            raise ValueError(f"key.shape = {key.shape}, must be {k_shape}")
        if value.shape != k_shape:
            raise ValueError(f"value.shape = {value.shape}, must be {k_shape}")
        t_shape = (batch_size, num)
        if token_idx.shape != t_shape:
            raise ValueError(f"token_idx.shape = {token_idx.shape}, must be {t_shape}")
        self.policy.validate_token_idx(token_idx)
        # Deal with dtype
        if self._dtype is None:
            self._dtype = query.dtype
        elif self._dtype != query.dtype:
            raise ValueError(
                f"query.dtype = {query.dtype}, must be {self._dtype}"
            )
        for x in (key, value):
            if x.dtype != query.dtype:
                raise ValueError("query, key, value must have same dtype")
        # Deal with device
        new_device = query.device
        for x in (key, value, token_idx):
            if x.device != new_device:
                raise ValueError("query, key, value, token_idx must be on same device")
        self.driver.set_or_check_device(new_device)
        self.policy.set_or_check_device(new_device)
