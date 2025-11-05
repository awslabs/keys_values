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
from typing import List, Tuple, Optional

import torch

from litgpt.config import Config

from keys_values.kvcache.base import KVCacheReplayLog
from keys_values.kvcache.gradient.autograd_hooks import NodeAnnotation


class MonitorCellComputationAutogradHooks:
    """
    This is a variant of :class:`CellComputationAutogradHooks`, which is not
    used for packing and unpacking saved tensors, but for monitoring them in
    order to test hypotheses in which order tensors to be packed arise.

    This code is not used during normal operations, but should be kept around
    in order to modify :class:`CellComputationAutogradHooks` and
    :class:`TrainingAttnWeightsReplayCache` in case something changes in the
    internals of forward-backward computations on the model, and in particular
    in multi-head self attention.

    The difficulty with non-reentrant autograd saved tensors hooks is that we
    cannot identify the node in the graph for an argument to the pack hook.
    We need to identify the node by the shape and the order it arises. Also,
    not all possible internal tensors are stored in the graph. The purpose of
    this class is to test a hypothesis about which tensors of a certain size
    the pack hook is called for in which order. Once this is understood,
    :class:`CellComputationAutogradHooks` can be implemented to follow this
    order.

    Current hypothesis tested here:

    We have `shape1 = (batch_size * n_head, head_size, cache_length)` and
    `shape2 = (batch_size * n_head, cache_length, head_size)`. We only consider
    :meth:`pack_hook` inputs `x` which have one of these shapes. Their order is:

    * If `prefill_length == cache_length`: Chunk 0. These tensors do not have
      annotations.
      - Loop over layers:
        - `f(queries) [shape2]`
        - `f(keys) [shape1]`
        - `values [shape2]`
    * Chunk 1: These tensors have annotations.
      - Loop over layers:
        - `f(keys) [shape1]`. `kind` is "cat" or "scatter"
        - `values [shape2]`. `kind` is "cat" or "scatter"
    * Chunk 2: These tensors have annotations.
      [...]

    Here, `f(x) = alpha * x, alpha = head_dim ** *(-0.25)`.

    For annotations, `annotation.delta` contains the tensor value of `keys` or
    `values` with shape `(batch_size, n_query_groups, cache_length, head_size)`.
    Testing means bringing this into `shape1` or `shape2` and compare against
    the input argument `x`.

    """
    def __init__(
        self,
        config: Config,
        batch_size: int,
        cache_length: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Args:
            batch_size: Maximum batch size
            n_query_groups: Number of query groups
            n_head: Number of attention heads
            cache_length: Cache length
            head_size: Head size
            num_layers: Maximum number of layers per cell
            device: Device
            dtype: Data type for nodes which are annotated

        """
        self.batch_size = batch_size
        self.n_query_groups = config.n_query_groups
        self.n_head = config.n_head
        self.cache_length = cache_length
        self.head_size = config.head_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        norm_head_size = config.attention_scores_scalar or config.head_size
        self._alpha = norm_head_size ** (-0.25)
        # To be initialized in `initialize_cell`
        self.eff_num_layers = None
        self.num_chunks = None
        self.first_layer_idx = None
        self.first_chunk_idx = None
        self._node_annotations: List[NodeAnnotation] = []
        self._shape1 = None
        self._shape2 = None
        self._delta_shape = None
        self._skip_initial_num = None
        self._num_matched = None

    def initialize_cell(
        self,
        eff_num_layers: int,
        num_chunks: int,
        first_layer_idx: int,
        first_chunk_idx: int,
        replay_logs: List[KVCacheReplayLog],
    ):
        """
        Has to be called before a new cell is processed.

        Args:
            eff_num_layers: Number of layers in this cell
            num_chunks: Number of chunks in this cell
            first_layer_idx: Index of first layer in this cell
            first_chunk_idx: Index of first chunk in this cell
            replay_logs: Replay logs for these layers
        """
        if not (0 <= eff_num_layers <= self.num_layers):
            raise ValueError(f"eff_num_layers {eff_num_layers} must be in [0, {self.num_layers}]")
        assert num_chunks > 0
        assert first_chunk_idx >= 0
        assert first_chunk_idx >= 0
        assert len(replay_logs) == eff_num_layers
        self.eff_num_layers = eff_num_layers
        self.num_chunks = num_chunks
        self.first_layer_idx = first_layer_idx
        self.first_chunk_idx = first_chunk_idx
        # Derived
        # Note: Don't use `= []`. List object is referred to by
        # :class:`TrainingAttnWeightsReplayCache` objects
        self._node_annotations.clear()
        dim0 = self.batch_size * self.n_head
        self._shape1 = (dim0, self.head_size, self.cache_length)
        self._shape2 = (dim0, self.cache_length, self.head_size)
        self._delta_shape = (self.batch_size, self.n_query_groups, self.cache_length, self.head_size)
        self._skip_initial_num = 0
        if first_chunk_idx == 0:
            init_length = replay_logs[0].token_chunks[0].shape[-1]
            # If the initial (prefill) chunk has length `cache_length`, we need
            # to skip 3 tensors per layer initially.
            if init_length == self.cache_length:
                self._skip_initial_num = 3 * eff_num_layers
        self._num_matched = 0

    @property
    def node_annotations(self) -> List[NodeAnnotation]:
        return self._node_annotations

    def _pop_annotation(
        self, x: torch.Tensor,
    ) -> Tuple[Optional[NodeAnnotation], str]:
        result = None
        msg = "No annotations"
        if self.node_annotations:
            result = self.node_annotations[0]
            if result.delta is not None and result.delta.shape == self._delta_shape:
                msg = ""
                self.node_annotations.pop(0)
            else:
                if result.delta is None:
                    msg = "annotation.delta is None"
                else:
                    shape = tuple(int(d) for d in result.delta.shape)
                    msg = f"annotation.delta {shape}"
                result = None
        return result, msg

    def pack_hook(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = tuple(int(d) for d in x.shape)
        if x_shape == self._shape1 or x_shape == self._shape2:
            self._num_matched += 1
            if self._num_matched > self._skip_initial_num:
                annotation, error_msg = self._pop_annotation(x)
                if annotation is not None:
                    expected_x = self._transform_annotated_tensor(
                        annotation.delta, res_shape=x_shape
                    )
                    ex_shape = tuple(int(d) for d in expected_x.shape)
                    assert ex_shape == x_shape  # Sanity check
                    if torch.allclose(x, expected_x, atol=1e-6, rtol=1e-4):
                        print(f"YES:  pack_hook, x {x_shape}. Matches ({annotation.layer_idx}, {annotation.chunk_idx}, {annotation.node_idx}, {annotation.kind})")
                    else:
                        print(f"UUPS: pack_hook, x {x_shape}. Does not match ({annotation.layer_idx}, {annotation.chunk_idx}, {annotation.node_idx}, {annotation.kind})")
                else:
                    print(f"UUPS: pack_hook, x {x_shape}. {error_msg}")
        return x

    def _transform_annotated_tensor(
        self, delta: torch.Tensor, res_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        From our experiments, these arguments arise when computing SDPA with
        `torch.nn.functional.scaled_dot_product_attention`:

        * `shape1`: `alpha * repeat_kv_and_reshape(keys.mT)`
        * `shape2`: `repeat_kv_and_reshape(values)`

        Here, `repeat_kv_and_reshape` repeats the `n_query_groups` dimension if
        `n_query_groups < n_head`, then reshapes by combining all but the final
        two dimensions.

        """
        is_keys = res_shape == self._shape1
        if is_keys:
            delta = delta.mT
        if self.n_head > self.n_query_groups:
            q_per_kv = self.n_head // self.n_query_groups
            delta = delta.unsqueeze(2).expand(-1, -1, q_per_kv, -1, -1)
        if is_keys:
            return self._alpha * delta.reshape(-1, *delta.shape[-2:])
        else:
            return delta.reshape(-1, *delta.shape[-2:])

    def unpack_hook(self, x: torch.Tensor) -> torch.Tensor:
        return x
