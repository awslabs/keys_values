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
import torch


class NextPositionsMixin:
    """
    Mixin for KV cache implementations for which the next `num` overwrite
    positions in the KV cache buffers only depend on the size of the chunk
    with which :meth:`forward` is called.

    Many policies adhere to this property, but not all. Some policies make
    their decision dependent on `query`, `key`, or `value` passed to
    :meth:`forward`, not just their length.

    Note: Only KV caches inheriting this mixin can be used in
    :class:`SingleTokenFastInference`.
    """

    def next_positions(self, num: int) -> torch.Tensor:
        """
        Returns:
            Batched positions where key and value tensors in next :meth:`forward`
            call are written to, shape `(batch_size, n_query_groups, num)`,
            where `num <= max_forward_length()`.
        """
        raise NotImplementedError
