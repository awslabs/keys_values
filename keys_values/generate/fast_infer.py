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

from keys_values.model import GPT


class SingleTokenFastInference:
    """
    Driver for fast inference in a single new token. This is needed during
    token generation.

    Concepts:
    - Group single token calls into chunks of size `chunk_size`. All expensive
      computations are done at chunk boundaries only.
    - Use compiled graph (`torch.compile`) for the complete single token forward
      expression
    - Within a chunk, KV buffers are maintained as tensors on GPU and are not
      written back. If KV caches use quantization, this is not used.
    - Both writing back cache buffers and policy decisions (i.e., which slots to
      overwrite next) are done at chunk boundaries only.

    Choices:
    - The inference graph returns the updated KV buffers as outputs, and they
      replace the tensors used as input. An alternative would be to only return
      the new keys, values and write them into existing buffers (in place).
      TODO: Could compare w.r.t. speed and peak GPU memory usage.
    - Once `chunk_size` tokens are sampled, we need to update the KV caches (not
      just the buffer nodes replicated here) and make decisions for the next
      `chunk_size` slots. This is done by running a normal forward pass through
      the model, updating internal states of caches as needed.
      An alternative would be to write back the buffer node replicas and change
      inner states according to the fixed decisions. We'd not have to recompute
      the updates then. But this would be difficult to support generically, and
      a chunk update of KV caches is cheap anyways.
      TODO: If the added time for the extra update is significant, we could
      explore this in the future.
    """

    def __init__(
        self,
        model: GPT,
        batch_size: int,
        chunk_size: int,
    ):
        self.model = model
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self._next_positions = None
        self._num_tokens_left_in_chunk = 0

    def _check_init_args(self):
        pass  # TODO!

    @property
    def num_tokens_left_in_chunk(self) -> int:
        return self._num_tokens_left_in_chunk

    def update(self, token_idx: torch.Tensor):
        """
        Runs forward pass `model.forward` with tokens `token_idx`. Afterwards,
        `_next_positions` is updated from the KV caches. These indexes cover
        the next `chunk_size` tokens. The defaylt use case is to call this
        method once `num_tokens_left_in_chunk == 0`, passing the `num_chunks`
        tokens which have been sampled for the chunk. But this method can be
        called also if `num_tokens_left_in_chunk > 0`, and `token_idx` can
        have a length different from `chunk_size - num_tokens_left_in_chunk`.

        Args:
            token_idx: New token indices, `(batch_size, num)`

        """
        if token_idx.ndim != 2 or token_idx.shape[0] != self.batch_size:
            raise ValueError(f"token_idx.shape = {token_idx.shape}: Must be 2D and first axis {self.batch_size}")
        # Run forward pass. This updates all KV caches
        logits = self.model(token_idx)
        self._next_positions = [
            cache.next_positions(self.chunk_size).clone()
            for cache in self.model.get_kv_caches()
        ]
        self._num_tokens_left_in_chunk = self.chunk_size
