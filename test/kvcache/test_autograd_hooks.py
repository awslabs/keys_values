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
from itertools import product

import torch
import pytest

from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.gradient.autograd_hooks import (
    CellComputationAutogradHooks,
    NodeAnnotation,
    MAX_DELTA_TRANS_LENGTH,
    create_random_index,
)
from keys_values.kvcache.gradient.train_attn_weights_replay_new import (
    TrainingAttnWeightsReplayCacheNew,
)
from keys_values.kvcache.test_utils import (
    random_tensor,
    available_backends,
    random_index,
)
from keys_values.utils import expand_index, repeat_interleave, randint_torch


@pytest.mark.parametrize(
    "device, dtype",
    product(available_backends(), [torch.float32, torch.bfloat16]),
)
def test_extract_delta(device, dtype):
    seed = 31415927
    torch.random.manual_seed(seed)

    n_head = 32
    n_query_groups = 8
    head_size = 64
    batch_size = 4
    cache_length = 4096
    params = KVCacheParams(
        max_batch_size=batch_size,
        n_query_groups=n_query_groups,
        cache_length=cache_length,
        head_size=head_size,
        n_head=n_head,
        dtype=dtype,
    )
    num_repeats = 16

    index_kwargs = dict(dtype=torch.int64, device=device)
    for _ in range(num_repeats):
        keys = random_tensor(params, device=device)
        chunk_size = randint_torch(1, cache_length // 2)
        input_pos = cache_length + 16
        token_positions = random_index(
            params,
            0,
            cache_length,
            device=device,
        )
        delta_index = random_index(
            params,
            0,
            cache_length,
            num=chunk_size,
            device=device,
        )
        token_positions.scatter_(
            -1,
            delta_index,
            torch.arange(
                input_pos,
                input_pos + chunk_size,
                **index_kwargs,
            )
            .view(1, 1, -1)
            .expand(batch_size, n_query_groups, -1),
        )
        delta_index = expand_index(delta_index, head_size).to(dtype=torch.int32)
        # Transform as in `sdpa_wrapper.scaled_dot_product_attention`
        sort_index = torch.argsort(token_positions, dim=-1).to(dtype=torch.int32)
        keys_after = keys.gather(2, expand_index(sort_index, head_size))
        keys_after = repeat_interleave(keys_after, n_head)
        assert keys_after.shape == (batch_size, n_head, cache_length, head_size)
        # Annotation as in `TrainingAttnWeightsReplayCacheNew._create_node_after_creator`
        index_len = delta_index.shape[2]
        if index_len >= MAX_DELTA_TRANS_LENGTH:
            ext_index = delta_index[:, :, :MAX_DELTA_TRANS_LENGTH, :]
        else:
            shape = (
                batch_size,
                n_query_groups,
                MAX_DELTA_TRANS_LENGTH - index_len,
                head_size,
            )
            index2 = create_random_index(
                shape=shape,
                length=cache_length,
                device=device,
                dtype=torch.int32,
            )
            ext_index = torch.cat((delta_index, index2), dim=2)
        delta = repeat_interleave(keys.gather(2, ext_index), n_head)
        assert delta.shape == (batch_size, n_head, MAX_DELTA_TRANS_LENGTH, head_size)
        ext_index = repeat_interleave(
            TrainingAttnWeightsReplayCacheNew._transform_index(
                index=ext_index,
                sort_index=sort_index,
            ),
            n_head,
        )
        annotation = NodeAnnotation(
            kind="ext-key",
            layer_idx=0,
            chunk_idx=2,
            shape=tuple(keys.shape),
            index=ext_index,
            delta=delta,
            positions=None,
            extra_info={"sort_index": sort_index},
        )
        parg_delta = CellComputationAutogradHooks._delta_for_pack_argument(
            x=keys_after,
            annotation=annotation,
        )
        torch.testing.assert_close(delta, parg_delta)
