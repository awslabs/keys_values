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

from keys_values.config import Config
from keys_values.kvcache.base import DefaultKVCacheReplayLog, KVCacheParams
from keys_values.kvcache.gradient.autograd_hooks import (
    CellComputationAutogradHooks,
    PackArgumentAsAnnotation,
)
from keys_values.kvcache.gradient.annotation import (
    NodeAnnotation,
    create_random_index,
    MAX_DELTA_TRANS_LENGTH,
)
from keys_values.kvcache.test_utils import (
    random_tensor,
    available_backends,
    random_index,
)
from keys_values.utils import expand_index, repeat_interleave, randint_torch


def _transform_index(
    index: torch.Tensor,
    sort_index: torch.Tensor,
) -> torch.Tensor:
    batch_size, n_query_groups, num, head_size = index.shape
    si_len = sort_index.shape[-1]
    assert sort_index.shape == (batch_size, n_query_groups, si_len)
    sort_index = sort_index.to(dtype=index.dtype)
    index = index[:, :, :, 0]
    result = (
        torch.empty_like(sort_index)
        .scatter_(
            2,
            sort_index,
            torch.arange(
                si_len,
                dtype=index.dtype,
                device=index.device,
            )
            .view(1, 1, -1)
            .expand(batch_size, n_query_groups, -1),
        )
        .gather(2, index)
    )
    return expand_index(result, head_size)


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
            _transform_index(
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


@pytest.mark.parametrize(
    "device, num_states",
    product(available_backends(), [2, 3, 5]),
)
def test_unpack_walks_multi_chunk_annotation_chain(device, num_states):
    """
    Regression test for issue #148: `_unpack_from_annotation` used to require
    the final buffer to be at most one chunk ahead of the annotation being
    unpacked. With long generated regions spanning 3+ chunks, autograd's
    backward can be served for intermediate chunks without unpacking their
    "scatter-*" annotations, so the gap can grow beyond one chunk, and the
    backward failed with `ValueError: ... final chunk_idx = 3, must be in
    [1, 2]`.

    We build a chain of ground-truth buffer states `1, ..., num_states`,
    linked by "scatter-value" annotations (the annotation with
    `chunk_idx == c` reconstructs state `c` from state `c + 1`), set the
    final buffer to state `num_states`, and then unpack the annotation for
    chunk 1 directly (gap of `num_states - 1` chunks). The unpack must walk
    the chain, and the intermediate states (applied early) must still be
    served when their IDs are unpacked later.

    """
    seed = 31415927
    torch.random.manual_seed(seed)
    dtype = torch.float32

    batch_size = 2
    n_head = 4
    n_query_groups = 2
    head_size = 8
    cache_length = 32
    chunk_size = 8
    layer_idx = 0
    kind = "scatter-value"

    config = Config(
        n_layer=1,
        n_head=n_head,
        n_query_groups=n_query_groups,
        n_embd=n_head * head_size,
        block_size=cache_length + num_states * chunk_size,
        vocab_size=48,
        rotary_percentage=1,
    )
    params = KVCacheParams(
        max_batch_size=batch_size,
        n_query_groups=n_query_groups,
        cache_length=cache_length,
        head_size=head_size,
        n_head=n_head,
        dtype=dtype,
    )
    hooks = CellComputationAutogradHooks(
        config=config,
        batch_size=batch_size,
    )
    token_kwargs = dict(dtype=torch.int64, device=device)
    replay_log = DefaultKVCacheReplayLog(
        token_chunks=[torch.zeros(batch_size, cache_length, **token_kwargs)]
        + [
            torch.zeros(batch_size, chunk_size, **token_kwargs)
            for _ in range(num_states)
        ],
        cache_length=cache_length,
        max_prefill_length=cache_length,
        grace_period=0,
    )
    hooks.initialize_cell(
        eff_num_layers=1,
        num_chunks=num_states + 1,
        first_layer_idx=layer_idx,
        first_chunk_idx=0,
        cache_lengths=[cache_length],
        replay_logs=[replay_log],
    )

    # Ground-truth buffer states 1, ..., num_states. State `c + 1` arises
    # from state `c` by scattering new values at duplicate-free indexes; the
    # annotation with `chunk_idx == c` stores this index and the overwritten
    # old values (`delta`), as in
    # `TrainingAttnWeightsReplayCache._create_node_before_creator`
    buffer_kwargs = dict(dtype=dtype, device=device)
    states = {
        1: torch.randn(
            batch_size, n_query_groups, cache_length, head_size, **buffer_kwargs
        )
    }
    for c in range(2, num_states + 1):
        prev = states[c - 1]
        index = expand_index(
            random_index(params, 0, cache_length, num=chunk_size, device=device),
            head_size,
        )
        hooks.node_annotations.append_safe(
            NodeAnnotation(
                kind=kind,
                layer_idx=layer_idx,
                chunk_idx=c - 1,
                shape=tuple(prev.shape),
                index=index,
                delta=prev.gather(2, index),
            )
        )
        new_values = torch.randn(
            batch_size, n_query_groups, chunk_size, head_size, **buffer_kwargs
        )
        states[c] = prev.scatter(2, index, new_values)
    hooks.node_annotations.set_final(
        x=states[num_states],
        layer_idx=layer_idx,
        chunk_idx=num_states,
        kind=kind,
    )
    # Simulate the forward/backward boundary: unmatched "scatter" annotations
    # are entered into `_packed_arg_for_id` under fresh IDs
    hooks._match_annotations(flush_pack_args=True)
    ids = {
        e.annot.chunk_idx: idd
        for idd, e in hooks._packed_arg_for_id.items()
        if isinstance(e, PackArgumentAsAnnotation)
    }
    assert set(ids.keys()) == set(range(1, num_states))

    # Unpack the annotation for chunk 1 with the final buffer at chunk
    # `num_states`. Before the fix, this raised ValueError for
    # `num_states > 2`
    x1 = hooks.unpack_hook(ids[1])
    torch.testing.assert_close(x1, states[1])
    assert hooks.node_annotations.get_final(layer_idx, kind)[1] == 1

    # Intermediate annotations were applied early; their reconstructed
    # states must still be served when autograd unpacks their IDs
    for c in range(2, num_states):
        x_c = hooks.unpack_hook(ids[c])
        torch.testing.assert_close(x_c, states[c])
    # All parked states have been consumed
    assert not hooks._id_to_unpacked
