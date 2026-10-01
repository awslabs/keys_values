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

    # The intermediate annotations were applied early. They were inserted by
    # the flush purely to keep the chain complete (no autograd node refers
    # to them), so no claim was recorded and no full-size state was parked
    # for them. Their entries only hold the small per-chunk deltas until the
    # cell is cleared.
    assert not hooks._early_applied
    assert hooks._parked_bytes == 0
    assert not hooks._id_to_unpacked


@pytest.mark.parametrize("device", available_backends())
def test_unpack_out_of_order_request_after_walking_past(device):
    """
    Second regression test for issue #148. After the chain-walk fix, long
    32k runs hit the mirror case: the buffer had been walked *past* a chunk,
    and autograd then asked for that chunk's state, failing with
    `final chunk_idx = 14, must be >= 15`.

    This happens when an annotation is applied early (to serve an `ext-*`
    annotation for the same chunk, or as part of a chain walk), the buffer is
    then walked further down by a later request, and only afterwards does
    autograd unpack the early-applied annotation's own ID. The state must be
    parked when it is applied early, so it can still be served.

    Here we unpack chunk 2 first (walking final 3 -> 2), then chunk 1
    (walking 2 -> 1), then ask for chunk 2 again -- which is only possible if
    chunk 2's state was retained.

    """
    seed = 271828
    torch.random.manual_seed(seed)
    dtype = torch.float32

    batch_size = 2
    n_head = 4
    n_query_groups = 2
    head_size = 8
    cache_length = 32
    chunk_size = 8
    num_states = 3
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

    buffer_kwargs = dict(dtype=dtype, device=device)
    states = {
        1: torch.randn(
            batch_size, n_query_groups, cache_length, head_size, **buffer_kwargs
        )
    }
    annotations = {}
    for c in range(2, num_states + 1):
        prev = states[c - 1]
        index = expand_index(
            random_index(params, 0, cache_length, num=chunk_size, device=device),
            head_size,
        )
        annot = NodeAnnotation(
            kind=kind,
            layer_idx=layer_idx,
            chunk_idx=c - 1,
            shape=tuple(prev.shape),
            index=index,
            delta=prev.gather(2, index),
        )
        annotations[c - 1] = annot
        hooks.node_annotations.append_safe(annot)
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

    # Register the chunk-2 annotation as a *matched* pack argument (i.e., an
    # ID the autograd graph really refers to), so it must be served even
    # after the buffer has been walked past it. The chunk-1 annotation is
    # flushed as usual.
    matched_id = 4242
    hooks._add_packed_annotation(
        matched_id,
        PackArgumentAsAnnotation(
            annot=annotations[2],
            target_dtype=None,
        ),
    )
    hooks.node_annotations.nodes.remove(annotations[2])
    hooks._match_annotations(flush_pack_args=True)
    chunk1_id = next(
        idd
        for idd, e in hooks._packed_arg_for_id.items()
        if isinstance(e, PackArgumentAsAnnotation) and e.annot.chunk_idx == 1
    )

    # Walk down to chunk 1. This applies the chunk-2 annotation early (as
    # part of the chain walk) and must park its state
    x1 = hooks.unpack_hook(chunk1_id)
    torch.testing.assert_close(x1, states[1])
    assert hooks.node_annotations.get_final(layer_idx, kind)[1] == 1

    # Now autograd asks for chunk 2, whose state the buffer has moved past.
    # Before the fix this raised `final chunk_idx = 1, must be >= 2`
    x2 = hooks.unpack_hook(matched_id)
    torch.testing.assert_close(x2, states[2])


@pytest.mark.parametrize("device", available_backends())
def test_parked_memory_is_bounded_and_released(device):
    """
    Memory guarantee for the chain walk (issue #148 review question).

    States rebuilt ahead of their unpack request are parked so the late
    request can be served. This test pins the two properties that bound the
    cost:

    1. Only states whose ID the autograd graph can actually request are
       parked. IDs inserted by the flush purely to keep the chain complete
       are never parked, so walking a long chain of them costs nothing.
    2. A parked state is released as soon as its request arrives, so the
       peak is set by how many requests are outstanding at once, not by the
       chain length.

    """
    torch.random.manual_seed(31415927)
    dtype = torch.float32
    batch_size = 2
    n_head = 4
    n_query_groups = 2
    head_size = 8
    cache_length = 32
    chunk_size = 8
    num_states = 6  # long chain: gap of 5 from the final buffer
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
    hooks = CellComputationAutogradHooks(config=config, batch_size=batch_size)
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

    buffer_kwargs = dict(dtype=dtype, device=device)
    states = {
        1: torch.randn(
            batch_size, n_query_groups, cache_length, head_size, **buffer_kwargs
        )
    }
    annotations = {}
    for c in range(2, num_states + 1):
        prev = states[c - 1]
        index = expand_index(
            random_index(params, 0, cache_length, num=chunk_size, device=device),
            head_size,
        )
        annot = NodeAnnotation(
            kind=kind,
            layer_idx=layer_idx,
            chunk_idx=c - 1,
            shape=tuple(prev.shape),
            index=index,
            delta=prev.gather(2, index),
        )
        annotations[c - 1] = annot
        hooks.node_annotations.append_safe(annot)
        states[c] = prev.scatter(
            2,
            index,
            torch.randn(
                batch_size, n_query_groups, chunk_size, head_size, **buffer_kwargs
            ),
        )
    hooks.node_annotations.set_final(
        x=states[num_states],
        layer_idx=layer_idx,
        chunk_idx=num_states,
        kind=kind,
    )

    # Case 1: the whole chain is flush-inserted (orphan) except the one being
    # requested. Walking 5 links must park nothing at all.
    matched_id = 9001
    hooks._add_packed_annotation(
        matched_id, PackArgumentAsAnnotation(annot=annotations[1], target_dtype=None)
    )
    hooks.node_annotations.nodes.remove(annotations[1])
    hooks._match_annotations(flush_pack_args=True)

    x1 = hooks.unpack_hook(matched_id)
    torch.testing.assert_close(x1, states[1])
    log = hooks.annotation_usage_log()
    assert log.parked_peak_count == 0, "orphan chain links must not be parked"
    assert log.parked_peak_bytes == 0
    assert not hooks._id_to_unpacked

    # Case 2: every chain link is a real (matched) ID, so each is parked when
    # applied early -- and released when its request arrives. The peak equals
    # the number of outstanding requests, and memory returns to zero.
    hooks.initialize_cell(
        eff_num_layers=1,
        num_chunks=num_states + 1,
        first_layer_idx=layer_idx,
        first_chunk_idx=0,
        cache_lengths=[cache_length],
        replay_logs=[replay_log],
    )
    ids = {}
    for chunk_idx, annot in annotations.items():
        ids[chunk_idx] = 9100 + chunk_idx
        hooks._add_packed_annotation(
            ids[chunk_idx], PackArgumentAsAnnotation(annot=annot, target_dtype=None)
        )
    hooks.node_annotations.set_final(
        x=states[num_states],
        layer_idx=layer_idx,
        chunk_idx=num_states,
        kind=kind,
    )
    hooks._match_annotations(flush_pack_args=True)

    # Worst case: request the oldest state first, forcing the full walk
    torch.testing.assert_close(hooks.unpack_hook(ids[1]), states[1])
    log = hooks.annotation_usage_log()
    one_buffer_bytes = states[1].numel() * states[1].element_size()
    # num_states - 2 intermediate links get applied early and parked
    assert log.parked_peak_count == num_states - 2
    assert log.parked_peak_bytes == (num_states - 2) * one_buffer_bytes
    # Bound claimed in review: at most (chunks_per_cell - 1) buffers per
    # (layer, kind). Never a function of model size or step count.
    assert log.parked_peak_count <= (num_states - 1)

    # Serving the outstanding requests releases everything
    for chunk_idx in range(2, num_states):
        torch.testing.assert_close(hooks.unpack_hook(ids[chunk_idx]), states[chunk_idx])
    assert not hooks._id_to_unpacked
    assert hooks._parked_bytes == 0


@pytest.mark.parametrize("device", available_backends())
def test_late_ext_request_served_from_parked_state(device):
    """
    Third variant of the issue-#148 ordering violation, observed in a real
    36-layer run with 1500-chunk sequences:

        ext-key (28,1494): final chunk_idx = 1493, must be in [1494, 1495]

    An "ext-*" request arrives AFTER the buffer has been walked past its
    chunk. The state an ext at chunk c needs is the buffer after chunk c;
    once the walk moves `final` below c, the live buffer can no longer serve
    it. The fix parks a copy when the state for chunk c is produced while an
    ext annotation for (layer, kind, c) is still outstanding, and serves the
    late request from that copy (`apply_ext_annotation` does not touch the
    live buffer).

    Here: states V1 -> V2 -> V3 (final at 3). An ext-value annotation for
    chunk 2 is outstanding. Unpack scatter chunk 2 (final 3 -> 2, ext state
    parked), then scatter chunk 1 (final 2 -> 1, buffer now PAST chunk 2),
    then the ext request for chunk 2 arrives. Pre-fix code raises exactly
    Matthias's error; fixed code serves the extended V2.

    """
    torch.random.manual_seed(271828)
    dtype = torch.float32
    batch_size = 2
    n_head = 4
    n_query_groups = 2
    head_size = 8
    cache_length = 32
    chunk_size = 8
    num_states = 3
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
    hooks = CellComputationAutogradHooks(config=config, batch_size=batch_size)
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

    buffer_kwargs = dict(dtype=dtype, device=device)
    states = {
        1: torch.randn(
            batch_size, n_query_groups, cache_length, head_size, **buffer_kwargs
        )
    }
    scatter_annots = {}
    for c in range(2, num_states + 1):
        prev = states[c - 1]
        index = expand_index(
            random_index(params, 0, cache_length, num=chunk_size, device=device),
            head_size,
        )
        scatter_annots[c - 1] = NodeAnnotation(
            kind=kind,
            layer_idx=layer_idx,
            chunk_idx=c - 1,
            shape=tuple(prev.shape),
            index=index,
            delta=prev.gather(2, index),
        )
        states[c] = prev.scatter(
            2,
            index,
            torch.randn(
                batch_size, n_query_groups, chunk_size, head_size, **buffer_kwargs
            ),
        )
    hooks.node_annotations.set_final(
        x=states[num_states],
        layer_idx=layer_idx,
        chunk_idx=num_states,
        kind=kind,
    )

    # Register: both scatter annotations as matched pack args, plus an
    # OUTSTANDING ext-value annotation for chunk 2 (plain GQA extension:
    # no reorder info, shape has n_head in dim 1)
    ids = {1: 9101, 2: 9102}
    for c, idd in ids.items():
        hooks._add_packed_annotation(
            idd, PackArgumentAsAnnotation(annot=scatter_annots[c], target_dtype=None)
        )
    ext_shape = (batch_size, n_head, cache_length, head_size)
    ext_annot = NodeAnnotation(
        kind="ext-value",
        layer_idx=layer_idx,
        chunk_idx=2,
        shape=ext_shape,
        index=None,
        delta=None,
    )
    ext_id = 9200
    hooks._add_packed_annotation(
        ext_id, PackArgumentAsAnnotation(annot=ext_annot, target_dtype=None)
    )
    hooks._match_annotations(flush_pack_args=True)

    # Descending scatter unpacks walk the buffer: 3 -> 2 -> 1
    torch.testing.assert_close(hooks.unpack_hook(ids[2]), states[2])
    torch.testing.assert_close(hooks.unpack_hook(ids[1]), states[1])
    assert hooks.node_annotations.get_final(layer_idx, kind)[1] == 1

    # NOW the ext request for chunk 2 arrives -- buffer already past it.
    # Pre-fix: ValueError "final chunk_idx = 1, must be in [2, 3]"
    x_ext = hooks.unpack_hook(ext_id)
    from keys_values.utils import repeat_interleave

    torch.testing.assert_close(x_ext, repeat_interleave(states[2], n_head))
    # The parked copy was released on fetch
    assert not hooks._ext_states
    assert hooks._parked_bytes == 0


@pytest.mark.parametrize("device", available_backends())
def test_in_order_ext_first_ordering_parks_nothing(device):
    """
    Performance regression test for issue #152.

    The common ordering in long fine-tuning runs is "ext one step early":
    for each chunk c (descending), autograd asks for `ext-*` at chunk c
    while the buffer is one step ahead (final = c + 1), then for the
    "scatter-*" annotation of chunk c. The pre-#149 code served this with
    zero copies (apply the scatter early, serve its own request later from
    the live buffer). The first #149 fix instead parked a CPU copy of every
    early-applied state, which meant a blocking GPU-CPU round trip per
    chunk per layer per kind, and GPU utilization dropped to 20-40% on
    multi-hundred-chunk workloads (issue #152).

    This test replays that ordering and asserts that NOTHING is ever
    parked: all claims are served from the live buffer, so the lazy scheme
    is copy-free on the fast path.
    """
    torch.random.manual_seed(1618033)
    dtype = torch.float32
    batch_size = 2
    n_head = 4
    n_query_groups = 2
    head_size = 8
    cache_length = 32
    chunk_size = 8
    num_states = 4
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
    hooks = CellComputationAutogradHooks(config=config, batch_size=batch_size)
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

    buffer_kwargs = dict(dtype=dtype, device=device)
    states = {
        1: torch.randn(
            batch_size, n_query_groups, cache_length, head_size, **buffer_kwargs
        )
    }
    scatter_annots = {}
    for c in range(2, num_states + 1):
        prev = states[c - 1]
        index = expand_index(
            random_index(params, 0, cache_length, num=chunk_size, device=device),
            head_size,
        )
        scatter_annots[c - 1] = NodeAnnotation(
            kind=kind,
            layer_idx=layer_idx,
            chunk_idx=c - 1,
            shape=tuple(prev.shape),
            index=index,
            delta=prev.gather(2, index),
        )
        states[c] = prev.scatter(
            2,
            index,
            torch.randn(
                batch_size, n_query_groups, chunk_size, head_size, **buffer_kwargs
            ),
        )
    hooks.node_annotations.set_final(
        x=states[num_states],
        layer_idx=layer_idx,
        chunk_idx=num_states,
        kind=kind,
    )

    # Register scatter annotations for chunks 1..num_states-1 and ext-value
    # annotations for the same chunks
    ext_shape = (batch_size, n_head, cache_length, head_size)
    scatter_ids = {}
    ext_ids = {}
    next_id = 9300
    for c in range(1, num_states):
        scatter_ids[c] = next_id
        hooks._add_packed_annotation(
            next_id,
            PackArgumentAsAnnotation(annot=scatter_annots[c], target_dtype=None),
        )
        next_id += 1
        ext_ids[c] = next_id
        hooks._add_packed_annotation(
            next_id,
            PackArgumentAsAnnotation(
                annot=NodeAnnotation(
                    kind="ext-value",
                    layer_idx=layer_idx,
                    chunk_idx=c,
                    shape=ext_shape,
                    index=None,
                    delta=None,
                ),
                target_dtype=None,
            ),
        )
        next_id += 1
    hooks._match_annotations(flush_pack_args=True)

    from keys_values.utils import repeat_interleave

    # The common in-order sequence: per chunk c (descending), ext(c) arrives
    # one step early, then scatter(c) itself
    for c in range(num_states - 1, 0, -1):
        x_ext = hooks.unpack_hook(ext_ids[c])
        torch.testing.assert_close(x_ext, repeat_interleave(states[c], n_head))
        x_scatter = hooks.unpack_hook(scatter_ids[c])
        torch.testing.assert_close(x_scatter, states[c])
        # The fast path must never copy: nothing parked, ever
        assert hooks._parked_bytes == 0
        assert not hooks._parked_states
        assert not hooks._ext_states

    log = hooks.annotation_usage_log()
    assert log.parked_peak_count == 0
    assert log.parked_peak_bytes == 0
    assert not hooks._early_applied
