from itertools import product
import random

import torch
import pytest

from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.gradient.autograd_hooks import (
    CellComputationAutogradHooks,
    NodeAnnotation,
)
from keys_values.kvcache.test_utils import (
    random_tensor,
    available_backends,
    random_index,
)
from keys_values.utils import expand_index, repeat_interleave


@pytest.mark.parametrize(
    "dtype, device",
    product([torch.float32, torch.bfloat16], available_backends()),
)
def test_extract_delta(dtype, device):
    seed = 31415927
    random.seed(seed)
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
        device=device,
    )
    num_repeats = 16

    index_kwargs = dict(dtype=torch.int64, device=device)
    for _ in range(num_repeats):
        keys = random_tensor(params)
        chunk_size = random.randint(1, cache_length // 2)
        input_pos = cache_length + 16
        token_positions = random_index(params, 0, cache_length)
        new_pos = random_index(params, 0, cache_length, num=chunk_size)
        token_positions.scatter_(
            -1,
            new_pos,
            torch.arange(
                input_pos, input_pos + chunk_size, **index_kwargs,
            ).view(1, 1, -1).expand(batch_size, n_query_groups, -1)
        )
        index_e = expand_index(new_pos, head_size)
        delta = keys.gather(2, index_e)
        assert delta.shape == (batch_size, n_query_groups, chunk_size, head_size)
        # Transform as in `sdpa_wrapper.scaled_dot_product_attention`
        sort_index = torch.argsort(token_positions, dim=-1)
        keys = keys.gather(2, expand_index(sort_index, head_size))
        keys = repeat_interleave(keys, n_head)
        assert keys.shape == (batch_size, n_head, cache_length, head_size)
        # Annotation as in `TrainingAttnWeightsReplayCacheNew._create_node_before_creator`
        annotation = NodeAnnotation(
            kind="scatter-ext-key",
            layer_idx=0,
            chunk_idx=2,
            shape=tuple(keys.shape),
            index=index_e,
            delta=delta,
            positions=None,
            extra_info={"sort_index": sort_index},
        )
        annot_delta = CellComputationAutogradHooks._delta_for_annotation(
            annotation=annotation,
            n_head=n_head,
        )
        parg_delta = CellComputationAutogradHooks._delta_for_pack_argument(
            x=keys,
            annotation=annotation,
            n_head=n_head,
        )
        torch.testing.assert_close(annot_delta, parg_delta)
