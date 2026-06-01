# Debug test in `test/kvcache/test_ring_attention.py`

## Initial prompt

Consider the test `test_sdpa_distributed_vs_single_on_chunk` in `test/kvcache/test_ring_attention.py`.
Instances are parameterized by `n_head, n_query_groups, q_len, kv_len_per_rank, dtype, input_pos, num_devices, do_q_lens, is_1d`.
All instances of this test pass, except for `(24, 8,   8, 256,  torch.float16,  256 * 5 + 15, 5,  True, False)`,
which fails with:

```text
        for rank, (d_output, s_output) in enumerate(
            zip(dist_outputs, single_outputs)
        ):
            print(f"Outputs for rank {rank}")
>           torch.testing.assert_close(d_output, s_output, atol=atol, rtol=rtol)
E           AssertionError: Tensor-likes are not close!
E
E           Mismatched elements: 374 / 12288 (3.0%)
E           Greatest absolute difference: 0.011260986328125 at index (1, 17, 1, 15) (up to 0.0005 allowed)
E           Greatest relative difference: 447.25 at index (1, 17, 1, 119) (up to 0.1 allowed)

test/kvcache/test_ring_attention.py:242: AssertionError
```

This seems to depend on the `num_devices` value. If `(256 * 5 + 15, 5)` is replaced
by `(256 * num_devices, num_devices)`, then the test passes for
`num_devices in (2, 3, 4, 8)`, but it fails for `num_devices in (5, 6, 7)`, with
similar errors as above.

* Please analyze this and tell me what is wrong
* Append the summary of your findings to `ai_dev/debug_ring_attention_test.md`

## Findings

### Root cause: off-by-one cap in `_get_q_len_per_rank`

The bug is in [`keys_values/kvcache/parallel/ring_attention.py:121`](keys_values/kvcache/parallel/ring_attention.py#L121),
inside `RingAttentionDriver._get_q_len_per_rank`:

```python
# BUGGY
end = min(start + num_p1, q_len_min)
```

should be:

```python
# FIXED
end = min(start + num_p1, ndevs)
```

### What the function is supposed to do

New tokens at positions `input_pos, input_pos+1, ..., input_pos+q_len-1` are
distributed so that token at position `p` goes to rank `p % num_devices`.
With `start = input_pos % ndevs`, the `num_p1 = q_len % ndevs` "extra" tokens
(those that get `q_len_min + 1 = q_len_max` instead of `q_len_min`) go to ranks
`start, start+1, ..., (start + num_p1 - 1) % ndevs`, wrapping around modulo `ndevs`.

The code implements this wrap-around as two segments:
1. The non-wrapping part: `result[start : end] = q_len_max`, where `end = min(start + num_p1, ndevs)`
2. The wrap-around remainder: `result[0 : num_rem] = q_len_max`, where `num_rem = start + num_p1 - end`

### Why `q_len_min` is wrong

The boundary for when to switch from the non-wrapping to the wrap-around segment
is `ndevs` (the index where we "fall off" the rank list). But the code uses `q_len_min`
instead, which equals `q_len // ndevs`.

When `q_len_min < ndevs` (i.e., `q_len < 2 * ndevs`) **and** `num_p1 > q_len_min`
(i.e., `q_len % ndevs > q_len // ndevs`), these are different values, and the cap
`q_len_min` prematurely ends the first segment. The remainder `num_rem` is computed
relative to that wrong cap, so:
- The wrap-around segment `result[0:num_rem]` is run as intended, but with an inflated
  `num_rem` that double-counts some rank-0 positions.
- The "middle" ranks in `[q_len_min, ndevs)` that should have gotten `q_len_max` are
  skipped.
- Net effect: `sum(result) < q_len` — the total count of assigned tokens is wrong,
  and one rank is systematically under-assigned.

### Numerical example: `ndevs=5, q_len=8, start=0`

| | value |
|---|---|
| `q_len_min` | 1 |
| `num_p1` | 3 |
| `end` (buggy) | `min(0+3, 1) = 1` |
| `end` (fixed) | `min(0+3, 5) = 3` |

**Buggy execution:**
- Sets `result[0] = 2` (range `[0, 1)`)
- `num_rem = 0 + 3 - 1 = 2`, sets `result[0] = 2`, `result[1] = 2`
- Final: `[2, 2, 1, 1, 1]`, sum = **7** (wrong)

**Fixed execution:**
- Sets `result[0] = result[1] = result[2] = 2` (range `[0, 3)`)
- `num_rem = 0 + 3 - 3 = 0`, no wrap-around
- Final: `[2, 2, 2, 1, 1]`, sum = **8** (correct)

### Why only `num_devices` ∈ {5, 6, 7} fail for `q_len=8`

The condition that triggers the bug is `num_p1 > q_len_min`, i.e., `q_len % N > q_len // N`.
For `q_len=8`:

| `ndevs` | `q_len_min` | `num_p1` | `num_p1 > q_len_min`? |
|---------|-------------|----------|-----------------------|
| 2 | 4 | 0 | — (no extra tokens) |
| 3 | 2 | 2 | 2 > 2? No |
| 4 | 2 | 0 | — |
| **5** | **1** | **3** | **3 > 1? Yes → BUG** |
| **6** | **1** | **2** | **2 > 1? Yes → BUG** |
| 7 | 1 | 1 | 1 > 1? No |
| 8 | 1 | 0 | — |

The bug is active for `ndevs ∈ {5, 6}` with `q_len=8`, which exactly matches the
observed test failures. For `ndevs=7`, `num_p1 = 1 = q_len_min = 1`, so
`min(start + 1, 1) = min(start + 1, ndevs)` whenever `start=0`, which is why `ndevs=7`
also passes when `start = ip % ndevs = 0`.

### Impact

`q_len_per_rank[rank_s]` is passed to `sdpa_ring_flexatt_offdiag` as `q_len_for_s`,
which determines `thresh = kv_len - q_len_for_s` — the boundary in the off-diagonal
causal mask. A wrong `q_len_for_s` shifts `thresh` by 1 for the affected rank,
unmasking or masking the wrong KV entries, causing the attention output mismatch
observed in the test.
