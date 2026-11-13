# Understand Sparse Attention Implemented in OOMB

Applied to codebase: https://github.com/wenhaoli-xmu/OOMB

## Initial prompt

Here is the codebase for fine-tuning a transformer model in the presence of
sparse attention with selective key-value (KV) caching. Please explain to me
exactly which selection strategy is implemented here, and point to
relevant code.

* The training script is `train/train.py`. The relevant configuration is
  `train/config_blockwise_sparse.json`, in particular
  `method = "blockwise-tp-sparse"`. I am only interested in the "sparse"
  part in this codebase.
* A KV cache selection strategy is this. During transformer inference with
  exact (non-sparse) attention, tensors `keys`, `values` are built of
  shape `(batch, num_heads, num_tokens, head_size)`, where `batch` is
  batch dimension, `num_heads` number of heads, `num_tokens` the number
  of tokens processed so far, `head_size` the embedding dimension per head.
  When a new chunk of `num_new` extra tokens is processed, this leads to
  tensors `query_new`, `key_new`, `value_new` of shape
  `(batch, num_heads, num_new, head_size)`. In exact attention, we
  append `key_new`, `value_new` to the cache `keys`, `values` and
  compute scaled dot product attention between `query_new` and the
  updated `keys`, `values`.
* In *sparse attention*, we only store a small fraction of the entries
  of the KV cache `keys`, `values`. Think about KV information as
  `head_size` sized vectors, indexed by `(b, h, j)` into
  `(batch, num_heads, num_tokens)`.
* A KV cache selection strategy decides, when new information
  `query_new`, `key_new`, `value_new` of shape
  `(batch, num_heads, num_new, head_size)` comes in, where to write the
  new `key_new`, `value_new` in the limited sized cache. It returns a
  multi-index of `(b, h, j)` entries, pointing to the entries in the
  hypothetical full-sized cache which are not masked out.
* My current best guess is that the selection policy is implemented in
  `chunkoptim/cache/topk_cache.py`, in `SimpleSparseCacheManager.select`.
  They maintain some `self.avg`, a kind of average of past keys, then
  compute something like attention weights in `scores` with `query`, which
  seem what I call `query_new` above. Then, they apply `topk` to obtain
  an index, which is appended to `self.idx`. There is also a parameter
  `self.page_budget`.
* Please step through the code in order to confirm that this is indeed
  where the selection is happening. If not, please point me to where it
  happens instead.
* Please also explain how `self.avg` is built over updates. I also do not
  understand the role of `self.page_budget`. I also need to know where
  `self.idx` is used to update the KV cache and compute the sparse
  multi-head attention.
* Finally, if you recognize this code as implementation of something that
  is clearly described elsewhere, in a paper or the documentation of
  another open source library, please point this out to me.

## Answer

### 1. Confirmation: selection is in `SimpleSparseCacheManager.select`

Yes, your guess is correct. The entry point in the transformer forward pass is
[train_blockwise_tp_sparse.py:163](chunkoptim/modifiers/train_blockwise_tp_sparse.py#L163):

```python
kv_cache[self.layer_idx].select(ques, stage)   # select which pages to attend to
kv_cache[self.layer_idx].update(keys, vals, stage)   # then store the new pages
```

`select` is always called **before** `update`, so it scores the new queries against *previously
stored* pages only. The newly arriving `key_new, value_new` are always stored
(they become the "recent" pages, see §4).

### 2. How `self.avg` is built

In [topk_cache.py:51-58](chunkoptim/cache/topk_cache.py#L51-L58):

```python
def update(self, key, val, stage):
    key, val = super().update(key, val, stage)         # pads + splits into pages
    avg = torch.cat([k.mean(dim=1, keepdim=True) for k in key], dim=1)
    avg = avg.permute(0, 2, 3, 1).contiguous()
    self.avg = avg if self.avg is None else torch.cat([self.avg, avg], dim=-1)
```

After `super().update`, `key` is a Python list of page tensors each of shape `(batch, page_size, num_kv_heads, head_dim)`. The mean over `dim=1` (the token dimension within the page) produces one `head_dim`-sized representative vector per page. After the permute, `self.avg` accumulates to shape:

```
(batch, num_kv_heads, head_dim, total_pages_stored_so_far)
```

So `self.avg[:, h, :, j]` is the **mean key vector** of historical page `j` for KV head `h`. It is a compact, single-vector summary of each page, growing by one column per page as the sequence is processed chunk by chunk.

### 3. The role of `self.page_budget`

`page_budget = 64` (from [config_blockwise_sparse.json:7](train/config_blockwise_sparse.json#L7)). This is the number of historical pages each query tile is allowed to attend to. With `page_size = 128`, that covers at most `64 × 128 = 8192` historical tokens. When the total number of stored pages is ≤ `page_budget`, `select` appends `None` and full attention is used ([topk_cache.py:41-43](chunkoptim/cache/topk_cache.py#L41-L43)). Sparsification only activates once there are more pages than the budget.

### 4. The selection algorithm, step by step

[topk_cache.py:37-48](chunkoptim/cache/topk_cache.py#L37-L48):

```python
query = query.transpose(1, 2).contiguous()           # (B, H, num_new, D)
         .unflatten(1, (self.num_kv_heads, -1))       # (B, KV_H, GRP, num_new, D)
         .sum(2)                                       # (B, KV_H, num_new, D)  — sum across GQA group
score = (query @ self.avg[..., :end])                 # (B, KV_H, num_new, num_pages)
         .softmax(dim=-1)
         .unflatten(2, (-1, self.page_size))           # (B, KV_H, num_q_tiles, page_size, num_pages)
         .sum([1, 3])                                  # (B, num_q_tiles, num_pages)  — sum over heads and tokens
index = score.topk(k=self.page_budget, dim=-1).indices  # (B, num_q_tiles, page_budget)
self.idx.append(index.to(torch.int32))
```

Concretely:
- The new queries are **grouped-query-aggregated** (summed over the GQA group heads).
- Each is **dot-producted against all stored page mean-keys** to get a soft attention score over pages.
- Softmax normalises these scores.
- Scores are summed over all query tokens within each query tile (block of `page_size` tokens) and over all KV heads.
- **Top `page_budget` historical pages** are selected per query tile.

The result `self.idx[-1]`, shape `(batch, num_query_tiles, page_budget)`, maps each query tile to its set of selected historical page indices.

### 5. Where `self.idx` is consumed

**Sparse attention kernel call** in [flash_paged_topk.py:479-492](chunkoptim/ops/flash_paged_topk.py#L479-L492):

```python
if manager.idx[-1] is not None:
    o, lse, ctx.softmax_scale = _flash_sparse_attn_forward(
        q,
        manager.page_table,       # GPU pointers to all pages (key/val/grad)
        manager.idx[-1],          # select_table: (B, num_q_tiles, page_budget)
        ...
        recent_offset=sum(manager.last_update_pages[:-1]),
        recent_pages=manager.last_update_pages[-1])
```

The Triton forward kernel [flash_paged_topk.py:117-178](chunkoptim/ops/flash_paged_topk.py#L117-L178) has **two separate loops**:

1. **Selected historical pages** (lines 117–145): iterates over the `page_budget` page indices from `select_table`; no causal mask (all historical, hence fully visible).
2. **Current-chunk pages** (lines 147–177): iterates over `recent_pages` pages just added; applies a causal mask `q_idx >= k_idx` for within-chunk causality.

This two-loop structure is what makes the sparse attention both correct (causal for the current chunk) and efficient (only `page_budget` historical pages fetched and computed per tile).

The backward pass ([flash_paged_topk.py:517-530](chunkoptim/ops/flash_paged_topk.py#L517-L530)) reuses the same `manager.idx[-1]` selection, so **gradients only flow back through the selected pages**. Unselected historical pages receive zero gradient in this forward pass.

### 6. Known prior work

This is a recognizable pattern: **page-level query-driven KV selection**. The closest published description is:

**Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference** (Tang et al., ICML 2024). Quest also organises the KV cache into fixed-size pages, computes per-page key statistics, scores them against the current query, and selects top-k pages for exact attention. The difference is that Quest uses the **per-page (min, max) key statistics** to compute an upper bound on attention logits (a theoretically tighter approximation), while this codebase uses the **per-page mean key** as a simpler heuristic proxy.

The two-loop structure (selected pages + recent tokens always attended) is a known pattern in works like **StreamingLLM** (Xiao et al., 2023) and **InfLLM** (Xiao et al., 2024), which also distinguish "distant" (selectively retrieved) from "local/recent" (always retained) KV context.

The other ops files in this repo — [flash_paged_moba.py](chunkoptim/ops/flash_paged_moba.py) and [flash_paged_nsa.py](chunkoptim/ops/flash_paged_nsa.py) — correspond to **MoBA** (Mixture of Block Attention, He et al., 2025) and **NSA** (Native Sparse Attention, Yuan et al., 2025), which are alternative sparse attention kernels in the same codebase but use different selection policies.


## Question

Thanks. Can you explain to me what a "page" is when it comes to storing
key-value information? Say the full cache for `keys` has shape
`(batch, num_kv_heads, num_tokens, head_dim)`, where `num_tokens` is the
number of tokens processed so far.

What is a page of size 128? Is it (1) something of shape `(128, head_dim)`
and indexed to some `(b, h)`, batch position and head? Or is it (2) something
of shape `(batch, num_kv_heads, 128, head_dim)`, so that a page necessarily
contains KV information over all batch positions and heads?

## Answer

It's option **(2)**: a page covers all batch positions and all KV heads,
but only `page_size` *consecutive tokens*.

The evidence is in [kv_cache.py:136-140](chunkoptim/cache/kv_cache.py#L136-L140),
inside `SimpleCacheManager.update`:

```python
if key.shape[1] > self.page_size:
    key = list(torch.chunk(key, chunks=key.shape[1] // self.page_size, dim=1))
    val = list(torch.chunk(val, chunks=val.shape[1] // self.page_size, dim=1))
else:
    key, val = [key], [val]
```

The input `key` arrives with shape `(batch, num_tokens, num_kv_heads, head_dim)`
(token dimension is `dim=1`, because `head_first=False` is used in the
projection call). `torch.chunk(..., dim=1)` splits along the token dimension,
so each page tensor has shape `(batch, page_size, num_kv_heads, head_dim)`.

Equivalently, in your notation with `num_kv_heads` before the token dimension,
a page is a contiguous slice `keys[:, :, j*128 : (j+1)*128, :]` of the full
cache — it simultaneously covers all batch elements and all heads, but only
token positions `[j·128, (j+1)·128)`.

This is also why the mean-key representative `self.avg` has shape
`(batch, num_kv_heads, head_dim, num_pages)` — the mean is taken jointly over
all `page_size` tokens in the page, but kept separate per batch element and
per KV head.


## Question

I don't understand one point. The `select` method in `topk_cache.py` appends
an index of shape `(batch, num_q_tiles, page_budget)` to `self.idx`, and this
is passed to `flash_paged_sparse_attn_func` called in `self_attn_forward` in
`train_blockwise_tp_sparse.py`. This means that different pages are selected
for each `(b, j)`, `b` batch position, `j` tile index. On the other hand, you
say that a page covers all batch positions. How does this go together? Please
look into the flash attention code to explain this.

## Answer

The resolution lies in how `stride_kvb` is computed and what `data_ptr()`
points to.

### The page pointers point to batch=0 of the full original tensor

Pages are created by `torch.chunk(key, ..., dim=1)`
([kv_cache.py:137](chunkoptim/cache/kv_cache.py#L137)), where `key` has shape
`(batch, num_tokens, num_kv_heads, head_dim)`. `torch.chunk` on `dim=1`
produces **non-contiguous views** of the original tensor. Each page has:

- shape: `(batch, page_size, num_kv_heads, head_dim)`  
- **stride along dim=0 (batch): `num_tokens × num_kv_heads × head_dim`**

The page's `data_ptr()` stored in the page table points to element
`[0, j·page_size, 0, 0]` of the original tensor — that is, batch=0 of that
page's token range.

### The kernel uses `stride_kvb = num_kv × num_kv_heads × head_dim`

Look at [flash_paged_topk.py:369](chunkoptim/ops/flash_paged_topk.py#L369):

```python
stride_kvb = num_kv * num_kv_heads * kv_head_dim
```

This matches exactly the batch stride of the chunked page tensors above. Then
[line 103](chunkoptim/ops/flash_paged_topk.py#L103):

```python
kv_offs = off_b * stride_kvb + off_kv_h * stride_kvh + (offs_n[:, None] * stride_kvn + offs_d[None, :])
```

Each kernel instance is launched for one `(batch, head)` pair
(`off_hb = tl.program_id(1)`), so `off_b` is fixed per instance. The KV load is:

```python
k_page_ptr = tl.load(T + kv_block_idx * 4)   # raw pointer to element [0, j*page_size, 0, 0]
k = tl.load(k_page_ptr + kv_offs, ...)         # + off_b * stride_kvb navigates to batch off_b
```

This correctly addresses element
`key[off_b, j·page_size + offs_n, off_kv_h, offs_d]`.

### How the per-batch-element selection is used

The selection table is indexed ([line 114](chunkoptim/ops/flash_paged_topk.py#L114)):

```python
sel_ptrs = ST + off_b * stride_sel_b + start_m_block * stride_sel_q
```

So kernel instance `(b=1, h=5, q_tile=3)` reads page indices from
`self.idx[-1][1, 3, :]`, while `(b=0, h=5, q_tile=3)` reads from
`self.idx[-1][0, 3, :]`. These can be completely different sets of page indices.

For each selected page index `kv_block_idx`, the instance loads the page pointer
from `T` and then uses `off_b * stride_kvb` to read only batch=1's data out of
that page.

### Summary

The apparent contradiction resolves as follows: a **page** spans all batch
elements in memory (it is a non-contiguous view of the original
`(batch, num_tokens, ...)` tensor), but each kernel instance independently
selects which pages it wants (per batch element, via
`self.idx[-1][b, q_tile, :]`) and reads only its own batch slice from each
page via the `off_b * stride_kvb` offset. The key to making this work is that
`stride_kvb` passed to the kernel matches the actual batch stride of the
chunked page tensors.
