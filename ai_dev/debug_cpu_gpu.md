# Debug CPU/GPU Transfer Code

## Initial prompt

I am trying to write code running on an Nvidia A100 GPU, computing gradients with
activation checkpointing. The method of interest is
`keys_values.kvcache.gradient.accumulate.GradientAccumulator.run`.

* This method is computing gradients for a number of layers of a model, using
  activation checkpointing.
* It is running a loop over cells, starting in line 732. This loop runs backwards
  over cells, so `col_idx` is decreasing.
* The computation for a cell has inputs `cell_inputs`, `head_gradients_top`,
  `k_buffers` (list), `v_buffers` (list). They are loaded from CPU using
  `get_inputs_slice`, `get_head_gradients_slice`, and `self._get_checkpoints`.
  This happens in `self._load_from_cpu`.
* It computes gradients using `Pytorch autograd`, but also has outputs
  `head_gradients_bottom`. They are stored to CPU, using `write_head_gradients_slice`.
  This happens in `self._write_to_cpu`.
* `get_inputs_slice`, `get_head_gradients_slice`, `write_head_gradients_slice` are
  defined in `keys_values.kvcache.gradient.main.LongContextGradientModel._backward_accumulate_gradients_nocheck`,
  which is the method calling `GradientAccumulator.run`. These functions use
  `self.layer_checkpoints.get_checkpoint` and `self.layer.layer_checkpoints.set_checkpoint`.
* For the configuration used, `self.layer_checkpoints` is of type
  `keys_values.kvcache.gradient.checkpoints.LayerInputDefaultCheckpoints`. It is
  created in the `LongContextGradientModel._create_layer_checkpointers` method.

My goal is to write code for which transfer of memory from CPU to GPU, transfer of
memory from GPU to CPU, and GPU computation all run in parallel inside each
iteration of the loop.

* I am trying to do this with double buffers (type `DoubleBuffer`) for all inputs
  loaded from CPU and all outputs written to CPU.
* Inputs are prefetched, so loaded for cell `col_idx - 1` in iteration `col_idx`.
* Outputs saving to CPU is delayed, done for cell `col_idx + 1` in iteration
  `col_idx`.
* I use streams `device_to_host_stream`, `host_to_device_stream`, and the current
  stream, they are all different `torch.cuda.Stream` objects.
* All CPU memory involved is pinned, i.e. created with `pin_memory=True`.
* All transfers from CPU to GPU are done by `.to(device, non_blocking=True)`, all
  transfers from GPU to CPU by `.copy_(..., non_blocking=True)`.

But when I run and profile this code, I see:

* Memory transfer and GPU computation do not run in parallel
* Almost all activity is on the main thread
* The total time when using 3 streams is about the same as using a single stream.

Can you explain what is wrong here? What am I missing? Can you detect any
violations (e.g., transfers without `non_blocking=True`, or CPU tensors involved
which are not pinned)? Can you think of other reasons why memory transfer and
computation does not run in parallel?

I have run Nvidia Nsight Compute profiling for two variants. Result files are:

* `~/tmp/profiling/cpu_gpu_noasync.nsys-rep`: Only one stream is used, so
  `device_to_host_stream=None, host_to_device_stream=None`.
* `~/tmp/profiling/cpu_gpu_async.nsys-rep`: 3 streams are used, so
  `device_to_host_stream=None`, `host_to_device_stream` are independent
  `torch.cuda.Stream` objects.

The same code was run. You can see that:

* Running times are comparable (whereas I'd expect the time for 3 streams to
  be shorter)
* Memory transfer and computation does not happen in parallel.


## Answer Claude Code

### Root Causes: Why Memory Transfer and Computation Don't Run in Parallel

1. sync_all(events_for_sync) blocks everything — the main culprit

At accumulate.py:830, after dispatching both _load_from_cpu and _write_to_cpu, the code immediately calls sync_all(events_for_sync). This calls event.synchronize() on the CPU, which stalls the Python thread until both the host-to-device and device-to-host streams have finished. Only then does computation proceed on the main stream.

The correct approach is to not synchronize on the CPU at all — instead, use stream.wait_event(event) to express cross-stream ordering constraints on the GPU, and let the GPU hardware schedule transfers and compute in parallel. event.synchronize() is a CPU barrier; you need current_stream.wait_event(h2d_event) (a GPU-side barrier).

2. _get_checkpoints runs on the default/main stream — not inside host_to_device_stream

In _load_from_cpu (accumulate.py:524), the with stream_decorator(host_to_device_stream) context wraps get_inputs_slice and get_head_gradients_slice, but then _get_checkpoints is called inside this context (accumulate.py:530-537). Let's check what _get_checkpoints does:

In accumulate.py:993, it calls checkpoints.get_checkpoint(chunk_idx=chunk_idx, out=buffers), which leads to _get_checkpoint in checkpoints.py:533 — a .to(device, non_blocking=True) transfer. That is inside the stream context, so it's fine.

But then on accumulate.py:996-997:
```python
k_buffers.append(k_and_v.keys().clone())
v_buffers.append(k_and_v.values().clone())
```
These .clone() calls are also inside the host_to_device_stream context, which is correct. However, k_and_v.keys() returns a tensor that was just asynchronously transferred — reading it immediately via .clone() may force a sync or create a dependency on the transfer completing. This is potentially fine if the clone is enqueued on the same stream, but the concern is whether the tensor is already "arrived."

3. write_head_gradients_slice / set_checkpoint_slice doesn't use non_blocking=True for the GPU→CPU copy
Look at checkpoints.py:988-993 — LayerInputDefaultCheckpoints._set_checkpoint calls cp_int.set_checkpoint_slice(...), which in turn does:

```python
self.k[pos][...].copy_(key, non_blocking=True)
self.v[pos][...].copy_(value, non_blocking=True)
```
These are inside _write_to_cpu via write_head_gradients_slice, which is wrapped in stream_decorator(device_to_host_stream). That's correct.

But notice — _set_checkpoint in checkpoints.py:988-993 calls .to(device=device) without non_blocking=True:

```python
return cp_int.set_checkpoint_slice(
    chunk_idx=layer_idx,
    key=buffers[:, None, :, :ne2].to(device=device),    # <-- BLOCKING!
    value=buffers[:, None, :, ne2:].to(device=device),  # <-- BLOCKING!
    input_pos=rstart,
)
```
This .to(device=device) at checkpoints.py:990-992 moves the GPU tensor to CPU synchronously before the async copy_ runs. However, device=torch.device("cpu") here — since buffers are GPU tensors being sliced and sent to CPU, this is a GPU→CPU transfer that is blocking (non_blocking is not set, defaults to False). This is a concrete violation.

4. get_checkpoint / get_inputs_slice calls torch.cat which may block.

In checkpoints.py:745-755, LayerInputCheckpoints.get_checkpoint calls:
```python
parts = [self._get_checkpoint(...) for ...]
return torch.cat(parts, dim=1)
```
The torch.cat is enqueued on whichever stream is current at the time — if we're inside stream_decorator(host_to_device_stream), it should run on that stream. But after get_inputs_slice returns the tensor and it's stored in the DoubleBuffer, the main stream later reads it with pair_cell_inputs.read() without waiting for host_to_device_stream to finish the cat. The sync_all before reading is supposed to fix this, but see issue #1 above.

5. sync_all is placed before reading from double buffers — correct intent, wrong mechanism

The call sequence in the loop is:

- _load_from_cpu(...) → dispatches on host_to_device_stream, records event
- _write_to_cpu(...) → dispatches on device_to_host_stream, records event
- pair_cell_inputs.read() — reads tensor loaded in previous iteration
- GPU computation (forward_computation + backward)
- sync_all(events_for_sync) — CPU-blocks until transfers from step 1+2 are done
- flip_all(...) — swap double buffers

Step 5 (sync_all) blocks the CPU thread until transfers finish. But at this point, the GPU computation (step 4) was already dispatched — it's running on the GPU. So sync_all here blocks the CPU thread, but the GPU may be busy. The problem is that the transfers in step 1 (for the next iteration) might not overlap with the GPU compute in step 4 because:

- The host_to_device_stream is dispatched at step 1 before the GPU compute is dispatched at step 4, but these are independent streams — the GPU should overlap them.
- However, sync_all is called after compute is dispatched (step 5), and sync_all is a CPU-side event.synchronize(). This doesn't prevent GPU overlap. But it does mean the CPU can't proceed to the next loop iteration until both transfers are done, which prevents prefetching the next-next cell's data from the CPU side.

### Summary of Issues

1.	sync_all uses CPU-blocking event.synchronize() instead of GPU-side stream.wait_event()	accumulate.py:132, accumulate.py:830	Primary: stalls CPU thread, preventing overlap
2.	.to(device=device) without non_blocking=True in _set_checkpoint	checkpoints.py:990-992	Blocking GPU→CPU transfer inside the write path
3.	_get_checkpoints calls .clone() immediately on asynchronously-transferred data	accumulate.py:996-997	Potential implicit sync
4.	Transfer events should be waited on by the compute stream via current_stream.wait_event(), not by the CPU	Design issue in run	GPU can't schedule overlap without inter-stream dependencies

### How to Fix

The core fix is to replace the CPU-blocking synchronization with GPU-side stream ordering:

```python
# Instead of event.synchronize() in sync_all:
# Have the main compute stream wait for h2d transfers before reading
torch.cuda.current_stream().wait_event(h2d_event)

# And: after compute, before d2h transfer starts reading GPU output,
# have d2h stream wait for compute to finish
d2h_event = torch.cuda.current_stream().record_event()
device_to_host_stream.wait_event(d2h_event)
This expresses ordering to the GPU scheduler, letting it overlap transfers with compute, while the CPU thread never stalls.
```

Also fix the blocking transfer in _set_checkpoint (checkpoints.py:990):

```python
# Change:
key=buffers[:, None, :, :ne2].to(device=device),
# To:
key=buffers[:, None, :, :ne2].to(device=device, non_blocking=True)
```


## Second analysis (after fixes)

Thanks, this code runs faster. But I am puzzled about one thing.

I ran the new code (with changes based on your hints) twice, once with
`grad.async_cpu_transfer=False` and once with `grad.async_cpu_transfer=True`.

If `grad.async_cpu_transfer=False`, the code should not use multiple streams at
all:

* In `keys_values.kvcache.gradient.main.LongContextGradientModel`:
  `self.device_to_host_stream=None` and `self.host_to_device_stream=None`
* Then, in `_checkpoint_layer_input` and `checkpoint_layer_input_sync`, nothing
  stream-related is done, so copy from GPU to CPU happens sequentially.
* Also, in `keys_values.kvcache.gradient.accumulate.GradientAccumulator.run`,
  everything should be as before.

I profiled both cases with `Nvidia Nsight Compute`:

* `grad.async_cpu_transfer=False`: Results in `~/tmp/profiling/cpu_gpu_after_noasync.sqlite`
* `grad.async_cpu_transfer=True`: Results in `~/tmp/profiling/cpu_gpu_after_async.sqlite`

While the total time for `grad.async_cpu_transfer=False` is a little more (221 secs)
than for `grad.async_cpu_transfer=True` (214 secs), the difference is quite small,
and I also do not see different patterns.

Can you check again, also inspecting the `sqlite` files, whether we are still missing
something? Can you confirm that CPU-GPU memory transfer nows runs in parallel with GPU
computation if `grad.async_cpu_transfer=True`? And if not, can you figure out why not?

Please append your final summary to the file `ai_dev/debug_cpu_gpu.md` in markdown
format.


## Answer: Analysis of async vs noasync profiles after fixes

### Short answer

**Yes, CPU-GPU memory transfer does now run in parallel with GPU computation** when
`grad.async_cpu_transfer=True`. The profiling confirms it. The reason the speedup is
small (~7 sec out of 221 sec, or ~3%) is that transfers are a relatively small part
of the total wall time, and the dominant cost — PyTorch's CUDA memory allocator
calling `cudaStreamSynchronize` — limits how much benefit overlapping can provide.

### Profiling data summary

All times are GPU-side (from Nsight Systems SQLite, `CUPTI_ACTIVITY_KIND_*` tables).

| Metric | noasync (221 sec) | async (214 sec) |
|---|---|---|
| GPU kernel time | 158.9 sec | 159.0 sec |
| HtoD transfer time | 11.6 sec | 11.8 sec |
| DtoH transfer time | 5.1 sec | 5.2 sec |
| `cudaStreamSynchronize` time (CPU-blocked) | 77.4 sec | 72.0 sec |
| `cudaHostAlloc` time | 17.6 sec | 17.5 sec |
| GPU wall time | 212.2 sec | 205.5 sec |

Observations:
- GPU kernel time is **identical** — the same compute work is done.
- Transfer volumes are **identical** — same data moved in both modes.
- `cudaHostAlloc` (17.5 sec) all happens during the **forward pass** (before the
  backward loop), not during the backward loop itself. It is pinned memory creation
  for `LayerInputDefaultCheckpoints` and KV cache checkpoint buffers.

### Confirmation that overlap works

The SQLite data shows three separate CUDA streams during the backward loop:

- Stream 7: main compute stream (kernels + some memcpy)
- Stream 13: DtoH transfers (`device_to_host_stream`)
- Stream 17: HtoD transfers (`host_to_device_stream`)

A query counting how much of the transfer time on streams 13/17 coincides with
active kernels on stream 7 gives:

```
overlapped transfer time:  8.43 sec  (82% of transfers)
sequential transfer time:  1.81 sec  (18% of transfers)
```

So 82% of the transfer operations on the dedicated transfer streams run **while
the main stream is simultaneously executing kernels**. The overlap is real.

### Why the speedup is only ~7 sec despite 8.4 sec of overlapped transfers

The `cudaStreamSynchronize` calls on the main stream 7 account for 72–77 sec
of wall time in both modes. These are **not from our code**; they come from
PyTorch's CUDA caching memory allocator (`CUDACachingAllocator`). When the
allocator needs to reclaim freed GPU tensors that are still logically in-flight
on a stream, it must synchronize that stream before it can reuse the memory.

The pattern is extremely regular: approximately **2 sec of allocator-induced
stream sync per backward cell** × 36 cells ≈ 72 sec total. Each cell contributes:

- ~1.06 sec between two layer-shard calls: `deallocate_kv_cache_buffers` inside
  `GradientAccumulator.run`'s `finally` block triggers GPU memory reclamation for
  the inference replay cache buffers.
- ~0.9 sec during the backward pass itself: PyTorch's autograd engine allocates
  and frees many tensors during `scalar_output.backward()`, causing repeated
  allocator syncs.

The async mode saves time because transfers on streams 13/17 can still proceed
**during** these allocator stalls on stream 7. The allocator sync freezes stream 7
and the CPU thread, but streams 13 and 17 are unaffected and continue to execute
independently on the GPU hardware. The `cudaStreamSynchronize` time is ~5 sec
shorter in async mode (77.4 → 72.0 sec), which is the overlap benefit. The
remaining ~2 sec savings comes from direct overlap with active compute.

### Remaining issue: blocking `.to(device)` in quantized checkpoint path

In `KVCacheBufferQuantizedCheckpoints.get_checkpoint_slice`
([checkpoints.py:401–402](keys_values/kvcache/gradient/checkpoints.py#L401)):

```python
keys = keys.to(device)    # BLOCKING — no non_blocking=True
values = values.to(device)
```

This is in the quantized path (`qname != "default"`). For these profiling runs the
default (non-quantized) path was used, so this did not affect results. If the
quantized path is ever used with async transfers, this needs to be fixed.

### Can we do better?

The 72 sec of allocator sync is the main remaining bottleneck. Some options:

1. **Avoid re-allocating inference replay caches every layer row**: Currently,
   `_create_inference_replay_caches` allocates GPU buffers and `_deallocate_buffers`
   frees them for every call to `GradientAccumulator.run`. The ~1.06 sec sync per
   row is largely this allocation/deallocation cycle. Pre-allocating these buffers
   once and reusing them across rows would eliminate the largest single sync.

2. **Use `torch.cuda.memory.set_per_process_memory_fraction` or
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**: The expandable segments
   allocator in PyTorch avoids some synchronizations during reallocation.

3. **Accept the current state**: The 7 sec / 3.2% improvement is real and correct.
   The compute kernel time (159 sec) cannot be reduced by transfer overlap — we are
   already limited by the compute and the unavoidable allocator syncs.


## Comment

I acted on your recommendations and also did some further refactoring. Please
consider the following:

* Branch `main`: Before the changes I asked for.
* Branch `cpu_gpu2`: After the changes I asked for. The file
  `~/git/keys_values/diffs` contains the diffs between before and after.

When I run the following command with the code before the change (so branch `main`):
```bash
python3 keys_values/__main__.py finetune_long_lora \
    Qwen/Qwen3-4B-Instruct-2507 \
    --out_dir /home/ubuntu/out/finetune/neurips_exp/lora/qwen3_4b/debug16_before \
    --precision bf16-true \
    --verbose some \
    --devices 1 \
    --data Helmet \
        --data.dataset_key nq \
        --data.max_length 64k \
        --data.metadata_dir /home/ubuntu/out/finetune/data \
        --data.trainloader_longest_first True \
    --train.save_interval 10 \
        --train.micro_batch_size 2 \
        --train.epochs 5 \
        --train.average_loss_per_batch True \
    --eval.interval 10 \
        --eval.initial_validation False \
        --eval.use_sample_metric False \
    --kv_cache.cache_length 32768 \
        --kv_cache.chunk_size 2048 \
        --kv_cache.name h2o-torch-quantized8 \
        --kv_cache.normalize_scores True \
    --grad.layers_per_cell 1 \
        --grad.layercp_qname default \
        --grad.cachecp_qname torch-quantized8 \
    --optimizer.name AdamW \
        --optimizer.learning_rate 0.0005
```

Things work fine.

When I run this command with the code after the change (so branch `cpu_gpu2`):
```bash
python3 keys_values/__main__.py finetune_long_lora \
    Qwen/Qwen3-4B-Instruct-2507 \
    --out_dir /home/ubuntu/out/finetune/neurips_exp/lora/qwen3_4b/debug16_after \
    --precision bf16-true \
    --verbose some \
    --devices 1 \
    --data Helmet \
        --data.dataset_key nq \
        --data.max_length 64k \
        --data.metadata_dir /home/ubuntu/out/finetune/data \
        --data.trainloader_longest_first True \
    --train.save_interval 10 \
        --train.micro_batch_size 2 \
        --train.epochs 5 \
        --train.average_loss_per_batch True \
    --eval.interval 10 \
        --eval.initial_validation False \
        --eval.use_sample_metric False \
    --kv_cache.cache_length 32768 \
        --kv_cache.chunk_size 2048 \
        --kv_cache.name h2o-torch-quantized8 \
        --kv_cache.normalize_scores True \
    --grad.layers_per_cell 1 \
        --grad.layercp_qname default \
        --grad.cachecp_qname torch-quantized8 \
        --grad.async_cpu_transfer True \
    --optimizer.name AdamW \
        --optimizer.learning_rate 0.0005
```

I am getting errors (tail of outputs, dropping some lines, using [...]):
```
Epoch 0 | iter   1 | loss train: 35.250, val_loss valid: n/a | iter time: 177.845 s

[rank 0 | 2026-05-27 12:25:52]: Starting gradient computation.

batch_size      = 2
seq_length      = 60916
cache_length    = 32768
chunk_sizes     = [32768, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 1524]
layers_per_cell = 1
chunks_per_cell = [1, 12, 2]

Forward pass over 15 chunks, grouped into 3 cells (training mode)
100%|██████████| 3/3 [00:21<00:00,  7.12s/it]

Deallocate KV cache buffers

Allocate storage for backward computation

Running backward pass over 36 rows of cells, 36 layers, using activation checkpointing
36it [01:35,  2.65s/it]

There were unmatched pack arguments in some cells. Use --verbose all for full information.
  2 unmatched in (35, 13):  10 matches,  15 comparisons,   0 scatter/cat,   4 4D indexes
  4 unmatched in (35,  1):  70 matches, 105 comparisons,   0 scatter/cat,  24 4D indexes
  2 unmatched in (34, 13):  10 matches,  15 comparisons,   0 scatter/cat,   4 4D indexes
  4 unmatched in (34,  1):  70 matches, 105 comparisons,   0 scatter/cat,  24 4D indexes
[...]
  2 unmatched in ( 0, 13):  10 matches,  15 comparisons,   0 scatter/cat,   4 4D indexes
  4 unmatched in ( 0,  1):  70 matches, 105 comparisons,   0 scatter/cat,  24 4D indexes

Deallocated 28 arrays [total: 0.041 GB].
Optimizer update done.
From check_for_nan_model_weights: transformer.h.0.attn.qkv.lora_A has 40960 NaNs [shape=torch.Size([16, 2560]), numel=40960]
From check_for_nan_model_weights: transformer.h.0.attn.qkv.lora_B has 40960 NaNs [shape=torch.Size([5120, 8]), numel=40960]
From check_for_nan_model_weights: transformer.h.1.attn.qkv.lora_A has 40960 NaNs [shape=torch.Size([16, 2560]), numel=40960]
From check_for_nan_model_weights: transformer.h.1.attn.qkv.lora_B has 40960 NaNs [shape=torch.Size([5120, 8]), numel=40960]
From check_for_nan_model_weights: transformer.h.2.attn.qkv.lora_A has 40960 NaNs [shape=torch.Size([16, 2560]), numel=40960]
From check_for_nan_model_weights: transformer.h.2.attn.qkv.lora_B has 40960 NaNs [shape=torch.Size([5120, 8]), numel=40960]
From check_for_nan_model_weights: transformer.h.3.attn.qkv.lora_A has 40960 NaNs [shape=torch.Size([16, 2560]), numel=40960]
From check_for_nan_model_weights: transformer.h.3.attn.qkv.lora_B has 40960 NaNs [shape=torch.Size([5120, 8]), numel=40960]
[...]
From check_for_nan_model_weights: transformer.h.33.attn.qkv.lora_A has 40960 NaNs [shape=torch.Size([16, 2560]), numel=40960]
From check_for_nan_model_weights: transformer.h.33.attn.qkv.lora_B has 40960 NaNs [shape=torch.Size([5120, 8]), numel=40960]
From check_for_nan_model_weights: transformer.h.34.attn.qkv.lora_A has 40960 NaNs [shape=torch.Size([16, 2560]), numel=40960]
From check_for_nan_model_weights: transformer.h.34.attn.qkv.lora_B has 40960 NaNs [shape=torch.Size([5120, 8]), numel=40960]
From check_for_nan_model_weights: transformer.h.35.attn.qkv.lora_A has 40960 NaNs [shape=torch.Size([16, 2560]), numel=40960]
From check_for_nan_model_weights: transformer.h.35.attn.qkv.lora_B has 40960 NaNs [shape=torch.Size([5120, 8]), numel=40960]

GPU memory at training step 1:
Memory on cuda:0: Used 9.315 GB, Free 30.179 GB


Epoch 0 | iter   2 | loss train: 20.875, val_loss valid: n/a | iter time: 122.751 s

[rank 0 | 2026-05-27 12:27:54]: Starting gradient computation.

batch_size      = 2
seq_length      = 53473
cache_length    = 32768
chunk_sizes     = [32768, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 225]
layers_per_cell = 1
chunks_per_cell = [1, 11]

Forward pass over 12 chunks, grouped into 2 cells (training mode)
100%|██████████| 2/2 [00:17<00:00,  8.59s/it]

Deallocate KV cache buffers

Allocate storage for backward computation

Running backward pass over 36 rows of cells, 36 layers, using activation checkpointing
35it [01:30,  2.58s/it]
Traceback (most recent call last):
  File "/home/ubuntu/git/keys_values/keys_values/__main__.py", line 145, in <module>
    main()
  File "/home/ubuntu/git/keys_values/keys_values/__main__.py", line 141, in main
    auto_cli(PARSER_DATA)
  File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/jsonargparse/_cli.py", line 129, in auto_cli
    return _run_component(component, init.get(subcommand))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/jsonargparse/_cli.py", line 227, in _run_component
    return component(**cfg)
           ^^^^^^^^^^^^^^^^
  File "/home/ubuntu/git/keys_values/keys_values/finetune/longcontext_lora.py", line 225, in setup
    setup_internal(
  File "/home/ubuntu/git/keys_values/keys_values/finetune/longcontext_full.py", line 542, in setup_internal
    fabric.launch(
  File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1010, in launch
    return self._wrap_and_launch(function, self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1121, in _wrap_and_launch
    return to_run(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1126, in _wrap_with_setup
    return to_run(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/git/keys_values/keys_values/finetune/longcontext_full.py", line 888, in main
    token_counts = fit(
                   ^^^^
  File "/home/ubuntu/git/keys_values/keys_values/finetune/longcontext_full.py", line 1583, in fit
    loss.backward()
  File "/home/ubuntu/git/keys_values/keys_values/kvcache/gradient/main.py", line 124, in backward
    self._model.backward()
  File "/home/ubuntu/git/keys_values/keys_values/kvcache/gradient/main.py", line 583, in backward
    self._backward_accumulate_gradients()
  File "/home/ubuntu/git/keys_values/keys_values/kvcache/gradient/main.py", line 1003, in _backward_accumulate_gradients
    self._backward_accumulate_gradients_nocheck(0)
  File "/home/ubuntu/git/keys_values/keys_values/kvcache/gradient/main.py", line 1265, in _backward_accumulate_gradients_nocheck
    self.accumulator.run(
  File "/home/ubuntu/git/keys_values/keys_values/kvcache/gradient/accumulate.py", line 856, in run
    scalar_output.backward()
  File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/torch/_tensor.py", line 630, in backward
    torch.autograd.backward(
  File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/torch/autograd/__init__.py", line 364, in backward
    _engine_run_backward(
  File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/torch/autograd/graph.py", line 865, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/git/keys_values/keys_values/kvcache/gradient/accumulate.py", line 839, in <lambda>
    lambda x: self.autograd_hooks.unpack_hook(x),
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/git/keys_values/keys_values/kvcache/gradient/autograd_hooks.py", line 785, in unpack_hook
    x = self._unpack_from_annotation(annotation)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/git/keys_values/keys_values/kvcache/gradient/autograd_hooks.py", line 852, in _unpack_from_annotation
    raise ValueError(
ValueError: Annotation scatter-key (0,1): (2, 8, 32768, 128): final chunk_idx = 11, must be in [1, 2]
```

* Weights become NaN
* Exception later on

Can you please analyze what is going wrong here?

* Am I missing any synchronization?
* Are the different streams interfering in a bad way?

Please append your final summary to the file `ai_dev/debug_cpu_gpu.md` in markdown
format.



## Round 3: NaN weights and annotation `chunk_idx` error (`cpu_gpu2` branch)

### Symptoms

1. **NaN weights** in all LoRA parameters (all 36 layers) after the first training iteration.
2. **`ValueError: Annotation scatter-key (0,1): (2, 8, 32768, 128): final chunk_idx = 11,
   must be in [1, 2]`** in the second iteration, thrown from `autograd_hooks.py:852`
   during `scalar_output.backward()`.

### Root causes

There are two distinct synchronization bugs introduced in the `cpu_gpu2` branch.

---

#### Bug 1 (primary): `torch.cuda.synchronize()` removed from end of forward pass

**Where**: `long_context.py`, in `_forward_internal`, after the last call to
`_checkpoint_layer_input`. The diff replaced:

```python
if self._do_checkpoint_layer_input() and torch.cuda.is_available():
    torch.cuda.synchronize()
```

with:

```python
self._checkpoint_layer_input_sync(self.config.n_layer)
```

**What goes wrong**: During the forward pass, `_checkpoint_layer_input` enqueues GPU→CPU
copies on `device_to_host_stream`. These copies write layer input checkpoints to
CPU-pinned memory. The replacement `_checkpoint_layer_input_sync` only establishes
GPU-side stream ordering (via `stream.wait_event(event)`) — it does **not** block the
CPU thread and does **not** synchronize the DMA engine.

After the forward pass returns and `backward()` is called, `_backward_accumulate_gradients`
→ `accumulate.run` → `_load_from_cpu` starts. `_load_from_cpu` runs `get_inputs_slice`
on the `host_to_device_stream`, which reads from the same CPU-pinned memory via
`.to(device, non_blocking=True)`. If the d2h DMA writes are still in flight at this
point, the h2d DMA reads corrupt data.

There is no cross-stream synchronization between `device_to_host_stream` (writing
checkpoints) and `host_to_device_stream` (reading them). The only ordering that was
guaranteeing correctness was the old `torch.cuda.synchronize()`, which drained all
streams before the CPU thread could proceed to the backward. Without it, both NaN weights
(all 36 layers) and the annotation mismatch (see below) follow.

---

#### Bug 2 (secondary): first layer's checkpoint races with `wte()`

**Where**: `long_context.py` → `_checkpoint_layer_input` for `block_idx=0`.

**What goes wrong**: For the very first transformer block, `_checkpoint_layer_input` is
called with `x = embeddings` where `embeddings` is the output of `wte(input_ids)`. That
kernel has been **enqueued** on the main stream but may not have **executed** yet. The
d2h copy is immediately enqueued on `device_to_host_stream` with no prior
`wait_event(main_event)`. If the d2h DMA engine starts before the main stream's `wte()`
kernel writes to GPU memory, the checkpoint captures stale or garbage data.

For all subsequent layers (`block_idx > 0`), `_checkpoint_layer_input_sync(N-1)` correctly
orders things by enqueueing:

```python
self.device_to_host_stream.wait_event(cuda_d2h_compute_event)  # computed after block N-1
```

This means d2h cannot start copying layer N's input until block N-1's forward is done —
but layer 0 has no such guard.

---

### Why the annotation error shows `chunk_idx = 11`

The annotation error is a downstream consequence of Bug 1. With corrupted layer-input
checkpoints, `cell_inputs` in the backward forward pass contain wrong data. The
`TrainingAttnWeightsReplayCache` forward then produces an autograd graph whose annotation
nodes do not follow the expected scatter-unwinding sequence. Specifically, the graph may
omit annotation nodes for intermediate chunks (2 through 10), so the unpack hook is never
called for them. When the hook is finally called for `chunk_idx=1` (the first chunk in the
cell), `final_idx` in `_node_annotations.chunk_idx_keys[0]` is still `11` (set during
the corrupted forward), rather than the expected `2` (set by a preceding unpack of
`chunk_idx=2`). This triggers:

```
ValueError: Annotation scatter-key (0,1): final chunk_idx = 11, must be in [1, 2]
```

---

### Is there missing synchronization elsewhere (streams, double buffers)?

No other synchronization issues were found. Specifically:

- **Double-buffer flip logic** is correct: `write_pos` cycles correctly and `read()`
  always returns data from the prior iteration.
- **`main_stream_waits_for_events(events_for_wait)`** correctly has the main stream
  wait (GPU-side) for h2d events before consuming prefetched tensors.
- **`streams_wait_for_event((h2d, d2h), main_event)`** correctly serializes transfer
  streams behind the previous iteration's compute.
- **`_kv_cache_checkpoints` shared `quant_buffers`** are only used sequentially within
  `_get_checkpoints` on h2d stream; no concurrent access with main stream.
- **`_buffers_per_length[cache_length][0]`** used by `_get_checkpoints` as a temporary
  dequantization buffer does not alias anything accessed by `forward_computation` on the
  main stream.

---

### Fix

**Fix for Bug 1**: Re-add a full synchronization between the end of the forward-pass
checkpoint writes and the start of the backward pass reads. The cleanest place is at the
start of `_backward_accumulate_gradients_nocheck`, or equivalently at the top of
`LongContextGradientModel.backward()`, using:

```python
if self.device_to_host_stream is not None:
    self.device_to_host_stream.synchronize()
```

This drains only `device_to_host_stream` (not all streams), which is sufficient and
more targeted than `torch.cuda.synchronize()`.

**Fix for Bug 2**: In `_checkpoint_layer_input`, before starting the d2h copy for
any layer, record a main-stream event and have d2h wait for it:

```python
if self.device_to_host_stream is not None:
    main_event = torch.cuda.current_stream().record_event()
    self.device_to_host_stream.wait_event(main_event)
with stream_decorator(self.device_to_host_stream):
    self.layer_checkpoints.set_checkpoint(...)
    ...
```

This ensures the d2h copy for layer N does not start until the main stream has
committed all work that produced the tensor being copied.
