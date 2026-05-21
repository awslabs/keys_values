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
