# Debug `test_ring_attention_distributed.py`

## Initial prompt

Consider the test `test_sdpa_distributed_vs_single_on_prefill` in
`test/kvcache/test_ring_attention_distributed.py`. When running this test with
the inputs `(4, 2, 512, torch.float16, 3)`, I am getting an exception and the
code hangs. I'd like you to debug my code and tell me what is wrong.

I am running this on a EC2 P4 instance with GPUs 0, 1, 2, 3 free:

```bash
CUDA_VISIBLE_DEVICES="0,1,2" python test/kvcache/test_ring_attention_distributed.py
```

Output is:
```bash
You are using a CUDA device ('NVIDIA A100-SXM4-40GB') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/3
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/3
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/3
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 3 processes
----------------------------------------------------------------------------------------------------

[Rank 0]: Sampling data
[Rank 1]: Broadcasting data
[Rank 2]: Broadcasting data
[Rank 0]: Broadcasting data
[Rank 0]: Created driver
[rank0]:[W701 09:03:30.190545785 ProcessGroupNCCL.cpp:4025] Warning: [PG ID 0 PG GUID 0(default_pg) Rank 0] An unbatched P2P op (send/recv) was called on this ProcessGroup with size 3.  In eager initialization mode, unbatched P2P ops are treated as independent collective ops, and are thus serialized with all other ops on this ProcessGroup, including other P2P ops. To avoid serialization, either create additional independent ProcessGroups for the P2P ops or use batched P2P ops. You can squash this warning by setting the environment variable TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING to false. (function operator())
[Rank 2]: Created driver
[Rank 1]: Created driver
[rank1]:[W701 09:03:30.636975753 ProcessGroupNCCL.cpp:4025] Warning: [PG ID 0 PG GUID 0(default_pg) Rank 1] An unbatched P2P op (send/recv) was called on this ProcessGroup with size 3.  In eager initialization mode, unbatched P2P ops are treated as independent collective ops, and are thus serialized with all other ops on this ProcessGroup, including other P2P ops. To avoid serialization, either create additional independent ProcessGroups for the P2P ops or use batched P2P ops. You can squash this warning by setting the environment variable TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING to false. (function operator())
[rank2]:[W701 09:03:30.636992435 ProcessGroupNCCL.cpp:4025] Warning: [PG ID 0 PG GUID 0(default_pg) Rank 2] An unbatched P2P op (send/recv) was called on this ProcessGroup with size 3.  In eager initialization mode, unbatched P2P ops are treated as independent collective ops, and are thus serialized with all other ops on this ProcessGroup, including other P2P ops. To avoid serialization, either create additional independent ProcessGroups for the P2P ops or use batched P2P ops. You can squash this warning by setting the environment variable TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING to false. (function operator())
[rank2]: Traceback (most recent call last):
[rank2]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 355, in <module>
[rank2]:     test_sdpa_distributed_vs_single_on_prefill(4, 2, 512, torch.float16, 3)
[rank2]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 253, in test_sdpa_distributed_vs_single_on_prefill
[rank2]:     fabric.launch(
[rank2]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1010, in launch
[rank2]:     return self._wrap_and_launch(function, self, *args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1120, in _wrap_and_launch
[rank2]:     return launcher.launch(to_run, *args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/strategies/launchers/subprocess_script.py", line 108, in launch
[rank2]:     return function(*args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1126, in _wrap_with_setup
[rank2]:     return to_run(*args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 316, in run_sdpa_distributed_vs_single_on_prefill
[rank2]:     outputs = driver(
[rank2]:               ^^^^^^^
[rank2]:   File "/home/ubuntu/git/keys_values/keys_values/kvcache/parallel/ring_attention.py", line 191, in __call__
[rank2]:     reqs.append(dist.isend(tensor=buff_keys.read(), dst=rank_send))
[rank2]:                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 2491, in isend
[rank2]:     return group.send([tensor], group_dst, tag)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]: RuntimeError: ncclComm != nullptr INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp":4043, please report a bug to PyTorch. Parent communicator missing in eager initialization mode.
[rank1]: Traceback (most recent call last):
[rank1]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 355, in <module>
[rank1]:     test_sdpa_distributed_vs_single_on_prefill(4, 2, 512, torch.float16, 3)
[rank1]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 253, in test_sdpa_distributed_vs_single_on_prefill
[rank1]:     fabric.launch(
[rank1]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1010, in launch
[rank1]:     return self._wrap_and_launch(function, self, *args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1120, in _wrap_and_launch
[rank1]:     return launcher.launch(to_run, *args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/strategies/launchers/subprocess_script.py", line 108, in launch
[rank1]:     return function(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1126, in _wrap_with_setup
[rank1]:     return to_run(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 316, in run_sdpa_distributed_vs_single_on_prefill
[rank1]:     outputs = driver(
[rank1]:               ^^^^^^^
[rank1]:   File "/home/ubuntu/git/keys_values/keys_values/kvcache/parallel/ring_attention.py", line 191, in __call__
[rank1]:     reqs.append(dist.isend(tensor=buff_keys.read(), dst=rank_send))
[rank1]:                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 2491, in isend
[rank1]:     return group.send([tensor], group_dst, tag)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]: RuntimeError: ncclComm != nullptr INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp":4043, please report a bug to PyTorch. Parent communicator missing in eager initialization mode.
```

Hints:
* The code where things go wrong is `keys_values.kvcache.parallel.ring_attention.RingAttentionDriver`
* I am trying to implement RingAttention, a distributed algorithm for computing
  multi-head self-attention, where buffers need to be sent to and received
  by other ranks in parallel to their computations.
* I am using streams `self._send_stream` and `self._recv_stream` to allow transmission
  to happen in parallel with computation (default stream)
* I am using double buffering `keys_values.kvcache.parallel.ring_attention.DoubleBuffer` to ensure that
  buffers are not read at the same time as being written to
* Append the summary of your findings to `ai_dev/debug_ring_attention_distributed_test.md`

## Findings

### Error

```
RuntimeError: ncclComm != nullptr INTERNAL ASSERT FAILED at
"ProcessGroupNCCL.cpp":4043 — Parent communicator missing in eager
initialization mode.
```

All three ranks crash at the first `dist.isend` call in the ring loop
([ring_attention.py:191](../keys_values/kvcache/parallel/ring_attention.py)).

### Root cause

`dist.isend` / `dist.irecv` are called inside `with torch.cuda.stream(self._send_stream)` and
`with torch.cuda.stream(self._recv_stream)` context managers. The assumption was that this
would dispatch the NCCL communication onto those streams so that it can overlap with
default-stream computation.

That assumption is wrong. NCCL P2P operations (`send` / `recv`) **do not respect
`torch.cuda.stream()` context managers**. In PyTorch's eager NCCL initialization mode
(introduced in PyTorch ~2.4), NCCL creates its communicator once, associated with the
default stream (or whichever stream is current at the point the process group is first
used for a collective — in this case `fabric.broadcast`). When `dist.isend` is then called
from inside a non-default stream context, NCCL cannot find a parent communicator associated
with `self._send_stream` and aborts with the "Parent communicator missing" assert.

The PyTorch warning printed just before the crash also hints at this:
> *An unbatched P2P op (send/recv) was called on this ProcessGroup... In eager
> initialization mode, unbatched P2P ops are treated as independent collective ops, and
> are thus serialized with all other ops on this ProcessGroup.*

### Fix

Remove the `with stream_decorator(self._send_stream):` and
`with stream_decorator(self._recv_stream):` wrappers around the `dist.isend` / `dist.irecv`
calls. Issue them from the default stream context directly:

```python
reqs.append(dist.isend(tensor=buff_keys.read(),   dst=rank_send))
reqs.append(dist.isend(tensor=buff_values.read(), dst=rank_send))
reqs.append(dist.irecv(tensor=buff_keys.write(),  src=rank_recv))
reqs.append(dist.irecv(tensor=buff_values.write(), src=rank_recv))
```

NCCL will schedule the transfer on its own internal communication stream, naturally
overlapping with subsequent default-stream computation (the `ring_att_comp(...)` call).
This is exactly the overlap that the `_send_stream` / `_recv_stream` wrappers were
trying — but failing — to achieve.

After this change, `self._send_stream` and `self._recv_stream` are unused and can be
removed from `RingAttentionDriver.__init__`, along with `stream_decorator`,
`streams_wait_for_event`, and `main_stream_waits_for_events` if they are no longer needed
elsewhere.

The double buffering and `req.wait()` synchronization logic is otherwise correct and
should be left as-is.

## Response 1

I did as you said and ran the test again. Get a similar failure:
```bash
You are using a CUDA device ('NVIDIA A100-SXM4-40GB') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/3
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/3
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/3
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 3 processes
----------------------------------------------------------------------------------------------------

[Rank 0]: Sampling data
[Rank 2]: Broadcasting data
[Rank 1]: Broadcasting data
[Rank 0]: Broadcasting data
[Rank 0]: Created driver
[rank0]:[W701 09:33:37.745071077 ProcessGroupNCCL.cpp:4025] Warning: [PG ID 0 PG GUID 0(default_pg) Rank 0] An unbatched P2P op (send/recv) was called on this ProcessGroup with size 3.  In eager initialization mode, unbatched P2P ops are treated as independent collective ops, and are thus serialized with all other ops on this ProcessGroup, including other P2P ops. To avoid serialization, either create additional independent ProcessGroups for the P2P ops or use batched P2P ops. You can squash this warning by setting the environment variable TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING to false. (function operator())
[Rank 1]: Created driver
[rank1]:[W701 09:33:37.179345078 ProcessGroupNCCL.cpp:4025] Warning: [PG ID 0 PG GUID 0(default_pg) Rank 1] An unbatched P2P op (send/recv) was called on this ProcessGroup with size 3.  In eager initialization mode, unbatched P2P ops are treated as independent collective ops, and are thus serialized with all other ops on this ProcessGroup, including other P2P ops. To avoid serialization, either create additional independent ProcessGroups for the P2P ops or use batched P2P ops. You can squash this warning by setting the environment variable TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING to false. (function operator())
[rank1]: Traceback (most recent call last):
[rank1]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 355, in <module>
[rank1]:     test_sdpa_distributed_vs_single_on_prefill(4, 2, 512, torch.float16, 3)
[rank1]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 253, in test_sdpa_distributed_vs_single_on_prefill
[rank1]:     fabric.launch(
[rank1]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1010, in launch
[rank1]:     return self._wrap_and_launch(function, self, *args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1120, in _wrap_and_launch
[rank1]:     return launcher.launch(to_run, *args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/strategies/launchers/subprocess_script.py", line 108, in launch
[rank1]:     return function(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1126, in _wrap_with_setup
[rank1]:     return to_run(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 316, in run_sdpa_distributed_vs_single_on_prefill
[rank1]:     outputs = driver(
[rank1]:               ^^^^^^^
[rank1]:   File "/home/ubuntu/git/keys_values/keys_values/kvcache/parallel/ring_attention.py", line 165, in __call__
[rank1]:     reqs.append(dist.isend(tensor=buff_keys.read(), dst=rank_send))
[rank1]:                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 2491, in isend
[rank1]:     return group.send([tensor], group_dst, tag)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]: RuntimeError: ncclComm != nullptr INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp":4043, please report a bug to PyTorch. Parent communicator missing in eager initialization mode.
[Rank 2]: Created driver
[rank2]:[W701 09:33:37.189965554 ProcessGroupNCCL.cpp:4025] Warning: [PG ID 0 PG GUID 0(default_pg) Rank 2] An unbatched P2P op (send/recv) was called on this ProcessGroup with size 3.  In eager initialization mode, unbatched P2P ops are treated as independent collective ops, and are thus serialized with all other ops on this ProcessGroup, including other P2P ops. To avoid serialization, either create additional independent ProcessGroups for the P2P ops or use batched P2P ops. You can squash this warning by setting the environment variable TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING to false. (function operator())
[rank2]: Traceback (most recent call last):
[rank2]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 355, in <module>
[rank2]:     test_sdpa_distributed_vs_single_on_prefill(4, 2, 512, torch.float16, 3)
[rank2]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 253, in test_sdpa_distributed_vs_single_on_prefill
[rank2]:     fabric.launch(
[rank2]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1010, in launch
[rank2]:     return self._wrap_and_launch(function, self, *args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1120, in _wrap_and_launch
[rank2]:     return launcher.launch(to_run, *args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/strategies/launchers/subprocess_script.py", line 108, in launch
[rank2]:     return function(*args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1126, in _wrap_with_setup
[rank2]:     return to_run(*args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 316, in run_sdpa_distributed_vs_single_on_prefill
[rank2]:     outputs = driver(
[rank2]:               ^^^^^^^
[rank2]:   File "/home/ubuntu/git/keys_values/keys_values/kvcache/parallel/ring_attention.py", line 165, in __call__
[rank2]:     reqs.append(dist.isend(tensor=buff_keys.read(), dst=rank_send))
[rank2]:                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 2491, in isend
[rank2]:     return group.send([tensor], group_dst, tag)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]: RuntimeError: ncclComm != nullptr INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp":4043, please report a bug to PyTorch. Parent communicator missing in eager initialization mode.
```

* Something else must be wrong
* Append the summary of your findings to `ai_dev/debug_ring_attention_distributed_test.md`

## Second round findings

The stream-wrapper removal was a correct but insufficient fix. The second run confirms this:
the error still occurs at bare `dist.isend` at line 165, with the same "parent communicator
missing" assert. The PyTorch warning printed before the crash contains the actual answer:

> *"To avoid serialization, either create additional independent ProcessGroups for the P2P
> ops or **use batched P2P ops**."*

### Root cause (revised)

In NCCL eager initialization mode, **individual `isend`/`irecv` are not supported for
process groups with more than 2 ranks**. NCCL requires that P2P operations on a group of
size N > 2 go through `dist.batch_isend_irecv`. That function creates a properly-scoped
P2P sub-communicator derived from the parent collective communicator. Calling standalone
`isend`/`irecv` skips this derivation, so NCCL has no communicator to use and asserts.

The stream wrapper issue was real (calling NCCL ops from a non-default stream context is
wrong), but even after fixing that, the underlying use of individual `isend`/`irecv` still
fails for the same fundamental reason.

### Fix applied

Replaced the four individual `isend`/`irecv` calls with a single `dist.batch_isend_irecv`
call using `dist.P2POp` descriptors:

```python
reqs = dist.batch_isend_irecv([
    dist.P2POp(dist.isend, buff_keys.read(),    rank_send),
    dist.P2POp(dist.isend, buff_values.read(),  rank_send),
    dist.P2POp(dist.irecv, buff_keys.write(),   rank_recv),
    dist.P2POp(dist.irecv, buff_values.write(), rank_recv),
])
```

`batch_isend_irecv` returns a list of work handles, so `for req in reqs: req.wait()`
continues to work unchanged.

## Response 2

Running this again gives:

```bash
You are using a CUDA device ('NVIDIA A100-SXM4-40GB') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/3
Initializing distributed: GLOBAL_RANK: 2, MEMBER: 3/3
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/3
[Rank 1]: Broadcasting data
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 3 processes
----------------------------------------------------------------------------------------------------

[Rank 2]: Broadcasting data
[Rank 0]: Sampling data
[Rank 0]: Broadcasting data
[Rank 0]: Created driver
[Rank 1]: Created driver
[Rank 2]: Created driver
[rank2]: Traceback (most recent call last):
[rank2]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 355, in <module>
[rank2]:     test_sdpa_distributed_vs_single_on_prefill(4, 2, 512, torch.float16, 3)
[rank2]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 253, in test_sdpa_distributed_vs_single_on_prefill
[rank2]:     fabric.launch(
[rank2]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1010, in launch
[rank2]:     return self._wrap_and_launch(function, self, *args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1120, in _wrap_and_launch
[rank2]:     return launcher.launch(to_run, *args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/strategies/launchers/subprocess_script.py", line 108, in launch
[rank2]:     return function(*args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1126, in _wrap_with_setup
[rank2]:     return to_run(*args, **kwargs)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 316, in run_sdpa_distributed_vs_single_on_prefill
[rank2]:     outputs = driver(
[rank2]:               ^^^^^^^
[rank2]:   File "/home/ubuntu/git/keys_values/keys_values/kvcache/parallel/ring_attention.py", line 165, in __call__
[rank2]:     reqs = dist.batch_isend_irecv([
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 2842, in batch_isend_irecv
[rank2]:     p2p_op.op(
[rank2]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 2491, in isend
[rank2]:     return group.send([tensor], group_dst, tag)
[rank2]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]: RuntimeError: ncclComm != nullptr INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp":4043, please report a bug to PyTorch. Parent communicator missing in eager initialization mode.
[rank1]: Traceback (most recent call last):
[rank1]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 355, in <module>
[rank1]:     test_sdpa_distributed_vs_single_on_prefill(4, 2, 512, torch.float16, 3)
[rank1]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 253, in test_sdpa_distributed_vs_single_on_prefill
[rank1]:     fabric.launch(
[rank1]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1010, in launch
[rank1]:     return self._wrap_and_launch(function, self, *args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1120, in _wrap_and_launch
[rank1]:     return launcher.launch(to_run, *args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/strategies/launchers/subprocess_script.py", line 108, in launch
[rank1]:     return function(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/lightning/fabric/fabric.py", line 1126, in _wrap_with_setup
[rank1]:     return to_run(*args, **kwargs)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ubuntu/git/keys_values/test/kvcache/test_ring_attention_distributed.py", line 316, in run_sdpa_distributed_vs_single_on_prefill
[rank1]:     outputs = driver(
[rank1]:               ^^^^^^^
[rank1]:   File "/home/ubuntu/git/keys_values/keys_values/kvcache/parallel/ring_attention.py", line 165, in __call__
[rank1]:     reqs = dist.batch_isend_irecv([
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 2842, in batch_isend_irecv
[rank1]:     p2p_op.op(
[rank1]:   File "/home/ubuntu/virtenvs/keys_values/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py", line 2491, in isend
[rank1]:     return group.send([tensor], group_dst, tag)
[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]: RuntimeError: ncclComm != nullptr INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp":4043, please report a bug to PyTorch. Parent communicator missing in eager initialization mode.
[rank: 1] Child process with PID 2053468 terminated with code 1. Forcefully terminating all other processes to avoid zombies 🧟
Killed
```

* Something else must be wrong.
* Append the summary of your findings to `ai_dev/debug_ring_attention_distributed_test.md`

## Third round findings

### Observation

`batch_isend_irecv` did not help: the trace now shows it calling `p2p_op.op()` → `isend`
→ `group.send` and crashing identically. This PyTorch version's `batch_isend_irecv` simply
iterates over the `P2POp` list and calls each one individually — it is not special-cased
to use a different NCCL path.

Critically, **rank 0 never crashes** — it is only killed as collateral when the subprocess
manager tears down after ranks 1 and 2 fail. The crash is therefore asymmetric: rank 0
works, ranks 1 and 2 do not.

### Root cause (final)

The asymmetry is caused by **device mismatch at process group creation time**.

In NCCL eager initialization mode, PyTorch creates the NCCL communicator the first time
the process group is *used* for a collective (here: during `fabric.broadcast`). The
communicator is registered under the CUDA device that was **current on that rank at that
moment**. For Lightning with the default `"auto"` strategy and a subprocess launcher,
`cuda:0` is the initial default device on every subprocess. Lightning sets each rank's
correct device (`cuda:rank`) only when the rank function body starts executing — but by the
time `fabric.broadcast` fires, it is possible that NCCL registers the parent communicator
under `cuda:0` for all ranks.

When ranks 1 and 2 later try to issue a P2P op on tensors living on `cuda:1`/`cuda:2`,
NCCL looks up the parent communicator by device, finds only the one registered under
`cuda:0`, and asserts "parent communicator missing."

Rank 0 is unaffected because its device happens to match the one the communicator was
registered under.

### Fix applied

Create a **dedicated `dist.new_group()` for P2P ops** inside `RingAttentionDriver.__init__`,
which is called *after* Lightning has already set the rank's device. `dist.new_group` calls
`ncclCommInitRank` (not `ncclCommSplit`), so it initialises a fresh communicator from
scratch against the *current* device at call time — the correct `cuda:rank` device.

```python
self._p2p_group = dist.new_group()
```

All four `P2POp` descriptors are then passed this group:

```python
reqs = dist.batch_isend_irecv([
    dist.P2POp(dist.isend, buff_keys.read(),    rank_send, self._p2p_group),
    dist.P2POp(dist.isend, buff_values.read(),  rank_send, self._p2p_group),
    dist.P2POp(dist.irecv, buff_keys.write(),   rank_recv, self._p2p_group),
    dist.P2POp(dist.irecv, buff_values.write(), rank_recv, self._p2p_group),
])
```
