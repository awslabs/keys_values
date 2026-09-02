# PyTorch virtual memory allocation to EFS volume

## Initial prompt

I have PyTorch code which allocates a lot of memory on CPU. Concretely, I am
allocated buffers of size 80.5 MB, but about 6000 of them. This fails. I suspect
this fills up virtual memory, because the local disk is not large (only about 100 GB
free).

I have another disk mounted at `/mnt/efs` which is much larger. How do I make
PyTorch write its virtual memory pages there?

## Answer

1. Cannot use swap file: EFS not supported 
2. Use `np.memmap` [preferred]
3. Use `torch.FloatStorage.from_file(...)`

## Question

This sounds complex. Can I use Option 3? With your suggestion, I need to make a
choice: Either use `np.memmap` and do everything low level. Or use PyTorch
allocation. Ideally, I'd like PyTorch to use its allocation, but only if the standard
swap disk space is too small, then use /mnt/efs.

## Answer

Unfortunately there's no OS-level automatic fallback — EFS simply can't be used as
swap. You have to make the decision in code.

But Option 3 is actually no more complex than a regular torch.empty. Here's a clean
wrapper that does exactly what you want — normal allocation unless there's not
enough RAM+swap:

```python
import os
import torch
import psutil

EFS_TMP = '/mnt/efs/tmp_buffers'
os.makedirs(EFS_TMP, exist_ok=True)
_buf_counter = 0

def alloc_buffer(n_bytes: int, dtype=torch.float32) -> torch.Tensor:
    global _buf_counter
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    available = vm.available + sw.free
    if available > n_bytes * 1.1:  # 10% headroom
        return torch.empty(n_bytes // torch.tensor([], dtype=dtype).element_size(), dtype=dtype)
    # fall back to EFS-backed storage
    path = os.path.join(EFS_TMP, f'buf_{_buf_counter}.bin')
    _buf_counter += 1
    n_elems = n_bytes // torch.tensor([], dtype=dtype).element_size()
    storage = torch.FloatStorage.from_file(path, shared=True, size=n_elems)
    return torch.FloatTensor(storage)
```

One caveat: Linux overcommits memory by default — `torch.empty` won't fail even
when memory is actually exhausted; the OOM killer just terminates the process
later. So the `psutil` check is the right approach (proactive, not reactive), and
try/except MemoryError won't reliably catch it.

You'll still need to clean up the EFS files when done, but that's a single
rm -rf /mnt/efs/tmp_buffers at the end.

## Question

This is nice. Based on this idea, could you write the following code? A class
`FileBasedExtraMemoryManager`. It should be created with passing the path to a temp
directory (on EFS). It should have methods:
- `allocate(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor`: Just does
  what your code above does. Returns `torch.empty` result if available, otherwise
  tensor based on `torch.FloatStorage.from_file`. The class should manage the file
  names and buffer counter. I am wondering why you use `torch.FloatTensor`, would
  `torch.tensor` work as well? I need tensors of different `dtype`.
- `cleanup()`: Removes the files. This should be like a destructor in C++

Also, I'd like a solution so that a single object of this class is created, and that 
this object is destroyed (and `cleanup` is called) once my program terminates.
Ideally, this works also if the program terminates via an exception or because of
Ctrl-C.

## Answer

There's already a `gpu_memory.py` — I'll put this alongside it as `cpu_memory.py`.

To answer your question about `torch.tensor` vs `torch.FloatTensor`:
`torch.tensor(data)` copies data from Python/numpy objects and won't accept a
storage. `torch.FloatTensor(storage)` works but is hardcoded to `float32`. The
modern general approach is `torch.empty(0, dtype=dtype).set_(storage)`, which
works for any dtype by creating an empty tensor and reassigning its backing storage.

Created: `keys_values/cpu_memory.py`.

Written to cpu_memory.py. A few notes on the design:

Automatic cleanup coverage:

* Normal exit, sys.exit(), unhandled exception → atexit fires ✓
* Ctrl-C (KeyboardInterrupt) → Python's default SIGINT handler raises an
  exception, so atexit fires ✓
* SIGTERM (e.g. EC2 instance shutdown, kill) → explicit signal handler installed,
  which calls cleanup() then re-raises the signal ✓
* SIGKILL → cannot be caught by any process; nothing can help here

Usage:

```python
from keys_values.cpu_memory import get_memory_manager

mgr = get_memory_manager('/mnt/efs/tmp_buffers')
buf = mgr.allocate((20_000_000,), dtype=torch.float32)  # 80 MB, float32
```

`torch.UntypedStorage` requires PyTorch ≥ 1.11. If you're on an older version,
let me know and I'll switch to the typed storage map.
