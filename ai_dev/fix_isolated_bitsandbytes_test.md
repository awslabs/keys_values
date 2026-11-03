# Fix isolated bitsandbytes quantization test

## Prompt summary

Find a real, unassigned issue in an AWS machine-learning repository involving
C++ or CUDA, confirm that no pull request already addresses it, implement the
smallest reliable fix, and validate it before proposing a pull request.

For issue #173, reproduce or diagnose why
`test_explore_bitsandbytes` passes as part of `test_quantization.py` but fails
when selected in isolation. Stop before pushing so the changes can be reviewed.

## AI-assisted analysis

The test declares `cuda:0`, but its call to `random_tensor` did not pass that
device. Git history showed that `KVCacheParams.device` was removed during a
device/dtype refactor and `random_tensor` gained an explicit `device` argument;
this test call was not migrated. Consequently, the CUDA-only test exercised a
CPU bitsandbytes path and depended on quantization state initialized by earlier
tests.

The fix passes the declared CUDA device to `random_tensor` and asserts the
resulting tensor's placement. This makes the test exercise its intended GPU
path and removes its dependency on test execution order.
