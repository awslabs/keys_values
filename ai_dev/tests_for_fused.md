# Writing unit tests for `keys_values/fused` operators with Claude Code

## Initial prompt

Please write comparison unit tests for a number of fused operators:

- `keys_values.fused.fused_rmsnorm.fused_rmsnorm`
- `keys_values.fused.fused_rope.fused_apply_rope`
- `keys_values.fused.fused_swiglu.fused_swiglu`

Each represents a PyTorch expression, and the test should compare the output of
the fused operator with the expression.

- I need tests for `forward` (output of operator) and `backward` (gradients
  with respect to inputs). For `backward`, create a scalar function as random
  linear combination of the outputs and compare gradients of that.
- The tests should iterate over a number of different input shapes
- Please write the tests in the style of tests in `tests`.
  Good examples are `tests.attention.test_flex_attention.test_comparison` or
  all tests in `tests.kvcache.test_sdpa_op`.

Put the new tests into the module `tests.fused`.

I have created a new branch `add_fused_tests`. Please work in this branch.

Please use the virtual environment `~/virtenvs/keysvals` for running Python
code.

## Comment

You implemented the baseline comparisons as functions `_rmsnorm_eager`,
`_rope_eager`, `_swiglu_eager`. This is OK, but I'd also like to compare against
the existing code that is used if the fused operators are not active. This would
be:

- `keys_values.model.RMSNorm`
- `litgpt.model.apply_rope`
- `keys_values.lora.LlaMAMLP` and `litgpt.model.LlaMAMLP`

Can you add these to the comparisons?

### Claude Code answer

What was added to each test:

* test_fused_rmsnorm_forward/backward — now also compares against kv_model.RMSNorm and litgpt.model.RMSNorm module instances (weight is copied in so all three see identical parameters). Both forward and backward loop over all three references.

* test_fused_apply_rope_forward/backward (3-D and 4-D) — now also compares against litgpt.model.apply_rope. A small helper _rope_via_litgpt reshapes the (T, D) cos/sin to the (1, T, D) form that apply_rope requires.

* test_fused_swiglu_forward/backward — unchanged structurally (the fused_swiglu op is a pure activation with no module state to compare against).

* Two new test functions test_fused_swiglu_llamamlp_forward and test_fused_swiglu_llamamlp_backward — run a real litgpt.model.LLaMAMLP and kv_lora.LLaMAMLP forward pass with set_fused_swiglu_enabled(True) and compare the output/gradients against the same module with the fused kernel off. This is the end-to-end module comparison you asked for. The flag is always restored in a finally block to avoid test pollution.

## Comment

I ran the tests on a GPU instance and obtained many failures.

Let us start with `test_fused_apply_rope_forward`.

I ran:
```bash
pytest test/fused/test_fused_ops.py -k test_fused_apply_rope_forward
```

Two configurations failed:
```text
_________________________________________________ test_fused_apply_rope_forward[8-32-128-dtype5] __________________________________________________

BH = 8, T = 32, D = 128, dtype = torch.bfloat16

    @_RunIf(min_cuda_gpus=1)
    @pytest.mark.parametrize("BH, T, D, dtype", _ROPE_PARAMS_3D)
    def test_fused_apply_rope_forward(BH, T, D, dtype):
        seed = 31415927
        torch.manual_seed(seed)
        device = torch.device("cuda", 0)

        x = torch.randn(BH, T, D, device=device, dtype=dtype)
        cos = torch.randn(T, D, device=device, dtype=dtype)
        sin = torch.randn(T, D, device=device, dtype=dtype)

        out_fused = fused_apply_rope(x, cos, sin)

        atol, rtol = _tolerances(dtype)
        references = {
            "eager": _rope_eager(x, cos, sin),
            "litgpt.apply_rope": _rope_via_litgpt(x, cos, sin),
        }
        for name, out_ref in references.items():
            print(f"rope fwd 3D: BH={BH}, T={T}, D={D}, dtype={dtype}, ref={name}")
>           torch.testing.assert_close(out_fused, out_ref, atol=atol, rtol=rtol)
E           AssertionError: Tensor-likes are not close!
E
E           Mismatched elements: 2 / 32768 (0.0%)
E           Greatest absolute difference: 0.0146484375 at index (2, 4, 125) (up to 0.01 allowed)
E           Greatest relative difference: 0.1953125 at index (6, 4, 68) (up to 0.01 allowed)

test/fused/test_fused_ops.py:259: AssertionError
-------------------------------------------------------------- Captured stdout call ---------------------------------------------------------------
rope fwd 3D: BH=8, T=32, D=128, dtype=torch.bfloat16, ref=eager
rope fwd 3D: BH=8, T=32, D=128, dtype=torch.bfloat16, ref=litgpt.apply_rope
_________________________________________________ test_fused_apply_rope_forward[2-64-256-dtype8] __________________________________________________

BH = 2, T = 64, D = 256, dtype = torch.bfloat16

    @_RunIf(min_cuda_gpus=1)
    @pytest.mark.parametrize("BH, T, D, dtype", _ROPE_PARAMS_3D)
    def test_fused_apply_rope_forward(BH, T, D, dtype):
        seed = 31415927
        torch.manual_seed(seed)
        device = torch.device("cuda", 0)

        x = torch.randn(BH, T, D, device=device, dtype=dtype)
        cos = torch.randn(T, D, device=device, dtype=dtype)
        sin = torch.randn(T, D, device=device, dtype=dtype)

        out_fused = fused_apply_rope(x, cos, sin)

        atol, rtol = _tolerances(dtype)
        references = {
            "eager": _rope_eager(x, cos, sin),
            "litgpt.apply_rope": _rope_via_litgpt(x, cos, sin),
        }
        for name, out_ref in references.items():
            print(f"rope fwd 3D: BH={BH}, T={T}, D={D}, dtype={dtype}, ref={name}")
>           torch.testing.assert_close(out_fused, out_ref, atol=atol, rtol=rtol)
E           AssertionError: Tensor-likes are not close!
E
E           Mismatched elements: 2 / 32768 (0.0%)
E           Greatest absolute difference: 0.015625 at index (0, 36, 247) (up to 0.01 allowed)
E           Greatest relative difference: 0.05419921875 at index (1, 6, 144) (up to 0.01 allowed)

test/fused/test_fused_ops.py:259: AssertionError
-------------------------------------------------------------- Captured stdout call ---------------------------------------------------------------
rope fwd 3D: BH=2, T=64, D=256, dtype=torch.bfloat16, ref=eager
rope fwd 3D: BH=2, T=64, D=256, dtype=torch.bfloat16, ref=litgpt.apply_rope
```

One issue I noted in your code: `sin` and `cos` are supposed to be sines and
cosines, but you sample them independently using `torch.randn`. You can check
`litgpt.model.build_rope_cache` how they are created.

Can you please fix the tests? Do not simply reduce the required tolerance. Absolute
differences of 0.015 are too large.

### Claude Code answer:

The root cause was that the Triton kernel and the eager reference both compute in float32 internally, but with random independent cos/sin (which can be large, e.g. O(1) from randn), the intermediate products x * cos and rot_x * sin can be large enough that adding them produces bf16 rounding differences of ~0.015. Since real RoPE always satisfies cos² + sin² = 1, the values are bounded in [-1, 1], keeping all intermediate products close to the magnitude of x and eliminating the cancellation. The fix replaces torch.randn cos/sin with proper unit-circle values built from random angles.

## Comment

Next I ran:
```bash
pytest test/fused/test_fused_ops.py -k test_fused_apply_rope_backward
```

All configurations fail. For example:
```text
_________________________________________________ test_fused_apply_rope_backward[4-33-64-dtype18] _________________________________________________

BH = 4, T = 33, D = 64, dtype = torch.float32

    @_RunIf(min_cuda_gpus=1)
    @pytest.mark.parametrize("BH, T, D, dtype", _ROPE_PARAMS_3D)
    def test_fused_apply_rope_backward(BH, T, D, dtype):
        seed = 31415927
        torch.manual_seed(seed)
        device = torch.device("cuda", 0)

        _x = torch.randn(BH, T, D, device=device, dtype=dtype)
        cos, sin = _build_cos_sin(T, D, device)
        cos, sin = cos.to(dtype), sin.to(dtype)

        def _run(fn):
            x = _copy_with_grad(_x)
            _random_linear_loss(fn(x)).backward()
            return x.grad

        grad_fused = _run(lambda x: fused_apply_rope(x, cos, sin))
        references = {
            "eager": _run(lambda x: _rope_eager(x, cos, sin)),
            "litgpt.apply_rope": _run(lambda x: _rope_via_litgpt(x, cos, sin)),
        }

        print(f"rope bwd 3D: BH={BH}, T={T}, D={D}, dtype={dtype}")
        atol, rtol = _tolerances(dtype)
        for name, grad_ref in references.items():
>           torch.testing.assert_close(grad_fused, grad_ref, atol=atol, rtol=rtol)
E           AssertionError: Tensor-likes are not close!
E
E           Mismatched elements: 8448 / 8448 (100.0%)
E           Greatest absolute difference: 6.446865081787109 at index (3, 8, 20) (up to 0.0001 allowed)
E           Greatest relative difference: 17691.923828125 at index (1, 23, 15) (up to 0.0001 allowed)

test/fused/test_fused_ops.py:300: AssertionError
-------------------------------------------------------------- Captured stdout call ---------------------------------------------------------------
rope bwd 3D: BH=4, T=33, D=64, dtype=torch.float32
```

I think the problem may be you call `_run` twice, and different linear combinations
are used, because `_random_linear_loss` is called twice. I think you need to fix
the random combination and use the same. Can you try and fix this?

Note that also all other "backward" tests fail. This could be the same reason.
Could you rework all of them?

### Answer of Claude Code

Good — no remaining _random_linear_loss calls (the grep exit code 1 means no matches found), and all 210 tests still collect.

The fix applied uniformly across all backward tests: instead of calling torch.randn_like(y) inside the loss function on each invocation (which produces a different random tensor every time), each test now generates one coeff = torch.randn(output_shape, device=device) before any forward/backward run, and passes it explicitly to _linear_loss. Every run in the loop then differentiates exactly the same scalar function, making the gradient comparison meaningful.

