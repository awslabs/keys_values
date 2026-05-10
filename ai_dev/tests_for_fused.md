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
