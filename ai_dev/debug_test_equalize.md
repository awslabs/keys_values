# Write and debug tests in `test_equalize.py`

Consider the test `test_communication_plan` in `test/kvcache/test_equalize.py`.
Running this test, it fails in several instances:
```python
test/kvcache/test_equalize.py .F.F.F                                                                                                        [100%]

==================================================================== FAILURES =====================================================================
____________________________________________________ test_communication_plan[4-2-64-128-517-4] ____________________________________________________

batch_size = 4, n_query_groups = 2, q_len = 64, cache_length = 128, input_pos = 517, num_devices = 4

    @pytest.mark.parametrize(
        "batch_size, n_query_groups, q_len, cache_length, input_pos, num_devices",
        [
            (2, 4, 8, 256, 256 * 2 + 11, 2),
            (4, 2, 64, 128, 128 * 4 + 5, 4),
            (3, 4, 16, 512, 512 * 8 + 127, 8),
            (5, 8, 13, 256, 256 * 3 + 15, 3),
            (1, 4, 21, 256, 256 * 5 + 15, 5),
            (4, 2, 15, 256, 256 * 7 + 15, 7),
        ],
    )
    def test_communication_plan(batch_size, n_query_groups, q_len, cache_length, input_pos, num_devices):
        active_dimensions = (0, 1)
        device = torch.device("cpu")

        essentially_1d = active_dimensions == ()
        if essentially_1d:
            raise NotImplementedError("TODO!")
        overwrite_pos = _sample_overwrite_pos(
            batch_size,
            n_query_groups,
            q_len,
            cache_length,
            num_devices,
            active_dimensions,
            device,
        )
        comm_plan = _get_communication_plan(
            num_devices=num_devices,
            input_pos=input_pos,
            overwrite_pos=overwrite_pos,
            active_dimensions=active_dimensions,
        )

        # Test correctness of plan
        # (1) Does the plan really equalize?
        delta_per_rank = _get_delta_per_rank(
            num_devices=num_devices,
            q_len=q_len,
            input_pos=input_pos,
            overwrite_pos=overwrite_pos,
            essentially_1d=essentially_1d,
        )
        print("delta_per_rank:")
        print(
            "\n".join(
                f"{(b, h)}: {delta_per_rank[b, h, :].tolist()}"
                for b in range(batch_size) for h in range(n_query_groups)
            )
        )
        print("\ncommunication_plan:")
        print("\n".join(f"{k}: {v.numpy()}" for k, v in comm_plan.items()))
        for (src_rank, trg_rank), plan in comm_plan.items():
            for row in plan:
                b, h, num = tuple(int(x) for x in row)
                assert num > 0, (src_rank, trg_rank, b, h, num)
                delta_per_rank[b, h, src_rank] += num
                delta_per_rank[b, h, trg_rank] -= num
>       assert torch.all(delta_per_rank == 0).item()
E       assert False
E        +  where False = <built-in method item of Tensor object at 0x148d0ed00>()
E        +    where <built-in method item of Tensor object at 0x148d0ed00> = tensor(False).item
E        +      where tensor(False) = <built-in method all of type object at 0x105a8fad0>(tensor([[[ 0,  0,  0,  0],\n         [ 0,  0,  0,  0]],\n\n        [[ 0,  0,  0,  0],\n         [ 0,  0,  0,  0]],\n\n        [[ 0,  0,  0,  0],\n         [ 0,  0,  0,  0]],\n\n        [[ 0,  0,  0,  0],\n         [ 8,  0, -8,  0]]]) == 0)
E        +        where <built-in method all of type object at 0x105a8fad0> = torch.all

test/kvcache/test_equalize.py:126: AssertionError
-------------------------------------------------------------- Captured stdout call ---------------------------------------------------------------
delta_per_rank:
(0, 0): [-9, 9, 5, -5]
(0, 1): [0, 4, -4, 0]
(1, 0): [0, -2, -4, 6]
(1, 1): [-4, 1, 4, -1]
(2, 0): [0, -1, 1, 0]
(2, 1): [0, -5, -3, 8]
(3, 0): [-2, -1, 5, -2]
(3, 1): [4, 4, -5, -3]

communication_plan:
(0, 1): [[0 0 9]]
(0, 2): [[1 1 4]
 [3 0 2]
 [3 1 4]]
(1, 2): [[2 0 1]
 [3 0 1]]
(1, 3): [[1 0 2]
 [2 1 5]]
(2, 1): [[0 1 4]
 [3 1 1]]
(2, 3): [[1 0 4]
 [2 1 3]]
(3, 1): [[1 1 1]
 [3 1 3]]
(3, 2): [[0 0 5]
 [3 0 2]]
____________________________________________________ test_communication_plan[5-8-13-256-783-3] ____________________________________________________

batch_size = 5, n_query_groups = 8, q_len = 13, cache_length = 256, input_pos = 783, num_devices = 3

    @pytest.mark.parametrize(
        "batch_size, n_query_groups, q_len, cache_length, input_pos, num_devices",
        [
            (2, 4, 8, 256, 256 * 2 + 11, 2),
            (4, 2, 64, 128, 128 * 4 + 5, 4),
            (3, 4, 16, 512, 512 * 8 + 127, 8),
            (5, 8, 13, 256, 256 * 3 + 15, 3),
            (1, 4, 21, 256, 256 * 5 + 15, 5),
            (4, 2, 15, 256, 256 * 7 + 15, 7),
        ],
    )
    def test_communication_plan(batch_size, n_query_groups, q_len, cache_length, input_pos, num_devices):
        active_dimensions = (0, 1)
        device = torch.device("cpu")

        essentially_1d = active_dimensions == ()
        if essentially_1d:
            raise NotImplementedError("TODO!")
        overwrite_pos = _sample_overwrite_pos(
            batch_size,
            n_query_groups,
            q_len,
            cache_length,
            num_devices,
            active_dimensions,
            device,
        )
        comm_plan = _get_communication_plan(
            num_devices=num_devices,
            input_pos=input_pos,
            overwrite_pos=overwrite_pos,
            active_dimensions=active_dimensions,
        )

        # Test correctness of plan
        # (1) Does the plan really equalize?
        delta_per_rank = _get_delta_per_rank(
            num_devices=num_devices,
            q_len=q_len,
            input_pos=input_pos,
            overwrite_pos=overwrite_pos,
            essentially_1d=essentially_1d,
        )
        print("delta_per_rank:")
        print(
            "\n".join(
                f"{(b, h)}: {delta_per_rank[b, h, :].tolist()}"
                for b in range(batch_size) for h in range(n_query_groups)
            )
        )
        print("\ncommunication_plan:")
        print("\n".join(f"{k}: {v.numpy()}" for k, v in comm_plan.items()))
        for (src_rank, trg_rank), plan in comm_plan.items():
            for row in plan:
                b, h, num = tuple(int(x) for x in row)
                assert num > 0, (src_rank, trg_rank, b, h, num)
                delta_per_rank[b, h, src_rank] += num
                delta_per_rank[b, h, trg_rank] -= num
>       assert torch.all(delta_per_rank == 0).item()
E       assert False
E        +  where False = <built-in method item of Tensor object at 0x13fe4e580>()
E        +    where <built-in method item of Tensor object at 0x13fe4e580> = tensor(False).item
E        +      where tensor(False) = <built-in method all of type object at 0x105a8fad0>(tensor([[[ 0,  0,  0],\n         [ 0,  0,  0],\n         [ 0,  0,  0],\n         [ 0,  0,  0],\n         [ 0,  0,  0],\n   ...],\n         [ 0,  0,  0],\n         [ 0,  0,  0],\n         [-4,  0,  4],\n         [ 0,  0,  0],\n         [ 0,  0,  0]]]) == 0)
E        +        where <built-in method all of type object at 0x105a8fad0> = torch.all

test/kvcache/test_equalize.py:126: AssertionError
-------------------------------------------------------------- Captured stdout call ---------------------------------------------------------------
delta_per_rank:
(0, 0): [-3, 0, 3]
(0, 1): [3, -1, -2]
(0, 2): [0, 1, -1]
(0, 3): [0, -1, 1]
(0, 4): [-2, 0, 2]
(0, 5): [0, 1, -1]
(0, 6): [-2, 0, 2]
(0, 7): [-4, 6, -2]
(1, 0): [1, -1, 0]
(1, 1): [-2, 1, 1]
(1, 2): [0, 1, -1]
(1, 3): [-2, -2, 4]
(1, 4): [-1, -1, 2]
(1, 5): [-3, 1, 2]
(1, 6): [1, -1, 0]
(1, 7): [2, -1, -1]
(2, 0): [0, -2, 2]
(2, 1): [1, -1, 0]
(2, 2): [1, 1, -2]
(2, 3): [-3, 2, 1]
(2, 4): [-1, 3, -2]
(2, 5): [1, -2, 1]
(2, 6): [3, -2, -1]
(2, 7): [-2, 0, 2]
(3, 0): [0, 1, -1]
(3, 1): [2, -2, 0]
(3, 2): [1, 1, -2]
(3, 3): [-1, -2, 3]
(3, 4): [-2, 0, 2]
(3, 5): [0, 0, 0]
(3, 6): [-3, 2, 1]
(3, 7): [0, -3, 3]
(4, 0): [-2, 0, 2]
(4, 1): [-1, 0, 1]
(4, 2): [0, 0, 0]
(4, 3): [-3, 3, 0]
(4, 4): [-1, -1, 2]
(4, 5): [-3, 1, 2]
(4, 6): [0, 0, 0]
(4, 7): [0, -1, 1]

communication_plan:
(0, 1): [[0 7 4]
 [1 5 1]
 [2 4 1]
 [2 5 1]
 [4 3 3]
 [4 5 1]]
(0, 2): [[0 0 3]
 [0 4 2]
 [0 6 2]
 [1 1 1]
 [1 3 2]
 [1 4 1]
 [2 2 1]
 [2 3 1]
 [2 7 2]
 [3 2 1]
 [3 3 1]
 [3 4 2]
 [3 6 1]
 [4 0 2]
 [4 1 1]
 [4 4 1]]
(1, 0): [[0 1 1]
 [1 0 1]
 [1 1 1]
 [1 6 1]
 [1 7 1]
 [2 1 1]
 [2 3 2]
 [2 6 2]
 [3 1 2]
 [3 6 2]]
(1, 2): [[0 3 1]
 [1 3 2]
 [1 4 1]
 [2 0 2]
 [2 5 1]
 [3 3 2]
 [3 7 3]
 [4 4 1]
 [4 7 1]]
(2, 0): [[0 1 2]
 [1 5 2]
 [1 7 1]
 [2 6 1]
 [4 5 2]]
(2, 1): [[0 2 1]
 [0 5 1]
 [0 7 2]
 [1 2 1]
 [2 2 1]
 [2 4 2]
 [3 0 1]
 [3 2 1]]
___________________________________________________ test_communication_plan[4-2-15-256-1807-7] ____________________________________________________

batch_size = 4, n_query_groups = 2, q_len = 15, cache_length = 256, input_pos = 1807, num_devices = 7

    @pytest.mark.parametrize(
        "batch_size, n_query_groups, q_len, cache_length, input_pos, num_devices",
        [
            (2, 4, 8, 256, 256 * 2 + 11, 2),
            (4, 2, 64, 128, 128 * 4 + 5, 4),
            (3, 4, 16, 512, 512 * 8 + 127, 8),
            (5, 8, 13, 256, 256 * 3 + 15, 3),
            (1, 4, 21, 256, 256 * 5 + 15, 5),
            (4, 2, 15, 256, 256 * 7 + 15, 7),
        ],
    )
    def test_communication_plan(batch_size, n_query_groups, q_len, cache_length, input_pos, num_devices):
        active_dimensions = (0, 1)
        device = torch.device("cpu")

        essentially_1d = active_dimensions == ()
        if essentially_1d:
            raise NotImplementedError("TODO!")
        overwrite_pos = _sample_overwrite_pos(
            batch_size,
            n_query_groups,
            q_len,
            cache_length,
            num_devices,
            active_dimensions,
            device,
        )
        comm_plan = _get_communication_plan(
            num_devices=num_devices,
            input_pos=input_pos,
            overwrite_pos=overwrite_pos,
            active_dimensions=active_dimensions,
        )

        # Test correctness of plan
        # (1) Does the plan really equalize?
        delta_per_rank = _get_delta_per_rank(
            num_devices=num_devices,
            q_len=q_len,
            input_pos=input_pos,
            overwrite_pos=overwrite_pos,
            essentially_1d=essentially_1d,
        )
        print("delta_per_rank:")
        print(
            "\n".join(
                f"{(b, h)}: {delta_per_rank[b, h, :].tolist()}"
                for b in range(batch_size) for h in range(n_query_groups)
            )
        )
        print("\ncommunication_plan:")
        print("\n".join(f"{k}: {v.numpy()}" for k, v in comm_plan.items()))
        for (src_rank, trg_rank), plan in comm_plan.items():
            for row in plan:
                b, h, num = tuple(int(x) for x in row)
                assert num > 0, (src_rank, trg_rank, b, h, num)
                delta_per_rank[b, h, src_rank] += num
                delta_per_rank[b, h, trg_rank] -= num
>       assert torch.all(delta_per_rank == 0).item()
E       assert False
E        +  where False = <built-in method item of Tensor object at 0x148cfe9e0>()
E        +    where <built-in method item of Tensor object at 0x148cfe9e0> = tensor(False).item
E        +      where tensor(False) = <built-in method all of type object at 0x105a8fad0>(tensor([[[ 0, -2,  2,  0,  0,  0,  0],\n         [ 0,  0,  0,  0,  0,  0,  0]],\n\n        [[ 0,  0,  0,  0,  0,  0,  0],...        [ 0,  0,  0,  0,  0,  0,  0]],\n\n        [[ 0,  0,  0,  0,  0,  0,  0],\n         [ 0,  0,  0,  0,  0,  0,  0]]]) == 0)
E        +        where <built-in method all of type object at 0x105a8fad0> = torch.all

test/kvcache/test_equalize.py:126: AssertionError
-------------------------------------------------------------- Captured stdout call ---------------------------------------------------------------
delta_per_rank:
(0, 0): [0, -2, 1, 1, -1, 1, 0]
(0, 1): [1, -1, 0, 1, 0, -1, 0]
(1, 0): [0, -1, -2, 1, 2, 0, 0]
(1, 1): [0, 0, 1, 1, 1, -2, -1]
(2, 0): [1, -2, 2, -1, 0, 0, 0]
(2, 1): [1, -1, -1, -1, 2, -2, 2]
(3, 0): [1, -1, -1, 0, 1, 0, 0]
(3, 1): [3, 0, 2, -1, -1, -2, -1]

communication_plan:
(1, 0): [[0 1 1]
 [3 0 1]]
(1, 2): [[2 0 2]]
(1, 3): [[0 0 1]
 [1 0 1]]
(1, 6): [[2 1 1]]
(2, 0): [[2 1 1]]
(2, 1): [[0 0 1]]
(2, 4): [[1 0 2]
 [3 0 1]]
(2, 5): [[1 1 1]]
(3, 0): [[2 0 1]]
(3, 2): [[3 1 1]]
(3, 6): [[2 1 1]]
(4, 0): [[3 1 1]]
(4, 5): [[0 0 1]]
(5, 0): [[3 1 2]]
(5, 3): [[0 1 1]
 [1 1 1]]
(5, 4): [[2 1 2]]
(6, 2): [[3 1 1]]
(6, 4): [[1 1 1]]
============================================================= short test summary info =============================================================
FAILED test/kvcache/test_equalize.py::test_communication_plan[4-2-64-128-517-4] - assert False
FAILED test/kvcache/test_equalize.py::test_communication_plan[5-8-13-256-783-3] - assert False
FAILED test/kvcache/test_equalize.py::test_communication_plan[4-2-15-256-1807-7] - assert False
```

* Please analyze this and tell me what is wrong
* Append the summary of your findings to `ai_dev/debug_test_equalize.md`

---

## Root Cause Analysis (2026-07-03)

### The Bug

The bug is in `_get_communication_plan` in
`keys_values/kvcache/parallel/equalize.py`, lines 127–139.

The greedy equalization loop tracks which rank has a **deficit** (too few
overwrite positions, `delta < 0`) and which has an **excess** (too many,
`delta > 0`):

```python
ranks_smallest = torch.argmin(delta_per_rank, dim=-1, keepdim=True)  # deficit
ranks_largest  = torch.argmax(delta_per_rank, dim=-1, keepdim=True)  # excess
```

It then constructs two candidate `(from, to)` rank pairs:

```python
args1 = (ranks_smallest, ranks_largest)  # from=deficit, to=excess
args2 = (ranks_largest,  ranks_smallest) # from=excess,  to=deficit
```

and selects between them with:

```python
torch.where(largest >= -smallest, args1, args2)
```

The intent appears to have been "pick whichever direction minimises
oscillation", but the condition flips the meaning of `src_rank`/`trg_rank`
arbitrarily depending on which magnitude is larger in each greedy round.

The rest of the codebase — `_get_allocations` and the test — uses a **fixed
convention**: `src_rank` = the rank that *gives away* old cache slots (the
deficit rank, which needs to free up `num` slots), and `trg_rank` = the rank
that *receives* them (the excess rank, which has more overwrite positions than
needed).

The test models this as:
```python
delta_per_rank[b, h, src_rank] += num   # deficit rank gains free slots → toward 0
delta_per_rank[b, h, trg_rank] -= num   # excess rank loses positions  → toward 0
```

When the condition is FALSE and `args2` is selected, `src_rank` becomes the
**excess** rank and `trg_rank` becomes the **deficit** rank. The test then
adds `num` to the excess entry (moves further from 0) and subtracts `num`
from the deficit entry (moves further from 0), leaving a non-zero residual.

### Concrete Example

For the failing case `[4-2-64-128-517-4]`, batch `(b=3, h=1)`:

```
delta = [4, 4, -5, -3]
```

In the first greedy round:
- `largest = 4` (rank 0 or 1), `smallest = -5` (rank 2)
- condition: `4 >= 5` → **FALSE** → picks `args2 = (rank_largest=0, rank_smallest=2)`
- plan records `(src=0, trg=2, mass=4)`, but rank 0 is the *excess* not the
  source

Test applies: `delta[0] += 4` → 8, `delta[2] -= 4` → -9 (both diverge from 0).
After processing all plan entries for `(b=3, h=1)`, the residual is
`[8, 0, -8, 0]` instead of `[0, 0, 0, 0]`.

The same flip happens for other `(b, h)` slices in the other failing cases
wherever `largest < -smallest` in any greedy round.

### Fix

Remove the conditional flip entirely. Always use `args1` so `src_rank` is
consistently the deficit rank and `trg_rank` is consistently the excess rank:

```python
# Before (buggy):
extra = torch.cat(
    (torch.where(largest >= -smallest, args1, args2), min_vals),
    dim=-1,
).unsqueeze(-1)

# After (fixed):
extra = torch.cat(
    (args1, min_vals),
    dim=-1,
).unsqueeze(-1)
```

This makes the direction of transfer unconditionally `deficit → excess` in every
round and for every `(b, h)`, which is the convention expected by
`_get_allocations` and the test.

