# Scratch Notes for LeetCode Problems

## 2958

- Maintain running histogram `hist` and `start`
- Being `start=0`: Find `hist`, `end` so that `end` is largest possible
- Method `extend(start, end, hist, elem, k) -> new_end`
- Loop over `start` until `new_end` hits end

# 1833

costs = [1,3,2,4,1], coins = 7

num_of_costs:
0: 2 *
1: 1 *
2: 1 *
3: 1
?: 0

num_bought: 4 -> returned
coins: 3

cost: 3
num: 1
cost_here: 3


## 2161

- Easy if relative orderings are ignored
- Split left and right into two ranges each:
  [left] = [left_orig | left_moved]
  [right] = [right_moved | right_orig]
- Important: After a swap:
  [left] = [left_orig | left_moved | new_left_orig]
  ==> [left_orig | new_left_orig | left_moved]
  Bubble left 1-by-1 (do this directly)
  [right] = [right_orig_new | right_moved | right_orig]
  ==> [right_moved | right_orig_new | right_orig]
- Bubble right 1-by-1 (do this directly)
- At end: Have to revert left_moved, right_moved
- Count pivot like left; sort out at end

[left_orig | left_moved | pivots | right_moved | right_orig]

Bubble left: left_done < left_pos:
- nums[left_done] = nums[left_pos]
  nums[i + 1] = nums[i], i = left_done:left_pos

Bubble right: right_done > right_pos
- nums[right_done] = nums[right_pos]
  nums[i] = nums[i + 1], i = right_pos:right_done


nums = [-3,4,3,2]
pivot = 2

nums:       [-3, 2, 3, 4]
left_pos:   2
left_done   1
right_pos:  3
right_done: 3

elem: -3


## 3629

- Need prime testing for numbers in `1, ..., 106`. Just build a boolean list
- Sounds like recursion!
- But why would this stop? We can jump backwards!
  Because we have an upper bound!


[1, 2, 4, 6]


## 2657

- Maintain `counts`
- How many `counts[i] == 2`?


## 2029

`1 <= stones.length <= 105`
`1 <= stones[i] <= 104`

- Recursive:
  - Inputs: List, sum_removed, is_alice
  - Output: Win if optimal play?
- Sum is 0 or not div by 3
- Restrict moves to x s.t. `(x + sum_removed) % 3 != 0`
- End if 2 left (1 is trivial)

Advanced:
- Avoid int lists and arithmetic: Just boolean (two lists instead of 1)


## 3514

- Simple solution: O(n^3 * log n)
- Better:
  - S = {x ^ y}: O(n^2 * log n) -> Can be O(n^2) long!


## 1846

Goal: Decrease largest entry as little as possible!
Idea:
- Sort increasing
- Change `x[0] = 1`
- Move along: For any `diff > 1`: Decrease 2nd
- Return final value

Better:
- Count different values
- Iterate over x with c > 0. Alternate between using x and
  prev_x + 1, ..., x - 1, until n = len(arr) steps are done

arr = [2,2,1,2,1]

counts = [(1, 2), (2, 3)]

num_left: 3
pref_x: 2
x: 2
c: 3
==> 2

arr = [100,1,1000]

counts = [(1, 1), (100, 1), (1000, 1)]

num_left: 2
pref_x: 1
x: 100
new_fill 98
c: 1
==> 1 + 2 = 3


## 2812

- Dynamic programming
- Create matrix `safeness[r][c]`: Max. safeness factor of any path from
  `(r, c)` to `G = (n - 1, n - 1)`
- Function `min_distance` (or table?)
- Order in which `safeness` cells are computed?

- First: Compute `min_distance` matrix
- Maintain list of positions and safety value `safety_val`:
  From these, there is a path to G with safety `safety_val`, and these are
  the best so far
- Always expand to neighbors of these positions:
  - Collect positions where new `safety_val` is best
  - Only expand these: Tick off in binary matrix
- Stop once expanded to `(0, 0)`

LEARNED ABOUT DP:

If no obvious linear ordering:
- Keep current top scorers
- Expand: Consider all candidates (new neighbors), but only expand those which
  obtain the new top score
==> This IS the correct ordering then!


## 3020

- Isolate numbers which appear >1 times
