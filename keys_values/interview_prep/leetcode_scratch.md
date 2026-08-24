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
