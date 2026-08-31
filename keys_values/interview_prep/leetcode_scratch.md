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


## 3737

- Single items: Special case
- Brute force
- Isolate positions of `target`. Use them

`nums[start:(tpos + 1)]`: `i + 1` equal to `target`

(i, tpos) -> (i + 1, tpos_next)
- Skip if 2 * i < tpos - start
- 

i + 1 > tpos + 1 - start - (i + 1) = tpos - start - i
2 * i + 1 > tpos - start
2 * i >= tpos - start

nums = [1,2,2,3], target = 2

target_pos: [1, 2]
len_nums: 4

--> [(1, 2, 2), (2,), (2, 2), (2, 2, 3), (2,)]
num_subarrays: 5

start: 2
pos_tpos: 1

(2, 4)

i: 0
tpos: 2
tpos_next: 4
diff: 0
result: 1


## 1872

- Greedy: Pick x s.t. score is max/min
  Not clear why this would be optimal!
- Later: Denies opponent score possibility
- A: Does not make sense to score negative!
  ==> Recursive on all where score is positive


scores = [-1,2,-3,4,-5]

_scorediff_for(start=0, val_first=0, is_alice=True)

player_sgn = 1
scores: [1 - (-7) = 8, 2 - (-3) = 5, -3]
i: 3
curr_sum: 2

_scorediff_for(start=4, val_first=2, is_alice=False) [2] -> -3


_scorediff_for(start=2, val_first=1, is_alice=False) [1] -> -7

player_sgn: -1
scores: [-2 - 5 = -7, -3]
i: 1
x: 4
curr_sum: 2


_scorediff_for(start=3, val_first=-2, is_alice=True) [3] -> 5

player_sgn: 1
scores: [2 - (-3) = 5, -3]
i: 0
curr_sum: 2


## 2948

- sorted_nums = sorted(nums), with sortind
- Go over gaps between neighboring. For any gap > limit: If they are in wrong
  order: This is where it stops!
- OK: But have to return a full array!
- AND: Careful with equal values!

nums = [1,7,6,18,2,1], limit = 3

nums: [1, 6, 7, 18, 1, 2]

i: 4


## 2075

encodedText: "iveo    eed   l te   olc"
rows: 4
cols: 6
num_parts: 4

  0    1    2    3    4    5    6    7    8    9
['i', 'v', 'e', 'o', ' ', ' ', ' ', ' ', 'e', 'e',
 'd', ' ', ' ', ' ', 'l', ' ', 't', 'e', ' ', ' ',
 ' ', 'o', 'l', 'c']

j: 3
decoded: "i love leetcode"


dec[i + j * rows] = mat[i][j + i] = enc[i * (cols + 1) + j]


## 2126

- Collide with astroids in non-decreasing order

10, [3 5 9 19 21]
10 >= 3
13 >= 5
18 >= 9
27 >= 19
27+19 >= 21


## 3751

- Process in groups of 10

num1 = 105
num2 = 106

a = 5, b = 100, num2 - b + 1 = 7

--> rng2 = (5, 7)

min(num2 - b + 1, 10)

num1: 198
num2: 202

total_waviness: 3

curr_num: 210
digits: [2, 0, 0]
num_peaks: 0
rng1: (1, 10)
a: 0
b: 200
rng2: (0, 3)
num_intersect: max(0, 3 - 1) = 2


## 3532

- Can move i <--> j iff |nums[i] - nums[j]| <= maxDiff
- Nodes 0:n can be clustered by gaps between adjacent `nums[i]` larger than `maxDiff`

nums = [2,5,6,8], maxDiff = 2
queries = [[0,1],[0,2],[1,3],[2,3]]

cluster_ranges = [(0, 1), (1, 4)]

result = [False, False, True, True]


## 3534

nums = [1, 8, 3, 4, 2]
sorted_nums = [(0, 1), (4, 2), (2, 3), (3, 4), (1, 8)]
0 -> 0
1 -> 4
2 -> 2
3 -> 3
4 -> 1


## 2492

- Path from 1 to n exists
- For any road r reachable from 1: Go to r, come back to 1, go to n
- Minimum score is min() over roads reachable from 1
- Road (ai, bi) reachable from 1 iff ai or bi reachable from 1

Plan:
- Connected component of 1 -> C
- Min over roads with ai or bi in C
- Can do the minimum on the fly

[[1,2,9],
 [2,3,6],
 [2,4,5],
 [1,4,7]]

edges:
1: [(2,9), (4,7)] -> []
2: [(1,9), (3,6), (4,5)] -> [(1,9), (4,5)]
3: [(2,6)]
4: [(2,5), (1,7)]

min_score: 5
nodes: {1, 2, 3, 4}

min_score: 5 (!)
extra_nodes: [3]
node: 4
neighbors: [(2,5), (1,7)]
new_neighbors: [(2,5), (1,7)]
