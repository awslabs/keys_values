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


## 1358

- Substring: len >= 3
- Iterate over start
- Find smallest end -> all longer as well
- When start increased: One letter may drop to 0: Search for that one!

abc, abca, abcab, abcabc, bca, bcab, bcabc, cab, cabc, abc

s = "abcabc"
n = 6

start: 3
end: 6
num_substrings: 4 + 3 + 2 + 1
hist: {a: 1, b: 1, c: 1}
elem: 'c'

s = "aaacb"
n = 5

start: 3
end: 6
num_substrings: 1 + 1 + 1 -> 3
hist: {b: 1, c: 1}
elem: 'a'

s = "abc"
n = 3

start: 1
end: 3
num_substrings: 1 -> 1
hist: {b: 1, c: 1}
elem: 'a'


## 3568

- First find S and number of L
- Is this DP? If so, w.r.t. which score?
  ==> Looks tough!
- Otherwise: just recursive?
- Just a BFS w.r.t. number of steps? Blows up!
  Can't we fuse paths to the same cell? By just keeping the best so far?
  What does "best" mean?
  - Most L collected
  - Most energy left to move
  How to combine these?
  Heuristic: First, number of L collected. Second, energy left to move
- One issue: Once L is collected, it becomes ".". But not for all paths, only
  the ones who collected it!
  ==> Have to store L positions, and for each path: Which have been collected?
- Another issue: How to detect that this is not possible??

Testing:

L.S
RXL

m = 2, n = 3, energy = 3
start_pos: (0, 2)
litter_pos: {(0, 0): 0, (1, 2): 1}
num_litter: 2

num_steps: 5
paths: {(1, 0): ([10], 3), (0, 1): ([10], 1)}

new_paths: {(0, 0): ([10], 2)}

AHH! Cycle!

Need one more rule:
For each R state:
- Retain best coll_patt so far
- If come back and it's not better: Do not allow for this path
==> Prevents cycles due to infinite "recharging"

Test:

S.
XL

m = n = 2, energy = 2
start_pos: (0, 0)
litter_pos: {(1, 1): 0}
rest_pos: {}

num_steps: 1
paths: {(0, 1): ([0], 1)}

new_paths: {(0, 1): ([0], 1)}


## 1861

- Each row is an i.i.d. problem
- Row i becomes column m - i - 1 in the result

Each row:
- "*" remain where they are
- Anything in between


## 115

- Recursive?

s = "babgbag", t = "bag"

full_len_s: 7
full_len_t: 3

num(0, 0):
  first_t: b
  len_s: 7
  len_t: 3
  off = 1, 2, 3, 4
  result = num(1, 1) + num(3, 1) + num(5, 1) = 3 + 1 + 1 = 5

num(1, 1):
  first_t: a
  len_s: 6
  len_t: 2
  off = 1, 2, 3, 4
  result = num(2, 2) + num(6, 2) = 2 + 1 = 3

num(3, 1):
  first_t: a
  start_t: 3 -> 5
  len_s: 2
  ==> 1

num(5, 1):
  ==> 1

num(2, 2):
  first_t: g
  start_t: 2 -> 3
  len_s: 4
  len_t: 1
  ==> 2

num(6, 2):
  first_t: g
  len_s: 1
  len_t: 1
  ==> 1

num(s = "rabbbit", t = "rabbit")
= num(s="abbbit", t="abbit")
= num(s="bbbit", t="bbit")
= num(s="bbit", t="bit") + num(s="bit", t="bit") = num(s="bbit", t="bit") + 1

num(s="bbit", t="bit")
= num(s="bit", t="it") + num(s="it", t="it")
= num(s="it", t="it") + num(s="it", t="it") = 2


## 39

Properties/constraints:
- candidates: distinct entries
- number of unique combinations is < 150

Ideas:
- Work with count vectors
- Sort increasing, then from the right
- Use recursion

candidates = [2,3,6,7], target = 7

cs(end=4, target=7):
  cand_last: 7
  result: []
  num_last: 0, 1
  -> cs(end=3, target=7); then l + [0]
  -> append [0, 0, 0, 1]

cs(end=3, target=7)
  cand_last: 6
  result: []
  num_last: 0, 1

Would memoization work here?


## 134

Properties/constraints:
- Both n and gas, cost values can be large (but also 0)
- Need to find starting station which works to go around the whole ring

Observations:
- Need to start at i where `gas[i] >= cost[i]`
- Use `delta[i] = gas[i] - cost[i]`
- Only start where `delta[i] > 0`. Note that `delta[i] == 0` steps are "free"
- Consider using `cumsum` as precomputation!


gas = [1,2,3,4,5], cost = [3,4,5,1,2]
delta = [-2, -2, -2, 3, 3]
delta_cumsum = [0, -2, -4, -6, -3, 0]

pos in [3, 4]:
  pos: 3
  # first
  off: -6
  # second
  off: 6
  ==> Return 3


## 1927

Properties/constraints:
- num has even length, can be very long
- "?" can be replaced by digits
- Positions do not matter, but only: How many ? are left/right, and
  what are the current sums

State: `(sum_left, sum_right, free_left, free_right)`

Greedy idea:
- A tries to maximize `abs(sum_left - sum_right)` with every move
- B tries to minimize `abs(sum_left - sum_right)` with every move

num = "?329 5???"
==> (14, 5, 1, 3)
A: 9L -> (23, 5, 0, 3)
B: 9R -> (23, 14, 0, 2)
A: 0R -> (23, 14, 0, 1)
B: 9R -> (23, 23, 0, 0)

Greedy solution can be done in O(1), from start state:
(sum_left, sum_right, free_left, free_right)

delta = sum_left - sum_right
- If delta == 0:
  A wins if free_left + free_right is odd; B otherwise.
  Namely: 