from collections import defaultdict, Counter
import math
from itertools import accumulate
from typing import List, Optional, Dict, Set, Tuple, Union


# === Medium ===


# OK
class Solution_3302:
    """
    https://leetcode.com/problems/find-the-lexicographically-smallest-valid-sequence/description/?envType=daily-question&envId=2026-08-20

    You are given two strings word1 and word2.

    A string x is called almost equal to y if you can change at most one
    character in x to make it identical to y.

    A sequence of indices seq is called valid if:

    - The indices are sorted in ascending order.
    - Concatenating the characters at these indices in word1 in the same order
      results in a string that is almost equal to word2.

    Return an array of size word2.length representing the valid sequence of
    indices. If no such sequence of indices exists, return an empty array.

    Note that the answer must represent the lexicographically smallest array,
    not the corresponding string formed by those indices.

    """

    def find_subseq(self, word: str, seq: str) -> Optional[List[int]]:
        len_seq = len(seq)
        result = []
        pos = 0
        elem = seq[pos]
        for i, x in enumerate(word):
            if x == elem:
                result.append(i)
                pos += 1
                if pos == len_seq:
                    return result
                else:
                    elem = seq[pos]
        return None

    def lexico_min(self, a: List[int], b: List[int]) -> List[int]:
        assert len(a) == len(b)
        for x, y in zip(a, b):
            if x < y:
                return a
            elif x > y:
                return b
        return a

    def validSequence(self, word1: str, word2: str) -> List[int]:
        len1 = len(word1)
        len2 = len(word2)
        assert len1 > len2 >= 1
        if len2 == 1:
            return [0]
        stop_index = [len1] * len2
        # `result` is lexico smallest valid index so far, or `stop_index` if none
        # has been found so far
        result = stop_index
        # `prefix` maintains positions of initial part of `word2` in `word1`
        prefix = []
        curr_pos = 0  # Current position in `word1`
        for skip_pos, skip_elem in enumerate(word2):
            if skip_pos == len2 - 1:
                # Skip final. Stop after that
                candidate = prefix + [curr_pos]
                result = self.lexico_min(result, candidate)
                break
            # Skip position `skip_pos`, then match the rest
            off = curr_pos + 1
            postfix = self.find_subseq(word1[off:], word2[(skip_pos + 1) :])
            if postfix is not None:
                candidate = prefix + [curr_pos] + [x + off for x in postfix]
                result = self.lexico_min(result, candidate)
            # Don't skip `skip_pos`, increase `prefix`
            off = curr_pos
            next_pos = self.find_subseq(word1[off:], skip_elem)
            if next_pos is None:
                # Cannot increase `prefix`: Stop
                break
            next_pos = next_pos[0] + off
            prefix.append(next_pos)
            curr_pos = next_pos + 1
            if curr_pos >= len1:
                break
        if result == stop_index:
            return []
        else:
            return result


# OK
class Solution_3517:
    def smallestPalindrome(self, s: str) -> str:
        """
        You are given a palindromic string s.

        Return lexico smallest palindromic permutation of s.

        """
        # Is this correct if length is odd?
        len_s = len(s)
        is_even = len_s % 2 == 0
        len_half = len_s // 2
        half_lst = sorted(s[:len_half])
        if is_even:
            middle = []
        else:
            middle = [s[len_half]]
        return "".join(half_lst + middle + list(reversed(half_lst)))


# OK
class Solution_486:
    """
    https://leetcode.com/problems/predict-the-winner/description/?envType=daily-question&envId=2026-08-20

    You are given an integer array nums. Two players are playing a game with
    this array: player 1 and player 2.

    Player 1 and player 2 take turns, with player 1 starting first. Both players
    start the game with a score of 0. At each turn, the player takes one of the
    numbers from either end of the array (i.e., nums[0] or nums[nums.length - 1])
    which reduces the size of the array by 1. The player adds the chosen number
    to their score. The game ends when there are no more elements in the array.

    Return true if Player 1 can win the game. If the scores of both players are
    equal, then player 1 is still the winner, and you should also return true.
    You may assume that both players are playing optimally.

    """

    def play_best_move(
        self,
        start: int,
        end: int,
        is_player1: bool,
    ) -> int:
        """
        If game is played on list `self.nums[start:end]`, and player 1 starts if
        `is_player == True`, otherwise player 2, the score difference between player 1
        and player 2 is returned if both play optimally.

        This is done recursively. If `end - start > 2`, call `play_best_move` for
        both options (choose left, choose right), the pick the one with better score
        difference for the current player.

        """
        sz = end - start
        elem1 = self.nums[start]
        elem2 = self.nums[end - 1]
        pl_sgn = 1 if is_player1 else -1
        if sz == 2:
            return pl_sgn * abs(elem1 - elem2)
        else:
            sdiff1 = (
                self.play_best_move(start + 1, end, not is_player1) + pl_sgn * elem1
            )
            sdiff2 = (
                self.play_best_move(start, end - 1, not is_player1) + pl_sgn * elem2
            )
            return max(sdiff1, sdiff2) if is_player1 else min(sdiff1, sdiff2)

    def predictTheWinner(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        self.nums = nums
        return self.play_best_move(0, len(nums), True) >= 0


# OK
class Solution_1140:
    """
    https://leetcode.com/problems/stone-game-ii/?envType=daily-question&envId=2026-08-20

    Alice and Bob continue their games with piles of stones. There are a number
    of piles arranged in a row, and each pile has a positive integer number of
    stones piles[i]. The objective of the game is to end with the most stones.

    Alice and Bob take turns, with Alice starting first.

    On each player's turn, that player can take all the stones in the first X
    remaining piles, where 1 <= X <= 2M. Then, we set M = max(M, X). Initially,
    M = 1.

    The game continues until all the stones have been taken.

    Assuming Alice and Bob play optimally, return the maximum number of stones
    Alice can get.

    """

    def max_score_for(
        self,
        start: int,
        m: int,
    ) -> int:
        # Say this is called for player A (it is symmetric for B)
        if m * 2 >= self.num_piles - start:
            # A takes all the rest
            return self.sum_all - self.cumsum[start]
        max_score = 0
        for x in range(1, m * 2 + 1):
            # A takes `x` piles, then B scores optimally, and A takes the rest
            score_other = self.max_score_for(start + x, max(m, x))
            all_rest = self.sum_all - self.cumsum[start + x]
            score_me = (
                self.cumsum[start + x] - self.cumsum[start] + all_rest - score_other
            )
            max_score = max(max_score, score_me)
        return max_score

    def stoneGameII(self, piles: List[int]) -> int:
        # General trick we use: We compute the cumulative sums for `piles`.
        # This allows to compute sums over consecutive ranges in O(1).
        self.piles = piles
        self.num_piles = len(piles)
        self.cumsum = []
        csum = 0
        for x in piles:
            self.cumsum.append(csum)
            csum += x
        self.sum_all = csum
        return self.max_score_for(0, 1)


# OK
# - Fixed bug: `return end` -> `return pos`
class Solution_2958:
    """
    https://leetcode.com/problems/length-of-longest-subarray-with-at-most-k-frequency/?envType=daily-question&envId=2026-08-20

    You are given an integer array `nums` and an integer `k`.

    The frequency of an element x is the number of times it occurs in an array.

    An array is called good if the frequency of each element in this array is
    less than or equal to `k`.

    Return the length of the longest good subarray of nums.

    A subarray is a contiguous non-empty sequence of elements within an array.

    Example 1:

    Input: nums = [1,2,3,1,2,3,1,2], k = 2
    Output: 6
    Explanation: The longest possible good subarray is [1,2,3,1,2,3] since the values 1, 2, and 3 occur at most twice in this subarray. Note that the subarrays [2,3,1,2,3,1] and [3,1,2,3,1,2] are also good.
    It can be shown that there are no good subarrays with length more than 6.

    Example 2:

    Input: nums = [1,2,1,2,1,2,1,2], k = 1
    Output: 2
    Explanation: The longest possible good subarray is [1,2] since the values 1 and 2 occur at most once in this subarray. Note that the subarray [2,1] is also good.
    It can be shown that there are no good subarrays with length more than 2.

    Example 3:

    Input: nums = [5,5,5,5,5,5,5], k = 4
    Output: 4
    Explanation: The longest possible good subarray is [5,5,5,5] since the value 5 occurs 4 times in this subarray.
    It can be shown that there are no good subarrays with length more than 4.

    Constraints:

    1 <= nums.length <= 105
    1 <= nums[i] <= 109
    1 <= k <= nums.length

    """

    def extend(
        self,
        end: int,
        hist: Dict[int, int],
    ) -> int:
        for pos in range(end, self.len_nums):
            x = self.nums[pos]
            curr = hist.get(x, 0)
            if curr >= self.k:
                return pos
            hist[x] = curr + 1
        return len(self.nums)

    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        """
        - `extend(0, {})` builds histogram for max good subarray starting at 0
        - Iterate: Increase `start` and decrease count. Now, `extend(end, hist)`
          extends subarray and histogram until max good again

        """
        self.nums = nums
        self.k = k
        self.len_nums = len(nums)
        hist: Dict[int, int] = dict()
        start = 0
        end = 0
        max_len = 0
        for elem in self.nums:
            end = self.extend(end, hist)
            max_len = max(max_len, end - start)
            if end == self.len_nums:
                break
            start += 1
            hist[elem] -= 1
        return max_len


# OK
# - Fixed bug: `num_of_cost[c] += 1` -> `num_of_cost[c - 1] += 1`
class Solution_1833:
    """
    https://leetcode.com/problems/maximum-ice-cream-bars/?envType=daily-question&envId=2026-08-20

    It is a sweltering summer day, and a boy wants to buy some ice cream bars.

    At the store, there are `n` ice cream bars. You are given an array `costs`
    of length `n`, where `costs[i]` is the price of the ith ice cream bar in coins.
    The boy initially has coins `coins` to spend, and he wants to buy as many ice
    cream bars as possible.

    Note: The boy can buy the ice cream bars in any order.

    Return the maximum number of ice cream bars the boy can buy with `coins`
    coins.

    You must solve the problem by counting sort.

    Example 1:

    Input: costs = [1,3,2,4,1], coins = 7
    Output: 4
    Explanation: The boy can buy ice cream bars at indices 0,1,2,4 for a total price of 1 + 3 + 2 + 1 = 7.

    Example 2:

    Input: costs = [10,6,8,7,7,8], coins = 5
    Output: 0
    Explanation: The boy cannot afford any of the ice cream bars.

    Example 3:

    Input: costs = [1,6,3,1,2,5], coins = 20
    Output: 6
    Explanation: The boy can buy all the ice cream bars for a total price of 1 + 6 + 3 + 1 + 2 + 5 = 18.

    Constraints:

    costs.length == n
    1 <= n <= 105
    1 <= costs[i] <= 105
    1 <= coins <= 108

    """

    def maxIceCream(self, costs: List[int], coins: int) -> int:
        # Sort `costs` and buy from cheapest upwards
        # Must use counting sort
        # Count number of bars for each cost value. Note that
        # `num_of_costs[c]` is for cost `c + 1`.
        num_of_cost = [0] * 105
        for c in costs:
            num_of_cost[c - 1] += 1
        num_bought = 0
        for cost, num in enumerate(num_of_cost):
            if num > 0:
                cost += 1
                cost_here = cost * num
                if cost_here >= coins:
                    num_bought += coins // cost
                    break
                num_bought += num
                coins -= cost_here
        return num_bought


# OK
# ==> DAMN!! DOES NOT WORK!!
class Solution_2161:
    """
    https://leetcode.com/problems/partition-array-according-to-given-pivot/?envType=daily-question&envId=2026-08-20

    You are given a 0-indexed integer array `nums` and an integer `pivot`.
    Rearrange `nums` such that the following conditions are satisfied:

    * Every element less than `pivot` appears before every element greater than
      pivot.
    * Every element equal to `pivot` appears in between the elements less than
      and greater than `pivot`.
    * The relative order of the elements less than `pivot` and the elements
      greater than `pivot` is maintained.

    More formally, consider every pi, pj where pi is the new position of the
    ith element and pj is the new position of the jth element. If i < j and
    both elements are smaller (or larger) than pivot, then pi < pj.

    Return `nums` after the rearrangement.

    Example 1:

    Input: nums = [9,12,5,10,14,3,10], pivot = 10
    Output: [9,5,3,10,10,12,14]
    Explanation:
    The elements 9, 5, and 3 are less than the pivot so they are on the left side of the array.
    The elements 12 and 14 are greater than the pivot so they are on the right side of the array.
    The relative ordering of the elements less than and greater than pivot is also maintained. [9, 5, 3] and [12, 14] are the respective orderings.

    Example 2:

    Input: nums = [-3,4,3,2], pivot = 2
    Output: [-3,2,4,3]
    Explanation:
    The element -3 is less than the pivot so it is on the left side of the array.
    The elements 4 and 3 are greater than the pivot so they are on the right side of the array.
    The relative ordering of the elements less than and greater than pivot is also maintained. [-3] and [4, 3] are the respective orderings.

    Constraints:

    * 1 <= nums.length <= 105
    * -106 <= nums[i] <= 106
    * pivot equals to an element of nums.

    """

    def swap(self, p1: int, p2: int):
        elem = self.nums[p1]
        self.nums[p1] = self.nums[p2]
        self.nums[p2] = elem

    def revert(self, first: int, last: int):
        if last < first + 1:
            return
        for i in range((last - first + 1) // 2):
            self.swap(first + i, last - i)

    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        self.nums = nums
        # In-place solution
        # - Pairwise swap solution: Ignores relative ordering
        # - Partition left into [done | moved], right into [moved | done]
        # - After each swap: Bubble entries left and/or right to bring them
        #   to "done" part
        # - "moved" parts have to be reverted at end
        # - pivot entries kept on left, sorted out at end
        left_pos = left_done = 0
        right_pos = right_done = len(nums) - 1
        while left_pos < right_pos:
            while left_pos < right_pos and nums[left_pos] <= pivot:
                elem = nums[left_pos]
                if elem < pivot:
                    # Bubble entry to the left
                    for i in range(left_done, left_pos):
                        nums[i + 1] = nums[i]
                    nums[left_done] = elem
                    left_done += 1
                left_pos += 1
            while left_pos < right_pos and nums[right_pos] > pivot:
                elem = nums[right_pos]
                # Bubble entry to right
                for i in range(right_pos, right_done):
                    nums[i] = nums[i + 1]
                nums[right_done] = elem
                right_done -= 1
                right_pos -= 1
            if left_pos >= right_pos:
                break
            self.swap(left_pos, right_pos)
            left_pos += 1
        # At this point:
        # - range(0, left_done): OK
        # - range(left_done, left_pos): Revert and move pivot right
        # - range(right_pos + 1, right_done + 1): Revert
        # - range(right_done + 1, N): OK
        self.revert(right_pos + 1, right_done)
        if left_pos > left_done:
            # Move pivot entries to right
            pos = left_done
            for x in nums[left_done:left_pos]:
                if x < pivot:
                    nums[pos] = x
                    pos += 1
            if pos < left_pos:
                nums[pos:left_pos] = [pivot] * (left_pos - pos)
            self.revert(left_done, pos - 1)
        return nums


# OK
class Solution_3635:
    """
    https://leetcode.com/problems/earliest-finish-time-for-land-and-water-rides-ii/?envType=daily-question&envId=2026-08-20

    You are given two categories of theme park attractions: land rides and
    water rides.

    * Land rides
      - landStartTime[i] – the earliest time the ith land ride can be boarded.
      - landDuration[i] – how long the ith land ride lasts.
    * Water rides
      - waterStartTime[j] – the earliest time the jth water ride can be boarded.
      - waterDuration[j] – how long the jth water ride lasts.

    A tourist must experience exactly one ride from each category, in either order.

    * A ride may be started at its opening time or any later moment.
    * If a ride is started at time t, it finishes at time t + duration.
    * Immediately after finishing one ride the tourist may board the other (if
      it is already open) or wait until it opens.

    Return the earliest possible time at which the tourist can finish both rides.

    Example 1:

    Input: landStartTime = [2,8], landDuration = [4,1], waterStartTime = [6], waterDuration = [3]

    Output: 9

    Explanation:

    Plan A (land ride 0 → water ride 0):
        Start land ride 0 at time landStartTime[0] = 2. Finish at 2 + landDuration[0] = 6.
        Water ride 0 opens at time waterStartTime[0] = 6. Start immediately at 6, finish at 6 + waterDuration[0] = 9.
    Plan B (water ride 0 → land ride 1):
        Start water ride 0 at time waterStartTime[0] = 6. Finish at 6 + waterDuration[0] = 9.
        Land ride 1 opens at landStartTime[1] = 8. Start at time 9, finish at 9 + landDuration[1] = 10.
    Plan C (land ride 1 → water ride 0):
        Start land ride 1 at time landStartTime[1] = 8. Finish at 8 + landDuration[1] = 9.
        Water ride 0 opened at waterStartTime[0] = 6. Start at time 9, finish at 9 + waterDuration[0] = 12.
    Plan D (water ride 0 → land ride 0):
        Start water ride 0 at time waterStartTime[0] = 6. Finish at 6 + waterDuration[0] = 9.
        Land ride 0 opened at landStartTime[0] = 2. Start at time 9, finish at 9 + landDuration[0] = 13.

    Plan A gives the earliest finish time of 9.

    Example 2:

    Input: landStartTime = [5], landDuration = [3], waterStartTime = [1], waterDuration = [10]

    Output: 14

    Explanation:

    Plan A (water ride 0 → land ride 0):
        Start water ride 0 at time waterStartTime[0] = 1. Finish at 1 + waterDuration[0] = 11.
        Land ride 0 opened at landStartTime[0] = 5. Start immediately at 11 and finish at 11 + landDuration[0] = 14.
    Plan B (land ride 0 → water ride 0):
        Start land ride 0 at time landStartTime[0] = 5. Finish at 5 + landDuration[0] = 8.
        Water ride 0 opened at waterStartTime[0] = 1. Start immediately at 8 and finish at 8 + waterDuration[0] = 18.

    Plan A provides the earliest finish time of 14.

    Constraints:

    * 1 <= n, m <= 5 * 104
    * landStartTime.length == landDuration.length == n
    * waterStartTime.length == waterDuration.length == m
    * 1 <= landStartTime[i], landDuration[i], waterStartTime[j], waterDuration[j] <= 105

    """

    def earliestFinishTime(
        self,
        landStartTime: List[int],
        landDuration: List[int],
        waterStartTime: List[int],
        waterDuration: List[int],
    ) -> int:
        land_earliest_time = min(x + y for x, y in zip(landStartTime, landDuration))
        water_earliest_time = min(x + y for x, y in zip(waterStartTime, waterDuration))
        return min(
            min(
                max(land_earliest_time, x) + y
                for x, y in zip(waterStartTime, waterDuration)
            ),
            min(
                max(water_earliest_time, x) + y
                for x, y in zip(landStartTime, landDuration)
            ),
        )


# OK
class Solution_1871:
    """
    https://leetcode.com/problems/jump-game-vii/description/?envType=daily-question&envId=2026-08-20

    You are given a 0-indexed binary string s and two integers minJump and
    maxJump. In the beginning, you are standing at index 0, which is equal
    to '0'. You can move from index i to index j if the following conditions
    are fulfilled:

    * i + minJump <= j <= min(i + maxJump, s.length - 1), and
    * s[j] == '0'.

    Return true if you can reach index s.length - 1 in s, or false otherwise.

    Example 1:

    Input: s = "011010", minJump = 2, maxJump = 3
    Output: true
    Explanation:
    In the first step, move from index 0 to index 3.
    In the second step, move from index 3 to index 5.

    Example 2:

    Input: s = "01101110", minJump = 2, maxJump = 3
    Output: false

    Constraints:

    * 2 <= s.length <= 105
    * s[i] is either '0' or '1'.
    * s[0] == '0'
    * 1 <= minJump <= maxJump < s.length

    """

    def _canReach(self, start: int) -> bool:
        pos = start
        if pos + self.minJump > self.fin_pos:
            return False
        elif pos + self.maxJump >= self.fin_pos:
            return True
        for i in range(pos + self.minJump, pos + self.maxJump + 1):
            if self.s[i] == "0" and self._canReach(i):
                return True
        return False

    def canReach(self, s: str, minJump: int, maxJump: int) -> bool:
        assert s[0] == "0"
        self.fin_pos = len(s) - 1
        if s[-1] != "0":
            return False
        elif self.fin_pos == 0:
            return True
        self.s = s
        self.minJump = minJump
        self.maxJump = maxJump
        return self._canReach(0)


# OK
# - Not happy with this solution: Could be really inefficient!
class Solution_3629:
    """
    https://leetcode.com/problems/minimum-jumps-to-reach-end-via-prime-teleportation/?envType=daily-question&envId=2026-08-20

    You are given an integer array `nums` of length `n`.

    You start at index 0, and your goal is to reach index `n - 1`.

    From any index `i`, you may perform one of the following operations:

    * Adjacent Step: Jump to index `i + 1` or `i - 1`, if the index is within
      bounds.
    * Prime Teleportation: If `nums[i]` is a prime number `p`, you may instantly
      jump to any index `j != i` such that `nums[j] % p == 0`.

    Return the minimum number of jumps required to reach index `n - 1`.

    Example 1:

    Input: nums = [1,2,4,6]

    Output: 2

    Explanation:

    One optimal sequence of jumps is:

    * Start at index i = 0. Take an adjacent step to index 1.
    * At index i = 1, nums[1] = 2 is a prime number. Therefore, we teleport to index i = 3 as nums[3] = 6 is divisible by 2.

    Thus, the answer is 2.

    Example 2:

    Input: nums = [2,3,4,7,9]

    Output: 2

    Explanation:

    One optimal sequence of jumps is:

    * Start at index i = 0. Take an adjacent step to index i = 1.
    * At index i = 1, nums[1] = 3 is a prime number. Therefore, we teleport to index i = 4 since nums[4] = 9 is divisible by 3.

    Thus, the answer is 2.

    Example 3:

    Input: nums = [4,6,5,8]

    Output: 3

    Explanation:

    * Since no teleportation is possible, we move through 0 → 1 → 2 → 3. Thus, the answer is 3.

    Constraints:

    * 1 <= n == nums.length <= 105
    * 1 <= nums[i] <= 106

    """

    def primeSieve(self, max_num: int) -> List[bool]:
        result = [True] * (max_num + 1)
        curr = 2
        limit = max_num // 2 + 1
        while curr < limit:
            for i in range(2 * curr, max_num + 1, curr):
                result[i] = False
            curr += 1
            while curr < limit and not result[curr]:
                curr += 1
        return result

    def _min_jumps(self, pos: int, num_done: int) -> int:
        if num_done >= self.max_jumps:
            return self.max_jumps
        if pos == self.max_jumps - 1:
            return 1
        entry = self.nums[pos]
        entry_is_prime = entry > 1 and self.is_prime[entry]
        if entry_is_prime and self.fin_entry % entry == 0:
            return 1
        # Start with jumps to neighbors
        min_val = self._min_jumps(pos + 1, num_done + 1) + 1
        if min_val == 2:
            return 2  # cannot be better than that
        if pos > 0:
            min_val = min(
                min_val,
                self._min_jumps(pos - 1, num_done + 1) + 1,
            )
            if min_val == 2:
                return 2
        if entry_is_prime:
            # Consider teleportation jumps
            candidates = [
                pnext
                for pnext in list(range(1, pos - 1))
                + list(range(pos + 2, self.max_jumps))
                if self.nums[pnext] % entry == 0
            ]
            for pnext in reversed(candidates):
                min_val = min(
                    min_val,
                    self._min_jumps(pnext, num_done + 1) + 1,
                )
                if min_val == 2:
                    return 2
        return min(min_val, self.max_jumps)

    def minJumps(self, nums: List[int]) -> int:
        max_num = 105
        assert all(1 <= x <= 105 for x in nums)
        if len(nums) == 1:
            return 0
        self.is_prime = self.primeSieve(max_num)
        self.nums = nums
        self.fin_entry = self.nums[-1]
        self.max_jumps = len(nums) - 1
        return self._min_jumps(0, 0)


# OK
class Solution_2657:
    """
    https://leetcode.com/problems/find-the-prefix-common-array-of-two-arrays/?envType=daily-question&envId=2026-08-20

    You are given two 0-indexed integer permutations A and B of length n.

    A prefix common array of A and B is an array C such that C[i] is equal to
    the count of numbers that are present at or before the index i in both A
    and B.

    Return the prefix common array of A and B.

    A sequence of n integers is called a permutation if it contains all
    integers from 1 to n exactly once.

    Example 1:

    Input: A = [1,3,2,4], B = [3,1,2,4]
    Output: [0,2,3,4]
    Explanation: At i = 0: no number is common, so C[0] = 0.
    At i = 1: 1 and 3 are common in A and B, so C[1] = 2.
    At i = 2: 1, 2, and 3 are common in A and B, so C[2] = 3.
    At i = 3: 1, 2, 3, and 4 are common in A and B, so C[3] = 4.

    Example 2:

    Input: A = [2,3,1], B = [3,1,2]
    Output: [0,1,3]
    Explanation: At i = 0: no number is common, so C[0] = 0.
    At i = 1: only 3 is common in A and B, so C[1] = 1.
    At i = 2: 1, 2, and 3 are common in A and B, so C[2] = 3.

    Constraints:

    * 1 <= A.length == B.length == n <= 50
    * 1 <= A[i], B[i] <= n
    * It is guaranteed that A and B are both a permutation of n integers.

    """

    def findThePrefixCommonArray(self, A: List[int], B: List[int]) -> List[int]:
        # Simple solution
        n = len(A)
        assert n == len(B)
        counts = [0] * n
        result = []
        for x, y in zip(A[:-1], B[:-1]):
            counts[x - 1] += 1
            counts[y - 1] += 1
            result.append(sum(c == 2 for c in counts))
        result.append(n)
        return result


# OK
class Solution_1344:
    """
    https://leetcode.com/problems/angle-between-hands-of-a-clock/?envType=daily-question&envId=2026-08-20

    Given two numbers, hour and minutes, return the smaller angle (in
    degrees) formed between the hour and the minute hand.

    Answers within 10-5 of the actual value will be accepted as correct.

    """

    def angleClock(self, hour: int, minutes: int) -> float:
        # - 60 minutes <-> 360 degrees
        angle_minute = float(minutes * 6)
        # - 12 hours <-> 360 degrees
        # - 1 hour: 30 degrees <-> 60 minutes
        angle_hour = (hour % 12) * 30 + minutes * 0.5
        min_angle = abs(angle_hour - angle_minute)
        return min(min_angle, 360 - min_angle)


class Solution_2029:
    """
    https://leetcode.com/problems/stone-game-ix/?envType=daily-question&envId=2026-08-25

    Alice and Bob continue their games with stones. There is a row of n stones,
    and each stone has an associated value. You are given an integer array
    stones, where stones[i] is the value of the ith stone.

    Alice and Bob take turns, with Alice starting first. On each turn, the
    player may remove any stone from stones. The player who removes a stone
    loses if the sum of the values of all removed stones is divisible by 3. Bob
    will win automatically if there are no remaining stones (even if it is
    Alice's turn).

    Assuming both players play optimally, return true if Alice wins and false
    if Bob wins.

    Example 1:

    Input: stones = [2,1]
    Output: true
    Explanation: The game will be played as follows:
    - Turn 1: Alice can remove either stone.
    - Turn 2: Bob removes the remaining stone.
    The sum of the removed stones is 1 + 2 = 3 and is divisible by 3. Therefore, Bob loses and Alice wins the game.

    Example 2:

    Input: stones = [2]
    Output: false
    Explanation: Alice will remove the only stone, and the sum of the values on the removed stones is 2.
    Since all the stones are removed and the sum of values is not divisible by 3, Bob wins the game.

    Example 3:

    Input: stones = [5,1,2,4,3]
    Output: false
    Explanation: Bob will always win. One possible way for Bob to win is shown below:
    - Turn 1: Alice can remove the second stone with value 1. Sum of removed stones = 1.
    - Turn 2: Bob removes the fifth stone with value 3. Sum of removed stones = 1 + 3 = 4.
    - Turn 3: Alices removes the fourth stone with value 4. Sum of removed stones = 1 + 3 + 4 = 8.
    - Turn 4: Bob removes the third stone with value 2. Sum of removed stones = 1 + 3 + 4 + 2 = 10.
    - Turn 5: Alice removes the first stone with value 5. Sum of removed stones = 1 + 3 + 4 + 2 + 5 = 15.
    Alice loses the game because the sum of the removed stones (15) is divisible by 3. Bob wins the game.

    Constraints:

        1 <= stones.length <= 105
        1 <= stones[i] <= 104

    """

    def caller_wins(
        self,
        stones_rem1: List[bool],
        stones_rem2: List[bool],
        sum_removed_rem1: bool,
        is_alice: bool,
    ) -> bool:
        # `sum_removed_rem1 = (sum_removed % 3) == 1`
        if len(stones_rem1) == 2:
            if (sum_removed_rem1 and all(stones_rem2)) or (
                not sum_removed_rem1 and all(stones_rem1)
            ):
                # All stones lead to sum divisible by 3
                return False
            if self.sum_all_divs_3:
                # Other play loses by picking the final stone
                return True
            return not is_alice
        if sum_removed_rem1:
            x_list = stones_rem2
            y_list = stones_rem1
        else:
            x_list = stones_rem1
            y_list = stones_rem2
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            # Value `sum_removed_rem1` arg:
            # - If sum_removed_rem1 == True:
            #   [sum + el] = [1 + el{0/1}] == 1 iff el == 0 iff not y
            # - If sum_removed_rem1 == False:
            #   [sum + el] = [2 + el{0/2}] == 1 iff el == 2 iff y
            if not x and not self.caller_wins(
                stones_rem1=stones_rem1[:i] + stones_rem1[(i + 1) :],
                stones_rem2=stones_rem2[:i] + stones_rem2[(i + 1) :],
                sum_removed_rem1=not y if sum_removed_rem1 else y,
                is_alice=not is_alice,
            ):
                return True
        return False

    # Fancy solution:
    # - Avoids all integer arithmetic (just boolean)
    # - Uses boolean lists only
    def stoneGameIX(self, stones: List[int]) -> bool:
        if len(stones) == 1:
            return False
        # Solution avoids having to do lots of int computation
        stones_rem1 = [x % 3 == 1 for x in stones]
        stones_rem2 = [x % 3 == 2 for x in stones]
        self.sum_all_divs_3 = sum(stones) % 3 == 0
        # Need initial loop: Cannot call `caller_wins` with sum 0
        for i, (x, y) in enumerate(zip(stones_rem1, stones_rem2)):
            if (x or y) and not self.caller_wins(
                stones_rem1=stones_rem1[:i] + stones_rem1[(i + 1) :],
                stones_rem2=stones_rem2[:i] + stones_rem2[(i + 1) :],
                sum_removed_rem1=x,
                is_alice=False,
            ):
                return True
        return False


# OK (note that XOR is x ^ y in Python)
class Solution_3514:
    """
    https://leetcode.com/problems/number-of-unique-xor-triplets-ii/?envType=daily-question&envId=2026-08-20

    A XOR triplet is defined as the XOR of three elements nums[i] XOR nums[j]
    XOR nums[k] where i <= j <= k.

    Return the number of unique XOR triplet values from all possible triplets
    (i, j, k).

    Example 1:

    Input: nums = [1,3]

    Output: 2

    Explanation: The possible XOR triplet values are:

        (0, 0, 0) → 1 XOR 1 XOR 1 = 1
        (0, 0, 1) → 1 XOR 1 XOR 3 = 3
        (0, 1, 1) → 1 XOR 3 XOR 3 = 1
        (1, 1, 1) → 3 XOR 3 XOR 3 = 3

    The unique XOR values are {1, 3}. Thus, the output is 2.

    Example 2:

    Input: nums = [6,7,8,9]

    Output: 4

    Explanation: The possible XOR triplet values are {6, 7, 8, 9}. Thus, the
    output is 4.

    Constraints:

    * 1 <= nums.length <= 1500
    * 1 <= nums[i] <= 1500

    """

    def uniqueXorTriplets(self, nums: List[int]) -> int:
        triples: Set[int] = set()
        for i, x in enumerate(nums):
            for _j, y in enumerate(nums[i:]):
                j = _j + i
                x_xor_y = x ^ y
                triples.update(x_xor_y ^ z for z in nums[j:])
        return len(triples)

    def uniqueXorTriplets_2(self, nums: List[int]) -> int:
        assert all(0 <= x < 2048 for x in nums)
        mask = [False] * 2048
        for i, x in enumerate(nums):
            for _j, y in enumerate(nums[i:]):
                j = _j + i
                x_xor_y = x ^ y
                for z in nums[j:]:
                    mask[x_xor_y ^ z] = True
        return sum(mask)


# OK
class Solution_1846:
    """
    https://leetcode.com/problems/maximum-element-after-decreasing-and-rearranging/?envType=daily-question&envId=2026-08-25

    You are given an array of positive integers `arr`. Perform some operations
    (possibly none) on `arr` so that it satisfies these conditions:

    * The value of the first element in `arr` must be 1.
    * The absolute difference between any 2 adjacent elements must be less than
      or equal to 1. In other words, `abs(arr[i] - arr[i - 1]) <= 1` for each i
      where 1 <= i < arr.length (0-indexed). `abs(x)` is the absolute value of
      x.

    There are 2 types of operations that you can perform any number of times:

    * Decrease the value of any element of `arr` to a smaller positive integer.
    * Rearrange the elements of `arr` to be in any order.

    Return the maximum possible value of an element in `arr` after performing
    the operations to satisfy the conditions.

    Example 1:

    Input: arr = [2,2,1,2,1]
    Output: 2
    Explanation:
    We can satisfy the conditions by rearranging arr so it becomes [1,2,2,2,1].
    The largest element in arr is 2.

    Example 2:

    Input: arr = [100,1,1000]
    Output: 3
    Explanation:
    One possible way to satisfy the conditions is by doing the following:
    1. Rearrange arr so it becomes [1,100,1000].
    2. Decrease the value of the second element to 2.
    3. Decrease the value of the third element to 3.
    Now arr = [1,2,3], which satisfies the conditions.
    The largest element in arr is 3.

    Example 3:

    Input: arr = [1,2,3,4,5]
    Output: 5
    Explanation: The array already satisfies the conditions, and the largest element is 5.

    Constraints:

    * 1 <= arr.length <= 10^5
    * 1 <= arr[i] <= 10^9

    """

    def maximumElementAfterDecrementingAndRearranging(self, arr: List[int]) -> int:
        # Elegant solution, does not need sorting, but just a histogram
        counts = defaultdict(int)
        for x in arr:
            counts[x] += 1
        counts = sorted(counts.items(), key=lambda x: x[0])
        num_left = len(arr)
        prev_x = 0
        for x, c in counts:
            new_fill = max(0, x - prev_x - 1)
            if new_fill >= num_left:
                return prev_x + num_left
            num_left -= new_fill
            prev_x = x
            if c >= num_left:
                return x
            num_left -= c


# OK: This was tough
class Solution_2812:
    """
    https://leetcode.com/problems/find-the-safest-path-in-a-grid/?envType=daily-question&envId=2026-08-25

    You are given a 0-indexed 2D matrix `grid` of size n x n, where (r, c)
    represents:

    * A cell containing a thief if `grid[r][c] = 1`
    * An empty cell if `grid[r][c] = 0`

    You are initially positioned at cell (0, 0). In one move, you can move to
    any adjacent cell in the grid, including cells containing thieves.

    The safeness factor of a path on the grid is defined as the minimum
    manhattan distance from any cell in the path to any thief in the grid.

    Return the maximum safeness factor of all paths leading to cell
    `(n - 1, n - 1)`.

    An adjacent cell of cell `(r, c)`, is one of the cells `(r, c + 1)`,
    `(r, c - 1)`, `(r + 1, c)` and `(r - 1, c)` if it exists.

    The Manhattan distance between two cells `(a, b)` and `(x, y)` is equal to
    `|a - x| + |b - y|`, where |val| denotes the absolute value of val.

    Example 1:

    Input: grid = [[1,0,0],[0,0,0],[0,0,1]]
    Output: 0
    Explanation: All paths from (0, 0) to (n - 1, n - 1) go through the thieves in cells (0, 0) and (n - 1, n - 1).

    Example 2:

    Input: grid = [[0,0,1],[0,0,0],[0,0,0]]
    Output: 2
    Explanation: The path depicted in the picture above has a safeness factor of 2 since:
    - The closest cell of the path to the thief at cell (0, 2) is cell (0, 0). The distance between them is | 0 - 0 | + | 0 - 2 | = 2.
    It can be shown that there are no other paths with a higher safeness factor.

    Example 3:

    Input: grid = [[0,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,0]]
    Output: 2
    Explanation: The path depicted in the picture above has a safeness factor of 2 since:
    - The closest cell of the path to the thief at cell (0, 3) is cell (1, 2). The distance between them is | 0 - 1 | + | 3 - 2 | = 2.
    - The closest cell of the path to the thief at cell (3, 0) is cell (3, 2). The distance between them is | 3 - 3 | + | 0 - 2 | = 2.
    It can be shown that there are no other paths with a higher safeness factor.

    Constraints:

        1 <= grid.length == n <= 400
        grid[i].length == n
        grid[i][j] is either 0 or 1.
        There is at least one thief in the grid.

    """

    def min_distance_to_thiefs(
        self,
        grid: List[List[int]],
    ) -> List[List[int]]:
        n = len(grid)
        thief_pos = [
            (r, c) for r, row in enumerate(grid) for c, el in enumerate(row) if el == 1
        ]
        min_distance = []
        for r in range(n):
            row = [min(abs(r - x) + abs(c - y) for x, y in thief_pos) for c in range(n)]
            min_distance.append(row)
        return min_distance

    @staticmethod
    def neighbors_not_yet_done(
        pos: Tuple[int, int],
        already_done: List[List[bool]],
    ) -> List[Tuple[int, int]]:
        max_p = len(already_done) - 1
        r, c = pos
        if r == 0:
            cands = [(r + 1, c)]
        elif r == max_p:
            cands = [(r - 1, c)]
        else:
            cands = [(r + 1, c), (r - 1, c)]
        if c == 0:
            cands += [(r, c + 1)]
        elif c == max_p:
            cands += [(r, c - 1)]
        else:
            cands += [(r, c + 1), (r, c - 1)]
        return [(x, y) for x, y in cands if not already_done[x][y]]

    def maximumSafenessFactor(self, grid: List[List[int]]) -> int:
        n = len(grid)
        min_distance = self.min_distance_to_thiefs(grid)
        if min_distance[0][0] == 0 or min_distance[n - 1][n - 1] == 0:
            return 0
        already_done = [[False] * n for _ in range(n)]
        already_done[n - 1][n - 1] = True
        best_positions = [(n - 1, n - 1)]
        safety_val = min_distance[n - 1][n - 1]
        # Loop over expansion rounds
        while True:
            cand_positions = [
                (npos, min(min_distance[npos[0]][npos[1]], safety_val))
                for pos in best_positions
                for npos in self.neighbors_not_yet_done(pos, already_done)
            ]
            # Only expand the best
            safety_val = max(c[1] for c in cand_positions)
            best_positions = [pos for pos, val in cand_positions if val == safety_val]
            for pos in best_positions:
                if pos == (0, 0):
                    return safety_val
                already_done[pos[0]][pos[1]] = True


# OK
class Solution_3020:
    """
    https://leetcode.com/problems/find-the-maximum-number-of-elements-in-subset/?envType=daily-question&envId=2026-08-25

    You are given an array of positive integers `nums`.

    You need to select a subset of `nums` which satisfies the following
    condition:

    * You can place the selected elements in a 0-indexed array such that it
      follows the pattern: `[x, x^2, x^4, ..., x^k/2, x^k, x^k/2, ..., x^4,
      x^2, x]` (Note that `k` can be any non-negative power of 2). For
      example, `[2, 4, 16, 4, 2]` and `[3, 9, 3]` follow the pattern while
      `[2, 4, 8, 4, 2]` does not.

    Return the maximum number of elements in a subset that satisfies these
    conditions.

    Example 1:

    Input: nums = [5,4,1,2,2]
    Output: 3
    Explanation: We can select the subset {4,2,2}, which can be placed in the array as [2,4,2] which follows the pattern and 22 == 4. Hence the answer is 3.

    Example 2:

    Input: nums = [1,3,2,4]
    Output: 1
    Explanation: We can select the subset {1}, which can be placed in the array as [1] which follows the pattern. Hence the answer is 1. Note that we could have also selected the subsets {2}, {3}, or {4}, there may be multiple subsets which provide the same answer.

    Constraints:

        2 <= nums.length <= 10^5
        1 <= nums[i] <= 10^9

    """

    def _len_for(
        self,
        x: int,
        at_least_once: Set[int],
        just_once: Set[int],
    ) -> int:
        z = x
        num = 1
        while True:
            z = z * z
            if z not in at_least_once:
                if z in just_once:
                    num += 1
                break
            num += 1
        return 2 * num - 1

    def maximumLength(self, nums: List[int]) -> int:
        counter = Counter(nums)
        # Pattern could be [1, 1, ..., 1] (odd length)
        num_1 = counter[1]
        if num_1 % 2 == 1:
            max_len = num_1
        else:
            max_len = 1  # Pattern [x] has length 1
        at_least_twice = {x for x, c in counter.items() if c >= 2}
        just_once = {x for x, c in counter.items() if c == 1}
        for x in at_least_twice:
            max_len = max(max_len, self._len_for(x, at_least_twice, just_once))
        return max_len


# OK
class Solution_3737:
    """
    https://leetcode.com/problems/count-subarrays-with-majority-element-i/?envType=daily-question&envId=2026-08-25

    You are given an integer array `nums` and an integer `target`.

    Return the number of subarrays (contiguous ranges) of `nums` in which
    `target` is the majority element.

    The majority element of a subarray is the element that appears strictly
    more than half of the times in that subarray.

    Example 1:

    Input: nums = [1,2,2,3], target = 2

    Output: 5

    Explanation:

    Valid subarrays with target = 2 as the majority element:

        nums[1..1] = [2]
        nums[2..2] = [2]
        nums[1..2] = [2,2]
        nums[0..2] = [1,2,2]
        nums[1..3] = [2,2,3]

    So there are 5 such subarrays.

    Example 2:

    Input: nums = [1,1,1,1], target = 1

    Output: 10

    Explanation:

    All 10 subarrays have 1 as the majority element.

    Example 3:

    Input: nums = [1,2,3], target = 4

    Output: 0

    Explanation:

    target = 4 does not appear in nums at all. Therefore, there cannot be any subarray where 4 is the majority element. Hence the answer is 0.

    Constraints:

        1 <= nums.length <= 1000
        1 <= nums[i] <= 10^9
        1 <= target <= 10^9

    """

    def _num_start_from(self, start: int, pos_tpos: int) -> int:
        result = 0
        for i, (tpos, tpos_next) in enumerate(
            zip(
                self.target_pos[pos_tpos:],
                self.target_pos[(pos_tpos + 1) :] + [self.len_nums],
            )
        ):
            # Array `nums[start:(tpos + 1)]`: `i + 1` equal to `target`
            # i + 1 > tpos + 1 - start - (i + 1) = tpos - start - i
            # <--> 2 * i >= tpos - start
            diff = 2 * i - (tpos - start)
            result += min(diff + 1, tpos_next - tpos)
        return result

    def countMajoritySubarrays(self, nums: List[int], target: int) -> int:
        # Idea:
        # - Loop over start positions `start`:
        #   - Step over positions where `nums[tpos] == target`
        #   - For each new `tpos`: How much ahead is count of `target` over
        #     count of others?
        self.target_pos = [i for i, x in enumerate(nums) if x == target]
        if not self.target_pos:
            return 0
        elif len(self.target_pos) == 1:
            return 1
        self.len_nums = len(nums)
        pos_tpos = 0
        num_subarrays = 0
        for start in range(self.len_nums):
            num_subarrays += self._num_start_from(start, pos_tpos)
            if start >= self.target_pos[pos_tpos]:
                pos_tpos += 1
                if pos_tpos == len(self.target_pos):
                    break
        return num_subarrays


# OK
class Solution_2948:
    """
    https://leetcode.com/problems/make-lexicographically-smallest-array-by-swapping-elements/?envType=daily-question&envId=2026-08-25

    You are given a 0-indexed array of positive integers `nums` and a positive
    integer `limit`.

    In one operation, you can choose any two indices i and j and swap `nums[i]` and
    `nums[j]` if `|nums[i] - nums[j]| <= limit`.

    Return the lexicographically smallest array that can be obtained by
    performing the operation any number of times.

    An array a is lexicographically smaller than an array b if in the first
    position where a and b differ, array a has an element that is less than
    the corresponding element in b. For example, the array `[2,10,3]` is
    lexicographically smaller than the array `[10,2,3]` because they differ
    at index 0 and `2 < 10`.

    Example 1:

    Input: nums = [1,5,3,9,8], limit = 2
    Output: [1,3,5,8,9]
    Explanation: Apply the operation 2 times:
    - Swap nums[1] with nums[2]. The array becomes [1,3,5,9,8]
    - Swap nums[3] with nums[4]. The array becomes [1,3,5,8,9]
    We cannot obtain a lexicographically smaller array by applying any more operations.
    Note that it may be possible to get the same result by doing different operations.

    Example 2:

    Input: nums = [1,7,6,18,2,1], limit = 3
    Output: [1,6,7,18,1,2]
    Explanation: Apply the operation 3 times:
    - Swap nums[1] with nums[2]. The array becomes [1,6,7,18,2,1]
    - Swap nums[0] with nums[4]. The array becomes [2,6,7,18,1,1]
    - Swap nums[0] with nums[5]. The array becomes [1,6,7,18,1,2]
    We cannot obtain a lexicographically smaller array by applying any more operations.

    Example 3:

    Input: nums = [1,7,28,19,10], limit = 3
    Output: [1,7,28,19,10]
    Explanation: [1,7,28,19,10] is the lexicographically smallest array we can obtain because we cannot apply the operation on any two indices.

    Constraints:

        1 <= nums.length <= 10^5
        1 <= nums[i] <= 10^9
        1 <= limit <= 10^9

    """

    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        # Idea: Bubble sort with restriction on swaps
        n = len(nums)
        if n == 1:
            return nums
        for i in range(n - 1):
            xl = nums[i]
            for j, xr in enumerate(nums[(i + 1) :]):
                if xr < xl <= xr + limit:
                    nums[j + i + 1] = xl
                    nums[i] = xr
                    xl = xr
        return nums


# OK
class Solution_2075:
    """
    https://leetcode.com/problems/decode-the-slanted-ciphertext/?envType=daily-question&envId=2026-08-25

    """

    def decodeCiphertext(self, encodedText: str, rows: int) -> str:
        len_text = len(encodedText)
        cols = len_text // rows
        assert len_text == rows * cols
        encoded = [x for x in encodedText]
        num_parts = cols - rows + 2
        decoded = []
        for j in range(num_parts):
            decoded.extend(encoded[j : len_text : (cols + 1)])
        return "".join(decoded).rstrip(" ")


# OK
class Solution_2126:
    """
    https://leetcode.com/problems/destroying-asteroids/?envType=daily-question&envId=2026-08-25

    You are given an integer `mass`, which represents the original mass of a
    planet. You are further given an integer array `asteroids`, where
    `asteroids[i]` is the mass of the ith asteroid.

    You can arrange for the planet to collide with the asteroids in any
    arbitrary order. If the mass of the planet is greater than or equal to
    the mass of the asteroid, the asteroid is destroyed and the planet gains
    the mass of the asteroid. Otherwise, the planet is destroyed.

    Return true if all asteroids can be destroyed. Otherwise, return false.

    Example 1:

    Input: mass = 10, asteroids = [3,9,19,5,21]
    Output: true
    Explanation: One way to order the asteroids is [9,19,5,3,21]:
    - The planet collides with the asteroid with a mass of 9. New planet mass: 10 + 9 = 19
    - The planet collides with the asteroid with a mass of 19. New planet mass: 19 + 19 = 38
    - The planet collides with the asteroid with a mass of 5. New planet mass: 38 + 5 = 43
    - The planet collides with the asteroid with a mass of 3. New planet mass: 43 + 3 = 46
    - The planet collides with the asteroid with a mass of 21. New planet mass: 46 + 21 = 67
    All asteroids are destroyed.

    Example 2:

    Input: mass = 5, asteroids = [4,9,23,4]
    Output: false
    Explanation:
    The planet cannot ever gain enough mass to destroy the asteroid with a mass of 23.
    After the planet destroys the other asteroids, it will have a mass of 5 + 4 + 9 + 4 = 22.
    This is less than 23, so a collision would not destroy the last asteroid.

    Constraints:

        1 <= mass <= 10^5
        1 <= asteroids.length <= 10^5
        1 <= asteroids[i] <= 10^5

    """

    def asteroidsDestroyed(self, mass: int, asteroids: List[int]) -> bool:
        sorted_asteroids = sorted(asteroids)
        cumsum = [mass]
        for x in sorted_asteroids[:-1]:
            mass += x
            cumsum.append(mass)
        return all(x >= y for x, y in zip(cumsum, sorted_asteroids))


# OK
class Solution_3751:
    """
    https://leetcode.com/problems/total-waviness-of-numbers-in-range-i/?envType=daily-question&envId=2026-08-25

    You are given two integers `num1` and `num2` representing an inclusive
    range `[num1, num2]`.

    The waviness of a number is defined as the total count of its peaks and
    valleys:

    * A digit is a peak if it is strictly greater than both of its immediate neighbors.
    * A digit is a valley if it is strictly less than both of its immediate neighbors.
    * The first and last digits of a number cannot be peaks or valleys.
    * Any number with fewer than 3 digits has a waviness of 0.

    Return the total sum of waviness for all numbers in the range `[num1, num2]`.

    Example 1:

    Input: num1 = 120, num2 = 130

    Output: 3

    Explanation:
    In the range [120, 130]:

        120: middle digit 2 is a peak, waviness = 1.
        121: middle digit 2 is a peak, waviness = 1.
        130: middle digit 3 is a peak, waviness = 1.
        All other numbers in the range have a waviness of 0.

    Thus, total waviness is 1 + 1 + 1 = 3.

    Example 2:

    Input: num1 = 198, num2 = 202

    Output: 3

    Explanation:
    In the range [198, 202]:

        198: middle digit 9 is a peak, waviness = 1.
        201: middle digit 0 is a valley, waviness = 1.
        202: middle digit 0 is a valley, waviness = 1.
        All other numbers in the range have a waviness of 0.

    Thus, total waviness is 1 + 1 + 1 = 3.

    Example 3:

    Input: num1 = 4848, num2 = 4848

    Output: 2

    Explanation:

    Number 4848: the second digit 8 is a peak, and the third digit 4 is a valley, giving a waviness of 2.

    Constraints:

        1 <= num1 <= num2 <= 10^5

    """

    def _digits(self, x: int) -> List[int]:
        return [int(c) for c in str(x)]

    def _peaks_and_range(
        self,
        digits: List[int],
    ) -> Tuple[int, Tuple[int, int]]:
        if len(digits) >= 3:
            num_peaks = sum(
                y > max(x, z) or y < min(x, z)
                for x, y, z in zip(digits[:-2], digits[1:-1], digits[2:])
            )
        else:
            num_peaks = 0
        x, y = digits[-2], digits[-1]
        if x < y:
            rng = (0, y)
        elif x > y:
            rng = (y + 1, 10)
        else:
            rng = (0, 0)  # empty
        return num_peaks, rng

    def totalWaviness(self, num1: int, num2: int) -> int:
        assert 1 <= num1 <= num2
        if num2 < 101:
            return 0
        curr_num = max(num1, 101)
        total_waviness = 0
        while curr_num <= num2:
            digits = self._digits(curr_num)
            num_peaks, rng1 = self._peaks_and_range(digits[:-1])
            a = digits[-1]
            b = curr_num - a
            rng2 = (a, min(10, num2 - b + 1))
            num_intersect = max(min(rng1[1], rng2[1]) - max(rng1[0], rng2[0]), 0)
            num_rem = min(num2 - curr_num + 1, 10 - a)
            total_waviness += num_peaks * num_rem + num_intersect
            curr_num = b + 10
        return total_waviness


# OK
class Solution_3532:
    """
    https://leetcode.com/problems/path-existence-queries-in-a-graph-i/?envType=daily-question&envId=2026-08-25

    You are given an integer `n` representing the number of nodes in a graph,
    labeled from 0 to `n - 1`.

    You are also given an integer array `nums` of length n sorted in
    non-decreasing order, and an integer `maxDiff`.

    An undirected edge exists between nodes i and j if the absolute difference
    between `nums[i]` and `nums[j]` is at most `maxDiff` (i.e.,
    `|nums[i] - nums[j]| <= maxDiff`).

    You are also given a 2D integer array queries. For each `queries[i] = [ui, vi]`,
    determine whether there exists a path between nodes `ui` and `vi`.

    Return a boolean array `answer`, where `answer[i]` is true if there
    exists a path between ui and vi in the ith query and false otherwise.

    Example 1:

    Input: n = 2, nums = [1,3], maxDiff = 1, queries = [[0,0],[0,1]]

    Output: [true,false]

    Explanation:

        Query [0,0]: Node 0 has a trivial path to itself.
        Query [0,1]: There is no edge between Node 0 and Node 1 because |nums[0] - nums[1]| = |1 - 3| = 2, which is greater than maxDiff.
        Thus, the final answer after processing all the queries is [true, false].

    Example 2:

    Input: n = 4, nums = [2,5,6,8], maxDiff = 2, queries = [[0,1],[0,2],[1,3],[2,3]]

    Output: [false,false,true,true]

    Explanation:

    The resulting graph is:

        Query [0,1]: There is no edge between Node 0 and Node 1 because |nums[0] - nums[1]| = |2 - 5| = 3, which is greater than maxDiff.
        Query [0,2]: There is no edge between Node 0 and Node 2 because |nums[0] - nums[2]| = |2 - 6| = 4, which is greater than maxDiff.
        Query [1,3]: There is a path between Node 1 and Node 3 through Node 2 since |nums[1] - nums[2]| = |5 - 6| = 1 and |nums[2] - nums[3]| = |6 - 8| = 2, both of which are within maxDiff.
        Query [2,3]: There is an edge between Node 2 and Node 3 because |nums[2] - nums[3]| = |6 - 8| = 2, which is equal to maxDiff.
        Thus, the final answer after processing all the queries is [false, false, true, true].

    Constraints:

        1 <= n == nums.length <= 10^5
        0 <= nums[i] <= 10^5
        nums is sorted in non-decreasing order.
        0 <= maxDiff <= 10^5
        1 <= queries.length <= 10^5
        queries[i] == [ui, vi]
        0 <= ui, vi < n

    """

    def _get_cluster_ranges(
        self,
        nums: List[int],
        maxDiff: int,
    ) -> List[Tuple[int, int]]:
        result = []
        start = 0
        for i, (a, b) in enumerate(zip(nums[:-1], nums[1:])):
            if b > a + maxDiff:
                result.append((start, i + 1))
                start = i + 1
        result.append((start, len(nums)))
        return result

    def pathExistenceQueries(
        self,
        n: int,
        nums: List[int],
        maxDiff: int,
        queries: List[List[int]],
    ) -> List[bool]:
        assert n == len(nums)
        cluster_ranges = self._get_cluster_ranges(nums, maxDiff)
        return [
            any(a <= u < b and a <= v < b for a, b in cluster_ranges)
            for u, v in queries
        ]


# OK
# - First try had glitch: Extend {1} -> {1, 2, 4} with edges (1,2), (1,4),
#   but ignored edge (2,4) which had lower distance!
class Solution_2492:
    """
    https://leetcode.com/problems/minimum-score-of-a-path-between-two-cities/?envType=daily-question&envId=2026-08-25

    You are given a positive integer n representing n cities numbered from 1
    to n. You are also given a 2D array `roads` where
    `roads[i] = [ai, bi, distancei]` indicates that there is a bidirectional
    road between cities ai and bi with a distance equal to `distancei`. The
    cities graph is not necessarily connected.

    The score of a path between two cities is defined as the minimum
    distance of a road in this path.

    Return the minimum possible score of a path between cities 1 and n.

    Note:

    * A path is a sequence of roads between two cities.
    * It is allowed for a path to contain the same road multiple times, and you can visit cities 1 and n multiple times along the path.
    * The test cases are generated such that there is at least one path between 1 and n.

    Example 1:

    Input: n = 4, roads = [[1,2,9],[2,3,6],[2,4,5],[1,4,7]]
    Output: 5
    Explanation: The path from city 1 to 4 with the minimum score is: 1 -> 2 -> 4. The score of this path is min(9,5) = 5.
    It can be shown that no other path has less score.

    Example 2:

    Input: n = 4, roads = [[1,2,2],[1,3,4],[3,4,7]]
    Output: 2
    Explanation: The path from city 1 to 4 with the minimum score is: 1 -> 2 -> 1 -> 3 -> 4. The score of this path is min(2,2,4,7) = 2.

    Constraints:

        2 <= n <= 10^5
        1 <= roads.length <= 10^5
        roads[i].length == 3
        1 <= ai, bi <= n
        ai != bi
        1 <= distancei <= 104
        There are no repeated edges.
        There is at least one path between 1 and n.

    """

    def _extend_connected_component(
        self,
        nodes: Set[int],
        edges: Dict[int, List[Tuple[int, int]]],
    ) -> Optional[int]:
        min_score = None
        extra_nodes = []
        for node in nodes:
            neighbors = edges.get(node)
            if neighbors is not None:
                for other, score in neighbors:
                    extra_nodes.append(other)
                    min_score = score if min_score is None else min(min_score, score)
                del edges[node]
        if extra_nodes:
            nodes.update(extra_nodes)
        return min_score

    def minScore(self, n: int, roads: List[List[int]]) -> int:
        edges: Dict[int, List[Tuple[int, int]]] = dict()
        for a, b, dist in roads:
            for src, trg in ((a, b), (b, a)):
                lst = edges.get(src)
                entry = (trg, dist)
                if lst is None:
                    edges[src] = [entry]
                else:
                    lst.append(entry)
        min_score = None
        nodes: Set[int] = {1}
        while True:
            score = self._extend_connected_component(nodes, edges)
            if score is None:
                # Component could not be extended
                break
            min_score = score if min_score is None else min(min_score, score)
        return min_score


# OK
class Solution_1358:
    """
    https://leetcode.com/problems/number-of-substrings-containing-all-three-characters/?envType=daily-question&envId=2026-08-25

    Given a string s consisting only of characters a, b and c.

    Return the number of substrings containing at least one occurrence of all
    these characters a, b and c.

    Example 1:

    Input: s = "abcabc"
    Output: 10
    Explanation: The substrings containing at least one occurrence of the
    characters a, b and c are:
    "abc", "abca", "abcab", "abcabc", "bca", "bcab", "bcabc", "cab", "cabc" and
    "abc" (again).

    Example 2:

    Input: s = "aaacb"
    Output: 3
    Explanation: The substrings containing at least one occurrence of the
    characters a, b and c are "aaacb", "aacb" and "acb".

    Example 3:

    Input: s = "abc"
    Output: 1

    Constraints:

        3 <= s.length <= 5 x 10^4
        s only consists of 'a', 'b' or 'c' characters.

    """

    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)
        start, end = 0, 3
        num_substrings = 0
        hist = Counter(s[:end])
        while len(hist) < 3:
            hist[s[end]] += 1
            end += 1
            if end == n:
                if len(hist) < 3:
                    return 0
        while True:
            # At this point:
            # s[start:end] has all 3
            # Note that end == n is allowed
            num_substrings += n - end + 1
            while hist[s[start]] > 1:
                hist[s[start]] -= 1
                start += 1
                num_substrings += n - end + 1
            elem = s[start]
            del hist[elem]
            start += 1
            if end == n:
                break
            while s[end] != elem:
                hist[s[end]] += 1
                end += 1
                if end == n:
                    break
            hist[elem] = 1
            end += 1
            if end == n:
                num_substrings += 1
                break
        return num_substrings


# OK:
# This was not easy!
# - Overlooked infinite recharge issue initially
# - Initial bug: Removed paths with 0 energy before checking if they visit the
#   final L
# General idea here:
# - Breadth-first traversal
# - Condense paths to final cell: Always represent best one there
# - Criterion for best: Heuristic, first num_l_collected, second num_energy
# - Ensure Ls are picked up once only
# - Avoid infinite recharging
#
# AI comments on solution (CoderPad):
# - BFS right idea, but state for table cannot just be position
# - State should be (pos, coll_path, num_energy)
# - With that state: Can just keep one entry, no need to order them
# - Infinite recharging: Don't allow to return to same recharge state. May have
#   to block several recharge states this way
#
# Lesson learned:
# - BFS is the right idea, but how to represent paths?
# - My "solution" is not bad: Smaller state table, but heuristic ordering may
#   fail
# - Study the constraints:
#   Limited number of L cells -> Can include coll_path in the state definition
#   Energy only up to 50 -> Can include num_energy in state definition
class Solution_3568:
    """
    https://leetcode.com/problems/minimum-moves-to-clean-the-classroom/?envType=daily-question&envId=2026-08-25

    You are given an m x n grid classroom where a student volunteer is tasked
    with cleaning up litter scattered around the room. Each cell in the grid is
    one of the following:

    * 'S': Starting position of the student
    * 'L': Litter that must be collected (once collected, the cell becomes empty)
    * 'R': Reset area that restores the student's energy to full capacity,
           regardless of their current energy level (can be used multiple times)
    * 'X': Obstacle the student cannot pass through
    * '.': Empty space

    You are also given an integer `energy`, representing the student's maximum
    energy capacity. The student starts with this energy from the starting
    position 'S'.

    Each move to an adjacent cell (up, down, left, or right) costs 1 unit of
    energy. If the energy reaches 0, the student can only continue if they are on
    a reset area 'R', which resets the energy to its maximum capacity energy.

    Return the minimum number of moves required to collect all litter items, or
    -1 if it's impossible.

    Example 1:

    Input: classroom = ["S.", "XL"], energy = 2

    Output: 2

    Explanation:

        The student starts at cell (0, 0) with 2 units of energy.
        Since cell (1, 0) contains an obstacle 'X', the student cannot move directly downward.
        A valid sequence of moves to collect all litter is as follows:
            Move 1: From (0, 0) → (0, 1) with 1 unit of energy and 1 unit remaining.
            Move 2: From (0, 1) → (1, 1) to collect the litter 'L'.
        The student collects all the litter using 2 moves. Thus, the output is 2.

    Example 2:

    Input: classroom = ["LS", "RL"], energy = 4

    Output: 3

    Explanation:

        The student starts at cell (0, 1) with 4 units of energy.
        A valid sequence of moves to collect all litter is as follows:
            Move 1: From (0, 1) → (0, 0) to collect the first litter 'L' with 1 unit of energy used and 3 units remaining.
            Move 2: From (0, 0) → (1, 0) to 'R' to reset and restore energy back to 4.
            Move 3: From (1, 0) → (1, 1) to collect the second litter 'L'.
        The student collects all the litter using 3 moves. Thus, the output is 3.

    Example 3:

    Input: classroom = ["L.S", "RXL"], energy = 3

    Output: -1

    Explanation:

    No valid path collects all 'L'.

    Constraints:

        1 <= m == classroom.length <= 20
        1 <= n == classroom[i].length <= 20
        classroom[i][j] is one of 'S', 'L', 'R', 'X', or '.'
        1 <= energy <= 50
        There is exactly one 'S' in the grid.
        There are at most 10 'L' cells in the grid.

    """
    def neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        y, x = pos
        result = []
        if x > 0 and self.classroom[y][x - 1] != "X":
            result.append((y, x - 1))
        if x < self.n - 1 and self.classroom[y][x + 1] != "X":
            result.append((y, x + 1))
        if y > 0 and self.classroom[y - 1][x] != "X":
            result.append((y - 1, x))
        if y < self.m - 1 and self.classroom[y + 1][x] != "X":
            result.append((y + 1, x))
        return result

    def do_step(
        self,
        paths: Dict[Tuple[int, int], Tuple[List[bool], int]],
        full_energy: int,
    ) -> Union[Dict[Tuple[int, int], Tuple[List[bool], int]], str]:
        new_paths = dict()
        for pos, vals in paths.items():
            coll_patt, num_energy = vals
            if all(coll_patt):
                return "success"  # We are done
            if num_energy > 0:
                num_done = sum(coll_patt)
                for npos in self.neighbors(pos):
                    cell = self.classroom[npos[0]][npos[1]]
                    _coll_patt = coll_patt
                    _num_energy = num_energy - 1
                    _num_done = num_done
                    if cell == "L":
                        lpos = self.litter_pos[npos]
                        if not coll_patt[lpos]:
                            # L not yet collected before
                            _coll_patt = [True if i == lpos else x for i, x in enumerate(coll_patt)]
                            _num_done += 1
                    elif cell == "R":
                        # Prevent infinite "recharging": Must make progress between
                        # recharging on the same rest place, i.e. pick at least one
                        # more litter
                        if num_done <= self.rest_pos[npos]:
                            continue
                        _num_energy = full_energy
                        self.rest_pos[npos] = num_done
                    _vals = (_coll_patt, _num_energy)
                    _vals2 = new_paths.get(npos)
                    if _vals2 is None:
                        new_paths[npos] = _vals
                    else:
                        # Keep better of the two
                        # Ordering is heuristic: num_done counts more than
                        # num_energy (except must have num_energy > 0)
                        _num_done2 = sum(_vals2[0])
                        if (_num_done, _num_energy) > (_num_done2, _vals2[1]):
                            new_paths[npos] = _vals

        if new_paths:
            return new_paths
        else:
            return "failure"

    def minMoves(self, classroom: List[str], energy: int) -> int:
        self.m = len(classroom)
        self.n = len(classroom[0])
        assert all(len(row) == self.n for row in classroom)
        assert energy > 0
        self.litter_pos = dict()
        num_litter = 0
        start_pos = None
        self.rest_pos = dict()
        for y, row in enumerate(classroom):
            for x, cell in enumerate(row):
                if cell == "S":
                    start_pos = (y, x)
                elif cell == "L":
                    self.litter_pos[(y, x)] = num_litter
                    num_litter += 1
                elif cell == "R":
                    self.rest_pos[(y, x)] = -1
        if num_litter == 0:
            return 0
        self.classroom = classroom
        # Data structure:
        # `paths[pos]` for positions `(y, x)`, contains tuple
        # `(coll_patt, num_energy)`. We proceed in steps. After `num_steps`
        # rounds, all paths represented by `paths` are this many steps long.
        # `coll_patt` is a boolean list of length `num_litter`, where we mark
        # which litter cells have been visited. This is needed, since otherwise
        # L cells could be counted several times.
        paths = {start_pos: ([False] * num_litter, energy)}
        num_steps = 0
        while True:
            paths = self.do_step(paths, energy)
            if isinstance(paths, str):
                return num_steps if paths == "success" else -1
            num_steps += 1


# OK
class Solution_1861:
    """
    https://leetcode.com/problems/rotating-the-box/?envType=daily-question&envId=2026-08-25

    You are given an m x n matrix of characters boxGrid representing a side-view
    of a box. Each cell of the box is one of the following:

    * A stone '#'
    * A stationary obstacle '*'
    * Empty '.'

    The box is rotated 90 degrees clockwise, causing some of the stones to fall
    due to gravity. Each stone falls down until it lands on an obstacle, another
    stone, or the bottom of the box. Gravity does not affect the obstacles'
    positions, and the inertia from the box's rotation does not affect the
    stones' horizontal positions.

    It is guaranteed that each stone in boxGrid rests on an obstacle, another
    stone, or the bottom of the box.

    Return an n x m matrix representing the box after the rotation described
    above.

    Example 1:

    Input: boxGrid = [["#",".","#"]]
    Output: [["."],
             ["#"],
             ["#"]]

    Example 2:

    Input: boxGrid = [["#",".","*","."],
                      ["#","#","*","."]]
    Output: [["#","."],
             ["#","#"],
             ["*","*"],
             [".","."]]

    Example 3:

    Input: boxGrid = [["#","#","*",".","*","."],
                      ["#","#","#","*",".","."],
                      ["#","#","#",".","#","."]]
    Output: [[".","#","#"],
             [".","#","#"],
             ["#","#","*"],
             ["#","*","."],
             ["#",".","*"],
             ["#",".","."]]

    Constraints:

        m == boxGrid.length
        n == boxGrid[i].length
        1 <= m, n <= 500
        boxGrid[i][j] is either '#', '*', or '.'.

    """
    def rotateRow(self, row_str: str) -> str:
        # Idea:
        # - "*" stay where they are
        # - Parts between "*" are ordered first "." then "#"
        # Also, we work on strings rather than lists of characters, because
        # Python is better with strings
        parts = row_str.split("*")
        new_parts = []
        for part in parts:
            len_part = len(part)
            num_stones = sum(x == "#" for x in part)
            new_parts.append("." * (len_part - num_stones) + "#" * num_stones)
        return "*".join(new_parts)

    def rotateTheBox(self, boxGrid: List[List[str]]) -> List[List[str]]:
        # Rotating by 90 degrees clockwise: Row i becomes column m - i - 1. That
        # is why we need to reverse.
        result_cols = reversed([self.rotateRow("".join(row)) for row in boxGrid])
        # Transpose of list of lists is done with zip, but need to convert tuples
        # into lists (as requested)
        return [list(l) for l in zip(*result_cols)]


# CHECK
# - Passes the tests
# - Can think of better solution?
class Solution_39:
    """
    https://leetcode.com/problems/combination-sum/

    Given an array of distinct integers `candidates` and a target integer `target`,
    return a list of all unique combinations of candidates where the chosen numbers
    sum to `target`. You may return the combinations in any order.

    The same number may be chosen from candidates an unlimited number of times. Two
    combinations are unique if the frequency of at least one of the chosen numbers
    is different.

    The test cases are generated such that the number of unique combinations that
    sum up to target is less than 150 combinations for the given input.

    Example 1:

    Input: candidates = [2,3,6,7], target = 7
    Output: [[2,2,3],[7]]
    Explanation:
    2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
    7 is a candidate, and 7 = 7.
    These are the only two combinations.

    Example 2:

    Input: candidates = [2,3,5], target = 8
    Output: [[2,2,2,2],[2,3,3],[3,5]]

    Example 3:

    Input: candidates = [2], target = 1
    Output: []

    Constraints:

        1 <= candidates.length <= 30
        2 <= candidates[i] <= 40
        All elements of candidates are distinct.
        1 <= target <= 40

    """
    def _combination_sum(self, end: int, target: int) -> List[List[int]]:
        while self.candidates[end - 1] > target:
            end -= 1
            if end == 0:
                return []
        cand_last = self.candidates[end - 1]
        if end == 1:
            return [] if target % cand_last != 0 else [[target // cand_last]]
        result = []
        for num_last in range(0, target // cand_last + 1):
            el_last = num_last * cand_last
            if el_last < target:
                result.extend(
                    l + [num_last]
                    for l in self._combination_sum(end - 1, target - el_last)
                )
            elif el_last == target:
                result.append([0] * (end - 1) + [num_last])
        return result

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        assert len(candidates) > 0
        self.candidates = candidates
        result_as_counts = self._combination_sum(len(candidates), target)
        return [
            [
                x
                for cand, num in zip(self.candidates, cvec)
                for x in [cand] * num
            ]
            for cvec in result_as_counts
        ]


# OK
# - My solution works
# - AI solution is just much simpler, but not obvious at all why it works,
#   so WTF!
#
# From AI:
# - If sum(delta) < 0: Solution cannot exist. For any start, we'd fail with the
#   final step
# - If sum(delta) >= 0: Solution must exist. Why? Say you start from 0. For some
#   position pos, sum(delta[:(pos+1)]) is smallest. Assume this is negative. If we
#   start from pos, cumulative sums from there on cannot be negative at least
#   until the end, because that would imply a smaller cum-sum value. In fact, the
#   sum from pos until the end must be >= -sum(delta[:(pos+1)]), since the total
#   sum(delta) >= 0. Also, sum(delta[:(i+1)]) >= sum(delta[:(pos+1)]) for any
#   i <= pos by definition of pos, and so this is a solution.
# - The rest of the AI solution is just trying to be clever. We could just as
#   well track the minimum cumulative sum.
class Solution_134:
    """
    https://leetcode.com/problems/gas-station/

    There are n gas stations along a circular route, where the amount of gas at
    the ith station is `gas[i]`.

    You have a car with an unlimited gas tank and it costs `cost[i]` of gas to
    travel from the ith station to its next (i + 1)th station. You begin the
    journey with an empty tank at one of the gas stations.

    Given two integer arrays `gas` and `cost`, return the starting gas station's
    index if you can travel around the circuit once in the clockwise direction,
    otherwise return -1. If there exists a solution, it is guaranteed to be unique.

    Example 1:

    Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
    Output: 3
    Explanation:
    Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
    Travel to station 4. Your tank = 4 - 1 + 5 = 8
    Travel to station 0. Your tank = 8 - 2 + 1 = 7
    Travel to station 1. Your tank = 7 - 3 + 2 = 6
    Travel to station 2. Your tank = 6 - 4 + 3 = 5
    Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
    Therefore, return 3 as the starting index.

    Example 2:

    Input: gas = [2,3,4], cost = [3,4,3]
    Output: -1
    Explanation:
    You can't start at station 0 or 1, as there is not enough gas to travel to the next station.
    Let's start at station 2 and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
    Travel to station 0. Your tank = 4 - 3 + 2 = 3
    Travel to station 1. Your tank = 3 - 3 + 3 = 3
    You cannot travel back to station 2, as it requires 4 unit of gas but you only have 3.
    Therefore, you can't travel around the circuit once no matter where you start.

    Constraints:

        n == gas.length == cost.length
        1 <= n <= 10^5
        0 <= gas[i], cost[i] <= 10^4
        The input is generated such that the answer is unique.

    """
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        delta = [g - c for g, c in zip(gas, cost)]
        # delta_cumsum[i] = sum(delta[:(i + 1)])
        delta_cumsum = [0] + list(accumulate(delta))
        # Starting from largest `delta` seems sensible
        for pos, _ in sorted(enumerate(delta), key=lambda x: x[1], reverse=True):
            if delta[pos] >= 0:
                # Need to ensure:
                # - sum(delta[pos:j]) >= 0 for all j in range(pos + 1, n + 1)
                # - sum(delta[pos:]) + sum(delta[:j]) >= 0 for all j in range(1, pos + 1)
                off = delta_cumsum[pos]
                if any(x < off for x in delta_cumsum[(pos + 1):]):
                    continue
                off = delta_cumsum[-1] - delta_cumsum[pos]
                if any(x < -off for x in delta_cumsum[:(pos + 1)]):
                    continue
                return pos
        return -1

    # AI solution: Why does this work?
    # ==> Sucks. Just being very clever here!
    def canCompleteCircuit_optimal(self, gas: List[int], cost: List[int]) -> int:
        total_tank = 0
        tank = 0
        start = 0
        for i in range(len(gas)):
            diff = gas[i] - cost[i]
            total_tank += diff
            tank += diff
            if tank < 0:
                start = i + 1
                tank = 0
        return start if total_tank >= 0 else -1

    def canCompleteCircuit_also_optimal(self, gas: List[int], cost: List[int]) -> int:
        len_gas = len(gas)
        total_tank = 0
        start = 0
        tank_after_start = 0
        for i in range(len_gas):
            diff = gas[i] - cost[i]
            total_tank += diff
            if total_tank < tank_after_start:
                start = i + 1
                tank_after_start = total_tank
        return start % len_gas if total_tank >= 0 else -1


# OK: This is a optimal solution
class Solution_1927:
    """
    https://leetcode.com/problems/sum-game/?envType=daily-question&envId=2026-08-25

    Alice and Bob take turns playing a game, with Alice starting first.

    You are given a string `num` of even length consisting of digits and '?'
    characters. On each turn, a player will do the following if there is still at
    least one '?' in `num`:

    * Choose an index i where `num[i] == '?'`.
    * Replace `num[i]` with any digit between '0' and '9'.

    The game ends when there are no more '?' characters in `num`.

    For Bob to win, the sum of the digits in the first half of `num` must be equal
    to the sum of the digits in the second half. For Alice to win, the sums must
    not be equal.

    For example, if the game ended with num = "243801", then Bob wins because
    2+4+3 = 8+0+1. If the game ended with num = "243803", then Alice wins because
    2+4+3 != 8+0+3.

    Assuming Alice and Bob play optimally, return true if Alice will win and
    false if Bob will win.

    Example 1:

    Input: num = "5023"
    Output: false
    Explanation: There are no moves to be made.
    The sum of the first half is equal to the sum of the second half: 5 + 0 = 2 + 3.

    Example 2:

    Input: num = "25??"
    Output: true
    Explanation: Alice can replace one of the '?'s with '9' and it will be
    impossible for Bob to make the sums equal.

    Example 3:

    Input: num = "?3295???"
    Output: false
    Explanation: It can be proven that Bob will always win. One possible outcome is:
    - Alice replaces the first '?' with '9'. num = "93295???".
    - Bob replaces one of the '?' in the right half with '9'. num = "932959??".
    - Alice replaces one of the '?' in the right half with '2'. num = "9329592?".
    - Bob replaces the last '?' in the right half with '7'. num = "93295927".
    Bob wins because 9 + 3 + 2 + 9 = 5 + 9 + 2 + 7.

    Constraints:

    * 2 <= num.length <= 10^5
    * num.length is even.
    * num consists of only digits and '?'.

    """
    def sumGame(self, num: str) -> bool:
        # Positions do not matter. Just state:
        #   (delta, free_left, free_right), where delta = sum_left - sum_right
        # Optimal approach is to play greedy:
        # - A maximizes `abs(delta)` with every move
        # - B minimizes `abs(delta)` with every move
        # - A plays 9 or 0 only, until the end
        #
        # Don't have to really play, but can predict outcome from start
        # state (delta, free_left, free_right)
        assert len(num) % 2 == 0
        half = len(num) // 2
        delta = free_left = free_right = 0
        for x in num[:half]:
            if x == "?":
                free_left += 1
            else:
                delta += int(x)
        for x in num[half:]:
            if x == "?":
                free_right += 1
            else:
                delta -= int(x)
        if delta == 0:
            # B wins if free_left == free_right: B always plays the same as A,
            # but on the other side.
            # A wins if free_left != free_right: A plays 9 on larger side, B
            # counters with 9 on other side, but last move is for A
            return free_left != free_right
        if delta < 0:
            # Flip everything around: Works by symmetry
            temp = free_left
            free_left, free_right = free_right, temp
            delta = -delta
        # At this point: delta > 0
        # If free_left > free_right: A wins (always 9 on left)
        # If free_left == free_right: A wins (always 9 on left)
        # If free_left < free_right: First, A and B play 9 left and right
        #   for free_left steps. Then, delta is the same and there are
        #   free_right - free_left right slots left.
        if free_left >= free_right:
            return True  # A wins (always 9 on the left)
        free_right -= free_left  # Free slots on the right only
        if free_right % 2 == 0:
            # B plays last
            # If delta > rhs: A wins by A:0, B:9 always
            # If delta < rhs: A wins by A:9, B:0 always
            # If delta == rhs: B wins by always playing 9 - A
            rhs = (free_right // 2) * 9
            return delta != rhs
        else:
            # A plays last
            # Same argument with delta vs rhs, but A can always move away
            # from 0 then
            return True


class Solution_3756:
    """
    https://leetcode.com/problems/concatenate-non-zero-digits-and-multiply-by-sum-ii/?envType=daily-question&envId=2026-08-25

    You are given a string s of length m consisting of digits. You are also given a 2D integer array queries, where queries[i] = [li, ri].

    For each queries[i], extract the s[li..ri]. Then, perform the following:

        Form a new integer x by concatenating all the non-zero digits from the substring in their original order. If there are no non-zero digits, x = 0.
        Let sum be the sum of digits in x. The answer is x * sum.

    Return an array of integers answer where answer[i] is the answer to the ith query.

    Since the answers may be very large, return them modulo 109 + 7.

    Example 1:

    Input: s = "10203004", queries = [[0,7],[1,3],[4,6]]

    Output: [12340, 4, 9]

    Explanation:

        s[0..7] = "10203004"
            x = 1234
            sum = 1 + 2 + 3 + 4 = 10
            Therefore, answer is 1234 * 10 = 12340.
        s[1..3] = "020"
            x = 2
            sum = 2
            Therefore, the answer is 2 * 2 = 4.
        s[4..6] = "300"
            x = 3
            sum = 3
            Therefore, the answer is 3 * 3 = 9.

    Example 2:

    Input: s = "1000", queries = [[0,3],[1,1]]

    Output: [1, 0]

    Explanation:

        s[0..3] = "1000"
            x = 1
            sum = 1
            Therefore, the answer is 1 * 1 = 1.
        s[1..1] = "0"
            x = 0
            sum = 0
            Therefore, the answer is 0 * 0 = 0.

    Example 3:

    Input: s = "9876543210", queries = [[0,9]]

    Output: [444444137]

    Explanation:

        s[0..9] = "9876543210"
            x = 987654321
            sum = 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1 = 45
            Therefore, the answer is 987654321 * 45 = 44444444445.
            We return 44444444445 modulo (109 + 7) = 444444137.

    Constraints:

        1 <= m == s.length <= 10^5
        s consists of digits only.
        1 <= queries.length <= 10^5
        queries[i] = [li, ri]
        0 <= li <= ri < m

     """
    pass


# === Hard ===


# HIER: This is hard!
class Solution_1872:
    """
    https://leetcode.com/problems/stone-game-viii/?envType=daily-question&envId=2026-08-25

    Alice and Bob take turns playing a game, with Alice starting first.

    There are `n` stones arranged in a row. On each player's turn, while the
    number of stones is more than one, they will do the following:

    * Choose an integer `x > 1`, and remove the leftmost `x` stones from the
      row.
    * Add the sum of the removed stones' values to the player's score.
    * Place a new stone, whose value is equal to that sum, on the left side
      of the row.

    The game stops when only one stone is left in the row.

    The score difference between Alice and Bob is (Alice's score - Bob's score).
    Alice's goal is to maximize the score difference, and Bob's goal is to
    minimize the score difference.

    Given an integer array stones of length `n` where `stones[i]` represents
    the value of the ith stone from the left, return the score difference
    between Alice and Bob if they both play optimally.

    Example 1:

    Input: stones = [-1,2,-3,4,-5]
    Output: 5
    Explanation:
    - Alice removes the first 4 stones, adds (-1) + 2 + (-3) + 4 = 2 to her score, and places a stone of
      value 2 on the left. stones = [2,-5].
    - Bob removes the first 2 stones, adds 2 + (-5) = -3 to his score, and places a stone of value -3 on
      the left. stones = [-3].
    The difference between their scores is 2 - (-3) = 5.

    Example 2:

    Input: stones = [7,-6,5,10,5,-2,-6]
    Output: 13
    Explanation:
    - Alice removes all stones, adds 7 + (-6) + 5 + 10 + 5 + (-2) + (-6) = 13 to her score, and places a
      stone of value 13 on the left. stones = [13].
    The difference between their scores is 13 - 0 = 13.

    Example 3:

    Input: stones = [-10,-12]
    Output: -22
    Explanation:
    - Alice can only make one move, which is to remove both stones. She adds (-10) + (-12) = -22 to her
      score and places a stone of value -22 on the left. stones = [-22].
    The difference between their scores is (-22) - 0 = -22.

    Constraints:

        n == stones.length
        2 <= n <= 105
        -104 <= stones[i] <= 104

    """

    def _signed_scorediff_for(
        self,
        start: int,
        val_first: int,
        player_sgn: int,
    ) -> int:
        curr_sum = val_first
        scores = []
        for i, x in enumerate(self.stones[start:-1]):
            curr_sum += x
            if curr_sum * player_sgn > 0:
                scores.append(
                    player_sgn
                    * (
                        curr_sum
                        - self._signed_scorediff_for(
                            start=start + i + 1,
                            val_first=curr_sum,
                            player_sgn=-player_sgn,
                        )
                    )
                )
        # Ending the game by taking the rest always needs to be considered:
        scores.append(player_sgn * (curr_sum + self.stones[-1]))
        return max(scores)

    def stoneGameVIII(self, stones: List[int]) -> int:
        self.stones = stones
        self.n = len(stones)
        return self._signed_scorediff_for(
            start=0,
            val_first=0,
            player_sgn=1,
        )


# OK (having done 3532 first helped!)
class Solution_3534:
    """
    https://leetcode.com/problems/path-existence-queries-in-a-graph-ii/description/?envType=daily-question&envId=2026-08-25

    You are given an integer n representing the number of nodes in a graph,
    labeled from 0 to n - 1.

    You are also given an integer array `nums` of length n and an integer
    `maxDiff`.

    An undirected edge exists between nodes i and j if the absolute difference
    between nums[i] and nums[j] is at most maxDiff (i.e.,
    |nums[i] - nums[j]| <= maxDiff).

    You are also given a 2D integer array queries. For each
    queries[i] = [ui, vi], find the minimum distance between nodes ui and vi.
    If no path exists between the two nodes, return -1 for that query.

    Return an array answer, where answer[i] is the result of the ith query.

    Note: The edges between the nodes are unweighted.

    Example 1:

    Input: n = 5, nums = [1,8,3,4,2], maxDiff = 3, queries = [[0,3],[2,4]]

    Output: [1,1]

    Explanation:

    The resulting graph is:

    Query	Shortest Path	Minimum Distance
    [0, 3]	0 → 3	1
    [2, 4]	2 → 4	1

    Thus, the output is [1, 1].

    Example 2:

    Input: n = 5, nums = [5,3,1,9,10], maxDiff = 2, queries = [[0,1],[0,2],[2,3],[4,3]]

    Output: [1,2,-1,1]

    Explanation:

    The resulting graph is:

    Query	Shortest Path	Minimum Distance
    [0, 1]	0 → 1	1
    [0, 2]	0 → 1 → 2	2
    [2, 3]	None	-1
    [4, 3]	3 → 4	1

    Thus, the output is [1, 2, -1, 1].

    Example 3:

    Input: n = 3, nums = [3,6,1], maxDiff = 1, queries = [[0,0],[0,1],[1,2]]

    Output: [0,-1,-1]

    Explanation:

    There are no edges between any two nodes because:

        Nodes 0 and 1: |nums[0] - nums[1]| = |3 - 6| = 3 > 1
        Nodes 0 and 2: |nums[0] - nums[2]| = |3 - 1| = 2 > 1
        Nodes 1 and 2: |nums[1] - nums[2]| = |6 - 1| = 5 > 1

    Thus, no node can reach any other node, and the output is [0, -1, -1].

    Constraints:

        1 <= n == nums.length <= 10^5
        0 <= nums[i] <= 10^5
        0 <= maxDiff <= 10^5
        1 <= queries.length <= 10^5
        queries[i] == [ui, vi]
        0 <= ui, vi < n

    """

    def _get_cluster_ranges(
        self,
        nums: List[int],
        maxDiff: int,
    ) -> List[Tuple[int, int]]:
        result = []
        start = 0
        for i, (a, b) in enumerate(zip(nums[:-1], nums[1:])):
            if b > a + maxDiff:
                result.append((start, i + 1))
                start = i + 1
        result.append((start, len(nums)))
        return result

    def pathExistenceQueries(
        self,
        n: int,
        nums: List[int],
        maxDiff: int,
        queries: List[List[int]],
    ) -> List[int]:
        assert n == len(nums)
        ind, sorted_nums = zip(
            *sorted(
                enumerate(nums),
                key=lambda x: x[1],
            )
        )
        remap = dict(enumerate(ind))
        cluster_ranges = self._get_cluster_ranges(sorted_nums, maxDiff)
        result = []
        for ou, ov in queries:
            if ou == ov:
                result.append(0)
                continue
            u = remap[ou]
            v = remap[ov]
            if u > v:
                u = v
                v = remap[ou]
            if any(a <= u and v < b for a, b in cluster_ranges):
                num_steps = 1
                num_u = sorted_nums[u]
                for w in range(u + 1, v + 1):
                    num_w = sorted_nums[w]
                    if num_w > num_u + maxDiff:
                        num_steps += 1
                        u = w - 1
                        num_u = sorted_nums[u]
                result.append(num_steps)
            else:
                result.append(-1)
        return result


# OK
# - My solution is VERY inefficient, scales linear in k
# - Mistake: Did not look at constraints! k can be very large, but coins cannot
#   (neither number of them, nor their values)
class Solution_3116:
    """
    https://leetcode.com/problems/kth-smallest-amount-with-single-denomination-combination/?envType=daily-question&envId=2026-08-25

    You are given an integer array `coins` representing coins of different
    denominations and an integer `k`.

    You have an infinite number of coins of each denomination. However, you are
    not allowed to combine coins of different denominations.

    Return the kth smallest amount that can be made using these coins.

    Example 1:

    Input: coins = [3,6,9], k = 3

    Output: 9

    Explanation: The given coins can make the following amounts:
    Coin 3 produces multiples of 3: 3, 6, 9, 12, 15, etc.
    Coin 6 produces multiples of 6: 6, 12, 18, 24, etc.
    Coin 9 produces multiples of 9: 9, 18, 27, 36, etc.
    All of the coins combined produce: 3, 6, 9, 12, 15, etc.

    Example 2:

    Input: coins = [5,2], k = 7

    Output: 12

    Explanation: The given coins can make the following amounts:
    Coin 5 produces multiples of 5: 5, 10, 15, 20, etc.
    Coin 2 produces multiples of 2: 2, 4, 6, 8, 10, 12, etc.
    All of the coins combined produce: 2, 4, 5, 6, 8, 10, 12, 14, 15, etc.

    Constraints:

        1 <= coins.length <= 15
        1 <= coins[i] <= 25
        1 <= k <= 2 * 10^9
        coins contains pairwise distinct integers.

    """

    def findKthSmallest(self, coins: List[int], k: int) -> int:
        next = [c for c in coins]
        rank = 0
        curr = None
        while rank < k:
            curr = next[0]
            mpos = []
            for pos, val in enumerate(next):
                if val < curr:
                    curr = val
                    mpos = [pos]
                elif val == curr:
                    mpos.append(pos)
            rank += 1
            for p in mpos:
                next[p] += coins[p]
        return curr

    # Solution from AI:
    # - countMultiplesUpTo(amount): Number of 1 <= x <= amount covered by any of the
    #   coins. Uses inclusion-exclusion principle, iterating over all binary masks
    #   of length number of coins
    # - Then use binary search to find amount s.t. countMultiplesUpTo(amount) == k.
    #   Here, amount <= min(coins) * k up front, which would mean that only the
    #   minimum coin is used.
    def findKthSmallest_muchbetter(self, coins: List[int], k: int) -> int:
        numCoins = len(coins)

        def countMultiplesUpTo(amount: int) -> int:
            # Inclusion-exclusion over all non-empty subsets of coins:
            # count of numbers <= amount divisible by lcm(subset), with sign
            # +1 for odd-sized subsets and -1 for even-sized subsets.
            totalCount = 0
            for mask in range(1, 1 << numCoins):
                subsetLcm = 1
                subsetSize = 0
                for i in range(numCoins):
                    if mask & (1 << i):
                        subsetLcm = subsetLcm * coins[i] // math.gcd(subsetLcm, coins[i])
                        subsetSize += 1
                sign = 1 if subsetSize % 2 == 1 else -1
                totalCount += sign * (amount // subsetLcm)
            return totalCount

        lowerBound, upperBound = 1, min(coins) * k
        while lowerBound < upperBound:
            midAmount = (lowerBound + upperBound) // 2
            if countMultiplesUpTo(midAmount) >= k:
                upperBound = midAmount
            else:
                lowerBound = midAmount + 1
        return lowerBound


# OK
# - My solution is standard-recursive, but can be very slow
# - First fix: Cache results in dict (memoization)
# - Even better: Standard bottom-up DP
class Solution_115:
    """
    https://leetcode.com/problems/distinct-subsequences/description/?envType=daily-question&envId=2026-08-25

    Given two strings s and t, return the number of distinct subsequences of s which equals t.

    The test cases are generated so that the answer fits on a 32-bit signed integer.

    Example 1:

    Input: s = "rabbbit", t = "rabbit"
    Output: 3
    Explanation:
    As shown below, there are 3 ways you can generate "rabbit" from s.
    rabbbit
    rabbbit
    rabbbit

    Example 2:

    Input: s = "babgbag", t = "bag"
    Output: 5
    Explanation:
    As shown below, there are 5 ways you can generate "bag" from s.
    babgbag
    babgbag
    babgbag
    babgbag
    babgbag

    Constraints:

        1 <= s.length, t.length <= 1000
        s and t consist of English letters.

    """
    def _num_distinct(self, start_s: int, start_t: int) -> int:
        cache_key = (start_s, start_t)
        cached = self.memory.get(cache_key)
        if cached is not None:
            return cached
        first_t = self.t[start_t]
        while self.s[start_s] != first_t:
            start_s += 1
            if start_s >= self.full_len_s:
                self.memory[cache_key] = 0
                return 0
        len_s = self.full_len_s - start_s
        len_t = self.full_len_t - start_t
        result = None
        if len_s < len_t:
            result = 0
        elif len_s == len_t:
            result = int(self.s[start_s:] == self.t[start_t:])
        elif len_t == 1:
            result = sum(x == first_t for x in self.s[(start_s + 1):]) + 1
        if result is None:
            result = self._num_distinct(start_s + 1, start_t + 1)
            for off in range(1, len_s - len_t + 1):
                if self.s[start_s + off] == first_t:
                    result += self._num_distinct(start_s + 1 + off, start_t + 1)
        self.memory[cache_key] = result
        return result

    def numDistinct(self, s: str, t: str) -> int:
        self.s = s
        self.t = t
        self.full_len_s = len(s)
        self.full_len_t = len(t)
        self.memory: Dict[Tuple[int, int], int] = dict()
        return self._num_distinct(0, 0)


