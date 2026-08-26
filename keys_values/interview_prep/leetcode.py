from collections import defaultdict
from typing import List, Optional, Dict, Set


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
            postfix = self.find_subseq(word1[off:], word2[(skip_pos + 1):])
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
            sdiff1 = self.play_best_move(start + 1, end, not is_player1) + pl_sgn * elem1
            sdiff2 = self.play_best_move(start, end - 1, not is_player1) + pl_sgn * elem2
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
            score_me = self.cumsum[start + x] - self.cumsum[start] + all_rest - score_other
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
                    num_bought += (coins // cost)
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
        waterDuration: List[int]
    ) -> int:
        land_earliest_time = min(
            x + y for x, y in zip(landStartTime, landDuration)
        )
        water_earliest_time = min(
            x + y for x, y in zip(waterStartTime, waterDuration)
        )
        return min(
            min(
                max(land_earliest_time, x) + y
                for x, y in zip(waterStartTime, waterDuration)
            ),
            min(
                max(water_earliest_time, x) + y
                for x, y in zip(landStartTime, landDuration)
            )
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
                for pnext in list(range(1, pos - 1)) + list(range(pos + 2, self.max_jumps))
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
            if (
                sum_removed_rem1 and all(stones_rem2)
            ) or (
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
                stones_rem1=stones_rem1[:i] + stones_rem1[(i + 1):],
                stones_rem2=stones_rem2[:i] + stones_rem2[(i + 1):],
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
                stones_rem1=stones_rem1[:i] + stones_rem1[(i + 1):],
                stones_rem2=stones_rem2[:i] + stones_rem2[(i + 1):],
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
    def maximumSafenessFactor(self, grid: List[List[int]]) -> int:
        pass


# === Hard ===


class Solution_1872:
    """
    https://leetcode.com/problems/stone-game-viii/?envType=daily-question&envId=2026-08-25

    Alice and Bob take turns playing a game, with Alice starting first.

    There are n stones arranged in a row. On each player's turn, while the number of stones is more than one, they will do the following:

        Choose an integer x > 1, and remove the leftmost x stones from the row.
        Add the sum of the removed stones' values to the player's score.
        Place a new stone, whose value is equal to that sum, on the left side of the row.

    The game stops when only one stone is left in the row.

    The score difference between Alice and Bob is (Alice's score - Bob's score). Alice's goal is to maximize the score difference, and Bob's goal is the minimize the score difference.

    Given an integer array stones of length n where stones[i] represents the value of the ith stone from the left, return the score difference between Alice and Bob if they both play optimally.

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
    def stoneGameVIII(self, stones: List[int]) -> int:
        pass
