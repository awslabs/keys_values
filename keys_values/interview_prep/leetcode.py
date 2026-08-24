from typing import List, Optional, Dict


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


class Solution_3629:
    """
    https://leetcode.com/problems/minimum-jumps-to-reach-end-via-prime-teleportation/?envType=daily-question&envId=2026-08-20

    You are given an integer array nums of length n.

    You start at index 0, and your goal is to reach index n - 1.

    From any index i, you may perform one of the following operations:

    * Adjacent Step: Jump to index i + 1 or i - 1, if the index is within bounds.
    * Prime Teleportation: If nums[i] is a p, you may instantly jump to any index j != i such that nums[j] % p == 0.

    Return the minimum number of jumps required to reach index n - 1.

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
    def minJumps(self, nums: List[int]) -> int:
        pass  # TODO!


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
        pass  # TODO!


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
        pass  # TODO!
