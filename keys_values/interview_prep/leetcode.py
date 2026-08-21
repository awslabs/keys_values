from typing import List, Optional


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
