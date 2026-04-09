from typing import List
from env.reward_policy import clamp_open_score, finalize_open_score


class RewardEngine:
    """
    Applies trajectory-level reward shaping
    (beyond per-step grading)
    """

    def __init__(self):
        self.previous_scores: List[float] = []

    def _clamp_open_interval(self, value: float) -> float:
        return clamp_open_score(value)

    def reset(self):
        self.previous_scores = []

    def adjust_reward(self, base_score: float, history: List[str]) -> float:
        """
        Modify reward based on behavior patterns
        """

        adjusted = base_score

        # ✅ Penalty: repetitive behavior
        if self._is_repetitive(history):
            adjusted -= 0.1

        # ✅ Bonus: improvement over time
        if self._is_improving(base_score):
            adjusted += 0.05

        # ✅ Clamp to strict (0, 1)
        adjusted = self._clamp_open_interval(adjusted)

        # Track scores
        self.previous_scores.append(adjusted)

        return finalize_open_score(adjusted)

    def _is_repetitive(self, history: List[str]) -> bool:
        """
        Detect repeated actions
        """
        if len(history) < 2:
            return False

        return history[-1] == history[-2]

    def _is_improving(self, current_score: float) -> bool:
        """
        Reward increasing performance
        """
        if not self.previous_scores:
            return False

        return current_score > self.previous_scores[-1]
