from typing import List


class RewardEngine:
    """
    Applies trajectory-level reward shaping
    (beyond per-step grading)
    """

    def __init__(self):
        self.previous_scores: List[float] = []

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

        # ✅ Clamp to [0, 1]
        adjusted = max(0.0, min(1.0, adjusted))

        # Track scores
        self.previous_scores.append(adjusted)

        return round(adjusted, 3)

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