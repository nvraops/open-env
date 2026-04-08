from abc import ABC, abstractmethod
from typing import Dict
from env.models import Action, Reward, Label


class BaseGrader(ABC):
    """
    Abstract base grader
    All task-specific graders must inherit from this
    """

    @abstractmethod
    def grade(self, sample: Dict, action: Action) -> Reward:
        pass

    # ✅ Utility: label scoring
    def score_label(self, true_label: str, predicted_label: Label) -> float:
        if true_label == predicted_label.value:
            return 1.0
        return 0.0

    # ✅ Utility: confidence calibration
    def score_confidence(self, confidence: float, correct: bool) -> float:
        if correct:
            return confidence  # reward high confidence when correct
        else:
            return 1 - confidence  # penalize overconfidence

    # ✅ Utility: reasoning quality (basic heuristic)
    def score_reasoning(self, reasoning: str) -> float:
        length = len(reasoning.split())

        if length > 20:
            return 1.0
        elif length > 10:
            return 0.7
        elif length > 5:
            return 0.4
        else:
            return 0.1

    # ✅ Final aggregation
    def combine_scores(
        self,
        label_score: float,
        confidence_score: float,
        reasoning_score: float
    ) -> float:
        return (
            0.5 * label_score +
            0.3 * confidence_score +
            0.2 * reasoning_score
        )