from abc import ABC, abstractmethod
from typing import Dict
from env.models import Action, Reward, Label
<<<<<<< HEAD
from env.reward_policy import clamp_open_score, finalize_open_score, scale_score_to_band
=======
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc


class BaseGrader(ABC):
    """
    Abstract base grader
    All task-specific graders must inherit from this
    """

    def clamp_open_interval(self, value: float) -> float:
<<<<<<< HEAD
        return clamp_open_score(value)

    def finalize_score(self, value: float) -> float:
        return finalize_open_score(value)

    def scale_score_to_band(
        self,
        value: float,
        min_score: float,
        max_score: float,
    ) -> float:
        return scale_score_to_band(value, min_score, max_score)
=======
        return min(0.999, max(0.001, value))
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc

    @abstractmethod
    def grade(self, sample: Dict, action: Action) -> Reward:
        pass

    # ✅ Utility: label scoring
    def score_label(self, true_label: str, predicted_label: Label) -> float:
        if true_label == predicted_label.value:
            return 0.999
        return 0.001

    # ✅ Utility: confidence calibration
    def score_confidence(self, confidence: float, correct: bool) -> float:
        if correct:
            return self.clamp_open_interval(confidence)
        else:
            return self.clamp_open_interval(1 - confidence)

    # ✅ Utility: reasoning quality (basic heuristic)
    def score_reasoning(self, reasoning: str) -> float:
        length = len(reasoning.split())

        if length > 20:
<<<<<<< HEAD
            return 0.999
=======
            return 1.0
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc
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
        return self.clamp_open_interval(
            0.5 * label_score +
            0.3 * confidence_score +
            0.2 * reasoning_score
        )
