from typing import Dict
from env.models import Action, Reward
from graders.base import BaseGrader


class EasyGrader(BaseGrader):
    """
    Easy Task:
    - Clear TRUE/FALSE claims
    - Minimal ambiguity
    """

    def grade(self, sample: Dict, action: Action) -> Reward:
        true_label = sample.get("label")

        # ✅ Label correctness
        label_score = self.score_label(true_label, action.label)
        is_correct = true_label == action.label.value

        # ✅ Confidence scoring
        confidence_score = self.score_confidence(
            action.confidence,
            is_correct
        )

        # ✅ Reasoning scoring
        reasoning_score = self.score_reasoning(action.reasoning)

        # ✅ Final score
        final_score = self.combine_scores(
            label_score,
            confidence_score,
            reasoning_score
        )

        # ✅ Feedback
        if is_correct:
            feedback = "Correct classification."
        else:
            feedback = f"Incorrect. Expected {true_label}, got {action.label.value}."

        return Reward(
            score=round(final_score, 3),
            feedback=feedback
        )
