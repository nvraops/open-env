from typing import Dict
from env.models import Action, Reward
from graders.base import BaseGrader


class MediumGrader(BaseGrader):
    """
    Medium Task:
    - Includes MISLEADING cases
    - Requires context understanding
    """

    def grade(self, sample: Dict, action: Action) -> Reward:
        true_label = sample.get("label")
        context = sample.get("context", "").lower()
        reasoning = action.reasoning.lower()

        # ✅ Label scoring
        label_score = self.score_label(true_label, action.label)
        is_correct = true_label == action.label.value

        # ✅ Confidence scoring
        confidence_score = self.score_confidence(
            action.confidence,
            is_correct
        )

        # ✅ Context-aware reasoning scoring
        reasoning_score = self._context_reasoning_score(context, reasoning)

        # ✅ Final score
        final_score = self.combine_scores(
            label_score,
            confidence_score,
            reasoning_score
        )

        # ✅ Feedback
        if is_correct:
            feedback = "Correct with context understanding."
        else:
            feedback = (
                f"Incorrect. Expected {true_label}, got {action.label.value}. "
                f"Check context alignment."
            )

<<<<<<< HEAD
        final_score = self.scale_score_to_band(final_score, 0.35, 0.78)

        return Reward(
            score=final_score,
=======
        return Reward(
            score=round(final_score, 3),
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc
            feedback=feedback
        )

    def _context_reasoning_score(self, context: str, reasoning: str) -> float:
        """
        Check if reasoning uses context keywords
        """
        if not context:
            return self.score_reasoning(reasoning)

        context_words = set(context.split())
        reasoning_words = set(reasoning.split())

        overlap = context_words.intersection(reasoning_words)

        if len(overlap) > 5:
            return 0.999
        elif len(overlap) > 2:
            return 0.7
        elif len(overlap) > 0:
            return 0.4
        else:
            return 0.2
