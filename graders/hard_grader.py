from typing import Dict
from env.models import Action, Reward
from graders.base import BaseGrader


class HardGrader(BaseGrader):
    """
    Hard Task:
    - Complex misinformation
    - Requires contradiction detection
    - Multi-signal reasoning evaluation
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

        # ✅ Advanced reasoning scoring
        reasoning_score = self._advanced_reasoning_score(
            context,
            reasoning,
            true_label
        )

        # ✅ Final score (slightly stricter weighting)
        final_score = (
            0.4 * label_score +
            0.3 * confidence_score +
            0.3 * reasoning_score
        )

        # 🚨 Penalty for contradiction
        contradiction_penalty = self._contradiction_penalty(
            context,
            reasoning,
            true_label
        )

        final_score = self.clamp_open_interval(final_score - contradiction_penalty)

        # ✅ Feedback
        if is_correct and contradiction_penalty == 0:
            feedback = "Strong reasoning and correct classification."
        elif contradiction_penalty > 0:
            feedback = "Reasoning contradicts known context."
        else:
            feedback = (
                f"Incorrect. Expected {true_label}, got {action.label.value}. "
                f"Improve reasoning depth."
            )

        return Reward(
            score=round(final_score, 3),
            feedback=feedback
        )

    def _advanced_reasoning_score(
        self,
        context: str,
        reasoning: str,
        true_label: str
    ) -> float:
        """
        Evaluate reasoning quality using:
        - Context overlap
        - Logical indicators
        """

        context_words = set(context.split())
        reasoning_words = set(reasoning.split())

        overlap = len(context_words.intersection(reasoning_words))

        logic_keywords = [
            "because", "therefore", "however",
            "evidence", "study", "data", "suggests"
        ]

        logic_score = sum(1 for word in logic_keywords if word in reasoning)

        # Combine signals
        if overlap > 5 and logic_score >= 2:
            return 0.999
        elif overlap > 3:
            return 0.7
        elif overlap > 1:
            return 0.5
        else:
            return 0.2

    def _contradiction_penalty(
        self,
        context: str,
        reasoning: str,
        true_label: str
    ) -> float:
        """
        Penalize if reasoning contradicts context
        """

        # Simple contradiction signals
        contradiction_words = ["no evidence", "false", "not true"]

        if true_label == "FALSE":
            for phrase in contradiction_words:
                if phrase in context and phrase not in reasoning:
                    return 0.2  # missed contradiction

        if true_label == "TRUE":
            if "false" in reasoning:
                return 0.2

        return 0.0
    
