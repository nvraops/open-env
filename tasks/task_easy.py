from pydantic import BaseModel, Field
from typing import List, Optional
from env.reward_policy import MAX_OPEN_SCORE, MIN_OPEN_SCORE, clamp_open_score

# Action model: the agent selects which statements are misinformation
class EasyTaskAction(BaseModel):
    selected_statements: List[int]  # indices of statements marked as false

# Observation model: the environment provides statements
class EasyTaskObservation(BaseModel):
    statements: List[str]

# Reward model
class EasyTaskReward(BaseModel):
    reward: float = Field(..., gt=0.0, lt=1.0, ge=MIN_OPEN_SCORE, le=MAX_OPEN_SCORE)

# The task environment
class EasyTask:
    def __init__(self):
        self.statements = [
            "The Earth is flat.",
            "Water boils at 100°C.",
            "Vaccines cause autism.",
            "Python is a programming language.",
        ]
        # indices of misinformation
        self.misinformation_indices = [0, 2]
        self.done = False

    def reset(self):
        self.done = False
        return EasyTaskObservation(statements=self.statements)

    def step(self, action: EasyTaskAction):
        reward = 0.001
        for idx in action.selected_statements:
            if idx in self.misinformation_indices:
                reward += 0.5  # partial credit
            else:
                reward -= 0.25  # penalty for false positive

        reward = clamp_open_score(reward)
        self.done = True
        return EasyTaskObservation(statements=self.statements), EasyTaskReward(reward=reward), self.done, {}
