<<<<<<< HEAD
from pydantic import BaseModel, Field
from typing import List, Optional
from env.reward_policy import MAX_OPEN_SCORE, MIN_OPEN_SCORE, clamp_open_score
=======
from pydantic import BaseModel
from typing import List, Optional
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc

# Action model: the agent selects which statements are misinformation
class EasyTaskAction(BaseModel):
    selected_statements: List[int]  # indices of statements marked as false

# Observation model: the environment provides statements
class EasyTaskObservation(BaseModel):
    statements: List[str]

# Reward model
class EasyTaskReward(BaseModel):
<<<<<<< HEAD
    reward: float = Field(..., gt=0.0, lt=1.0, ge=MIN_OPEN_SCORE, le=MAX_OPEN_SCORE)
=======
    reward: float
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc

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
<<<<<<< HEAD
        reward = 0.001
=======
        reward = 0.0
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc
        for idx in action.selected_statements:
            if idx in self.misinformation_indices:
                reward += 0.5  # partial credit
            else:
                reward -= 0.25  # penalty for false positive

<<<<<<< HEAD
        reward = clamp_open_score(reward)
        self.done = True
        return EasyTaskObservation(statements=self.statements), EasyTaskReward(reward=reward), self.done, {}
=======
        reward = max(0.0, min(reward, 1.0))
        self.done = True
        return EasyTaskObservation(statements=self.statements), EasyTaskReward(reward=reward), self.done, {}
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc
