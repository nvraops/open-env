<<<<<<< HEAD
from pydantic import BaseModel, Field
from typing import List
from env.reward_policy import MAX_OPEN_SCORE, MIN_OPEN_SCORE, clamp_open_score
=======
from pydantic import BaseModel
from typing import List
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc

class MediumTaskAction(BaseModel):
    explanations: List[str]  # one explanation per statement

class MediumTaskObservation(BaseModel):
    statements: List[str]

class MediumTaskReward(BaseModel):
<<<<<<< HEAD
    reward: float = Field(..., gt=0.0, lt=1.0, ge=MIN_OPEN_SCORE, le=MAX_OPEN_SCORE)
=======
    reward: float
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc

class MediumTask:
    def __init__(self):
        self.statements = [
            "Eating chocolate cures COVID-19.",
            "The sun revolves around the Earth."
        ]
        self.correct_explanations = [
            "No scientific evidence; COVID-19 is viral.",
            "Incorrect; Earth revolves around the Sun."
        ]
        self.done = False

    def reset(self):
        self.done = False
        return MediumTaskObservation(statements=self.statements)

    def step(self, action: MediumTaskAction):
<<<<<<< HEAD
        reward = 0.001
=======
        reward = 0.0
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc
        for a, c in zip(action.explanations, self.correct_explanations):
            if c.lower() in a.lower():
                reward += 0.5
            else:
                reward -= 0.25
<<<<<<< HEAD
        reward = clamp_open_score(reward)
        self.done = True
        return MediumTaskObservation(statements=self.statements), MediumTaskReward(reward=reward), self.done, {}
=======
        reward = max(0.0, min(reward, 1.0))
        self.done = True
        return MediumTaskObservation(statements=self.statements), MediumTaskReward(reward=reward), self.done, {}
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc
