from pydantic import BaseModel, Field
from typing import List
from env.reward_policy import MAX_OPEN_SCORE, MIN_OPEN_SCORE, clamp_open_score

class MediumTaskAction(BaseModel):
    explanations: List[str]  # one explanation per statement

class MediumTaskObservation(BaseModel):
    statements: List[str]

class MediumTaskReward(BaseModel):
    reward: float = Field(..., gt=0.0, lt=1.0, ge=MIN_OPEN_SCORE, le=MAX_OPEN_SCORE)

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
        reward = 0.0
        for a, c in zip(action.explanations, self.correct_explanations):
            if c.lower() in a.lower():
                reward += 0.5
            else:
                reward -= 0.25
        reward = clamp_open_score(reward)
        self.done = True
        return MediumTaskObservation(statements=self.statements), MediumTaskReward(reward=reward), self.done, {}
