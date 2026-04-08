from pydantic import BaseModel
from typing import List

class MediumTaskAction(BaseModel):
    explanations: List[str]  # one explanation per statement

class MediumTaskObservation(BaseModel):
    statements: List[str]

class MediumTaskReward(BaseModel):
    reward: float

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
        reward = max(0.0, min(reward, 1.0))
        self.done = True
        return MediumTaskObservation(statements=self.statements), MediumTaskReward(reward=reward), self.done, {}