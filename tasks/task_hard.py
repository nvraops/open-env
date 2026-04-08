from pydantic import BaseModel
from typing import List

class HardTaskAction(BaseModel):
    classifications: List[int]  # 0 = true, 1 = false

class HardTaskObservation(BaseModel):
    article: str
    statements: List[str]

class HardTaskReward(BaseModel):
    reward: float

class HardTask:
    def __init__(self):
        self.article = "Many myths about health circulate online..."
        self.statements = [
            "Drinking bleach can cure diseases.",
            "Washing hands prevents infections.",
            "5G towers spread viruses."
        ]
        self.misinformation_indices = [0, 2]
        self.done = False

    def reset(self):
        self.done = False
        return HardTaskObservation(article=self.article, statements=self.statements)

    def step(self, action: HardTaskAction):
        reward = 0.0
        for idx, cls in enumerate(action.classifications):
            if idx in self.misinformation_indices and cls == 1:
                reward += 0.33
            elif idx not in self.misinformation_indices and cls == 0:
                reward += 0.33
            else:
                reward -= 0.1
        reward = max(0.0, min(reward, 1.0))
        self.done = True
        return HardTaskObservation(article=self.article, statements=self.statements), HardTaskReward(reward=reward), self.done, {}