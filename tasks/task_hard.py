<<<<<<< HEAD
from pydantic import BaseModel, Field
from typing import List
from env.reward_policy import MAX_OPEN_SCORE, MIN_OPEN_SCORE, clamp_open_score
=======
from pydantic import BaseModel
from typing import List
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc

class HardTaskAction(BaseModel):
    classifications: List[int]  # 0 = true, 1 = false

class HardTaskObservation(BaseModel):
    article: str
    statements: List[str]

class HardTaskReward(BaseModel):
<<<<<<< HEAD
    reward: float = Field(..., gt=0.0, lt=1.0, ge=MIN_OPEN_SCORE, le=MAX_OPEN_SCORE)
=======
    reward: float
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc

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
<<<<<<< HEAD
        reward = 0.001
=======
        reward = 0.0
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc
        for idx, cls in enumerate(action.classifications):
            if idx in self.misinformation_indices and cls == 1:
                reward += 0.33
            elif idx not in self.misinformation_indices and cls == 0:
                reward += 0.33
            else:
                reward -= 0.1
<<<<<<< HEAD
        reward = clamp_open_score(reward)
        self.done = True
        return HardTaskObservation(article=self.article, statements=self.statements), HardTaskReward(reward=reward), self.done, {}
=======
        reward = max(0.0, min(reward, 1.0))
        self.done = True
        return HardTaskObservation(article=self.article, statements=self.statements), HardTaskReward(reward=reward), self.done, {}
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc
