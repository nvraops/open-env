from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
<<<<<<< HEAD
from env.reward_policy import MAX_OPEN_SCORE, MIN_OPEN_SCORE
=======
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc


# ✅ Strict label control (VERY IMPORTANT for grading)
class Label(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    MISLEADING = "MISLEADING"


class Observation(BaseModel):
    """
    What the agent receives at each step
    """
    claim: str
    context: Optional[str] = ""
    history: List[str] = Field(default_factory=list)


class Action(BaseModel):
    """
    What the agent must output
    """
    label: Label
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=5)


class Reward(BaseModel):
    """
    Feedback from environment
    """
<<<<<<< HEAD
    score: float = Field(..., gt=0.0, lt=1.0, ge=MIN_OPEN_SCORE, le=MAX_OPEN_SCORE)
    feedback: str
=======
    score: float = Field(..., ge=0.0, le=1.0)
    feedback: str
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc
