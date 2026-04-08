from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


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
    score: float = Field(..., ge=0.0, le=1.0)
    feedback: str