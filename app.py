from fastapi import FastAPI
from pydantic import BaseModel
import json
from pathlib import Path

from env.core import MisinfoEnv
from env.models import Action
from graders.easy_grader import EasyGrader
from graders.medium_grader import MediumGrader
from graders.hard_grader import HardGrader

app = FastAPI()

# Global env state
env = None

DATA_DIR = Path(__file__).resolve().parent / "data"


# -------- Request Schema --------
class StepRequest(BaseModel):
    label: str
    confidence: float
    reasoning: str


# -------- Helpers --------
def load_data(task="easy"):
    file = DATA_DIR / f"{task}.json"
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_grader(task="easy"):
    if task == "easy":
        return EasyGrader()
    elif task == "medium":
        return MediumGrader()
    elif task == "hard":
        return HardGrader()
    else:
        return EasyGrader()


# -------- REQUIRED ENDPOINTS --------

@app.post("/openenv/reset")
def reset(task: str = "easy"):
    """
    Initializes environment
    """
    global env

    data = load_data(task)
    grader = get_grader(task)
    env = MisinfoEnv(data, grader)

    obs = env.reset()

    return {
        "observation": {
            "claim": obs.claim
        }
    }


@app.post("/openenv/step")
def step(req: StepRequest):
    """
    Takes one step in environment
    """
    global env

    if env is None:
        return {"error": "Environment not initialized. Call /reset first."}

    action = Action(
        label=req.label,
        confidence=req.confidence,
        reasoning=req.reasoning
    )

    obs, reward, done, info = env.step(action)

    return {
        "observation": {
            "claim": obs.claim
        },
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/")
def health():
    return {"status": "ok"}