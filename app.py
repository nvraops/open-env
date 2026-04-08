import os
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from env.core import MisinfoEnv
from env.models import Action
from graders.easy_grader import EasyGrader
from graders.hard_grader import HardGrader
from graders.medium_grader import MediumGrader
from inference import get_action, load_data

app = FastAPI(title="misinfo_env", version="1.0.0")


def _get_task_name() -> str:
    task_name = (os.getenv("TASK_NAME") or "easy").lower()
    return task_name if task_name in {"easy", "medium", "hard"} else "easy"


def _get_grader(task_name: str):
    if task_name == "easy":
        return EasyGrader()
    if task_name == "medium":
        return MediumGrader()
    return HardGrader()


TASK_NAME = _get_task_name()
ENV = MisinfoEnv(load_data(TASK_NAME), _get_grader(TASK_NAME))


class StepInput(BaseModel):
    observation: Any = None
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}


@app.get("/")
def home():
    return {"message": "API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "misinfo_env",
        "description": "OpenEnv environment for misinformation detection.",
    }


@app.get("/schema")
def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "context": {"type": "string"},
                "history": {"type": "array", "items": {"type": "string"}},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "current_index": {"type": "integer"},
                "done": {"type": "boolean"},
                "history": {"type": "array", "items": {"type": "string"}},
            },
        },
    }


@app.post("/reset")
@app.post("/openenv/reset")
@app.get("/openenv/reset")
def reset_env():
    observation = ENV.reset()
    return observation.model_dump()


@app.get("/state")
def state_env():
    return ENV.state_info()


@app.post("/step")
def step_env(action: Action):
    observation, reward, done, info = ENV.step(action)
    return {
        "observation": observation.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.post("/openenv/step")
def step_inference(input_data: StepInput):
    action = get_action(
        observation=input_data.observation,
        reward=input_data.reward,
        done=input_data.done,
        info=input_data.info,
    )
    return {"action": action}
