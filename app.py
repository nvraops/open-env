from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

# 👇 IMPORT YOUR LOGIC
from inference.run import get_action   # <-- adjust if function name differs

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running"}

@app.get("/openenv/reset")
def reset_env():
    return {
        "status": "success",
        "message": "Environment reset successful"
    }

class StepInput(BaseModel):
    observation: Any = None
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}

@app.post("/openenv/step")
def step_env(input_data: StepInput):

    # 👇 CALL YOUR MODEL / LOGIC
    action = get_action(
        observation=input_data.observation,
        reward=input_data.reward,
        done=input_data.done,
        info=input_data.info
    )

    return {
        "action": action
    }
