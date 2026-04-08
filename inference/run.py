# run.py
import json
import asyncio
from pathlib import Path
from typing import List, Optional, Any, Dict
import os
import sys

# Allow imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

from env.core import MisinfoEnv
from env.models import Action
from graders.easy_grader import EasyGrader
from graders.medium_grader import MediumGrader
from graders.hard_grader import HardGrader
from rewards.reward import RewardEngine

load_dotenv()

# -----------------------------
# Environment & model config
# -----------------------------
API_KEY = os.getenv("OPENAI_API_KEY") 
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"
TASK_NAME = os.getenv("TASK_NAME") or "all"
MAX_STEPS = int(os.getenv("MAX_STEPS") or 5)
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD") or 0.1)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# -----------------------------
# Load data & grader
# -----------------------------
def load_data(task_name: str):
    task_file = DATA_DIR / f"{task_name}.json"
    if not task_file.exists():
        raise FileNotFoundError(f"Task data not found: {task_file}")
    with task_file.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_grader(task_name: str):
    if task_name == "easy":
        return EasyGrader()
    if task_name == "medium":
        return MediumGrader()
    if task_name == "hard":
        return HardGrader()
    raise ValueError(f"Unknown task grader: {task_name}")

def warn_no_api_key():
    print("[WARN] No API key found. Using fallback responses.", flush=True)

# -----------------------------
# Logging helpers
# -----------------------------
def log_start(task: str, env_name: str, model: str):
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# -----------------------------
# Prompt builder
# -----------------------------
def build_prompt(step: int, last_claim: str, last_feedback: str, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return f"""
Step: {step}
Last claim: {last_claim!r}
Last feedback: {last_feedback}
Previous steps:
{history_block}
Provide your label and reasoning for the next claim.
""".strip()

# -----------------------------
# Mock model response
# -----------------------------
def get_model_response(client, prompt: str):
    # Fallback logic
    if "not" in prompt.lower():
        return {
            "label": "MISLEADING",
            "confidence": 0.7,
            "reasoning": "Detected negative claim pattern"
        }
    else:
        return {
            "label": "TRUE",
            "confidence": 0.7,
            "reasoning": "Seems factual"
        }

# -----------------------------
# Wrapper for FastAPI
# -----------------------------
def get_action(observation: Any, reward: float, done: bool, info: Dict[str, Any]):
    """
    Return an action string based on current step inputs.
    Can be called by /openenv/step endpoint.
    """
    last_claim = getattr(observation, "claim", "") if observation else ""
    last_feedback = ""
    history = []

    prompt = build_prompt(step=0, last_claim=last_claim, last_feedback=last_feedback, history=history)
    response = get_model_response(None, prompt)
    return response.get("label", "MISLEADING")

# -----------------------------
# Run tasks for CLI / local test
# -----------------------------
async def run_task(task_name: str, client, reward_engine):
    data = load_data(task_name)
    grader = get_grader(task_name)
    env = MisinfoEnv(data, grader)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=f"misinfo-detection-{task_name}", env_name="misinfo_env", model=MODEL_NAME)

    try:
        observation = env.reset()

        while not env.state.is_done() and steps_taken < MAX_STEPS:
            step = steps_taken + 1
            prompt = build_prompt(step, observation.claim, "", history)
            response = get_model_response(client, prompt)

            action = Action(
                label=response.get("label", "MISLEADING"),
                confidence=response.get("confidence", 0.5),
                reasoning=response.get("reasoning", "No reasoning provided")
            )

            observation, reward_value, done, info = env.step(action)
            reward = reward_engine.adjust_reward(reward_value, history)
            done = done or (step >= MAX_STEPS)

            log_step(step, action.label, reward, done, None)

            rewards.append(reward)
            history.append(f"Step {step}: {action.label} -> reward {reward:.2f}")
            steps_taken = step

        score = sum(rewards) / MAX_STEPS if MAX_STEPS > 0 else 0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print("ERROR:", str(e))

    finally:
        log_end(success, steps_taken, score, rewards)
        result = {
            "task": task_name,
            "status": "success" if success else "failed",
            "score": score,
            "steps": steps_taken
        }
        print(json.dumps(result))
        print()  # newline

async def main():
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL) if API_KEY else None
    if not API_KEY:
        warn_no_api_key()
    reward_engine = RewardEngine()

    tasks_to_run = ["easy", "medium", "hard"]
    if TASK_NAME in tasks_to_run:
        tasks_to_run = [TASK_NAME]

    for task_name in tasks_to_run:
        await run_task(task_name, client, reward_engine)

if __name__ == "__main__":
    asyncio.run(main())
