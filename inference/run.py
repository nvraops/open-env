# run.py
import json
import asyncio
from pathlib import Path
from typing import List, Optional, Any, Dict
import os
import sys

# Allow imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from env.core import MisinfoEnv
from env.models import Action
from env.reward_policy import clamp_open_score, finalize_open_score
from graders.easy_grader import EasyGrader
from graders.medium_grader import MediumGrader
from graders.hard_grader import HardGrader
from rewards.reward import RewardEngine

load_dotenv()

# -----------------------------
# Environment & model config
# -----------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
TASK_NAME = os.getenv("TASK_NAME") or "all"
MAX_STEPS = int(os.getenv("MAX_STEPS") or 5)
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD") or 0.1)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EMIT_JSON_RESULT = (os.getenv("EMIT_JSON_RESULT") or "0").lower() in {"1", "true", "yes"}


def clamp_open_interval(value: float) -> float:
    return clamp_open_score(value)


def finalize_public_score(value: float) -> float:
    return finalize_open_score(value)


def format_open_interval_2dp(value: float) -> str:
    display_value = min(0.99, max(0.01, round(clamp_open_interval(value), 2)))
    return f"{display_value:.2f}"


def resolve_api_config(
    base_url: str,
    openai_api_key: Optional[str],
    hf_token: Optional[str],
    model_name: str,
) -> tuple[str, Optional[str], str]:
    normalized = (base_url or "").lower()
    if openai_api_key and openai_api_key.startswith("sk-or-v1"):
        return "https://openrouter.ai/api/v1", openai_api_key, _normalize_model_name(
            model_name, provider="openrouter"
        )
    if "api.openai.com" in normalized:
        return normalized_base_url(base_url), openai_api_key, _normalize_model_name(
            model_name, provider="openai"
        )
    if "huggingface" in normalized or "hf.space" in normalized:
        return normalized_base_url(base_url), hf_token, _normalize_model_name(
            model_name, provider="huggingface"
        )
    api_key = openai_api_key or hf_token
    provider = "openrouter" if (openai_api_key or "").startswith("sk-or-v1") else "openai"
    return normalized_base_url(base_url), api_key, _normalize_model_name(
        model_name, provider=provider
    )


def normalized_base_url(base_url: str) -> str:
    return (base_url or "https://api.openai.com/v1").strip().rstrip("/")


def _normalize_model_name(model_name: str, provider: str) -> str:
    model_name = (model_name or "gpt-4.1-mini").strip()
    if provider == "openrouter" and "/" not in model_name and model_name.startswith("gpt-"):
        return f"openai/{model_name}"
    if provider == "openai" and model_name.startswith("openai/"):
        return model_name.split("/", 1)[1]
    return model_name

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

# -----------------------------
# Logging helpers
# -----------------------------
def log_start(task: str, env_name: str, model: str):
    print(f"[START] task={task} env={env_name} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={format_open_interval_2dp(reward)} "
        f"done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(format_open_interval_2dp(r) for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

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
# Model response
# -----------------------------
def _fallback_response(prompt: str):
    if "not" in prompt.lower():
        return {
            "label": "MISLEADING",
            "confidence": 0.7,
            "reasoning": "Detected negative claim pattern",
        }
    return {
        "label": "TRUE",
        "confidence": 0.7,
        "reasoning": "Seems factual",
    }


def get_model_response(client, prompt: str):
    if client is None:
        return _fallback_response(prompt)

    system_prompt = (
        "You classify misinformation claims. "
        "Reply with strict JSON containing label, confidence, and reasoning. "
        "label must be one of TRUE, FALSE, MISLEADING."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        label = str(parsed.get("label", "MISLEADING")).upper()
        if label not in {"TRUE", "FALSE", "MISLEADING"}:
            label = "MISLEADING"
        confidence = parsed.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(max(confidence, 0.0), 1.0)
        reasoning = str(parsed.get("reasoning", "No reasoning provided"))
        if len(reasoning) < 5:
            reasoning = "Model reasoning unavailable"
        return {
            "label": label,
            "confidence": confidence,
            "reasoning": reasoning,
        }
    except Exception:
        return _fallback_response(prompt)

# -----------------------------
# Wrapper for FastAPI
# -----------------------------
def get_action(observation: Any, reward: float, done: bool, info: Dict[str, Any]):
    """
    Return an action string based on current step inputs.
    Can be called by /openenv/step endpoint.
    """
    last_claim = getattr(observation, "claim", "") if observation else ""
    last_feedback = str(info.get("feedback", "")) if isinstance(info, dict) else ""
    history = []

    prompt = build_prompt(step=0, last_claim=last_claim, last_feedback=last_feedback, history=history)
    base_url, api_key, model_name = resolve_api_config(
        API_BASE_URL, OPENAI_API_KEY, HF_TOKEN, MODEL_NAME
    )
    client = OpenAI(api_key=api_key, base_url=base_url) if (OpenAI and api_key and base_url) else None
    original_model_name = MODEL_NAME
    try:
        globals()["MODEL_NAME"] = model_name
        response = get_model_response(client, prompt)
    finally:
        globals()["MODEL_NAME"] = original_model_name
    return response.get("label", "MISLEADING")

# -----------------------------
# Run tasks for CLI / local test
# -----------------------------
async def run_task(task_name: str, client, reward_engine):
    data = load_data(task_name)
    grader = get_grader(task_name)
    env = MisinfoEnv(data, grader)
    reward_engine.reset()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_feedback = ""
    available_steps = min(MAX_STEPS, len(data))

    log_start(task=task_name, env_name="misinfo_env", model=MODEL_NAME)

    try:
        observation = env.reset()

        while not env.state.is_done() and steps_taken < MAX_STEPS:
            step = steps_taken + 1
            prompt = build_prompt(step, observation.claim, last_feedback, history)
            response = get_model_response(client, prompt)

            action = Action(
                label=response.get("label", "MISLEADING"),
                confidence=response.get("confidence", 0.5),
                reasoning=response.get("reasoning", "No reasoning provided")
            )

            observation, reward_value, done, info = env.step(action)
            reward = reward_engine.adjust_reward(reward_value, history)
            done = done or (step >= MAX_STEPS)
            last_feedback = str(info.get("feedback", "")) if isinstance(info, dict) else ""

            log_step(step, action.label, reward, done, None)

            rewards.append(reward)
            history.append(f"Step {step}: {action.label} -> reward {reward:.3f}")
            steps_taken = step

        raw_score = sum(rewards) / available_steps if available_steps > 0 else 0.001
        score = finalize_public_score(raw_score)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        score = finalize_public_score(score)
        print("ERROR:", str(e))

    finally:
        log_end(success, steps_taken, score, rewards)
        result = {
            "task": task_name,
            "status": "success" if success else "failed",
            "score": finalize_public_score(score),
            "steps": steps_taken
        }
        if EMIT_JSON_RESULT:
            print(json.dumps(result))
        print()  # newline

async def main():
    global API_BASE_URL, MODEL_NAME
    if OpenAI is None:
        raise RuntimeError("openai package is required for inference execution")
    API_BASE_URL, api_key, MODEL_NAME = resolve_api_config(
        API_BASE_URL, OPENAI_API_KEY, HF_TOKEN, MODEL_NAME
    )
    if api_key is None:
        raise ValueError("Missing API key for configured API_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=API_BASE_URL)
    reward_engine = RewardEngine()

    tasks_to_run = ["easy", "medium", "hard"]
    if TASK_NAME in tasks_to_run:
        tasks_to_run = [TASK_NAME]

    for task_name in tasks_to_run:
        await run_task(task_name, client, reward_engine)

if __name__ == "__main__":
    asyncio.run(main())
