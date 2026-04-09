---
title: Open Env 07 04 2026
emoji: 📈
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
---

<<<<<<< HEAD
# Misinfo OpenEnv

An OpenEnv environment for misinformation classification across `easy`, `medium`, and `hard` tasks.

## What Was Fixed

This project was updated to satisfy the Meta/OpenEnv Phase 2 validation rule that all task scores must be strictly inside `(0, 1)`.

The code now guarantees:

- no reward returns exactly `0.0`
- no reward returns exactly `1.0`
- grader outputs are clamped to `0.001` through `0.999`
- task reward outputs are clamped to `0.001` through `0.999`
- inference summary scores are also clamped to the open interval

## Key Files

- `graders/base.py`: shared clamp logic and safe score finalization
- `graders/easy_grader.py`
- `graders/medium_grader.py`
- `graders/hard_grader.py`
- `tasks/task_easy.py`
- `tasks/task_medium.py`
- `tasks/task_hard.py`
- `rewards/reward.py`: safe reward shaping after rounding
- `inference/run.py`: provider-aware API key resolution and safe summary scoring

## Inference Configuration

Set your credentials in `.env`.

### Option 1: OpenAI

```env
OPENAI_API_KEY=sk-...
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
```

### Option 2: OpenRouter-style key

```env
OPENAI_API_KEY=sk-or-v1-...
MODEL_NAME=gpt-4o-mini
```

`inference/run.py` now detects `sk-or-v1` keys automatically and routes them to:

```text
https://openrouter.ai/api/v1
```

It also normalizes the model name to `openai/gpt-4o-mini` when needed.

## Run Locally

```powershell
python inference/run.py
```

Typical output format:

```text
[START] task=easy env=misinfo_env model=openai/gpt-4o-mini
[STEP] step=1 action=Label.TRUE reward=0.99 done=false error=null
[STEP] step=2 action=Label.FALSE reward=0.99 done=false error=null
[STEP] step=3 action=Label.FALSE reward=0.99 done=true error=null
[END] success=true steps=3 score=0.997 rewards=0.99,0.99,0.99
```

Set `EMIT_JSON_RESULT=1` if you also want the optional JSON task summary line.

## Validation Note

The important requirement for submission is:

```text
0 < score < 1
```

This repository now enforces that rule across graders, task rewards, shaped rewards, and final inference scores.
=======
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> b8610d1af8aceffc20032bfb7d83086f6cf268dc
