# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Duck Hunt VLM environment for the OpenEnv hackathon. The agent outputs tool calls `shoot(x, y, horizon)` where `horizon` is a latency-aware prediction window. Training uses Unsloth + TRL GRPO on Qwen3-VL-8B from pixels.

## Commands
```bash
# Install
pip install -e ".[server]"       # server deps
pip install -e ".[client]"       # client deps

# Run server
uvicorn duckhunt_env.server.app:app --host 0.0.0.0 --port 7860

# Smoke test
python -c "from duckhunt_env.client import DuckHuntEnv"

# Run reference tests (from horizon_min, use as model)
cd ~/horizon_min/duck_hunt_openenv && python -m pytest tests/ -v
```

## Architecture

### Reference Code (source of truth)
The complete working implementation lives at `~/horizon_min/duck_hunt_openenv/`:
- `server/game_engine.py` — `Duck`, `Match`, `Round` classes; port this
- `server/renderer.py` — Headless PIL renderer using sprite sheet; port this
- `server/environment.py` — `DuckHuntEnvironment` (plain class, not OpenEnv); adapt to OpenEnv
- `server/config.py` — All game constants
- `duck_hunt_env/models.py` — `ShootAction`, `DuckHuntObservation` dataclasses
- `duck_hunt_env/client.py` — HTTP client using `requests`
- `tests/` — `pytest` test suite; run against the new server

Training reference at `~/horizon_min/training/src/`:
- `reward.py` — `compute_reward()` with 5-signal reward
- `config.py` — `RewardConfig`, `FullConfig` dataclasses

### This Repo — Target Structure (to be built)
```
duckhunt_env/
  assets/          # sprites.png, background.jpg, arcadeclassic.ttf (already present)
  pyproject.toml   # package config (already present)
  server/
    app.py         # create_app(DuckHuntEnvironment) from openenv.core.env_server.app
    duckhunt_environment.py  # inherits openenv.core.env_server.interfaces.Environment
    game_engine.py # ported from ~/horizon_min (DO NOT change game logic)
    renderer.py    # ported from ~/horizon_min
  client/
    models.py      # DuckHuntAction (Pydantic v2), DuckHuntObservation (Pydantic v2)
    duckhunt_client.py  # HTTPEnvClient[DuckHuntAction, DuckHuntObservation]
training/
  reward.py        # Multi-signal reward function
  prompts.py       # System prompt for VLM
notebooks/
  train_duckhunt_grpo.ipynb  # Colab training notebook (required deliverable)
```

### Game Engine Hierarchy
`Round` → contains 5 `Match`es → each Match has 2 `Duck`s, 3 bullets, 30s timer.
- Game over when `total_misses >= 4`
- Duck speed scales with round number
- Ducks bounce off walls; random direction changes
- Shot detection: circle-based collision against hitbox center

### OpenEnv 0.2.1 Integration
- `DuckHuntEnvironment` MUST inherit `openenv.core.env_server.interfaces.Environment`
- App MUST call `create_app(DuckHuntEnvironment)` from `openenv.core.env_server.app`
- Client MUST use `HTTPEnvClient[DuckHuntAction, DuckHuntObservation]`

### Action Format (regex-critical)
```
shoot(x=0.42, y=0.31, horizon=3)
```
Regex: `r'shoot\(x=([\d.]+),\s*y=([\d.]+),\s*horizon=(\d+)\)'`
- `x`, `y`: float 0.0–1.0 (normalized screen coords)
- `horizon`: int 0–30 (frames to advance before shot resolves)
- Latency buckets (randomly assigned per episode): `[100, 200, 300, 400, 500, 600]` ms

### Reward Signals (5 total)
| Outcome | Reward |
|---|---|
| Hit | +1.0 |
| Double kill | +2.5 |
| Miss | -0.3 + proximity_bonus (0.5 × exp(-5.0 × dist)) |
| No target | -0.5 |
| Invalid format | -1.0 |
| Horizon penalty (hits only) | -0.1 × (horizon / 30) |

### Rendering
- All PIL rendering MUST be headless (no display dependency)
- Frame output size: 512×512 (resized from 800×500)
- Observations include 4 frames as base64 PNG strings
- Sprite sheet at `duckhunt_env/assets/sprites.png`; renderer imports from assets relative to `__file__`

## Key Constants
```python
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 500
FPS = 30
FRAMES_PER_OBSERVATION = 4
FRAME_OUTPUT_SIZE = (512, 512)
MAX_HORIZON = 30
LATENCY_OPTIONS_MS = [100, 200, 300, 400, 500, 600]
DUCKS_PER_MATCH = 2
BULLETS_PER_MATCH = 3
MATCHES_PER_ROUND = 5
MAX_MISSES = 4
```

## Custom Claude Skills
- `/build-server` — scaffold server + client files (app.py, environment, models, client)
- `/port-engine` — port game_engine.py + renderer.py from horizon_min
- `/build-reward` — build training/reward.py + training/prompts.py

## Stack
- Python 3.11 (venv at `./venv` uses 3.9 — use system Python 3.11 for new work)
- OpenEnv 0.2.1 (`openenv-core[core]>=0.2.1`)
- FastAPI + uvicorn, Pydantic v2, Pillow
- TRL ≥ 0.12 (GRPO), Unsloth (VLM fine-tuning), vLLM (inference)
