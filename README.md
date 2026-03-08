# Duck Hunt VLM — Horizon Minimization via GRPO

A language model learns to play Duck Hunt — and discovers that **shooting sooner beats predicting further**.

The model outputs `shoot(x, y, horizon)` where `horizon` controls how many frames into the future it leads its shot. A larger horizon gives the model more time to predict where a duck is going, but ducks bounce randomly off walls — so longer predictions accumulate more error. The training reward penalizes large horizons on successful hits, pushing the model toward the **minimum prediction window** needed per shot.

This is built on top of **[OpenEnv 0.2.1](https://openenv.dev)** — a standardized protocol for agentic game environments.

**Training in progress — target 60%+ hit rate after 500 GRPO steps.**

> **[Trained Model](https://huggingface.co/vineethsaivs/duckhunt-qwen35-grpo)** · Base model: [Qwen3-8B](https://huggingface.co/unsloth/Qwen3-8B)
> **[W&B Training Run](https://wandb.ai/vineethsaivs-university-of-the-pacific/duckhunt-grpo)**

---

## The Challenge

Duck Hunt is deceptively hard for an AI. Ducks move fast, bounce unpredictably off screen edges, and the model must account for its own processing latency — the time between seeing frames and the shot actually landing. At 300ms latency (9 frames at 30 FPS), a duck traveling at 6 pixels/frame has already moved 54 pixels by the time the shot arrives. **The model has to lead its shots.**

The horizon tradeoff makes it harder still. The model can wait longer for a clearer trajectory prediction — but every extra frame of prediction is a frame where the duck might bounce and invalidate that prediction. A model that masters this learns to adapt per shot: short horizons for straight-flying ducks, longer ones near screen edges, and different horizons across varying latency conditions.

---

## What's Novel

This project makes three original contributions on top of the reference architecture:

**1. Latency-aware action space**
The action `shoot(x, y, horizon)` explicitly encodes the model's belief about how far ahead to aim. The current network latency (in ms) is included in every observation, forcing the model to reason about it explicitly rather than absorbing it as implicit bias.

**2. Horizon minimization reward**
Successful hits are penalized proportionally to the horizon used: `-0.1 × (horizon / 30)`. This is analogous to a surgeon being rewarded for minimal incision size — the model must learn not just *how* to hit, but *how quickly* it can commit to a shot. Hence the name: horizon minimization.

**3. OpenEnv 0.2.1 integration**
The Duck Hunt environment is built as a first-class OpenEnv `Environment[DuckHuntAction, DuckHuntObservation, State]`, with full snapshot/restore support for deterministic reward computation and a WebSocket client for stateful multi-step episodes.

---

## How It Works

```
Game State (text description + metadata)
         │
         ▼
   Qwen3-8B (LoRA rank 16)
         │
         ▼
  shoot(x=0.42, y=0.31, horizon=3)
         │
         ▼
  DuckHuntEnvironment.step()
         │
         ├── hit?    →  +1.0 reward  −  0.1×(horizon/30) horizon penalty
         ├── miss?   →  −0.3  +  0.5×exp(−5×distance) proximity bonus
         └── invalid → −1.0
         │
         ▼
   GRPO update (G=4 completions per state)
```

The training pipeline uses **Group Relative Policy Optimization (GRPO)** — the same RL algorithm used to train DeepSeek-R1. For each game state, 4 completions are sampled, scored against the environment, and the model is updated toward better-rewarded outputs using a clipped surrogate objective.

---

## Action Space

The model outputs a single tool call:

```python
shoot(x=0.42, y=0.31, horizon=3)
```

| Parameter | Type | Range | Description |
|-----------|------|--------|-------------|
| `x` | float | 0.0–1.0 | Horizontal screen position (0=left, 1=right) |
| `y` | float | 0.0–1.0 | Vertical screen position (0=top, 1=bottom) |
| `horizon` | int | 0–30 | Extra frames to simulate before shot resolves |

The total prediction lookahead is `latency_frames + horizon`. The model must learn to pick `horizon` based on the `latency_ms` field in the observation.

---

## Reward Function

| Outcome | Reward | Notes |
|---------|--------|-------|
| Hit one duck | +1.0 | Primary signal |
| Double kill | +2.5 | Both ducks hit in same step |
| Horizon penalty | −0.1 × (h/30) | Applied on hits — penalizes lazy high-horizon shots |
| Miss | −0.3 + 0.5×exp(−5d) | Graduated: near misses penalized less than wild shots |
| No target | −0.5 | Shot fired when no duck is visible |
| Invalid output | −1.0 | Model failed to produce parseable shoot() call |

Two reward signals are combined: **accuracy** (did it hit?) and **format** (is the output a valid tool call?). The format signal provides early training signal before the model learns to aim.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING LAYER (H100)                      │
│   Qwen3-8B + GRPO + Unsloth + TRL GRPOTrainer               │
│   Reward: duckhunt_accuracy + duckhunt_format                │
└──────────────────────┬──────────────────────────────────────┘
                       │  HTTP / WebSocket
┌──────────────────────▼──────────────────────────────────────┐
│               OPENENV SERVER LAYER  (port 7860)              │
│   DuckHuntEnvironment(Environment[Action, Obs, State])       │
│   FastAPI · /reset · /step · /state · /health                │
└──────────────────────┬──────────────────────────────────────┘
                       │  Python
┌──────────────────────▼──────────────────────────────────────┐
│                   GAME ENGINE LAYER                          │
│   DuckHuntGame: reset, advance_frame, shoot()               │
│   Renderer: headless PIL, 4 frames per observation          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                     ASSETS LAYER                             │
│   sprites.png · background.jpg · crosshairs.png             │
│   arcadeclassic.ttf                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Training Setup

| Config | Value |
|--------|-------|
| Base model | [unsloth/Qwen3-8B](https://huggingface.co/unsloth/Qwen3-8B) |
| Fine-tuning | LoRA rank 16, bf16, ~0.53% of parameters |
| Dataset | 200 game-state snapshots collected at training start |
| GRPO completions | G=4 per state |
| Batch size | 2 per device × 4 gradient accumulation = 8 effective |
| Learning rate | 5e-6, cosine schedule |
| Training steps | 500 |
| Latency buckets | 100ms, 200ms, 300ms, 400ms, 500ms, 600ms |
| Infrastructure | NVIDIA H100 80GB (Northflank) |
| Logging | [Weights & Biases](https://wandb.ai/vineethsaivs-university-of-the-pacific/duckhunt-grpo) |

---

## Quick Start

```bash
# Clone
git clone https://github.com/vineethsaivs/openenv-duckhunt-horizon-min
cd openenv-duckhunt-horizon-min

# Install
pip install -e .
pip install fastapi uvicorn pillow pydantic httpx openenv-core

# Start the Duck Hunt environment server
python -m uvicorn duckhunt_env.server.app:app --port 7860

# In another terminal — try a shot
curl -X POST http://localhost:7860/reset
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"x": 0.5, "y": 0.3, "horizon": 2, "latency_bucket": 200}'
```

### Run Training (H100 recommended)

```bash
# Install H100 dependencies
pip install -r requirements-h100.txt

# Train
python training/train_grpo.py \
  --max-steps 500 \
  --output-dir outputs/duckhunt_grpo \
  --hub-model-id your-username/duckhunt-qwen3-grpo
```

### Use the Client

```python
from duckhunt_env.client.duckhunt_client import DuckHuntEnv

env = DuckHuntEnv.from_local(host="localhost", port=7860)
obs = env.reset()

while not obs.done:
    # Model decides where to shoot and how far to lead
    action = {"x": 0.45, "y": 0.30, "horizon": 3}
    obs = env.step(action)
    print(f"Result: {obs.last_action_result}, Reward: {obs.reward:.2f}")

env.close()
```

---

## Project Structure

```
openenv-duckhunt-horizon-min/
├── duckhunt_env/
│   ├── server/
│   │   ├── game_engine.py          # Duck physics, shooting, state management
│   │   ├── renderer.py             # Headless PIL renderer (4 frames/step)
│   │   ├── duckhunt_environment.py # OpenEnv Environment[Action, Obs, State]
│   │   ├── app.py                  # FastAPI server
│   │   ├── config.py               # Game constants
│   │   └── Dockerfile              # Container (python:3.11-slim)
│   ├── client/
│   │   ├── duckhunt_client.py      # WebSocket client for multi-step episodes
│   │   └── models.py               # Pydantic v2 action/observation models
│   ├── assets/                     # sprites.png, background.jpg, font
│   └── openenv.yaml                # OpenEnv deployment metadata
├── training/
│   ├── train_grpo.py               # GRPO training script (Qwen3-8B + Unsloth)
│   ├── reward.py                   # 5-signal reward function
│   └── prompts.py                  # System prompt templates + shoot() schema
├── requirements-h100.txt           # H100 training dependencies
├── pyproject.toml                  # Editable install
└── CLAUDE.md                       # Claude Code context
```

---

## Game Environment Parameters

| Parameter | Value |
|-----------|-------|
| Screen | 800 × 500 px |
| Coordinates | 0.0–1.0 normalized |
| FPS | 30 |
| Ducks per round | 2 |
| Bullets per round | 3 |
| Game over on | 4 misses |
| Latency buckets | 100, 200, 300, 400, 500, 600 ms |
| Horizon range | 0–30 frames |
| Frames per observation | 4 sequential frames |

---

## Why This Matters

Duck Hunt is a low-stakes proxy for a real problem: **AI agents deployed in latency-sensitive environments must reason about time, not just space.** A robot arm, a drone, a trading algorithm — all face the same constraint. The horizon minimization objective formalizes a principle that applies beyond games: *act as early as you can, not as late as you must.*

This demonstrates that small language models can learn reactive, latency-aware reasoning through reinforcement learning alone — no human demonstrations required.

---

## Built At

**OpenEnv Hackathon — San Francisco, March 2026**
Built in one day with Claude Code + Northflank H100.

Inspired by [horizon_min](https://github.com/dmayboroda/horizon_min) by [@dmayboroda](https://github.com/dmayboroda).

---

## License

MIT
