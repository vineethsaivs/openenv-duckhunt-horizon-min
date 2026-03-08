"""GRPO training script for Duck Hunt on H100.

Uses Unsloth FastModel + TRL GRPOTrainer to train Qwen3.5-9B
from game screenshots via Group Relative Policy Optimization.

Usage:
    python training/train_grpo.py \
        --max-steps 500 \
        --output-dir outputs/duckhunt_grpo \
        --push-to-hub --hub-model-id user/duckhunt-qwen3.5-grpo
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset, Features, Value
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_grpo")


# ===================================================================
#  1. CLI
# ===================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Duck Hunt GRPO Training (Qwen3.5-9B)")
    p.add_argument("--env-url", type=str, default=None,
                    help="OpenEnv server URL (if None, starts server locally)")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--num-samples", type=int, default=200,
                    help="Game-state snapshots to collect")
    p.add_argument("--output-dir", type=str, default="outputs/")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-model-id", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ===================================================================
#  2. Server management
# ===================================================================
def start_server(port: int = 7860) -> subprocess.Popen:
    """Start the Duck Hunt server as a subprocess."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "duckhunt_env.server.app:app",
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    logger.info("Starting server: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for server to be ready
    import httpx
    for _attempt in range(30):
        try:
            r = httpx.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                logger.info("Server ready on port %d", port)
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.kill()
    raise RuntimeError("Server failed to start within 30 seconds")


def stop_server(proc: subprocess.Popen):
    """Stop the server subprocess."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        logger.info("Server stopped")


# ===================================================================
#  3. Data collection — local game engine
# ===================================================================
def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _snapshot_duck(duck) -> dict:
    return {
        "x": duck.x, "y": duck.y,
        "dx": duck.dx, "dy": duck.dy,
        "state": duck.state.value,
        "sprite_dir": duck.sprite_dir,
    }


def _rng_serial(obj):
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(type(obj))


def collect_snapshots(num_samples: int, seed: int = 42) -> list[dict]:
    """Collect game-state snapshots using the local engine."""
    from duckhunt_env.server.game_engine import DuckHuntGame
    from duckhunt_env.server.renderer import Renderer
    from duckhunt_env.server.config import (
        FRAME_OUTPUT_SIZE, FRAMES_PER_OBSERVATION,
        LATENCY_OPTIONS_MS, FPS,
    )

    game = DuckHuntGame()
    renderer = Renderer(output_size=FRAME_OUTPUT_SIZE)
    samples = []

    random.seed(seed)
    game.reset(seed=seed)

    latency_ms = random.choice(LATENCY_OPTIONS_MS)
    latency_frames = int(latency_ms / 1000 * FPS)

    for i in range(num_samples):
        if game.is_over():
            game.reset()
            latency_ms = random.choice(LATENCY_OPTIONS_MS)
            latency_frames = int(latency_ms / 1000 * FPS)

        # Ensure ducks are flying
        for _ in range(50):
            if game.ducks_remaining > 0:
                break
            game.advance_frame(5)
            if game.is_over():
                game.reset()
                latency_ms = random.choice(LATENCY_OPTIONS_MS)
                latency_frames = int(latency_ms / 1000 * FPS)

        # Render frames
        frames_b64 = []
        for _ in range(FRAMES_PER_OBSERVATION):
            state = game.get_state()
            img = renderer.render_and_resize(state, game.frame_counter)
            frames_b64.append(_pil_to_b64(img.convert("RGB")))
            game.advance_frame(1)

        # Capture snapshot for deterministic replay
        match = game._round.current_match if game._round else None
        snapshot = {
            "duck_a": _snapshot_duck(match.duck_a) if match else {},
            "duck_b": _snapshot_duck(match.duck_b) if match else {},
            "round_number": game.round_number,
            "bullets_remaining": game.bullets_remaining,
            "latency_frames": latency_frames,
            "rng_state": json.dumps(random.getstate(), default=_rng_serial),
        }

        samples.append({
            "frames_b64": frames_b64,
            "latency_ms": latency_ms,
            "latency_frames": latency_frames,
            "ducks_flying": game.ducks_remaining,
            "snapshot": snapshot,
        })

        # Random action to advance the game
        x = random.random()
        y = random.random()
        horizon = random.randint(0, 15)
        game.shoot(x, y, advance_frames=latency_frames + horizon)

        if (i + 1) % 50 == 0:
            logger.info("Collected %d / %d snapshots", i + 1, num_samples)

    logger.info("Collection complete: %d snapshots", len(samples))
    return samples


# ===================================================================
#  4. Build HuggingFace dataset
# ===================================================================
def build_dataset(samples: list[dict]) -> Dataset:
    """Build dataset with Qwen3.5 multimodal chat format.

    Each row stores a JSON-serialised list of chat messages with
    base64 image data URIs inline.  The set_transform deserialises
    at access time so the trainer sees native message lists.
    """
    from training.prompts import format_system_prompt

    prompts = []
    snapshots = []
    latencies = []

    for sample in samples:
        frames_b64 = sample["frames_b64"]
        latency_ms = sample["latency_ms"]
        num_frames = len(frames_b64)
        latency_frames = sample.get("latency_frames", 0)
        if latency_frames == 0:
            latency_frames = int(latency_ms / 1000 * 30)

        system_text = format_system_prompt(
            num_frames=num_frames,
            processing_latency_frames=latency_frames,
        )

        # Qwen3.5 multimodal format: images as data URIs in content list
        user_content = []
        for b64 in frames_b64:
            user_content.append({
                "type": "image",
                "image": f"data:image/png;base64,{b64}",
            })
        user_content.append({
            "type": "text",
            "text": (
                f"{num_frames} frames, "
                f"{sample.get('ducks_flying', '?')} ducks flying, "
                f"latency {latency_frames} frames. Shoot now."
            ),
        })

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_content},
        ]

        prompts.append(json.dumps(messages))
        snapshots.append(json.dumps(sample.get("snapshot") or {}))
        latencies.append(latency_ms)

    ds = Dataset.from_dict(
        {
            "prompt": prompts,
            "snapshot": snapshots,
            "latency_ms": latencies,
        },
        features=Features({
            "prompt": Value("string"),
            "snapshot": Value("string"),
            "latency_ms": Value("int32"),
        }),
    )

    # Transform: deserialise JSON prompts to message lists at access time
    ds.set_transform(_prompt_transform)

    logger.info("Dataset built: %d rows", len(ds))
    return ds


def _prompt_transform(batch: dict) -> dict:
    """Deserialise JSON prompts to message lists."""
    new_prompts = []
    for prompt_json in batch["prompt"]:
        messages = json.loads(prompt_json)
        new_prompts.append(messages)
    return {
        "prompt": new_prompts,
        "snapshot": batch["snapshot"],
        "latency_ms": batch["latency_ms"],
    }


# ===================================================================
#  5. Reward functions for GRPOTrainer
# ===================================================================
def _make_accuracy_reward():
    """Reward function that simulates the shot from the stored snapshot."""
    from duckhunt_env.server.game_engine import Duck, DuckState
    from duckhunt_env.server.config import SCREEN_WIDTH, SCREEN_HEIGHT
    from training.reward import (
        parse_action, REWARD_INVALID, REWARD_HIT, REWARD_DOUBLE_KILL,
        REWARD_MISS, REWARD_NO_TARGET, PROXIMITY_BONUS, PROXIMITY_DECAY,
        LAMBDA_HORIZON, MAX_HORIZON,
    )

    def _restore_duck(data: dict) -> Duck:
        duck = object.__new__(Duck)
        duck.x = data["x"]
        duck.y = data["y"]
        duck.dx = data["dx"]
        duck.dy = data["dy"]
        duck.state = DuckState(data["state"])
        duck.sprite_dir = data["sprite_dir"]
        return duck

    def _rng_restore(state_json: str):
        raw = json.loads(state_json)
        raw[1] = tuple(raw[1])
        random.setstate(tuple(raw))

    def accuracy_reward(completions, snapshot, **kwargs) -> list[float]:
        rewards = []
        for text, snap_json in zip(completions, snapshot):
            if isinstance(text, list):
                text = text[0].get("content", "") if text else ""

            action = parse_action(text)
            if action is None:
                rewards.append(REWARD_INVALID)
                continue

            snap = json.loads(snap_json) if isinstance(snap_json, str) else snap_json
            if not snap or not snap.get("duck_a"):
                rewards.append(0.0)
                continue

            if snap.get("rng_state"):
                _rng_restore(snap["rng_state"])

            duck_a = _restore_duck(snap["duck_a"])
            duck_b = _restore_duck(snap["duck_b"])
            round_num = snap.get("round_number", 1)
            latency_frames = snap.get("latency_frames", 0)

            had_target = (
                duck_a.state == DuckState.FLYING
                or duck_b.state == DuckState.FLYING
            )

            total_advance = latency_frames + action.horizon
            for _ in range(total_advance):
                duck_a.update(round_num)
                duck_b.update(round_num)

            if not had_target:
                rewards.append(REWARD_NO_TARGET)
                continue

            px = int(action.x * SCREEN_WIDTH)
            py = int(action.y * SCREEN_HEIGHT)
            hit_a = duck_a.check_hit(px, py)
            hit_b = duck_b.check_hit(px, py)

            if hit_a and hit_b:
                base = REWARD_DOUBLE_KILL
            elif hit_a or hit_b:
                base = REWARD_HIT
            else:
                base = REWARD_MISS

            penalty = 0.0
            if hit_a or hit_b:
                penalty = LAMBDA_HORIZON * (action.horizon / MAX_HORIZON)

            proximity = 0.0
            if not (hit_a or hit_b):
                da_norm = (duck_a.x / SCREEN_WIDTH, duck_a.y / SCREEN_HEIGHT)
                db_norm = (duck_b.x / SCREEN_WIDTH, duck_b.y / SCREEN_HEIGHT)
                dist_a = math.sqrt(
                    (action.x - da_norm[0]) ** 2 + (action.y - da_norm[1]) ** 2
                )
                dist_b = math.sqrt(
                    (action.x - db_norm[0]) ** 2 + (action.y - db_norm[1]) ** 2
                )
                min_dist = min(dist_a, dist_b)
                proximity = PROXIMITY_BONUS * math.exp(-PROXIMITY_DECAY * min_dist)

            rewards.append(base - penalty + proximity)

        return rewards

    accuracy_reward.__name__ = "duckhunt_accuracy"
    return accuracy_reward


def _make_format_reward():
    """Reward function that scores output format validity."""
    from training.reward import parse_action

    def format_reward(completions, **kwargs) -> list[float]:
        rewards = []
        for text in completions:
            if isinstance(text, list):
                text = text[0].get("content", "") if text else ""
            action = parse_action(text)
            rewards.append(1.0 if action is not None else 0.0)
        return rewards

    format_reward.__name__ = "duckhunt_format"
    return format_reward


# ===================================================================
#  6. Model loading (Unsloth FastModel — Qwen3.5-9B)
# ===================================================================
def load_model(args: argparse.Namespace):
    """Load Qwen3.5-9B with bf16 LoRA via Unsloth FastModel."""
    from unsloth import FastModel

    logger.info("Loading Qwen3.5-9B with bf16 via Unsloth FastModel ...")

    model, tokenizer = FastModel.from_pretrained(
        "unsloth/Qwen3.5-9B",
        load_in_4bit=False,
        load_in_16bit=True,
        fast_inference=False,
        use_gradient_checkpointing="unsloth",
    )

    model = FastModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_rslora=False,
        use_dora=False,
    )

    logger.info("Model loaded: Qwen3.5-9B, bf16 LoRA rank 16")
    return model, tokenizer


# ===================================================================
#  7. Training
# ===================================================================
def train(args: argparse.Namespace):
    """Main training entry point."""
    from trl import GRPOConfig, GRPOTrainer

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # --- Auto-start server if no --env-url ---
    server_proc = None
    if args.env_url is None:
        server_proc = start_server(port=7860)
        args.env_url = "http://localhost:7860"

    try:
        # --- Collect data ---
        logger.info("Collecting %d game-state snapshots ...", args.num_samples)
        samples = collect_snapshots(args.num_samples, seed=args.seed)
        dataset = build_dataset(samples)

        # --- Load model ---
        model, tokenizer = load_model(args)

        # --- Reward functions ---
        accuracy_fn = _make_accuracy_reward()
        format_fn = _make_format_reward()

        # --- GRPO config ---
        output_dir = args.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        report_to = "none"
        try:
            import wandb
            report_to = "wandb"
            wandb.init(
                project="duckhunt-grpo",
                name=f"qwen3.5-9b-grpo-{args.max_steps}steps",
                config={
                    "model": "unsloth/Qwen3.5-9B",
                    "lora_rank": 16,
                    "precision": "bf16",
                    "max_steps": args.max_steps,
                    "num_samples": args.num_samples,
                },
            )
        except ImportError:
            logger.warning("wandb not available, logging disabled")

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-6,
            warmup_ratio=0.05,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            bf16=True,
            # GRPO specifics
            num_generations=4,
            max_completion_length=256,
            temperature=0.7,
            # Logging & checkpointing
            logging_steps=1,
            save_steps=50,
            save_total_limit=3,
            report_to=report_to,
            # Reward weights: [accuracy, format]
            reward_weights=[1.0, 0.3],
        )

        # --- Workaround: TRL/PEFT version mismatch ---
        if not hasattr(model, 'warnings_issued'):
            model.warnings_issued = {}

        # --- Trainer ---
        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dataset,
            reward_funcs=[accuracy_fn, format_fn],
            processing_class=tokenizer,
        )

        logger.info("Starting GRPO training for %d steps ...", args.max_steps)
        trainer.train()

        # --- Save final ---
        final_dir = Path(output_dir) / "final"
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info("Training complete. Final model saved to %s", final_dir)

        # --- Push to Hub ---
        if args.push_to_hub and args.hub_model_id:
            _push_to_hub(args.hub_model_id, str(final_dir))

    finally:
        if server_proc:
            stop_server(server_proc)


def _push_to_hub(repo_id: str, checkpoint_dir: str):
    """Push final model to HuggingFace Hub."""
    from huggingface_hub import HfApi, ModelCard

    logger.info("Pushing %s -> %s ...", checkpoint_dir, repo_id)

    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=repo_id,
        ignore_patterns=["optimizer.pt", "scheduler.pt"],
    )

    card_content = f"""\
---
library_name: peft
base_model: unsloth/Qwen3.5-9B
tags:
  - grpo
  - reinforcement-learning
  - duck-hunt
---

# {repo_id.split('/')[-1]}

LoRA adapter for [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B),
fine-tuned with GRPO to play Duck Hunt.

## Training

- **Method**: GRPO (Group Relative Policy Optimization)
- **LoRA rank**: 16, bf16 via Unsloth
- **Framework**: TRL GRPOTrainer

## Usage

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```
"""
    card = ModelCard(card_content)
    card.push_to_hub(repo_id)
    logger.info("Pushed to https://huggingface.co/%s", repo_id)


# ===================================================================
#  Entry point
# ===================================================================
if __name__ == "__main__":
    args = parse_args()
    train(args)
