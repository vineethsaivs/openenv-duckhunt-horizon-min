"""Evaluation metrics for Duck Hunt adaptive latency + noise environment.

Computes three novel metrics from training data (collected snapshots or
training outputs):

METRIC 1 — Horizon Efficiency Ratio
    For every HIT, compare the horizon used vs. the minimum horizon that
    would still achieve a hit (via snapshot replay). Ratio = avg across
    hits. Perfect = 1.0, higher = wasteful.

METRIC 2 — Adaptation Speed
    When latency shifts (detected by latency_ms changes between consecutive
    snapshots), count shots until the next hit. Lower = faster adaptation.

METRIC 3 — Dual Inference Score
    Compare hit rate during compound uncertainty (latency shifted within
    last 5 steps) vs. stable periods. Reports both rates and ratio.
    Noise shifts are not tracked in snapshot data, so this reports the
    latency-only version.

Usage:
    python training/evaluate.py --results-dir /path/to/outputs/
    python training/evaluate.py --snapshots-file snapshots.json
"""

from __future__ import annotations

import argparse
import json
import math
import logging
import random
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


# -------------------------------------------------------------------
# Snapshot replay helpers (reused from train_grpo.py _make_accuracy_reward)
# -------------------------------------------------------------------

def _restore_duck(data: dict):
    """Restore a Duck object from snapshot data."""
    from duckhunt_env.server.game_engine import Duck, DuckState
    duck = object.__new__(Duck)
    duck.x = data["x"]
    duck.y = data["y"]
    duck.dx = data["dx"]
    duck.dy = data["dy"]
    duck.state = DuckState(data["state"])
    duck.sprite_dir = data["sprite_dir"]
    return duck


def _rng_restore(state_json: str):
    """Restore RNG state from JSON."""
    raw = json.loads(state_json)
    raw[1] = tuple(raw[1])
    random.setstate(tuple(raw))


def _replay_shot(snap: dict, x: float, y: float, horizon: int) -> bool:
    """Replay a shot from snapshot and return True if it hits."""
    from duckhunt_env.server.game_engine import DuckState
    from duckhunt_env.server.config import SCREEN_WIDTH, SCREEN_HEIGHT

    if not snap or not snap.get("duck_a"):
        return False

    if snap.get("rng_state"):
        _rng_restore(snap["rng_state"])

    duck_a = _restore_duck(snap["duck_a"])
    duck_b = _restore_duck(snap["duck_b"])
    round_num = snap.get("round_number", 1)
    latency_frames = snap.get("latency_frames", 0)

    total_advance = latency_frames + horizon
    for _ in range(total_advance):
        duck_a.update(round_num)
        duck_b.update(round_num)

    px = int(x * SCREEN_WIDTH)
    py = int(y * SCREEN_HEIGHT)
    return duck_a.check_hit(px, py) or duck_b.check_hit(px, py)


# -------------------------------------------------------------------
# Metric 1: Horizon Efficiency Ratio
# -------------------------------------------------------------------

def compute_horizon_efficiency(samples: list[dict]) -> dict:
    """For each HIT, find minimum horizon that still hits via replay."""
    from training.reward import parse_action

    ratios = []
    hits_evaluated = 0
    hits_skipped = 0

    for sample in samples:
        completion = sample.get("completion", "")
        action = parse_action(completion)
        if action is None:
            continue

        result = sample.get("result", "")
        if result not in ("hit", "double_kill"):
            continue

        snap_raw = sample.get("snapshot")
        if not snap_raw:
            hits_skipped += 1
            continue

        snap = json.loads(snap_raw) if isinstance(snap_raw, str) else snap_raw
        if not snap.get("duck_a"):
            hits_skipped += 1
            continue

        # Find minimum horizon that still hits
        min_horizon = action.horizon
        for h in range(action.horizon):
            if _replay_shot(snap, action.x, action.y, h):
                min_horizon = h
                break

        if min_horizon == 0:
            ratio = 1.0 if action.horizon == 0 else float(action.horizon + 1)
        else:
            ratio = action.horizon / min_horizon

        ratios.append(ratio)
        hits_evaluated += 1

    avg_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
    return {
        "avg_ratio": round(avg_ratio, 4),
        "hits_evaluated": hits_evaluated,
        "hits_skipped": hits_skipped,
        "perfect_count": sum(1 for r in ratios if r == 1.0),
    }


# -------------------------------------------------------------------
# Metric 2: Adaptation Speed
# -------------------------------------------------------------------

def compute_adaptation_speed(samples: list[dict]) -> dict:
    """Measure shots-to-hit after each detected latency shift."""
    recovery_counts = []
    prev_latency = None
    shots_since_shift = None

    for sample in samples:
        latency_ms = sample.get("latency_ms")
        result = sample.get("result", "")

        if prev_latency is not None and latency_ms != prev_latency:
            # Shift detected — start counting
            if shots_since_shift is not None and shots_since_shift > 0:
                # Previous shift never recovered; record as the count so far
                recovery_counts.append(shots_since_shift)
            shots_since_shift = 0

        if shots_since_shift is not None:
            shots_since_shift += 1
            if result in ("hit", "double_kill"):
                recovery_counts.append(shots_since_shift)
                shots_since_shift = None

        prev_latency = latency_ms

    # Handle trailing unrecovered shift
    if shots_since_shift is not None and shots_since_shift > 0:
        recovery_counts.append(shots_since_shift)

    avg_recovery = (
        sum(recovery_counts) / len(recovery_counts)
        if recovery_counts
        else float("nan")
    )
    return {
        "avg_shots_to_recover": round(avg_recovery, 2),
        "shift_events": len(recovery_counts),
        "fastest_recovery": min(recovery_counts) if recovery_counts else None,
        "slowest_recovery": max(recovery_counts) if recovery_counts else None,
    }


# -------------------------------------------------------------------
# Metric 3: Dual Inference Score
# -------------------------------------------------------------------

def compute_dual_inference(samples: list[dict]) -> dict:
    """Compare hit rate during compound uncertainty vs. stable periods.

    Compound uncertainty = latency shifted within last 5 steps.
    Noise shifts are not tracked in snapshot data, so this is the
    latency-only version.
    """
    WINDOW = 5

    # Build shift event indices
    shift_steps: set[int] = set()
    prev_latency = None
    for i, sample in enumerate(samples):
        latency_ms = sample.get("latency_ms")
        if prev_latency is not None and latency_ms != prev_latency:
            shift_steps.add(i)
        prev_latency = latency_ms

    uncertain_hits = 0
    uncertain_total = 0
    stable_hits = 0
    stable_total = 0

    for i, sample in enumerate(samples):
        result = sample.get("result", "")
        is_hit = result in ("hit", "double_kill")

        # Check if any shift occurred within the last WINDOW steps
        in_uncertainty = any(
            s in shift_steps for s in range(max(0, i - WINDOW + 1), i + 1)
        )

        if in_uncertainty:
            uncertain_total += 1
            if is_hit:
                uncertain_hits += 1
        else:
            stable_total += 1
            if is_hit:
                stable_hits += 1

    uncertain_rate = (
        uncertain_hits / uncertain_total if uncertain_total > 0 else float("nan")
    )
    stable_rate = (
        stable_hits / stable_total if stable_total > 0 else float("nan")
    )
    ratio = uncertain_rate / stable_rate if stable_rate > 0 else float("nan")

    return {
        "uncertain_hit_rate": round(uncertain_rate, 4),
        "uncertain_samples": uncertain_total,
        "stable_hit_rate": round(stable_rate, 4),
        "stable_samples": stable_total,
        "ratio": round(ratio, 4),
        "shift_events": len(shift_steps),
        "note": "Latency-only (noise shifts not tracked in snapshot data)",
    }


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_samples(results_dir: str | None, snapshots_file: str | None) -> list[dict]:
    """Load evaluation samples from results dir or snapshots file."""
    samples = []

    if snapshots_file:
        path = Path(snapshots_file)
        if path.exists():
            with open(path) as f:
                samples = json.load(f)
            logger.info("Loaded %d samples from %s", len(samples), path)
            return samples

    if results_dir:
        rdir = Path(results_dir)
        # Try common output patterns
        for pattern in ["samples.json", "snapshots.json", "eval_data.json",
                        "training_results.json", "*/samples.json"]:
            for p in rdir.glob(pattern):
                with open(p) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    samples.extend(data)
                    logger.info("Loaded %d samples from %s", len(data), p)

        # Try JSONL files
        for p in rdir.glob("*.jsonl"):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
            logger.info("Loaded samples from %s (total: %d)", p, len(samples))

    if not samples:
        logger.warning("No samples found. Generating synthetic eval data.")
        samples = _generate_synthetic_samples()

    return samples


def _generate_synthetic_samples(n: int = 200) -> list[dict]:
    """Generate synthetic samples using the game engine for evaluation."""
    from duckhunt_env.server.game_engine import DuckHuntGame, DuckState
    from duckhunt_env.server.config import (
        SCREEN_WIDTH, SCREEN_HEIGHT, FRAMES_PER_OBSERVATION,
        LATENCY_OPTIONS_MS, FPS,
    )

    game = DuckHuntGame()
    samples = []

    random.seed(42)
    game.reset(seed=42)

    latency_ms = random.choice(LATENCY_OPTIONS_MS)
    latency_frames = int(latency_ms / 1000 * FPS)
    step_since_shift = 0
    steps_until_shift = random.randint(5, 15)

    for i in range(n):
        if game.is_over():
            game.reset()
            latency_ms = random.choice(LATENCY_OPTIONS_MS)
            latency_frames = int(latency_ms / 1000 * FPS)
            step_since_shift = 0
            steps_until_shift = random.randint(5, 15)

        for _ in range(50):
            if game.ducks_remaining > 0:
                break
            game.advance_frame(5)
            if game.is_over():
                game.reset()
                latency_ms = random.choice(LATENCY_OPTIONS_MS)
                latency_frames = int(latency_ms / 1000 * FPS)
                step_since_shift = 0
                steps_until_shift = random.randint(5, 15)

        game.advance_frame(FRAMES_PER_OBSERVATION)

        match = game._round.current_match if game._round else None
        duck_a_data = _snapshot_duck_raw(match.duck_a) if match else {}
        duck_b_data = _snapshot_duck_raw(match.duck_b) if match else {}

        snap = {
            "duck_a": duck_a_data,
            "duck_b": duck_b_data,
            "round_number": game.round_number,
            "bullets_remaining": game.bullets_remaining,
            "latency_frames": latency_frames,
            "rng_state": json.dumps(random.getstate(), default=_rng_serial),
        }

        x = random.random()
        y = random.random()
        horizon = random.randint(0, 15)

        result, _base_reward, distance = game.shoot(
            x, y, advance_frames=latency_frames + horizon,
        )

        completion = f"shoot(x={x:.4f}, y={y:.4f}, horizon={horizon})"

        samples.append({
            "completion": completion,
            "result": result,
            "distance": round(distance, 4),
            "latency_ms": latency_ms,
            "latency_frames": latency_frames,
            "snapshot": snap,
        })

        step_since_shift += 1
        if step_since_shift >= steps_until_shift:
            latency_ms = random.choice(LATENCY_OPTIONS_MS)
            latency_ms += int(random.gauss(0, 50))
            latency_ms = max(50, min(800, latency_ms))
            latency_frames = int(latency_ms / 1000 * FPS)
            step_since_shift = 0
            steps_until_shift = random.randint(5, 15)

    logger.info("Generated %d synthetic samples for evaluation", len(samples))
    return samples


def _snapshot_duck_raw(duck) -> dict:
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


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Duck Hunt adaptive latency/noise metrics"
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory containing training outputs",
    )
    parser.add_argument(
        "--snapshots-file", type=str, default=None,
        help="JSON file of collected snapshots with completions/results",
    )
    args = parser.parse_args()

    if not args.results_dir and not args.snapshots_file:
        logger.info("No input specified, generating synthetic eval data")

    samples = load_samples(args.results_dir, args.snapshots_file)
    logger.info("Evaluating %d samples", len(samples))

    # Compute metrics
    m1 = compute_horizon_efficiency(samples)
    m2 = compute_adaptation_speed(samples)
    m3 = compute_dual_inference(samples)

    results = {
        "horizon_efficiency": m1,
        "adaptation_speed": m2,
        "dual_inference": m3,
        "num_samples": len(samples),
    }

    # Print table
    print()
    print("=" * 60)
    print("  DUCK HUNT EVALUATION METRICS")
    print("=" * 60)
    print()

    print("METRIC 1: Horizon Efficiency Ratio")
    print(f"  Avg ratio (used/min):  {m1['avg_ratio']}")
    print(f"  Hits evaluated:        {m1['hits_evaluated']}")
    print(f"  Perfect (ratio=1.0):   {m1['perfect_count']}")
    print(f"  Hits skipped:          {m1['hits_skipped']}")
    print()

    print("METRIC 2: Adaptation Speed")
    print(f"  Avg shots to recover:  {m2['avg_shots_to_recover']}")
    print(f"  Shift events:          {m2['shift_events']}")
    print(f"  Fastest recovery:      {m2['fastest_recovery']}")
    print(f"  Slowest recovery:      {m2['slowest_recovery']}")
    print()

    print("METRIC 3: Dual Inference Score")
    print(f"  Uncertain hit rate:    {m3['uncertain_hit_rate']}")
    print(f"  Uncertain samples:     {m3['uncertain_samples']}")
    print(f"  Stable hit rate:       {m3['stable_hit_rate']}")
    print(f"  Stable samples:        {m3['stable_samples']}")
    print(f"  Ratio (uncert/stable): {m3['ratio']}")
    print(f"  Note: {m3['note']}")
    print()
    print("=" * 60)

    # Save results
    out_dir = Path(args.results_dir) if args.results_dir else Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
