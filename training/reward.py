"""Multi-signal reward function for Duck Hunt GRPO training.

Five signals:
  1. Hit: +1.0,  double_kill: +2.5
  2. Horizon penalty on hits: -0.1 * (horizon / 30)
  3. Miss: -0.3 + proximity_bonus  (0.5 * exp(-5.0 * distance))
  4. No target: -0.5
  5. Invalid format: -1.0

The regex matches:  shoot(x=0.42, y=0.31, horizon=3)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REWARD_HIT = 1.0
REWARD_DOUBLE_KILL = 2.5
REWARD_MISS = -0.3
REWARD_NO_TARGET = -0.5
REWARD_INVALID = -1.0
LAMBDA_HORIZON = 0.1
MAX_HORIZON = 30
PROXIMITY_BONUS = 0.5
PROXIMITY_DECAY = 5.0

# Regex that matches:  shoot(x=0.42, y=0.31, horizon=3)
SHOOT_PATTERN = re.compile(
    r'shoot\(x=([\d.]+),\s*y=([\d.]+),\s*horizon=(\d+)\)'
)


# ---------------------------------------------------------------------------
# Parsed action
# ---------------------------------------------------------------------------
@dataclass
class ParsedAction:
    x: float
    y: float
    horizon: int


def parse_action(text: str) -> ParsedAction | None:
    """Extract a shoot() call from *text*.  Returns None on failure."""
    m = SHOOT_PATTERN.search(text)
    if m is None:
        return None
    x = max(0.0, min(1.0, float(m.group(1))))
    y = max(0.0, min(1.0, float(m.group(2))))
    horizon = max(0, min(MAX_HORIZON, int(m.group(3))))
    return ParsedAction(x=x, y=y, horizon=horizon)


# ---------------------------------------------------------------------------
# Single-step reward (with optional game result)
# ---------------------------------------------------------------------------
def compute_step_reward(
    action: ParsedAction | None,
    result: dict | None = None,
) -> float:
    """Compute the scalar reward for one completion.

    Parameters
    ----------
    action : ParsedAction | None
        Parsed action, or None if the model output was unparseable.
    result : dict | None
        If provided, a dict from the environment with keys:
        ``result`` ("hit" | "double_kill" | "miss" | "no_target"),
        ``distance`` (float, normalised distance to nearest duck).
        When None (format-only mode), valid parses score 0.0.
    """
    if action is None:
        return REWARD_INVALID

    # No game result → format-only scoring
    if result is None:
        return 0.0

    outcome = result.get("result", "miss")
    distance = result.get("distance", 1.0)

    # Base reward by outcome
    if outcome == "double_kill":
        base = REWARD_DOUBLE_KILL
    elif outcome == "hit":
        base = REWARD_HIT
    elif outcome == "no_target":
        return REWARD_NO_TARGET
    else:  # miss
        base = REWARD_MISS

    # Horizon penalty (hits only)
    penalty = 0.0
    if outcome in ("hit", "double_kill"):
        penalty = LAMBDA_HORIZON * (action.horizon / MAX_HORIZON)

    # Proximity bonus (misses only — gives gradient signal)
    proximity = 0.0
    if outcome == "miss":
        proximity = PROXIMITY_BONUS * math.exp(-PROXIMITY_DECAY * distance)

    return base - penalty + proximity


# ---------------------------------------------------------------------------
# GRPO batch reward function
# ---------------------------------------------------------------------------
def duckhunt_reward(
    completions: list[str],
    results: list[dict] | None = None,
) -> list[float]:
    """Compute rewards for a batch of model completions.

    This is the function passed to TRL's GRPOTrainer as a reward function.

    Parameters
    ----------
    completions : list[str]
        Raw decoded model outputs.
    results : list[dict] | None
        Per-completion environment outcomes.  When None the reward is
        format-only (valid parse → 0.0, invalid → -1.0).

    Returns
    -------
    list[float]
        One reward per completion.
    """
    rewards = []
    for i, text in enumerate(completions):
        action = parse_action(text)
        env_result = results[i] if results is not None else None
        rewards.append(compute_step_reward(action, env_result))
    return rewards
