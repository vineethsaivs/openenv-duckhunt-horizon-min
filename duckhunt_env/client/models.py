"""Pydantic models for the Duck Hunt OpenEnv environment."""

from typing import Literal

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class DuckHuntAction(Action):
    """Action: shoot at normalised (x, y) with a horizon prediction window."""

    x: float = Field(..., ge=0.0, le=1.0, description="Normalised x coordinate (0.0–1.0)")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalised y coordinate (0.0–1.0)")
    horizon: int = Field(..., ge=0, le=30, description="Frames to advance before shot resolves (0–30)")


class DuckHuntObservation(Observation):
    """Observation returned after reset or step."""

    # Visual data
    frames: list[str] = Field(default_factory=list, description="Base64-encoded PNG frames")
    num_frames: int = Field(default=0, description="Number of frames in this observation")

    # Game progress
    round_number: int = Field(default=1, description="Current round (1, 2, 3, …)")
    match_number: int = Field(default=1, description="Current match within the round (1–5)")

    # Current match state
    ducks_flying: int = Field(default=0, description="Number of ducks currently flying (0–2)")
    bullets_remaining: int = Field(default=3, description="Bullets left in this match (0–3)")
    match_ducks_hit: int = Field(default=0, description="Ducks hit in this match (0–2)")

    # Round / game progress
    round_ducks_hit: int = Field(default=0, description="Ducks hit in the current round (0–10)")
    total_misses: int = Field(default=0, description="Cumulative misses towards game over")

    # Hardware simulation
    processing_latency_ms: int = Field(default=0, description="Hidden from model — always 0")
    recent_results: list[dict] = Field(default_factory=list, description="Last N shot outcomes with result and distance for latency inference")

    # Feedback from last action
    last_action_result: Literal["hit", "miss", "double_kill", "no_target"] | None = Field(
        default=None, description="Result of the most recent shot"
    )
    last_ducks_hit: int = Field(default=0, description="Ducks hit by the last shot (0–2)")

    # Step tracking
    step_count: int = Field(default=0, description="Total steps taken this episode")
