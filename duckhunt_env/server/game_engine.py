"""Duck Hunt Game Engine

Ported from ~/horizon_min/duck_hunt_openenv/server/game_engine.py
Wrapped with DuckHuntGame interface per port-engine spec.
"""

import math
import random
from enum import Enum

from .config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    HITBOX_WIDTH,
    HITBOX_HEIGHT,
    SPEED_BASE,
    SPEED_VARIANCE,
    BOUNCE_DY_MIN,
    BOUNCE_DY_MAX,
    BOUNCE_TOP_DY_MIN,
    BOUNCE_TOP_DY_MAX,
    BOUNCE_BOTTOM_DY_MIN,
    BOUNCE_BOTTOM_DY_MAX,
    BULLETS_PER_MATCH,
    MATCH_DURATION_FRAMES,
    MATCHES_PER_ROUND,
    DUCKS_PER_MATCH,
    MAX_MISSES,
    REWARD_HIT,
    REWARD_DOUBLE_KILL,
    REWARD_MISS,
    REWARD_SHOOT_NOTHING,
)


class DuckState(Enum):
    FLYING = "flying"
    FALLING = "falling"
    ESCAPED = "escaped"


class Duck:
    """A duck that flies around and can be shot."""

    def __init__(self, round_number: int):
        # Random spawn position (left or right edge, bottom half)
        spawn_left = random.choice([True, False])
        if spawn_left:
            self.x = 0
        else:
            self.x = SCREEN_WIDTH

        self.y = random.randint(SCREEN_HEIGHT // 2, SCREEN_HEIGHT - HITBOX_HEIGHT)

        # Speed based on round number
        speed_range = range(SPEED_BASE + round_number, SPEED_BASE + SPEED_VARIANCE + round_number)
        speed = random.choice(list(speed_range))

        # Initial direction (move toward center)
        if spawn_left:
            self.dx = speed
        else:
            self.dx = -speed

        self.dy = random.randint(BOUNCE_DY_MIN, BOUNCE_DY_MAX)
        # Ensure dy is not 0
        if self.dy == 0:
            self.dy = random.choice([-1, 1])

        self.state = DuckState.FLYING
        self._update_sprite_dir()

    def _update_sprite_dir(self):
        """Update sprite direction based on velocity."""
        if self.dx > 0:
            if self.dy < 0:
                self.sprite_dir = "up_right"
            elif self.dy > 0:
                self.sprite_dir = "down_right"
            else:
                self.sprite_dir = "right"
        else:
            if self.dy < 0:
                self.sprite_dir = "up_left"
            elif self.dy > 0:
                self.sprite_dir = "down_left"
            else:
                self.sprite_dir = "left"

    def update(self, round_number: int):
        """Update duck position and state."""
        if self.state == DuckState.FLYING:
            self.x += self.dx
            self.y += self.dy
            self._check_boundaries(round_number)
            self._check_escaped()

        elif self.state == DuckState.FALLING:
            self.y += 4  # Fall straight down

    def _check_boundaries(self, round_number: int):
        """Check boundaries and apply bounce logic."""
        speed_range = range(SPEED_BASE + round_number, SPEED_BASE + SPEED_VARIANCE + round_number)
        speed = random.choice(list(speed_range))
        coin_toss = random.choice([-1, 1])

        # Left edge
        if self.x <= 0:
            self.dx = speed
            self.dy = random.randint(BOUNCE_DY_MIN, BOUNCE_DY_MAX)
            if self.dy == 0:
                self.dy = random.choice([-1, 1])
            self._update_sprite_dir()

        # Right edge
        elif self.x >= SCREEN_WIDTH - HITBOX_WIDTH:
            self.dx = -speed
            self.dy = random.randint(BOUNCE_DY_MIN, BOUNCE_DY_MAX)
            if self.dy == 0:
                self.dy = random.choice([-1, 1])
            self._update_sprite_dir()

        # Top edge
        elif self.y <= 0:
            self.dx = speed * coin_toss
            self.dy = random.randint(BOUNCE_TOP_DY_MIN, BOUNCE_TOP_DY_MAX)
            self._update_sprite_dir()

        # Bottom half
        elif self.y >= SCREEN_HEIGHT // 2:
            self.dx = speed * coin_toss
            self.dy = random.randint(BOUNCE_BOTTOM_DY_MIN, BOUNCE_BOTTOM_DY_MAX)
            self._update_sprite_dir()

    def _check_escaped(self):
        """Check if duck escaped off top of screen."""
        if self.y + HITBOX_HEIGHT < 0:
            self.state = DuckState.ESCAPED

    def check_hit(self, target_x: int, target_y: int) -> bool:
        """Check if target coordinates hit this duck."""
        if self.state != DuckState.FLYING:
            return False

        # Hitbox check (hitbox is at duck position with size HITBOX_WIDTH x HITBOX_HEIGHT)
        if target_x < self.x or target_x > self.x + HITBOX_WIDTH:
            return False
        if target_y < self.y or target_y > self.y + HITBOX_HEIGHT:
            return False

        return True

    def hit(self):
        """Mark duck as hit and start falling."""
        self.state = DuckState.FALLING
        self.dx = 0
        self.dy = 4
        self.sprite_dir = "falling"

    @property
    def is_finished(self) -> bool:
        """Check if duck is done (fell off screen or escaped)."""
        if self.state == DuckState.ESCAPED:
            return True
        if self.state == DuckState.FALLING and self.y >= SCREEN_HEIGHT // 2:
            return True
        return False

    @property
    def center_norm(self) -> tuple[float, float]:
        """Return normalized (x, y) center of the duck."""
        cx = (self.x + HITBOX_WIDTH / 2) / SCREEN_WIDTH
        cy = (self.y + HITBOX_HEIGHT / 2) / SCREEN_HEIGHT
        return (cx, cy)


class Match:
    """A single match with two ducks."""

    def __init__(self, round_number: int):
        self.round_number = round_number
        self.duck_a = Duck(round_number)
        self.duck_b = Duck(round_number)
        self.bullets_remaining = BULLETS_PER_MATCH
        self.frames_elapsed = 0
        self.ducks_hit = 0

    def advance_frames(self, n: int):
        """Advance the match by n frames."""
        for _ in range(n):
            if self.is_complete:
                break
            self.duck_a.update(self.round_number)
            self.duck_b.update(self.round_number)
            self.frames_elapsed += 1

    def process_shot(self, x: int, y: int) -> tuple[bool, bool]:
        """Process a shot at (x, y). Returns (hit_a, hit_b)."""
        if self.bullets_remaining <= 0:
            return (False, False)

        self.bullets_remaining -= 1

        hit_a = self.duck_a.check_hit(x, y)
        hit_b = self.duck_b.check_hit(x, y)

        if hit_a:
            self.duck_a.hit()
            self.ducks_hit += 1

        if hit_b:
            self.duck_b.hit()
            self.ducks_hit += 1

        return (hit_a, hit_b)

    @property
    def is_complete(self) -> bool:
        """Check if match is complete."""
        both_resolved = (
            self.duck_a.state != DuckState.FLYING
            and self.duck_b.state != DuckState.FLYING
        )
        time_up = self.frames_elapsed >= MATCH_DURATION_FRAMES
        return both_resolved or time_up

    def get_flying_count(self) -> int:
        """Count ducks that are still flying."""
        count = 0
        if self.duck_a.state == DuckState.FLYING:
            count += 1
        if self.duck_b.state == DuckState.FLYING:
            count += 1
        return count

    def get_state(self) -> dict:
        """Return current match state for rendering."""
        return {
            "duck_a": {
                "x": self.duck_a.x,
                "y": self.duck_a.y,
                "state": self.duck_a.state.value,
                "sprite_dir": self.duck_a.sprite_dir,
            },
            "duck_b": {
                "x": self.duck_b.x,
                "y": self.duck_b.y,
                "state": self.duck_b.state.value,
                "sprite_dir": self.duck_b.sprite_dir,
            },
            "bullets_remaining": self.bullets_remaining,
            "frames_elapsed": self.frames_elapsed,
            "ducks_hit": self.ducks_hit,
            "flying_count": self.get_flying_count(),
        }


class Round:
    """A round consisting of multiple matches."""

    def __init__(self, round_number: int):
        self.round_number = round_number
        self.current_match = Match(round_number)
        self.matches_completed = 0
        self.total_ducks_hit = 0

    def advance_to_next_match(self):
        """Complete current match and start next one."""
        self.total_ducks_hit += self.current_match.ducks_hit
        self.matches_completed += 1

        if self.matches_completed < MATCHES_PER_ROUND:
            self.current_match = Match(self.round_number)

    @property
    def is_complete(self) -> bool:
        """Check if round is complete."""
        return self.matches_completed >= MATCHES_PER_ROUND

    def get_misses(self) -> int:
        """Return total missed ducks so far."""
        return (self.matches_completed * DUCKS_PER_MATCH) - self.total_ducks_hit


# ---------------------------------------------------------------------------
# DuckHuntGame — high-level wrapper per port-engine spec
# ---------------------------------------------------------------------------

class DuckHuntGame:
    """Top-level game interface wrapping Round/Match/Duck.

    Provides: reset(), advance_frame(), shoot(x, y, advance_frames),
    is_over(), score, ducks_remaining, bullets_remaining, misses.
    """

    def __init__(self):
        self._round: Round | None = None
        self._round_number: int = 0
        self._total_misses: int = 0
        self._score: int = 0
        self._frame_counter: int = 0

    # -- lifecycle ----------------------------------------------------------

    def reset(self, seed: int | None = None):
        """Reset the game. Optionally seed RNG for deterministic replay."""
        if seed is not None:
            random.seed(seed)
        self._round_number = 1
        self._total_misses = 0
        self._score = 0
        self._frame_counter = 0
        self._round = Round(self._round_number)

    def advance_frame(self, n: int = 1):
        """Advance the current match by *n* frames."""
        if self._round is None or self.is_over():
            return
        self._round.current_match.advance_frames(n)
        self._frame_counter += n

    # -- shooting -----------------------------------------------------------

    def shoot(
        self, x: float, y: float, advance_frames: int = 0
    ) -> tuple[str, float, float]:
        """Shoot at normalised coordinates after advancing.

        Parameters
        ----------
        x, y : float  (0.0–1.0 normalised screen coords)
        advance_frames : int  (latency + horizon frames to advance first)

        Returns
        -------
        (result, base_reward, distance)
            result: "hit" | "double_kill" | "miss" | "no_target"
            base_reward: float
            distance: float — normalised distance to nearest duck
        """
        if self._round is None or self.is_over():
            return ("no_target", REWARD_SHOOT_NOTHING, 1.0)

        match = self._round.current_match
        flying_before = match.get_flying_count()

        # Advance by latency + horizon
        if advance_frames > 0:
            match.advance_frames(advance_frames)
            self._frame_counter += advance_frames

        # No ducks were flying when observation was taken
        if flying_before == 0:
            return ("no_target", REWARD_SHOOT_NOTHING, 1.0)

        # Distance to nearest duck (normalised) — computed *after* advance
        dist_a = _norm_distance(x, y, match.duck_a)
        dist_b = _norm_distance(x, y, match.duck_b)
        distance = min(dist_a, dist_b)

        # Convert to pixel coords and fire
        px = int(x * SCREEN_WIDTH)
        py = int(y * SCREEN_HEIGHT)
        hit_a, hit_b = match.process_shot(px, py)

        if hit_a and hit_b:
            result, base_reward = "double_kill", REWARD_DOUBLE_KILL
        elif hit_a or hit_b:
            result, base_reward = "hit", REWARD_HIT
        else:
            result, base_reward = "miss", REWARD_MISS

        self._score += int(hit_a) + int(hit_b)

        # Handle match / round completion
        self._maybe_advance_match()

        return (result, base_reward, distance)

    # -- queries ------------------------------------------------------------

    def is_over(self) -> bool:
        return self._total_misses >= MAX_MISSES

    @property
    def score(self) -> int:
        return self._score

    @property
    def ducks_remaining(self) -> int:
        if self._round is None:
            return 0
        return self._round.current_match.get_flying_count()

    @property
    def bullets_remaining(self) -> int:
        if self._round is None:
            return 0
        return self._round.current_match.bullets_remaining

    @property
    def misses(self) -> int:
        return self._total_misses

    @property
    def round_number(self) -> int:
        return self._round_number

    @property
    def frame_counter(self) -> int:
        return self._frame_counter

    # -- state for renderer / observation -----------------------------------

    def get_state(self) -> dict:
        """Return current match state dict (for renderer)."""
        if self._round is None:
            return {}
        return self._round.current_match.get_state()

    def get_duck_positions_norm(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return normalised (x, y) centres of both ducks."""
        if self._round is None:
            return ((0.0, 0.0), (0.0, 0.0))
        m = self._round.current_match
        return (m.duck_a.center_norm, m.duck_b.center_norm)

    # -- snapshot for deterministic replay ----------------------------------

    def snapshot(self) -> dict:
        """Capture RNG state + game state for deterministic replay."""
        return {
            "rng_state": random.getstate(),
            "round_number": self._round_number,
            "total_misses": self._total_misses,
            "score": self._score,
            "frame_counter": self._frame_counter,
        }

    def restore(self, snap: dict):
        """Restore from a snapshot (must re-seed then replay frames)."""
        random.setstate(snap["rng_state"])
        self._round_number = snap["round_number"]
        self._total_misses = snap["total_misses"]
        self._score = snap["score"]
        self._frame_counter = snap["frame_counter"]
        self._round = Round(self._round_number)

    # -- internal -----------------------------------------------------------

    def _maybe_advance_match(self):
        """Check if the current match is done; advance round if so."""
        if self._round is None:
            return
        match = self._round.current_match
        if not match.is_complete:
            return

        match_misses = DUCKS_PER_MATCH - match.ducks_hit
        self._total_misses += match_misses

        if self.is_over():
            return

        self._round.advance_to_next_match()

        if self._round.is_complete:
            self._round_number += 1
            self._round = Round(self._round_number)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _norm_distance(x: float, y: float, duck: Duck) -> float:
    """Euclidean distance from normalised shot (x, y) to duck centre."""
    cx, cy = duck.center_norm
    return math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
