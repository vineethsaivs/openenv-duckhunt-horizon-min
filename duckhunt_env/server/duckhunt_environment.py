"""Duck Hunt OpenEnv Environment — inherits openenv Environment ABC."""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from ..client.models import DuckHuntAction, DuckHuntObservation
from .game_engine import DuckHuntGame
from .renderer import Renderer
from .config import (
    FRAMES_PER_OBSERVATION,
    FRAME_OUTPUT_SIZE,
    LATENCY_OPTIONS_MS,
    FPS,
    MAX_HORIZON,
)


class DuckHuntEnvironment(Environment[DuckHuntAction, DuckHuntObservation, State]):
    """OpenEnv 0.2.1 Duck Hunt environment.

    Each episode: agent receives frames + game metadata, replies with
    shoot(x, y, horizon). The game advances by (latency + horizon) frames
    before resolving the shot.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._game = DuckHuntGame()
        self._renderer = Renderer(output_size=FRAME_OUTPUT_SIZE)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._latency_ms: int = 0
        self._latency_frames: int = 0

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, episode_id=None, **kwargs) -> DuckHuntObservation:
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        self._game.reset(seed=seed)

        # Random latency for this episode
        self._latency_ms = random.choice(LATENCY_OPTIONS_MS)
        self._latency_frames = int(self._latency_ms / 1000 * FPS)

        # Render initial frames
        frames_b64 = self._render_n_frames(FRAMES_PER_OBSERVATION)

        return self._build_observation(
            frames=frames_b64,
            last_action_result=None,
            last_ducks_hit=0,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: DuckHuntAction, timeout_s=None, **kwargs) -> DuckHuntObservation:
        self._state.step_count += 1

        advance = self._latency_frames + action.horizon
        result, _base_reward, _distance = self._game.shoot(
            action.x, action.y, advance_frames=advance,
        )

        ducks_hit = {"hit": 1, "double_kill": 2}.get(result, 0)

        # Render new frames after the shot
        frames_b64 = self._render_n_frames(FRAMES_PER_OBSERVATION)

        return self._build_observation(
            frames=frames_b64,
            last_action_result=result,
            last_ducks_hit=ducks_hit,
        )

    # ------------------------------------------------------------------
    # state property (required by ABC)
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _render_n_frames(self, n: int) -> list[str]:
        """Render n frames, advancing the game each time."""
        frames = []
        for _ in range(n):
            game_state = self._game.get_state()
            img = self._renderer.render_and_resize(game_state, self._game.frame_counter)
            frames.append(self._renderer.image_to_base64(img))
            self._game.advance_frame(1)
        return frames

    def _build_observation(
        self,
        frames: list[str],
        last_action_result: str | None,
        last_ducks_hit: int,
    ) -> DuckHuntObservation:
        game = self._game
        return DuckHuntObservation(
            frames=frames,
            num_frames=len(frames),
            round_number=game.round_number,
            match_number=(game._round.matches_completed + 1) if game._round else 1,
            ducks_flying=game.ducks_remaining,
            bullets_remaining=game.bullets_remaining,
            match_ducks_hit=game._round.current_match.ducks_hit if game._round else 0,
            round_ducks_hit=(
                game._round.total_ducks_hit + (game._round.current_match.ducks_hit if game._round else 0)
            ) if game._round else 0,
            total_misses=game.misses,
            processing_latency_ms=self._latency_ms,
            last_action_result=last_action_result,
            last_ducks_hit=last_ducks_hit,
            step_count=self._state.step_count,
            done=game.is_over(),
            reward=0.0,
        )
