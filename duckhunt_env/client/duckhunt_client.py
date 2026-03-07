"""Duck Hunt OpenEnv WebSocket client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import DuckHuntAction, DuckHuntObservation


class DuckHuntEnv(EnvClient[DuckHuntAction, DuckHuntObservation, State]):
    """WebSocket client for the Duck Hunt environment.

    Example:
        >>> with DuckHuntEnv(base_url="http://localhost:7860") as env:
        ...     result = env.reset(seed=42)
        ...     obs = result.observation
        ...     result = env.step(DuckHuntAction(x=0.5, y=0.3, horizon=5))
    """

    def _step_payload(self, action: DuckHuntAction) -> Dict:
        return action.model_dump(exclude={"metadata"})

    def _parse_result(self, payload: Dict) -> StepResult[DuckHuntObservation]:
        obs_data = payload.get("observation", {})
        observation = DuckHuntObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", False),
                "reward": payload.get("reward"),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
