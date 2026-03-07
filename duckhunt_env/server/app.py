"""FastAPI application for the Duck Hunt OpenEnv environment."""

from openenv.core.env_server.http_server import create_app

from ..client.models import DuckHuntAction, DuckHuntObservation
from .duckhunt_environment import DuckHuntEnvironment

app = create_app(
    DuckHuntEnvironment,
    DuckHuntAction,
    DuckHuntObservation,
    env_name="duckhunt",
    max_concurrent_envs=1,
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
