"""FastAPI application exposing the OpenEnv-compatible reliability environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv is required to serve this environment. Install project dependencies first."
    ) from exc

from openenv_models import ReliabilityAction, ReliabilityObservation
from server.live_system_environment import LiveSystemReliabilityEnvironment


app = create_app(
    LiveSystemReliabilityEnvironment,
    ReliabilityAction,
    ReliabilityObservation,
    env_name="adaptive-rl-reliability",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
