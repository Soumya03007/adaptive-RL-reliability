try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies first."
    ) from exc

from reliability_game.models import ReliabilityGameAction, ReliabilityGameObservation
from reliability_game.server.reliability_game_environment import ReliabilityGameEnvironment


app = create_app(
    ReliabilityGameEnvironment,
    ReliabilityGameAction,
    ReliabilityGameObservation,
    env_name="reliability_game",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.port == 8000:
        main()
    else:
        main(port=args.port)
