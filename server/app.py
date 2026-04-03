"""OpenEnv-compatible FastAPI application for the reliability environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from openenv.core.env_server.http_server import create_app

from openenv_models import ReliabilityAction, ReliabilityObservation
from server.environment import LiveSystemReliabilityEnvironment


def _web_enabled() -> bool:
    return os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in {"true", "1", "yes"}


def _build_env() -> LiveSystemReliabilityEnvironment:
    return LiveSystemReliabilityEnvironment()


def _configure_readme_path() -> None:
    # OpenEnv's web UI looks at ENV_README_PATH when README is not in /app/README.md.
    if os.getenv("ENV_README_PATH"):
        return
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    if readme_path.exists():
        os.environ["ENV_README_PATH"] = str(readme_path)


_configure_readme_path()

app = create_app(
    _build_env,
    ReliabilityAction,
    ReliabilityObservation,
    env_name="adaptive-rl-reliability",
)


if not _web_enabled():

    @app.get("/", tags=["Environment Info"])
    def root() -> dict[str, Any]:
        return {
            "name": "adaptive-rl-reliability",
            "status": "ok",
            "docs": "/docs",
            "health": "/health",
            "web_interface_enabled": False,
        }


def main(host: str = "0.0.0.0", port: int | None = None) -> None:
    import uvicorn

    resolved_port = port if port is not None else int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=resolved_port)


if __name__ == "__main__":
    main()
