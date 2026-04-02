"""Persistent FastAPI application exposing the OpenEnv-compatible reliability environment."""

from __future__ import annotations

from threading import Lock
from typing import Any

from fastapi import Body, FastAPI, HTTPException, status
from openenv.core.env_server.http_server import deserialize_action, serialize_observation
from openenv.core.env_server.types import (
    EnvironmentMetadata,
    HealthResponse,
    HealthStatus,
    ResetRequest,
    ResetResponse,
    SchemaResponse,
    StepRequest,
    StepResponse,
)
from pydantic import ValidationError

from openenv_models import (
    ReliabilityAction,
    ReliabilityObservation,
    ReliabilityState,
)
from server.live_system_environment import LiveSystemReliabilityEnvironment


app = FastAPI(
    title="OpenEnv Environment HTTP API",
    version="1.0.0",
    description="Persistent HTTP API for the adaptive RL reliability environment.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

_env_lock = Lock()
_env = LiveSystemReliabilityEnvironment()


@app.post("/reset", response_model=ResetResponse, tags=["Environment Control"])
def reset(request: ResetRequest = Body(default_factory=ResetRequest)) -> ResetResponse:
    with _env_lock:
        kwargs = request.model_dump(exclude_unset=True)
        observation = _env.reset(**kwargs)
        return ResetResponse(**serialize_observation(observation))


@app.post("/step", response_model=StepResponse, tags=["Environment Control"])
def step(request: StepRequest) -> StepResponse:
    try:
        action = deserialize_action(request.action, ReliabilityAction)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=exc.errors(),
        ) from exc

    with _env_lock:
        kwargs = request.model_dump(exclude_unset=True, exclude={"action"})
        observation = _env.step(action, **kwargs)
        return StepResponse(**serialize_observation(observation))


@app.get("/state", response_model=ReliabilityState, tags=["State Management"])
def get_state() -> ReliabilityState:
    with _env_lock:
        return _env.state


@app.get("/metadata", response_model=EnvironmentMetadata, tags=["Environment Info"])
def metadata() -> EnvironmentMetadata:
    return _env.get_metadata()


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    return HealthResponse(status=HealthStatus.HEALTHY)


@app.get("/schema", response_model=SchemaResponse, tags=["Schema"])
def schema() -> SchemaResponse:
    return SchemaResponse(
        action=ReliabilityAction.model_json_schema(),
        observation=ReliabilityObservation.model_json_schema(),
        state=ReliabilityState.model_json_schema(),
    )


@app.get("/", tags=["Environment Info"])
def root() -> dict[str, Any]:
    return {
        "name": "adaptive-rl-reliability",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
