"""Typed client for the adaptive live-system reliability OpenEnv server."""

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from openenv_models import ReliabilityAction, ReliabilityObservation, ReliabilityState


class ReliabilityControlEnv(
    EnvClient[ReliabilityAction, ReliabilityObservation, ReliabilityState]
):
    """Persistent WebSocket client for the reliability-control environment."""

    def _step_payload(self, action: ReliabilityAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[ReliabilityObservation]:
        observation_data = dict(payload.get("observation", {}))
        observation_data["reward"] = payload.get("reward")
        observation_data["done"] = payload.get("done", False)
        observation = ReliabilityObservation.model_validate(observation_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> ReliabilityState:
        return ReliabilityState.model_validate(payload)
