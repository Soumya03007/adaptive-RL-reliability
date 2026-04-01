from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    ReliabilityGameAction,
    ReliabilityGameObservation,
    ReliabilityGameState,
)


class ReliabilityGameEnv(
    EnvClient[
        ReliabilityGameAction,
        ReliabilityGameObservation,
        ReliabilityGameState,
    ]
):
    def _step_payload(self, action: ReliabilityGameAction) -> Dict:
        return {"choice": action.choice}

    def _parse_result(self, payload: Dict) -> StepResult[ReliabilityGameObservation]:
        observation = ReliabilityGameObservation(**payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ReliabilityGameState:
        return ReliabilityGameState(**payload)
