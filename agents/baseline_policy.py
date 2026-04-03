from __future__ import annotations

from pydantic import BaseModel, Field

from openenv_models import ActionCommand, ReliabilityObservation
from openenv_tasks import get_task_definition


class BaselineDecision(BaseModel):
    command: ActionCommand = Field(...)
    rationale: str = Field(default="", max_length=280)


def choose_baseline_decision(
    task_id: str,
    observation: ReliabilityObservation,
) -> BaselineDecision:
    task = get_task_definition(task_id)

    if (
        observation.traffic >= 0.28
        or observation.error_rate >= task.error_target * 0.20
        or observation.latency >= task.latency_target * 0.45
        or observation.cpu >= task.cpu_target * 0.45
    ):
        return BaselineDecision(
            command="scale_up",
            rationale="protect-capacity",
        )

    if (
        observation.traffic <= 0.18
        and observation.latency <= task.latency_target * 0.30
        and observation.cpu <= task.cpu_target * 0.30
        and observation.error_rate <= max(task.error_target * 0.10, 0.01)
    ):
        return BaselineDecision(
            command="scale_down",
            rationale="trim-excess-capacity",
        )

    return BaselineDecision(
        command="no_op",
        rationale="hold-steady",
    )
