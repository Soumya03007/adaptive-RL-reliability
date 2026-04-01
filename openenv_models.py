from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


ActionCommand = Literal["scale_down", "no_op", "scale_up"]
DifficultyLevel = Literal["easy", "medium", "hard"]


class ReliabilityReward(BaseModel):
    """Typed reward payload for auditability and baseline reporting."""

    base_reward: float = Field(..., ge=0.0, le=1.0)
    health_reward: float = Field(..., ge=0.0, le=1.0)
    progress_bonus: float = Field(..., ge=0.0, le=1.0)
    outage_penalty: float = Field(..., ge=0.0, le=1.0)
    final_reward: float = Field(..., ge=0.0, le=1.0)


class ReliabilityAction(Action):
    """Action chosen by an agent controlling the live system."""

    command: ActionCommand = Field(
        ...,
        description="Scaling decision for the live system.",
    )
    rationale: str = Field(
        default="",
        description="Short explanation for why the action was selected.",
        max_length=280,
    )


class ReliabilityObservation(Observation):
    """Typed observation returned by the OpenEnv server."""

    task_id: str = Field(..., description="Stable task identifier.")
    task_title: str = Field(..., description="Human-readable task title.")
    difficulty: DifficultyLevel = Field(..., description="Task difficulty bucket.")
    objective: str = Field(..., description="What the agent is trying to accomplish.")
    latency: float = Field(..., ge=0.0, le=2.0)
    cpu: float = Field(..., ge=0.0, le=2.0)
    error_rate: float = Field(..., ge=0.0, le=1.0)
    traffic: float = Field(..., ge=0.0, le=1.5)
    step_count: int = Field(..., ge=0)
    remaining_steps: int = Field(..., ge=0)
    healthy_steps: int = Field(..., ge=0)
    healthy_ratio: float = Field(..., ge=0.0, le=1.0)
    current_grader_score: float = Field(..., ge=0.0, le=1.0)
    last_action: ActionCommand | None = Field(
        default=None,
        description="Previous scaling decision, if any.",
    )
    available_actions: list[ActionCommand] = Field(
        default_factory=lambda: ["scale_down", "no_op", "scale_up"],
        description="Valid actions for the next step.",
    )
    reward_breakdown: ReliabilityReward | None = Field(
        default=None,
        description="Structured reward decomposition for the last step.",
    )


class ReliabilityState(State):
    """Internal environment state exposed through OpenEnv state()."""

    task_id: str = Field(..., description="Current task identifier.")
    task_title: str = Field(..., description="Current task title.")
    difficulty: DifficultyLevel = Field(..., description="Difficulty of the current task.")
    objective: str = Field(..., description="Task objective.")
    max_steps: int = Field(..., ge=1)
    latency: float = Field(..., ge=0.0, le=2.0)
    cpu: float = Field(..., ge=0.0, le=2.0)
    error_rate: float = Field(..., ge=0.0, le=1.0)
    traffic: float = Field(..., ge=0.0, le=1.5)
    healthy_steps: int = Field(default=0, ge=0)
    action_changes: int = Field(default=0, ge=0)
    episode_return: float = Field(default=0.0, ge=0.0)
    current_grader_score: float = Field(default=0.0, ge=0.0, le=1.0)
    final_grader_score: float | None = Field(default=None, ge=0.0, le=1.0)
    terminated_reason: str | None = Field(default=None)
    action_history: list[ActionCommand] = Field(default_factory=list)
    reward_breakdown: ReliabilityReward | None = Field(default=None)
