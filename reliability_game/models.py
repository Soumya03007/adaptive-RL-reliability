from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class ReliabilityGameAction(Action):
    choice: Literal["scale_down", "hold", "scale_up"] = Field(
        ...,
        description="Scaling move selected by the agent for the current step.",
    )


class ReliabilityGameObservation(Observation):
    task_name: str = Field(default="", description="Current task identifier.")
    task_title: str = Field(default="", description="Human-friendly task name.")
    objective: str = Field(default="", description="What the agent is trying to achieve.")
    difficulty: int = Field(default=1, description="Relative task difficulty, starting at 1.")
    latency: float = Field(default=0.0, description="Current latency level.")
    cpu: float = Field(default=0.0, description="Current CPU load.")
    error_rate: float = Field(default=0.0, description="Current error rate.")
    traffic: float = Field(default=0.0, description="Current traffic intensity.")
    step_index: int = Field(default=0, description="Current step index within the episode.")
    steps_remaining: int = Field(default=0, description="Steps left before the time limit.")
    last_action: str = Field(default="none", description="Most recent action label.")
    base_reward: float = Field(default=0.0, description="Raw normalized reward before rubric shaping.")
    grader_score: float = Field(default=0.0, description="Current task grader score in the range 0-1.")
    task_completed: bool = Field(default=False, description="Whether the episode currently satisfies all task checks.")
    checks: dict[str, bool] = Field(
        default_factory=dict,
        description="Named grader checks for task completion.",
    )


class ReliabilityGameState(State):
    task_name: str = Field(default="", description="Active task identifier.")
    difficulty: int = Field(default=1, description="Active task difficulty.")
    cumulative_reward: float = Field(default=0.0, description="Sum of step rewards so far.")
    success_steps: int = Field(default=0, description="Steps that passed all instantaneous task checks.")
    last_action: str = Field(default="none", description="Latest action label.")
