from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from envs.reliability_env import ACTION_LABELS, ReliabilityEnv
from openenv_models import (
    ReliabilityAction,
    ReliabilityObservation,
    ReliabilityReward,
    ReliabilityState,
)
from openenv_tasks import GraderResult, get_task_definition, grade_trajectory


ACTION_TO_INDEX = {label: index for index, label in ACTION_LABELS.items()}


class LiveSystemReliabilityEnvironment(
    Environment[ReliabilityAction, ReliabilityObservation, ReliabilityState]
):
    """OpenEnv wrapper around the reliability-control simulator."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, default_task_id: str | None = None):
        super().__init__()
        self.default_task_id = get_task_definition(default_task_id).id
        self._task = get_task_definition(self.default_task_id)
        self._env = ReliabilityEnv(self._task.env_config())
        self._trajectory: list[dict] = []
        self._state = self._build_state(episode_id=str(uuid4()))

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        task_name: str | None = None,
        **_: object,
    ) -> ReliabilityObservation:
        selected_task = task_id or task_name or self.default_task_id
        self._task = get_task_definition(selected_task)
        self._env = ReliabilityEnv(self._task.env_config())
        self._env.reset(seed=seed)
        self._trajectory = []
        self._state = self._build_state(episode_id=episode_id or str(uuid4()))
        return self._build_observation(
            reward_breakdown=ReliabilityReward(
                base_reward=0.0,
                health_reward=0.0,
                progress_bonus=0.0,
                outage_penalty=0.0,
                final_reward=0.0,
            ),
            grader_result=grade_trajectory(self._task, [], action_changes=0),
            done=False,
        )

    def step(
        self,
        action: ReliabilityAction,
        timeout_s: float | None = None,
        **_: object,
    ) -> ReliabilityObservation:
        del timeout_s

        action_index = ACTION_TO_INDEX[action.command]
        _, base_reward, simulator_done, info = self._env.step(action_index)

        previous_action = self._state.action_history[-1] if self._state.action_history else None
        self._state.step_count += 1
        self._state.action_history.append(action.command)
        if previous_action is not None and previous_action != action.command:
            self._state.action_changes += 1

        snapshot = info["metrics"]
        health_reward = self._health_reward(snapshot)
        progress_bonus = self._progress_bonus(health_reward)
        outage_penalty = (
            0.35 if simulator_done and self._state.step_count < self._task.max_steps else 0.0
        )
        final_reward = max(
            0.0,
            min(
                1.0,
                0.55 * base_reward + 0.35 * health_reward + 0.10 * progress_bonus - outage_penalty,
            ),
        )
        reward_breakdown = ReliabilityReward(
            base_reward=base_reward,
            health_reward=health_reward,
            progress_bonus=progress_bonus,
            outage_penalty=outage_penalty,
            final_reward=final_reward,
        )

        if self._is_healthy(snapshot):
            self._state.healthy_steps += 1

        done = simulator_done or self._state.step_count >= self._task.max_steps
        terminated_reason = info.get("done_reason")
        if not simulator_done and self._state.step_count >= self._task.max_steps:
            terminated_reason = "max_steps_reached"

        self._trajectory.append(
            {
                **snapshot,
                "reward": final_reward,
                "done": done,
                "action": action.command,
            }
        )

        grader_result = grade_trajectory(
            self._task,
            self._trajectory,
            action_changes=self._state.action_changes,
            max_steps=self._task.max_steps,
        )

        self._state.latency = snapshot["latency"]
        self._state.cpu = snapshot["cpu"]
        self._state.error_rate = snapshot["error_rate"]
        self._state.traffic = snapshot["traffic"]
        self._state.episode_return += final_reward
        self._state.current_grader_score = grader_result.score
        self._state.final_grader_score = grader_result.score if done else None
        self._state.terminated_reason = terminated_reason if done else None
        self._state.reward_breakdown = reward_breakdown

        return self._build_observation(
            reward_breakdown=reward_breakdown,
            grader_result=grader_result,
            done=done,
            terminated_reason=terminated_reason if done else None,
        )

    @property
    def state(self) -> ReliabilityState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="adaptive-rl-reliability",
            description=(
                "A real-world autoscaling simulator for reliability-aware control of a live service."
            ),
            version="0.2.0",
        )

    def _build_state(self, episode_id: str) -> ReliabilityState:
        snapshot = self._env.snapshot()
        return ReliabilityState(
            episode_id=episode_id,
            step_count=0,
            task_id=self._task.id,
            task_title=self._task.title,
            difficulty=self._task.difficulty,
            objective=self._task.objective,
            max_steps=self._task.max_steps,
            latency=snapshot["latency"],
            cpu=snapshot["cpu"],
            error_rate=snapshot["error_rate"],
            traffic=snapshot["traffic"],
            healthy_steps=0,
            action_changes=0,
            episode_return=0.0,
            current_grader_score=0.0,
            final_grader_score=None,
            terminated_reason=None,
            action_history=[],
            reward_breakdown=None,
        )

    def _build_observation(
        self,
        reward_breakdown: ReliabilityReward,
        grader_result: GraderResult,
        done: bool,
        terminated_reason: str | None = None,
    ) -> ReliabilityObservation:
        snapshot = self._env.snapshot()
        last_action = self._state.action_history[-1] if self._state.action_history else None
        return ReliabilityObservation(
            task_id=self._task.id,
            task_title=self._task.title,
            difficulty=self._task.difficulty,
            objective=self._task.objective,
            latency=snapshot["latency"],
            cpu=snapshot["cpu"],
            error_rate=snapshot["error_rate"],
            traffic=snapshot["traffic"],
            step_count=self._state.step_count,
            remaining_steps=max(self._task.max_steps - self._state.step_count, 0),
            healthy_steps=self._state.healthy_steps,
            healthy_ratio=grader_result.healthy_ratio,
            current_grader_score=grader_result.score,
            last_action=last_action,
            reward=reward_breakdown.final_reward,
            done=done,
            reward_breakdown=reward_breakdown,
            metadata={
                "grader": grader_result.model_dump(),
                "pass_score": self._task.pass_score,
                "healthy_ratio_target": self._task.healthy_ratio_target,
                "terminated_reason": terminated_reason,
            },
        )

    def _health_reward(self, snapshot: dict[str, float]) -> float:
        checks = [
            1.0 if snapshot["latency"] <= self._task.latency_target else 0.0,
            1.0 if snapshot["cpu"] <= self._task.cpu_target else 0.0,
            1.0 if snapshot["error_rate"] <= self._task.error_target else 0.0,
        ]
        return sum(checks) / len(checks)

    def _is_healthy(self, snapshot: dict[str, float]) -> bool:
        return (
            snapshot["latency"] <= self._task.latency_target
            and snapshot["cpu"] <= self._task.cpu_target
            and snapshot["error_rate"] <= self._task.error_target
        )

    def _progress_bonus(self, health_reward: float) -> float:
        if health_reward < (2.0 / 3.0):
            return 0.0
        return min(self._state.step_count / self._task.max_steps, 1.0)


__all__ = ["LiveSystemReliabilityEnvironment"]
