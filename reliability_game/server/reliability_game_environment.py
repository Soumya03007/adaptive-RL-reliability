import random
from uuid import uuid4

import numpy as np
from openenv.core.env_server.interfaces import Environment

from reliability_game.graders import EpisodeMetrics, ReliabilityTaskRubric, grade_episode
from reliability_game.models import (
    ReliabilityGameAction,
    ReliabilityGameObservation,
    ReliabilityGameState,
)
from reliability_game.tasks import TASK_SEQUENCE, get_task_spec


ACTION_TO_INDEX = {
    "scale_down": 0,
    "hold": 1,
    "scale_up": 2,
}


class ReliabilityGameEnvironment(
    Environment[ReliabilityGameAction, ReliabilityGameObservation, ReliabilityGameState]
):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self.rng = random.Random()
        self._set_task(TASK_SEQUENCE[0])
        self._state = ReliabilityGameState(
            episode_id=str(uuid4()),
            step_count=0,
            task_name=self.task.name,
            difficulty=self.task.difficulty,
        )
        self._rewards: list[float] = []
        self._latencies: list[float] = []
        self._cpus: list[float] = []
        self._errors: list[float] = []
        super().__init__(rubric=self.rubric)
        self._reset_state_variables()

    def _set_task(self, task_name: str) -> None:
        self.task = get_task_spec(task_name)
        self.rubric = ReliabilityTaskRubric(self.task)

    def _reset_state_variables(self) -> None:
        self.latency = self.rng.uniform(0.3, 0.6)
        self.cpu = self.rng.uniform(0.3, 0.6)
        self.error_rate = self.rng.uniform(0.0, 0.1)
        if self.task.name == "balanced":
            traffic_base = (0.15, 0.45)
        elif self.task.name == "high_traffic":
            traffic_base = (0.6, 0.95)
        else:
            traffic_base = (0.3, 0.7)
        self.traffic = self.rng.uniform(*traffic_base)
        self.last_action = "none"

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_name: str | None = None,
        **kwargs,
    ) -> ReliabilityGameObservation:
        if seed is not None:
            self.rng.seed(seed)
        if task_name is not None:
            self._set_task(task_name)

        self._reset_rubric()
        self._rewards.clear()
        self._latencies.clear()
        self._cpus.clear()
        self._errors.clear()
        self._reset_state_variables()
        self._state = ReliabilityGameState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=self.task.name,
            difficulty=self.task.difficulty,
            cumulative_reward=0.0,
            success_steps=0,
            last_action="none",
        )
        return self._build_observation(done=False, reward=0.0, grader_score=0.0, checks={})

    def step(
        self,
        action: ReliabilityGameAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> ReliabilityGameObservation:
        action_idx = ACTION_TO_INDEX[action.choice]
        previous_action = self.last_action

        self._apply_action(action_idx)
        self._apply_traffic()
        self._apply_noise()
        self._apply_failures()
        self._clamp_state()

        self._state.step_count += 1
        self.last_action = action.choice

        base_reward = self._compute_base_reward(action_idx, previous_action)
        self._rewards.append(base_reward)
        self._latencies.append(self.latency)
        self._cpus.append(self.cpu)
        self._errors.append(self.error_rate)

        done = (
            self.latency >= 2.0
            or self.cpu >= 2.0
            or self.error_rate >= 1.0
            or self._state.step_count >= self.task.max_steps
        )

        preview_grade = grade_episode(
            self.task,
            EpisodeMetrics(
                episode_length=self._state.step_count,
                mean_reward=float(sum(self._rewards) / len(self._rewards)),
                avg_latency=float(sum(self._latencies) / len(self._latencies)),
                avg_cpu=float(sum(self._cpus) / len(self._cpus)),
                avg_error=float(sum(self._errors) / len(self._errors)),
                terminated=done and self._state.step_count < self.task.max_steps,
            ),
        )
        observation = self._build_observation(
            done=done,
            reward=base_reward,
            grader_score=preview_grade.grader_score,
            checks=preview_grade.checks,
            task_completed=preview_grade.completed,
        )
        shaped_reward = self._apply_rubric(action, observation)
        observation.reward = shaped_reward
        observation.base_reward = base_reward

        self._state.cumulative_reward += shaped_reward
        self._state.last_action = action.choice
        if preview_grade.completed:
            self._state.success_steps += 1
        return observation

    @property
    def state(self) -> ReliabilityGameState:
        return self._state

    def _build_observation(
        self,
        done: bool,
        reward: float,
        grader_score: float,
        checks: dict[str, bool],
        task_completed: bool = False,
    ) -> ReliabilityGameObservation:
        return ReliabilityGameObservation(
            done=done,
            reward=reward,
            task_name=self.task.name,
            task_title=self.task.title,
            objective=self.task.objective,
            difficulty=self.task.difficulty,
            latency=float(self.latency),
            cpu=float(self.cpu),
            error_rate=float(self.error_rate),
            traffic=float(self.traffic),
            step_index=self._state.step_count,
            steps_remaining=max(0, self.task.max_steps - self._state.step_count),
            last_action=self.last_action,
            base_reward=float(reward),
            grader_score=float(grader_score),
            task_completed=task_completed,
            checks=dict(checks),
            metadata={
                "tasks": list(TASK_SEQUENCE),
                "success_steps": self._state.success_steps,
            },
        )

    def _apply_action(self, action_idx: int) -> None:
        if action_idx == 2:
            self.latency *= self.rng.uniform(0.5, 0.8)
            self.cpu *= self.rng.uniform(1.1, 1.4)
        elif action_idx == 0:
            self.latency *= self.rng.uniform(1.2, 1.6)
            self.cpu *= self.rng.uniform(0.7, 1.0)

    def _apply_traffic(self) -> None:
        if self.task.name == "high_traffic":
            volatility = (1.0, 1.3)
        else:
            volatility = (0.9, 1.2)
        self.traffic *= self.rng.uniform(*volatility)
        self.latency *= 1 + 0.5 * self.traffic
        self.cpu *= 1 + 0.7 * self.traffic

    def _apply_noise(self) -> None:
        if self.task.name == "failure_heavy":
            latency_noise = (0.95, 1.25)
            cpu_noise = (0.95, 1.15)
        else:
            latency_noise = (0.9, 1.2)
            cpu_noise = (0.9, 1.1)
        self.latency *= self.rng.uniform(*latency_noise)
        self.cpu *= self.rng.uniform(*cpu_noise)

    def _apply_failures(self) -> None:
        if self.task.name == "balanced":
            failure_prob = 0.04
            impact = (1.2, 1.8)
        elif self.task.name == "failure_heavy":
            failure_prob = 0.2
            impact = (1.8, 2.8)
        else:
            failure_prob = 0.1
            impact = (1.5, 2.5)

        if self.rng.random() < failure_prob:
            self.latency *= self.rng.uniform(*impact)
            self.error_rate += 0.2

    def _clamp_state(self) -> None:
        self.latency = float(np.clip(self.latency, 0.0, 2.0))
        self.cpu = float(np.clip(self.cpu, 0.0, 2.0))
        self.error_rate = float(np.clip(self.error_rate, 0.0, 1.0))
        self.traffic = float(np.clip(self.traffic, 0.0, 1.5))

    def _compute_base_reward(self, action_idx: int, previous_action: str) -> float:
        cpu_weight = 2.0 if self.task.name == "cost_sensitive" else 1.0
        scale_action_weight = 0.08 if self.task.name == "cost_sensitive" else 0.03
        action_change_weight = 0.12 if self.task.name == "cost_sensitive" else 0.08
        max_penalty = 3.0 * 2.0 + 5.0 * 1.0 + cpu_weight * 2.0 + scale_action_weight + action_change_weight
        penalty = 3.0 * self.latency + 5.0 * self.error_rate + cpu_weight * self.cpu

        if action_idx in (0, 2):
            penalty += scale_action_weight
        if previous_action != "none" and previous_action != self.last_action:
            penalty += action_change_weight

        reward = 1.0 - penalty / max_penalty
        return float(np.clip(reward, 0.0, 1.0))

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata.name = "ReliabilityGameEnvironment"
        metadata.description = (
            "Mini-game RL environment for reliability-aware service scaling with "
            "ordered tasks, automated graders, and reward shaping."
        )
        metadata.version = "0.1.0"
        return metadata
