from dataclasses import dataclass

from openenv.core.rubrics import Rubric

from .tasks import TaskSpec


@dataclass(frozen=True)
class EpisodeMetrics:
    episode_length: int
    mean_reward: float
    avg_latency: float
    avg_cpu: float
    avg_error: float
    terminated: bool


@dataclass(frozen=True)
class GradeResult:
    completed: bool
    grader_score: float
    checks: dict[str, bool]


def grade_episode(task: TaskSpec, metrics: EpisodeMetrics) -> GradeResult:
    checks = {
        "survived_long_enough": metrics.episode_length >= task.min_episode_length,
        "latency_within_limit": metrics.avg_latency <= task.max_avg_latency,
        "cpu_within_limit": metrics.avg_cpu <= task.max_avg_cpu,
        "error_within_limit": metrics.avg_error <= task.max_avg_error,
        "reward_within_limit": metrics.mean_reward >= task.min_mean_reward,
        "avoided_failure_termination": not metrics.terminated,
    }
    grader_score = sum(1.0 for passed in checks.values() if passed) / len(checks)
    return GradeResult(
        completed=all(checks.values()),
        grader_score=grader_score,
        checks=checks,
    )


class ReliabilityTaskRubric(Rubric):
    def __init__(self, task: TaskSpec):
        super().__init__()
        self.task = task
        self._rewards: list[float] = []
        self._latencies: list[float] = []
        self._cpus: list[float] = []
        self._errors: list[float] = []

    def reset(self) -> None:
        self._rewards.clear()
        self._latencies.clear()
        self._cpus.clear()
        self._errors.clear()
        self.last_score = None

    def forward(self, action, observation) -> float:
        self._rewards.append(float(observation.base_reward))
        self._latencies.append(float(observation.latency))
        self._cpus.append(float(observation.cpu))
        self._errors.append(float(observation.error_rate))

        grade = grade_episode(
            self.task,
            EpisodeMetrics(
                episode_length=observation.step_index,
                mean_reward=sum(self._rewards) / len(self._rewards),
                avg_latency=sum(self._latencies) / len(self._latencies),
                avg_cpu=sum(self._cpus) / len(self._cpus),
                avg_error=sum(self._errors) / len(self._errors),
                terminated=bool(observation.done),
            ),
        )

        completion_bonus = 0.15 if observation.done and grade.completed else 0.0
        shaped_reward = min(
            1.0,
            0.75 * float(observation.base_reward)
            + 0.25 * grade.grader_score
            + completion_bonus,
        )
        return float(max(0.0, shaped_reward))
