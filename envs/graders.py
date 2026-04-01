from dataclasses import asdict, dataclass

from envs.tasks import TaskSpec


@dataclass(frozen=True)
class EpisodeMetrics:
    episode_length: int
    mean_reward: float
    avg_latency: float
    avg_cpu: float
    avg_error: float
    avg_traffic: float
    terminated: bool


@dataclass(frozen=True)
class EpisodeGrade:
    completed: bool
    grader_score: float
    checks: dict[str, bool]
    summary: dict[str, float | bool]

    def to_dict(self) -> dict:
        return {
            "completed": self.completed,
            "grader_score": self.grader_score,
            "checks": dict(self.checks),
            "summary": dict(self.summary),
        }


def grade_episode(task: TaskSpec, metrics: EpisodeMetrics) -> EpisodeGrade:
    checks = {
        "survived_long_enough": metrics.episode_length >= task.min_episode_length,
        "latency_within_limit": metrics.avg_latency <= task.max_avg_latency,
        "cpu_within_limit": metrics.avg_cpu <= task.max_avg_cpu,
        "error_within_limit": metrics.avg_error <= task.max_avg_error,
        "reward_within_limit": metrics.mean_reward >= task.min_mean_reward,
        "avoided_failure_termination": not metrics.terminated,
    }
    grader_score = sum(1.0 for passed in checks.values() if passed) / len(checks)
    return EpisodeGrade(
        completed=all(checks.values()),
        grader_score=grader_score,
        checks=checks,
        summary=asdict(metrics),
    )
