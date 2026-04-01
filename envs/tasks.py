from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    difficulty: int
    title: str
    objective: str
    min_episode_length: int
    max_avg_latency: float
    max_avg_cpu: float
    max_avg_error: float
    min_mean_reward: float

    def to_dict(self) -> dict:
        return asdict(self)


TASK_SPECS = {
    "balanced": TaskSpec(
        name="balanced",
        difficulty=1,
        title="Stability Bootcamp",
        objective="Keep the service healthy for a full episode under baseline traffic.",
        min_episode_length=30,
        max_avg_latency=1.25,
        max_avg_cpu=1.30,
        max_avg_error=0.28,
        min_mean_reward=0.52,
    ),
    "high_traffic": TaskSpec(
        name="high_traffic",
        difficulty=2,
        title="Traffic Surge",
        objective="Survive a traffic spike while keeping average latency and error under control.",
        min_episode_length=32,
        max_avg_latency=1.32,
        max_avg_cpu=1.40,
        max_avg_error=0.30,
        min_mean_reward=0.48,
    ),
    "failure_heavy": TaskSpec(
        name="failure_heavy",
        difficulty=3,
        title="Chaos Monkey",
        objective="Maintain service reliability despite frequent failures and noisy dynamics.",
        min_episode_length=28,
        max_avg_latency=1.42,
        max_avg_cpu=1.48,
        max_avg_error=0.34,
        min_mean_reward=0.44,
    ),
    "cost_sensitive": TaskSpec(
        name="cost_sensitive",
        difficulty=4,
        title="Efficiency Gauntlet",
        objective="Stay stable while minimizing wasteful scaling and excess CPU usage.",
        min_episode_length=34,
        max_avg_latency=1.28,
        max_avg_cpu=1.12,
        max_avg_error=0.28,
        min_mean_reward=0.50,
    ),
}

DEFAULT_TASK_ORDER = tuple(
    task_name
    for task_name, _ in sorted(
        TASK_SPECS.items(),
        key=lambda item: item[1].difficulty,
    )
)


def get_task_spec(task_name: str) -> TaskSpec:
    if task_name not in TASK_SPECS:
        available = ", ".join(sorted(TASK_SPECS))
        raise KeyError(f"Unknown task '{task_name}'. Available tasks: {available}")
    return TASK_SPECS[task_name]
