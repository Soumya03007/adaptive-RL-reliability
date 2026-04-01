from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    difficulty: int
    title: str
    objective: str
    max_steps: int
    min_episode_length: int
    max_avg_latency: float
    max_avg_cpu: float
    max_avg_error: float
    min_mean_reward: float


TASKS = {
    "balanced": TaskSpec(
        name="balanced",
        difficulty=1,
        title="Stability Bootcamp",
        objective="Keep the platform healthy through a full baseline round.",
        max_steps=50,
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
        objective="Absorb a demand spike without letting latency or errors spiral.",
        max_steps=50,
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
        objective="Hold the service together while failures hit more frequently.",
        max_steps=50,
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
        objective="Finish the round with healthy service and disciplined resource use.",
        max_steps=50,
        min_episode_length=34,
        max_avg_latency=1.28,
        max_avg_cpu=1.12,
        max_avg_error=0.28,
        min_mean_reward=0.50,
    ),
}

TASK_SEQUENCE = tuple(
    task_name
    for task_name, _ in sorted(TASKS.items(), key=lambda item: item[1].difficulty)
)


def get_task_spec(task_name: str) -> TaskSpec:
    if task_name not in TASKS:
        available = ", ".join(sorted(TASKS))
        raise KeyError(f"Unknown task '{task_name}'. Available tasks: {available}")
    return TASKS[task_name]
