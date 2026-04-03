from __future__ import annotations

from statistics import mean

from pydantic import BaseModel, Field

from configs.loader import get_default_task_name, load_env_config


class TaskDefinition(BaseModel):
    id: str
    title: str
    difficulty: str
    objective: str
    env_variant: str
    max_steps: int = Field(..., ge=1)
    latency_target: float = Field(..., gt=0.0, le=2.0)
    cpu_target: float = Field(..., gt=0.0, le=2.0)
    error_target: float = Field(..., ge=0.0, le=1.0)
    cpu_efficiency_min: float = Field(..., ge=0.0, le=2.0)
    cpu_efficiency_max: float = Field(..., ge=0.0, le=2.0)
    max_scale_fraction: float = Field(..., ge=0.0, le=1.0)
    healthy_ratio_target: float = Field(..., ge=0.0, le=1.0)
    pass_score: float = Field(..., ge=0.0, le=1.0)
    score_weights: dict[str, float] = Field(default_factory=dict)
    description: str = Field(default="")

    def env_config(self) -> dict:
        return load_env_config(self.env_variant)


class GraderResult(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    survival_ratio: float = Field(..., ge=0.0, le=1.0)
    healthy_ratio: float = Field(..., ge=0.0, le=1.0)
    slo_score: float = Field(..., ge=0.0, le=1.0)
    efficiency_score: float = Field(..., ge=0.0, le=1.0)
    control_score: float = Field(..., ge=0.0, le=1.0)
    action_discipline: float = Field(..., ge=0.0, le=1.0)
    terminated_early: bool
    summary: str


TASKS: dict[str, TaskDefinition] = {
    "balanced": TaskDefinition(
        id="balanced",
        title="Balanced Autoscaling",
        difficulty="easy",
        objective=(
            "Keep a production service healthy during routine traffic. "
            "Avoid outages and keep scaling churn low."
        ),
        env_variant="balanced",
        max_steps=18,
        latency_target=1.05,
        cpu_target=1.05,
        error_target=0.16,
        cpu_efficiency_min=0.25,
        cpu_efficiency_max=0.75,
        max_scale_fraction=0.60,
        healthy_ratio_target=0.70,
        pass_score=0.80,
        score_weights={
            "survival_ratio": 0.35,
            "healthy_ratio": 0.25,
            "slo_score": 0.20,
            "efficiency_score": 0.15,
            "control_score": 0.05,
        },
        description=(
            "Easy task: hold the service inside standard SLO thresholds under normal load."
        ),
    ),
    "high_traffic": TaskDefinition(
        id="high_traffic",
        title="Rush-Hour Capacity Management",
        difficulty="medium",
        objective=(
            "Keep latency and errors controlled during a sustained traffic surge. "
            "Scale fast enough to absorb load without oscillating."
        ),
        env_variant="high_traffic",
        max_steps=22,
        latency_target=1.12,
        cpu_target=1.10,
        error_target=0.18,
        cpu_efficiency_min=0.50,
        cpu_efficiency_max=0.95,
        max_scale_fraction=0.65,
        healthy_ratio_target=0.60,
        pass_score=0.72,
        score_weights={
            "survival_ratio": 0.35,
            "healthy_ratio": 0.20,
            "slo_score": 0.15,
            "efficiency_score": 0.20,
            "control_score": 0.10,
        },
        description=(
            "Medium task: survive a heavier traffic profile while staying close to service SLOs."
        ),
    ),
    "failure_heavy": TaskDefinition(
        id="failure_heavy",
        title="Incident Response Under Failures",
        difficulty="hard",
        objective=(
            "Maintain a degraded live system through repeated failure spikes. "
            "Prioritize survival first, then recover toward healthy operating bounds."
        ),
        env_variant="failure_heavy",
        max_steps=26,
        latency_target=1.22,
        cpu_target=1.18,
        error_target=0.24,
        cpu_efficiency_min=0.55,
        cpu_efficiency_max=1.05,
        max_scale_fraction=0.70,
        healthy_ratio_target=0.45,
        pass_score=0.62,
        score_weights={
            "survival_ratio": 0.35,
            "healthy_ratio": 0.20,
            "slo_score": 0.15,
            "efficiency_score": 0.15,
            "control_score": 0.15,
        },
        description=(
            "Hard task: handle incident-like failure dynamics while minimizing total instability."
        ),
    ),
}


def list_task_definitions() -> list[TaskDefinition]:
    return [TASKS[name] for name in TASKS]


def get_task_definition(task_id: str | None = None) -> TaskDefinition:
    selected = task_id or get_default_task_name()
    if selected not in TASKS:
        available = ", ".join(sorted(TASKS))
        raise KeyError(f"Unknown task '{selected}'. Available tasks: {available}")
    return TASKS[selected]


def metric_health(value: float, target: float, hard_cap: float) -> float:
    if value <= target:
        return 1.0
    if value >= hard_cap:
        return 0.0
    return max(0.0, 1.0 - ((value - target) / (hard_cap - target)))


def metric_band(value: float, lower: float, upper: float, hard_cap: float) -> float:
    if lower <= value <= upper:
        return 1.0
    if value < lower:
        return max(0.0, value / max(lower, 1e-6))
    if value >= hard_cap:
        return 0.0
    return max(0.0, 1.0 - ((value - upper) / max(hard_cap - upper, 1e-6)))


def grade_trajectory(
    task: TaskDefinition,
    trajectory: list[dict],
    action_changes: int,
    max_steps: int | None = None,
) -> GraderResult:
    task_steps = max_steps or task.max_steps
    if not trajectory:
        return GraderResult(
            score=0.0,
            passed=False,
            survival_ratio=0.0,
            healthy_ratio=0.0,
            slo_score=0.0,
            efficiency_score=0.0,
            control_score=1.0,
            action_discipline=1.0,
            terminated_early=False,
            summary="No steps executed.",
        )

    step_count = len(trajectory)
    survival_ratio = min(step_count, task_steps) / task_steps

    per_step_slo = []
    per_step_efficiency = []
    healthy_steps = 0
    scale_actions = 0
    for step in trajectory:
        latency = metric_health(step["latency"], task.latency_target, 2.0)
        cpu = metric_health(step["cpu"], task.cpu_target, 2.0)
        error = metric_health(step["error_rate"], task.error_target, 1.0)
        step_slo = mean([latency, cpu, error])
        per_step_slo.append(step_slo)
        per_step_efficiency.append(
            metric_band(
                step["cpu"],
                task.cpu_efficiency_min,
                task.cpu_efficiency_max,
                2.0,
            )
        )
        if latency == 1.0 and cpu == 1.0 and error == 1.0:
            healthy_steps += 1
        if step.get("action") in {"scale_up", "scale_down"}:
            scale_actions += 1

    healthy_ratio = healthy_steps / task_steps
    slo_score = sum(per_step_slo) / task_steps
    efficiency_score = sum(per_step_efficiency) / task_steps
    max_changes = max(task_steps - 1, 1)
    change_score = max(0.0, 1.0 - (min(action_changes, max_changes) / max_changes))
    scale_fraction = scale_actions / task_steps
    if scale_fraction <= task.max_scale_fraction:
        control_score = 1.0
    else:
        control_score = max(
            0.0,
            1.0 - ((scale_fraction - task.max_scale_fraction) / max(1.0 - task.max_scale_fraction, 1e-6)),
        )
    action_discipline = 0.5 * change_score + 0.5 * control_score

    weights = task.score_weights
    raw_score = (
        weights["survival_ratio"] * survival_ratio
        + weights["healthy_ratio"] * healthy_ratio
        + weights["slo_score"] * slo_score
        + weights["efficiency_score"] * efficiency_score
        + weights["control_score"] * control_score
    )
    score = max(0.0, min(1.0, raw_score))
    terminated_early = step_count < task_steps and bool(trajectory[-1].get("done"))
    passed = (
        survival_ratio >= 1.0
        and healthy_ratio >= task.healthy_ratio_target
        and score >= task.pass_score
    )

    summary = (
        f"survival={survival_ratio:.2f}, healthy={healthy_ratio:.2f}, "
        f"slo={slo_score:.2f}, efficiency={efficiency_score:.2f}, control={control_score:.2f}"
    )
    return GraderResult(
        score=score,
        passed=passed,
        survival_ratio=survival_ratio,
        healthy_ratio=healthy_ratio,
        slo_score=slo_score,
        efficiency_score=efficiency_score,
        control_score=control_score,
        action_discipline=action_discipline,
        terminated_early=terminated_early,
        summary=summary,
    )
