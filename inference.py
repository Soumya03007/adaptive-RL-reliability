import json
import os
from pathlib import Path

from agents.baseline_policy import choose_baseline_decision
from openenv_models import ReliabilityAction
from openenv_tasks import get_task_definition, list_task_definitions
from server.environment import LiveSystemReliabilityEnvironment


BENCHMARK = os.getenv("BENCHMARK_NAME", "adaptive-rl-reliability")
POLICY_NAME = os.getenv("MODEL_NAME", "deterministic-rule-v1")
TASKS = os.getenv(
    "TASKS",
    ",".join(task.id for task in list_task_definitions()),
)
BASE_SEED = int(os.getenv("BASE_SEED", "42"))
OUTPUT_PATH = Path(os.getenv("INFERENCE_OUTPUT", "outputs/inference_report.json"))


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    done_text = str(done).lower()
    error_text = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_text} error={error_text}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_text}",
        flush=True,
    )


def run_episode(task_id: str, episode_seed: int) -> dict:
    task = get_task_definition(task_id)
    env = LiveSystemReliabilityEnvironment(default_task_id=task_id)
    rewards: list[float] = []
    steps_taken = 0
    success = False
    final_score = 0.0
    last_error: str | None = None

    log_start(task=task_id, env=BENCHMARK, model=POLICY_NAME)

    try:
        observation = env.reset(seed=episode_seed, task_id=task_id)
        while not observation.done:
            decision = choose_baseline_decision(task_id=task_id, observation=observation)
            observation = env.step(
                ReliabilityAction(
                    command=decision.command,
                    rationale=decision.rationale,
                )
            )
            reward = float(observation.reward or 0.0)
            rewards.append(reward)
            steps_taken = observation.step_count
            log_step(
                step=steps_taken,
                action=decision.command,
                reward=reward,
                done=observation.done,
                error=None,
            )

        final_score = float(env.state.final_grader_score or 0.0)
        success = final_score >= task.pass_score
    except Exception as exc:
        last_error = str(exc)
    finally:
        close_error = None
        close = getattr(env, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                close_error = str(exc)
        if close_error and last_error is None:
            last_error = close_error
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    return {
        "task_id": task_id,
        "episode_seed": episode_seed,
        "policy": POLICY_NAME,
        "score": final_score,
        "steps": steps_taken,
        "success": success,
        "error": last_error,
        "rewards": rewards,
    }


def main() -> None:
    selected_tasks = [part.strip() for part in TASKS.split(",") if part.strip()]
    for task_id in selected_tasks:
        get_task_definition(task_id)

    rows: list[dict] = []
    for task_offset, task_id in enumerate(selected_tasks):
        episode_seed = BASE_SEED + task_offset * 100
        rows.append(run_episode(task_id=task_id, episode_seed=episode_seed))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "benchmark": BENCHMARK,
                "policy": POLICY_NAME,
                "tasks": selected_tasks,
                "base_seed": BASE_SEED,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
