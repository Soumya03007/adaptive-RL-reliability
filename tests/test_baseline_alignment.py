import pytest

from agents.baseline_policy import choose_baseline_decision
from inference import run_episode
from openenv_tasks import list_task_definitions
from server.environment import LiveSystemReliabilityEnvironment


def test_task_catalog_covers_easy_medium_hard():
    tasks = list_task_definitions()

    assert [task.id for task in tasks] == ["balanced", "high_traffic", "failure_heavy"]
    assert [task.difficulty for task in tasks] == ["easy", "medium", "hard"]
    assert all(0.0 <= task.pass_score <= 1.0 for task in tasks)


def test_baseline_policy_covers_scale_down_hold_and_scale_up():
    env = LiveSystemReliabilityEnvironment(default_task_id="balanced")
    observation = env.reset(seed=42, task_id="balanced")

    scale_up = choose_baseline_decision(
        "balanced",
        observation.model_copy(
            update={
                "latency": 1.2,
                "cpu": 1.1,
                "error_rate": 0.18,
                "traffic": 1.0,
            }
        ),
    )
    hold = choose_baseline_decision(
        "balanced",
        observation.model_copy(
            update={
                "latency": 0.35,
                "cpu": 0.28,
                "error_rate": 0.015,
                "traffic": 0.22,
            }
        ),
    )
    scale_down = choose_baseline_decision(
        "balanced",
        observation.model_copy(
            update={
                "latency": 0.25,
                "cpu": 0.20,
                "error_rate": 0.01,
                "traffic": 0.15,
            }
        ),
    )

    assert scale_up.command == "scale_up"
    assert hold.command == "no_op"
    assert scale_down.command == "scale_down"


@pytest.mark.parametrize("task_id,seed", [("balanced", 42), ("high_traffic", 142), ("failure_heavy", 242)])
def test_inference_episode_is_reproducible(task_id: str, seed: int):
    first = run_episode(task_id=task_id, episode_seed=seed)
    second = run_episode(task_id=task_id, episode_seed=seed)

    assert first["score"] == second["score"]
    assert first["steps"] == second["steps"]
    assert first["success"] == second["success"]
    assert first["rewards"] == second["rewards"]
