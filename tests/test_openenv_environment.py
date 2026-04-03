from openenv_models import ReliabilityAction
from openenv_tasks import get_task_definition
from server.environment import LiveSystemReliabilityEnvironment


def test_reset_returns_balanced_task_observation():
    task = get_task_definition("balanced")
    env = LiveSystemReliabilityEnvironment(default_task_id="balanced")
    observation = env.reset(seed=42, task_id="balanced")

    assert observation.task_id == "balanced"
    assert observation.difficulty == "easy"
    assert observation.step_count == 0
    assert observation.remaining_steps == task.max_steps
    assert observation.reward == 0.0
    assert observation.done is False


def test_step_updates_state_and_returns_typed_reward_breakdown():
    env = LiveSystemReliabilityEnvironment(default_task_id="balanced")
    env.reset(seed=42, task_id="balanced")

    observation = env.step(ReliabilityAction(command="scale_up", rationale="smoke"))

    assert observation.step_count == 1
    assert observation.reward_breakdown is not None
    assert 0.0 <= observation.reward_breakdown.final_reward <= 1.0
    assert env.state.step_count == 1
    assert env.state.action_history == ["scale_up"]
