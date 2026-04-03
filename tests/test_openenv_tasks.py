from openenv_tasks import get_task_definition, grade_trajectory


def test_grader_rewards_healthy_complete_trajectory():
    task = get_task_definition("balanced")
    trajectory = [
        {
            "latency": 0.7,
            "cpu": 0.7,
            "error_rate": 0.05,
            "traffic": 0.5,
            "reward": 0.9,
            "done": False,
        }
        for _ in range(task.max_steps)
    ]

    result = grade_trajectory(task, trajectory, action_changes=0)

    assert result.score >= task.pass_score
    assert result.passed is True
    assert result.healthy_ratio == 1.0
    assert result.survival_ratio == 1.0


def test_grader_penalizes_short_unhealthy_trajectory():
    task = get_task_definition("failure_heavy")
    trajectory = [
        {
            "latency": 1.9,
            "cpu": 1.7,
            "error_rate": 0.6,
            "traffic": 1.1,
            "reward": 0.1,
            "done": True,
        }
        for _ in range(3)
    ]

    result = grade_trajectory(task, trajectory, action_changes=2)

    assert result.score < task.pass_score
    assert result.passed is False
    assert result.terminated_early is True
