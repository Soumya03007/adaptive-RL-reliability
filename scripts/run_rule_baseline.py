import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.baseline_policy import choose_baseline_decision
from openenv_models import ReliabilityAction
from openenv_tasks import list_task_definitions
from server.environment import LiveSystemReliabilityEnvironment


def run_episode(task_id: str, seed: int) -> dict:
    env = LiveSystemReliabilityEnvironment(default_task_id=task_id)
    observation = env.reset(seed=seed, task_id=task_id)

    while not observation.done:
        decision = choose_baseline_decision(
            task_id=task_id,
            observation=observation,
        )
        observation = env.step(
            ReliabilityAction(
                command=decision.command,
                rationale=decision.rationale,
            )
        )

    state = env.state
    return {
        "task_id": task_id,
        "seed": seed,
        "policy": "deterministic-rule-v1",
        "score": state.final_grader_score,
        "episode_return": state.episode_return,
        "steps": state.step_count,
        "terminated_reason": state.terminated_reason,
    }


def summarize(rows: list[dict]) -> dict[str, float]:
    scores = [row["score"] for row in rows if row["score"] is not None]
    return {
        "mean": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Run a deterministic local baseline for smoke testing.")
    parser.add_argument(
        "--tasks",
        default=",".join(task.id for task in list_task_definitions()),
        help="Comma-separated task ids",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/rule_baseline.json"))
    args = parser.parse_args()

    selected_tasks = [part.strip() for part in args.tasks.split(",") if part.strip()]
    rows = []
    task_summaries = {}

    for task_offset, task_id in enumerate(selected_tasks):
        task_rows = []
        for episode_index in range(args.episodes):
            row = run_episode(task_id, args.seed + task_offset * 100 + episode_index)
            rows.append(row)
            task_rows.append(row)
            print(
                f"Task {task_id} | Seed {row['seed']} | Score {row['score']:.3f} "
                f"| Return {row['episode_return']:.3f} | Steps {row['steps']}"
            )

        task_summaries[task_id] = summarize(task_rows)
        summary = task_summaries[task_id]
        print(
            f"TaskSummary {task_id} mean={summary['mean']:.3f} "
            f"min={summary['min']:.3f} max={summary['max']:.3f}"
        )

    report = {
        "episodes_per_task": args.episodes,
        "base_seed": args.seed,
        "tasks": selected_tasks,
        "task_summaries": task_summaries,
        "overall": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Overall mean={report['overall']['mean']:.3f} "
        f"min={report['overall']['min']:.3f} max={report['overall']['max']:.3f}"
    )
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
