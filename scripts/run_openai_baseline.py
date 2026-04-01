import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from pydantic import BaseModel, Field

from openenv_models import ActionCommand, ReliabilityAction, ReliabilityObservation
from openenv_tasks import get_task_definition, list_task_definitions
from server.live_system_environment import LiveSystemReliabilityEnvironment


DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")


class PlannerDecision(BaseModel):
    command: ActionCommand = Field(description="One of: scale_down, no_op, scale_up")
    rationale: str = Field(default="", max_length=280)


def build_prompt(task_id: str, observation: ReliabilityObservation) -> str:
    return (
        f"Task: {task_id}\n"
        f"Difficulty: {observation.difficulty}\n"
        f"Objective: {observation.objective}\n"
        f"Step: {observation.step_count}/{observation.step_count + observation.remaining_steps}\n"
        f"Latency: {observation.latency:.3f}\n"
        f"CPU: {observation.cpu:.3f}\n"
        f"Error rate: {observation.error_rate:.3f}\n"
        f"Traffic: {observation.traffic:.3f}\n"
        f"Healthy ratio so far: {observation.healthy_ratio:.3f}\n"
        f"Current grader score: {observation.current_grader_score:.3f}\n"
        f"Last action: {observation.last_action}\n"
        "Choose the next scaling action. Prefer preventing outages, then improving score, "
        "and avoid unnecessary action flips."
    )


def choose_action(
    client: OpenAI,
    model: str,
    task_id: str,
    observation: ReliabilityObservation,
    seed: int,
) -> PlannerDecision:
    completion = client.beta.chat.completions.parse(
        model=model,
        seed=seed,
        temperature=0,
        response_format=PlannerDecision,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an autoscaling agent for a live service. "
                    "Return a single safe action."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(task_id, observation),
            },
        ],
    )
    return completion.choices[0].message.parsed


def run_episode(
    client: OpenAI,
    model: str,
    task_id: str,
    episode_seed: int,
    planner_seed: int,
) -> dict:
    env = LiveSystemReliabilityEnvironment(default_task_id=task_id)
    observation = env.reset(seed=episode_seed, task_id=task_id)

    step_index = 0
    while not observation.done:
        decision = choose_action(
            client=client,
            model=model,
            task_id=task_id,
            observation=observation,
            seed=planner_seed + step_index,
        )
        observation = env.step(
            ReliabilityAction(
                command=decision.command,
                rationale=decision.rationale,
            )
        )
        step_index += 1

    state = env.state
    return {
        "task_id": task_id,
        "episode_seed": episode_seed,
        "planner_seed": planner_seed,
        "score": state.final_grader_score,
        "episode_return": state.episode_return,
        "steps": state.step_count,
        "terminated_reason": state.terminated_reason,
    }


def summarize_scores(rows: list[dict]) -> dict[str, float]:
    scores = [row["score"] for row in rows if row["score"] is not None]
    if not scores:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Run an OpenAI baseline across OpenEnv tasks.")
    parser.add_argument(
        "--tasks",
        default=",".join(task.id for task in list_task_definitions()),
        help="Comma-separated task ids",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per task")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/openai_baseline.json"),
        help="Where to save the JSON report",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required to run the OpenAI baseline.")

    selected_tasks = [part.strip() for part in args.tasks.split(",") if part.strip()]
    for task_id in selected_tasks:
        get_task_definition(task_id)

    client = OpenAI(api_key=api_key)
    rows: list[dict] = []

    for task_offset, task_id in enumerate(selected_tasks):
        task_rows = []
        for episode_index in range(args.episodes):
            episode_seed = args.seed + task_offset * 100 + episode_index
            planner_seed = args.seed + task_offset * 1000 + episode_index * 100
            row = run_episode(
                client=client,
                model=args.model,
                task_id=task_id,
                episode_seed=episode_seed,
                planner_seed=planner_seed,
            )
            rows.append(row)
            task_rows.append(row)
            print(
                " | ".join(
                    [
                        f"Task {task_id}",
                        f"EpisodeSeed {episode_seed}",
                        f"Score {row['score']:.3f}",
                        f"Return {row['episode_return']:.3f}",
                        f"Steps {row['steps']}",
                        f"Reason {row['terminated_reason']}",
                    ]
                )
            )

        task_summary = summarize_scores(task_rows)
        print(
            f"TaskSummary {task_id} mean={task_summary['mean']:.3f} "
            f"min={task_summary['min']:.3f} max={task_summary['max']:.3f}"
        )

    overall = summarize_scores(rows)
    report = {
        "model": args.model,
        "episodes_per_task": args.episodes,
        "base_seed": args.seed,
        "tasks": selected_tasks,
        "overall": overall,
        "rows": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Overall mean={overall['mean']:.3f} min={overall['min']:.3f} "
        f"max={overall['max']:.3f}"
    )
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
