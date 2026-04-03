import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from openai import APIStatusError
from pydantic import BaseModel, Field, ValidationError

from openenv_models import ActionCommand, ReliabilityAction, ReliabilityObservation
from openenv_tasks import get_task_definition, list_task_definitions
from server.environment import LiveSystemReliabilityEnvironment


DEFAULT_MODEL = os.environ.get(
    "OPENROUTER_MODEL",
    "nvidia/nemotron-3-super-120b-a12b:free",
)
DEFAULT_FALLBACK_MODELS = tuple(
    part.strip()
    for part in os.environ.get(
        "OPENROUTER_FALLBACK_MODELS",
        "",
    ).split(",")
    if part.strip()
)
DEFAULT_TIMEOUT = float(os.environ.get("OPENROUTER_TIMEOUT", "60"))
DEFAULT_API_URL = os.environ.get(
    "OPENROUTER_API_URL",
    "https://openrouter.ai/api/v1",
)


class PlannerDecision(BaseModel):
    command: ActionCommand = Field(description="One of: scale_down, no_op, scale_up")
    rationale: str = Field(default="", max_length=280)


PLANNER_SCHEMA = {
    "name": "planner_decision",
    "schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["scale_down", "no_op", "scale_up"],
            },
            "rationale": {
                "type": "string",
            },
        },
        "required": ["command", "rationale"],
        "additionalProperties": False,
    },
    "strict": True,
}


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
        "Choose exactly one next scaling action. Prioritize survival, keep CPU in an efficient range, "
        "avoid unnecessary scaling, and return valid JSON only."
    )


def parse_decision(message_content: str) -> PlannerDecision:
    try:
        return PlannerDecision.model_validate_json(message_content)
    except ValidationError:
        if "scale_up" in message_content:
            return PlannerDecision(command="scale_up", rationale="fallback-from-text")
        if "scale_down" in message_content:
            return PlannerDecision(command="scale_down", rationale="fallback-from-text")
        return PlannerDecision(command="no_op", rationale="fallback-from-text")


def should_retry_openrouter_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429, 500, 502, 503, 504}
    return False


def get_openrouter_status(exc: Exception) -> int | None:
    if isinstance(exc, APIStatusError):
        return exc.status_code
    return None


def choose_action(
    client: OpenAI,
    models: list[str],
    task_id: str,
    observation: ReliabilityObservation,
    verbose: bool = False,
) -> PlannerDecision:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an autoscaling agent for a live production service. "
                "Always return a JSON object with keys command and rationale."
            ),
        },
        {
            "role": "user",
            "content": build_prompt(task_id, observation),
        },
    ]

    last_error: Exception | None = None
    for model in models:
        for attempt in range(4):
            try:
                if verbose:
                    print(
                        f"[OpenRouter] task={task_id} step={observation.step_count} "
                        f"model={model} attempt={attempt + 1}",
                        flush=True,
                    )
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=80,
                    extra_body={"reasoning": {"enabled": True}},
                )
                message = completion.choices[0].message
                content = message.content or ""
                if verbose:
                    print(
                        f"[OpenRouter] task={task_id} step={observation.step_count} "
                        f"model={model} completed",
                        flush=True,
                    )
                return parse_decision(content)
            except Exception as exc:
                last_error = exc
                status = get_openrouter_status(exc)
                if verbose:
                    print(
                        f"[OpenRouter] task={task_id} step={observation.step_count} "
                        f"model={model} attempt={attempt + 1} failed: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                if not should_retry_openrouter_error(exc):
                    break
                # Switch models quickly on hard rate limits instead of spending all retries
                # on a provider that is actively throttling this request.
                if status == 429:
                    time.sleep(1)
                    break
                if attempt < 3:
                    time.sleep(2**attempt)

    if last_error is not None:
        raise RuntimeError(
            "OpenRouter inference failed after retries and model fallbacks. "
            "Try again later or switch to a different provider/model."
        ) from last_error
    raise RuntimeError("OpenRouter inference failed before any request completed.")


def run_episode(
    client: OpenAI,
    models: list[str],
    task_id: str,
    episode_seed: int,
    verbose: bool = False,
) -> dict:
    env = LiveSystemReliabilityEnvironment(default_task_id=task_id)
    observation = env.reset(seed=episode_seed, task_id=task_id)
    if verbose:
        print(
            f"[Episode] task={task_id} seed={episode_seed} started "
            f"latency={observation.latency:.3f} cpu={observation.cpu:.3f} "
            f"error={observation.error_rate:.3f} traffic={observation.traffic:.3f}",
            flush=True,
        )

    while not observation.done:
        decision = choose_action(
            client=client,
            models=models,
            task_id=task_id,
            observation=observation,
            verbose=verbose,
        )
        observation = env.step(
            ReliabilityAction(
                command=decision.command,
                rationale=decision.rationale,
            )
        )
        if verbose:
            print(
                f"[Step] task={task_id} seed={episode_seed} step={observation.step_count} "
                f"action={decision.command} score={observation.current_grader_score:.3f} "
                f"latency={observation.latency:.3f} cpu={observation.cpu:.3f} "
                f"error={observation.error_rate:.3f} traffic={observation.traffic:.3f} "
                f"done={observation.done}",
                flush=True,
            )

    state = env.state
    return {
        "task_id": task_id,
        "episode_seed": episode_seed,
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


def build_report(
    *,
    model: str,
    fallback_models: list[str],
    episodes_per_task: int,
    base_seed: int,
    tasks: list[str],
    rows: list[dict],
    status: str = "completed",
    error: str | None = None,
) -> dict:
    task_summaries: dict[str, dict[str, float]] = {}
    for task_id in tasks:
        task_rows = [row for row in rows if row["task_id"] == task_id]
        if task_rows:
            task_summaries[task_id] = summarize_scores(task_rows)

    report = {
        "provider": "openrouter",
        "model": model,
        "fallback_models": fallback_models,
        "episodes_per_task": episodes_per_task,
        "base_seed": base_seed,
        "tasks": tasks,
        "task_summaries": task_summaries,
        "overall": summarize_scores(rows),
        "rows": rows,
        "status": status,
    }
    if error:
        report["error"] = error
    return report


def write_report(output_path: Path, report: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Run an OpenAI-client baseline across OpenEnv tasks."
    )
    parser.add_argument(
        "--tasks",
        default=",".join(task.id for task in list_task_definitions()),
        help="Comma-separated task ids",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per task")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model name")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="API request timeout in seconds")
    parser.add_argument(
        "--fallback-models",
        default=",".join(DEFAULT_FALLBACK_MODELS),
        help="Comma-separated fallback models to try after the primary model",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-request and per-step progress logs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/openrouter_baseline.json"),
        help="Where to save the JSON report",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is required to run the OpenAI-client baseline.")

    selected_tasks = [part.strip() for part in args.tasks.split(",") if part.strip()]
    for task_id in selected_tasks:
        get_task_definition(task_id)

    client = OpenAI(
        api_key=api_key,
        base_url=DEFAULT_API_URL,
        timeout=args.timeout,
    )
    model_candidates = [args.model] + [
        part.strip()
        for part in args.fallback_models.split(",")
        if part.strip() and part.strip() != args.model
    ]
    rows: list[dict] = []

    for task_offset, task_id in enumerate(selected_tasks):
        task_rows = []
        for episode_index in range(args.episodes):
            episode_seed = args.seed + task_offset * 100 + episode_index
            try:
                row = run_episode(
                    client=client,
                    models=model_candidates,
                    task_id=task_id,
                    episode_seed=episode_seed,
                    verbose=args.verbose,
                )
            except Exception as exc:
                partial_report = build_report(
                    model=args.model,
                    fallback_models=model_candidates[1:],
                    episodes_per_task=args.episodes,
                    base_seed=args.seed,
                    tasks=selected_tasks,
                    rows=rows,
                    status="partial",
                    error=str(exc),
                )
                write_report(args.output, partial_report)
                print(f"Saved partial report to {args.output}")
                raise
            rows.append(row)
            task_rows.append(row)
            checkpoint_report = build_report(
                model=args.model,
                fallback_models=model_candidates[1:],
                episodes_per_task=args.episodes,
                base_seed=args.seed,
                tasks=selected_tasks,
                rows=rows,
                status="running",
            )
            write_report(args.output, checkpoint_report)
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

    report = build_report(
        model=args.model,
        fallback_models=model_candidates[1:],
        episodes_per_task=args.episodes,
        base_seed=args.seed,
        tasks=selected_tasks,
        rows=rows,
        status="completed",
    )
    write_report(args.output, report)
    overall = report["overall"]
    print(
        f"Overall mean={overall['mean']:.3f} min={overall['min']:.3f} "
        f"max={overall['max']:.3f}"
    )
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
