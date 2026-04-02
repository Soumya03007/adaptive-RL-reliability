import json
import os
import time
from pathlib import Path
from typing import Optional

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from openai import APIStatusError
from pydantic import BaseModel, Field, ValidationError

from openenv_models import ActionCommand, ReliabilityAction, ReliabilityObservation
from openenv_tasks import get_task_definition, list_task_definitions
from server.live_system_environment import LiveSystemReliabilityEnvironment


API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3.6-plus-preview:free")
API_KEY = os.getenv("HF_TOKEN")
BENCHMARK = os.getenv("BENCHMARK_NAME", "adaptive-rl-reliability")
TASKS = os.getenv(
    "TASKS",
    ",".join(task.id for task in list_task_definitions()),
)
BASE_SEED = int(os.getenv("BASE_SEED", "42"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))
OUTPUT_PATH = Path(os.getenv("INFERENCE_OUTPUT", "outputs/inference_report.json"))


class PlannerDecision(BaseModel):
    command: ActionCommand = Field(description="One of: scale_down, no_op, scale_up")
    rationale: str = Field(default="", max_length=280)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
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


def fallback_decision(observation: ReliabilityObservation) -> PlannerDecision:
    if (
        observation.traffic < 0.28
        and observation.latency < 0.45
        and observation.cpu < 0.45
        and observation.error_rate < 0.03
    ):
        return PlannerDecision(command="no_op", rationale="fallback-low-load")
    return PlannerDecision(command="scale_up", rationale="fallback-protect-capacity")


def apply_guardrail(
    decision: PlannerDecision,
    observation: ReliabilityObservation,
) -> PlannerDecision:
    if (
        observation.error_rate > 0.08
        or observation.latency > 0.85
        or observation.cpu > 0.85
    ):
        if decision.command != "scale_up":
            return PlannerDecision(command="scale_up", rationale="guardrail-scale-up")
    return decision


def should_retry(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429, 500, 502, 503, 504}
    return False


def choose_action(
    client: OpenAI,
    model_name: str,
    task_id: str,
    observation: ReliabilityObservation,
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
    for attempt in range(4):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_tokens=80,
                extra_body={"reasoning": {"enabled": True}},
            )
            content = completion.choices[0].message.content or ""
            return apply_guardrail(parse_decision(content), observation)
        except Exception as exc:
            last_error = exc
            if not should_retry(exc):
                break
            if attempt < 3:
                time.sleep(2**attempt)

    if last_error is not None:
        return fallback_decision(observation)
    return fallback_decision(observation)


def run_episode(client: OpenAI, task_id: str, episode_seed: int) -> dict:
    task = get_task_definition(task_id)
    env = LiveSystemReliabilityEnvironment(default_task_id=task_id)
    rewards: list[float] = []
    steps_taken = 0
    success = False
    final_score = 0.0
    last_error: str | None = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset(seed=episode_seed, task_id=task_id)
        while not observation.done:
            decision = choose_action(
                client=client,
                model_name=MODEL_NAME,
                task_id=task_id,
                observation=observation,
            )
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
        "score": final_score,
        "steps": steps_taken,
        "success": success,
        "error": last_error,
        "rewards": rewards,
    }


def main() -> None:
    if not API_KEY:
        raise EnvironmentError("HF_TOKEN is required.")

    selected_tasks = [part.strip() for part in TASKS.split(",") if part.strip()]
    for task_id in selected_tasks:
        get_task_definition(task_id)

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        timeout=REQUEST_TIMEOUT,
    )

    rows: list[dict] = []
    for task_offset, task_id in enumerate(selected_tasks):
        episode_seed = BASE_SEED + task_offset * 100
        rows.append(run_episode(client=client, task_id=task_id, episode_seed=episode_seed))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "api_base_url": API_BASE_URL,
                "model": MODEL_NAME,
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
