# Requirements Document

## Introduction

This feature makes the existing Adaptive RL Reliability codebase fully compliant with the OpenEnv specification and deployable to Hugging Face Spaces. The codebase already contains a PPO-based reinforcement learning environment (`ReliabilityEnv`) that simulates a microservice under stochastic traffic and failure conditions. The gaps to close are: typed Pydantic models wired into the OpenEnv interface, an `openenv.yaml` metadata file, a `state()` method, a baseline inference script using the OpenAI API, a Dockerfile, a Hugging Face Spaces deployment (`app.py` + Space config), standalone task graders, explicit difficulty labeling, and an updated README.

## Glossary

- **OpenEnv**: The open environment specification defined by the `openenv-core` library that standardises the `step()`, `reset()`, and `state()` interface and the `openenv validate` CLI tool.
- **ReliabilityEnv**: The existing Python class in `envs/reliability_env.py` that simulates a microservice with latency, CPU, error rate, and traffic state.
- **OpenEnvServer**: The OpenEnv server class (from `openenv-core`) that wraps an environment and exposes the compliant HTTP/JSON interface.
- **Observation**: A Pydantic model (`ReliabilityObservation`) that carries the full typed observation returned to an agent after each `reset()` or `step()`.
- **Action**: A Pydantic model (`ReliabilityAction`) that carries the agent's typed scaling decision (`scale_down`, `no_op`, `scale_up`) and an optional rationale string.
- **Reward**: A Pydantic model (`ReliabilityReward`) that carries the structured per-step reward decomposition.
- **State**: A Pydantic model (`ReliabilityState`) that carries the full internal environment state exposed through `state()`.
- **TaskDefinition**: A Pydantic model in `openenv_tasks.py` that describes a single graded task variant including difficulty, targets, and score weights.
- **GraderResult**: A Pydantic model returned by `grade_trajectory()` containing the final score and sub-scores for a completed episode.
- **Baseline_Script**: The script `scripts/baseline_openai.py` that runs an LLM agent against all tasks using the OpenAI API client.
- **HF_Space**: The Hugging Face Spaces deployment consisting of `app.py` and the Space YAML header in `README.md`.
- **openenv.yaml**: The metadata file at the repository root that describes the environment to the `openenv validate` CLI.
- **Dockerfile**: The container build file that packages the environment for reproducible execution.
- **GraderScore**: The scalar `0.0–1.0` episode-level score produced by `grade_trajectory()`.

---

## Requirements

### Requirement 1: Typed Pydantic Models for OpenEnv Interface

**User Story:** As a hackathon evaluator, I want the environment to expose typed Pydantic models for Observation, Action, Reward, and State, so that the `openenv validate` tool can verify interface compliance automatically.

#### Acceptance Criteria

1. THE `ReliabilityObservation` SHALL extend the OpenEnv `Observation` base class and include fields for `latency`, `cpu`, `error_rate`, `traffic`, `step_count`, `remaining_steps`, `healthy_steps`, `healthy_ratio`, `current_grader_score`, `last_action`, `available_actions`, `task_id`, `task_title`, `difficulty`, `objective`, and `reward_breakdown`.
2. THE `ReliabilityAction` SHALL extend the OpenEnv `Action` base class and include a `command` field constrained to the literal values `"scale_down"`, `"no_op"`, and `"scale_up"`, plus an optional `rationale` string field with a maximum length of 280 characters.
3. THE `ReliabilityReward` SHALL be a Pydantic `BaseModel` with float fields `base_reward`, `health_reward`, `progress_bonus`, `outage_penalty`, and `final_reward`, each constrained to the range `[0.0, 1.0]`.
4. THE `ReliabilityState` SHALL extend the OpenEnv `State` base class and include all fields needed to reconstruct the full internal environment state, including `task_id`, `task_title`, `difficulty`, `objective`, `max_steps`, `latency`, `cpu`, `error_rate`, `traffic`, `healthy_steps`, `action_changes`, `episode_return`, `current_grader_score`, `final_grader_score`, `terminated_reason`, `action_history`, and `reward_breakdown`.
5. WHEN a Pydantic model field receives a value outside its declared range, THE model SHALL raise a `ValidationError`.

---

### Requirement 2: OpenEnv-Compliant Environment Wrapper

**User Story:** As a hackathon evaluator, I want the environment to implement the full OpenEnv interface (`reset()`, `step()`, `state()`), so that any OpenEnv-compatible agent can interact with it without custom glue code.

#### Acceptance Criteria

1. THE `OpenEnvReliabilityEnv` SHALL implement a `reset(task_id: str | None)` method that initialises a new episode and returns a `ReliabilityObservation` instance.
2. WHEN `reset()` is called with a valid `task_id`, THE `OpenEnvReliabilityEnv` SHALL load the corresponding `TaskDefinition` and configure `ReliabilityEnv` with the matching `env_config`.
3. WHEN `reset()` is called with `task_id=None`, THE `OpenEnvReliabilityEnv` SHALL default to the `"balanced"` task.
4. THE `OpenEnvReliabilityEnv` SHALL implement a `step(action: ReliabilityAction)` method that accepts a `ReliabilityAction`, maps `command` to the integer action index, calls `ReliabilityEnv.step()`, and returns a tuple of `(ReliabilityObservation, ReliabilityReward, bool, dict)`.
5. THE `OpenEnvReliabilityEnv` SHALL implement a `state()` method that returns a `ReliabilityState` reflecting the current internal environment state at any point during an episode.
6. WHEN `step()` is called before `reset()`, THE `OpenEnvReliabilityEnv` SHALL raise a `RuntimeError` with a descriptive message.
7. WHEN the episode terminates (latency ≥ 2.0, cpu ≥ 2.0, error_rate ≥ 1.0, or max steps reached), THE `OpenEnvReliabilityEnv` SHALL set the `done` flag to `True` and populate `terminated_reason` in the returned state.

---

### Requirement 3: openenv.yaml Metadata File

**User Story:** As a hackathon evaluator, I want an `openenv.yaml` file at the repository root, so that the `openenv validate` CLI can discover and validate the environment without manual configuration.

#### Acceptance Criteria

1. THE `openenv.yaml` file SHALL exist at the repository root and contain the fields required by the OpenEnv specification: `name`, `version`, `description`, `entry_point`, `observation_type`, `action_type`, `reward_type`, `state_type`, and `tasks`.
2. THE `tasks` field in `openenv.yaml` SHALL list all three task variants (`balanced`, `high_traffic`, `failure_heavy`) with their `id`, `title`, `difficulty`, and `description`.
3. WHEN `openenv validate` is run against the repository, THE OpenEnv_CLI SHALL report zero validation errors.

---

### Requirement 4: Standalone Task Graders

**User Story:** As a hackathon evaluator, I want each task to have a clean standalone grader callable, so that scores can be computed independently of the training loop.

#### Acceptance Criteria

1. THE `openenv_tasks.py` module SHALL define a `TASKS` dictionary mapping task IDs to `TaskDefinition` instances for at least three tasks: `"balanced"` (easy), `"high_traffic"` (medium), and `"failure_heavy"` (hard).
2. THE `grade_trajectory(task, trajectory, action_changes, max_steps)` function SHALL accept a `TaskDefinition`, a list of per-step metric dicts, an integer action change count, and an optional max steps override, and SHALL return a `GraderResult`.
3. THE `GraderResult` SHALL include `score` (float, `[0.0, 1.0]`), `passed` (bool), `survival_ratio`, `healthy_ratio`, `slo_score`, `action_discipline`, `terminated_early`, and `summary` fields.
4. WHEN `grade_trajectory()` is called with an empty trajectory list, THE grader SHALL return a `GraderResult` with `score=0.0` and `passed=False`.
5. THE `score` field in `GraderResult` SHALL be computed as a weighted sum of `survival_ratio`, `healthy_ratio`, `slo_score`, and `action_discipline` using the weights defined in the corresponding `TaskDefinition`.
6. THE difficulty labels SHALL follow the ordering: `"balanced"` → `"easy"`, `"high_traffic"` → `"medium"`, `"failure_heavy"` → `"hard"`.

---

### Requirement 5: Baseline Inference Script Using OpenAI API

**User Story:** As a hackathon evaluator, I want a baseline inference script that runs an LLM agent against all tasks using the OpenAI API, so that I can reproduce baseline scores without a trained RL checkpoint.

#### Acceptance Criteria

1. THE `Baseline_Script` SHALL read the `OPENAI_API_KEY` environment variable and raise a descriptive error if it is not set.
2. THE `Baseline_Script` SHALL run the LLM agent against all tasks defined in `TASKS` using the `openai` Python client.
3. WHEN the LLM agent is invoked, THE `Baseline_Script` SHALL construct a prompt that includes the current `ReliabilityObservation` fields and the list of available actions.
4. THE `Baseline_Script` SHALL parse the LLM response to extract a valid `ActionCommand` and fall back to `"no_op"` if the response cannot be parsed.
5. WHEN an episode completes, THE `Baseline_Script` SHALL call `grade_trajectory()` and record the `GraderResult` for that task.
6. THE `Baseline_Script` SHALL run each task with a fixed seed for reproducibility and print a summary table of task ID, difficulty, and `GraderScore` to stdout.
7. THE `Baseline_Script` SHALL accept a `--model` CLI argument defaulting to `"gpt-4o-mini"` and a `--episodes` argument defaulting to `3`.

---

### Requirement 6: Dockerfile for Containerised Execution

**User Story:** As a hackathon evaluator, I want a Dockerfile that packages the environment, so that the environment can be run reproducibly in any container runtime.

#### Acceptance Criteria

1. THE `Dockerfile` SHALL use a Python base image compatible with the project's `requires-python = ">=3.12"` constraint.
2. THE `Dockerfile` SHALL install all runtime dependencies from `requirements.txt` or `pyproject.toml`.
3. THE `Dockerfile` SHALL set the default `CMD` to run the baseline inference script (`scripts/baseline_openai.py`).
4. WHEN the Docker image is built, THE build SHALL complete without errors on a standard Linux x86-64 host.
5. THE `Dockerfile` SHALL accept `OPENAI_API_KEY` as a runtime environment variable (not baked into the image).

---

### Requirement 7: Hugging Face Spaces Deployment

**User Story:** As a hackathon evaluator, I want the environment deployed to Hugging Face Spaces with a Gradio interface, so that I can interact with the environment and view live scores in a browser.

#### Acceptance Criteria

1. THE `app.py` file SHALL exist at the repository root and implement a Gradio interface that allows a user to select a task, step through the environment manually or run an auto-play loop, and view the current observation and grader score.
2. THE `app.py` SHALL import and use `OpenEnvReliabilityEnv` to drive the environment.
3. THE `README.md` SHALL include a Hugging Face Spaces YAML front-matter block at the top with at least the fields `title`, `emoji`, `colorFrom`, `colorTo`, `sdk`, `sdk_version`, `app_file`, and `tags` (including the `openenv` tag).
4. WHEN the Gradio app is launched, THE app SHALL display the current `latency`, `cpu`, `error_rate`, `traffic`, and `current_grader_score` values.
5. THE Space config `tags` field SHALL include `"openenv"` so the Space is discoverable by the hackathon leaderboard.

---

### Requirement 8: Updated README with Task Descriptions and Baseline Scores

**User Story:** As a developer or evaluator browsing the repository, I want the README to document all tasks with their difficulty levels and baseline scores, so that I can understand the benchmark at a glance.

#### Acceptance Criteria

1. THE `README.md` SHALL include a task table listing each task's `id`, `title`, `difficulty`, `objective`, and `pass_score`.
2. THE `README.md` SHALL document the baseline LLM agent scores produced by `scripts/baseline_openai.py` for each task.
3. THE `README.md` SHALL include instructions for running `openenv validate`, the baseline script, and the Gradio app.
4. THE `README.md` SHALL retain all existing documentation about the observation space, action space, reward formula, and PPO training.
