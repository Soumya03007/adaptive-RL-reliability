---
title: Adaptive RL Reliability
emoji: chart_with_upwards_trend
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Adaptive RL Reliability

Adaptive RL Reliability is an OpenEnv-compatible environment for live-system autoscaling. Instead of a game, it simulates a real operator task: choosing whether to scale a production service down, hold steady, or scale up while traffic fluctuates and failures appear.

The environment is designed around real-world reliability tradeoffs:
- protect latency and error-rate SLOs
- avoid CPU saturation
- avoid wasteful or oscillating scaling decisions
- survive longer under tougher dynamics

## Why this environment exists

Many RL examples are games or toy control loops. This repo focuses on a task humans actually perform in production systems: reliability-aware capacity control. The agent is effectively acting like a cautious on-call engineer or autoscaling policy trying to keep a live service healthy.

## OpenEnv compliance

This repo now includes the required OpenEnv surface:
- typed Pydantic models in `openenv_models.py`
- OpenEnv server in `server/app.py`
- OpenEnv environment implementation in `server/live_system_environment.py`
- `state` support via `ReliabilityState`
- manifest in `openenv.yaml`

Local validator status:

```powershell
openenv validate .
```

This passes locally.

## Action space

The agent chooses one discrete control action per step:

- `scale_down`
- `no_op`
- `scale_up`

These actions are represented by the typed `ReliabilityAction` model.

## Observation space

Each `ReliabilityObservation` contains:

- `task_id`, `task_title`, `difficulty`, `objective`
- `latency`
- `cpu`
- `error_rate`
- `traffic`
- `step_count`
- `remaining_steps`
- `healthy_steps`
- `healthy_ratio`
- `current_grader_score`
- `last_action`
- `reward_breakdown`
- `done`, `reward`, `metadata`

The observation is intentionally operator-friendly: it exposes the service health signals an autoscaling agent would reasonably use.

## Task set

The environment ships with three deterministic grader-backed tasks.

| Task | Difficulty | Objective | Pass target |
| --- | --- | --- | --- |
| `balanced` | easy | Hold a routine production service inside standard SLOs under normal load. | full horizon, healthy ratio >= 0.80, score >= 0.78 |
| `high_traffic` | medium | Absorb a traffic surge without letting latency and errors spiral. | full horizon, healthy ratio >= 0.72, score >= 0.72 |
| `failure_heavy` | hard | Ride through repeated failure spikes while recovering toward healthy bounds. | full horizon, healthy ratio >= 0.60, score >= 0.66 |

Task definitions and grader logic live in `openenv_tasks.py`.

## Graders

Each task uses a programmatic grader that returns a normalized score from `0.0` to `1.0`.

The grader is deterministic and combines:
- survival ratio across the task horizon
- healthy-step ratio against task thresholds
- SLO closeness for latency, CPU, and error rate
- action discipline, which discourages excessive action flipping

The grader returns both:
- a scalar `score`
- a boolean `passed`

This is separate from the shaped training reward so evaluation remains explicit and auditable.

## Reward design

The per-step reward is shaped over the full trajectory and is not just a terminal binary.

It combines:
- simulator base reward from latency, CPU, error rate, and action costs
- a health reward for being inside task thresholds
- a progress bonus for staying healthy later into the episode
- an outage penalty for terminating early

The reward breakdown is exposed via the typed `ReliabilityReward` model.

## Project layout

Relevant files:

- `openenv.yaml`: OpenEnv manifest
- `openenv_models.py`: typed Action, Observation, Reward, and State models
- `openenv_tasks.py`: task definitions and deterministic graders
- `server/live_system_environment.py`: OpenEnv environment wrapper
- `server/app.py`: FastAPI app for OpenEnv
- `scripts/run_hf_baseline.py`: Hugging Face Inference baseline runner
- `scripts/run_rule_baseline.py`: deterministic local smoke baseline
- `envs/reliability_env.py`: underlying service simulator
- `scripts/train_torchrl_ppo.py`: existing PPO training pipeline

## Setup

Install the package:

```powershell
pip install -e .
```

Generate the lockfile if needed:

```powershell
python -m uv lock
```

Validate the environment:

```powershell
openenv validate .
```

Run the server locally:

```powershell
python -m server.app
```

Or via the project script:

```powershell
python -m uv run server
```

## Docker

Build the container:

```powershell
docker build -t adaptive-rl-reliability .
```

Run it:

```powershell
docker run --rm -p 8000:8000 adaptive-rl-reliability
```

Note: the Dockerfile is present and validator-compatible. In this local session the Docker daemon was not running, so I could not complete a live `docker build` execution here.

## Hugging Face Spaces

This repo is structured for a Docker-based Hugging Face Space and tagged with `openenv` in the README metadata block.

Typical deployment flow:

```powershell
openenv push
```

Or manually push this repo to a Docker Space on Hugging Face.

## Baselines

### Deterministic local rule baseline

Checked locally with:

```powershell
python scripts/run_rule_baseline.py --episodes 5
```

Observed scores:

| Task | Mean score | Min | Max |
| --- | --- | --- | --- |
| `balanced` | `0.873` | `0.845` | `0.901` |
| `high_traffic` | `0.819` | `0.788` | `0.868` |
| `failure_heavy` | `0.763` | `0.740` | `0.798` |
| overall | `0.818` | `0.740` | `0.901` |

The JSON artifact is written to `outputs/rule_baseline.json`.

### Hugging Face baseline

Use the OpenAI-client baseline against an OpenRouter-hosted instruct model with:

```powershell
python scripts/run_hf_baseline.py --episodes 1
```

Requirements:
- `OPENROUTER_API_KEY` must be set
- optionally set `OPENROUTER_MODEL`
- optionally set `OPENROUTER_FALLBACK_MODELS`

Default model:

```text
nvidia/nemotron-3-super-120b-a12b:free
```

The baseline writes results to `outputs/openrouter_baseline.json`.

The runner uses the `openai` Python client with OpenRouter's OpenAI-compatible `base_url`, retries temporary provider errors, checkpoints after each finished episode, and can fall back to backup models.

## Existing PPO training path

The original TorchRL PPO training code is still available. That makes this repo useful in two ways:
- as an OpenEnv benchmark for agent-style control
- as an RL research sandbox for learned policies on the same simulator

Train PPO:

```powershell
python scripts/train_torchrl_ppo.py --task balanced
```

Evaluate a checkpoint:

```powershell
python scripts/eval_checkpoint.py training\runs\<run_name>\checkpoints\best.pt --episodes 20
```
