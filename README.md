# Adaptive RL Reliability

A reinforcement learning environment for reliability-aware service control.

The agent observes a simulated service state and chooses one of three scaling actions:

- `0`: scale down
- `1`: no-op
- `2`: scale up

The goal is to keep the system stable under stochastic traffic and failures while balancing latency, CPU usage, and error rate.

## Environment

Observation space:

- `latency`
- `cpu`
- `error_rate`
- `traffic`

Action space:

- discrete with 3 actions: `scale down`, `no-op`, `scale up`

Episode termination:

- `latency >= 2.0`
- `cpu >= 2.0`
- `error_rate >= 1.0`
- or max episode length via `StepCounter`

## Reward And Grader Score

Per-step reward is normalized to `0.0-1.0`.

- `1.0` means the system is healthy and efficient
- `0.0` means the system is in a very poor state

The reward is computed from a weighted penalty over:

- latency
- error rate
- cpu usage

Then normalized as:

```text
reward = 1 - penalty / max_penalty
```

For reporting, training also computes a normalized episode-level `GraderScore`:

```text
GraderScore = mean per-step reward over the episode
```

This is the main metric to use in demos and hackathon reporting because it stays in `0.0-1.0`.

## Reproducibility

Training and evaluation are seeded through `configs/config.py`.

Current default:

- `SEED = 42`

The environment uses its own seeded RNG, and training also seeds Python, NumPy, and PyTorch.

## Main Files

- `envs/reliability_env.py`: reliability simulator and reward
- `envs/torchrl_env.py`: TorchRL wrapper
- `models/policy.py`: policy network
- `models/value.py`: value network
- `scripts/train_torchrl_ppo.py`: main PPO training script
- `scripts/eval_checkpoint.py`: evaluate a saved checkpoint
- `configs/env_config.yaml`: environment dynamics and reward weights
- `configs/config.py`: training and evaluation settings

## How To Run

Install dependencies:

```powershell
pip install -e .
```

Smoke test the environment:

```powershell
python scripts/test_env.py
```

Run TorchRL PPO training:

```powershell
python scripts/train_torchrl_ppo.py
```

This creates a run directory like:

```text
training/runs/20260331_110231
```

Artifacts written per run:

- `run_config.json`
- `metrics.csv`
- `checkpoints/latest.pt`
- `checkpoints/best.pt`
- periodic checkpoints like `checkpoints/iter_00010.pt`

Evaluate the best checkpoint from a run:

```powershell
python scripts/eval_checkpoint.py training\runs\<run_name>\checkpoints\best.pt --episodes 20
```

Example:

```powershell
python scripts/eval_checkpoint.py training\runs\20260331_110231\checkpoints\best.pt --episodes 20
```

## What To Watch During Training

The most important console fields are:

- `TrainReward`: mean normalized step reward on the latest rollout
- `TrainAvg`: smoothed training reward
- `GraderScore`: normalized `0.0-1.0` evaluation score
- `EvalState`: average latency, CPU, and error during evaluation
- `EvalLen`: average episode length during evaluation

Good signs:

- `GraderScore` trends upward
- `EvalLen` increases
- `EvalState` latency and error decrease

## Suggested Hackathon Reporting

Use `GraderScore` as the headline metric.

Recommended wording:

- "Per-step reward is normalized to `0-1`."
- "Final evaluation uses `GraderScore`, the mean per-step reward across evaluation episodes."
- "Training and evaluation are seeded for reproducibility."
