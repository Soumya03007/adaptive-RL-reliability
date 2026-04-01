---
title: Reliability Game Environment Server
emoji: game_die
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - reliability
---

# Reliability Game

This package turns the reliability-control simulator into an OpenEnv mini-game with:

- a playable action loop
- four tasks with increasing difficulty
- explicit graders that verify task completion
- rubric-based reward shaping for automated evaluation

## Tasks

1. `balanced` / Stability Bootcamp
2. `high_traffic` / Traffic Surge
3. `failure_heavy` / Chaos Monkey
4. `cost_sensitive` / Efficiency Gauntlet

Each task defines a success objective and pass thresholds for:

- minimum episode length
- average latency
- average CPU
- average error rate
- mean reward

An episode is marked complete only when every check passes.

## Quick Start

```python
from reliability_game import ReliabilityGameAction, ReliabilityGameEnv

with ReliabilityGameEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_name="balanced", seed=42)
    while not result.done:
        result = env.step(ReliabilityGameAction(choice="hold"))
        print(result.observation.grader_score, result.observation.task_completed)
```

## Local Server

```powershell
python -m reliability_game.server.app --port 8000
```

## Validate

```powershell
openenv validate .
```
