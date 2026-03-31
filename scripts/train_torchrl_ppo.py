import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.distributions import Categorical
from torch.optim import Adam

from tensordict.nn import TensorDictModule
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import Compose, StepCounter
from torchrl.modules import ProbabilisticActor, ValueOperator

from configs.config import *
from configs.loader import get_default_task_name, load_env_config
from envs.reliability_env import ReliabilityEnv
from envs.torchrl_env import TorchRLEnvWrapper
from models.policy import PolicyNet
from models.value import ValueNet
from rl.buffer import build_buffer
from rl.collector import build_collector
from rl.ppo import build_ppo
from training.artifacts import (
    append_metrics_row,
    prepare_run_dir,
    save_checkpoint,
    write_run_metadata,
)
from utils.seeding import set_global_seed


def evaluate_policy(policy_net, env_config, device, episodes=5, max_steps=50, seed=None):
    eval_env = ReliabilityEnv(env_config)
    if seed is not None:
        eval_env.seed(seed)

    returns = []
    grader_scores = []
    episode_lengths = []
    state_history = []

    policy_net.eval()
    with torch.no_grad():
        for _ in range(episodes):
            obs = eval_env.reset()
            episode_return = 0.0

            for step in range(max_steps):
                state_history.append(obs.copy())
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                action = torch.argmax(policy_net(obs_tensor), dim=-1).item()
                obs, reward, done, _ = eval_env.step(action)
                episode_return += reward

                if done:
                    episode_lengths.append(step + 1)
                    break
            else:
                episode_lengths.append(max_steps)

            returns.append(episode_return)
            grader_scores.append(episode_return / episode_lengths[-1])

    policy_net.train()

    state_means = np.mean(np.asarray(state_history, dtype=np.float32), axis=0)
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "grader_score_mean": float(np.mean(grader_scores)),
        "grader_score_std": float(np.std(grader_scores)),
        "episode_length_mean": float(np.mean(episode_lengths)),
        "latency_mean": float(state_means[0]),
        "cpu_mean": float(state_means[1]),
        "error_mean": float(state_means[2]),
        "traffic_mean": float(state_means[3]),
    }


def summarize_rollout(data):
    next_obs = data["next", "observation"].reshape(-1, 4).detach().cpu()
    action_counts = torch.bincount(
        data["action"].reshape(-1).detach().cpu(),
        minlength=3,
    ).tolist()

    return {
        "reward_mean": data["next", "reward"].mean().item(),
        "latency_mean": next_obs[:, 0].mean().item(),
        "cpu_mean": next_obs[:, 1].mean().item(),
        "error_mean": next_obs[:, 2].mean().item(),
        "traffic_mean": next_obs[:, 3].mean().item(),
        "action_counts": [int(count) for count in action_counts],
    }


def main(max_batches=None, run_name=None, task_name=None):
    device = torch.device(DEVICE)
    set_global_seed(SEED)
    selected_task = task_name or get_default_task_name()
    env_config = load_env_config(selected_task)
    train_config = {
        "seed": SEED,
        "task_name": selected_task,
        "device": DEVICE,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "ppo_epochs": PPO_EPOCHS,
        "frames_per_batch": FRAMES_PER_BATCH,
        "total_frames": TOTAL_FRAMES,
        "max_steps": MAX_STEPS,
        "eval_interval": EVAL_INTERVAL,
        "eval_episodes": EVAL_EPISODES,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "clip_epsilon": CLIP_EPSILON,
        "entropy_coeff": ENTROPY_COEFF,
        "critic_coeff": CRITIC_COEFF,
        "grad_clip_norm": GRAD_CLIP_NORM,
    }

    run_dir = prepare_run_dir(RUNS_DIR, run_name=run_name)
    write_run_metadata(run_dir, train_config, env_config)
    metrics_path = run_dir / "metrics.csv"

    base_env = TorchRLEnvWrapper(ReliabilityEnv(env_config), device=str(device))
    base_env._set_seed(SEED)
    env = TransformedEnv(base_env, Compose([StepCounter(max_steps=MAX_STEPS)]))

    policy_net = PolicyNet().to(device)
    value_net = ValueNet().to(device)

    policy_module = ProbabilisticActor(
        module=TensorDictModule(policy_net, ["observation"], ["logits"]),
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        return_log_prob=True,
        spec=env.action_spec,
    )

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    collector = build_collector(env, policy_module, device)
    advantage_module, loss_module = build_ppo(policy_module, value_module)

    optimizer = Adam(
        list(policy_net.parameters()) + list(value_net.parameters()),
        lr=LR,
    )

    running_reward = 0.0
    best_grader_score = float("-inf")

    print(f"RunDir {run_dir}")

    for i, data in enumerate(collector):
        data = data.to(device)
        advantage_module(data)

        batch = data.reshape(-1)
        buffer = build_buffer(len(batch))
        buffer.extend(batch.cpu())

        for _ in range(PPO_EPOCHS):
            subdata = buffer.sample(min(BATCH_SIZE, len(buffer))).to(device)
            loss_vals = loss_module(subdata)
            loss = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP_NORM)
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        rollout_metrics = summarize_rollout(data)
        reward = rollout_metrics["reward_mean"]
        running_reward = 0.9 * running_reward + 0.1 * reward if i > 0 else reward

        if i % EVAL_INTERVAL == 0:
            eval_metrics = evaluate_policy(
                policy_net,
                env_config,
                device,
                episodes=EVAL_EPISODES,
                max_steps=MAX_STEPS,
                seed=SEED + 1000,
            )

            metrics_row = {
                "iteration": i,
                "train_reward": reward,
                "train_reward_ema": running_reward,
                "train_latency": rollout_metrics["latency_mean"],
                "train_cpu": rollout_metrics["cpu_mean"],
                "train_error": rollout_metrics["error_mean"],
                "train_traffic": rollout_metrics["traffic_mean"],
                "action_0": rollout_metrics["action_counts"][0],
                "action_1": rollout_metrics["action_counts"][1],
                "action_2": rollout_metrics["action_counts"][2],
                "eval_return_mean": eval_metrics["return_mean"],
                "eval_return_std": eval_metrics["return_std"],
                "grader_score_mean": eval_metrics["grader_score_mean"],
                "grader_score_std": eval_metrics["grader_score_std"],
                "eval_episode_length_mean": eval_metrics["episode_length_mean"],
                "eval_latency": eval_metrics["latency_mean"],
                "eval_cpu": eval_metrics["cpu_mean"],
                "eval_error": eval_metrics["error_mean"],
                "eval_traffic": eval_metrics["traffic_mean"],
            }
            append_metrics_row(metrics_path, metrics_row)

            checkpoint_metrics = {
                "train_reward": reward,
                "train_reward_ema": running_reward,
                "grader_score_mean": eval_metrics["grader_score_mean"],
                "grader_score_std": eval_metrics["grader_score_std"],
                "eval_episode_length_mean": eval_metrics["episode_length_mean"],
            }
            save_checkpoint(
                run_dir / "checkpoints" / "latest.pt",
                i,
                policy_net,
                value_net,
                optimizer,
                checkpoint_metrics,
            )

            if eval_metrics["grader_score_mean"] > best_grader_score:
                best_grader_score = eval_metrics["grader_score_mean"]
                save_checkpoint(
                    run_dir / "checkpoints" / "best.pt",
                    i,
                    policy_net,
                    value_net,
                    optimizer,
                    checkpoint_metrics,
                )

            if i > 0 and i % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(
                    run_dir / "checkpoints" / f"iter_{i:05d}.pt",
                    i,
                    policy_net,
                    value_net,
                    optimizer,
                    checkpoint_metrics,
                )

            print(
                " | ".join(
                    [
                        f"Iter {i}",
                        f"TrainReward {reward:.3f}",
                        f"TrainAvg {running_reward:.3f}",
                        (
                            "TrainState "
                            f"L={rollout_metrics['latency_mean']:.3f} "
                            f"C={rollout_metrics['cpu_mean']:.3f} "
                            f"E={rollout_metrics['error_mean']:.3f}"
                        ),
                        f"Act {rollout_metrics['action_counts']}",
                        (
                            "GraderScore "
                            f"{eval_metrics['grader_score_mean']:.3f}"
                            f"+/-{eval_metrics['grader_score_std']:.3f}"
                        ),
                        (
                            "EvalState "
                            f"L={eval_metrics['latency_mean']:.3f} "
                            f"C={eval_metrics['cpu_mean']:.3f} "
                            f"E={eval_metrics['error_mean']:.3f}"
                        ),
                        f"EvalLen {eval_metrics['episode_length_mean']:.1f}",
                    ]
                )
            )

        if max_batches is not None and i + 1 >= max_batches:
            break

    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on the reliability environment.")
    parser.add_argument("--run-name", default=None, help="Optional name for the output run directory")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional limit for smoke tests")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task variant to train on")
    args = parser.parse_args()

    main(max_batches=args.max_batches, run_name=args.run_name, task_name=args.task)
