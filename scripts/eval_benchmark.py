import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from configs.config import (
    BENCHMARK_SEEDS,
    BENCHMARK_TASKS,
    DEVICE,
    EVAL_EPISODES,
    MAX_STEPS,
)
from configs.loader import load_env_config
from models.policy import PolicyNet
from scripts.train_torchrl_ppo import evaluate_policy
from utils.seeding import set_global_seed


def parse_csv_arg(raw_value, cast):
    return [cast(part.strip()) for part in raw_value.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint across multiple tasks and seeds.")
    parser.add_argument("checkpoint", type=Path, help="Path to a checkpoint such as best.pt")
    parser.add_argument(
        "--tasks",
        default=",".join(BENCHMARK_TASKS),
        help="Comma-separated task variants to evaluate",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in BENCHMARK_SEEDS),
        help="Comma-separated evaluation seeds",
    )
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES, help="Episodes per task/seed pair")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Maximum steps per episode")
    parser.add_argument("--device", default=DEVICE, help="Torch device to use")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tasks = parse_csv_arg(args.tasks, str)
    seeds = parse_csv_arg(args.seeds, int)
    device = torch.device(args.device)

    set_global_seed(seeds[0] if seeds else 0)

    policy_net = PolicyNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy_net.load_state_dict(checkpoint["policy_state_dict"])

    task_scores = {}
    all_scores = []

    print(f"Checkpoint {checkpoint_path}")
    print(f"Iteration {checkpoint.get('iteration', 'unknown')}")

    for task_name in tasks:
        seed_scores = []

        for seed in seeds:
            metrics = evaluate_policy(
                policy_net,
                load_env_config(task_name),
                device,
                episodes=args.episodes,
                max_steps=args.max_steps,
                seed=seed,
            )
            seed_scores.append(metrics["grader_score_mean"])
            print(
                " | ".join(
                    [
                        f"Task {task_name}",
                        f"Seed {seed}",
                        f"GraderScore {metrics['grader_score_mean']:.3f}",
                        f"EvalLen {metrics['episode_length_mean']:.1f}",
                        (
                            "EvalState "
                            f"L={metrics['latency_mean']:.3f} "
                            f"C={metrics['cpu_mean']:.3f} "
                            f"E={metrics['error_mean']:.3f}"
                        ),
                    ]
                )
            )

        task_scores[task_name] = seed_scores
        all_scores.extend(seed_scores)

    print("\nTask Summary")
    for task_name, seed_scores in task_scores.items():
        print(
            f"{task_name}: mean={statistics.mean(seed_scores):.3f} "
            f"std={statistics.pstdev(seed_scores):.3f}"
        )

    print(
        "\nOverall "
        f"mean={statistics.mean(all_scores):.3f} "
        f"std={statistics.pstdev(all_scores):.3f}"
    )


if __name__ == "__main__":
    main()
