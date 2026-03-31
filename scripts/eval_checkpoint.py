import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from configs.config import DEVICE, EVAL_EPISODES, MAX_STEPS, SEED
from configs.loader import load_env_config
from models.policy import PolicyNet
from scripts.train_torchrl_ppo import evaluate_policy
from utils.seeding import set_global_seed


def load_run_config(checkpoint_path: Path):
    run_config_path = checkpoint_path.parent.parent / "run_config.json"
    if run_config_path.exists():
        return json.loads(run_config_path.read_text(encoding="utf-8"))
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Path to a checkpoint such as best.pt")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=SEED + 1000, help="Evaluation seed")
    parser.add_argument("--device", default=DEVICE, help="Torch device to use")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    run_config = load_run_config(checkpoint_path)
    env_config = run_config["env_config"] if run_config else load_env_config()
    device = torch.device(args.device)

    set_global_seed(args.seed)

    policy_net = PolicyNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy_net.load_state_dict(checkpoint["policy_state_dict"])

    metrics = evaluate_policy(
        policy_net,
        env_config,
        device,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    print(f"Checkpoint {checkpoint_path}")
    print(f"Iteration {checkpoint.get('iteration', 'unknown')}")
    print(
        " | ".join(
            [
                f"GraderScore {metrics['grader_score_mean']:.3f}+/-{metrics['grader_score_std']:.3f}",
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


if __name__ == "__main__":
    main()
