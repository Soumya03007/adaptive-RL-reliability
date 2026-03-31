import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import torch.optim as optim

from configs.loader import load_env_config
from envs.reliability_env import ReliabilityEnv
from envs.torchrl_env import TorchRLEnvWrapper
from models.policy import PolicyNet

# --- Setup ---
env = TorchRLEnvWrapper(ReliabilityEnv(load_env_config()))
policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=0.001)

num_episodes = 550

for episode in range(num_episodes):
    td = env.reset()

    log_probs = []
    rewards = []
    entropies = []

    # --- Rollout ---
    for step in range(50):
        obs = td["observation"].float()

        logits = policy(obs)
        probs = F.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        td["action"] = action
        step_td = env.step(td)
        next_td = step_td["next"]

        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        rewards.append(next_td["reward"].item())

        td = next_td

        if td["done"].item():
            break

    # --- Compute discounted returns ---
    returns = []
    G = 0

    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)

    # Normalize returns (helps stability)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

    # --- Compute loss ---
    loss = sum(
        -log_prob * G - 0.01 * entropy
        for log_prob, G, entropy in zip(log_probs, returns, entropies)
    )

    # --- Update policy ---
    optimizer.zero_grad()
    loss.backward()

    # Prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)

    optimizer.step()

    # --- Logging ---
    if episode % 10 == 0:
        print(f"Episode {episode} | Total Reward: {sum(rewards):.2f}")
