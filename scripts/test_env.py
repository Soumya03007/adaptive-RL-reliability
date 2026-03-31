import sys
from pathlib import Path

# Add parent directory to path so we can import envs
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.loader import load_env_config
from envs.reliability_env import ReliabilityEnv

env = ReliabilityEnv(load_env_config())
state = env.reset()

for step in range(20):
    action = step % 3  # cycle through scale down / no-op / scale up
    state, reward, done, _ = env.step(action)

    print(f"Step {step}")
    print("State:", state)
    print("Reward:", reward)
    print("-" * 30)

    if done:
        print("Episode ended")
        break
