import torch.nn as nn
from envs.reliability_env import ReliabilityEnv

class ValueNet(nn.Module):
    def __init__(self, obs_dim=None):
        super().__init__()
        input_dim = obs_dim or ReliabilityEnv.OBSERVATION_DIM
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs):
        return self.net(obs)

