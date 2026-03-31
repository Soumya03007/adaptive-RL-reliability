import torch.nn as nn
from envs.reliability_env import ReliabilityEnv

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ReliabilityEnv.OBS_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs):
        return self.net(obs)

