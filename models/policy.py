import torch
import torch.nn as nn
from envs.reliability_env import ReliabilityEnv

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(ReliabilityEnv.OBS_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.model(x)
