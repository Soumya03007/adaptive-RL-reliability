import torch
import torch.nn as nn
from envs.reliability_env import ReliabilityEnv

class PolicyNet(nn.Module):
    def __init__(self, obs_dim=None):
        super().__init__()
        input_dim = obs_dim or ReliabilityEnv.OBSERVATION_DIM
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self._init_policy_prior()

    def forward(self, x):
        return self.model(x)

    def _init_policy_prior(self):
        final_layer = self.model[-1]
        nn.init.zeros_(final_layer.weight)
        with torch.no_grad():
            # This environment is usually safer when the agent starts slightly
            # biased toward scaling up instead of idling or scaling down.
            final_layer.bias.copy_(torch.tensor([-0.35, -0.10, 0.45]))
