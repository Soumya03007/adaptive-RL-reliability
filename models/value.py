import torch.nn as nn

class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs):
        return self.net(obs)

