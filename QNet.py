import torch.nn as nn

class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,256), nn.ReLU(),
            nn.Linear(256,256),     nn.ReLU(),
            nn.Linear(256,256),     nn.ReLU(),
            nn.Linear(256, n_actions)
        )
    def forward(self, x):
        return self.net(x)  # [B, N_ACTIONS]