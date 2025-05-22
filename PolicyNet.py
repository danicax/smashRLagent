import torch
import torch.nn as nn
from torch.distributions import Normal

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim=17):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64,      64), nn.ReLU(),
        )
        self.mu     = nn.Linear(64, act_dim)         # mean for each of 17 dims
        self.logstd = nn.Parameter(torch.zeros(act_dim))  # log-std for each

    def forward(self, x):
        h      = self.net(x)
        mu     = self.mu(h)               # [B,17]
        logstd = self.logstd             # [17]
        return mu, logstd