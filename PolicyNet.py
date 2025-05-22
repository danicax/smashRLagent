import torch.nn as nn
import torch

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64),      nn.ReLU(),
            #nn.Linear(64, act_dim)
        )
        self.mu      = nn.Linear(64, act_dim)
        self.logstd  = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logstd