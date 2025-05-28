import torch.nn as nn
import torch

class PPOAgent(nn.Module):
    def __init__(self, obs_dim, n_buttons=11, n_analogs=6):
        super().__init__()
        hid = 128
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid),     nn.ReLU(),
        )

        # Bernoulli head for buttons
        self.button_logits = nn.Linear(hid, n_buttons)

        # Gaussian head for analogs
        self.analog_mu     = nn.Linear(hid, n_analogs)
        # one learnable log-std per analog dimension
        self.analog_logstd = nn.Parameter(torch.zeros(n_analogs))

        # Critic head
        self.value_head    = nn.Linear(hid, 1)

    def forward(self, x):
        h = self.shared(x)
        return {
            'logits': self.button_logits(h),        # for Bernoulli
            'mu'    : self.analog_mu(h),            # mean for Normal
            'logstd': self.analog_logstd,           # log-std for Normal
            'value' : self.value_head(h).squeeze(-1)
        }
    
    