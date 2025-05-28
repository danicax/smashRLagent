from Agents.base_agent import Agent
from PolicyNet import PolicyNet
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch
from torch.optim.lr_scheduler import ExponentialLR  # Add this import


class BCAgent(Agent):
    def __init__(self, obs_dim, act_dim, device="cpu"):
        super(BCAgent, self).__init__()
        self.policy_net = PolicyNet(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)

    def train(self, states, actions, next_states):
        self.policy_net.train()
        mu, std = self.policy_net(states)               # mu: [B,17], logstd: [17]
        dist = Normal(loc=mu, scale=std)
        
        # log_prob has shape [B], one log‐prob per sample
        logp    = dist.log_prob(actions).sum(dim=-1)
        # 4) negative log-likelihood
        loss    = -logp.mean()

        
        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping with max norm of 1.0
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def predict(self, state):
        self.policy_net.eval()
        mu, std = self.policy_net(state.unsqueeze(0))

        dist = Normal(loc=mu, scale=std)
        action = dist.sample().squeeze(0)
        return action.cpu().numpy()

    def validate(self, states, actions):
        self.policy_net.eval()
        mu, std = self.policy_net(states)
        dist = Normal(loc=mu, scale=std)
        logp = dist.log_prob(actions).sum(dim=-1)
        return -logp.sum().item()
        