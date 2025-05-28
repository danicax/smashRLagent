from Agents.base_agent import Agent
from PolicyNet import PolicyNet
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch



class IQLAgent(Agent):
    def __init__(self, obs_dim, act_dim, device="cpu", gamma=0.99, iql_expectile=0.9, AWAC_lambda=0.1, param_update_freq=1000):
        super(IQLAgent, self).__init__()
        self.Qnet = nn.Sequential(
            nn.Linear(obs_dim+act_dim, 64), nn.ReLU(),
            nn.Linear(64,      64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.Qtarget = nn.Sequential(
            nn.Linear(obs_dim+act_dim, 64), nn.ReLU(),
            nn.Linear(64,      64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.Vnet = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64,      64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.policy = PolicyNet(obs_dim, act_dim, device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        self.Qoptimizer = optim.Adam(self.Qnet.parameters(), lr=0.001)
        self.Voptimizer = optim.Adam(self.Vnet.parameters(), lr=0.001)

        self.gamma = gamma
        self.iql_expectile = iql_expectile
        self.AWAC_lambda = AWAC_lambda
        self.param_update_freq = param_update_freq
        self.num_param_updates = 0

    def reward_function(self, states, actions, next_states):
        def get_feats(state):
            p1_stock = state[:, 0]
            p1_percent = state[:, 1]
            p2_stock = state[:, 17]
            p2_percent = state[:, 18]
            return p1_stock, p1_percent, p2_stock, p2_percent
        
        a,b,c,d = get_feats(states)
        e,f,g,h = get_feats(next_states)

        p1_stock_diff = (e - a) * 10
        p2_stock_diff = (c - g) * 10


        p1_percent_diff = 0
        p2_percent_diff = 0

        if p1_stock_diff == 0:
            p1_percent_diff = (f - b) * 0.1

        if p2_stock_diff == 0:
            p2_percent_diff = (d - h) * 0.1
            
        return p1_stock_diff + p2_stock_diff + p1_percent_diff + p2_percent_diff

    def expectile_loss(self, diff):
        weight = torch.where(diff > 0, self.iql_expectile, 1 -  self.iql_expectile)
        return weight * (diff**2)

    def update_V(self, states, actions):
        with torch.no_grad():
            Q = self.Qtarget(torch.cat([states, actions], dim=-1))
        V = self.Vnet(states)
        loss = torch.mean(self.expectile_loss(Q - V))

        self.Voptimizer.zero_grad()
        loss.backward()
        self.Voptimizer.step()

        return loss.item()

    def update_Q(self, states, actions, next_states):
        with torch.no_grad():
            V = self.Vnet(next_states)
            target = self.reward_function(states, actions, next_states) + self.gamma * V
        Q = self.Qnet(torch.cat([states, actions], dim=-1))
        loss = torch.nn.functional.mse_loss(Q, target)

        self.Qoptimizer.zero_grad()
        loss.backward()
        self.Qoptimizer.step()

        return loss.item()

    def update_policy(self, states, actions, adv):
        mu, logstd = self.policy(states)
        dist = Normal(loc=mu, scale=torch.exp(logstd))
        loss = dist.log_prob(actions) * torch.exp(adv / self.AWAC_lambda)
        loss = loss.mean()

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()


    def estimate_advantage(self, states, actions):
        with torch.no_grad():
            V = self.Vnet(states)
            Q = self.Qnet(torch.cat([states, actions], dim=-1))
        return Q - V

    def train(self, states, actions, next_states):
        Vloss = self.update_V(states, actions)
        Qloss = self.update_Q(states, actions, next_states)
        adv = self.estimate_advantage(states, actions)
        Ploss = self.update_policy(states, actions, adv)

        self.num_param_updates += 1
        if self.num_param_updates % self.param_update_freq == 0:
            self.update_Q_target()

        return Vloss, Qloss, Ploss

    def update_Q_target(self):
        self.Qtarget.load_state_dict(self.Qnet.state_dict())

    def predict(self, state):
        # TODO: implement predict
        return 0

