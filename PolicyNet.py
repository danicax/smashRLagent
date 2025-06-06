import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal
import torch.nn.functional as F
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim=17):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128,      128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.mu     = nn.Linear(128, act_dim)         # mean for each of 17 dims
        # self.logstd = nn.Parameter(torch.zeros((obs_dim, act_dim)))  # log-std for each
        self.logstd = nn.Parameter(torch.zeros(act_dim))  # log-std for each

    def forward(self, x):
        h      = self.net(x)
        mu     = self.mu(h)               # [B,17]
        mu     = torch.sigmoid(mu)       # use sigmoid to ensure the output is between 0 and 1
        
        logstd = torch.clamp(self.logstd, min=-20, max=0)
        # logstd = torch.sigmoid(x @ self.logstd) * 6 - 5
        # logstd = torch.sigmoid(self.logstd(x)) * 6 - 5
        # logstd = torch.clamp(logstd, min=-5, max=1)
        std    = torch.exp(logstd)
        return Normal(loc=mu, scale=std)
    

# class AttentionLayer(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(AttentionLayer, self).__init__()
#         self.query = nn.Linear(input_dim, hidden_dim)
#         self.key = nn.Linear(input_dim, hidden_dim)
#         self.value = nn.Linear(input_dim, hidden_dim)
#         self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
#     def forward(self, x):
#         # x shape: [batch_size, seq_len, input_dim]
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)
        
#         # Scaled dot-product attention
#         attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
#         attention = F.softmax(attention, dim=-1)
#         # print(attention.shape)
#         # Apply attention to values
#         out = torch.matmul(attention, V)
#         return out

# class PolicyNet(nn.Module):
#     def __init__(self, obs_dim, act_dim, hidden_dim=4):
#         super().__init__()
#         self.attention = AttentionLayer(1, hidden_dim)
        
#         self.fc1 = nn.Linear(hidden_dim * obs_dim, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, act_dim)
        
#         # Separate layers for mean and std
#         self.mean_layer = nn.Linear(act_dim, act_dim)
#         self.std_layer = nn.Linear(act_dim, act_dim)
        
#     def forward(self, x):
#         # Apply attention
#         x = x.unsqueeze(-1)  # Add sequence dimension
#         x = self.attention(x)
#         # print(x.shape)
#         x = x.reshape(x.shape[0], -1)

        
#         # Process through MLP
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
        
#         # Get mean and std
#         mu = self.mean_layer(x)
#         logstd = torch.clamp(self.std_layer(x), min=-5, max=1)
#         std    = torch.exp(logstd)
        
#         return Normal(loc=mu, scale=std)
        