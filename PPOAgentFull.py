import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal
import melee
import argparse
from util import connect_to_console, make_obs_simple, unpack_and_send_simple, compute_reward, menu_helper
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time

class PPOPolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256, act_dim=17):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.cov_mat = torch.eye(act_dim, requires_grad=True)

        
    def forward(self, x):
        h = self.net(x)
        mu = torch.sigmoid(self.mu(h))
        dist = MultivariateNormal(mu, self.cov_mat)
        return dist

class PPOAgentFull():
    def __init__(self, obs_dim, act_dim=17, hidden_dim=256, max_steps_per_episode=1800, console_arguments=None, gamma=0.99, updates_per_episode=5, device='cpu'):
        self.max_steps_per_episode = max_steps_per_episode
        self.args = console_arguments
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.updates_per_episode = updates_per_episode
        self.clip_eps = 0.2
        self.device = device

        self.actor = PPOPolicyNet(obs_dim, hidden_dim=hidden_dim, act_dim=act_dim).to(device)
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        ).to(device)
        self.writer = SummaryWriter(log_dir=f'./logs/PPOAgentFull')

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4, weight_decay=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4, weight_decay=1e-5)


    def train(self, total_episodes):
        episode_num = 0
        while episode_num < total_episodes:
            episode_num += 1

            obs, actions, log_probs, rewards, batch_length = self.rollout()
            # print("obs.shape", obs.shape)
            # print("actions.shape", actions.shape)
            # print("log_probs.shape", log_probs.shape)
            # print("rewards.shape", rewards.shape)
            # print("batch_length", batch_length)

            self.writer.add_scalar('total_reward', sum(rewards), episode_num)
            
            rewards_to_go = self.compute_rewards_to_go(rewards)
            
            with torch.no_grad():
                V = self.critic(obs)
                A = rewards_to_go - V
                A = (A - A.mean()) / (A.std() + 1e-8)

            for _ in range(self.updates_per_episode):
                V, curr_log_probs = self.evaluate(obs, actions)
                ratio = torch.exp(curr_log_probs - log_probs)
                surr1 = ratio * A
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * A
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(V.squeeze(), rewards_to_go)
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # log actor and critic loss using tensorboard
                self.writer.add_scalar('actor_loss', actor_loss.item(), episode_num)
                self.writer.add_scalar('critic_loss', critic_loss.item(), episode_num)

            # save the model
            if episode_num % 10 == 0:
                torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(), 'actor_optimizer': self.actor_optimizer.state_dict(), 'critic_optimizer': self.critic_optimizer.state_dict()}, f'./models/PPOAgentFull_{episode_num}.pth')



    def evaluate(self, obs, act):
        v = self.critic(obs)
        dist = self.actor(obs)
        log_prob = dist.log_prob(act)
        return v, log_prob

    def compute_rewards_to_go(self, rewards):
        rewards_to_go = []
        discounted_reward = 0
        for i in range(len(rewards)-1, -1, -1):
            discounted_reward = rewards[i] + self.gamma * discounted_reward
            rewards_to_go.append(discounted_reward)
        return torch.tensor(rewards_to_go[::-1])


    def get_action(self, obs):
        with torch.no_grad():
            dist = self.actor(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action, log_prob
    
    def rollout(self):
        console, controller1, controller2 = connect_to_console(self.args)
        
        frames_in_game = 0
        obs_list = []
        action_list = []
        log_prob_list = []
        reward_list = []

        while True:
            gamestate = console.step()
            if gamestate is None:
                continue

            # If we're past menus, nothing to do—CPUs play themselves
            if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                frames_in_game += 1

                obs = make_obs_simple(gamestate)
                act, log_prob = self.get_action(obs.to(self.device))
                reward = compute_reward(gamestate)

                obs_list.append(obs)
                action_list.append(act)
                log_prob_list.append(log_prob)
                reward_list.append(reward)

                # break if we've reached the max steps per episode
                if frames_in_game > self.max_steps_per_episode:
                    console.stop()
                    time.sleep(5)
                    break

                # send the action to the controller and continue
                unpack_and_send_simple(controller1,act)
                continue

            menu_helper(gamestate, controller1, controller2)

            continue
            
        
        return (torch.stack(obs_list).to(self.device), 
                torch.stack(action_list).to(self.device), 
                torch.stack(log_prob_list).to(self.device), 
                torch.tensor(reward_list).to(self.device), 
                frames_in_game)



if __name__ == '__main__':
    dolphin_path = "/home/summertony717/School/cs224r/project/Slippi/squashfs-root/usr/bin"
    iso_path = "/home/summertony717/School/cs224r/project/melee.iso"

    # create an argparse for the console arguments
    parser = argparse.ArgumentParser(description='Run two CPUs vs each other using libmelee')
    parser.add_argument('--port1', '-p1', type=int,
                        help='Controller port for CPU 1', default=1)
    parser.add_argument('--port2', '-p2', type=int,
                        help='Controller port for CPU 2', default=2)
    parser.add_argument('--cpu-level1', type=int, default=9,
                        help='CPU difficulty for player 1 (0–9)')
    parser.add_argument('--cpu-level2', type=int, default=9,
                        help='CPU difficulty for player 2 (0–9)')
    parser.add_argument('--address', '-a', default="127.0.0.1",
                        help='IP address of Slippi/Wii')
    parser.add_argument('--dolphin_executable_path', '-e', default=dolphin_path,
                        help='Path to Dolphin executable')
    parser.add_argument('--iso', default=iso_path, type=str,
                        help='Path to Melee ISO')
    args = parser.parse_args()

    agent = PPOAgentFull(obs_dim=4, act_dim=2, hidden_dim=256, max_steps_per_episode=1800, console_arguments=args, gamma=0.99, updates_per_episode=5, device='cpu')
    agent.train(total_episodes=100)                         