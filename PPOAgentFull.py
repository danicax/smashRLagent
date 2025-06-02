import torch.nn as nn
import torch
from torch.distributions import Normal
import melee
import argparse
from util import connect_to_console, make_obs_simple, unpack_and_send_simple, compute_reward, menu_helper
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
import os
import matplotlib.pyplot as plt

class PPOPolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256, act_dim=17):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.logstd = nn.Linear(hidden_dim, act_dim)

        
    def forward(self, x):
        h = self.net(x)
        mu = torch.sigmoid(self.mu(h))
        std = torch.exp(torch.clamp(self.logstd(h), -20, 0))
        # std = 0.25
        dist = Normal(mu, std)
        return dist

class PPOAgentFull():
    def __init__(self, obs_dim, act_dim=17, hidden_dim=256, max_steps_per_episode=3000, console_arguments=None, gamma=0.98, updates_per_episode=5, device='cpu'):
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
        self.writer = SummaryWriter(log_dir=f'./logs/PPO_Hate_The_Void_Small_Net_0.998')

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4, weight_decay=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-4, weight_decay=1e-5)


    def train(self, total_episodes):
        episode_num = 0
        while episode_num < total_episodes:
            episode_num += 1

            obs, actions, log_probs, rewards, batch_length = self.rollout()

            self.writer.add_scalar('total_reward', self.compute_average_total_rewards(rewards), episode_num)
            
            rewards_to_go = self.compute_rewards_to_go(rewards)
            
            with torch.no_grad():
                V = self.critic(obs)
                A = rewards_to_go - V
                A = (A - A.mean()) / (A.std() + 1e-8)

            for _ in range(self.updates_per_episode):   
                V, curr_log_probs, dist = self.evaluate(obs, actions)
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

            # if episode_num % 5 == 0:
                # self.visualize_policy(episode_num)
                # self.visualize_critic(episode_num)
            # save the model
            if episode_num % 5 == 0:
                torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(), 'actor_optimizer': self.actor_optimizer.state_dict(), 'critic_optimizer': self.critic_optimizer.state_dict()}, 
                f'./models/PPO_Hate_The_Void_Small_Net_0.998_{episode_num}.pth')

    def evaluate(self, obs, act):
        v = self.critic(obs)
        dist = self.actor(obs)
        log_prob = dist.log_prob(act).sum(dim=-1)
        return v, log_prob, dist

    def compute_rewards_to_go(self, batched_rewards):
        rewards_to_go = []

        for rewards in reversed(batched_rewards):
            discounted_reward = 0
            for r in reversed(rewards):
                discounted_reward = r + self.gamma * discounted_reward
                rewards_to_go.append(discounted_reward)
        return torch.tensor(rewards_to_go[::-1])


    def compute_average_total_rewards(self, rewards):
        v = sum([sum(e) for e in rewards]) / len(rewards)
        return v


    def get_action(self, obs):
        with torch.no_grad():
            dist = self.actor(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    
    def rollout(self):
        console, controller1, controller2 = connect_to_console(self.args)
        
        frames_in_game = 0
        obs_list = []
        action_list = []
        log_prob_list = []
        reward_list = [[]]
        frames_list = []

        while True:
            gamestate = console.step()
            if gamestate is None:
                continue

            # If we're past menus, nothing to do—CPUs play themselves
            if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:

                obs = make_obs_simple(gamestate)
                act, log_prob = self.get_action(obs.to(self.device))
                reward = compute_reward(gamestate)

                obs_list.append(obs)
                action_list.append(act)
                log_prob_list.append(log_prob)
                reward_list[-1].append(reward)

                # break if we've reached the max steps per episode
                if frames_in_game >= self.max_steps_per_episode:
                    console.stop()
                    frames_list.append(frames_in_game - sum(frames_list))
                    obs_list = obs_list[:self.max_steps_per_episode]
                    action_list = action_list[:self.max_steps_per_episode]
                    log_prob_list = log_prob_list[:self.max_steps_per_episode]
                    # bootstrap the last reward
                    with torch.no_grad():
                        reward_list[-1][-1] += self.critic(obs).item()
                    reward_list[-1] = reward_list[-1][1:]

                    time.sleep(5)
                    break

                # send the action to the controller and continue
                unpack_and_send_simple(controller1,act)

                frames_in_game += 1
                continue

            menu_helper(gamestate, controller1, controller2)


            # We've reached the post game screen
            if len(frames_list) > 0 and frames_in_game - sum(frames_list) > 0:         # Died. Append the most negative reward.
                frames_list.append(frames_in_game - sum(frames_list))
                reward_list[-1].append(compute_reward(gamestate))
                reward_list[-1] = reward_list[-1][1:]                                   # remove the reward for the first state of the game
                reward_list.append([])
            else:                                                                       # already accounted for this death
                continue
            
        

        if reward_list[-1] == []:
            reward_list.pop()

        assert len(obs_list) == len(action_list) == len(log_prob_list) == self.max_steps_per_episode
        assert sum([len(e) for e in reward_list]) == self.max_steps_per_episode
        assert sum(frames_list) == self.max_steps_per_episode

        return (torch.stack(obs_list).to(self.device), 
                torch.stack(action_list).to(self.device), 
                torch.stack(log_prob_list).to(self.device), 
                reward_list, 
                frames_list)

    def draw_grid(self, data, name, epoch, color_bar=True, vmin=0, vmax=1):
        plt.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
        if color_bar:
            # limit the color bar to between -1 and 1
            plt.colorbar()
        plt.xticks(range(0, 201, 20), labels=range(-100, 101, 20))
        plt.yticks(range(0, 201, 20), labels=range(100, -101, -20))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(f'visuals/{name}_{epoch}.png')
        plt.close()

    def visualize_critic(self, epoch):
        with torch.no_grad():
            vs = []
            for y in range(100, -101, -1):
                v_row = []
                for x in range(-100, 101):
                    v_row.append(self.critic(torch.tensor([x, y, 60.0, 0.0]).to(self.device)))
                vs.append(v_row)

            # draw this on a 2d grid and save it
            self.draw_grid(vs, 'critic', epoch, color_bar=True, vmin=0, vmax=30)

    def visualize_policy(self, epoch):
        with torch.no_grad():
            x_mus = []
            x_stds = []
            jump_mus = []
            jump_stds = []
            for y in range(100, -101, -1):
                mus_row = []
                x_stds_row = []
                jump_mus_row = []
                jump_stds_row = []
                for x in range(-100, 101):
                    dist = self.actor(torch.tensor([x, y, 60.0, 0.0]).to(self.device))
                    mus_row.append(dist.mean[0])
                    x_stds_row.append(dist.stddev[0])
                    jump_mus_row.append(dist.mean[1])
                    jump_stds_row.append(dist.stddev[1])
                x_mus.append(mus_row)
                x_stds.append(x_stds_row)
                jump_mus.append(jump_mus_row)
                jump_stds.append(jump_stds_row)
            
            self.draw_grid(x_mus, 'x_mus', epoch, color_bar=True)
            # self.draw_grid(x_stds, 'x_stds', epoch, color_bar=True)
            self.draw_grid(jump_mus, 'jump_mus', epoch, color_bar=True)
            # self.draw_grid(jump_stds, 'jump_stds', epoch, color_bar=True)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


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

    agent = PPOAgentFull(obs_dim=4, act_dim=2, hidden_dim=32, max_steps_per_episode=1800, console_arguments=args, gamma=0.998, updates_per_episode=10, device='cpu')
    # agent.load_checkpoint(checkpoint_path='./models/PPO_Social_Distance_60.pth')
    agent.train(total_episodes=400)                         