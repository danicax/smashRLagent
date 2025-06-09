import sys
import melee
import random
import csv
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.distributions import Categorical
import argparse
import signal
from util import make_obs, make_obs_simple, min_dist_reward, stay_alive_reward, unpack_and_send, unpack_and_send_simple, connect_to_console, menu_helper, hit_and_death_detection
from PPOAgentFull import PPOAgentFull
import time
import numpy as np
from Agents.IQLAgent import IQLAgent

"""
   act = torch.tensor([
            float(controller_state.button[melee.enums.Button.BUTTON_A]),
            float(controller_state.button[melee.enums.Button.BUTTON_B]),
            float(controller_state.button[melee.enums.Button.BUTTON_D_DOWN]),
            float(controller_state.button[melee.enums.Button.BUTTON_D_LEFT]),
            float(controller_state.button[melee.enums.Button.BUTTON_D_RIGHT]),
            float(controller_state.button[melee.enums.Button.BUTTON_D_UP]),
            float(controller_state.button[melee.enums.Button.BUTTON_L]),
            float(controller_state.button[melee.enums.Button.BUTTON_R]),
            float(controller_state.button[melee.enums.Button.BUTTON_X]),
            float(controller_state.button[melee.enums.Button.BUTTON_Y]),
            float(controller_state.button[melee.enums.Button.BUTTON_Z]),
            # float(controller_state.button[melee.enums.Button.BUTTON_START]), # do we need this? @tony
            controller_state.main_stick[0], #x/y components
            controller_state.main_stick[1],
            controller_state.c_stick[0],
            controller_state.c_stick[1],
            controller_state.l_shoulder,
            controller_state.r_shoulder
        ], dtype=torch.float32)
"""



def check_port(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 4:
        raise argparse.ArgumentTypeError(
            f"{value} is an invalid controller port. Must be 1, 2, 3, or 4."
        )
    return ivalue

parser = argparse.ArgumentParser(description='Run two CPUs vs each other using libmelee')
parser.add_argument('--port1', '-p1', type=check_port,
                    help='Controller port for CPU 1', default=1)
parser.add_argument('--port2', '-p2', type=check_port,
                    help='Controller port for CPU 2', default=2)
parser.add_argument('--cpu-level1', type=int, default=9,
                    help='CPU difficulty for player 1 (0–9)')
parser.add_argument('--cpu-level2', type=int, default=9,
                    help='CPU difficulty for player 2 (0–9)')
parser.add_argument('--address', '-a', default="127.0.0.1",
                    help='IP address of Slippi/Wii')
parser.add_argument('--dolphin_executable_path', '-e', default=None,
                    help='Path to Dolphin executable')
parser.add_argument('--iso', default=None, type=str,
                    help='Path to Melee ISO')
# parser.add_argument('--model_path', type=str, default=None,
#                     help='Path to model')
parser.add_argument('--log_path', type=str, default=None,
                    help='Path to log file')
args = parser.parse_args()




def main(model_path, experiment_name):
    console, controller1, controller2 = connect_to_console(args)
    costume = 0

    # agent = PPOAgentFull(obs_dim=4, act_dim=2, hidden_dim=64, max_steps_per_episode=1800, console_arguments=args, gamma=0.99, updates_per_episode=10, device='cpu', experiment_name="eval")
    # agent.load_checkpoint(model_path)
    agent = IQLAgent(obs_dim=70, act_dim=17)
    state_dict = torch.load(model_path, map_location="cpu")
    agent.policy.load_state_dict(state_dict)
    agent.policy.eval()

    in_game_frame_count = 0
    total_reward = 0

    prev_gamestate = None
    gamestate = None

    time_to_hit = np.inf
    time_to_death = np.inf

    while True:
        prev_gamestate = gamestate
        gamestate = console.step()
        if gamestate is None:
            continue

        # If we're past menus, nothing to do—CPUs play themselves
        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            # obs = make_obs_simple(gamestate)
            obs = make_obs(gamestate)
            # reward = min_dist_reward(prev_gamestate, gamestate)
            # reward = stay_alive_reward(prev_gamestate, gamestate)
            reward = min_dist_reward(prev_gamestate, gamestate)
            # result = hit_and_death_detection(prev_gamestate, gamestate)
            # if "hit" in result:
                # time_to_hit = min(time_to_hit, in_game_frame_count)
            # if "death" in result:
                # time_to_death = min(time_to_death, in_game_frame_count)
            # reward = agent.reward_function(obs.unsqueeze(0), None, None).item()
            total_reward += reward
            # act = policy(obs)
            act = agent.predict(obs)
            in_game_frame_count += 1
            # unpack_and_send_simple(controller1,act)
            unpack_and_send(controller1,act)
            if in_game_frame_count > 3600:
                with open(f'eval_logs/{experiment_name}.txt', 'a+') as f:
                    # f.write(f"{model_path} Total reward: {total_reward} Time to hit: {time_to_hit} Time to death: {time_to_death}\n")
                    f.write(f"{model_path} Total reward: {total_reward}\n")
                console.stop()
                return
            continue


        # In the menus, select both CPUs and then autostart on the second
        menu_helper(gamestate, controller1, controller2)
        continue


experiment_name = "bc_min_dist"
# model_paths = glob.glob("checkpoints/iql_min_dist_expectile_0.8_*.pth")
model_paths = glob.glob(r"checkpoints/bc_*.pth")
model_paths = sorted(model_paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))
print(model_paths)

for model_path in model_paths[:]:
    for i in range(3):
        main(model_path, experiment_name)
        time.sleep(5)

# main(model_path="iql_5.pth")
