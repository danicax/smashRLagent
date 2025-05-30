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
from util import make_obs
# from PolicyNet import PolicyNet
# from QNet import QNet
# from PolicyNet import PolicyNet
# from Agents.BCAgent import BCAgent
from Agents.PPOAgent import PPOAgentSimple

from torch.distributions import Bernoulli, Normal

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




# def unpack_and_send(controller, action_tensor):
#     """
#     action_tensor: FloatTensor of shape [act_dim] in the same order you trained on:
#       [A, B, D_DOWN, D_LEFT, D_RIGHT, D_UP,
#        L, R, X, Y, Z, START,
#        main_x, main_y, c_x, c_y, raw_x, raw_y, l_shldr, r_shldr]
#     """
#     # First, clear last frame’s inputs
#     #controller.release_all()

#     # Booleans
#     # print("ACTION",action_tensor)
#     btns = [
#         melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_B, melee.enums.Button.BUTTON_D_DOWN,
#         melee.enums.Button.BUTTON_D_LEFT, melee.enums.Button.BUTTON_D_RIGHT,melee.enums. Button.BUTTON_D_UP,
#         melee.enums.Button.BUTTON_L, melee.enums.Button.BUTTON_R, melee.enums.Button.BUTTON_X,
#         melee.enums.Button.BUTTON_Y, melee.enums.Button.BUTTON_Z #, melee.enums.Button.BUTTON_START
#     ]

#     #Analog sticks
#     main_x, main_y = action_tensor[11].item(), action_tensor[12].item()
#     c_x,    c_y    = action_tensor[13].item(), action_tensor[14].item()
#     l_shoulder,    r_shoulder    = action_tensor[15].item(), action_tensor[16].item()

#     controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, main_x, main_y)
#     controller.tilt_analog(melee.enums.Button.BUTTON_C,    c_x,    c_y)
#     controller.press_shoulder(melee.enums.Button.BUTTON_L, l_shoulder)
#     controller.press_shoulder(melee.enums.Button.BUTTON_R, r_shoulder)
    
#     for i, b in enumerate(btns):
#         if action_tensor[i].item() >0.5:
#             controller.press_button(b)
def unpack_and_send(controller, action_tensor):
    """
    action_tensor: FloatTensor of shape [2] for main stick [x, y]
    """
    controller.release_all()
    main_x, main_y = action_tensor[0].item(), action_tensor[1].item()
    controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, main_x, main_y)


#Load the trained model

# model = PolicyNet(obs_dim=54, act_dim=17)
# state_dict = torch.load("D:\cs224rPython\trained_qnet_630.pth.pth", map_location="cpu")

# model = QNet(obs_dim=70, act_dim=17)
# state_dict = torch.load("trained_qnet_630.pth", map_location="cpu")

# def policy(obs):
#     with torch.no_grad():
#         mu, logstd = model(obs)  # → [1,17]
#         std = logstd.exp().unsqueeze(0)
#         dist   = Normal(mu, std)
#         sample = dist.sample()          # → [1,17]
#         action = sample.squeeze(0)      # → [17]
#         return action  # Integer action index

#agent = PPOAgent(obs_dim=70, n_buttons=11, n_analogs=6)
agent = PPOAgentSimple(obs_dim=70)
state_dict = torch.load("trained_PPO_pain_17.pth", map_location="cpu")
agent.load_state_dict(state_dict)
agent.eval()

# def policy(obs):
#     with torch.no_grad():
#         out = agent(obs.unsqueeze(0))  # Add batch dimension
#         logits = out['logits']
#         mu = out['mu']
#         logstd = out['logstd']
#         std = logstd.exp()
#         # Sample buttons (Bernoulli) and analogs (Normal)
#         btns = (torch.sigmoid(logits) > 0.5).float().squeeze(0)
#         analogs = mu.squeeze(0)  # Use mean for deterministic eval, or sample: Normal(mu, std).sample()
#         action = torch.cat([btns, analogs])
#         return action
    
def policy(obs):
    with torch.no_grad():
        out = agent(obs.unsqueeze(0))  # Add batch dimension
        mu = out['mu'].squeeze(0)      # [2]
        # For deterministic: use mu; for stochastic: sample from Normal
        # std = out['logstd'].exp()
        # dist = Normal(mu, std)
        # action = dist.sample()
        action = mu  # deterministic
        return action


# agent = BCAgent(obs_dim=70, act_dim=17)
# # model = PolicyNet(obs_dim=70, act_dim=17)
# state_dict = torch.load("trained_PPO_12.pth", map_location="cpu")
# agent.policy_net.load_state_dict(state_dict)
# agent.policy_net.eval()

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
args = parser.parse_args()

# No logging since these are vanilla CPUs
console = melee.Console(path=args.dolphin_executable_path,
                        slippi_address=args.address,
                        logger=None)

# Two virtual controllers
controller1 = melee.Controller(console=console,
                               port=args.port1,
                               type=melee.ControllerType.STANDARD)
controller2 = melee.Controller(console=console,
                               port=args.port2,
                               type=melee.ControllerType.STANDARD)

def signal_handler(sig, frame):
    console.stop()
    print("Shutting down cleanly…")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Launch Dolphin + ISO
console.run(iso_path=args.iso)

print("Connecting to console…")
if not console.connect():
    print("ERROR: Failed to connect to the console.")
    sys.exit(-1)
print("Console connected")

# Plug in BOTH controllers
print(f"Connecting controller on port {args.port1}…")
if not controller1.connect():
    print("ERROR: Failed to connect controller1.")
    sys.exit(-1)
print("Controller1 connected")

print(f"Connecting controller on port {args.port2}…")
if not controller2.connect():
    print("ERROR: Failed to connect controller2.")
    sys.exit(-1)
print("Controller2 connected")

prev_gamestate = None


costume = 0
for _ in range(0,150):
    gamestate = console.step()
    if gamestate is None:
        continue

    # If we're past menus, nothing to do—CPUs play themselves
    if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        continue

    # In the menus, select both CPUs and then autostart on the second
    melee.MenuHelper.menu_helper_simple(
        gamestate,
        controller1,
        melee.Character.FOX,
        melee.Stage.BATTLEFIELD,
        connect_code='',
        cpu_level=0,
        costume=costume,
        autostart=False,
        swag=False
    )
    melee.MenuHelper.menu_helper_simple(
        gamestate,
        controller2,
        melee.Character.FALCO,
        melee.Stage.BATTLEFIELD,
        connect_code='',
        cpu_level=0,
        costume=costume,
        autostart=True,    # <-- start when both have been selected
        swag=False
    )
while True:
    prev_gamestate = gamestate
    gamestate = console.step()
    if gamestate is None:
        continue
    
    if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        obs = make_obs(gamestate)
        #act = agent.predict(obs)
        #if prev_gamestate is not None and gamestate is not None:
        act = policy(obs)
        unpack_and_send(controller1,act)
       
    # if gamestate is None:
    #     continue
        continue;
    
    melee.MenuHelper.skip_postgame(controller1,gamestate)
    melee.MenuHelper.skip_postgame(controller2,gamestate)