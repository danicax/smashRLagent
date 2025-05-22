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
from torch.nn.functional import mse_loss
from torch.distributions import Categorical
import argparse
import signal
from util import make_obs
from QNet import QNet
from ReplayBuffer import ReplayBuffer
import math

from torch.distributions import Bernoulli, Normal


move_inputs = {
    # No-stick normals
    "Jab":            [1,0,0,0,0,0,0,0,0,0,0,   0.5,0.5,0.5,0.5,0.5,0.5],
    "Neutral Tilt":   [1,0,0,0,0,0,0,0,0,0,0,   0.5,0.5,0.5,0.5,0.5,0.5],

    # Movement normals
    "Dash Attack":    [1,0,0,0,0,0,0,0,0,0,0,   1.0,0.5,0.5,0.5,0.5,0.5],
    "Forward Tilt":   [1,0,0,0,0,0,0,0,0,0,0,   1.0,0.5,0.5,0.5,0.5,0.5],
    "Up Tilt":        [1,0,0,0,0,1,0,0,0,0,0,   0.5,1.0,0.5,0.5,0.5,0.5],
    "Down Tilt":      [1,0,0,0,0,0,0,0,0,0,0,   0.5,0.0,0.5,0.5,0.5,0.5],

    # Smash Attacks (C-Stick flick)
    "Forward Smash":  [1,0,0,0,0,0,0,0,0,0,0,   0.5,0.5,1.0,0.5,0.5,0.5],
    "Up Smash":       [1,0,0,0,0,0,0,0,0,0,0,   0.5,0.5,0.5,1.0,0.5,0.5],
    "Down Smash":     [1,0,0,0,0,0,0,0,0,0,0,   0.5,0.5,0.5,0.0,0.5,0.5],

    # Aerials
    "Neutral Air":    [1,0,0,0,0,0,0,0,0,0,0,   0.5,0.5,0.5,0.5,0.5,0.5],
    "Forward Air":    [1,0,0,0,0,0,0,0,0,0,0,   1.0,0.5,0.5,0.5,0.5,0.5],
    "Back Air":       [1,0,0,0,1,0,0,0,0,0,0,   0.0,0.5,0.5,0.5,0.5,0.5],
    "Up Air":         [1,0,0,0,0,1,0,0,0,0,0,   0.5,1.0,0.5,0.5,0.5,0.5],
    "Down Air":       [1,0,0,0,0,0,0,0,0,0,0,   0.5,0.0,0.5,0.5,0.5,0.5],

    # Specials
    "Neutral B":      [0,1,0,0,0,0,0,0,0,0,0,   0.5,0.5,0.5,0.5,0.5,0.5],
    "Side B →":       [0,1,0,0,0,0,0,0,0,0,0,   1.0,0.5,0.5,0.5,0.5,0.5],
    "Side B ←":       [0,1,0,0,0,0,0,0,0,0,0,   0.0,0.5,0.5,0.5,0.5,0.5],
    "Up B":           [0,1,0,0,0,0,0,0,0,0,0,   0.5,1.0,0.5,0.5,0.5,0.5],
    "Down B":         [0,1,0,0,0,0,0,0,0,0,0,   0.5,0.0,0.5,0.5,0.5,0.5],

    # Grabs & Throws
    "Grab":           [0,0,0,0,0,0,0,0,0,0,1,   0.5,0.5,0.5,0.5,0.5,0.5],
    "Pummel":         [1,0,0,0,0,0,0,0,0,0,1,   0.5,0.5,0.5,0.5,0.5,0.5],
    "Forward Throw":  [1,0,0,0,0,0,0,0,0,0,1,   1.0,0.5,0.5,0.5,0.5,0.5],
    "Back Throw":     [1,0,0,0,1,0,0,0,0,0,1,   0.0,0.5,0.5,0.5,0.5,0.5],
    "Up Throw":       [1,0,0,0,0,0,0,0,0,0,1,   0.5,1.0,0.5,0.5,0.5,0.5],
    "Down Throw":     [1,0,0,0,0,0,0,0,0,0,1,   0.5,0.0,0.5,0.5,0.5,0.5],
}
ACTIONS = torch.tensor(list(move_inputs.values()), dtype=torch.float32)

N_ACTIONS = len(ACTIONS)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_net    = QNet(obs_dim=54, n_actions=N_ACTIONS).to(device)
target_q = QNet(obs_dim=54, n_actions=N_ACTIONS).to(device)
target_q.load_state_dict(q_net.state_dict())
opt      = optim.Adam(q_net.parameters(), lr=1e-4)
buffer   = ReplayBuffer()
eps_start, eps_end, eps_decay = 1.0, 0.1, 100_000
gamma = 0.99
update_target_every = 1000
step = 0

print(move_list)

def unpack_and_send(controller, action_tensor):
    """
    action_tensor: FloatTensor of shape [act_dim] in the same order you trained on:
      [A, B, D_DOWN, D_LEFT, D_RIGHT, D_UP,
       L, R, X, Y, Z, 
       main_x, main_y, c_x, c_y, l_shldr, r_shldr]
    """
    # First, clear last frame’s inputs
    #controller.release_all()

    # Booleans
    print("ACTION",action_tensor)
    btns = [
        melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_B, melee.enums.Button.BUTTON_D_DOWN,
        melee.enums.Button.BUTTON_D_LEFT, melee.enums.Button.BUTTON_D_RIGHT,melee.enums. Button.BUTTON_D_UP,
        melee.enums.Button.BUTTON_L, melee.enums.Button.BUTTON_R, melee.enums.Button.BUTTON_X,
        melee.enums.Button.BUTTON_Y, melee.enums.Button.BUTTON_Z #, melee.enums.Button.BUTTON_START
    ]
    
    for i, b in enumerate(btns):
        # if(b==melee.enums.Button.BUTTON_L or b==melee.enums.Button.BUTTON_R):
        #     continue
        if action_tensor[i].item() >0.5:
            controller.press_button(b)
        else:
            controller.release_button(b)
            # if(b == melee.enums.Button.BUTTON_A):
            #     print("A")

    #Analog sticks
    main_x, main_y = action_tensor[11].item(), action_tensor[12].item()
    c_x,    c_y    = action_tensor[13].item(), action_tensor[14].item()
    l_shoulder,    r_shoulder    = action_tensor[15].item(), action_tensor[16].item()

    controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, main_x, main_y)
    controller.tilt_analog(melee.enums.Button.BUTTON_C,    c_x,    c_y)
    controller.press_shoulder(melee.enums.Button.BUTTON_L, l_shoulder)
    controller.press_shoulder(melee.enums.Button.BUTTON_R, r_shoulder)


# Load the trained model
# model = PolicyNet(obs_dim=54, act_dim=17)
# state_dict = torch.load("trained_policy_distribution.pth", map_location="cpu")
# model.load_state_dict(state_dict)
# model.eval()

# def policy(obs):
#     with torch.no_grad():
#         mu, logstd = model(obs)  # → [1,17]
#         std = logstd.exp().unsqueeze(0)
#         dist   = Normal(mu, std)
#         sample = dist.sample()          # → [1,17]
#         action = sample.squeeze(0)      # → [17]
#         return action  # Integer action index


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
        cpu_level=9,
        costume=costume,
        autostart=True,    # <-- start when both have been selected
        swag=False
    )

prev_gamestate = None
state = None
action_idx = random.randrange(N_ACTIONS)

while True:
    # if gamestate is None:
    #     continue

    # gamestate = console.step()
    # if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
    #     unpack_and_send(controller1,ACTIONS[a_idx])
    #     next_gs = console.step();
    #     next_obs = make_obs(next_gs)
    #     reward = compute_reward(gamestate,next_gs)

    #     buffer.push(obs,a_idx,reward,next_obs,False)

    #     obs = make_obs(gamestate)

    #     eps = eps_end + (eps_start - eps_end) * math.exp(-1. * step / eps_decay)
    #     if random.random() < eps:
    #         a_idx = random.randrange(N_ACTIONS)
    #     else:
    #         with torch.no_grad():
    #             q_vals = q_net(obs)      # [1, N_ACTIONS]
    #             a_idx  = q_vals.argmax(dim=1).item()

    #         r = compute_reward(gamestate)

            
    #         act = policy(obs) # TONY!!
    #         unpack_and_send(controller1,ACTIONS[a_idx])
        
    #         continue;
    # else:
    #     gamestate = console.step()

    if gamestate is None:
        continue
    prev_gamestate = gamestate
    gamestate = console.step()
    if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        if prev_gamestate is None:
            
            continue
        #unpack_and_send(controller1, ACTIONS[action_idx])

        continue

    
    melee.MenuHelper.skip_postgame(controller1,gamestate)
    melee.MenuHelper.skip_postgame(controller2,gamestate)