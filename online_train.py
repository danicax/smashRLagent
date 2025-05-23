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
    "BDash Attack":    [1,0,0,0,0,0,0,0,0,0,0,  0.0,0.5,0.5,0.5,0.5,0.5],
    "BForward Tilt":   [1,0,0,0,0,0,0,0,0,0,0,  0.0,0.5,0.5,0.5,0.5,0.5],

    "Up Tilt":        [1,0,0,0,0,1,0,0,0,0,0,   0.5,1.0,0.5,0.5,0.5,0.5],
    "Down Tilt":      [1,0,0,0,0,0,0,0,0,0,0,   0.5,0.0,0.5,0.5,0.5,0.5],

    # Smash Attacks (C-Stick flick)
    "Forward Smash":  [1,0,0,0,0,0,0,0,0,0,0,   0.5,0.5,1.0,0.5,0.5,0.5],
    "BForward Smash":  [1,0,0,0,0,0,0,0,0,0,0,  0.5,0.5,0.0,0.5,0.5,0.5],
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

    # Movement / positioning
    "Walk Left":   [0,0,0,0,0,0,0,0,0,0,0,   0.0,0.5,0.5,0.5,0.5,0.5],
    "Walk Right":  [0,0,0,0,0,0,0,0,0,0,0,   1.0,0.5,0.5,0.5,0.5,0.5],
    "Crouch":      [0,0,0,0,0,0,0,0,0,0,0,   0.5,0.0,0.5,0.5,0.5,0.5],
    
    # Aerial jump
    # (tap Y in neutral)
    "Jump":        [0,0,0,0,0,0,0,0,0,1,0,   0.5,0.5,0.5,0.5,0.5,0.5],
}
ACTIONS = torch.tensor(list(move_inputs.values()), dtype=torch.float32)

N_ACTIONS = len(move_inputs)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_net    = QNet(obs_dim=70, n_actions=N_ACTIONS).to(device)
target_q = QNet(obs_dim=70, n_actions=N_ACTIONS).to(device)
target_q.load_state_dict(q_net.state_dict())
opt      = optim.Adam(q_net.parameters(), lr=1e-4)
buffer   = ReplayBuffer()
eps_start, eps_end, eps_decay = 1.0, 0.5, 10_000
gamma = 0.99
update_target_every = 1000
step = 0
batch_size = 32

def compute_epsilon(step, eps_start=1.0, eps_end=0.5, eps_decay=10_000):
    """
    Exponentially decay ε from eps_start→eps_end over eps_decay steps.
    After many steps, ε → eps_end.
    """
    return eps_end + (eps_start - eps_end) * math.exp(-1. * step / eps_decay)

def unpack_and_send(controller, action_tensor):
    """
    action_tensor: FloatTensor of shape [act_dim] in the same order you trained on:
      [A, B, D_DOWN, D_LEFT, D_RIGHT, D_UP,
       L, R, X, Y, Z, 
       main_x, main_y, c_x, c_y, l_shldr, r_shldr]
    """
    # First, clear last frame’s inputs
    controller.release_all()

    # Booleans
    #print("ACTION",action_tensor)
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
prev_state = None
action_idx = random.randrange(N_ACTIONS)

def compute_reward(prev_gamestate, gamestate):
    # Compute the reward based on the game state
    # For now, just return a dummy reward
    if prev_gamestate is None or gamestate is None:
        return 0.0

    # Example: reward based on stock difference
    player_stock = (int(gamestate.players[1].stock) - int(prev_gamestate.players[1].stock)) * 10.0
    enemy_stock = -(int(gamestate.players[2].stock) - int(prev_gamestate.players[2].stock)) * 10.0

    player_hp = 0
    enemy_hp = 0

    if(player_stock == 0):
        player_hp = -(float(gamestate.players[1].percent) - float(prev_gamestate.players[1].percent)) * 0.1
    if(enemy_stock == 0):
        enemy_hp = (float(gamestate.players[2].percent) - float(prev_gamestate.players[2].percent)) * 0.1


    reward = player_stock + enemy_stock + player_hp + enemy_hp
    return reward

def check_done(gamestate):
    # Check if the game is over
    if gamestate.menu_state in [melee.Menu.POSTGAME_SCORES] or gamestate is None:
        return True
    return False

count = 0
num_games = 0
name = 0
num_train = 100000
while True:

    if gamestate is None:
        continue

    prev_gamestate = gamestate
    if prev_gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        #print("ACTION",action_idx)
        unpack_and_send(controller1, ACTIONS[action_idx])
        prev_state = make_obs(prev_gamestate)

    gamestate = console.step()
    done = check_done(gamestate)
    # if done:
    #     num_games += 1
    #     prev_gamestate = None
    #     prev_state = None
    #     action_idx = random.randrange(N_ACTIONS)
    if(num_games==num_train):
        num_games = 0
        name+=1
        torch.save(q_net.state_dict(), f"trained_qnet{name}.pth")

    if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        count +=1
        if prev_gamestate is None or prev_state is None:
            continue
        state = make_obs(gamestate)
        reward = compute_reward(prev_gamestate, gamestate)
        

        buffer.push(prev_state, action_idx, reward, state, done)

        # update Q
        if len(buffer) >= batch_size:
            num_games+=1
            batch = buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = batch

            # Compute the target Q-values
            with torch.no_grad():
                target_q_values = target_q(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * gamma * target_q_values

            # Compute the current Q-values
            q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute the loss
            loss = mse_loss(q_values, target_q_values)

            # Optimize the model
            opt.zero_grad()
            loss.backward()
            opt.step()

        # exploration/exploitation

        eps = compute_epsilon(count)
        if random.random() < eps:
            action_idx = random.randrange(N_ACTIONS)
        else:
            with torch.no_grad():
                q_vals = q_net(state)  # [1, N_ACTIONS]
                action_idx = q_vals.argmax().item()
                if done:
                    prev_gamestate = None
                    prev_state = None
                    action_idx = random.randrange(N_ACTIONS)
        #print("ACTION",action_idx)        

        
            
            
                #print(f"Q-network saved to trained_qnet{num_games}.pth")
        continue
    
    melee.MenuHelper.skip_postgame(controller1,gamestate)
    melee.MenuHelper.skip_postgame(controller2,gamestate)