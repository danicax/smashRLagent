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
from util import make_obs as make_obs
from QNet import QNet
from ReplayBuffer import ReplayBuffer
import math

from torch.distributions import Bernoulli, Normal

save_dir = "Double_DQN_Hard_XL_Everything"
os.makedirs(save_dir, exist_ok=True)

move_inputs = [
    [0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,-1,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,-1,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,-1],
    [0,0,0,0,0,0,0,0,0,0,0,0,1]
]
#ACTIONS = torch.tensor(list(move_inputs.values()), dtype=torch.float32)
ACTIONS = torch.tensor((move_inputs), dtype=torch.float32)

N_ACTIONS = len(move_inputs)
obs_dim = 70

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_net    = QNet(obs_dim=obs_dim, n_actions=N_ACTIONS).to(device)
target_q = QNet(obs_dim=obs_dim, n_actions=N_ACTIONS).to(device)
target_q.load_state_dict(q_net.state_dict())
opt      = optim.Adam(q_net.parameters(), lr=1e-6)
buffer   = ReplayBuffer()
eps_start, eps_end, eps_decay = 1.0, 0.1, 1_000
gamma = 0.995
update_target_every = 1000
step = 0
batch_size = 32

def compute_epsilon(step, eps_start=1.0, eps_end=0.1, eps_decay=1_000):
    """
    Exponentially decay ε from eps_start→eps_end over eps_decay steps.
    After many steps, ε → eps_end.
    """
    return eps_end + (eps_start - eps_end) * math.exp(-1. * step / eps_decay)

# def unpack_and_send(controller, action_tensor):
#     """
#     action_tensor: FloatTensor of shape [act_dim] in the same order you trained on:
#       [A, B, D_DOWN, D_LEFT, D_RIGHT, D_UP,
#        L, R, X, Y, Z, 
#        main_x, main_y, c_x, c_y, l_shldr, r_shldr]
#     """
#     # First, clear last frame’s inputs
#     controller.release_all()

#     # Booleans
#     if action_tensor[1].item() >0.5:
#         controller.press_button(melee.enums.Button.BUTTON_Y)
#     if action_tensor[2].item() >0.5:
#         controller.press_button(melee.enums.Button.BUTTON_L)

#     controller.tilt_analog_unit(melee.enums.Button.BUTTON_MAIN, action_tensor[0].item(), 0)

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
        melee.enums.Button.BUTTON_D_LEFT, melee.enums.Button.BUTTON_D_RIGHT,melee.enums.Button.BUTTON_D_UP,
        melee.enums.Button.BUTTON_L, melee.enums.Button.BUTTON_R, melee.enums.Button.BUTTON_X,
        melee.enums.Button.BUTTON_Z #, melee.enums.Button.BUTTON_START
    ]
    
    for i, b in enumerate(btns):
        if action_tensor[i].item() >0.5:
            controller.press_button(b)

    #Analog sticks
    main_x = action_tensor[10].item()
    c_x,    c_y    = action_tensor[11].item(), action_tensor[12].item()
    #l_shoulder,    r_shoulder    = action_tensor[13].item(), action_tensor[14].item()

    controller.tilt_analog_unit(melee.enums.Button.BUTTON_MAIN, main_x, 0)
    controller.tilt_analog_unit(melee.enums.Button.BUTTON_C,    c_x,    c_y)


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
    if gamestate is None or prev_gamestate is None:
        return 0.0
    # Example: reward based on stock difference
    player_stock = 0
    enemy_stock = 0

    if gamestate.players[1].stock < prev_gamestate.players[1].stock:
        
        player_stock = (int(gamestate.players[1].stock) - int(prev_gamestate.players[1].stock))*3
        #print("DEATH", player_stock)
    if gamestate.players[2].stock < prev_gamestate.players[2].stock:
        enemy_stock = -(int(gamestate.players[2].stock) - int(prev_gamestate.players[2].stock))*3

    player_hp = 0
    enemy_hp = 0

    if gamestate.players[1].off_stage:
        return -100

    if(player_stock == 0):
        if gamestate.players[1].percent > prev_gamestate.players[1].percent:
            player_hp = -(float(gamestate.players[1].percent) - float(prev_gamestate.players[1].percent)) * 0.001
    if(enemy_stock == 0):
        if gamestate.players[2].percent > prev_gamestate.players[2].percent:
            enemy_hp = (float(gamestate.players[2].percent) - float(prev_gamestate.players[2].percent)) * 0.001


    reward = player_stock + enemy_stock + player_hp + enemy_hp
    
    # if reward != 0:
    #     print("REWARD",reward)
    return reward
# def compute_reward(prev_gamestate, gamestate):
#     if gamestate is None:
#         return 0.0
    
#     if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
#         return 0.0
    
#     p1 = gamestate.players[1]
    
#     if p1.off_stage:
#         return -100
    
#     if gamestate.players[1].percent > prev_gamestate.players[1].percent:
#         return -(gamestate.players[1].percent - prev_gamestate.players[1].percent)

#     return 0

count = 0
num_games = 0
num_train = 5000
done = False
total_reward = 0
while True:

    
    if gamestate is None:
        continue

    prev_gamestate = gamestate
    
    if prev_gamestate is not None and prev_gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:

        unpack_and_send(controller1, ACTIONS[action_idx])
        prev_state = make_obs(prev_gamestate)

    gamestate = console.step()

    if gamestate is not None and gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        done = False
        count +=1
        if prev_gamestate is None or prev_state is None:
            continue
        state = make_obs(gamestate)
        reward = compute_reward(prev_gamestate, gamestate)
        #reward = compute_reward(gamestate)
        total_reward += reward
        
        
        # if len(buffer) == 0 or buffer.buf[-1][4]==False:
        #     if done:
        #         print("PUSHING DONE")
        buffer.push(prev_state, action_idx, reward, state, done)

        # update Q
        if len(buffer) >= batch_size:
           
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            # Compute the target Q-values
            with torch.no_grad():
            #     target_q_values = target_q(next_states).max(1)[0]
            #     target_q_values = rewards + gamma * target_q_values
            #with double DQN
                best_next_a    = q_net(next_states).argmax(dim=1, keepdim=True)  # [B,1]

                # 2) target network evaluates that action
                next_q_target  = target_q(next_states).gather(1, best_next_a).squeeze(1)  # [B]

                # 3) form the Double-DQN target
                target_q_values = rewards + (1-dones)*gamma * next_q_target

            # Compute the current Q-values
            q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute the loss
            loss = mse_loss(q_values, target_q_values)

            # Optimize the model
            opt.zero_grad()
            loss.backward()
            opt.step()
            if count % update_target_every == 0:
                print(f"[DQN] update {(count/update_target_every)}  "
                    f"loss={loss.item():.4f}  total_reward={total_reward:.4f}")
                total_reward = 0
                target_q.load_state_dict(q_net.state_dict())
            if count%num_train==0:
                num_games +=1
                torch.save(q_net.state_dict(), os.path.join(save_dir, f"trained_double_qnet_simple_{num_games}.pth"))

        # exploration/exploitation

        eps = compute_epsilon(count)
        if random.random() < eps:
            action_idx = random.randrange(N_ACTIONS)
        else:
            with torch.no_grad():
                q_vals = q_net(state)  # [1, N_ACTIONS]
                action_idx = q_vals.argmax().item()
        
        continue

    if gamestate is not None:
        if(done == False):
            if(len(buffer) > 0 and buffer.buf[-1][4]==False):
                print("Game ended, pushing last state")
                buffer.buf[-1] = (buffer.buf[-1][0], buffer.buf[-1][1], buffer.buf[-1][2], buffer.buf[-1][3], True)
        done = True
        #print("NONE")

        # done = True
        # prev_gamestate = None
        # prev_state = None
        # action_idx = random.randrange(N_ACTIONS)
    if gamestate is None:
        continue
    melee.MenuHelper.menu_helper_simple(
        gamestate,
        controller2,
        melee.Character.FALCO,
        melee.Stage.BATTLEFIELD,
        connect_code='',
        cpu_level=5,
        costume=costume,
        autostart=True,    # <-- start when both have been selected
        swag=False
    )
    melee.MenuHelper.skip_postgame(controller1,gamestate)
    melee.MenuHelper.skip_postgame(controller2,gamestate)