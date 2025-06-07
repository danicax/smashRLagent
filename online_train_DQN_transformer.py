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
from util import make_obs_simple as make_obs
#from QNet import QNet
from Agents.TransformerQNet import TransformerQNet as QNet
from ReplayBuffer import ReplayBufferTransformer as ReplayBuffer
import math
from collections import deque


from torch.distributions import Bernoulli, Normal

save_dir = "final_model_TRANSFORMER_stay_alive_simple"
os.makedirs(save_dir, exist_ok=True)

move_inputs = [
    [0,0,0],
    [0,1,0],
    [-1,0,0],
    [1,0,0],
    [0,0,1]
]
#ACTIONS = torch.tensor(list(move_inputs.values()), dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
ACTIONS = torch.tensor((move_inputs), dtype=torch.float32)
N_ACTIONS = len(move_inputs)

SEQ_LEN = 35  # or whatever your transformer expects
obs_dim = 6
state_seq = deque([torch.zeros(obs_dim, device=device) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)



q_net    = QNet(obs_dim=obs_dim, n_actions=N_ACTIONS,seq_len=SEQ_LEN)
target_q = QNet(obs_dim=obs_dim, n_actions=N_ACTIONS,seq_len=SEQ_LEN)
target_q.load_state_dict(q_net.state_dict())
opt      = optim.Adam(q_net.parameters(), lr=1e-4)
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
    if action_tensor[1].item() >0.5:
        controller.press_button(melee.enums.Button.BUTTON_Y)
    if action_tensor[2].item() >0.5:
        controller.press_button(melee.enums.Button.BUTTON_L)

    controller.tilt_analog_unit(melee.enums.Button.BUTTON_MAIN, action_tensor[0].item(), 0)


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
state = None

# def compute_reward(prev_gamestate, gamestate):
#     # Compute the reward based on the game state
#     # For now, just return a dummy reward
#     if prev_gamestate is None or gamestate is None:
#         return 0.0

#     p1 = gamestate.players[1]
#     p2 = gamestate.players[2]

#     dx = float(p1.position.x) - float(p2.position.x)
#     dy = float(p1.position.y) - float(p2.position.y)
#     dist = (dx ** 2 + dy ** 2) ** 0.5
#     reward = 1.0 / (dist + 1.0)
    
#     # if reward<0:
#     #     print("Reward: ", reward, "Player Stock: ", player_stock, "Enemy Stock: ", enemy_stock, "Player HP: ", player_hp, "Enemy HP: ", enemy_hp)
#     return reward
def get_padded_seq(seq_deque):
    """
    seq_deque: deque of <= SEQ_LEN Tensors,
    returns a Tensor [SEQ_LEN, obs_dim] where the "front" is padded with zeros.
    """
    current_len = len(seq_deque)
    if current_len < SEQ_LEN:
        pad = [torch.zeros(obs_dim, device=device)] * (SEQ_LEN - current_len)
        return torch.stack(pad + list(seq_deque), dim=0)
    else:
        return torch.stack(list(seq_deque), dim=0)
    
def compute_reward(prev_gamestate, gamestate):
    if gamestate is None:
        return 0.0
    
    if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        return 0.0
    
    p1 = gamestate.players[1]
    
    if p1.off_stage:
        return -100
    
    if gamestate.players[1].percent > prev_gamestate.players[1].percent:
        return -(gamestate.players[1].percent - prev_gamestate.players[1].percent)

    return 0

count = 0
num_games = 0
num_train = 5000
done = False
total_reward = 0
print("HI",device)
while True:
    
    if gamestate is None:
        continue
    prev_gamestate = gamestate
    
    if prev_gamestate is not None and prev_gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        
        unpack_and_send(controller1, ACTIONS[action_idx])

        if prev_state is None:
            tmp = make_obs(prev_gamestate)
            state_seq.append(tmp)
            prev_state = get_padded_seq(state_seq)
        else:
            prev_state = state
        #print("sending action: ", action_idx)
        
    gamestate = console.step()

    if gamestate is not None and gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        
        count +=1
        if prev_gamestate is None or prev_state is None:
            continue
        tmp = make_obs(gamestate)
        state_seq.append(tmp)
        state = get_padded_seq(state_seq)
        reward = compute_reward(prev_gamestate, gamestate)
        total_reward += reward
        
        buffer.push(prev_state, action_idx, reward, state, done)

        # # update Q
        if len(buffer) >= batch_size:
        
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            # Compute the target Q-values
            with torch.no_grad():
            #     target_q_values = target_q(next_states).max(1)[0]
            #     target_q_values = rewards + gamma * target_q_values
            #with double DQN
                #WTF LINE
                best_next_a = q_net(next_states).argmax(dim=1, keepdim=True)      # → [B, 1]
                #print("Best next action: ", best_next_a)

                # next_q_target = target_q(next_states).gather(1, best_next_a).squeeze(1)  # → [B]
                # target_q_values = rewards + (1 - dones) * gamma * next_q_target

            # Compute the current Q-values
            # q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # # Compute the loss
            # loss = mse_loss(q_values, target_q_values)

            # # Optimize the model
            # opt.zero_grad()
            # loss.backward()
            # opt.step()
            
            # if count % update_target_every == 0:
                
            #     print(f"[DQN] update {(count/update_target_every)}  "
            #         f"loss={loss.item():.4f}  total_reward={total_reward:.4f}")
            #     total_reward = 0
            #     target_q.load_state_dict(q_net.state_dict())
            # if count%num_train==0:
            #     num_games +=1
            #     torch.save(q_net.state_dict(), os.path.join(save_dir, f"trained_double_qnet_simple_transformer_{num_games}.pth"))

        # exploration/exploitation

        # eps = compute_epsilon(count)
        # if random.random() < eps:
        #     action_idx = random.randrange(N_ACTIONS)
        # else:
        #     with torch.no_grad():
        #         q_vals = q_net(state.unsqueeze(0))  # [1, N_ACTIONS]
        #         action_idx = q_vals.argmax().item()
        
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
    # prev_state = None
    state_seq = deque([torch.zeros(obs_dim, device=device) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
    #print("ENDED?")
    melee.MenuHelper.skip_postgame(controller1,gamestate)
    melee.MenuHelper.skip_postgame(controller2,gamestate)