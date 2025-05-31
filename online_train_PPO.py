import sys, signal, argparse
import os, glob
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli, Normal, Categorical

from Agents.PPOAgent import PPOAgent
from ReplayBuffer import ReplayBufferPPO

import melee
from util import make_obs   # reuse your existing helpers

GAMMA       = 0.99
LAMBDA      = 0.95
CLIP_EPS    = 0.2
LR_ACTOR    = 1e-6
LR_CRITIC   = 1e-6
ROLLOUT_LEN = 1024
UPDATE_EPOCHS = 4
MINI_BATCH  = 64
TOTAL_UPDATES = 1000

obs_dim = 70
n_actions = 17

buffer   = ReplayBufferPPO(ROLLOUT_LEN, obs_dim, n_actions)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 3) PPO update function ---
def ppo_update(agent, buf, opt_act, opt_crit):
    obs_b, act_b, old_logp_b, ret_b, adv_b = buf.get()
    # split stored action:
    btn_b    = act_b[:, :11]
    analog_b = act_b[:, 11:]

    total_actor_loss  = 0.0
    total_critic_loss = 0.0
    n_updates = 0

    for _ in range(UPDATE_EPOCHS):
        logits, mu, logstd, vals = (
            agent(obs_b)['logits'],
            agent(obs_b)['mu'],
            agent(obs_b)['logstd'],
            agent(obs_b)['value']
        )
        std = logstd.exp()

        dist_btn    = Bernoulli(logits=logits)
        dist_analog = Normal(mu, std)

        logp_btn    = dist_btn.log_prob(btn_b).sum(-1)
        logp_analog = dist_analog.log_prob(analog_b).sum(-1)
        logp        = logp_btn + logp_analog

        ratio = (logp - old_logp_b).exp()
        surr1 = ratio * adv_b
        surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv_b
        loss_act = -torch.min(surr1, surr2).mean()
        loss_crit = (ret_b - vals).pow(2).mean()

        opt_act.zero_grad(); loss_act.backward()
        opt_crit.zero_grad(); loss_crit.backward()
        opt_act.step();    opt_crit.step()

        total_actor_loss  += loss_act.item()
        total_critic_loss += loss_crit.item()
        n_updates += 1
    return total_actor_loss / n_updates, total_critic_loss / n_updates

        

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
        if action_tensor[i].item() >0.5:
            controller.press_button(b)

    #Analog sticks
    main_x, main_y = action_tensor[11].item(), action_tensor[12].item()
    c_x,    c_y    = action_tensor[13].item(), action_tensor[14].item()
    l_shoulder,    r_shoulder    = action_tensor[15].item(), action_tensor[16].item()

    controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, main_x, main_y)
    controller.tilt_analog(melee.enums.Button.BUTTON_C,    c_x,    c_y)
    controller.press_shoulder(melee.enums.Button.BUTTON_L, l_shoulder)
    controller.press_shoulder(melee.enums.Button.BUTTON_R, r_shoulder)


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
# action_idx = random.randrange(N_ACTIONS)
agent = PPOAgent(obs_dim).to(device)


opt_actor  = actor = optim.Adam(
    list(agent.shared.parameters())
  + list(agent.button_logits.parameters())
  + list(agent.analog_mu.parameters())
  + [agent.analog_logstd],        # it’s a Parameter, so needs to go in manually
  lr=LR_ACTOR
)
opt_critic = optim.Adam(
    list(agent.value_head.parameters()),
    lr=LR_CRITIC
)

def compute_reward(prev_gamestate, gamestate):
    # Compute the reward based on the game state
    # For now, just return a dummy reward
    if prev_gamestate is None or gamestate is None:
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

    if(player_stock == 0):
        if gamestate.players[1].percent > prev_gamestate.players[1].percent:
            player_hp = -(float(gamestate.players[1].percent) - float(prev_gamestate.players[1].percent)) * 0.001
    if(enemy_stock == 0):
        if gamestate.players[2].percent > prev_gamestate.players[2].percent:
            enemy_hp = (float(gamestate.players[2].percent) - float(prev_gamestate.players[2].percent)) * 0.001


    reward = player_stock + enemy_stock + player_hp + enemy_hp
    if gamestate.players[1].position.y > 0:
        reward += 0.001
    
    # if reward != 0:
    #     print("REWARD",reward)
    return reward


count = 0
num_games = 0
num_train = 50
done = False
action = None
rollout_steps  = 0
update_count   = 0

while True:

    
    if gamestate is None:
        continue
    if gamestate.menu_state == melee.Menu.UNKNOWN_MENU:
        melee.MenuHelper.skip_postgame(controller1,gamestate)
        melee.MenuHelper.skip_postgame(controller2,gamestate)
        continue

    prev_gamestate = gamestate
    
    if prev_gamestate is not None and prev_gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:

        #unpack_and_send(controller1, ) SEND ACTION HERE VIA UNPACK AND SEND
        prev_state = make_obs(prev_gamestate)

        with torch.no_grad():
            out     = agent(prev_state.unsqueeze(0)) 
            logits  = out['logits'].squeeze(0)    # [11]
            mu      = out['mu'].squeeze(0)        # [6]
            std     = out['logstd'].exp()         # [6]
            dist_b  = Bernoulli(logits=logits)
            dist_a  = Normal(mu, std)

            btn_a   = dist_b.sample()             # [11]
            analog_a= dist_a.sample()             # [6]
            action  = torch.cat([btn_a, analog_a], dim=-1)  # [17]
            #print("ACTION",action.shape, btn_a.shape, analog_a.shape)

            logp  = dist_b.log_prob(btn_a).sum() + dist_a.log_prob(analog_a).sum()
            value = out['value'].item()
        unpack_and_send(controller1, action)

    gamestate = console.step()

    if gamestate is not None and gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        done = False
        count +=1
        if prev_gamestate is None or prev_state is None:
            continue
        state = make_obs(gamestate)
        reward = compute_reward(prev_gamestate, gamestate)
        buffer.store(prev_state, action, logp, reward, out['value'].item())
        rollout_steps += 1


        if rollout_steps >= ROLLOUT_LEN:
        # bootstrap last value
            with torch.no_grad():
                last_val = agent(prev_state.unsqueeze(0))['value'].item()

            buffer.finish_path(last_val)   # compute GAE & returns
            actor_loss, critic_loss = ppo_update(agent, buffer, opt_actor, opt_critic)
            buffer.reset()                 # clear out old rollout
            rollout_steps = 0
            update_count += 1
            print(f"[PPO] update {update_count}/{TOTAL_UPDATES}  "
            f"actor_loss={actor_loss:.4f}  critic_loss={critic_loss:.4f}")
            count+=1
            if count%num_train==0:
                num_games +=1
                torch.save(agent.state_dict(), f"trained_PPO_pain_{num_games}.pth")

        
        continue

    #if gamestate is not None:
        
    if gamestate is None:
        continue

    if gamestate is not None:
        if(done == False and count>0):
            print("Game ended, pushing last state")
            if(buffer.ptr>0):
                buffer.finish_path(last_val=0.0)
                rollout_steps = 0
                update_count += 1
                ppo_update(agent, buffer, opt_actor, opt_critic)
                buffer.reset()
            
    done = True
    
    melee.MenuHelper.skip_postgame(controller1,gamestate)
    melee.MenuHelper.skip_postgame(controller2,gamestate)