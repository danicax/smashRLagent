import sys, signal, argparse
import os, glob
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli, Normal, Categorical
import torch.nn.functional as F

#from Agents.PPOAgent import PPOAgent
from Agents.PPOAgent import PPOAgentDiscrete
from ReplayBuffer import ReplayBufferPPODiscrete as ReplayBufferPPO

import melee
from util import make_obs as make_obs 

GAMMA       = 0.995
LAMBDA      = 0.95
CLIP_EPS    = 0.2
LR_ACTOR    = 1e-6
LR_CRITIC   = 1e-5
ROLLOUT_LEN = 1024
UPDATE_EPOCHS = 4
MINI_BATCH  = 64
TOTAL_UPDATES = 200

ENT_COEF_START = 1
ENT_COEF_END = 0.01
MAX_GRAD_NORM = 0.5

obs_dim = 70

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
ACTIONS = torch.tensor(move_inputs, dtype=torch.float32)
N_ACTIONS = len(move_inputs)

# --- 3) PPO update function ---
# def ppo_update(agent, buf, opt_act, opt_crit):
#     obs_b, act_b, old_logp_b, ret_b, adv_b = buf.get()
#     # split stored action:
#     btn_b    = act_b[:, :11]
#     analog_b = act_b[:, 11:]

#     total_actor_loss  = 0.0
#     total_critic_loss = 0.0
#     n_updates = 0

#     for _ in range(UPDATE_EPOCHS):
#         logits, mu, logstd, vals = (
#             agent(obs_b)['logits'],
#             agent(obs_b)['mu'],
#             agent(obs_b)['logstd'],
#             agent(obs_b)['value']
#         )
#         std = logstd.exp()

#         dist_btn    = Bernoulli(logits=logits)
#         dist_analog = Normal(mu, std)

#         logp_btn    = dist_btn.log_prob(btn_b).sum(-1)
#         logp_analog = dist_analog.log_prob(analog_b).sum(-1)
#         logp        = logp_btn + logp_analog

#         ratio = (logp - old_logp_b).exp()
#         surr1 = ratio * adv_b
#         surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv_b
#         loss_act = -torch.min(surr1, surr2).mean()
#         loss_crit = (ret_b - vals).pow(2).mean()

#         opt_act.zero_grad(); loss_act.backward()
#         opt_crit.zero_grad(); loss_crit.backward()
#         opt_act.step();    opt_crit.step()

#         total_actor_loss  += loss_act.item()
#         total_critic_loss += loss_crit.item()
#         n_updates += 1
#     return total_actor_loss / n_updates, total_critic_loss / n_updates



def ppo_update(agent, buf, opt_act, opt_crit, update_count):
    obs_b, act_b, old_logp_b, ret_b, adv_b = buf.get()
    N = obs_b.size(0)

    # 1) normalize with population std
    adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)

    # 2) precompute indices
    action_idx = act_b.long()  # [N]

    total_a, total_c, steps = 0.0, 0.0, 0

    for _ in range(UPDATE_EPOCHS):
        perm = torch.randperm(N, device=obs_b.device)
        for start in range(0, N, MINI_BATCH):
            mb = perm[start:start+MINI_BATCH]

            obs_m     = obs_b[mb]
            old_lp_m  = old_logp_b[mb]
            ret_m     = ret_b[mb]
            adv_m     = adv_b[mb]
            action_idx_m = action_idx[mb]

            out       = agent(obs_m)
            logits    = out['logits']
            vals      = out['value'].squeeze(-1)

            dist      = Categorical(logits=logits)
            logp_m    = dist.log_prob(action_idx_m)
            ratio     = torch.exp(logp_m - old_lp_m)

            s1        = ratio * adv_m
            s2        = torch.clamp(ratio,1-CLIP_EPS,1+CLIP_EPS)*adv_m
            loss_a    = -torch.min(s1, s2).mean()

            # entropy bonus
            decay_frac = max(0.0, 1.0 - update_count / float(TOTAL_UPDATES))
            ENT_COEF = ENT_COEF_END + (ENT_COEF_START - ENT_COEF_END) * decay_frac

            ent = dist.entropy().mean()
            loss_a -= ENT_COEF * ent

            # critic
            loss_c = F.mse_loss(vals, ret_m)

            # backward + clip + step
            opt_act.zero_grad()
            opt_crit.zero_grad()
            loss_a.backward(retain_graph=True)
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            opt_act.step()
            opt_crit.step()

            total_a += loss_a.item()
            total_c += loss_c.item()
            steps  += 1

    return total_a/steps, total_c/steps

num_left = 0
num_right = 0
num_down = 0
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



# MIN DIST REWARD
# def compute_reward(gamestate):
#     if gamestate is None:
#         return 0.0
    
#     p1 = gamestate.players[1]
#     p2 = gamestate.players[2]

#     dx = float(p1.position.x) - float(p2.position.x)
#     dy = float(p1.position.y) - float(p2.position.y)
#     dist = (dx ** 2 + dy ** 2) ** 0.5
#     reward = (1.0 / (dist + 1.0))*10
#     #print(reward)
#     return reward

# STAY ALIVE REWARD
# def compute_reward(prev_gamestate,gamestate):
#     if gamestate is None:
#         return 0.0
    
#     reward = +0.01

#     # 2) if Fox died between prev_gs and gs, give a big penalty
#     if gamestate.players[1].stock < prev_gamestate.players[1].stock:
#         reward = -1.0  # harsh death signal
#     return reward

# DO DMG REWARD
# def compute_reward(prev_gamestate, gamestate):
#     if gamestate is None:
#         return 0.0
    
#     p1 = gamestate.players[1]
#     p2 = gamestate.players[2]

#     dx = float(p1.position.x) - float(p2.position.x)
#     dy = float(p1.position.y) - float(p2.position.y)
#     dist = (dx ** 2 + dy ** 2) ** 0.5
#     reward = (1.0 / (dist + 1.0))*0.001

#     player_stock = 0
#     enemy_stock = 0

#     if gamestate.players[1].stock < prev_gamestate.players[1].stock:
#         player_stock = (int(gamestate.players[1].stock) - int(prev_gamestate.players[1].stock))*3
#     if gamestate.players[2].stock < prev_gamestate.players[2].stock:
#         enemy_stock = -(int(gamestate.players[2].stock) - int(prev_gamestate.players[2].stock))*3

#     player_hp = 0
#     enemy_hp = 0

#     if(player_stock == 0):
#         if gamestate.players[1].percent > prev_gamestate.players[1].percent:
#             player_hp = -(float(gamestate.players[1].percent) - float(prev_gamestate.players[1].percent)) * 0.001
#     if(enemy_stock == 0):
#         if gamestate.players[2].percent > prev_gamestate.players[2].percent:
#             enemy_hp = (float(gamestate.players[2].percent) - float(prev_gamestate.players[2].percent)) * 0.001


#     reward += player_stock + enemy_stock + player_hp + enemy_hp

    
#     return reward

# def compute_reward(prev_gamestate, gamestate):
#     if gamestate is None or prev_gamestate is None:
#         return 0.0
#     # Example: reward based on stock difference
#     player_stock = 0
#     enemy_stock = 0

#     if gamestate.players[1].stock < prev_gamestate.players[1].stock:
        
#         player_stock = (int(gamestate.players[1].stock) - int(prev_gamestate.players[1].stock))*3
#         #print("DEATH", player_stock)
#     if gamestate.players[2].stock < prev_gamestate.players[2].stock:
#         enemy_stock = -(int(gamestate.players[2].stock) - int(prev_gamestate.players[2].stock))*3

#     player_hp = 0
#     enemy_hp = 0

#     if(player_stock == 0):
#         if gamestate.players[1].percent > prev_gamestate.players[1].percent:
#             player_hp = -(float(gamestate.players[1].percent) - float(prev_gamestate.players[1].percent)) * 0.001
#     if(enemy_stock == 0):
#         if gamestate.players[2].percent > prev_gamestate.players[2].percent:
#             enemy_hp = (float(gamestate.players[2].percent) - float(prev_gamestate.players[2].percent)) * 0.001


#     reward = player_stock + enemy_stock + player_hp + enemy_hp
    
#     # if reward != 0:
#     #     print("REWARD",reward)
#     return reward
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

def check_port(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 4:
        raise argparse.ArgumentTypeError(
            f"{value} is an invalid controller port. Must be 1, 2, 3, or 4."
        )
    return ivalue

def main():
    global num_down, num_left, num_right

    buffer   = ReplayBufferPPO(ROLLOUT_LEN, obs_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        #In the menus, select both CPUs and then autostart on the second
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

    agent = PPOAgentDiscrete(obs_dim, N_ACTIONS).to(device)


    # opt_actor  = actor = optim.Adam(
    #     list(agent.shared.parameters())
    #   + list(agent.button_logits.parameters())
    #   + list(agent.analog_mu.parameters())
    #   + [agent.analog_logstd],        # it’s a Parameter, so needs to go in manually
    #   lr=LR_ACTOR
    # )
    opt_actor = optim.Adam(list(agent.policy_head.parameters()), lr=LR_ACTOR)
    opt_critic = optim.Adam(list(agent.value_head.parameters()), lr=LR_CRITIC)

    save_dir = "PPO_hard_XL_everything"
    os.makedirs(save_dir, exist_ok=True)


    count = 0
    num_games = 0
    num_train = 50
    done = False
    action = None
    rollout_steps  = 0
    update_count   = 0
    prev_gamestate = None
    prev_state = None
    prev_action = None
    prev_action_idx = None
    prev_logp = None
    prev_value = None
    action_counts = [0, 0, 0]
    epsilon = 0.1
    while True:

        gamestate = console.step()
        if gamestate is None:
            continue
        #ac_count = [0, 0, 0]
        
        if gamestate is not None and gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            controller1.release_all()
            state = make_obs(gamestate)
            with torch.no_grad():
                # out = agent(state.unsqueeze(0))
                # logits_x = out['logits_x'].squeeze(0)  # [3]
                # logits_y = out['logits_y'].squeeze(0)  # [3]
                # dist_x = Categorical(logits=logits_x)
                # dist_y = Categorical(logits=logits_y)

                # idx_x = dist_x.sample()
                # idx_y = dist_y.sample()
                # choices = torch.tensor([-1.0, 0.0, 1.0], device=device)  # choices for joystick
                # action = torch.stack([choices[idx_x], choices[idx_y]])
                # logp = dist_x.log_prob(idx_x) + dist_y.log_prob(idx_y)
                # value = out['value'].item()

                out = agent(state.unsqueeze(0))
                logits = out['logits'].squeeze(0)  # [N_ACTIONS]
                dist = Categorical(logits=logits)
                action_idx = dist.sample()
                action = ACTIONS[action_idx]  # This is your move_input row
                logp = dist.log_prob(action_idx)
                value = out['value'].item()

                prev_state = state
                prev_action = action
                prev_action_idx = action_idx
                prev_logp = logp
                prev_value = value
                prev_gamestate = gamestate
            unpack_and_send(controller1, action)

            #unpack_and_send_c2(controller2)

            #controller1.release_all()

            # state = make_obs(gamestate)
            # with torch.no_grad():
            #     out = agent(state.unsqueeze(0))
            #     logits_x = out['logits_x'].squeeze(0)  # [3]
            #     logits_y = out['logits_y'].squeeze(0)  # [3]
            #     a_logit = out['a_logit'].squeeze(0)

            #     dist_x = Categorical(logits=logits_x)
            #     dist_y = Categorical(logits=logits_y)
            #     dist_a = Bernoulli(logits=a_logit)

            #     idx_x = dist_x.sample()  # 0, 1, or 2
            #     #action_counts[idx_x] += 1
            #     #print("Action counts:", action_counts)
            #     idx_y = dist_y.sample()
            #     a_sample = dist_a.sample()
            #     # Map indices to values
            #     choices = torch.tensor([-1.0, 0.0, 1.0], device=device)  # choices for joystick
            #     action = torch.stack([choices[idx_x], choices[idx_y], a_sample])
            #     logp = dist_x.log_prob(idx_x) + dist_y.log_prob(idx_y) + dist_a.log_prob(a_sample)
            #     value = out['value'].item()

            #     prev_state = state
            #     prev_action = action
            #     prev_logp = logp
            #     prev_value = value
            #     prev_gamestate = gamestate
            # unpack_and_send(controller1, action)
            

        elif gamestate is not None:
            melee.MenuHelper.skip_postgame(controller1,gamestate)
            melee.MenuHelper.skip_postgame(controller2,gamestate)
            continue

        gamestate = console.step()
        if gamestate is None:
            continue

        if gamestate is not None and gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            done = False
            count +=1
            state = make_obs(gamestate)
            reward = compute_reward(prev_gamestate,gamestate)
            #reward = compute_reward(gamestate)
            buffer.store(prev_state, prev_action_idx, prev_logp, reward, prev_value)
            rollout_steps += 1

            if rollout_steps >= ROLLOUT_LEN:
            # bootstrap last value
                last_val = 0.0
                with torch.no_grad():
                    last_val = agent(prev_state.unsqueeze(0))['value'].item()

                buffer.finish_path(last_val)   # compute GAE & returns
                actor_loss, critic_loss = ppo_update(agent, buffer, opt_actor, opt_critic,update_count)
                              # clear out old rollout
                rollout_steps = 0
                update_count += 1
                avg_reward = buffer.rews[:buffer.ptr].sum().item() if buffer.ptr > 0 else 0.0

                # Print to console
                print(f"[PPO] update {update_count}/{TOTAL_UPDATES}  "
                    f"actor_loss={actor_loss:.4f}  critic_loss={critic_loss:.4f}  avg_reward={avg_reward:.4f}")

                # Log to file
                with open("training_log_PPO.txt", "a") as f:
                    f.write(f"{update_count},{actor_loss:.6f},{critic_loss:.6f},{avg_reward:.6f}\n")
                count+=1

                buffer.reset()   
                torch.save(agent.state_dict(), os.path.join(save_dir, f"FINAL_PPO_simple_stay_alive_{update_count}.pth"))

                
            continue
            
        if gamestate is None:
            continue

        if gamestate is not None:
            if(done == False and count>0):
                #print("Game ended, pushing last state")
                if(buffer.ptr>0):
                    buffer.finish_path(last_val=-10.0)
                    rollout_steps += 1
                    #update_count += 1
                    #actor_loss, critic_loss = ppo_update(agent, buffer, opt_actor, opt_critic)

                    # avg_reward = buffer.rews[:buffer.ptr].mean().item() if buffer.ptr > 0 else 0.0

                    # # Print to console
                    # print(f"[PPO] update {update_count}/{TOTAL_UPDATES}  "
                    #     f"actor_loss={actor_loss:.4f}  critic_loss={critic_loss:.4f}  avg_reward={avg_reward:.4f}")

                    # # Log to file
                    # with open("training_log_PPO.txt", "a") as f:
                    #     f.write(f"{update_count},{actor_loss:.6f},{critic_loss:.6f},{avg_reward:.6f}\n")
                    
                    #buffer.reset()   
                
        done = True
        
        melee.MenuHelper.skip_postgame(controller1,gamestate)
        melee.MenuHelper.skip_postgame(controller2,gamestate)
if __name__ == "__main__":
    main()