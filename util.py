from tqdm import tqdm
import melee
import os
import glob
import torch
import pickle
import json
import multiprocessing as mp
import sys

def make_obs(gamestate, max_projectiles=5):
    # 1) Player features extractor (17 floats each)
    def player_feats(p):
        return [
            float(p.stock),
            float(p.percent),
            p.position.x,
            p.position.y,
            float(p.character.value),
            float(p.action.value),
            float(p.action_frame),
            float(p.facing),
            float(p.shield_strength),
            float(p.jumps_left),
            float(p.on_ground),
            float(p.invulnerable),
            p.speed_air_x_self,
            p.speed_ground_x_self,
            p.speed_x_attack,
            p.speed_y_attack,
            p.speed_y_self,
        ]

    # get player slots 1 & 2
    ps = gamestate.players
    f1 = player_feats(ps.get(1)) if 1 in ps else [0.0]*17
    f2 = player_feats(ps.get(2)) if 2 in ps else [0.0]*17 #we dont want player2 data right?
    #f2 = [float(gamestate.players.get(2).character.value)]

    # 2) Stage as one float
    stage_feat = [float(gamestate.stage.value)]

    # 3) Projectile features: each is 7 floats
    proj_feats = []
    for proj in gamestate.projectiles[:max_projectiles]:
        proj_feats.extend([
            float(proj.type.value),
            float(proj.frame),
            float(proj.owner),
            proj.position.x,
            proj.position.y,
            proj.speed.x,
            proj.speed.y,
        ])
    # pad out to max_projectiles * 7
    needed = max_projectiles*7 - len(proj_feats)
    if needed > 0:
        proj_feats.extend([0.0]*needed)

    all_feats = f1 + f2 + stage_feat + proj_feats
    return torch.tensor(all_feats, dtype=torch.float32)

# def make_obs_simple(gamestate):
#     # 1) Player features extractor (17 floats each)
#     def player_feats(p):
#         return [
#             # float(p.stock),
#             # float(p.percent),
#             p.position.x,
#             p.position.y,
#             # float(p.character.value),
#             # float(p.action.value),
#             # float(p.action_frame),
#             # float(p.facing),
#             # float(p.shield_strength),
#             # float(p.jumps_left),
#             # float(p.on_ground),
#             # float(p.invulnerable),
#             # p.speed_air_x_self,
#             # p.speed_ground_x_self,
#             # p.speed_x_attack,
#             # p.speed_y_attack,
#             # p.speed_y_self,
#         ]

#     # get player slots 1 & 2
#     ps = gamestate.players
#     f1 = player_feats(ps.get(1)) if 1 in ps else [0.0]*2
#     f2 = player_feats(ps.get(2)) if 2 in ps else [0.0]*2 #we dont want player2 data right?
#     #f2 = [float(gamestate.players.get(2).character.value)]

#     # 2) Stage as one float
#     stage_feat = [float(gamestate.stage.value)]


#     all_feats = f1 + f2 + stage_feat
#     return torch.tensor(all_feats, dtype=torch.float32)


def make_obs_simple(gamestate):
    # 1) Player features extractor (17 floats each)
    def player_feats(p):
        return [
            #float(p.stock),
            #float(p.percent),
            p.position.x,
            p.position.y,
            # float(p.character.value),
            # float(p.action.value),
            # float(p.action_frame),
            # float(p.facing),
            # float(p.shield_strength),
            # float(p.jumps_left),
            # float(p.on_ground),
            # float(p.invulnerable),
            # p.speed_air_x_self,
            # p.speed_ground_x_self,
            # p.speed_x_attack,
            # p.speed_y_attack,
            # p.speed_y_self,
        ]

    # get player slots 1 & 2
    ps = gamestate.players
    f1 = player_feats(ps.get(1)) if 1 in ps else [0.0]*2
    f2 = player_feats(ps.get(2)) if 2 in ps else [0.0]*2 #we dont want player2 data right?
    #f2 = [float(gamestate.players.get(2).character.value)]

    # 2) Stage as one float
    #stage_feat = [float(gamestate.stage.value)]


    all_feats = f1 + f2# + stage_feat
    return torch.tensor(all_feats, dtype=torch.float32)

def get_controller_state(controller_state):
    # get the controller state
    
    print("hi", controller_state.button)
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
    return act


def connect_to_console(args):
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
    
    console.run(iso_path=args.iso)

    # print("Connecting to console…")
    if not console.connect():
        print("ERROR: Failed to connect to the console.")
        sys.exit(-1)
    # print("Console connected")

    # Plug in BOTH controllers
    # print(f"Connecting controller on port {args.port1}…")
    if not controller1.connect():
        print("ERROR: Failed to connect controller1.")
        sys.exit(-1)
    # print("Controller1 connected")

    # print(f"Connecting controller on port {args.port2}…")
    if not controller2.connect():
        print("ERROR: Failed to connect controller2.")
        sys.exit(-1)
    # print("Controller2 connected")

    return console, controller1, controller2


# def unpack_and_send_simple(controller, action_tensor):
#     """
#     action_tensor: FloatTensor of shape [2] for main stick [x, y]
#     """
#     controller.release_all()
#     main_x, main_y = action_tensor[0].item(), action_tensor[1].item()
    

#     print(main_x, main_y)
#     # normalize the main stick
#     if 0.25 < main_y < 0.75: 
#         main_y = 0.5

#     controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, main_x, main_y)

def unpack_and_send_simple(controller, action_tensor):
    """
    action_tensor: FloatTensor of shape [2] for main stick [x, y]
    """
    print(action_tensor)
    controller.release_all()
    main_x, button_x = action_tensor[0].item(), action_tensor[1].item()

    if button_x >= 0.5:
        controller.press_button(melee.enums.Button.BUTTON_X)

    # print(main_x, main_y)
    # normalize the main stick

    controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, main_x, 0.5)


def unpack_and_send(controller, action_tensor):
    """
    action_tensor: FloatTensor of shape [act_dim] in the same order you trained on:
      [A, B, D_DOWN, D_LEFT, D_RIGHT, D_UP,
       L, R, X, Y, Z, START,
       main_x, main_y, c_x, c_y, raw_x, raw_y, l_shldr, r_shldr]
    """
    # First, clear last frame’s inputs
    #controller.release_all()

    # Booleans
    # print("ACTION",action_tensor)
    btns = [
        melee.enums.Button.BUTTON_A, melee.enums.Button.BUTTON_B, melee.enums.Button.BUTTON_D_DOWN,
        melee.enums.Button.BUTTON_D_LEFT, melee.enums.Button.BUTTON_D_RIGHT,melee.enums. Button.BUTTON_D_UP,
        melee.enums.Button.BUTTON_L, melee.enums.Button.BUTTON_R, melee.enums.Button.BUTTON_X,
        melee.enums.Button.BUTTON_Y, melee.enums.Button.BUTTON_Z #, melee.enums.Button.BUTTON_START
    ]

    #Analog sticks
    main_x, main_y = action_tensor[11].item(), action_tensor[12].item()
    c_x,    c_y    = action_tensor[13].item(), action_tensor[14].item()
    l_shoulder,    r_shoulder    = action_tensor[15].item(), action_tensor[16].item()

    controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, main_x, main_y)
    controller.tilt_analog(melee.enums.Button.BUTTON_C,    c_x,    c_y)
    controller.press_shoulder(melee.enums.Button.BUTTON_L, l_shoulder)
    controller.press_shoulder(melee.enums.Button.BUTTON_R, r_shoulder)
    
    for i, b in enumerate(btns):
        if action_tensor[i].item() >0.5:
            controller.press_button(b)


# MIN DIST REWARD
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

#     return reward

# social distancing reward
# def compute_reward(gamestate):
#     if gamestate is None:
#         return 0.0
    
#     if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
#         return 0.0
    
#     p1 = gamestate.players[1]
#     p2 = gamestate.players[2]
    
#     dx = float(p1.position.x) - float(p2.position.x)
#     dy = float(p1.position.y) - float(p2.position.y)
#     dist = (dx ** 2 + dy ** 2) ** 0.5 / 100

#     stocks = float(p1.stock)

#     return dist + stocks
    
# hate the void reward
# def compute_reward(gamestate):
#     if gamestate is None:
#         return 0.0
    
#     if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
#         return 0.0
    
#     p1 = gamestate.players[1]
#     p2 = gamestate.players[2]
    
#     dx = float(p1.position.x) - float(p2.position.x)
#     dy = float(p1.position.y) - float(p2.position.y)
#     dist = (dx ** 2 + dy ** 2) ** 0.5 / 100
    
#     if p1.off_stage:
#         return -100

#     return dist

# stay alive reward
def compute_reward(prev_gamestate, gamestate):
    if gamestate is None or prev_gamestate is None:
        return 0.0
    
    if gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH] or prev_gamestate.menu_state not in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        return 0.0
    
    p1 = gamestate.players[1]
    
    if p1.off_stage:
        return -100
    
    if gamestate.players[1].percent > prev_gamestate.players[1].percent:
        return -(gamestate.players[1].percent - prev_gamestate.players[1].percent)

    return 0




def menu_helper(gamestate, controller1, controller2):
    # In the menus, select both CPUs and then autostart on the second
    melee.MenuHelper.menu_helper_simple(
        gamestate,
        controller1,
        melee.Character.FOX,
        melee.Stage.BATTLEFIELD,
        connect_code='',
        cpu_level=0,
        costume=0,
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
        costume=0,
        autostart=True,
        swag=False
    )