from tqdm import tqdm
import melee
import os
import glob
import torch
import pickle
import json
import multiprocessing as mp

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
            p.hitstun_frames_left
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

def get_controller_state(controller_state):
    # get the controller state
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
