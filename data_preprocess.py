from tqdm import tqdm
import melee
import os
import glob
import torch
import pickle
import json
import multiprocessing as mp
from util import make_obs, get_controller_state

def process_slp(slp, output_path):
    states = []
    actions = []
    game_id = os.path.basename(slp).split(".")[0]
    # console = melee.Console(system="file", allow_old_version=False, path=slp)
    console = melee.Console(system="file", path=slp)
    console.connect()
    try:
        while True:
            gs = console.step()
            if gs is None:
                break

            controller_state = gs.players[1].controller_state  # or whichever field holds the input
            act = get_controller_state(controller_state)

            obs = make_obs(gs, max_projectiles=5)
            states.append(obs)
            actions.append(act)
    except Exception as e:
        print(f"Error processing {game_id}: {e}")
        return 0

    # store the states and actions in a pkl file
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f"{game_id}.pkl"), "wb") as f:
        pickle.dump({"states": states, "actions": actions}, f)

    return len(actions)

# REPLACE WITH YOUR PATH NAME
# source_path = "replays"
source_path = "data"
# find all slp files in the data folder recursively
slp_paths = glob.glob(os.path.join(source_path, "**/*.slp"), recursive=True)
# slp_paths = glob.glob(os.path.join(source_path, "2025-05/*.slp"), recursive=True)
print(len(slp_paths))

# output_path = "data/processed"
num_train = 449
num_train = 50
num_val = 0

train_output_path = f"data/train_mini_{num_train}"
val_output_path = f"data/val_mini_{num_val}"

def make_train(x): return process_slp(x, train_output_path)
def make_val(x): return process_slp(x, val_output_path)

# use multiprocessing to speed up the process
print(f'working with {mp.cpu_count()} cores')

if __name__ == "__main__":
    with mp.Pool(processes=mp.cpu_count()) as pool:
        train = pool.map(make_train, slp_paths[:num_train])
        val = pool.map(make_val, slp_paths[num_train:num_train + num_val])

    metadata = {}
    for k, v in zip(slp_paths, train):
        metadata[os.path.basename(k)] = v

    with open(os.path.join(train_output_path, "metadata.json"), "w+") as f:
        json.dump(metadata, f)

    metadata = {}
    for k, v in zip(slp_paths[50:60], val):
        metadata[os.path.basename(k)] = v

    with open(os.path.join(val_output_path, "metadata.json"), "w+") as f:
        json.dump(metadata, f)
