from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import os, json, pickle


class MeleeDataset(Dataset):
    def __init__(self, data_path, cache_size=None):
        """
        Args:
            data_path (str): the path to the data
            cache_size (int): How many games to cache in the dataset. If None, all games will be cached.
        """
        self.data_path = data_path
        self.cache_size = cache_size

        # load the metadata
        with open(os.path.join(data_path, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

        # calculate the total length of the dataset
        self.length = 0
        for k, v in self.metadata.items():
            self.length += v

        # initialize the states and actions. These are LRU caches.
        self.states = OrderedDict()
        self.actions = OrderedDict()

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # find the game file that contains the idx-th state, action pair
        for k, v in self.metadata.items():
            if idx < v:
                curr_game_file = k
                break
            idx -= v

        # if the game file is not in the cache, load it
        if curr_game_file not in self.states:
            with open(os.path.join(self.data_path, curr_game_file.split(".")[0] + ".pkl"), "rb") as f:
                data = pickle.load(f)
            self.states[curr_game_file] = data["states"]
            self.actions[curr_game_file] = data["actions"]
            print(f"Loaded {curr_game_file} with {len(self.states[curr_game_file])} states and {len(self.actions[curr_game_file])} actions. The cache size is {len(self.states)}.")
            # if the cache is full, remove the least recently used game file
            if self.cache_size is not None and len(self.states) > self.cache_size:
                self.states.popitem(last=False)
                self.actions.popitem(last=False)

        state = self.states[curr_game_file][idx]
        action = self.actions[curr_game_file][idx]

        # move the game file to the end of the cache
        self.states.move_to_end(curr_game_file)
        self.actions.move_to_end(curr_game_file)

        return state, action
    
    
    
    