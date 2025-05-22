from collections import deque
import torch
import random
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buf = deque(maxlen=capacity)
    def push(self, s,a,r,s_next,d):
        self.buf.append((s,a,r,s_next,d))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s,a,r,s2,d = map(lambda x: torch.stack(x), zip(*batch))
        return s,a,r,s2,d
    def __len__(self):
        return len(self.buf)