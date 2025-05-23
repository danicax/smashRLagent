from collections import deque
import torch
import random
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buf = deque(maxlen=capacity)
    def push(self, s,a,r,s_next,d):
        self.buf.append((s,a,r,s_next,d))
    def sample(self, batch_size=32):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        s = torch.stack(s)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float32)
        s2 = torch.stack(s2)
        d = torch.tensor(d, dtype=torch.float32)
        return s, a, r, s2, d
    def __len__(self):
        return len(self.buf)