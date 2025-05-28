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
    


class ReplayBufferPPO:
    def __init__(self, capacity, obs_dim, action_dim):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs     = torch.zeros(capacity, obs_dim, device=device)
        self.actions = torch.zeros(capacity, action_dim, device=device)
        self.logps   = torch.zeros(capacity, device=device)
        self.rews    = torch.zeros(capacity, device=device)
        self.advs    = torch.zeros(capacity, device=device)
        self.returns = torch.zeros(capacity, device=device)
        self.vals    = torch.zeros(capacity, device=device)
        self.ptr = 0
        self.capacity = capacity

    def store(self, obs, action, logp, rew, val):
        idx = self.ptr
        self.obs[idx]     = obs
        self.actions[idx] = action
        self.logps[idx]   = logp
        self.rews[idx]    = rew
        self.vals[idx]    = val
        self.ptr += 1

    def finish_path(self, last_val=0, GAMMA=0.99, LAMBDA=0.95):
        # append last_val for bootstrap
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rews = torch.cat([self.rews[:self.ptr], torch.tensor([last_val], device=device)])
        vals = torch.cat([self.vals[:self.ptr], torch.tensor([last_val], device=device)])
        gae = 0
        for t in reversed(range(self.ptr)):
            delta = rews[t] + GAMMA * vals[t+1] - vals[t]
            gae = delta + GAMMA * LAMBDA * gae
            self.advs[t]    = gae
            self.returns[t] = gae + vals[t]

    def get(self):
        # normalize advantages
        advs = self.advs[:self.ptr]
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        return (
            self.obs[:self.ptr],
            self.actions[:self.ptr],
            self.logps[:self.ptr],
            self.returns[:self.ptr],
            advs
        )
    def reset(self):
        """Clear the buffer for the next rollout."""
        self.ptr = 0
        self.obs.zero_()
        self.actions.zero_()
        self.logps.zero_()
        self.rews.zero_()
        self.vals.zero_()
