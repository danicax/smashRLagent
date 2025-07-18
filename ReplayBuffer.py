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
        # 1) slice out exactly what we stored
        obs_b  = self.obs[:self.ptr]
        act_b = self.actions[:self.ptr]
        logp_b        = self.logps[:self.ptr]
        ret_b         = self.returns[:self.ptr]
        adv_b         = self.advs[:self.ptr]

        # 2) normalize advantages with population‐std (unbiased=False)
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)

        # detach so that nothing here tracks gradients
        return (obs_b.clone(), act_b.clone(),
                logp_b.clone(), ret_b.clone(),
                adv_b.clone())
    
    def reset(self):
        """Clear the buffer for the next rollout."""
        self.ptr = 0
        for b in (self.obs, self.actions, self.logps,
                    self.rews, self.vals, self.advs, self.returns):
            b.zero_()

class ReplayBufferTransformer:
    def __init__(self, capacity=100_000):
        self.capacity = capacity
        self.buf = []  # will store tuples (state_seq, action, reward, next_state_seq, done)
        self.position = 0

    def push(self, state_seq, action, reward, next_state_seq, done):
        """state_seq: Tensor[SEQ_LEN, obs_dim], next_state_seq: same shape."""
        if len(self.buf) < self.capacity:
            self.buf.append(None)
        self.buf[self.position] = (state_seq.clone(),
                                   action,
                                   reward,
                                   next_state_seq.clone(),
                                   done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Returns:
          state_seqs: Tensor [B, SEQ_LEN, obs_dim]
          actions:    LongTensor [B]
          rewards:    FloatTensor [B]
          next_seqs:  Tensor [B, SEQ_LEN, obs_dim]
          dones:      FloatTensor [B] (0 or 1)
        """
        batch = random.sample(self.buf, batch_size)
        state_seqs, actions, rewards, next_seqs, dones = zip(*batch)

        # Stack along new batch dimension:
        state_seqs = torch.stack(state_seqs, dim=0)    # [B, SEQ_LEN, obs_dim]
        next_seqs  = torch.stack(next_seqs,  dim=0)    # [B, SEQ_LEN, obs_dim]
        actions    = torch.tensor(actions, dtype=torch.long)   # [B]
        rewards    = torch.tensor(rewards, dtype=torch.float32) # [B]
        dones      = torch.tensor(dones,   dtype=torch.float32) # [B]
        return state_seqs, actions, rewards, next_seqs, dones

    def __len__(self):
        return len(self.buf)
    
class ReplayBufferPPODiscrete:
    def __init__(self, capacity, obs_dim):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs     = torch.zeros(capacity, obs_dim, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)  # store action indices
        self.logps   = torch.zeros(capacity, device=device)
        self.rews    = torch.zeros(capacity, device=device)
        self.advs    = torch.zeros(capacity, device=device)
        self.returns = torch.zeros(capacity, device=device)
        self.vals    = torch.zeros(capacity, device=device)
        self.ptr = 0
        self.capacity = capacity

    def store(self, obs, action_idx, logp, rew, val):
        idx = self.ptr
        self.obs[idx]     = obs
        self.actions[idx] = action_idx  # integer index
        self.logps[idx]   = logp
        self.rews[idx]    = rew
        self.vals[idx]    = val
        self.ptr += 1

    def finish_path(self, last_val=0, GAMMA=0.99, LAMBDA=0.95):
        device = self.obs.device
        rews = torch.cat([self.rews[:self.ptr], torch.tensor([last_val], device=device)])
        vals = torch.cat([self.vals[:self.ptr], torch.tensor([last_val], device=device)])
        gae = 0
        for t in reversed(range(self.ptr)):
            delta = rews[t] + GAMMA * vals[t+1] - vals[t]
            gae = delta + GAMMA * LAMBDA * gae
            self.advs[t]    = gae
            self.returns[t] = gae + vals[t]

    def get(self):
        obs_b  = self.obs[:self.ptr]
        act_b  = self.actions[:self.ptr]
        logp_b = self.logps[:self.ptr]
        ret_b  = self.returns[:self.ptr]
        adv_b  = self.advs[:self.ptr]
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)
        return (obs_b.clone(), act_b.clone(), logp_b.clone(), ret_b.clone(), adv_b.clone())

    def reset(self):
        self.ptr = 0
        for b in (self.obs, self.actions, self.logps, self.rews, self.vals, self.advs, self.returns):
            b.zero_()