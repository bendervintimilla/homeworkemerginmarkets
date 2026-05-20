"""Rollout buffer for PPO with Generalized Advantage Estimation (GAE).

PPO collects a batch of T env-steps from N parallel actors (here we keep N=1
to stay simple; vectorizing is a small extension). After collection we compute
advantages with GAE-lambda and returns, then train for K epochs over
mini-batches of the rollout.
"""

from __future__ import annotations

import numpy as np


class RolloutBuffer:
    def __init__(self, size: int, obs_shape: tuple[int, ...], gamma: float,
                 gae_lambda: float):
        self.size = size
        self.gamma = gamma
        self.lam = gae_lambda
        self.obs = np.zeros((size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.logp = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, logp, reward, value, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.logp[i] = logp
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = float(done)
        self.ptr += 1

    def compute_gae(self, last_value: float):
        """GAE-lambda. last_value is V(s_T) for bootstrapping the final step."""
        adv = 0.0
        for t in reversed(range(self.size)):
            next_value = last_value if t == self.size - 1 else self.values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_value * next_nonterminal \
                    - self.values[t]
            adv = delta + self.gamma * self.lam * next_nonterminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def get(self):
        return {
            "obs": self.obs,
            "actions": self.actions,
            "logp": self.logp,
            "advantages": self.advantages,
            "returns": self.returns,
            "values": self.values,
        }

    def reset(self):
        self.ptr = 0
