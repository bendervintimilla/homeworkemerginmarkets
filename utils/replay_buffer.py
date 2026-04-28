from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """Uniform replay buffer used by HW4 (DQN/DDQN).

    Stores transitions as numpy arrays so we can vectorize the sample step
    instead of looping in Python.
    """

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], dtype=np.float32):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.dtype = dtype
        self.size = 0
        self.idx = 0

        self.states = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, s, a, r, s_next, done):
        self.states[self.idx] = s
        self.next_states[self.idx] = s_next
        self.actions[self.idx] = a
        self.rewards[self.idx] = r
        self.dones[self.idx] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size
