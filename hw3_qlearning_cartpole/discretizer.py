"""State discretization for tabular Q-Learning on continuous CartPole.

CartPole-v1 returns 4 continuous observations:
    [cart position, cart velocity, pole angle, pole angular velocity]

Bounds for cart position and pole angle are bounded by the env's done condition
(|x|>2.4, |theta|>12 deg). Velocities are theoretically unbounded so we clip
to a reasonable range. The angle and angular velocity are the most informative
features so we give them more bins.
"""

from __future__ import annotations

import numpy as np


# Manually chosen clipping bounds. The env terminates outside [-2.4, 2.4] for x
# and outside ~[-0.21, 0.21] rad for theta (12 degrees), so picking slightly
# inside those is fine. Velocities clipped to ranges seen in practice.
LOW = np.array([-2.4, -3.0, -0.21, -3.5], dtype=np.float64)
HIGH = np.array([2.4, 3.0, 0.21, 3.5], dtype=np.float64)


class CartPoleDiscretizer:
    """Maps a continuous 4-d CartPole state to a flat integer state id."""

    def __init__(self, bins: tuple[int, int, int, int] = (3, 3, 12, 12)):
        self.bins = np.array(bins, dtype=np.int64)
        if (self.bins < 1).any():
            raise ValueError("All bin counts must be >= 1")
        # Pre-compute bin edges (without the outer endpoints – we'll use np.digitize).
        self._edges = [
            np.linspace(LOW[i], HIGH[i], self.bins[i] + 1)[1:-1]
            for i in range(4)
        ]
        self.n_states = int(np.prod(self.bins))

    def __call__(self, obs: np.ndarray) -> int:
        # digitize gives a value in [0, bins[i]) for each dim once we pass the
        # interior edges. Clip to [0, bins-1] to absorb extreme values safely.
        idxs = np.empty(4, dtype=np.int64)
        for i in range(4):
            idxs[i] = np.digitize(obs[i], self._edges[i])
            if idxs[i] >= self.bins[i]:
                idxs[i] = self.bins[i] - 1
            elif idxs[i] < 0:
                idxs[i] = 0
        # Flatten with row-major order.
        flat = 0
        for i in range(4):
            flat = flat * int(self.bins[i]) + int(idxs[i])
        return flat
