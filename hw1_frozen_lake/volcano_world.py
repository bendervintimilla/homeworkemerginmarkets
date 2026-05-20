"""Volcano environment from the course repo (castorgit/RL_course/volcano_rl).

Source: https://github.com/castorgit/RL_course/tree/main/volcano_rl
Vendored here so HW1 is self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from gymnasium import Env, spaces
    from gymnasium.envs.registration import register
    from gymnasium.utils import seeding
except ImportError:
    Env = object
    register = None

    class _Discrete:
        def __init__(self, n: int):
            self.n = n

        def contains(self, x: object) -> bool:
            return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    class _Spaces:
        Discrete = _Discrete

    class _Seeding:
        @staticmethod
        def np_random(seed: Optional[int] = None):
            return np.random.default_rng(seed), seed

    spaces = _Spaces()
    seeding = _Seeding()


NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

ACTION_TO_STR = {NORTH: "N", EAST: "E", SOUTH: "S", WEST: "W"}


@dataclass(frozen=True)
class Cell:
    symbol: str
    reward: float = 0.0
    terminal: bool = False


DEFAULT_MAP = (
    (".", ".", "L", "V"),
    ("S", ".", "L", "."),
    ("G", ".", ".", "."),
)
START_POS = (1, 0)

CELL_TYPES = {
    "S": Cell("S"),
    ".": Cell("."),
    "L": Cell("L", reward=-50.0, terminal=True),
    "V": Cell("V", reward=20.0, terminal=True),
    "G": Cell("G", reward=2.0, terminal=True),
}


class VolcanoWorldEnv(Env):
    """3x4 GridWorld inspired by FrozenLake.

    Layout:
        . . L V
        S . L .
        G . . .

    Terminal cells:
        Lava L: -50 reward
        View V: +20 reward (the risky goal)
        Safe G: +2 reward (the conservative goal)

    Actions: 0=N, 1=E, 2=S, 3=W. Slippery dynamics by default.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        is_slippery: bool = True,
        slip_probabilities: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    ):
        self.render_mode = render_mode
        self.is_slippery = is_slippery
        self.slip_probabilities = self._validate_slip_probabilities(slip_probabilities)

        self.desc = np.asarray(DEFAULT_MAP, dtype="U1")
        self.nrow, self.ncol = self.desc.shape
        self.nS = self.nrow * self.ncol
        self.nA = 4

        self.initial_state_distrib = np.zeros(self.nS)
        self.initial_state_distrib[self.to_s(*START_POS)] = 1.0

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        self.P = {state: {action: [] for action in range(self.nA)}
                  for state in range(self.nS)}
        self._build_transitions()

        self.np_random = None
        self.s = self.to_s(*START_POS)
        self.lastaction = None

    @staticmethod
    def _validate_slip_probabilities(slip_probabilities):
        if len(slip_probabilities) != 3:
            raise ValueError("slip_probabilities must have three values")
        total = sum(slip_probabilities)
        if not np.isclose(total, 1.0):
            raise ValueError("slip_probabilities must sum to 1.0")
        if any(p < 0 for p in slip_probabilities):
            raise ValueError("slip_probabilities cannot be negative")
        return slip_probabilities

    def to_s(self, row, col):
        return row * self.ncol + col

    def from_s(self, state):
        return divmod(state, self.ncol)

    def _move(self, row, col, action):
        if action == NORTH:
            row = max(row - 1, 0)
        elif action == EAST:
            col = min(col + 1, self.ncol - 1)
        elif action == SOUTH:
            row = min(row + 1, self.nrow - 1)
        elif action == WEST:
            col = max(col - 1, 0)
        else:
            raise ValueError(f"Unknown action: {action}")
        return row, col

    def _cell(self, row, col):
        return CELL_TYPES[self.desc[row, col]]

    def _transition_candidates(self, action):
        if not self.is_slippery:
            return ((action, 1.0),)
        return (
            ((action - 1) % self.nA, self.slip_probabilities[0]),
            (action, self.slip_probabilities[1]),
            ((action + 1) % self.nA, self.slip_probabilities[2]),
        )

    def _build_transitions(self):
        for row in range(self.nrow):
            for col in range(self.ncol):
                state = self.to_s(row, col)
                current_cell = self._cell(row, col)
                for action in range(self.nA):
                    transitions = self.P[state][action]
                    if current_cell.terminal:
                        transitions.append((1.0, state, 0.0, True))
                        continue
                    for actual_action, prob in self._transition_candidates(action):
                        next_row, next_col = self._move(row, col, actual_action)
                        next_state = self.to_s(next_row, next_col)
                        next_cell = self._cell(next_row, next_col)
                        transitions.append(
                            (prob, next_state, next_cell.reward, next_cell.terminal)
                        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        del options
        self.np_random, _ = seeding.np_random(seed)
        self.s = int(self.np_random.choice(self.nS, p=self.initial_state_distrib))
        self.lastaction = None
        return int(self.s), {"prob": 1.0}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        transitions = self.P[self.s][action]
        probs = [t[0] for t in transitions]
        idx = int(self.np_random.choice(len(transitions), p=probs))
        prob, next_state, reward, terminated = transitions[idx]
        self.s = next_state
        self.lastaction = action
        return int(next_state), float(reward), bool(terminated), False, {"prob": prob}

    def render(self):
        output = StringIO()
        row, col = self.from_s(self.s)
        desc = self.desc.tolist()
        desc[row][col] = f"[{desc[row][col]}]"
        output.write("\n".join(" ".join(cell for cell in line) for line in desc))
        if self.lastaction is not None:
            output.write(f"\nLast action: {ACTION_TO_STR[self.lastaction]}")
        output.write("\n")
        return output.getvalue()


def register_env() -> None:
    if register is None:
        raise ImportError("gymnasium is required to register Volcano")
    register(
        id="VolcanoWorld-v0",
        entry_point="hw1_frozen_lake.volcano_world:VolcanoWorldEnv",
    )
