"""Actor-critic networks (Keras) for PPO.

Two heads share no weights – this is simpler to debug and works well for the
state-vector tasks targeted in the course (LunarLander, Walker2D). For pixel
inputs you'd typically share a CNN trunk; not needed here.
"""

from __future__ import annotations

import numpy as np


def build_actor(obs_dim: int, n_actions: int, hidden=(64, 64)):
    """Discrete categorical policy: outputs logits over n_actions."""
    from tensorflow import keras

    obs_dim = int(obs_dim)
    n_actions = int(n_actions)
    hidden = tuple(int(h) for h in hidden)

    inp = keras.layers.Input(shape=(obs_dim,))
    x = inp
    for h in hidden:
        x = keras.layers.Dense(
            h, activation="tanh",
            kernel_initializer=keras.initializers.Orthogonal(np.sqrt(2.0)),
        )(x)
    logits = keras.layers.Dense(
        n_actions,
        kernel_initializer=keras.initializers.Orthogonal(0.01),
    )(x)
    return keras.Model(inp, logits, name="actor")


def build_critic(obs_dim: int, hidden=(64, 64)):
    from tensorflow import keras

    obs_dim = int(obs_dim)
    hidden = tuple(int(h) for h in hidden)

    inp = keras.layers.Input(shape=(obs_dim,))
    x = inp
    for h in hidden:
        x = keras.layers.Dense(
            h, activation="tanh",
            kernel_initializer=keras.initializers.Orthogonal(np.sqrt(2.0)),
        )(x)
    v = keras.layers.Dense(
        1, kernel_initializer=keras.initializers.Orthogonal(1.0),
    )(x)
    return keras.Model(inp, v, name="critic")
