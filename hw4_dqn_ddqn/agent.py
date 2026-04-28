"""DQN and Double-DQN agents implemented with Keras.

Key ideas (Mnih et al. 2015 + van Hasselt et al. 2016):

* Replay buffer + uniformly sampled mini-batches break the temporal correlation
  in transitions and reuse data, making the supervised regression at the core
  of Q-learning stable.
* Target network. We keep a slow-moving copy of the online network to compute
  the bootstrap target r + gamma * max_a Q_target(s', a). Without this the
  target moves with the parameters and training diverges.
* Double-DQN. The vanilla max in DQN systematically overestimates Q because
  noisy estimates skew the max upward. DDQN selects the next action with the
  online net and evaluates it with the target net:
        target = r + gamma * Q_target(s', argmax_a Q_online(s', a))
  This decouples action SELECTION from action EVALUATION and reduces the bias.

The two only differ in `_compute_target`. Same buffer, same loss, same loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Keras import is lazy so the rest of the repo doesn't pay the TF import cost.
def _build_keras_mlp(obs_dim: int, n_actions: int, hidden: tuple[int, ...]):
    import tensorflow as tf
    from tensorflow import keras

    # Keras 3 is strict about plain Python int for `units` (np.int64 fails).
    obs_dim = int(obs_dim)
    n_actions = int(n_actions)
    hidden = tuple(int(h) for h in hidden)

    model = keras.Sequential(name="q_network")
    model.add(keras.layers.Input(shape=(obs_dim,)))
    for h in hidden:
        model.add(keras.layers.Dense(h, activation="relu"))
    model.add(keras.layers.Dense(n_actions, activation="linear"))
    return model


@dataclass
class DQNConfig:
    obs_dim: int
    n_actions: int
    hidden: tuple[int, ...] = (128, 128)
    lr: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = 100_000
    batch_size: int = 64
    learning_starts: int = 1_000
    target_update_freq: int = 500   # gradient steps
    train_freq: int = 4              # env steps between train batches
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    grad_clip: float = 10.0
    huber_delta: float = 1.0
    double: bool = False             # set True for DDQN
    seed: int = 0


class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        # Defer TF imports to here so importing this module is light.
        import tensorflow as tf
        from tensorflow import keras

        self.cfg = cfg
        self.tf = tf
        self.keras = keras
        self.rng = np.random.default_rng(cfg.seed)

        self.online = _build_keras_mlp(cfg.obs_dim, cfg.n_actions, cfg.hidden)
        self.target = _build_keras_mlp(cfg.obs_dim, cfg.n_actions, cfg.hidden)
        self.target.set_weights(self.online.get_weights())

        self.optimizer = keras.optimizers.Adam(learning_rate=cfg.lr,
                                               clipnorm=cfg.grad_clip)
        self.loss_fn = keras.losses.Huber(delta=cfg.huber_delta)

        self._train_steps = 0  # number of gradient updates done

    # ---- behaviour ----
    def epsilon(self, env_step: int) -> float:
        frac = min(1.0, env_step / max(1, self.cfg.eps_decay_steps))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def act(self, obs: np.ndarray, eps: float) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(self.cfg.n_actions))
        # Single-state inference; cast to float32 for TF speed.
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        q = self.online(x, training=False).numpy().ravel()
        return int(np.argmax(q))

    # ---- learning ----
    def _compute_target(self, rewards, next_states, dones):
        if self.cfg.double:
            # Online net picks the action, target net evaluates it.
            next_q_online = self.online(next_states, training=False).numpy()
            next_actions = np.argmax(next_q_online, axis=1)
            next_q_target = self.target(next_states, training=False).numpy()
            next_value = next_q_target[np.arange(len(next_actions)), next_actions]
        else:
            next_q_target = self.target(next_states, training=False).numpy()
            next_value = next_q_target.max(axis=1)
        return rewards + self.cfg.gamma * (1.0 - dones) * next_value

    def train_step(self, batch) -> float:
        states, actions, rewards, next_states, dones = batch
        states = states.astype(np.float32)
        next_states = next_states.astype(np.float32)
        targets = self._compute_target(rewards, next_states, dones).astype(np.float32)

        with self.tf.GradientTape() as tape:
            q_all = self.online(states, training=True)
            # Gather Q for the action actually taken.
            idx = self.tf.stack(
                [self.tf.range(self.tf.shape(q_all)[0]), self.tf.cast(actions, self.tf.int32)],
                axis=1,
            )
            q_taken = self.tf.gather_nd(q_all, idx)
            loss = self.loss_fn(targets, q_taken)
        grads = tape.gradient(loss, self.online.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.online.trainable_variables))

        self._train_steps += 1
        if self._train_steps % self.cfg.target_update_freq == 0:
            self.target.set_weights(self.online.get_weights())
        return float(loss.numpy())
