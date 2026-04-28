"""PPO with clipping (Schulman et al. 2017).

For each rollout of T steps:
  1. Compute advantages A_t with GAE-lambda and returns R_t = A_t + V_phi(s_t).
  2. For K epochs over the rollout, sample mini-batches and minimise the
     PPO-clip surrogate together with a value-function MSE and an entropy bonus:

        L = - E[ min(rho_t A_t,  clip(rho_t, 1-eps, 1+eps) A_t) ]
            + c_v * E[(V_phi(s_t) - R_t)^2]
            - c_h * E[ H(pi(.|s_t)) ]

     where rho_t = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t).

We normalize advantages per mini-batch (a common practical trick) and clip
gradients globally. The discrete-action version uses a categorical policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .networks import build_actor, build_critic


@dataclass
class PPOConfig:
    obs_dim: int
    n_actions: int
    rollout_steps: int = 2048
    epochs: int = 10
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02   # early-stop policy update if exceeded
    seed: int = 0


class PPOAgent:
    def __init__(self, cfg: PPOConfig):
        import tensorflow as tf
        from tensorflow import keras

        self.cfg = cfg
        self.tf = tf
        self.keras = keras
        self.rng = np.random.default_rng(cfg.seed)

        self.actor = build_actor(cfg.obs_dim, cfg.n_actions)
        self.critic = build_critic(cfg.obs_dim)

        # Single optimizer applied to the union of params – cleaner than two.
        self.opt_actor = keras.optimizers.Adam(learning_rate=cfg.lr,
                                               clipnorm=cfg.max_grad_norm)
        self.opt_critic = keras.optimizers.Adam(learning_rate=cfg.lr,
                                                clipnorm=cfg.max_grad_norm)

    def policy(self, obs: np.ndarray):
        """Returns (action, logprob, value) for a single observation."""
        x = obs.reshape(1, -1).astype(np.float32)
        logits = self.actor(x, training=False).numpy().ravel()
        # Stable softmax + sample.
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        action = int(self.rng.choice(self.cfg.n_actions, p=probs))
        logp = float(np.log(probs[action] + 1e-12))
        value = float(self.critic(x, training=False).numpy().ravel()[0])
        return action, logp, value

    def value(self, obs: np.ndarray) -> float:
        x = obs.reshape(1, -1).astype(np.float32)
        return float(self.critic(x, training=False).numpy().ravel()[0])

    def update(self, batch: dict) -> dict:
        cfg = self.cfg
        tf = self.tf

        obs = batch["obs"]
        actions = batch["actions"]
        old_logp = batch["logp"]
        adv = batch["advantages"]
        ret = batch["returns"]

        # Normalize advantages (common stability trick).
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(obs)
        idxs = np.arange(n)
        last_kl = 0.0
        last_clipfrac = 0.0
        last_pg_loss = 0.0
        last_v_loss = 0.0
        last_entropy = 0.0
        early_stopped_at = cfg.epochs

        for epoch in range(cfg.epochs):
            self.rng.shuffle(idxs)
            for start in range(0, n, cfg.minibatch_size):
                mb = idxs[start:start + cfg.minibatch_size]
                pg_loss, v_loss, entropy, approx_kl, clipfrac = self._train_minibatch(
                    obs[mb], actions[mb], old_logp[mb], adv[mb], ret[mb],
                )
                last_pg_loss = float(pg_loss)
                last_v_loss = float(v_loss)
                last_entropy = float(entropy)
                last_kl = float(approx_kl)
                last_clipfrac = float(clipfrac)

            if cfg.target_kl is not None and last_kl > 1.5 * cfg.target_kl:
                early_stopped_at = epoch + 1
                break

        return {
            "pg_loss": last_pg_loss,
            "v_loss": last_v_loss,
            "entropy": last_entropy,
            "kl": last_kl,
            "clipfrac": last_clipfrac,
            "epochs_used": early_stopped_at,
        }

    def _train_minibatch(self, obs, actions, old_logp, adv, ret):
        tf = self.tf
        cfg = self.cfg
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_logp_tf = tf.convert_to_tensor(old_logp, dtype=tf.float32)
        adv_tf = tf.convert_to_tensor(adv, dtype=tf.float32)
        ret_tf = tf.convert_to_tensor(ret, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            logits = self.actor(obs, training=True)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            probs = tf.nn.softmax(logits, axis=-1)
            entropy = -tf.reduce_sum(probs * log_probs, axis=-1)
            entropy_mean = tf.reduce_mean(entropy)

            idx = tf.stack([tf.range(tf.shape(actions_tf)[0]), actions_tf], axis=1)
            new_logp = tf.gather_nd(log_probs, idx)

            ratio = tf.exp(new_logp - old_logp_tf)
            unclipped = ratio * adv_tf
            clipped = tf.clip_by_value(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_tf
            pg_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

            value = tf.squeeze(self.critic(obs, training=True), axis=-1)
            v_loss = 0.5 * tf.reduce_mean(tf.square(value - ret_tf))

            actor_loss = pg_loss - cfg.ent_coef * entropy_mean
            critic_loss = cfg.vf_coef * v_loss

            # Approximate KL (Schulman blog post): mean(old_logp - new_logp)
            approx_kl = tf.reduce_mean(old_logp_tf - new_logp)
            clipfrac = tf.reduce_mean(
                tf.cast(tf.greater(tf.abs(ratio - 1.0), cfg.clip_eps), tf.float32)
            )

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.opt_actor.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.opt_critic.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        del tape

        return pg_loss, v_loss, entropy_mean, approx_kl, clipfrac
