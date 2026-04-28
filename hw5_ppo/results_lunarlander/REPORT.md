# HW5 – PPO en LunarLander-v3

Total env-steps: 200000
Rollout: 2048, Epochs: 10, Mini-batch: 64, LR: 0.0003
GAE: gamma=0.99, lambda=0.95, clip=0.2

Retorno medio últimos 100 episodios: **62.24**
"Resuelto" en LunarLander es R >= 200.

## Notas de implementación

- Inicialización ortogonal (gain=sqrt(2) en hidden, 0.01 en logits, 1.0 en V).
  Sigue las guidelines del paper original y del *PPO Implementation Details* blog.
- Normalización de advantages a media 0, std 1 por minibatch.
- Early stopping por KL aprox > 1.5 * target_kl (target_kl=0.02).
- Entropy bonus = 0.01 (favorece exploración temprana).
- Value coefficient = 0.5, gradient clipping = 0.5.