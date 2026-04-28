# HW5 – Proximal Policy Optimization (PPO)

**Lecture 9-10** – métodos de policy gradient en Deep RL. PPO es el método
estándar de hecho hoy en día.

## Objetivos

1. Implementar **PPO** con clipping, **GAE-λ** y red actor-crítico desde cero
   en Keras (sin Stable-Baselines).
2. Entrenar en **LunarLander-v3** (acciones discretas) hasta superar el
   umbral "resuelto" (R ≥ 200 en 100 episodios consecutivos).
3. Familiarizarse con los detalles "que importan" en PPO: orthogonal init,
   advantage normalization, early-stop por KL, mini-batches.

## Componentes

- `networks.py` – actor (categorical logits) + crítico (V(s)) con orthogonal
  initialization estándar.
- `buffer.py` – `RolloutBuffer` con cómputo de GAE-λ.
- `ppo.py` – `PPOAgent` con loss clipped + value loss + entropy bonus,
  early stopping por KL, advantage norm, gradient clipping.
- `run.py` – driver completo con logging.

## Ejecución

```bash
# LunarLander-v3 hasta resolverlo (~500k-1M steps con CPU, mucho más rápido en GPU).
python -m hw5_ppo.run --env LunarLander --total-steps 800000

# Sanity check rápido en CartPole:
python -m hw5_ppo.run --env CartPole --total-steps 100000 --rollout-steps 1024
```

## Hiperparámetros

| Param | Valor | Comentario |
|---|---|---|
| Rollout T | 2048 | Compromiso entre varianza y stale data |
| Epochs K | 10 | "K passes per rollout" del paper |
| Mini-batch | 64 | OK con T=2048 → 32 updates/epoch |
| LR | 3e-4 | Adam, valor canónico Schulman |
| Clip ε | 0.2 | Estándar |
| γ | 0.99 | Episodios largos en LunarLander |
| GAE λ | 0.95 | Trade-off bias/variance estándar |
| Entropy coef | 0.01 | Mantiene exploración inicial |
| Target KL | 0.02 | Early stop si supera 1.5× |
| Grad clip | 0.5 | Estabilidad numérica |

## Resultado esperado

- Con ~500k steps PPO suele estar en R(100) ≈ 200-260 en LunarLander.
- En CartPole alcanza el máximo (500) muy rápido (<100k steps).
