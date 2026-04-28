# HW4 – DQN y Double-DQN

**Lecture 8** – sofisticaciones del DQN.

## Objetivos

1. Implementar **DQN** (Mnih et al. 2015) en Keras: red online + target,
   replay buffer uniforme, ε-greedy con decay, Huber loss, target update
   periódico.
2. Implementar **Double DQN** (van Hasselt et al. 2016) modificando solo el
   cálculo del target.
3. Entrenar y comparar en **CartPole-v1** y **LunarLander-v3**.

## Componentes

- `agent.py`
  - `DQNAgent` con flag `double=True/False` para alternar.
  - Construcción de la red Keras (MLP 2 capas).
  - Bucle de entrenamiento: GradientTape + Adam + clipping global.
- `run.py` – driver principal con configs por entorno.
- `utils/replay_buffer.py` – replay vectorizado.

## Ejecución

```bash
# CartPole rápido (~1-2 min con CPU)
python -m hw4_dqn_ddqn.run --env CartPole --steps 50000

# LunarLander (recomendado GPU; sin GPU tarda ~30-60 min)
python -m hw4_dqn_ddqn.run --env LunarLander --steps 300000
```

## Hiperparámetros recomendados

| Param | CartPole | LunarLander |
|---|---|---|
| Hidden | (64, 64) | (128, 128) |
| LR | 1e-3 | 5e-4 |
| Buffer | 50k | 100k |
| Batch | 64 | 64 |
| Target update | 200 grad-steps | 1000 grad-steps |
| Train freq | cada 1 step | cada 4 steps |
| ε decay | 10k | 200k |
| Learning starts | 500 | 10k |

## Lecturas esperadas

- **CartPole**: ambos llegan a >450 antes de los 30k steps. Diferencia
  pequeña porque la recompensa está acotada.
- **LunarLander**: DDQN típicamente alcanza >200 (umbral de "resuelto") con
  curvas más estables; DQN tiende a oscilar más debido a la sobreestimación.
