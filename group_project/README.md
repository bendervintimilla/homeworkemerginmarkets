# Group Project – Walker2D / Ant con PPO + Imitation Learning

40% de la nota. Presentación en clases 13-14.

## Plan

1. **PPO baseline** (`train_ppo.py`)
   - Stable-Baselines3 con `MlpPolicy [256, 256]`.
   - VecNormalize (obs y reward).
   - Entrenamiento de 1-2M steps en `Walker2d-v5` (o `Ant-v5`).
   - Callback de evaluación cada 10k steps.

2. **Imitation Learning – Behavior Cloning** (`imitation.py`)
   - Cargamos el PPO entrenado como experto.
   - Generamos 50-100 episodios del experto.
   - Entrenamos una red supervisada (MSE) que mapea obs → action.
   - Evaluamos la BC policy y comparamos.

3. **Extensión propuesta**
   - Inicializar PPO desde la BC policy y ver si converge antes / mejor.
   - Probar DAgger (recolección iterativa de trayectorias) en lugar de BC puro.
   - Comparar reward shaping (curriculum sobre velocidad lateral / altura).

## Instalación específica

```bash
pip install 'stable-baselines3[extra]' mujoco
```

`Walker2D-v5` y `Ant-v5` requieren MuJoCo (gratis, viene con `mujoco>=3.0`).

## Ejecución

```bash
# 1) Baseline
python -m group_project.train_ppo --env Walker2d --steps 1500000 --n-envs 4

# 2) BC desde el experto entrenado en (1)
python -m group_project.imitation \
    --expert group_project/runs/ppo_walker2d/final_model.zip \
    --vecnorm group_project/runs/ppo_walker2d/vecnorm.pkl \
    --env Walker2d --episodes 50 --epochs 30
```

## Métricas a reportar

| Métrica | Cómo |
|---|---|
| Curva de aprendizaje (return vs steps) | TensorBoard del `EvalCallback` |
| Retorno final ± std | media y std de 100 episodios deterministicos |
| Coste muestral (steps hasta R objetivo) | inspección de las curvas |
| Comparativa BC vs PPO | misma evaluación greedy |

## Notas de hardware

- PPO en MuJoCo es exigente: con CPU-only, 1M steps ≈ 1-2h en Walker2D-v5
  con 4 envs paralelos. Con GPU baja a ~30-45 min.
- Para iterar rápido en debugging: bajar `--steps` a 100k y `--n-envs` a 2.
