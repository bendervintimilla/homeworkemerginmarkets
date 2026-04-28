# HW1 – Frozen Lake con Monte Carlo y Temporal Difference

**Lecture 4** – Introducción a Gymnasium + algoritmos sin modelo.

## Objetivos

1. Familiarizarse con la API de Gymnasium.
2. Implementar control on-policy de **Monte Carlo** (first-visit, ε-soft).
3. Implementar **TD(0)** para evaluación de política y para control
   (regla de actualización tipo Q-learning).
4. Comparar ambos sobre `FrozenLake-v1` (slippery y non-slippery).

## Ficheros

- `monte_carlo.py` – control de MC first-visit con media incremental.
- `td_learning.py` – TD(0) prediction + TD control.
- `run.py` – orquesta entrenamiento, evaluación y plots.

## Ejecución

```bash
# Determinista (más fácil, converge rápido)
python -m hw1_frozen_lake.run --episodes 30000

# Resbaladizo (caso interesante: alta estocasticidad)
python -m hw1_frozen_lake.run --slippery --episodes 80000

# Mapa 8x8
python -m hw1_frozen_lake.run --map 8x8 --slippery --episodes 200000
```

Los resultados se guardan en `hw1_frozen_lake/results/` con curvas de
aprendizaje, mapas de valor V(s) y un REPORT.md.

## Hiperparámetros usados

| Param | Valor | Justificación |
|---|---|---|
| γ (discount) | 0.99 | Episodios cortos pero recompensa muy diferida |
| α (TD) | 0.1 | Estable para control tabular en este tamaño |
| ε schedule | 1.0 → 0.05 lineal | 80% de los episodios para decay |
| Episodios MC | 50k–80k | MC necesita muchos episodios completos |

## Resultados esperados (slippery=False)

- MC y TD convergen a ≈100% éxito.
- TD converge en menos episodios efectivos.

## Resultados esperados (slippery=True)

- Tasa de éxito ~70-80% (la estocasticidad limita el óptimo).
- TD muestra menos varianza en las curvas de aprendizaje.
