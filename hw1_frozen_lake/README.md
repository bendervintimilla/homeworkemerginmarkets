# HW1 – Gymnasium, Monte Carlo y TD(0)

**Lecture 4** – Introducción a Gymnasium y métodos sin modelo.

Cubre las cinco actividades del enunciado oficial:

1. Familiarización con los entornos (CartPole, Taxi, LunarLander, FrozenLake).
2. **Monte Carlo first-visit** en FrozenLake 4×4 y 8×8.
3. **Monte Carlo** en el entorno **Volcano** custom.
4. **Monte Carlo** en Taxi.
5. **TD(0)** en FrozenLake/Taxi.

## Entregable

`Assignment1_HW1.ipynb` – notebook Jupyter con las 5 actividades, las
visualizaciones (curvas de aprendizaje, mapas de valor V(s), políticas
greedy en ASCII), la tabla comparativa y las respuestas a las 5 preguntas
del spec.

## Estructura

| Fichero | Contenido |
|---|---|
| `monte_carlo.py` | MC first-visit clásico (Sutton & Barto Cap. 5.4). |
| `mc_generic.py` | MC genérico (first-visit / every-visit, α opcional, max_steps). |
| `td_learning.py` | TD(0) prediction + Q-Learning style TD-control. |
| `volcano_world.py` | Entorno Volcano (vendored de `castorgit/RL_course`). |
| `run.py` | Driver original con un solo entorno (compatibilidad). |
| `run_full.py` | **Driver completo HW1** – entrena los 5 escenarios y deja todo en `results_full/`. |
| `Assignment1_HW1.ipynb` | Notebook entregable. |

## Reproducir

```bash
python -m hw1_frozen_lake.run_full --seed 42
```

Defaults:
- FrozenLake 4×4 deterministic: 50 000 episodios.
- FrozenLake 4×4 slippery: 80 000 episodios.
- FrozenLake 8×8 deterministic: 120 000 episodios.
- Volcano (slippery): 50 000 episodios.
- Taxi: 20 000 episodios (MC + TD).

Tiempo total ≈ 5–8 min en CPU modesta. Genera:

- `comparative_table.md` – tabla con tiempos, episodios, retornos.
- `<env>_returns.png` – curvas de aprendizaje MC vs TD.
- `<env>_V_mc.png`, `<env>_V_td.png` – mapas de valor.
- `<env>.npz` – Q-tables y arrays de retornos para reabrir en el notebook.

## Hiperparámetros

| Param | FrozenLake | Volcano | Taxi |
|---|---|---|---|
| γ | 0.99 | 0.99 | 0.99 |
| α (TD) | 0.1 | – | 0.1 |
| α (MC) | 1/N (incremental) | 1/N | 0.1 (constante) |
| ε schedule | 1.0 → 0.05 (lineal) | 1.0 → 0.05 | 1.0 → 0.05 |
| max steps | – | – | 200 |

## Resultados esperados

- FrozenLake 4×4 deterministic: 100% éxito (MC y TD).
- FrozenLake 4×4 slippery: TD ≈ 0.74, MC ≈ 0.35 con 80k episodios.
- FrozenLake 8×8 deterministic: 100% éxito con 120k episodios.
- Volcano: política aversa al riesgo (sale por la salida segura, +2).
- Taxi: TD converge a ~+8 (resuelto), MC va más lento.
