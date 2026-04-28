# HW2 – SARSA y Q-Learning (Tabular RL)

**Lecture 5** – Métodos "caballo de batalla" del RL tabular.

## Objetivos

1. Implementar **SARSA** (on-policy TD control) y **Q-Learning** (off-policy
   TD control) desde cero.
2. Entrenar y evaluar en **CliffWalking-v0** y **Taxi-v3**.
3. Observar la diferencia de comportamiento entre ambos algoritmos: el caso de
   CliffWalking es la demostración canónica (Sutton & Barto 6.5).

## Ejecución

```bash
python -m hw2_sarsa_qlearning.run
```

Parámetros usados (defaults):
- CliffWalking: 2000 episodios, α=0.5, γ=0.99, ε 1.0→0.05.
- Taxi-v3: 10000 episodios, α=0.1, γ=0.99, ε 1.0→0.05.

## Lectura de resultados

- `CliffWalking-v0_returns.png` – durante el entrenamiento Q-Learning suele
  tener retornos medios PEORES que SARSA (porque al explorar se cae).
- `REPORT.md` – contiene las políticas greedy aprendidas en formato ASCII.
- En CliffWalking se ve cómo SARSA sale por la fila superior (camino largo y
  seguro) mientras Q-Learning va pegado al acantilado (camino óptimo y
  peligroso).
