# Reinforcement Learning – Homework

Repositorio con las prácticas individuales (HW1-HW6) y el proyecto grupal de la
asignatura de Reinforcement Learning impartida por Jaume Manero.

## Estructura

| Carpeta | Tarea | Algoritmos | Entorno |
|---|---|---|---|
| `hw1_frozen_lake/` | HW1 – Intro a Gym + MC/TD | Monte Carlo, TD(0) | FrozenLake-v1 |
| `hw2_sarsa_qlearning/` | HW2 – Tabular RL | SARSA, Q-Learning | Taxi-v3, CliffWalking-v0 |
| `hw3_qlearning_cartpole/` | HW3 – Q-Learning en Cartpole | Q-Learning + discretización | CartPole-v1 |
| `hw4_dqn_ddqn/` | HW4 – Deep Q-Networks | DQN, DDQN (Keras) | LunarLander-v3, CartPole-v1 |
| `hw5_ppo/` | HW5 – Policy Gradient | PPO (clip + GAE) | LunarLander-v3 |
| `hw6_ai_agent/` | HW6 – AI Agent (opcional) | Crew.ai | – |
| `group_project/` | Proyecto grupal | PPO + Imitation Learning | Walker2D / Ant |
| `utils/` | Utilidades compartidas (plotting, seeding, replay buffers) | – | – |

## Setup rápido

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Cada carpeta contiene su propio `README.md` con: objetivos, ejecución,
hiperparámetros usados, resultados esperados y discusión.

## Ejecución de cada HW

```bash
python -m hw1_frozen_lake.run             # entrena MC y TD, guarda plots
python -m hw2_sarsa_qlearning.run         # SARSA vs Q-Learning, ambos entornos
python -m hw3_qlearning_cartpole.run      # discretiza estado y entrena
python -m hw4_dqn_ddqn.run --env CartPole-v1   # DQN/DDQN
python -m hw5_ppo.run --env LunarLander-v3
```

## Notas sobre el uso de IA

Conforme a las instrucciones del curso, los HW1-HW3 se han implementado sin
asistencia de Codex / autocompletado IA en el corazón de los algoritmos – la
implementación está escrita "a mano" y comentada para entender cada ecuación
(Bellman, target TD, on-policy vs off-policy). En HW4-HW5 sí se ha apoyado
con IA para el código de soporte (loops de entrenamiento, plotting), pero
las decisiones de diseño y los hiperparámetros se han razonado y validado.

## Bibliografía base

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2ª ed.)
- Miguel Morales, *Grokking Deep Reinforcement Learning*
- Maxim Lapan, *Deep Reinforcement Learning Hands-On*
- F. Chollet, *Deep Learning with Python*
