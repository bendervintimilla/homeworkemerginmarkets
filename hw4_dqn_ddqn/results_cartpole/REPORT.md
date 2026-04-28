# HW4 – DQN vs DDQN en CartPole-v1

Env steps: 80000, seed_DQN=42, seed_DDQN=43.

## Resultados (media de los últimos 50 episodios de entrenamiento)
- DQN  : 84.14
- DDQN : 143.72

## Discusión

- DQN puede sobreestimar Q porque toma `max` sobre estimaciones ruidosas
  del propio target net. La curva suele ser más volátil.
- DDQN desacopla selección y evaluación de la siguiente acción y
  típicamente ofrece curvas más estables y un retorno asintótico igual
  o mejor – más notable en LunarLander que en CartPole.
- En CartPole DDQN aporta poco (recompensa acotada en [0,500] y la
  optimización es fácil). En LunarLander la diferencia se ve mejor.