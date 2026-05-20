# HW3 – Q-Learning tabular en CartPole-v1

**Lecture 7-8** – puente entre RL tabular y Deep RL.

## Objetivos

1. Aplicar Q-Learning a un entorno **continuo** mediante **discretización**
   manual del estado.
2. Diseñar un binning sensato (más resolución en las dimensiones críticas).
3. Confirmar que el agente "resuelve" CartPole (retorno medio ≥ 195 sobre
   100 episodios) y entender las limitaciones de discretizar.

## Uso

```bash
python -m hw3_qlearning_cartpole.run --episodes 15000

# Probar otros binnings:
python -m hw3_qlearning_cartpole.run --bins 1 1 6 12 --episodes 10000
```

## Bins recomendados

`(3, 3, 12, 12)` para `(x, x_dot, theta, theta_dot)`. Posición y velocidad del
carro son menos relevantes que el ángulo y la velocidad angular del polo
para mantener el equilibrio.

## Resultados esperados

- Mean return greedy (100 ep) entre **150 y 500**, dependiendo del seed y bins.
- Con `(3,3,12,12)` y 15k episodios, suele superar 195 (resuelto).

## Conexión con HW4

Esta práctica deja patente la **maldición de la dimensionalidad**: si
quisiéramos más resolución, la tabla crece rápidamente y muchos estados
quedan poco visitados. HW4 sustituye la tabla por una red neuronal (DQN) que
generaliza entre estados similares.
