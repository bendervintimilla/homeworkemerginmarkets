# HW1 – FrozenLake-v1 (4x4, slippery=False)

Episodes: 50000, gamma=0.99, alpha_TD=0.1, eps: 1.0 -> 0.05.

## Greedy policy success rate (2000 eval episodes)
- Monte Carlo : success=1.000, mean_len=6.00
- TD-control  : success=1.000, mean_len=6.00

## Greedy policy (MC)
```
v > v <
v < v <
> v v <
< > > <
```

## Greedy policy (TD)
```
v > v <
v < v <
> v v <
< > > <
```

## Discusión
- MC necesita episodios completos para actualizar; converge más lento por
  episodio pero con menos sesgo (no bootstrapping).
- TD(0) actualiza en cada paso (bootstrap) – aprende online y aprovecha
  trajectorias parciales. Suele converger antes en este entorno.
- En FrozenLake slippery la varianza de retornos es alta y MC paga el
  precio (más episodios para una buena estimación de Q).