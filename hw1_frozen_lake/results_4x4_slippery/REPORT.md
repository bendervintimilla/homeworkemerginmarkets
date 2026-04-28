# HW1 – FrozenLake-v1 (4x4, slippery=True)

Episodes: 80000, gamma=0.99, alpha_TD=0.1, eps: 1.0 -> 0.05.

## Greedy policy success rate (2000 eval episodes)
- Monte Carlo : success=0.348, mean_len=23.57
- TD-control  : success=0.739, mean_len=44.01

## Greedy policy (MC)
```
> ^ > ^
< < > <
^ v v <
< > v <
```

## Greedy policy (TD)
```
< ^ ^ ^
< < > <
^ v < <
< > v <
```

## Discusión
- MC necesita episodios completos para actualizar; converge más lento por
  episodio pero con menos sesgo (no bootstrapping).
- TD(0) actualiza en cada paso (bootstrap) – aprende online y aprovecha
  trajectorias parciales. Suele converger antes en este entorno.
- En FrozenLake slippery la varianza de retornos es alta y MC paga el
  precio (más episodios para una buena estimación de Q).