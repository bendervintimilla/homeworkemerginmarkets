# HW2 – SARSA vs Q-Learning

## Resultados de evaluación greedy (500 episodios)

| Entorno | SARSA return | SARSA len | Q-Learn return | Q-Learn len |
|---|---|---|---|---|
| CliffWalking-v1 | -17.00 | 17.00 | -13.00 | 13.00 |
| Taxi-v4 | 7.93 | 13.07 | 7.97 | 13.03 |

## Política aprendida en CliffWalking

Mapa: fila 0 = arriba, fila 3 = abajo. La fila 3 (excepto el inicio en (3,0)
y la meta en (3,11)) es un acantilado: caer da -100 y reinicia el episodio.

### SARSA (on-policy, prudente)
```
v > > > > > > > > > v v
> > ^ < ^ ^ > > ^ ^ v v
^ ^ ^ ^ ^ ^ ^ ^ ^ > > v
^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
```

### Q-Learning (off-policy, agresivo)
```
> > > > > > > > > > > v
> > > > > > > > > > > v
> > > > > > > > > > > v
^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
```

## Discusión

- **SARSA** converge a una política que **se aleja del acantilado** porque
  aprende el valor de la política ε-greedy que ejecuta: incluso con ε bajo
  el riesgo residual de caer del acantilado lastra Q en los estados
  pegados al borde.
- **Q-Learning** converge a la política óptima (el camino directo por el
  borde) porque su target ignora la exploración (max sobre acciones).
  Durante el entrenamiento sus retornos son peores (cae a la lava al
  explorar) pero la política greedy final es mejor.
- En **Taxi-v3** ambos convergen a una política buena (~8 de retorno),
  pero Q-Learning suele aprender un poco más rápido.