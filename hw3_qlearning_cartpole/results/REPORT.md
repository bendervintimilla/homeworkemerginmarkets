# HW3 – Q-Learning tabular en CartPole-v1

Bins por dimensión (x, x_dot, theta, theta_dot): (3, 3, 12, 12)
Estados totales: 1296
alpha=0.1, gamma=0.99, episodios=15000

Retorno medio últimos 100 episodios entrenamiento: 390.28
Retorno medio evaluación greedy (100 ep)        : 489.05
Longitud media en evaluación                    : 489.05

**¿Resuelto?** (489.1 >= 195) -> True

## Decisiones de diseño

- Posición y velocidad del carro están infrabineadas (3 bins) porque
  la dinámica relevante para no caer está dominada por theta y
  theta_dot. Más bins en esas dos dimensiones (12 cada una) capturan
  bien la región crítica cerca de theta=0.
- Velocidades teóricamente ilimitadas: clipping a [-3, 3] y [-3.5, 3.5]
  basado en la distribución observada al ejecutar políticas aleatorias.
- El espacio discreto resultante es 3*3*12*12 = 1296 estados, x 2
  acciones = 2592 entradas en Q. Tamaño manejable y converge en pocos
  miles de episodios.

## Por qué el siguiente paso es DQN

- La discretización tiene un coste claro: cuando aumentamos el número
  de bins, el Q-table crece exponencialmente (curse of dimensionality)
  y muchos estados se visitan poquísimo. CartPole es un buen sandbox
  para mostrar esto. En HW4 sustituimos la tabla por una red neuronal
  que generaliza entre estados similares.