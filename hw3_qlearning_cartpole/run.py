"""HW3 driver: tabular Q-Learning on CartPole-v1 via state discretization.

Goal: show that even without function approximation, careful discretization +
Q-Learning can solve CartPole-v1 (mean return >= 195 over 100 episodes – the
classic "solved" threshold). This is the bridge to HW4 where we replace the
table with a neural net (DQN).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from utils import plot_learning_curve, set_global_seed

from .discretizer import CartPoleDiscretizer
from .qlearning import train, evaluate


def main():
    parser = argparse.ArgumentParser(description="HW3 - Tabular Q-Learning on CartPole")
    parser.add_argument("--episodes", type=int, default=15_000)
    parser.add_argument("--bins", nargs=4, type=int, default=[3, 3, 12, 12],
                        help="Bins per dim: x, x_dot, theta, theta_dot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--out", default="hw3_qlearning_cartpole/results")
    args = parser.parse_args()

    set_global_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = gym.make("CartPole-v1")
    disc = CartPoleDiscretizer(tuple(args.bins))
    print(f"\n[HW3] Discretized state space: {disc.n_states} states "
          f"(bins={tuple(args.bins)})")
    print(f"[HW3] Training Q-Learning for {args.episodes} episodes...")

    res = train(env, disc, num_episodes=args.episodes,
                alpha=args.alpha, gamma=args.gamma, seed=args.seed)

    eval_env = gym.make("CartPole-v1")
    mean_r, mean_len = evaluate(eval_env, res.Q, disc, episodes=100)

    plot_learning_curve(res.returns,
                        title=f"CartPole-v1 tabular Q-Learning ({disc.n_states} states)",
                        out_path=out / "learning_curve.png", window=200)

    last_100 = res.returns[-100:].mean()
    report = [
        "# HW3 – Q-Learning tabular en CartPole-v1",
        "",
        f"Bins por dimensión (x, x_dot, theta, theta_dot): {tuple(args.bins)}",
        f"Estados totales: {disc.n_states}",
        f"alpha={args.alpha}, gamma={args.gamma}, episodios={args.episodes}",
        "",
        f"Retorno medio últimos 100 episodios entrenamiento: {last_100:.2f}",
        f"Retorno medio evaluación greedy (100 ep)        : {mean_r:.2f}",
        f"Longitud media en evaluación                    : {mean_len:.2f}",
        "",
        f"**¿Resuelto?** ({mean_r:.1f} >= 195) -> {mean_r >= 195}",
        "",
        "## Decisiones de diseño",
        "",
        "- Posición y velocidad del carro están infrabineadas (3 bins) porque",
        "  la dinámica relevante para no caer está dominada por theta y",
        "  theta_dot. Más bins en esas dos dimensiones (12 cada una) capturan",
        "  bien la región crítica cerca de theta=0.",
        "- Velocidades teóricamente ilimitadas: clipping a [-3, 3] y [-3.5, 3.5]",
        "  basado en la distribución observada al ejecutar políticas aleatorias.",
        "- El espacio discreto resultante es 3*3*12*12 = 1296 estados, x 2",
        "  acciones = 2592 entradas en Q. Tamaño manejable y converge en pocos",
        "  miles de episodios.",
        "",
        "## Por qué el siguiente paso es DQN",
        "",
        "- La discretización tiene un coste claro: cuando aumentamos el número",
        "  de bins, el Q-table crece exponencialmente (curse of dimensionality)",
        "  y muchos estados se visitan poquísimo. CartPole es un buen sandbox",
        "  para mostrar esto. En HW4 sustituimos la tabla por una red neuronal",
        "  que generaliza entre estados similares.",
    ]
    (out / "REPORT.md").write_text("\n".join(report))

    print(f"\n=== HW3 results ===")
    print(f"  Train mean (last 100): {last_100:.2f}")
    print(f"  Eval mean (100 ep)   : {mean_r:.2f}")
    print(f"  Solved (>= 195)      : {mean_r >= 195}")
    print(f"  -> {out}/")


if __name__ == "__main__":
    main()
