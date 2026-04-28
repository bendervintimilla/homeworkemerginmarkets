"""HW2 driver: SARSA vs. Q-Learning on Taxi-v3 and CliffWalking-v0.

This homework illustrates the key practical difference between on-policy and
off-policy TD control. Cliff Walking shows it textbook-style: Q-Learning
prefers the optimal-but-risky path along the cliff, SARSA learns a safer path
because while exploring it occasionally falls and that cost is reflected in
its on-policy Q.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from utils import plot_multiple_curves, set_global_seed

from .algorithms import sarsa, q_learning, evaluate_greedy


CLIFF_SHAPE = (4, 12)
ARROWS = {0: "^", 1: ">", 2: "v", 3: "<"}  # CliffWalking action map


def _resolve_env(base: str, version_preference: list[str]) -> str:
    """Try IDs like 'Taxi-v4', 'Taxi-v3' in order and return the first one that
    is registered in the current Gymnasium install."""
    for v in version_preference:
        env_id = f"{base}-{v}"
        try:
            gym.spec(env_id)
            return env_id
        except Exception:
            continue
    raise RuntimeError(f"No registered version of {base} (tried {version_preference})")


def render_cliff_policy(Q: np.ndarray) -> str:
    policy = Q.argmax(axis=1)
    rows = []
    for r in range(CLIFF_SHAPE[0]):
        row = " ".join(ARROWS[int(policy[r * CLIFF_SHAPE[1] + c])]
                       for c in range(CLIFF_SHAPE[1]))
        rows.append(row)
    return "\n".join(rows)


def run_one(env_id: str, episodes: int, seed: int, out: Path,
            alpha: float = 0.1) -> dict:
    env = gym.make(env_id)
    print(f"\n[HW2] === {env_id} ===")
    print(f"      |S|={env.observation_space.n}, |A|={env.action_space.n}, "
          f"episodes={episodes}, alpha={alpha}")

    print("[HW2] Training SARSA...")
    sarsa_res = sarsa(env, num_episodes=episodes, alpha=alpha,
                      gamma=0.99, seed=seed)
    print("[HW2] Training Q-Learning...")
    ql_res = q_learning(env, num_episodes=episodes, alpha=alpha,
                        gamma=0.99, seed=seed)

    eval_env = gym.make(env_id)
    sarsa_ret, sarsa_len = evaluate_greedy(eval_env, sarsa_res.Q, episodes=500)
    ql_ret, ql_len = evaluate_greedy(eval_env, ql_res.Q, episodes=500)

    plot_multiple_curves(
        {"SARSA": sarsa_res.returns, "Q-Learning": ql_res.returns},
        title=f"{env_id} – training returns",
        out_path=out / f"{env_id}_returns.png",
        window=max(50, episodes // 50),
        ylabel="Episode return",
    )
    plot_multiple_curves(
        {"SARSA": sarsa_res.lengths, "Q-Learning": ql_res.lengths},
        title=f"{env_id} – episode length",
        out_path=out / f"{env_id}_lengths.png",
        window=max(50, episodes // 50),
        ylabel="Steps",
    )

    return {
        "env": env_id,
        "sarsa_eval": (sarsa_ret, sarsa_len),
        "ql_eval": (ql_ret, ql_len),
        "Q_sarsa": sarsa_res.Q,
        "Q_ql": ql_res.Q,
    }


def main():
    parser = argparse.ArgumentParser(description="HW2 - SARSA vs Q-Learning")
    parser.add_argument("--taxi-episodes", type=int, default=10_000)
    parser.add_argument("--cliff-episodes", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="hw2_sarsa_qlearning/results")
    args = parser.parse_args()

    set_global_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cliff_id = _resolve_env("CliffWalking", ["v1", "v0"])
    taxi_id = _resolve_env("Taxi", ["v4", "v3"])
    cliff = run_one(cliff_id, args.cliff_episodes, args.seed, out,
                    alpha=0.5)  # tabular cliff converges fast with bigger alpha
    taxi = run_one(taxi_id, args.taxi_episodes, args.seed, out, alpha=0.1)

    cliff_policy_sarsa = render_cliff_policy(cliff["Q_sarsa"])
    cliff_policy_ql = render_cliff_policy(cliff["Q_ql"])

    report = [
        "# HW2 – SARSA vs Q-Learning",
        "",
        "## Resultados de evaluación greedy (500 episodios)",
        "",
        "| Entorno | SARSA return | SARSA len | Q-Learn return | Q-Learn len |",
        "|---|---|---|---|---|",
        f"| {cliff_id} | {cliff['sarsa_eval'][0]:.2f} | {cliff['sarsa_eval'][1]:.2f} | "
        f"{cliff['ql_eval'][0]:.2f} | {cliff['ql_eval'][1]:.2f} |",
        f"| {taxi_id} | {taxi['sarsa_eval'][0]:.2f} | {taxi['sarsa_eval'][1]:.2f} | "
        f"{taxi['ql_eval'][0]:.2f} | {taxi['ql_eval'][1]:.2f} |",
        "",
        "## Política aprendida en CliffWalking",
        "",
        "Mapa: fila 0 = arriba, fila 3 = abajo. La fila 3 (excepto el inicio en (3,0)",
        "y la meta en (3,11)) es un acantilado: caer da -100 y reinicia el episodio.",
        "",
        "### SARSA (on-policy, prudente)",
        "```",
        cliff_policy_sarsa,
        "```",
        "",
        "### Q-Learning (off-policy, agresivo)",
        "```",
        cliff_policy_ql,
        "```",
        "",
        "## Discusión",
        "",
        "- **SARSA** converge a una política que **se aleja del acantilado** porque",
        "  aprende el valor de la política ε-greedy que ejecuta: incluso con ε bajo",
        "  el riesgo residual de caer del acantilado lastra Q en los estados",
        "  pegados al borde.",
        "- **Q-Learning** converge a la política óptima (el camino directo por el",
        "  borde) porque su target ignora la exploración (max sobre acciones).",
        "  Durante el entrenamiento sus retornos son peores (cae a la lava al",
        "  explorar) pero la política greedy final es mejor.",
        "- En **Taxi-v3** ambos convergen a una política buena (~8 de retorno),",
        "  pero Q-Learning suele aprender un poco más rápido.",
    ]
    (out / "REPORT.md").write_text("\n".join(report))

    print("\n=== HW2 results ===")
    for r in (cliff, taxi):
        print(f"  {r['env']:<20} SARSA={r['sarsa_eval'][0]:.2f} | "
              f"Q-Learn={r['ql_eval'][0]:.2f}")
    print(f"  -> {out}/")


if __name__ == "__main__":
    main()
