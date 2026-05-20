"""HW1 driver: train MC and TD on FrozenLake-v1 (slippery and non-slippery),
evaluate the learned greedy policy, and dump plots + a small report.

Usage:
    python -m hw1_frozen_lake.run
    python -m hw1_frozen_lake.run --slippery --episodes 80000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from utils import plot_multiple_curves, save_fig, set_global_seed
from utils.plotting import plot_value_grid

from .monte_carlo import mc_control, derive_policy, state_value_from_q
from .td_learning import td_control, td0_evaluation


ARROWS = {0: "<", 1: "v", 2: ">", 3: "^"}  # FrozenLake action map


def evaluate_policy(env: gym.Env, policy: np.ndarray, episodes: int = 1000,
                    seed: int = 123) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    rewards = np.zeros(episodes)
    lengths = np.zeros(episodes)
    for i in range(episodes):
        s, _ = env.reset(seed=int(rng.integers(1 << 31)))
        done, truncated = False, False
        ep_r, steps = 0.0, 0
        while not (done or truncated):
            a = int(policy[s])
            s, r, done, truncated, _ = env.step(a)
            ep_r += r
            steps += 1
        rewards[i] = ep_r
        lengths[i] = steps
    return float(rewards.mean()), float(lengths.mean())


def render_policy_grid(policy: np.ndarray, shape: tuple[int, int]) -> str:
    rows = []
    for r in range(shape[0]):
        row = " ".join(ARROWS[int(policy[r * shape[1] + c])] for c in range(shape[1]))
        rows.append(row)
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="HW1 - FrozenLake MC vs TD")
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--slippery", action="store_true",
                        help="Use slippery FrozenLake (default: non-slippery, easier).")
    parser.add_argument("--map", default="4x4", choices=["4x4", "8x8"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="hw1_frozen_lake/results")
    args = parser.parse_args()

    set_global_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "FrozenLake-v1",
        is_slippery=args.slippery,
        map_name=args.map,
    )
    shape = (4, 4) if args.map == "4x4" else (8, 8)

    print(f"\n[HW1] Environment: FrozenLake-v1 ({args.map}, slippery={args.slippery})")
    print(f"      |S| = {env.observation_space.n}, |A| = {env.action_space.n}")
    print(f"      Episodes: {args.episodes}\n")

    # --- Monte Carlo control ---
    print("[HW1] Training Monte Carlo first-visit control...")
    Q_mc, ret_mc, _ = mc_control(env, num_episodes=args.episodes,
                                 gamma=0.99, seed=args.seed)
    pi_mc = derive_policy(Q_mc)
    V_mc = state_value_from_q(Q_mc)

    # --- TD control (Q-learning style) ---
    print("[HW1] Training TD-control (Q-learning update)...")
    Q_td, ret_td, _ = td_control(env, num_episodes=args.episodes,
                                 alpha=0.1, gamma=0.99, seed=args.seed)
    pi_td = derive_policy(Q_td)
    V_td = state_value_from_q(Q_td)

    # --- TD(0) policy evaluation of the MC-learned policy ---
    print("[HW1] Evaluating MC's policy via TD(0) prediction...")
    V_eval = td0_evaluation(env, pi_mc, num_episodes=10_000, alpha=0.1,
                            gamma=0.99, seed=args.seed)

    # --- Greedy policy evaluation (true performance) ---
    eval_env = gym.make("FrozenLake-v1", is_slippery=args.slippery,
                        map_name=args.map)
    mc_succ, mc_len = evaluate_policy(eval_env, pi_mc, episodes=2000)
    td_succ, td_len = evaluate_policy(eval_env, pi_td, episodes=2000)

    # --- Plots ---
    plot_multiple_curves(
        {"Monte Carlo": ret_mc, "TD-control (Q-learning)": ret_td},
        title=f"FrozenLake-v1 ({args.map}, slippery={args.slippery}) – returns",
        out_path=out / "learning_curves.png",
        window=500, ylabel="Episode return",
    )
    plot_value_grid(V_mc, shape, "V_MC(s) = max_a Q(s,a)", out / "V_mc.png")
    plot_value_grid(V_td, shape, "V_TD(s) = max_a Q(s,a)", out / "V_td.png")
    plot_value_grid(V_eval, shape, "V^pi_MC via TD(0) prediction",
                    out / "V_td0_eval_pi_mc.png")

    # --- Text report ---
    report = [
        f"# HW1 – FrozenLake-v1 ({args.map}, slippery={args.slippery})",
        "",
        f"Episodes: {args.episodes}, gamma=0.99, alpha_TD=0.1, eps: 1.0 -> 0.05.",
        "",
        "## Greedy policy success rate (2000 eval episodes)",
        f"- Monte Carlo : success={mc_succ:.3f}, mean_len={mc_len:.2f}",
        f"- TD-control  : success={td_succ:.3f}, mean_len={td_len:.2f}",
        "",
        "## Greedy policy (MC)",
        "```",
        render_policy_grid(pi_mc, shape),
        "```",
        "",
        "## Greedy policy (TD)",
        "```",
        render_policy_grid(pi_td, shape),
        "```",
        "",
        "## Discusión",
        "- MC necesita episodios completos para actualizar; converge más lento por",
        "  episodio pero con menos sesgo (no bootstrapping).",
        "- TD(0) actualiza en cada paso (bootstrap) – aprende online y aprovecha",
        "  trajectorias parciales. Suele converger antes en este entorno.",
        "- En FrozenLake slippery la varianza de retornos es alta y MC paga el",
        "  precio (más episodios para una buena estimación de Q).",
    ]
    (out / "REPORT.md").write_text("\n".join(report))

    print("\n=== HW1 results ===")
    print(f"  MC : success={mc_succ:.3f}")
    print(f"  TD : success={td_succ:.3f}")
    print(f"  Plots & report -> {out}/")


if __name__ == "__main__":
    main()
