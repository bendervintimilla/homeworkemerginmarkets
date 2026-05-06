"""HW1 full pipeline – Activities 2-5 of the spec.

Activity 2: MC on FrozenLake 4x4 and 8x8.
Activity 3: MC on Volcano (custom env).
Activity 4: MC on Taxi.
Activity 5: TD(0) on FrozenLake.

Outputs everything under hw1_frozen_lake/results_full/ so the notebook can
just load the .npz and the plots without re-training.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from utils import plot_learning_curve, plot_multiple_curves, set_global_seed
from utils.plotting import plot_value_grid

from .mc_generic import mc_control, evaluate_greedy as mc_eval
from .td_learning import td_control, td0_evaluation
from .volcano_world import VolcanoWorldEnv


def _resolve(name: str, versions: list[str]) -> str:
    for v in versions:
        env_id = f"{name}-{v}"
        try:
            gym.spec(env_id)
            return env_id
        except Exception:
            continue
    raise RuntimeError(f"No registered version of {name}")


def run_frozenlake(map_name: str, slippery: bool, episodes: int, seed: int,
                   out: Path) -> dict:
    print(f"\n[A2] FrozenLake-v1 ({map_name}, slippery={slippery}), {episodes} ep")
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=slippery)
    shape = (4, 4) if map_name == "4x4" else (8, 8)

    t0 = time.time()
    res_mc = mc_control(env, num_episodes=episodes, gamma=0.99, seed=seed)
    t_mc = time.time() - t0

    t0 = time.time()
    Q_td, ret_td, len_td = td_control(env, num_episodes=episodes, alpha=0.1,
                                       gamma=0.99, seed=seed)
    t_td = time.time() - t0

    eval_env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=slippery)
    mc_succ, mc_len = mc_eval(eval_env, res_mc.Q, episodes=1000)
    td_succ, td_len = mc_eval(eval_env, Q_td, episodes=1000)

    tag = f"frozenlake_{map_name}_{'slippery' if slippery else 'deterministic'}"
    plot_multiple_curves(
        {"MC first-visit": res_mc.returns, "TD-control": ret_td},
        title=f"FrozenLake-v1 {map_name} (slippery={slippery})",
        out_path=out / f"{tag}_returns.png",
        window=max(100, episodes // 50),
    )
    plot_value_grid(res_mc.Q.max(axis=1), shape, f"V_MC – {tag}",
                    out / f"{tag}_V_mc.png")
    plot_value_grid(Q_td.max(axis=1), shape, f"V_TD – {tag}",
                    out / f"{tag}_V_td.png")

    np.savez(
        out / f"{tag}.npz",
        Q_mc=res_mc.Q, returns_mc=res_mc.returns,
        Q_td=Q_td, returns_td=ret_td,
    )
    return {
        "env": f"FrozenLake-v1 {map_name} {'slippery' if slippery else 'det'}",
        "episodes": episodes,
        "mc_time_s": t_mc, "td_time_s": t_td,
        "mc_eval_return": mc_succ, "mc_eval_len": mc_len,
        "td_eval_return": td_succ, "td_eval_len": td_len,
        "mc_train_last100": float(res_mc.returns[-100:].mean()),
        "td_train_last100": float(ret_td[-100:].mean()),
    }


def run_volcano(episodes: int, seed: int, out: Path) -> dict:
    print(f"\n[A3] Volcano (slippery), {episodes} ep")
    env = VolcanoWorldEnv(is_slippery=True)
    t0 = time.time()
    res = mc_control(env, num_episodes=episodes, gamma=0.99, seed=seed)
    t_mc = time.time() - t0
    eval_env = VolcanoWorldEnv(is_slippery=True)
    mean_r, mean_l = mc_eval(eval_env, res.Q, episodes=1000)

    plot_learning_curve(res.returns, title="Volcano – MC training returns",
                        out_path=out / "volcano_returns.png", window=200)
    plot_value_grid(res.Q.max(axis=1), (3, 4), "V_MC – Volcano",
                    out / "volcano_V.png")
    np.savez(out / "volcano.npz", Q=res.Q, returns=res.returns)
    return {
        "env": "Volcano (slippery)",
        "episodes": episodes,
        "mc_time_s": t_mc, "td_time_s": np.nan,
        "mc_eval_return": mean_r, "mc_eval_len": mean_l,
        "td_eval_return": np.nan, "td_eval_len": np.nan,
        "mc_train_last100": float(res.returns[-100:].mean()),
        "td_train_last100": np.nan,
    }


def run_taxi(episodes: int, seed: int, out: Path) -> dict:
    taxi_id = _resolve("Taxi", ["v4", "v3"])
    print(f"\n[A4] {taxi_id}, {episodes} ep MC + TD")
    env = gym.make(taxi_id)
    t0 = time.time()
    res = mc_control(env, num_episodes=episodes, gamma=0.99, alpha=0.1,
                     max_steps=200, seed=seed)
    t_mc = time.time() - t0

    t0 = time.time()
    Q_td, ret_td, _ = td_control(env, num_episodes=episodes, alpha=0.1,
                                  gamma=0.99, seed=seed)
    t_td = time.time() - t0

    eval_env = gym.make(taxi_id)
    mc_r, mc_l = mc_eval(eval_env, res.Q, episodes=500)
    td_r, td_l = mc_eval(eval_env, Q_td, episodes=500)

    plot_multiple_curves(
        {"MC first-visit": res.returns, "TD-control": ret_td},
        title=f"{taxi_id} – training returns",
        out_path=out / "taxi_returns.png", window=max(100, episodes // 50),
    )
    np.savez(out / "taxi.npz", Q_mc=res.Q, returns_mc=res.returns,
             Q_td=Q_td, returns_td=ret_td)
    return {
        "env": taxi_id,
        "episodes": episodes,
        "mc_time_s": t_mc, "td_time_s": t_td,
        "mc_eval_return": mc_r, "mc_eval_len": mc_l,
        "td_eval_return": td_r, "td_eval_len": td_l,
        "mc_train_last100": float(res.returns[-100:].mean()),
        "td_train_last100": float(ret_td[-100:].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fl-4x4-episodes", type=int, default=50_000)
    parser.add_argument("--fl-4x4-slippery-episodes", type=int, default=80_000)
    parser.add_argument("--fl-8x8-episodes", type=int, default=120_000)
    parser.add_argument("--volcano-episodes", type=int, default=50_000)
    parser.add_argument("--taxi-episodes", type=int, default=20_000)
    parser.add_argument("--out", default="hw1_frozen_lake/results_full")
    args = parser.parse_args()

    set_global_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    summary.append(run_frozenlake("4x4", False, args.fl_4x4_episodes, args.seed, out))
    summary.append(run_frozenlake("4x4", True, args.fl_4x4_slippery_episodes,
                                  args.seed, out))
    summary.append(run_frozenlake("8x8", False, args.fl_8x8_episodes, args.seed, out))
    summary.append(run_volcano(args.volcano_episodes, args.seed, out))
    summary.append(run_taxi(args.taxi_episodes, args.seed, out))

    # Markdown table for the notebook + REPORT.
    cols = ["env", "episodes", "mc_time_s", "td_time_s",
            "mc_eval_return", "mc_eval_len",
            "td_eval_return", "td_eval_len",
            "mc_train_last100", "td_train_last100"]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for s in summary:
        row = []
        for c in cols:
            v = s[c]
            if isinstance(v, float):
                row.append(f"{v:.3f}" if not np.isnan(v) else "—")
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines)

    (out / "comparative_table.md").write_text(table)
    np.save(out / "summary.npy", np.array(summary, dtype=object))
    print("\n=== HW1 comparative table ===")
    print(table)
    print(f"\nArtifacts -> {out}/")


if __name__ == "__main__":
    main()
