"""Side-by-side arrow visualization of the greedy policy each algorithm
learned on CliffWalking.

The notebook used an ASCII trace to compare SARSA's safe-path vs Q-Learning's
cliff-edge path. The arrow grid is the same information but visually direct
and shows the policy at every state, not just along the trajectory.

CliffWalking action encoding (gymnasium):
  0 = UP   1 = RIGHT   2 = DOWN   3 = LEFT
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ACTION_VECS = {
    0: (0, -1),   # UP    : (dx, dy) in screen coords (y grows downward)
    1: (1, 0),    # RIGHT
    2: (0, 1),    # DOWN
    3: (-1, 0),   # LEFT
}
LABELS = {
    "sarsa": "SARSA (on-policy)",
    "q_learning": "Q-Learning (off-policy)",
    "expected_sarsa": "Expected SARSA",
    "double_q": "Double Q-Learning",
}


def cliff_policy_from_Q(Q_seeds: np.ndarray) -> np.ndarray:
    """Average Q across seeds, then take greedy action. Q shape (seeds, S, A).
    Returns (4, 12) array of actions in {0,1,2,3}."""
    Q_mean = Q_seeds.mean(axis=0)
    actions = Q_mean.argmax(axis=1)
    # CliffWalking state layout: row-major, 4 rows × 12 cols, top-left = state 0.
    return actions.reshape(4, 12)


def draw_cliff_grid(ax, title: str):
    """Draw the cliffwalking 4×12 grid with cliff, start, goal markers."""
    n_rows, n_cols = 4, 12
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)  # inverted y so row 0 is on top
    ax.set_aspect("equal")
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="#aaaaaa", linewidth=0.6)

    # Cliff: bottom row, cols 1..10
    for c in range(1, n_cols - 1):
        ax.add_patch(Rectangle((c - 0.5, n_rows - 1 - 0.5), 1, 1,
                                facecolor="#fde2e2", edgecolor="#cc0000",
                                linewidth=0.8, zorder=0))
        ax.text(c, n_rows - 1, "C", ha="center", va="center",
                fontsize=9, color="#990000", fontweight="bold", zorder=5)

    # Start: bottom-left
    ax.add_patch(Rectangle((-0.5, n_rows - 1 - 0.5), 1, 1,
                            facecolor="#d4f4dd", edgecolor="#2b8a3e",
                            linewidth=0.8, zorder=0))
    ax.text(0, n_rows - 1, "S", ha="center", va="center",
            fontsize=10, color="#2b6e3a", fontweight="bold", zorder=5)

    # Goal: bottom-right
    ax.add_patch(Rectangle((n_cols - 1 - 0.5, n_rows - 1 - 0.5), 1, 1,
                            facecolor="#ffe9b3", edgecolor="#b86e00",
                            linewidth=0.8, zorder=0))
    ax.text(n_cols - 1, n_rows - 1, "G", ha="center", va="center",
            fontsize=10, color="#7a4a00", fontweight="bold", zorder=5)

    ax.set_title(title, fontsize=11)


def draw_policy_arrows(ax, policy: np.ndarray):
    """policy: (4, 12) array of actions. Skip cliff/start/goal cells."""
    n_rows, n_cols = policy.shape
    for r in range(n_rows):
        for c in range(n_cols):
            # Skip terminal cells and cliff cells
            is_cliff = (r == n_rows - 1) and (0 < c < n_cols - 1)
            is_start = (r == n_rows - 1) and (c == 0)
            is_goal = (r == n_rows - 1) and (c == n_cols - 1)
            if is_cliff or is_goal:
                continue
            a = int(policy[r, c])
            dx, dy = ACTION_VECS[a]
            arrow_color = "#222222" if not is_start else "#0c5320"
            ax.annotate(
                "",
                xy=(c + 0.35 * dx, r + 0.35 * dy),
                xytext=(c - 0.30 * dx, r - 0.30 * dy),
                arrowprops=dict(arrowstyle="-|>",
                                color=arrow_color,
                                lw=1.6,
                                mutation_scale=14),
                zorder=4,
            )


def main():
    runs = np.load("hw2_sarsa_qlearning/results/four_algos.npz")
    out_assets = Path("hw2_sarsa_qlearning/assets")
    out_assets.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 7),
                              gridspec_kw={"hspace": 0.30, "wspace": 0.05})
    for ax, algo in zip(axes.flat,
                        ["sarsa", "q_learning", "expected_sarsa", "double_q"]):
        Q_seeds = runs[f"cliff_{algo}_Q"]
        policy = cliff_policy_from_Q(Q_seeds)
        draw_cliff_grid(ax, LABELS[algo])
        draw_policy_arrows(ax, policy)

    fig.suptitle(
        "Greedy policies learned by 4 TD-control algorithms on CliffWalking\n"
        "(arrows = argmax action averaged across 10 seeds)",
        fontsize=13, y=1.00
    )
    out = out_assets / "policy_arrows.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
