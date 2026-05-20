"""Visualize the learned Q-table and policy on the Taxi-v3 grid.

The PDF explicitly asks (for both Activity 1 and Activity 2):
  "Visualize the Q-Table: Display the Q-values for each state-action pair"
  "Visualize the Policy: Show the optimal policy on the grid"

Taxi has 500 states encoded as:
    state = ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + destination

with:
    taxi_row, taxi_col in {0..4}
    passenger_loc in {0,1,2,3} = {R,G,Y,B} or 4 = "in taxi"
    destination   in {0,1,2,3} = {R,G,Y,B}

We can't visualize all 500 states at once. Instead, we pick a CANONICAL
configuration (passenger at R = (0,0), destination at G = (0,4)) and show
the policy + V(s) over the 5×5 taxi-position grid. This is the textbook
way of visualizing Taxi: pick a scenario and show what the agent would do
from every possible taxi position.

We do this for all 4 algorithms side-by-side, plus a V(s) heatmap.

Taxi action encoding (gymnasium):
    0 = South   1 = North   2 = East   3 = West   4 = Pickup   5 = Dropoff
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


# Taxi action vectors for the 4 directional actions
ACTION_VECS = {
    0: (0, 1),   # South (y grows down in screen coords)
    1: (0, -1),  # North
    2: (1, 0),   # East
    3: (-1, 0),  # West
}
ACTION_LABELS = {0: "S", 1: "N", 2: "E", 3: "W", 4: "PU", 5: "DO"}

# Taxi map (5x5). Color-coded landmarks at R, G, Y, B.
# These are at: R=(0,0), G=(0,4), Y=(4,0), B=(4,3) in gymnasium Taxi-v3.
LANDMARKS = {
    "R": (0, 0),
    "G": (0, 4),
    "Y": (4, 0),
    "B": (4, 3),
}
LANDMARK_COLORS = {"R": "#ffadad", "G": "#caffbf", "Y": "#fdffb6", "B": "#a0c4ff"}

LABELS = {
    "sarsa": "SARSA (on-policy)",
    "q_learning": "Q-Learning (off-policy)",
    "expected_sarsa": "Expected SARSA",
    "double_q": "Double Q-Learning",
}


def encode_taxi_state(taxi_row: int, taxi_col: int,
                      passenger_loc: int, destination: int) -> int:
    """Replicate the gymnasium Taxi-v3 state encoding."""
    return ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + destination


def policy_for_config(Q: np.ndarray, passenger_loc: int,
                      destination: int) -> tuple[np.ndarray, np.ndarray]:
    """For a fixed (passenger_loc, destination), return:
       - action grid of shape (5, 5)   : best action per taxi position
       - V-value grid of shape (5, 5)  : max_a Q(s, a) per taxi position
    """
    actions = np.zeros((5, 5), dtype=np.int64)
    V = np.zeros((5, 5))
    for r in range(5):
        for c in range(5):
            s = encode_taxi_state(r, c, passenger_loc, destination)
            actions[r, c] = int(Q[s].argmax())
            V[r, c] = float(Q[s].max())
    return actions, V


def draw_taxi_grid(ax, title: str, passenger_loc: int, destination: int):
    """Draw the 5x5 taxi grid with landmark colors."""
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(4.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color="#aaaaaa", linewidth=0.6)

    # Landmark cells
    landmark_names = ["R", "G", "Y", "B"]
    for i, name in enumerate(landmark_names):
        r, c = LANDMARKS[name]
        color = LANDMARK_COLORS[name]
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                facecolor=color, edgecolor=color, zorder=0))

    # Mark passenger origin (P) and destination (D)
    p_name = landmark_names[passenger_loc] if passenger_loc < 4 else None
    d_name = landmark_names[destination]
    if p_name:
        pr, pc = LANDMARKS[p_name]
        ax.text(pc, pr - 0.32, "P", ha="center", va="center",
                fontsize=11, fontweight="bold", color="#aa0000")
    dr, dc = LANDMARKS[d_name]
    ax.text(dc, dr + 0.32, "D", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#005500")

    ax.set_title(title, fontsize=10)


def draw_taxi_policy_arrows(ax, actions: np.ndarray, passenger_loc: int,
                              destination: int):
    """Draw arrows for the policy on the 5x5 grid. Special markers for
    Pickup (4) and Dropoff (5)."""
    landmark_names = ["R", "G", "Y", "B"]
    p_name = landmark_names[passenger_loc] if passenger_loc < 4 else None
    d_name = landmark_names[destination]

    for r in range(5):
        for c in range(5):
            a = int(actions[r, c])
            if a in (0, 1, 2, 3):
                dx, dy = ACTION_VECS[a]
                ax.annotate(
                    "",
                    xy=(c + 0.30 * dx, r + 0.30 * dy),
                    xytext=(c - 0.25 * dx, r - 0.25 * dy),
                    arrowprops=dict(arrowstyle="-|>",
                                     color="#222222",
                                     lw=1.4,
                                     mutation_scale=12),
                    zorder=4,
                )
            elif a == 4:  # pickup
                ax.text(c, r, "PU", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="#0066cc",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="#e6f0ff", edgecolor="#0066cc"))
            elif a == 5:  # dropoff
                ax.text(c, r, "DO", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="#cc6600",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="#fff0e6", edgecolor="#cc6600"))


def main():
    runs = np.load("hw2_sarsa_qlearning/results/four_algos.npz")
    out_assets = Path("hw2_sarsa_qlearning/assets")
    out_assets.mkdir(parents=True, exist_ok=True)

    # Canonical configuration: passenger at R (0,0) waiting, destination G (0,4)
    passenger_loc = 0  # R
    destination = 1    # G

    # --- Policy arrows for all 4 algos ---
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
    for ax, algo in zip(axes, ["sarsa", "q_learning", "expected_sarsa", "double_q"]):
        Q_seeds = runs[f"taxi_{algo}_Q"]   # (seeds, S, A)
        Q_mean = Q_seeds.mean(axis=0)
        actions, V = policy_for_config(Q_mean, passenger_loc, destination)
        draw_taxi_grid(ax, LABELS[algo], passenger_loc, destination)
        draw_taxi_policy_arrows(ax, actions, passenger_loc, destination)
    fig.suptitle(
        "Taxi-v3 greedy policy   (Passenger waiting at R, Destination G)\n"
        "Arrows = best action per taxi position (averaged across 10 seeds)",
        fontsize=12, y=1.04)
    out = out_assets / "taxi_policy_grid.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # --- V(s) heatmap for SARSA and Q-Learning ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    Vs = {}
    for algo in ["sarsa", "q_learning"]:
        Q_seeds = runs[f"taxi_{algo}_Q"]
        Q_mean = Q_seeds.mean(axis=0)
        _, V = policy_for_config(Q_mean, passenger_loc, destination)
        Vs[algo] = V
    vmin = min(v.min() for v in Vs.values())
    vmax = max(v.max() for v in Vs.values())
    for ax, algo in zip(axes, ["sarsa", "q_learning"]):
        V = Vs[algo]
        im = ax.imshow(V, cmap="viridis", vmin=vmin, vmax=vmax)
        for r in range(5):
            for c in range(5):
                ax.text(c, r, f"{V[r, c]:.1f}",
                        ha="center", va="center",
                        color="white" if V[r, c] < (vmin + vmax) / 2 else "black",
                        fontsize=9, fontweight="bold")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_title(f"V(s) = max_a Q(s,a)   ·   {LABELS[algo]}",
                     fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.85, label="V")
    fig.suptitle(
        "Taxi-v3 V*(s) heatmap   (Passenger at R, Destination G)\n"
        "Brighter = higher value (closer to passenger / destination)",
        fontsize=12, y=1.02)
    out2 = out_assets / "taxi_q_heatmap.png"
    fig.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out2}")

    # --- Second config: passenger ALREADY in taxi, heading to Y ---
    passenger_loc = 4  # In taxi
    destination = 2    # Y
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
    for ax, algo in zip(axes, ["sarsa", "q_learning", "expected_sarsa", "double_q"]):
        Q_seeds = runs[f"taxi_{algo}_Q"]
        Q_mean = Q_seeds.mean(axis=0)
        actions, V = policy_for_config(Q_mean, passenger_loc, destination)
        draw_taxi_grid(ax, LABELS[algo], passenger_loc, destination)
        draw_taxi_policy_arrows(ax, actions, passenger_loc, destination)
    fig.suptitle(
        "Taxi-v3 greedy policy   (Passenger ALREADY IN TAXI, Destination Y)\n"
        "Arrows = best action per taxi position (averaged across 10 seeds)",
        fontsize=12, y=1.04)
    out3 = out_assets / "taxi_policy_grid_pickup.png"
    fig.savefig(out3, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out3}")


if __name__ == "__main__":
    main()
