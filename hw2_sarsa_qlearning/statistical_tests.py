"""Statistical hypothesis testing for the algorithm comparison claims.

The original notebook reports mean returns like "SARSA: -19.11 ± 1.05 vs
Q-Learning: -36.45 ± 8.80" on CliffWalking. With n=10 seeds and skewed
distributions (Q-Learning's std is 8x SARSA's), eyeballing the means is
not enough to claim the difference is real. This module runs the proper
non-parametric test.

We use the **paired Wilcoxon signed-rank test** because:
  1. Same seed -> same starting Q, same epsilon schedule, same env reset
     sequence. The runs are PAIRED by seed, not independent samples.
  2. n=10 is small. The Wilcoxon is non-parametric and does not assume
     normality (Q-Learning's per-seed returns are clearly non-Gaussian
     due to occasional cliff-falls during evaluation windows).
  3. The paired version has higher power than the unpaired Mann-Whitney
     when the pairing structure is real, which it is here.

We complement the p-value with **Cohen's d for paired samples** as the
effect size, because for n=10 'significant' is cheap and 'how big is
the effect' is what actually matters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------
#  Effect size: Cohen's d for paired samples
# --------------------------------------------------------------------------

def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference for paired samples.

    d = mean(diff) / std(diff)

    Rule of thumb (Cohen 1988):
      |d| < 0.2  : negligible
      0.2 - 0.5  : small
      0.5 - 0.8  : medium
      > 0.8      : large
    """
    diff = a - b
    return float(diff.mean() / diff.std(ddof=1))


# --------------------------------------------------------------------------
#  Per-seed final-window returns: the unit of analysis
# --------------------------------------------------------------------------

def per_seed_final_return(returns: np.ndarray, last_n: int = 200) -> np.ndarray:
    """Collapse a (seeds, episodes) array to (seeds,) by averaging the last
    `last_n` episodes per seed. This is the standard RL evaluation window."""
    return returns[:, -last_n:].mean(axis=1)


# --------------------------------------------------------------------------
#  Main comparison driver
# --------------------------------------------------------------------------

def paired_comparison(name_a: str, returns_a: np.ndarray,
                      name_b: str, returns_b: np.ndarray,
                      last_n: int = 200) -> dict:
    """Run the full paired comparison and return a dict suitable for JSON."""
    a = per_seed_final_return(returns_a, last_n)
    b = per_seed_final_return(returns_b, last_n)

    # Two-sided Wilcoxon signed-rank: H0 = no difference, H1 = some difference.
    # We also report the one-sided 'a > b' alternative.
    w_two = stats.wilcoxon(a, b, alternative="two-sided", zero_method="zsplit")
    w_greater = stats.wilcoxon(a, b, alternative="greater", zero_method="zsplit")

    d = cohens_d_paired(a, b)

    return {
        "name_a": name_a,
        "name_b": name_b,
        "n_pairs": int(len(a)),
        "mean_a": float(a.mean()),
        "std_a": float(a.std(ddof=1)),
        "mean_b": float(b.mean()),
        "std_b": float(b.std(ddof=1)),
        "mean_diff": float((a - b).mean()),
        "wilcoxon_W": float(w_two.statistic),
        "p_value_two_sided": float(w_two.pvalue),
        "p_value_one_sided_a_greater": float(w_greater.pvalue),
        "cohens_d_paired": d,
        "per_seed_a": a.tolist(),
        "per_seed_b": b.tolist(),
    }


def main():
    runs = np.load("hw2_sarsa_qlearning/results/four_algos.npz")

    comparisons = []

    # --- CliffWalking: the canonical SARSA vs Q-Learning matchup -----------
    print("=" * 70)
    print("CLIFFWALKING — paired Wilcoxon signed-rank tests")
    print("=" * 70)
    for pair in [
        ("sarsa", "q_learning"),
        ("expected_sarsa", "q_learning"),
        ("double_q", "q_learning"),
        ("sarsa", "expected_sarsa"),
    ]:
        a, b = pair
        ra = runs[f"cliff_{a}_returns"]
        rb = runs[f"cliff_{b}_returns"]
        result = paired_comparison(a, ra, b, rb)
        result["env"] = "cliff"
        comparisons.append(result)
        print(f"\n  {a:>16s}  vs  {b:<16s}")
        print(f"    means    : {result['mean_a']:+8.3f}  vs  {result['mean_b']:+8.3f}")
        print(f"    diff     : {result['mean_diff']:+8.3f}")
        print(f"    Wilcoxon : W = {result['wilcoxon_W']:.1f}, "
              f"p (two-sided) = {result['p_value_two_sided']:.4g}")
        print(f"    Cohen's d: {result['cohens_d_paired']:+.3f}")

    # --- Taxi: should show NO significant differences ---------------------
    print("\n" + "=" * 70)
    print("TAXI — same tests (expected: all p > 0.05, all algos converge)")
    print("=" * 70)
    for pair in [
        ("sarsa", "q_learning"),
        ("expected_sarsa", "double_q"),
    ]:
        a, b = pair
        ra = runs[f"taxi_{a}_returns"]
        rb = runs[f"taxi_{b}_returns"]
        result = paired_comparison(a, ra, b, rb)
        result["env"] = "taxi"
        comparisons.append(result)
        print(f"\n  {a:>16s}  vs  {b:<16s}")
        print(f"    means    : {result['mean_a']:+8.3f}  vs  {result['mean_b']:+8.3f}")
        print(f"    Wilcoxon : p (two-sided) = {result['p_value_two_sided']:.4g}")
        print(f"    Cohen's d: {result['cohens_d_paired']:+.3f}")

    out = Path("hw2_sarsa_qlearning/results/statistical_tests.json")
    out.write_text(json.dumps(comparisons, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
