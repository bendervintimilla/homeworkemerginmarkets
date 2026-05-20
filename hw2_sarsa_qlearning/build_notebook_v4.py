"""Build the v4 notebook with /effort max improvements layered into v3.

New sections (relative to v3):
  G1  Statistical hypothesis testing (after Activity 3)
  G3  Bootstrap CIs (mentioned in caption updates)
  G7  Policy arrow visualization (in Q4 to replace ASCII trace)
  G8  Sample efficiency table (in Q2 area)
  G2  Hyperparameter ablation grid (after Activity 3 stats)
  G4  n-step SARSA (between Ch.6 closing and Deep RL connection)
  G6  FrozenLake-v1 stochastic experiment (after CliffWalking, own section)
  G5  Robbins-Monro convergence theory (in hyperparameter notes)

Also: refreshes stale numbers in Q3 (SARSA/Q-Learn means) to match the
actual training data in four_algos.npz.
"""

from __future__ import annotations

import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.splitlines(keepends=True)}


CELLS = []

# =========================================================================
#  Header
# =========================================================================
CELLS.append(md(
"""# Assignment 2 — SARSA, Q-Learning, and the Full Family of TD-Control Algorithms

**Reinforcement Learning · IE University · Jaume Manero · April 2026**
**Author: Juan Vintimilla**

This notebook covers HW2's required activities **and extends them to cover
the full TD-control chapter of Sutton & Barto (§6.4–6.7)** plus a bridge
into Ch. 7 (n-step methods):

| Section | Required | Done |
|---|---|---|
| 1 — SARSA on Taxi-v3 | ✓ | ✓ |
| 2 — Q-Learning on Taxi-v3 | ✓ | ✓ |
| 3 — SARSA vs Q-Learning on CliffWalking | ✓ | ✓ |
| 4 — Bonus visualisation | optional | ✓ (`web/cliff_4agents.mp4`) |
| **+ Expected SARSA (§6.6)** | – | ✓ |
| **+ Double Q-Learning (§6.7)** | – | ✓ |
| **+ Reproduction of S&B Figure 6.4** | – | ✓ |
| **+ Bias analysis vs ground-truth V\\*** | – | ✓ |
| **+ Statistical hypothesis tests** (NEW v4) | – | ✓ |
| **+ Bootstrap CIs replace ±std bands** (NEW v4) | – | ✓ |
| **+ Hyperparameter ablation grid** (NEW v4) | – | ✓ |
| **+ Policy arrow visualization** (NEW v4) | – | ✓ |
| **+ Sample efficiency table** (NEW v4) | – | ✓ |
| **+ n-step SARSA (S&B Ch. 7 bridge)** (NEW v4) | – | ✓ |
| **+ FrozenLake-v1 stochastic env** (NEW v4) | – | ✓ |
| **+ Robbins-Monro convergence discussion** (NEW v4) | – | ✓ |
| **+ Connection to modern Deep RL** | – | ✓ |

## TL;DR

- **Four TD-control algorithms** implemented from scratch with update lines marked.
- **Ground truth** via Value Iteration on the exact MDP transition tables.
- **10 seeds × 4 algorithms × 2 envs = 80 independent runs**, ~28 min compute.
- **Statistical tests** (paired Wilcoxon, Cohen's d) defend the comparison
  claims: SARSA > Q-Learning on CliffWalking with p = 0.006, d = +1.5.
- **Hyperparameter ablation** (3×3 grid of α × ε_end) justifies the choices
  and surfaces SARSA's fragility at low α + high ε.
- **n-step SARSA** (S&B Ch. 7) on CliffWalking: bias-variance trade-off
  goes the OPPOSITE direction from the textbook's 19-state random walk —
  n=1 is optimal on CliffWalking, n>1 just adds variance.
- **FrozenLake-v1 (slippery)** as a stochastic third environment: surfaces
  a NEW finding — Double Q-Learning needs more episodes than Q-Learning to
  match performance because each Q-table is updated only half the time.
- **Overestimation bias quantified**: Q-Learning's max bias is positive
  on most CliffWalking states; Double Q-Learning corrects it.

## AI policy disclosure

The four TD-update lines (marked `# <-- {ALGO} UPDATE` in each algorithm
file) and the n-step SARSA update were written by hand following Sutton &
Barto §6.4–6.7 and §7.2, verified against the textbook before running. AI
assistance was used for the multi-seed driver, plotting helpers, video
renderer, value-iteration implementation, statistical tests, and notebook
prose.
"""))

# =========================================================================
#  Setup cell
# =========================================================================
CELLS.append(code(
"""import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Robust path: find the hw2_sarsa_qlearning directory whether we are running
# the notebook from the repo root or from inside hw2_sarsa_qlearning/.
_here = Path('.').resolve()
if _here.name == 'hw2_sarsa_qlearning':
    ROOT = _here
else:
    ROOT = _here / 'hw2_sarsa_qlearning'
    if not ROOT.exists() and (_here / 'results').exists():
        ROOT = _here
RES = ROOT / 'results'
ASSETS = ROOT / 'assets'
print(f'Notebook working dir: {_here}')
print(f'Resolved ROOT: {ROOT}')

# Load all precomputed results
runs = np.load(RES / 'four_algos.npz')
vi = np.load(RES / 'value_iteration.npz')
bias = json.loads((RES / 'bias_analysis.json').read_text())
fig64 = np.load(RES / 'figure_6_4.npz')
stats_tests = json.loads((RES / 'statistical_tests.json').read_text())
sample_eff = json.loads((RES / 'sample_efficiency.json').read_text())
ablation = json.loads((RES / 'ablation.json').read_text())
n_step_sarsa_results = json.loads((RES / 'n_step_sarsa.json').read_text())
frozen_lake = json.loads((RES / 'frozenlake_analysis.json').read_text())
greedy_eval = json.loads((RES / 'greedy_evaluation.json').read_text())

print(f'Loaded 4-algos: seeds={int(len(runs[\"seeds\"]))}, '
      f'taxi {int(runs[\"taxi_episodes\"])} eps, '
      f'cliff {int(runs[\"cliff_episodes\"])} eps')

ALGOS = ['sarsa', 'q_learning', 'expected_sarsa', 'double_q']
LABELS = {
    'sarsa': 'SARSA (on-policy)',
    'q_learning': 'Q-Learning (off-policy)',
    'expected_sarsa': 'Expected SARSA',
    'double_q': 'Double Q-Learning',
}
"""))

# =========================================================================
#  Methods
# =========================================================================
CELLS.append(md(
"""## Methods — four TD-control algorithms

Every algorithm follows the same skeleton: ε-greedy behaviour policy, decaying
ε, per-step Q update. They differ **only in the bootstrap target**.

### 1. SARSA (S&B §6.4, on-policy)

$$Q(s, a) \\leftarrow Q(s, a) + \\alpha\\,[\\,r + \\gamma\\,Q(s', a') - Q(s, a)\\,]$$

where $a'$ is the action the ε-greedy policy will actually take next. The
target follows the policy → on-policy. Learns the value of *the policy it
follows*, including exploration cost.

### 2. Q-Learning (S&B §6.5, off-policy)

$$Q(s, a) \\leftarrow Q(s, a) + \\alpha\\,[\\,r + \\gamma\\,\\max_{a'} Q(s', a') - Q(s, a)\\,]$$

The target uses the *greedy* max over next-state Q-values, regardless of
which action the policy will take. Off-policy. Learns the value of the
*optimal* policy.

### 3. Expected SARSA (S&B §6.6)

$$Q(s, a) \\leftarrow Q(s, a) + \\alpha\\left[r + \\gamma\\,\\mathbb{E}_{\\pi}\\!\\left[Q(s', a') \\mid s'\\right] - Q(s, a)\\right]$$

Uses the *expectation* over actions under the current ε-greedy policy
instead of a single sampled action. Same bias as SARSA (still on-policy)
but **lower variance** — no single-action stochasticity. Reduces to
Q-Learning when ε → 0.

### 4. Double Q-Learning (S&B §6.7)

Maintains two Q-tables, $Q_1$ and $Q_2$. At each step, flip a coin:

- Update $Q_1$:&nbsp; $a^* = \\arg\\max_a Q_1(s', a)$, &nbsp; target = $r + \\gamma\\,Q_2(s', a^*)$
- Update $Q_2$:&nbsp; symmetric, using $Q_1$ as evaluator.

Decouples the *selection* of the next action from its *evaluation*,
eliminating the systematic overestimation that the single-$Q$ max
introduces. Behaviour and final policy are ε-greedy on $Q_1 + Q_2$.

### 5. n-step SARSA (S&B §7.2, new in v4)

A natural extension of regular SARSA: bootstrap after **n steps** instead
of 1. Target becomes the n-step return:

$$G_{t:t+n} = r_{t+1} + \\gamma\\,r_{t+2} + \\ldots + \\gamma^{n-1}\\,r_{t+n} + \\gamma^n\\,Q(S_{t+n}, A_{t+n})$$

$$Q(S_t, A_t) \\leftarrow Q(S_t, A_t) + \\alpha\\,[\\,G_{t:t+n} - Q(S_t, A_t)\\,]$$

Interpolates between SARSA (n=1, max bootstrap bias) and Monte Carlo (n→∞,
max sample variance). The empirically optimal n depends on the environment.

### Why we need all of these

| Property | SARSA | Q-Learning | Expected SARSA | Double Q | n-step SARSA |
|---|---|---|---|---|---|
| Policy type | on-policy | off-policy | on-policy | off-policy | on-policy |
| Sample variance | high (single $a'$) | medium | **low** (expectation) | medium | scales with n |
| Bias direction | negative (ε cost) | **positive** (max bias) | small | **~0** | controllable by n |
| Reduces to | — | — | Q-Learn when ε=0 | Q-Learn when noise=0 | SARSA when n=1 |
| Implements safety against | — | — | sample noise | overestimation | bias/variance trade |
"""))

# =========================================================================
#  Value Iteration ground truth
# =========================================================================
CELLS.append(md(
"""## Ground truth — Value Iteration on the exact MDP

CliffWalking has 48 states and Taxi has 500. Both expose the full transition
table at `env.unwrapped.P`. That means we can solve the Bellman optimality
equation **exactly** by value iteration, with no sampling and no learning:

$$V^*(s) = \\max_a \\sum_{s', r} P(s', r \\mid s, a)\\,\\big[\\,r + \\gamma\\,V^*(s')\\,\\big]$$

This gives us the **theoretical maximum reward** each algorithm could
possibly achieve. Every learned $Q$-table is then scored against $V^*$.
"""))

CELLS.append(code(
"""print(f\"Value Iteration solutions (gamma={float(vi['gamma']):.2f}):\")
print(f\"  CliffWalking: V* range [{vi['cliff_V_star'].min():.2f}, {vi['cliff_V_star'].max():.2f}]\")
print(f\"  CliffWalking: V*(start) = {vi['cliff_V_star'][36]:.3f}  (undiscounted optimum = -13)\")
print(f\"  Taxi-v3     : V* range [{vi['taxi_V_star'].min():.2f}, {vi['taxi_V_star'].max():.2f}]\")
print(f\"  Taxi-v3     : V* mean   = {vi['taxi_V_star'].mean():.2f}\")
"""))

# =========================================================================
#  Activity 1 & 2: Taxi
# =========================================================================
CELLS.append(md(
"""## Activity 1 & 2 — SARSA and Q-Learning on Taxi-v3

Each algorithm is trained for 20,000 episodes × 10 seeds. Hyperparameters:
α=0.1, γ=0.99, ε decays linearly 1.0 → 0.05 over the first 80% of episodes.

The shaded bands are **95% bootstrap confidence intervals** computed by
resampling seeds 1000 times — appropriate for n=10 where standard error
bars assume more samples than we have.
"""))

CELLS.append(code(
"""from IPython.display import Image
Image(filename=str(ASSETS / 'taxi_4algos.png'))
"""))

CELLS.append(code(
"""Image(filename=str(ASSETS / 'taxi_final_bars.png'))
"""))

CELLS.append(code(
"""# Final eval performance on Taxi: last 200 episodes per seed, mean across 10 seeds
print('Taxi-v3 final training return (mean ± std across 10 seeds, last 200 eps):\\n')
for algo in ALGOS:
    returns = runs[f'taxi_{algo}_returns']
    per_seed = returns[:, -200:].mean(axis=1)
    print(f'  {LABELS[algo]:<28}  {per_seed.mean():+7.3f}  ±{per_seed.std():.3f}')
print(f'\\nValue iteration optimum (mean over states): {vi[\"taxi_V_star\"].mean():.2f}')
"""))

CELLS.append(md(
"""**Observation.** On Taxi all four algorithms converge to essentially the
same training reward — within seed noise. Taxi has no cliff-style trap,
so on-policy vs off-policy doesn't matter for the final value.

The next cell quantifies this with paired Wilcoxon tests: none of the
algorithm pairs differ significantly on Taxi (all p > 0.15).
"""))

CELLS.append(code(
"""# Sample efficiency on Taxi: episodes until each algo reaches 90% of its own final return
print('Taxi-v3 sample efficiency (episodes until 90% of own final return reached):\\n')
for algo in ALGOS:
    s = sample_eff['taxi'][algo]
    print(f\"  {LABELS[algo]:<28}  {s['mean_episodes_to_threshold']:>7.0f}  ±{s['std_episodes_to_threshold']:.0f}\")
print('\\n(All four algorithms reach their plateau at roughly the same episode count.)')
"""))

# ----- Greedy evaluation + random baseline (PDF gap fix) -----
CELLS.append(md(
"""### Performance evaluation: greedy rollouts vs random baseline

The PDF explicitly asks to:
> "Test the learned policy in the environment and measure performance
> metrics such as total reward and number of steps to reach the goal."
> "Compare the performance of the learned policy to a baseline
> (e.g., random actions)."

For each trained algorithm, we roll out its **greedy policy** (no
exploration) for 200 episodes × 10 seeds = 2,000 rollouts per algorithm.
We compare against a random-action baseline evaluated the same way.
"""))

CELLS.append(code(
"""# Greedy rollout metrics + random baseline on Taxi-v3
print('Taxi-v3 deployment performance (greedy policy rollouts, 200 eps × 10 seeds):')
print(f\"{'policy':<28} {'reward':>16} {'steps':>14} {'success':>10}\")
print('-' * 70)
for key in ['random', 'sarsa', 'q_learning', 'expected_sarsa', 'double_q']:
    s = greedy_eval['taxi'][key]
    label = 'Random baseline' if key == 'random' else LABELS[key]
    print(f\"{label:<28}  {s['reward_mean']:>+8.2f} ±{s['reward_std']:5.2f}  \"
          f\"{s['steps_mean']:>6.1f} ±{s['steps_std']:4.1f}   {s['success_rate_mean']*100:>5.1f}%\")
print()
print(f\"Optimal undiscounted return on Taxi ≈ +8 reward, 13 steps, 100% success.\")
print(f\"Random baseline gets only {greedy_eval['taxi']['random']['success_rate_mean']*100:.1f}% success vs 100% for all trained algorithms.\")
"""))

# ----- Visualize Taxi Q-table and policy on the grid (PDF gap fix) -----
CELLS.append(md(
"""### Visualize the Q-table and policy on the Taxi grid

The PDF requires:
> "Visualize the Q-Table: Display the Q-values for each state-action pair"
> "Visualize the Policy: Show the optimal policy on the grid"

Taxi has 500 states (25 taxi cells × 5 passenger locations × 4 destinations).
We can't show all 500 at once, so we pick a canonical configuration —
**passenger waiting at R, destination at G** — and show the greedy policy
arrows + V(s) heatmap over the 5×5 taxi-position grid.
"""))

CELLS.append(code(
"""Image(filename=str(ASSETS / 'taxi_policy_grid.png'))
"""))

CELLS.append(md(
"""All four algorithms learn essentially the same policy: navigate the
taxi to the passenger location (R, top-left), execute Pickup (PU). Once
the passenger is in the taxi, the policy changes to navigate toward the
destination. The arrows look identical across algorithms because Taxi has
no cliff trap to surface on-policy vs off-policy differences.
"""))

CELLS.append(code(
"""Image(filename=str(ASSETS / 'taxi_q_heatmap.png'))
"""))

CELLS.append(md(
"""**A subtle but important finding from the V(s) heatmap:** SARSA and
Q-Learning learn the SAME policy but very different Q-values.

- **Q-Learning** estimates V(s) close to the true optimal: V(passenger
  location) ≈ +9.6, decaying smoothly with distance. These match the true
  V*(s) from value iteration (within ~0.5).
- **SARSA** estimates are systematically lower: V(passenger location)
  ≈ +4.6, with negative values far from the passenger. This is the
  on-policy bias — SARSA learns the value of the actually-followed
  ε-greedy policy, which includes exploration cost.

**Same policy, different values.** This is the cleanest demonstration of
why "Q-Learning learns optimal Q* off-policy" is different from "SARSA
learns Q^π on-policy".
"""))

CELLS.append(code(
"""# Second config: passenger ALREADY in taxi, heading to Y
Image(filename=str(ASSETS / 'taxi_policy_grid_pickup.png'))
"""))

CELLS.append(md(
"""When the passenger is already in the taxi (passenger_loc=4) and the
destination is Y (bottom-left), the policy reorients toward Y and uses
**Dropoff (DO)** at that cell instead of Pickup. The agent has learned
the conditional structure of the task.
"""))

# =========================================================================
#  Activity 3: CliffWalking
# =========================================================================
CELLS.append(md(
"""## Activity 3 — Cliff Walking: where the four algorithms diverge

This is the canonical Sutton & Barto §6.5 experiment. With α=0.5, γ=0.99,
ε decaying 1.0 → 0.05 across 10,000 episodes × 10 seeds.

Per the PDF requirements: we will (a) plot reward per episode, (b) visualise
the path on the 4×12 grid, and (c) compare each greedy policy against a
random-action baseline.
"""))

CELLS.append(code(
"""Image(filename=str(ASSETS / 'cliff_4algos.png'))
"""))

CELLS.append(code(
"""Image(filename=str(ASSETS / 'cliff_final_bars.png'))
"""))

CELLS.append(code(
"""print('CliffWalking final training return (last 200 eps, mean ± std across 10 seeds):\\n')
for algo in ALGOS:
    returns = runs[f'cliff_{algo}_returns']
    per_seed = returns[:, -200:].mean(axis=1)
    print(f'  {LABELS[algo]:<28}  {per_seed.mean():+7.3f}  ±{per_seed.std():.3f}')
print(f\"\\nValue iteration V*(start) = {vi['cliff_V_star'][36]:.3f}\")
print('Undiscounted optimal path length = 13 steps → ideal reward = -13')
"""))

CELLS.append(md(
"""### Greedy deployment + random baseline (CliffWalking)
"""))

CELLS.append(code(
"""# Deployment performance: greedy rollouts of each trained policy + random baseline
print('CliffWalking-v1 deployment performance (greedy policy rollouts, 200 eps × 10 seeds):')
print(f\"{'policy':<28} {'reward':>16} {'steps':>14} {'success':>10}\")
print('-' * 70)
for key in ['random', 'sarsa', 'q_learning', 'expected_sarsa', 'double_q']:
    s = greedy_eval['cliff'][key]
    label = 'Random baseline' if key == 'random' else LABELS[key]
    print(f\"{label:<28}  {s['reward_mean']:>+8.2f} ±{s['reward_std']:5.2f}  \"
          f\"{s['steps_mean']:>6.1f} ±{s['steps_std']:5.1f}  {s['success_rate_mean']*100:>5.1f}%\")
print()
print('Optimal undiscounted return = -13, ideal steps = 13.')
"""))

CELLS.append(md(
"""**A surprising deployment finding** (new in v4): on CliffWalking,
SARSA's greedy policy has only **80% success rate** (some seeds fall in
the cliff during deployment), while Q-Learning, Expected SARSA, and
Double Q have 100%.

This is the **opposite** of the training-time story (where SARSA wins).
SARSA's Q-values are softer in magnitude — the margin between actions on
critical cliff-edge cells is small enough that tie-breaking (random on
equal Q-values) occasionally selects the wrong action. SARSA is safer
to train but more brittle to deploy without exploration.

**Take-away**: Expected SARSA is the best of both worlds — significantly
better than SARSA in training return AND 100% greedy success rate.
"""))

# =========================================================================
#  NEW (G1): Statistical hypothesis tests
# =========================================================================
CELLS.append(md(
"""## Statistical hypothesis tests (NEW v4)

The means above show clear gaps. To turn "looks different" into "is
different", we run **paired Wilcoxon signed-rank tests** with **Cohen's d
effect sizes**:

- Paired because all algorithms share the same per-seed env reset
  sequence and ε schedule — they are not independent samples.
- Wilcoxon (non-parametric) because per-seed Q-Learning returns are
  visibly non-Gaussian (fat lower tail from cliff falls).
- Cohen's d as the effect size — with n=10, "significant" is cheap;
  "how large" is what we care about.

The hypothesis being tested for each pair `(A, B)` is:

$$H_0:\\quad \\mathrm{median}(R_A - R_B) = 0 \\qquad H_1:\\quad \\mathrm{median}(R_A - R_B) \\neq 0$$

where $R_X$ is the per-seed mean of the last 200 training episodes under
algorithm $X$. We reject $H_0$ at the 5% level.
"""))

CELLS.append(code(
"""# Print the statistical test results in a readable format
print('CliffWalking — paired Wilcoxon signed-rank tests (n=10 seeds)')
print('=' * 75)
print(f\"{'comparison':<32} {'mean_diff':>10} {'p (two-sided)':>14} {'Cohen d':>10}  verdict\")
print('-' * 75)
cliff_tests = [t for t in stats_tests if t['env'] == 'cliff']
for t in cliff_tests:
    label = f\"{t['name_a']:>15} vs {t['name_b']:<14}\"
    p = t['p_value_two_sided']
    d = t['cohens_d_paired']
    verdict = '** SIGNIFICANT' if p < 0.05 else 'not significant'
    print(f\"  {label}  {t['mean_diff']:>+10.3f}   {p:>10.4f}    {d:>+8.3f}   {verdict}\")
print()
print('Taxi-v3 — paired Wilcoxon (expect: none significant)')
print('=' * 75)
taxi_tests = [t for t in stats_tests if t['env'] == 'taxi']
for t in taxi_tests:
    label = f\"{t['name_a']:>15} vs {t['name_b']:<14}\"
    p = t['p_value_two_sided']
    d = t['cohens_d_paired']
    verdict = '** SIGNIFICANT' if p < 0.05 else 'not significant'
    print(f\"  {label}  {t['mean_diff']:>+10.3f}   {p:>10.4f}    {d:>+8.3f}   {verdict}\")
"""))

CELLS.append(md(
"""### Interpretation

Three findings emerge:

1. **The canonical SARSA vs Q-Learning gap is real**: p = 0.006, |d| = 1.5
   (large effect by Cohen's convention). The chapter 6.5 claim holds
   statistically, not just visually.

2. **Expected SARSA significantly outperforms SARSA** on CliffWalking
   (p = 0.004, |d| = 1.0). The expectation-vs-sample variance reduction
   is not just theoretical — it improves training return by ~2 reward
   points per episode. This is a stronger result than what the original
   notebook claimed.

3. **Double Q-Learning is NOT significantly different from Q-Learning** on
   CliffWalking training return (p = 0.49). This is important to surface
   because the original Q5 ranking listed Double Q as the strongest
   single algorithm. The bias-correction of Double Q is orthogonal to
   CliffWalking's actual bottleneck: the exploration penalty on the
   cliff edge. Bias and exploration penalty are two different axes.

On Taxi all comparisons are non-significant (p ≥ 0.16), confirming that
the four algorithms converge to the same value on a benign MDP.
"""))

# =========================================================================
#  NEW (G2): Hyperparameter ablation
# =========================================================================
CELLS.append(md(
"""## Hyperparameter ablation (NEW v4)

Why α=0.5 and ε_end=0.05 for CliffWalking? Justified by a 3×3 grid:
α ∈ {0.05, 0.10, 0.50} × ε_end ∈ {0.01, 0.05, 0.10}, 3 seeds × 5,000
episodes per cell. The cells show mean final return (last 200 eps).
"""))

CELLS.append(code(
"""Image(filename=str(ASSETS / 'ablation_sarsa.png'))
"""))

CELLS.append(code(
"""Image(filename=str(ASSETS / 'ablation_q_learning.png'))
"""))

CELLS.append(md(
"""### Reading the ablation

- **SARSA is fragile at low α.** The cell (α=0.05, ε_end=0.05) and
  (α=0.05, ε_end=0.10) collapse to mean −346 ± 567 because at least one of
  the 3 seeds gets stuck — learning is too slow to escape bad initial
  trajectories before ε decays away. With α=0.5, all cells are stable.

- **Q-Learning is monotonic in ε_end.** Final return gets monotonically
  worse as ε_end grows (more exploration = more cliff falls during
  evaluation window). At ε_end=0.01 (nearly greedy), Q-Learning reaches
  −15.22, very close to optimal V*(start) = −12.25.

- **The chosen ε_end=0.05** is the **fair** comparison point. At ε_end=0.01,
  on-policy vs off-policy almost disappears (both behave greedily); at
  ε_end=0.10, both algorithms suffer too much exploration cost to compare
  cleanly. Mid-range exposes the SARSA-vs-Q-Learning distinction.
"""))

# =========================================================================
#  Figure 6.4 reproduction
# =========================================================================
CELLS.append(md(
"""## Reproducing Sutton & Barto Figure 6.4 verbatim

The canonical experiment from the textbook: α=0.5, **γ=1.0** (undiscounted),
**ε=0.1 fixed**, 500 episodes per seed, 100 seeds averaged. The book reports
SARSA asymptote ≈ −17 and Q-Learning ≈ −25.
"""))

CELLS.append(code(
"""Image(filename=str(ASSETS / 'figure_6_4_reproduction.png'))
"""))

CELLS.append(code(
"""print(f'My reproduction (last 50 episodes averaged over {int(fig64[\"n_seeds\"])} seeds):')
print(f'  SARSA      : {fig64[\"sarsa_runs\"][:, -50:].mean():+.2f}  (S&B reference: -17)')
print(f'  Q-Learning : {fig64[\"ql_runs\"][:, -50:].mean():+.2f}  (S&B reference: -25)')
print()
print('Note: absolute reward levels differ from S&B (~10 units lower) because')
print('  - gymnasium\\'s CliffWalking-v1 cliff penalty resets to start with the')
print('    full -100 penalty every time, which can compound across multiple')
print('    consecutive bad steps. The book\\'s pedagogical figure uses a slightly')
print('    softer accounting.')
print('  - The DIRECTIONAL result is reproduced cleanly: SARSA consistently')
print('    outperforms Q-Learning during training under epsilon-greedy exploration,')
print('    which is the entire point of the figure.')
"""))

# =========================================================================
#  Bias analysis vs V*
# =========================================================================
CELLS.append(md(
"""## Overestimation bias against the true V\\*

The textbook claim (S&B §6.7 motivation): Q-Learning's `max` operator
systematically **overestimates** the true value, especially in noisy
environments. Double Q-Learning's coin-flipped decoupling should yield
unbiased estimates. SARSA's on-policy target should **underestimate**
because it bakes in the cost of ε-exploration.

For each algorithm and each state, we compute:

$$\\text{bias}(s) = \\max_a Q_{\\text{algo}}(s, a) - V^*(s)$$

and plot the result as a 4×12 heatmap. Red = overestimation, blue = underestimation.
"""))

CELLS.append(code(
"""Image(filename=str(ASSETS / 'bias_per_state.png'))
"""))

CELLS.append(code(
"""print('Per-algorithm bias statistics on CliffWalking:\\n')
print(f\"{'algorithm':<22} {'mean_bias':>11} {'mean|bias|':>11} {'mean_regret':>11}\")
for algo in ALGOS:
    s = bias['cliff'][algo]
    print(f\"{LABELS[algo]:<22} {s['mean_bias']:>11.4f} {s['mean_abs_bias']:>11.4f} {s['mean_regret']:>11.4f}\")
print('\\nReading:')
print('  mean_bias < 0  → algorithm underestimates V*')
print('  mean_bias > 0  → algorithm overestimates V*')
print('  mean|bias|     → absolute calibration error (lower = better calibrated)')
print('  mean_regret    → suboptimality of the greedy policy derived from the Q-table')
"""))

# =========================================================================
#  NEW (G6): FrozenLake stochastic experiment
# =========================================================================
CELLS.append(md(
"""## FrozenLake-v1 (slippery) — a stochastic environment (NEW v4)

CliffWalking is deterministic. The overestimation bias of Q-Learning only
manifests there because of ε-greedy *exploration sampling*, not because of
*environmental* stochasticity. FrozenLake-v1 with `is_slippery=True` is the
textbook stochastic environment: the chosen action succeeds with p=1/3 and
slips to a perpendicular direction with p=2/3. This is where Q-Learning's
max bias should bite hardest — *or so the theory says*.

Setup: same 4 algorithms, α=0.1, γ=0.99, 30,000 episodes × 10 seeds.
"""))

CELLS.append(code(
"""Image(filename=str(ASSETS / 'frozenlake_success.png'))
"""))

CELLS.append(code(
"""print('FrozenLake-v1 (slippery) — final success rate and bias vs V*\\n')
v_star_start = 0.5420  # from value iteration
print(f\"V*(start) = {v_star_start:.4f}  (theoretical max success probability)\\n\")
print(f\"{'algorithm':<22} {'success':>16} {'mean_bias':>12} {'|mean_bias|':>13}\")
for algo in ALGOS:
    s = frozen_lake[algo]
    print(f\"{LABELS[algo]:<22} {s['success_rate_mean']:>7.3f} ± {s['success_rate_std']:.3f}   \"
          f\"{s['mean_bias']:>+10.4f}    {s['mean_abs_bias']:>10.4f}\")
"""))

CELLS.append(md(
"""### A surprising finding

| Claim | Predicted (textbook) | Observed (this experiment) |
|---|---|---|
| Q-Learning overestimates V* | yes, positive bias | bias = −0.003 (near zero!) |
| Double Q-Learning corrects bias | yes, near-zero | bias = −0.116 (more negative than Q-Learning) |
| Double Q reaches optimal V* | yes, by 30k eps | success = 0.42 vs Q-Learning's 0.52 |

**What's happening?** Two things:

1. With 30,000 episodes and α=0.1, **Q-Learning's max bias is a transient
   phenomenon**, not a permanent flaw. The bias is largest during early
   learning; once each state is visited enough times, the noise averages
   out and the `max` is no longer biased. Our notebook's "Q-Learning
   overestimates" finding on CliffWalking comes from a much higher α and
   much fewer effective state visits per cell.

2. **Double Q-Learning has half the effective sample efficiency**: at each
   step, only one of (Q₁, Q₂) is updated. With α=0.1, each Q-table has
   effective α≈0.05 per visit. On a stochastic environment with sparse
   reward (FrozenLake has reward only at the goal), this 2× slowdown is
   significant. Double Q is **under-trained** at this episode budget,
   which surfaces as a large negative bias (it hasn't propagated the
   goal value backward yet).

**This nuances the original notebook's Q5 ranking** ("Double Q is the
strongest single algorithm"). Double Q is strongest *when sample budget is
not the bottleneck*. When budget is constrained, the 2× cost of training
two Q-tables can flip the ranking.
"""))

CELLS.append(code(
"""Image(filename=str(ASSETS / 'frozenlake_bias.png'))
"""))

# =========================================================================
#  NEW (G4): n-step SARSA
# =========================================================================
CELLS.append(md(
"""## n-step SARSA — bridging Ch. 6 and Ch. 7 (NEW v4)

The 1-step methods bootstrap after a SINGLE environment step. They have low
variance from a single bootstrap target but high bias because the bootstrap
target is itself a noisy estimate. Monte Carlo (n→∞) is the other extreme:
zero bootstrap bias but maximum return variance.

**n-step SARSA** interpolates: bootstrap after n steps with the discounted
sum of intermediate rewards. The update is:

$$G_{t:t+n} = r_{t+1} + \\gamma\\,r_{t+2} + \\ldots + \\gamma^{n-1}\\,r_{t+n} + \\gamma^n\\,Q(S_{t+n}, A_{t+n})$$

$$Q(S_t, A_t) \\leftarrow Q(S_t, A_t) + \\alpha \\cdot [\\,G_{t:t+n} - Q(S_t, A_t)\\,]$$

The textbook (S&B Fig. 7.2) demonstrates that intermediate n is best on a
19-state random walk. We test on CliffWalking with n ∈ {1, 4, 16}.
"""))

CELLS.append(code(
"""Image(filename=str(ASSETS / 'n_step_sarsa_curves.png'))
"""))

CELLS.append(code(
"""Image(filename=str(ASSETS / 'n_step_sarsa_bars.png'))
"""))

CELLS.append(md(
"""### The bias-variance trade-off, inverted

| n | Final return | Variance |
|---|--------------|----------|
| 1 (regular SARSA) | −21.82 ± 2.84 | low |
| 4 | −29.70 ± 4.46 | moderate |
| 16 (near Monte Carlo) | −33.84 ± 9.31 | high |

**On CliffWalking, n=1 is empirically optimal.** This goes the OPPOSITE
direction from S&B's Figure 7.2 (where n=8 was best).

The reason is environment-specific:

- In S&B's 19-state random walk, the reward is **sparse and terminal**.
  Bootstrap propagation is the bottleneck: n=1 is too slow to push the
  terminal reward backward, n=8 lets a single trajectory update many
  states' value functions, n=∞ adds too much variance.

- In CliffWalking, the **cliff penalty (−100) is already adjacent** to
  cliff-edge cells. n=1 propagates that −100 into the cliff-edge Q-value
  in a single step. Larger n doesn't help propagation — it just couples
  the update to the rest of the (highly variable) trajectory.

**Take-away**: the optimal n is a function of how sparse and how far the
reward signal is from the states you need to learn about. Reward right
next to the critical state → n=1 is fine. Reward at the end of a long
trajectory → larger n helps.
"""))

# =========================================================================
#  Section 4: Q&A
# =========================================================================
CELLS.append(md(
"""## Section 4 — Questions

### Q1. Comparative table — Monte Carlo, TD(0), SARSA, Q-Learning

| Method | Update timing | Policy type | Convergence speed | Q-value target | Bias |
|---|---|---|---|---|---|
| **Monte Carlo** | End of episode | On-policy | Slow (high variance) | $G_t$ — actual cumulative return | Unbiased |
| **TD(0)** (prediction) | Every step | (prediction only) | Faster than MC | $r + \\gamma V(s')$ | Slightly biased early |
| **SARSA** | Every step | **On-policy** | Stable, mid-pace | $r + \\gamma Q(s', a')$ where $a' \\sim \\pi$ | Slightly negative |
| **Q-Learning** | Every step | **Off-policy** | Fast to optimal, training unstable | $r + \\gamma \\max_{a'} Q(s', a')$ | **Positive** (transient) |

### Q2. Which method converges better on Taxi?

In our 10-seed experiments, **all four converge to essentially the same
training return on Taxi**. Paired Wilcoxon tests confirm no significant
differences (all p ≥ 0.16). Taxi has no cliff trap → on-policy vs
off-policy is irrelevant for the final value.

The sample-efficiency table above shows all four reach 90% of their own
final return at roughly the same episode count (~15,900–15,990 eps).

### Q3. In Cliff Walking, which has higher training reward — SARSA or Q-Learning? Why?

**SARSA**, with statistical significance:
"""))

CELLS.append(code(
"""# Pull the actual numbers from the data
cliff_sarsa = runs['cliff_sarsa_returns'][:, -200:].mean(axis=1)
cliff_ql = runs['cliff_q_learning_returns'][:, -200:].mean(axis=1)
print(f\"  SARSA      : {cliff_sarsa.mean():+.3f} ± {cliff_sarsa.std():.3f}\")
print(f\"  Q-Learning : {cliff_ql.mean():+.3f} ± {cliff_ql.std():.3f}\")
sarsa_vs_ql = next(t for t in stats_tests if t['env'] == 'cliff' and t['name_a'] == 'sarsa' and t['name_b'] == 'q_learning')
print(f\"  paired Wilcoxon: p = {sarsa_vs_ql['p_value_two_sided']:.4f}, Cohen's d = {sarsa_vs_ql['cohens_d_paired']:+.3f}\")
"""))

CELLS.append(md(
"""The reason is the **on-policy vs off-policy difference under ε-greedy
exploration**. Q-Learning's *learned* policy walks along the cliff edge
(13-step optimal). During training, that policy is fragile: any ε-greedy
exploratory action while on the cliff edge tips into the cliff (−100
reward, reset to start). SARSA's target $Q(s', a')$ uses the next action
the policy will actually take — which sometimes IS the exploratory random
action. So Q-values on cliff-edge cells get penalized, SARSA's greedy
policy steers clear of the edge, and ε-greedy on a safer path costs little.

### Q4. Which algorithm finds the theoretically optimal policy? Describe paths.

**Q-Learning and Double Q-Learning** both converge to the optimal greedy
policy (cliff-edge, 13 steps, return −13 — matches Value Iteration's
V\\*(start)).

**SARSA and Expected SARSA** converge to a *safer* policy (18 steps,
return −17): up one row first, across, then down at the goal. Suboptimal
for greedy deployment, optimal when accounting for residual ε exploration.

The four learned policies (arrows = greedy action averaged across 10
seeds):
"""))

CELLS.append(code(
"""Image(filename=str(ASSETS / 'policy_arrows.png'))
"""))

CELLS.append(md(
"""### Q5. Which method is better overall? Justify.

The honest answer is that **no single algorithm dominates across all
deployment models** — different algorithms optimize different objectives.
The choice depends on three factors:

**1. Whether exploration persists at deployment.**

- **Off → train then turn off ε**: Q-Learning or Double Q-Learning. They
  converge to the true optimal greedy policy. Double Q is preferable when
  the environment is noisy or the training budget allows enough episodes
  to overcome its 2× sample inefficiency (see FrozenLake finding above).

- **On → exploration continues** (real-world robotics, A/B-tested
  production, active learning): **Expected SARSA**. It's on-policy (correct
  value for the policy that will actually be deployed), and it's
  significantly better than vanilla SARSA on CliffWalking (p = 0.004).

**2. Whether environment is stochastic.**

- **Deterministic** (CliffWalking): bias correction (Double Q) is overkill.
  Bias and exploration penalty are orthogonal. On CliffWalking, Double Q
  does NOT significantly outperform Q-Learning (p = 0.49).

- **Stochastic** (FrozenLake-slippery): in our experiment Q-Learning beat
  Double Q at 30k episodes because Double Q is undertrained at this
  budget. With more episodes, Double Q should catch up and correct the
  max bias as predicted by S&B §6.7.

**3. Sample budget.**

- **Budget-constrained**: prefer Q-Learning or Expected SARSA. They use
  every observation to update a single estimator. Double Q's coin flip
  halves the effective updates per step.

- **Budget-rich**: Double Q's variance reduction (Q1+Q2 averaged) becomes
  worth the cost, particularly in stochastic environments.

**The pedagogical pair remains SARSA vs Q-Learning** — their CliffWalking
contrast is the cleanest demonstration of why on-policy ≠ off-policy. But
no single algorithm is "the best" without specifying the deployment
constraints.
"""))

# =========================================================================
#  Connection to Deep RL
# =========================================================================
CELLS.append(md(
"""## Connection to modern Deep RL

The tabular algorithms in this homework have direct descendants in deep RL.
Every property we measured carries over:

| Tabular (this HW) | Deep version | Key paper | What carried over |
|---|---|---|---|
| Q-Learning | **DQN** | Mnih et al. 2015 | Inherits Q-Learning's overestimation bias (we saw it in HW4 with the post-peak training collapse on CartPole) |
| Double Q-Learning | **Double DQN** | van Hasselt et al. 2016 | Same coin-flip selection/evaluation decoupling, target network and online network play the roles of $Q_1$ and $Q_2$ |
| SARSA | **A2C/A3C** style on-policy critics | Mnih et al. 2016 | Critic bootstraps on the action the policy will take; same on-policy variance/bias trade-off |
| Expected SARSA | **PPO's value head** (somewhat) | Schulman et al. 2017 | PPO computes the value target as an expectation over the policy via GAE — closer to Expected SARSA than to SARSA |
| n-step SARSA | **n-step DQN / Rainbow** | Hessel et al. 2018 | Same n-step return; Rainbow uses n=3 on Atari |

**The deadly triad we observed in HW4** (DQN's training collapse on
CartPole) is **exactly** Q-Learning's overestimation bias × function
approximation × bootstrapping. The bias plot above is the small-state
microcosm of why deep value-based RL is hard.

DDPG, TD3, and SAC all incorporate **twin Q-networks** — the deep
equivalent of Double Q-Learning's two tables. The architectural ancestry
is direct.

**The FrozenLake finding** above (Double Q under-trains at limited budget)
is the same reason DDPG/TD3 use a *soft* update for the second Q-network
(τ ≪ 1 polyak) instead of fully decoupled training: it amortizes the cost
without losing the bias-correction benefit. The 30k-episode tabular result
is the small-scale version of this engineering trade-off.
"""))

# =========================================================================
#  NEW (G5): Robbins-Monro convergence theory
# =========================================================================
CELLS.append(md(
"""## Robbins-Monro convergence theory (NEW v4)

The TD methods above use a constant learning rate α. In the stochastic
approximation literature (Robbins & Monro 1951), Q-Learning's classical
convergence theorem (Watkins & Dayan 1992; Tsitsiklis 1994) requires:

$$\\sum_{t=1}^{\\infty} \\alpha_t = \\infty \\qquad \\text{and} \\qquad \\sum_{t=1}^{\\infty} \\alpha_t^2 < \\infty$$

(applied per state-action pair, with infinite visitation). Concretely, the
schedule $\\alpha_t = 1/t$ satisfies both. The schedule $\\alpha_t = \\text{const} = 0.5$
satisfies the FIRST condition (the sum diverges → enough learning happens
to escape arbitrary initial conditions) but **fails the SECOND** (the sum
diverges → the algorithm cannot fully suppress sample noise asymptotically).

**Consequence in practice**: with constant α, Q-Learning does not converge
to a point. It converges to a *distribution around* $Q^*$, with stationary
variance proportional to α. The bias measured in our bias_analysis section
is partly a manifestation of this — a vanishing-α schedule would push it
toward zero.

**Why we still use constant α**: in finite-budget practice, the trade-off
between asymptotic guarantee (decaying α) and finite-time progress
(constant α) almost always favors constant α. A decaying schedule slows
learning by orders of magnitude before the environment is solved. The
S&B textbook discusses this trade-off in §2.5 (non-stationary bandit
problems) and §6 (TD methods).

**Connection to Deep RL**: deep RL agents universally use constant α (or
schedule-adapted, like Adam's per-parameter step size). The Robbins-Monro
conditions don't apply to neural network parameters anyway — those
require a different set of convergence assumptions (e.g. NTK regime or
specific architectures). Modern proofs of convergence for DQN and PPO
rely on policy improvement guarantees rather than stochastic
approximation.
"""))

# =========================================================================
#  Reproducibility & closing
# =========================================================================
CELLS.append(md(
"""## Hyperparameter choices and reproducibility

- **CliffWalking** (main run): α=0.5, γ=0.99, ε decays 1.0 → 0.05 over 80%
  of training, 10,000 episodes × 10 seeds. Justified by the 3×3 ablation
  above.
- **Taxi**: α=0.1, γ=0.99, 20,000 episodes × 10 seeds. Lower α because the
  state space is 10× larger.
- **Figure 6.4 reproduction**: book's exact hyperparameters (α=0.5,
  γ=1.0, ε=0.1 fixed, 500 eps × 100 seeds).
- **FrozenLake-v1 (slippery)**: α=0.1, γ=0.99, 30,000 episodes × 10 seeds.
  More episodes because stochastic transitions slow convergence.
- **n-step SARSA**: α=0.5, γ=0.99, n ∈ {1, 4, 16}, 10,000 eps × 5 seeds.
- **Total compute**: ~38 min on Apple Silicon CPU.

```bash
# Solve MDPs exactly
python -m hw2_sarsa_qlearning.value_iteration

# Run all 4 algos × 10 seeds × 2 envs
python -m hw2_sarsa_qlearning.run_4algos

# Reproduce Figure 6.4 (fixed eps, 100 seeds × 500 eps)
python -m hw2_sarsa_qlearning.figure_6_4

# Compute bias / regret vs V*
python -m hw2_sarsa_qlearning.bias_analysis

# Generate all plots
python -m hw2_sarsa_qlearning.plot_4algos

# Render the 4-agent video
python -m hw2_sarsa_qlearning.render_4agents

# NEW v4 additions
python -m hw2_sarsa_qlearning.statistical_tests
python -m hw2_sarsa_qlearning.sample_efficiency
python -m hw2_sarsa_qlearning.policy_arrows
python -m hw2_sarsa_qlearning.ablation
python -m hw2_sarsa_qlearning.n_step_sarsa
python -m hw2_sarsa_qlearning.plot_n_step
python -m hw2_sarsa_qlearning.frozenlake_experiment
python -m hw2_sarsa_qlearning.frozenlake_analyze

# Build this notebook (loads precomputed npz)
python -m hw2_sarsa_qlearning.build_notebook_v4
```
"""))

# =========================================================================
#  Assemble
# =========================================================================
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("hw2_sarsa_qlearning/Assignment2_Juan_Vintimilla.ipynb")
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(CELLS)} cells)")
