# Final Project — Deep Q-Learning: Continuous Maze

## 1. What is the project?

An agent must cross a **continuous 2D maze** from its start position on the left
(0.1, 0.5) to a goal circle on the right (0.9, 0.5), avoiding **seven red danger
zones** (instant death, −100) and **two black walls** (movement blocked, −1).

Unlike Project A, the state space here is **continuous** — the agent's position
is a pair of real numbers in [0, 1]², so states essentially never repeat and a
tabular Q-table cannot be used. Instead, a neural network approximates
Q(s, a): this is **Deep Q-Learning (DQN)**.

The environment (`env.py`) was provided and is used **unmodified**.

---

## 2. The Environment (`env.py` — provided, fixed)

| Property | Value |
|---|---|
| Observation | (x, y) position, continuous in [0, 1]² |
| Actions | Discrete(4): 0 = Up, 1 = Down, 2 = Left, 3 = Right |
| Step size | 0.05 normalized units per action |
| Start | (0.1, 0.5) — left edge, centre height |
| Goal | circle at (0.9, 0.5), radius 0.05 → **+10, episode ends** |
| Danger zones | 7 red rectangles → **−100, episode ends** |
| Walls | 2 black rectangles → **−1, movement blocked** |
| Other steps | reward 0 |

### The maze layout

```
        walls (top)          ██████████
        danger strip            ▓▓▓▓
   ▓▓                                       ▓▓        <- corner dangers (y 0.7–0.8)
              ▓▓▓
   S ───────► ▓▓▓  central bar (x 0.45–0.55, ● G      <- start & goal at y = 0.5
              ▓▓▓               y 0.4–0.6)
   ▓▓                                       ▓▓        <- corner dangers (y 0.2–0.3)
        danger strip            ▓▓▓▓
        walls (bottom)       ██████████
```

The **central danger bar** sits directly on the straight line between start and
goal, so the agent cannot simply "walk right". It must detour above (y ≈ 0.65)
or below (y ≈ 0.35) the bar — through the corridor between the central bar and
the corner danger squares — then come back to the centre line to hit the goal.

### Why this is hard

A uniformly random agent reaches the goal in only **~0.1% of episodes**
(measured over 2 000 episodes, 200 steps each). Nearly every random episode
ends in a danger zone. The reward is also **sparse**: reward is 0 everywhere
except at walls, dangers, and the goal, so an untrained agent gets no signal
pointing towards the goal.

---

## 3. Design of the solution

### 3.1 Why a DQN (`dqn.py`)

Q-learning needs Q(s, a) for every state. With a continuous state there are
infinitely many states, so we replace the table with a small **multi-layer
perceptron** that takes (x, y) and outputs 4 Q-values, one per action. The
three classic DQN components are all implemented:

| Component | Purpose |
|---|---|
| **Q-network** | maps state → Q-values; trained with the Bellman target |
| **Target network** | frozen copy of the Q-network, synced every 250 gradient steps; makes the TD target stable instead of chasing itself |
| **Replay buffer** | 50 000 past transitions, sampled in random mini-batches of 64; breaks the correlation between consecutive steps |

### 3.2 Reward shaping (`train.py` — wrapper, env untouched)

Two problems had to be solved to make DQN learn this maze, and both were found
experimentally (see §6, "What failed first"):

**Problem 1 — sparse reward.** With reward 0 on almost every step, DQN gets no
learning signal until the agent stumbles into the goal — which random
exploration essentially never does (~0.1%).

**Fix:** *potential-based reward shaping* (Ng et al., 1999). Every step earns a
bonus proportional to the decrease in distance to a target. Shaping of this
form provably does not change which policy is optimal.

**Problem 2 — the shaping trap.** Naive "distance to goal" shaping pulls the
agent **straight into the central danger bar**, because that is the direction
of the goal. Training runs with plain goal-distance shaping produced exactly
0 successful episodes out of 3 000.

**Fix:** the potential routes through a **waypoint above the bar** (0.5, 0.68)
while the agent is left of x = 0.42, then switches to the goal. The shaping
now guides the agent along a survivable path instead of the fatal straight
line. First goal contact happened at episode 6 (vs never).

Two further details prevent degenerate behaviour:

- a **step penalty** (−0.05/step) so dithering in place to farm shaping noise
  is strictly worse than making progress;
- a **step limit** (120) because the base environment never ends an episode on
  its own — without it, a cautious agent can wander forever.

### 3.3 Training stability

The environment's −100 danger reward is 10× larger than any other signal.
With plain MSE loss, a single danger-death mini-batch produces gradients large
enough to wipe out a partially learned policy — we observed success rates
climb to 36% and then collapse to 0%. Three standard fixes solved it:

| Fix | Effect |
|---|---|
| **Huber loss** (instead of MSE) | TD errors above 1.0 contribute linear, not quadratic, gradients |
| **Gradient clipping** (max-norm 10) | caps the size of any single update |
| **Best-checkpoint restore** | the weights from the best 100-episode window (by success rate) are what gets saved, not whatever the last episode left behind |
| **Exploration restart** | if ε has decayed to its floor and no episode in the last 300 reached the goal, ε is reset to 0.5 — recovers runs where the random seed never stumbled onto the goal before exploration ran out |

With these, the same configuration goes from "peaks at 36%, collapses" to
**100% success at its best checkpoint and 50/50 in greedy evaluation, with a
22-step path (theoretical optimum ≈ 21–22)**.

Two further "textbook" upgrades were also tried and **rejected after
measurement** (3 seeds each, same budget): **Double DQN** brought no benefit —
one seed failed outright (0/20 eval), and the working seeds matched vanilla —
and **soft (Polyak) target updates** roughly doubled CPU training time with no
stability gain. Both remain available as `DQNAgent(double_dqn=..., tau=...)`
options, off by default. A plausible reading: Double DQN exists to curb
optimistic overestimation, but with −100 cliffs everywhere this task, if
anything, benefits from vanilla DQN's optimism.

Robustness across seeds (vanilla config, 3 000 episodes): seeds 7 / 13 / 42
all reached 92–96% final success and 20/20 greedy evaluation with paths of
21–28 steps, so convergence does not hinge on a lucky seed.

---

## 4. Network architecture and hyperparameters — and why

| Parameter | Value | Why |
|---|---|---|
| Architecture | 2 → **64 → 64** → 4, ReLU | The input is only 2-dimensional, but the Q-surface has to bend around 7 danger zones; one small layer (32) underfits (see Experiment A), while 128×128 learns no better and trains slower |
| Optimizer / lr | Adam, **1e-3** | Standard DQN choice; 1e-4 is too slow to converge in 2 000 episodes, 5e-3 oscillates (see Experiment B) |
| Loss | **Huber** + grad-clip 10 | Robustness to the −100 outlier rewards (see §3.3) |
| γ (discount) | **0.99** | The optimal path is ~23 steps; γ=0.99 keeps the goal visible from the start (0.99²³ ≈ 0.79) |
| ε schedule | 1.0 → 0.05, ×0.995/episode | Reaches the floor around episode 600 — enough exploration to find the detour, then mostly exploitation to refine it; the exploration-restart safety net (§3.3) covers unlucky seeds |
| Replay buffer | 50 000 | Roughly the last ~700 episodes; big enough to keep rare successful trajectories in the sampling pool for a long time |
| Batch size | 64 | Standard; one gradient step per environment step |
| Target update | every 250 steps | Frequent enough to track improvement, infrequent enough to keep TD targets stable |
| Episodes | 3 000 | Success rate plateaus at ~95% around episode 1 300; the extra budget leaves room for an exploration restart on unlucky seeds; costs ~70 s of wall-clock time |
| shaping_scale | 10.0 | Keeps the total shaping over a full crossing (~8–13) the same order of magnitude as the +10 goal reward, so neither signal drowns the other |
| max_steps | 120 | ~5× the optimal path length (23 steps) |

---

## 5. Agent behaviour (what the trained agent does)

The greedy policy learned by the final network (see `solution.gif`, 22 steps):

1. **Walks right** from the start, dropping 3 steps to y ≈ 0.35 — just below
   the central danger bar (which starts at y = 0.4) while staying above the
   corner danger squares (which end at y = 0.3).
2. **Crosses the maze** rightward along that narrow y ≈ 0.35 lane.
3. **Climbs back up** to the centre line once past the bar and walks right
   into the goal circle.

Two things are worth noting. First, the agent did **not** take the northern
route suggested by the shaping waypoint — it found the mirror-image southern
detour instead, and a tighter one than the hand-coded reference path. This is
exactly what the theory promises: potential-based shaping accelerates learning
but does not dictate the final policy; the Q-network converged to its own
**22-step route, at the theoretical optimum of ≈ 21–22 steps** (3 down + 16
right + 3 up, ending inside the goal's radius). Second, the policy is
deterministic-robust: in greedy evaluation the agent reached the goal in
**50 out of 50 episodes, all in exactly 22 steps**.

The training curve also showcases the exploration-restart mechanism from
§3.3: this run found the goal early by luck (≈38% success around episode
400), collapsed back to 0%, ground along the −100 floor until the restart
fired near episode 900, then broke through for good — stabilising at 95–100%
from episode ~1300, with the saved model taken from a 100%-success window at
episode 2950. The remaining ~5% of training losses after episode 1300 are the
ε = 0.05 floor of random exploration actions, not policy mistakes.

---

## 6. What failed first (experimental record)

This is the sequence of designs that did **not** work, and why — kept here
because the final design is motivated entirely by these failures:

| Attempt | Result | Diagnosis |
|---|---|---|
| Plain DQN, raw sparse reward | 0 successes | no learning signal; random exploration finds goal ~0.1% of the time |
| + goal-distance shaping | 0 successes in 3 000 eps | shaping pulls the agent straight into the central danger bar |
| + step penalty & timeout penalty | 0 successes | same trap; agent instead learned "never move right" — Q-values for action Right were −37 to −270 everywhere |
| + waypoint shaping | first success at ep 6; peaks 36%, collapses to 0% | policy learned, then catastrophically forgotten — MSE loss amplifies −100 TD errors |
| + Huber loss, grad clip, best-checkpoint | **97% success, 50/50 greedy** | working design |
| + Double DQN, soft target updates | no gain; 1 of 3 seeds failed; ~2× slower | rejected — vanilla DQN's optimism seems to help against −100 cliffs (§3.3) |
| + exploration restart, 3-seed benchmark | **100% best checkpoint, 50/50 greedy at optimal 22 steps** | final design |

---

## 7. Experiments (`experiments.py`)

Run with:

```bash
python experiments.py
```

Produces `experiment_A_architecture.png` and `experiment_B_learning_rate.png`.
Both experiments train fresh agents for 2 000 episodes per configuration and
report the rolling success rate — for this task a much clearer metric than raw
reward, which mixes shaping bonuses with penalties.

### Experiment A — Network architecture

Compares Q-networks of `(32,)`, `(64, 64)` and `(128, 128)` hidden units with
all hyperparameters fixed (lr = 1e-3, 2 000 episodes each).

**Results (final 200 episodes):**

| Architecture | Final success | Final avg reward |
|---|---|---|
| 1 layer, 32 units | **0.0%** | −91.77 |
| 2 layers, 64 units | 96.5% | 8.73 |
| 2 layers, 128 units | **98.0%** | 10.93 |

**Analysis.** The single 32-unit layer never reached the goal once in 2 000
episodes — its reward curve wanders between −100 and −20 without ever
breaking through. Even though the input is only 2-dimensional, the Q-surface
the network must represent is highly non-linear: seven separate "cliffs" of
−100 around the danger zones with narrow safe corridors between them. One
small hidden layer cannot carve that surface, so the agent keeps walking into
dangers it cannot distinguish from safe cells.

Both two-layer networks solve the task. (128, 128) starts succeeding slightly
earlier (~episode 700 vs ~800) and ends marginally higher (98% vs 96.5%), but
the difference is within run-to-run noise, and each gradient step is ~2×
slower. **(64, 64) is the right choice**: the smallest network that reliably
solves the task.

### Experiment B — Learning rate

Compares Adam learning rates 1e-4, 1e-3, 5e-3 with the (64, 64) network fixed.

**Results (final 200 episodes):**

| Learning rate | Final success | Final avg reward |
|---|---|---|
| 1e-4 | **0.0%** | −87.99 |
| 1e-3 | **96.0%** | 8.23 |
| 5e-3 | 91.0% | 2.45 |

**Analysis.** lr = 1e-4 is a complete failure at this episode budget: updates
are so small that the rare early goal-reaching transitions in the replay
buffer fade from the sampling pool before the network has propagated their
value backwards along the path. Its curve shows it learning to avoid danger
(reward rising to ~−10) but never learning to reach the goal.

lr = 5e-3 was expected to be unstable, and the curves show exactly that
pattern in a milder form than feared: it learns *fastest* (60% success by
episode 370, ~99% by 750) but keeps wobbling — dips to 86–90% recur throughout
training, and it ends lower than 1e-3 (91.0% vs 96.0%, and noticeably lower
final reward, 2.45 vs 8.23, because its dips include more danger deaths).

**lr = 1e-3 is the sweet spot**: a few hundred episodes slower to take off
than 5e-3, but it climbs to ~96% and stays there with the smallest
oscillation. This matches the standard DQN recommendation.

### Overall conclusion

Both experiments show the same lesson from opposite directions: this task has
a **cliff-shaped failure mode** (danger zones), and both under-capacity
(32-unit network) and under-training (lr = 1e-4) fail *completely* rather than
degrading gracefully — the agent either learns to thread the corridors or it
never reaches the goal at all. Between configurations that do solve the task,
the differences are modest, and the moderate settings — (64, 64), lr = 1e-3 —
give the best stability-per-cost.

---

## 8. Project structure

```
project_b/
├── env.py             # PROVIDED environment — unmodified
├── dqn.py             # QNetwork, ReplayBuffer, DQNAgent (Huber + grad clip)
├── train.py           # ShapedMazeEnv wrapper (waypoint shaping), training loop
├── main.py            # Train the final agent; saves model, curve, GIF
├── evaluate.py        # Load a saved model, record a greedy episode as GIF
├── experiments.py     # Experiment A (architecture) and B (learning rate)
├── PROJECT_DOC.md     # This document
├── dqn_model.pt       # Trained Q-network weights (written by main.py)
├── training_curve.png # Episodes vs reward + success rate (written by main.py)
├── solution.gif       # Greedy agent reaching the goal (written by main.py)
├── experiment_A_architecture.png   # (written by experiments.py)
└── experiment_B_learning_rate.png  # (written by experiments.py)
```

## 9. How to run

**Train the final agent, save the training curve and the solution GIF:**

```bash
cd project_b
python main.py
```

**Re-generate the GIF from the saved model without retraining:**

```bash
python evaluate.py
```

**Run the two tuning experiments:**

```bash
python experiments.py
```
