# Final Project — Q-Learning: Delivery Robot

## 1. What is the project?

A robot navigates an **8×8 city grid** to complete a two-phase delivery task:

- **Phase 1** — find the package `P` and pick it up
- **Phase 2** — carry the package to the customer `C`

The robot learns purely through trial and error using **Q-learning**: it receives rewards and penalties as it moves around, and gradually figures out the best path to take.

---

## 2. The Environment (`env.py`)

### 2.1 State Space

The robot's situation at any moment is described by three numbers:

| Variable | Values | Meaning |
|---|---|---|
| `row` | 0 – 7 | Which row the robot is in |
| `col` | 0 – 7 | Which column the robot is in |
| `has_package` | 0 or 1 | Whether the robot is carrying the package |

This gives **8 × 8 × 2 = 128 possible states**.

### 2.2 Action Space

At every step the robot chooses one of 4 actions:

| Action | Direction |
|---|---|
| 0 | Up |
| 1 | Down |
| 2 | Right |
| 3 | Left |

If the robot tries to move into a wall or off the edge of the grid, it simply stays in place.

### 2.3 Special States

The map has **11 special cells** across **4 distinct mechanic types**:

| Symbol | Name | Count | What happens | Reward |
|---|---|---|---|---|
| **W** | Wall | 4 | Movement blocked; robot stays | −0.05 |
| **X** | Danger | 3 | Traffic jam; episode ends immediately | −15 |
| **B** | Bonus | 2 | Fuel station; gives reward first time only | +3 |
| **P** | Package | 1 | Sub-goal; robot picks it up, Phase 2 begins | +5 |
| **C** | Customer | 1 | Terminal goal; only counts if robot has package | +20 |

Every other step costs **−0.01** — a small penalty that encourages the robot to find short paths rather than wander.  If the robot takes more than **100 steps** without finishing, the episode ends with **−10**.

### 2.4 Map Layout

```
     0    1    2    3    4    5    6    7
  +----+----+----+----+----+----+----+----+
0 | R  |    |    |    |    | B  |    |    |
  +----+----+----+----+----+----+----+----+
1 |    |    |    | W  |    |    | P  |    |
  +----+----+----+----+----+----+----+----+
2 |    | X  |    | W  |    |    |    |    |
  +----+----+----+----+----+----+----+----+
3 |    |    |    | W  | W  |    |    |    |
  +----+----+----+----+----+----+----+----+
4 |    |    |    |    |    | X  |    |    |
  +----+----+----+----+----+----+----+----+
5 |    |    |    |    |    |    |    | B  |
  +----+----+----+----+----+----+----+----+
6 |    |    | X  |    |    |    |    |    |
  +----+----+----+----+----+----+----+----+
7 |    |    |    |    |    |    |    | C  |
  +----+----+----+----+----+----+----+----+
```

`R` = robot start (0,0), `P` = package (1,6), `C` = customer (7,7)

The 4 walls form an **L-shaped barrier** in the top-left area that forces the robot to plan a route around it rather than going straight.

---

## 3. The Algorithm — Q-Learning (`q_learning.py`)

### 3.1 The Q-Table

Q-learning stores a table of **Q-values**: one number for every (state, action) pair.

```
Q[row, col, has_package, action]   →   shape: (8, 8, 2, 4)
```

A Q-value represents **"how much total future reward the agent expects if it takes this action from this state and then acts optimally afterwards"**.

At the start all Q-values are 0. The agent updates them as it experiences the environment.

### 3.2 The Bellman Update

After every step the agent applies this update rule:

```
Q(s, a)  ←  Q(s, a)  +  α · [ r  +  γ · max Q(s', a')  −  Q(s, a) ]
                                 └─────────────────────────┘
                                        TD target
```

Breaking this down:

- `r` — the reward just received
- `γ · max Q(s', a')` — the discounted value of the best next state
- The term in brackets is the **TD error**: how wrong our current Q-value was
- `α` scales how much we correct for that error

### 3.3 Exploration vs Exploitation (ε-greedy)

The agent uses **ε-greedy** to decide whether to explore or exploit:

- With probability **ε** → take a **random action** (explore)
- With probability **1 − ε** → take the **best known action** (exploit)

Early in training ε is high so the agent explores widely. Over time ε decays so the agent increasingly trusts what it has learned.

---

## 4. Hyperparameters (`main.py`)

| Parameter | Value | What it controls | Why this value |
|---|---|---|---|
| `alpha` (α) | 0.1 | **Learning rate** — how much the Q-value changes on each update | 0.1 is a safe middle ground: fast enough to learn, stable enough not to oscillate |
| `gamma` (γ) | 0.99 | **Discount factor** — how much future rewards matter vs immediate ones | Close to 1 means the agent plans far ahead, which is important for a multi-step delivery task |
| `epsilon` | 1.0 | **Starting exploration rate** | Start at 1.0 so the agent explores completely at first |
| `epsilon_min` | 0.05 | **Minimum exploration rate** | Keep 5% random exploration always, so the agent never fully stops discovering |
| `epsilon_decay` | 0.9995 | **How fast ε shrinks per episode** | Multiply by 0.9995 each episode. Over 30 000 episodes ε goes from 1.0 → ~0.05 smoothly |
| `no_episodes` | 30 000 | **How many training episodes to run** | Enough for ε to decay fully and for Q-values to converge |

### Why γ = 0.99 specifically?

The delivery task requires the robot to pick up the package AND then deliver it — these two goals can be many steps apart. With γ = 0.99 the agent can look ~100 steps ahead before the reward becomes negligible (0.99^100 ≈ 0.37). A lower γ like 0.9 would make it short-sighted — it would stop caring about the delivery goal.

### Why ε-decay = 0.9995?

With 30 000 episodes:
- At episode 1 000: ε ≈ 0.61 (still exploring a lot)
- At episode 5 000: ε ≈ 0.08 (mostly exploiting)
- At episode 10 000+: ε ≈ 0.05 (at the floor, mostly exploiting)

This gives enough exploration early to discover the package and customer, and enough exploitation later to refine the optimal path.

---

## 5. Reading the Policy Plot

After training, `visualize()` produces two side-by-side grid plots.

### What each plot shows

**Left — Phase 1**: The robot has no package. The arrows show the path it would follow to reach `P`.

**Right — Phase 2**: The robot is carrying the package. The arrows show the path it would follow to deliver to `C`.

### How to read the cells

| What you see | What it means |
|---|---|
| **Arrow** (↑ ↓ → ←) | The best action the agent learned for that cell |
| **Number** below arrow | The Q-value: how much total reward the agent expects from here |
| **Dark blue** cell | High Q-value — the agent strongly wants to be here |
| **Light blue** cell | Lower Q-value — less valuable position |
| **Coloured** cell (W/X/B/P/C) | Special state — see legend |

### What a good policy looks like

- In Phase 1, arrows in the free cells should generally point **toward P** (row 1, col 6)
- In Phase 2, arrows should generally point **toward C** (row 7, col 7)
- The agent should route **around** walls and **away from** danger cells
- The path may go slightly out of its way to collect a **Bonus** cell if it is on the route

### What Q-values tell you

High Q-values near the goal confirm the agent has learned that being close to the goal is valuable. Low or negative Q-values near danger cells confirm the agent has learned to avoid them. If cells near `X` (danger) have low Q-values, the policy is working correctly.

---

## 6. Experiments (`experiments.py`)

Run with:

```bash
python experiments.py
```

This produces two figures and saves them as PNG files.

---

### Experiment A — Epsilon-Decay Strategy

**What we compare:** Two epsilon-decay schedules with everything else held constant (α=0.1, γ=0.99).

| Strategy | ε_decay | Behaviour |
|---|---|---|
| Slow decay | 0.9998 | ε stays high for ~10 000 episodes before reaching 0.05 |
| Fast decay | 0.9990 | ε reaches 0.05 after ~3 000 episodes |

**Why this is meaningful:** The decay schedule determines the *exploration-exploitation trade-off* over time. This is a central concept in reinforcement learning — the agent must balance discovering new paths (exploration) against following the best path it already knows (exploitation).

**What the plot shows:**
- The **top panel** shows the smoothed total reward per episode for each strategy.
- The **bottom panel** shows how ε drops over time for each strategy, so you can see exactly when each agent transitions from exploring to exploiting.

**Expected finding:** Slow decay reaches a higher reward earlier because the agent has more time to discover the package and the delivery route before it stops exploring. Fast decay may plateau at a lower reward if it commits to a suboptimal path too early.

---

### Experiment B — Learning Rate (Alpha)

**What we compare:** Three values of α with everything else held constant (ε_decay=0.9995, γ=0.99).

| Alpha | Characteristic |
|---|---|
| 0.01 | Very small updates — slow but stable convergence |
| 0.1 | Balanced — standard starting point |
| 0.5 | Large updates — fast early learning but noisy |

**Why this is meaningful:** α directly controls the magnitude of the Bellman update:

```
Q(s,a)  ←  Q(s,a)  +  α · TD_error
```

A large α makes Q-values jump toward each new experience aggressively. A small α averages over many experiences. In a deterministic environment like this one, moderate α works best — too small is unnecessarily slow, too large causes oscillation.

**What the plot shows:** Three smoothed reward curves on one graph. The curves reveal how fast each α learns and how stable it is once it converges.

**Expected finding:** α=0.1 converges cleanly. α=0.01 shows a steady but slow rise that may still be climbing at episode 20 000. α=0.5 rises quickly but has a noisier curve, showing the instability of large updates.

---

## 7. Project Structure

```
final_project/
├── env.py            # The environment: grid, rewards, special states
├── q_learning.py     # The agent: Q-table, Bellman update, train, visualize
├── main.py           # Train and visualize the final agent
├── experiments.py    # Experiment A (ε-decay) and Experiment B (alpha)
├── PROJECT_DOC.md    # This document
└── q_table.npy       # Saved Q-table (written after training)
```

---

## 8. How to Run

**Train the final agent and show the policy plot:**
```bash
cd final_project
python main.py
```

**Run the experiments and produce the comparison plots:**
```bash
python experiments.py
```

Set `do_train = True` in `main.py` to train from scratch, or `False` to load a saved Q-table and only visualize. Set `render_training = True` to watch the robot learn in real time (much slower).
