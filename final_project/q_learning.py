import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class QLearningAgent:
    """
    Tabular Q-learning agent for the DeliveryRobotEnv.

    State is (row, col, has_package) so the Q-table has shape:
        (grid_size, grid_size, 2, n_actions)

    Epsilon-greedy policy balances exploration vs exploitation.
    Q-values updated with the Bellman equation on every step.
    """

    def __init__(self, grid_size, n_actions, alpha, gamma,
                 epsilon, epsilon_min, epsilon_decay):
        self.alpha         = alpha          # Learning rate
        self.gamma         = gamma          # Discount factor
        self.epsilon       = epsilon        # Current exploration rate
        self.epsilon_min   = epsilon_min    # Floor for epsilon
        self.epsilon_decay = epsilon_decay  # Multiplicative decay per episode

        # has_package is 0 or 1, so the 3rd dimension has size 2
        self.q_table = np.zeros((grid_size, grid_size, 2, n_actions))

    def select_action(self, state):
        """Epsilon-greedy: explore randomly or exploit the best known action."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.q_table.shape[-1])  # Explore
        return int(np.argmax(self.q_table[state]))             # Exploit

    def update(self, state, action, reward, next_state):
        """Bellman update: move Q(s,a) toward the TD target."""
        td_target = reward + self.gamma * np.max(self.q_table[next_state])
        td_error  = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="q_table.npy"):
        np.save(path, self.q_table)
        print(f"Q-table saved → {path}")

    def load(self, path="q_table.npy"):
        self.q_table = np.load(path)
        print(f"Q-table loaded ← {path}")


def _run_episode(env, agent, render=False):
    """Run one episode and return the total reward."""
    obs, _ = env.reset()
    state  = tuple(obs)
    total  = 0
    while True:
        action              = agent.select_action(state)
        next_obs, done, reward, _ = env.step(action)
        if render:
            env.render()
        agent.update(state, action, reward, tuple(next_obs))
        state = tuple(next_obs)
        total += reward
        if done:
            break
    agent.decay_epsilon()
    return total


def train(env, agent, no_episodes, render=False):
    """
    Train for a fixed number of episodes, printing progress every 1 000.
    Used by main.py for the final trained agent.
    """
    for episode in range(no_episodes):
        total = _run_episode(env, agent, render)
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1:>6} | "
                  f"Reward: {total:>8.2f} | "
                  f"Epsilon: {agent.epsilon:.4f}")
    env.close()
    print("\nTraining complete.")


def train_and_record(env, agent, no_episodes):
    """
    Same as train() but returns a list of per-episode total rewards.
    Used by experiments.py to produce learning curves.
    """
    rewards = []
    for _ in range(no_episodes):
        rewards.append(_run_episode(env, agent))
    env.close()
    return rewards


def visualize(agent, env):
    """
    Two side-by-side grid plots showing the greedy policy learned by the agent.

    Left plot  — Phase 1: robot has no package, navigating toward P.
    Right plot — Phase 2: robot is carrying the package, navigating toward C.

    Reading the plot
    ----------------
    - Blue shading : how valuable that cell is (darker = higher Q-value).
      Brighter blue means the agent strongly prefers to be there.
    - Arrow        : the best action the agent would take from that cell
                     (↑ Up, ↓ Down, → Right, ← Left).
    - Number       : the exact max Q-value for that cell.
    - Coloured cells are special states — see the legend below the plots.
    """
    from matplotlib.patches import Patch

    q_table     = agent.q_table          # shape: (grid, grid, 2, 4)
    grid_size   = q_table.shape[0]
    arrows      = {0: "↑", 1: "↓", 2: "→", 3: "←"}

    package_rc  = tuple(env.package)
    customer_rc = tuple(env.customer)
    walls       = {tuple(w) for w in env.wall_states}
    dangers     = {tuple(d) for d in env.danger_states}
    bonuses     = {tuple(b) for b in env.bonus_states}
    special     = walls | dangers | bonuses | {package_rc, customer_rc}

    # Colours match the pygame render exactly
    CELL_STYLE = {
        "wall":     ((0.27, 0.27, 0.27), "W", "white"),
        "danger":   ((0.82, 0.20, 0.20), "X", "white"),
        "bonus":    ((0.00, 0.73, 0.76), "B", "white"),
        "package":  ((0.94, 0.78, 0.00), "P", "black"),
        "customer": ((0.20, 0.71, 0.20), "C", "white"),
    }

    def cell_type(rc):
        if rc in walls:       return "wall"
        if rc in dangers:     return "danger"
        if rc in bonuses:     return "bonus"
        if rc == package_rc:  return "package"
        if rc == customer_rc: return "customer"
        return None

    titles = [
        "Phase 1 — Fetch the Package (P)\n"
        "Agent has no package; arrows show the path it would take to P",
        "Phase 2 — Deliver to Customer (C)\n"
        "Agent is carrying the package; arrows show the path it would take to C",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        "Learned Greedy Policy  —  Arrows = optimal action, Colour = Q-value",
        fontsize=13, fontweight="bold", y=1.01
    )

    for phase, ax in enumerate(axes):
        phase_q   = q_table[:, :, phase, :]        # (grid, grid, 4)
        best_act  = np.argmax(phase_q, axis=2)
        best_q    = np.max(phase_q, axis=2)

        # Mask special cells from the heatmap colouring
        mask = np.zeros((grid_size, grid_size), dtype=bool)
        for rc in special:
            mask[rc] = True

        free = best_q[~mask]
        sns.heatmap(
            best_q, ax=ax, cmap="Blues", mask=mask,
            cbar=True, linewidths=0.8, linecolor="lightgray",
            vmin=free.min() if free.size else 0,
            vmax=free.max() if free.size else 1,
            annot=False
        )

        for row in range(grid_size):
            for col in range(grid_size):
                rc = (row, col)
                x, y = col + 0.5, row + 0.5
                ct = cell_type(rc)

                if ct is not None:
                    color, label, tc = CELL_STYLE[ct]
                    ax.add_patch(plt.Rectangle((col, row), 1, 1,
                                               color=color, zorder=2))
                    ax.text(x, y, label, color=tc, ha="center", va="center",
                            fontsize=13, fontweight="bold", zorder=3)
                else:
                    # Arrow (large, centred) + Q-value (small, below)
                    ax.text(x, y - 0.12, arrows[best_act[row, col]],
                            color="black", ha="center", va="center",
                            fontsize=16, fontweight="bold", zorder=3)
                    ax.text(x, y + 0.28, f"{best_q[row, col]:.1f}",
                            color="#444444", ha="center", va="center",
                            fontsize=7.5, zorder=3)

        ax.set_title(titles[phase], fontsize=10, pad=8)
        ax.set_xlabel("Column", fontsize=9)
        ax.set_ylabel("Row", fontsize=9)
        ax.tick_params(labelsize=8)

    # Legend in its own clearly separated row below both plots
    legend_handles = [
        Patch(color=(0.27, 0.27, 0.27), label="W  Wall — movement blocked (−0.05)"),
        Patch(color=(0.82, 0.20, 0.20), label="X  Danger — episode ends (−15)"),
        Patch(color=(0.00, 0.73, 0.76), label="B  Bonus — one-time reward (+3)"),
        Patch(color=(0.94, 0.78, 0.00), label="P  Package — sub-goal (+5)"),
        Patch(color=(0.20, 0.71, 0.20), label="C  Customer — terminal goal (+20)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=5,
        frameon=True, fontsize=9,
        bbox_to_anchor=(0.5, -0.08),
        title="Special cells", title_fontsize=9
    )

    plt.tight_layout()
    plt.savefig("policy_plot.png", dpi=150, bbox_inches="tight")
    print("Policy plot saved → policy_plot.png")
    plt.show()
