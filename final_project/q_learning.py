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


def train(env, agent, no_episodes, render=False):
    """
    Run Q-learning for a fixed number of episodes and save the Q-table.
    Prints a progress line every 1 000 episodes.

    Note: DeliveryRobotEnv.step() returns (obs, done, reward, info) — the
    done/reward order is non-standard, so we unpack accordingly.
    """
    for episode in range(no_episodes):
        obs, _ = env.reset()
        state  = tuple(obs)         # (row, col, has_package)
        total_reward = 0

        while True:
            action = agent.select_action(state)

            next_obs, done, reward, _ = env.step(action)   # non-standard order!

            if render:
                env.render()

            next_state    = tuple(next_obs)
            agent.update(state, action, reward, next_state)

            state        = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1:>6} | "
                  f"Reward: {total_reward:>8.2f} | "
                  f"Epsilon: {agent.epsilon:.4f}")

    env.close()
    print("\nTraining complete.")


def visualize(agent, env):
    """
    One figure with two side-by-side greedy-policy plots:
      Left  — Phase 1: robot has no package, heading to pick it up (P)
      Right — Phase 2: robot is carrying the package, heading to customer (C)

    Each cell shows:
      arrow  — the greedy action the agent would take
      number — the max Q-value for that cell
    Special cells are coloured and labelled so they stand out:
      P (yellow)  — package location
      C (green)   — customer / delivery goal
      X (red)     — traffic / danger (episode ends with -15)
      W (gray)    — wall (movement blocked)
    """
    q_table     = agent.q_table        # (grid, grid, 2, 4)
    grid_size   = q_table.shape[0]
    arrows      = {0: "↑", 1: "↓", 2: "→", 3: "←"}

    package_rc  = tuple(env.package)
    customer_rc = tuple(env.customer)
    dangers     = [tuple(d) for d in env.danger_states]
    walls       = [tuple(w) for w in env.wall_states]

    special = set(dangers) | set(walls) | {package_rc, customer_rc}

    phase_titles = ["Phase 1: Go fetch the package  (P)",
                    "Phase 2: Deliver to customer  (C)"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Learned Greedy Policy", fontsize=14, fontweight="bold")

    for phase, ax in enumerate(axes):
        phase_q       = q_table[:, :, phase, :]        # (grid, grid, 4)
        best_actions  = np.argmax(phase_q, axis=2)
        best_q_values = np.max(phase_q, axis=2)

        # Mask special cells so the heatmap doesn't colour them
        mask = np.zeros((grid_size, grid_size), dtype=bool)
        for rc in special:
            mask[rc] = True

        sns.heatmap(best_q_values, annot=False, cmap="Blues",
                    ax=ax, cbar=True, mask=mask,
                    linewidths=0.5, linecolor="lightgray",
                    vmin=best_q_values[~mask].min() if (~mask).any() else 0,
                    vmax=best_q_values[~mask].max() if (~mask).any() else 1)

        # Draw every cell
        for row in range(grid_size):
            for col in range(grid_size):
                rc  = (row, col)
                x   = col + 0.5
                y   = row + 0.5

                if rc in walls:
                    ax.add_patch(plt.Rectangle((col, row), 1, 1,
                                 color=(0.27, 0.27, 0.27), zorder=2))
                    ax.text(x, y, "W", color="white", ha="center", va="center",
                            fontsize=11, fontweight="bold", zorder=3)

                elif rc in dangers:
                    ax.add_patch(plt.Rectangle((col, row), 1, 1,
                                 color=(0.86, 0.20, 0.20), zorder=2))
                    ax.text(x, y, "X", color="white", ha="center", va="center",
                            fontsize=11, fontweight="bold", zorder=3)

                elif rc == package_rc:
                    ax.add_patch(plt.Rectangle((col, row), 1, 1,
                                 color=(1.0, 0.82, 0.0), zorder=2))
                    ax.text(x, y, "P", color="black", ha="center", va="center",
                            fontsize=13, fontweight="bold", zorder=3)

                elif rc == customer_rc:
                    ax.add_patch(plt.Rectangle((col, row), 1, 1,
                                 color=(0.20, 0.75, 0.20), zorder=2))
                    ax.text(x, y, "C", color="white", ha="center", va="center",
                            fontsize=13, fontweight="bold", zorder=3)

                else:
                    q_val = best_q_values[row, col]
                    ax.text(x, y - 0.15, arrows[best_actions[row, col]],
                            color="black", ha="center", va="center",
                            fontsize=15, fontweight="bold", zorder=3)
                    ax.text(x, y + 0.25, f"{q_val:.1f}",
                            color="dimgray", ha="center", va="center",
                            fontsize=7, zorder=3)

        ax.set_title(phase_titles[phase], fontsize=11)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color=(1.0, 0.82, 0.0),    label="P — Package"),
        Patch(color=(0.20, 0.75, 0.20),  label="C — Customer"),
        Patch(color=(0.86, 0.20, 0.20),  label="X — Traffic (danger)"),
        Patch(color=(0.27, 0.27, 0.27),  label="W — Wall (blocked)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4,
               frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.show()
