# =============================================================================
# experiments.py — DQN hyperparameter and architecture experiments
# =============================================================================
#
# How to run
# ----------
#     cd project_b
#     python experiments.py
#
# Runs two independent experiments and saves their plots as PNGs.
#
# Experiment A — Network architecture
# ------------------------------------
# We compare three Q-network sizes with all hyperparameters fixed:
#
#   (32,)      — one small hidden layer. May lack the capacity to carve the
#                Q-surface around seven danger zones from just an (x, y) input.
#   (64, 64)   — two medium layers. Enough capacity for this 2D task without
#                being slow or prone to overfitting the replay buffer.
#   (128, 128) — two large layers. More capacity than the task needs; each
#                gradient step is slower, and more parameters take longer to
#                converge on the same amount of data.
#
# Experiment B — Learning rate
# ------------------------------------
# We compare three Adam learning rates with the (64, 64) network fixed:
#
#   1e-4 — small, stable updates but slow: may not converge in 2 000 episodes.
#   1e-3 — the standard DQN choice; learns fast and stays stable.
#   5e-3 — aggressive updates; risks oscillation and catastrophic forgetting
#          right after the policy starts working.
#
# Each configuration reports the rolling success rate (fraction of episodes
# where the agent actually reached the goal) — for this task that is a much
# clearer signal than raw reward, which mixes shaping bonuses with penalties.
# =============================================================================

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dqn import DQNAgent
from train import ShapedMazeEnv, train

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Shared settings — kept constant across ALL experiments
# ---------------------------------------------------------------------------
N_EPISODES = 2_000
SMOOTH_WIN = 100     # rolling-average window for the curves


def smooth(values, window=SMOOTH_WIN):
    """Rolling average over `window` episodes."""
    return np.convolve(values, np.ones(window) / window, mode="valid")


def run_config(hidden_sizes, lr):
    """Train one fresh agent and return its per-episode history."""
    env = ShapedMazeEnv(render_mode=None, shaping_scale=10.0,
                        step_penalty=0.05, max_steps=120)
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        hidden_sizes=hidden_sizes,
        lr=lr,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_capacity=50_000,
        batch_size=64,
        target_update_freq=250,
    )
    return train(env, agent, n_episodes=N_EPISODES, log_every=200)


def plot_histories(histories, title, path):
    """Two panels per experiment: smoothed reward (top) and success rate (bottom)."""
    colors = ["steelblue", "seagreen", "tomato"]
    x = np.arange(SMOOTH_WIN - 1, N_EPISODES)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for (label, hist), color in zip(histories.items(), colors):
        axes[0].plot(x, smooth(hist["reward"]), label=label, color=color, linewidth=2)
        success = np.array(hist["reached_goal"], dtype=float)
        axes[1].plot(x, smooth(success) * 100, label=label, color=color, linewidth=2)

    axes[0].set_ylabel("Total reward per episode\n(rolling avg)")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].set_ylabel("Success rate (%)\n(rolling avg)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylim(-5, 105)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {path}")
    plt.close()


def summarize(histories):
    for label, hist in histories.items():
        final_success = np.mean(hist["reached_goal"][-200:]) * 100
        final_reward = np.mean(hist["reward"][-200:])
        print(f"  {label:<22} final success {final_success:5.1f}%   "
              f"final avg reward {final_reward:7.2f}")


# ---------------------------------------------------------------------------
# Experiment A — Network architecture
# ---------------------------------------------------------------------------
def run_experiment_A():
    print("=" * 60)
    print("Experiment A: Network architecture (lr=1e-3 fixed)")
    print("  (32,)   vs   (64, 64)   vs   (128, 128)")
    print("=" * 60)

    architectures = [
        ("1 layer, 32 units", (32,)),
        ("2 layers, 64 units", (64, 64)),
        ("2 layers, 128 units", (128, 128)),
    ]
    histories = {}
    for label, hidden in architectures:
        print(f"\nTraining: {label}")
        histories[label] = run_config(hidden_sizes=hidden, lr=1e-3)

    print("\nExperiment A results:")
    summarize(histories)
    plot_histories(histories, "Experiment A — Q-Network Architecture",
                   HERE / "experiment_A_architecture.png")
    return histories


# ---------------------------------------------------------------------------
# Experiment B — Learning rate
# ---------------------------------------------------------------------------
def run_experiment_B():
    print("\n" + "=" * 60)
    print("Experiment B: Learning rate (architecture (64, 64) fixed)")
    print("  lr = 1e-4   vs   1e-3   vs   5e-3")
    print("=" * 60)

    learning_rates = [
        ("lr = 1e-4", 1e-4),
        ("lr = 1e-3", 1e-3),
        ("lr = 5e-3", 5e-3),
    ]
    histories = {}
    for label, lr in learning_rates:
        print(f"\nTraining: {label}")
        histories[label] = run_config(hidden_sizes=(64, 64), lr=lr)

    print("\nExperiment B results:")
    summarize(histories)
    plot_histories(histories, "Experiment B — Learning Rate",
                   HERE / "experiment_B_learning_rate.png")
    return histories


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_experiment_A()
    run_experiment_B()
    print("\nAll experiments complete.")
