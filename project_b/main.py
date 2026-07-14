"""
main.py — Train the DQN agent on the continuous maze and save all final outputs.

How to run
----------
Train from scratch and save outputs::

    cd project_b
    python main.py

To skip training and just regenerate the plots/GIF from a saved network, set
``do_train = False`` at the top of this file. To watch the agent during
training (much slower), set ``render_training = True``.

For the exam demo (load trained weights, no training), use evaluate.py
instead — it is the dedicated test-mode entry point::

    python evaluate.py --episodes 10

Outputs written to project_b/
    dqn_model.pth       — trained Q-network weights (best checkpoint)
    training_curve.png  — episodes vs total reward + success rate
    solution.gif        — greedy agent navigating from start to goal
"""
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dqn import DQNAgent
from train import ShapedMazeEnv, train

HERE = Path(__file__).parent

# Seed everything for reproducibility. Training still works from any seed —
# the exploration-restart logic in train() recovers unlucky runs — but a fixed
# seed makes the saved outputs repeatable. 42 was the best of the seeds
# benchmarked in the A/B test (fastest convergence, near-optimal 22-step path).
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Settings ──────────────────────────────────────────────────────────────────
do_train        = True
do_plot         = True
do_gif          = True
render_training = False   # Set True to watch the agent during training (slow)

model_path = HERE / "dqn_model.pth"

# ── Hyperparameters ───────────────────────────────────────────────────────────
n_episodes    = 3_000
hidden_sizes  = (64, 64)  # Q-network architecture: 2 -> 64 -> 64 -> 4
lr            = 1e-3      # Adam learning rate
gamma         = 0.99      # Discount factor
epsilon       = 1.0       # Start fully exploring
epsilon_min   = 0.05      # Never drop below 5% exploration
epsilon_decay = 0.995     # Multiply epsilon by this after every episode
buffer_size   = 50_000    # Replay buffer capacity
batch_size    = 64        # Mini-batch size per gradient step
target_update = 250       # Gradient steps between target-network syncs

# Environment wrapper (see train.py for why shaping is needed)
shaping_scale = 10.0      # Weight of the potential-based shaping term
step_penalty  = 0.05      # Flat per-step cost — discourages dithering
max_steps     = 120       # Steps before an episode is truncated


def plot_training_curve(history, path):
    """Two panels: smoothed episode reward (top) and success rate (bottom)."""
    window = 50
    rewards = np.array(history["reward"])
    successes = np.array(history["reached_goal"], dtype=float)
    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
    success_rate = np.convolve(successes, np.ones(window) / window, mode="valid") * 100
    x = np.arange(window - 1, len(rewards))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})
    fig.suptitle("DQN Training Curve — Continuous Maze", fontsize=13, fontweight="bold")

    axes[0].plot(rewards, alpha=0.15, color="steelblue", linewidth=0.5)
    axes[0].plot(x, smooth, color="steelblue", linewidth=2,
                 label=f"Reward ({window}-episode rolling average)")
    axes[0].set_ylabel("Total reward per episode")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, success_rate, color="seagreen", linewidth=2)
    axes[1].set_ylabel("Success rate (%)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylim(-5, 105)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Training curve saved -> {path}")
    plt.close()


if __name__ == "__main__":
    env = ShapedMazeEnv(
        render_mode="human" if render_training else None,
        shaping_scale=shaping_scale,
        step_penalty=step_penalty,
        max_steps=max_steps,
    )
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        hidden_sizes=hidden_sizes,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_size,
        batch_size=batch_size,
        target_update_freq=target_update,
    )

    if do_train:
        print(f"Training for {n_episodes:,} episodes...\n")
        history = train(env, agent, n_episodes=n_episodes, render=render_training)
        agent.save(model_path)
        if do_plot:
            plot_training_curve(history, HERE / "training_curve.png")
    else:
        agent.load(model_path)

    if do_gif:
        from evaluate import record_episode

        gif_env = ShapedMazeEnv(render_mode="human", shaping_scale=shaping_scale,
                                step_penalty=step_penalty, max_steps=max_steps)
        # Greedy policy + deterministic env -> a single attempt is definitive
        if not record_episode(agent, gif_env, HERE / "solution.gif"):
            print("Warning: greedy agent did not reach the goal — "
                  "inspect training_curve.png and consider retraining.")
        gif_env.close()
