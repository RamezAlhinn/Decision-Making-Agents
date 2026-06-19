# =============================================================================
# experiments.py — Q-learning Experiments
# =============================================================================
#
# This file runs two independent experiments and saves their plots as PNGs.
#
# Experiment A — Epsilon-decay strategy
# --------------------------------------
# We compare two decay schedules while keeping alpha and gamma fixed.
# The decay schedule controls how fast the agent shifts from exploring
# (random actions) to exploiting (best known actions).
#
#   Strategy 1 — Slow decay  (ε_decay = 0.9998)
#     ε stays high for longer → agent explores more before committing.
#     Useful when the environment is large or rewards are sparse, because
#     the agent needs more time to discover good paths.
#
#   Strategy 2 — Fast decay  (ε_decay = 0.9990)
#     ε drops quickly → agent starts exploiting early.
#     Can converge faster on simple maps but risks getting stuck in a
#     suboptimal policy if it stops exploring before finding the goal.
#
# Theory link: exploration-exploitation trade-off (lecture content).
# The expected result is that slow decay reaches a higher final reward
# because it discovers the package and delivery path more reliably.
#
# Experiment B — Learning rate (alpha)
# --------------------------------------
# We compare three values of α while keeping the epsilon schedule fixed.
# α controls how strongly each new experience updates the Q-values:
#
#   α = 0.01 — very small updates; Q-values change slowly.
#     The agent is cautious and stable, but needs many more episodes
#     to converge. May still be improving at 20 000 episodes.
#
#   α = 0.1  — balanced; the standard starting point.
#     Updates are meaningful but not so large that they overshoot.
#     Usually converges well within 20 000 episodes.
#
#   α = 0.5  — large updates; the agent reacts strongly to each step.
#     Learns fast early but can oscillate: one bad episode can undo
#     progress from many good ones, causing a noisy reward curve.
#
# Theory link: the Bellman update rule — Q(s,a) ← Q(s,a) + α·TD_error.
# Large α amplifies the TD error correction; small α dampens it.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from env import DeliveryRobotEnv
from q_learning import QLearningAgent, train_and_record

# ---------------------------------------------------------------------------
# Shared settings — kept constant across ALL experiments
# ---------------------------------------------------------------------------
NO_EPISODES = 20_000   # enough for ε to fully decay and curves to stabilise
GAMMA       = 0.99
SMOOTH_WIN  = 300      # window for rolling average (smooths noisy reward curves)


def smooth(rewards, window=SMOOTH_WIN):
    """Rolling average over `window` episodes."""
    return np.convolve(rewards, np.ones(window) / window, mode="valid")


def make_agent(alpha, epsilon_decay):
    """Helper — builds a fresh headless env + agent with the given hyperparameters."""
    env = DeliveryRobotEnv(random_start=True, headless=True)
    agent = QLearningAgent(
        grid_size     = env.grid_size,
        n_actions     = env.action_space.n,
        alpha         = alpha,
        gamma         = GAMMA,
        epsilon       = 1.0,
        epsilon_min   = 0.05,
        epsilon_decay = epsilon_decay,
    )
    return env, agent


# ---------------------------------------------------------------------------
# Experiment A — Epsilon-decay strategy comparison
# ---------------------------------------------------------------------------
def run_experiment_A():
    print("=" * 60)
    print("Experiment A: Epsilon-decay strategy")
    print("  Slow decay  ε_decay=0.9998   (α=0.1 fixed)")
    print("  Fast decay  ε_decay=0.9990   (α=0.1 fixed)")
    print("=" * 60)

    strategies = [
        ("Slow decay  (ε_decay=0.9998)", 0.9998),
        ("Fast decay  (ε_decay=0.9990)", 0.9990),
    ]
    colors  = ["steelblue", "tomato"]
    results = {}

    for label, decay in strategies:
        print(f"\nTraining: {label}")
        env, agent = make_agent(alpha=0.1, epsilon_decay=decay)
        rewards = train_and_record(env, agent, NO_EPISODES)
        results[label] = rewards
        print(f"  Final avg reward (last 1000 eps): "
              f"{np.mean(rewards[-1000:]):.2f}")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle("Experiment A — Epsilon-Decay Strategy Comparison",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    x_smooth = np.arange(SMOOTH_WIN - 1, NO_EPISODES)

    for (label, _), color in zip(strategies, colors):
        raw = results[label]
        ax.plot(raw, alpha=0.12, color=color, linewidth=0.6)
        ax.plot(x_smooth, smooth(raw), label=label, color=color, linewidth=2)

    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Total reward per episode", fontsize=10)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, NO_EPISODES)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    ax.grid(alpha=0.3)

    # Annotation explaining what to look for
    ax.annotate(
        "Slow decay keeps ε high longer → more exploration → finds the\n"
        "package earlier → higher reward sooner",
        xy=(NO_EPISODES * 0.55, ax.get_ylim()[0] * 0.6),
        fontsize=8, color="gray", style="italic"
    )

    # ── Epsilon schedule panel ────────────────────────────────────────────
    ax2 = axes[1]
    eps = np.ones(NO_EPISODES)
    for (label, decay), color in zip(strategies, colors):
        e = 1.0
        curve = []
        for _ in range(NO_EPISODES):
            curve.append(e)
            e = max(0.05, e * decay)
        ax2.plot(curve, label=label, color=color, linewidth=1.5)

    ax2.set_xlabel("Episode", fontsize=10)
    ax2.set_ylabel("ε (exploration rate)", fontsize=10)
    ax2.set_xlim(0, NO_EPISODES)
    ax2.set_ylim(0, 1.05)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig("experiment_A_epsilon.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved → experiment_A_epsilon.png")
    plt.close()


# ---------------------------------------------------------------------------
# Experiment B — Learning rate (alpha) comparison
# ---------------------------------------------------------------------------
def run_experiment_B():
    print("\n" + "=" * 60)
    print("Experiment B: Learning rate (alpha)")
    print("  α = 0.01   α = 0.1   α = 0.5   (ε_decay=0.9995 fixed)")
    print("=" * 60)

    alphas  = [0.01, 0.1, 0.5]
    colors  = ["steelblue", "seagreen", "tomato"]
    results = {}

    for alpha in alphas:
        print(f"\nTraining: α = {alpha}")
        env, agent = make_agent(alpha=alpha, epsilon_decay=0.9995)
        rewards = train_and_record(env, agent, NO_EPISODES)
        results[alpha] = rewards
        print(f"  Final avg reward (last 1000 eps): "
              f"{np.mean(rewards[-1000:]):.2f}")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Experiment B — Learning Rate (α) Comparison",
                 fontsize=13, fontweight="bold")

    x_smooth = np.arange(SMOOTH_WIN - 1, NO_EPISODES)

    for alpha, color in zip(alphas, colors):
        raw = results[alpha]
        ax.plot(raw, alpha=0.12, color=color, linewidth=0.6)
        ax.plot(x_smooth, smooth(raw),
                label=f"α = {alpha}", color=color, linewidth=2)

    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Total reward per episode", fontsize=10)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, NO_EPISODES)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))
    ax.grid(alpha=0.3)

    # Annotations pointing to the characteristic behaviour of each alpha
    ymin, ymax = ax.get_ylim()
    mid = NO_EPISODES // 2
    ax.annotate("α=0.01: slow to converge,\nvery stable curve",
                xy=(mid, smooth(results[0.01])[mid - SMOOTH_WIN + 1]),
                xytext=(mid - 5000, ymin + (ymax - ymin) * 0.25),
                fontsize=8, color="steelblue",
                arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.8))
    ax.annotate("α=0.5: learns fast early\nbut noisier curve",
                xy=(2000, smooth(results[0.5])[2000 - SMOOTH_WIN + 1]),
                xytext=(3500, ymin + (ymax - ymin) * 0.55),
                fontsize=8, color="tomato",
                arrowprops=dict(arrowstyle="->", color="tomato", lw=0.8))

    plt.tight_layout()
    plt.savefig("experiment_B_alpha.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved → experiment_B_alpha.png")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_experiment_A()
    run_experiment_B()
    print("\nAll experiments complete.")
