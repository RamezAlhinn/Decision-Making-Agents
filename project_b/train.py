# =============================================================================
# train.py — training loop + reward shaping wrapper for the DQN agent
# =============================================================================
#
# The provided ContinuousMazeEnv (env.py) is intentionally sparse: every
# non-terminal step gives reward 0, hitting a wall gives -1, entering a
# danger zone gives -100 and ends the episode, and reaching the goal gives
# +10 and ends the episode. With reward 0 almost everywhere, a DQN agent
# has no gradient signal to tell "closer to the goal" apart from "further
# from the goal" until it stumbles into the goal by pure random exploration
# — which is extremely unlikely given the danger zones blocking the direct
# path (see PROJECT_DOC.md for the maze layout).
#
# ShapedMazeEnv wraps ContinuousMazeEnv WITHOUT modifying it: it adds a
# small potential-based shaping term (the change in distance to the goal)
# to every step's reward. This is a standard technique (Ng et al., 1999,
# "Policy invariance under reward transformations") that speeds up learning
# without changing which policy is optimal, since it only rewards making
# progress towards the goal, and that potential telescopes to zero for any
# episode ending at the same state.
# =============================================================================

from pathlib import Path

import numpy as np
import gymnasium as gym

from env import ContinuousMazeEnv

HERE = Path(__file__).parent


#   A single waypoint placed just above the central danger bar
#   (0.45, 0.4, 0.55, 0.6). Straight-line distance-to-goal shaping pulls the
#   agent directly into this bar, since that is the shortest path to
#   goal_pos; routing the potential through a waypoint first removes that
#   trap while still only depending on env.py's public, fixed goal/zones.
WAYPOINT = np.array([0.5, 0.68], dtype=np.float32)
WAYPOINT_SWITCH_X = 0.42   # once the agent is past this x, head straight for the goal


class ShapedMazeEnv(gym.Wrapper):
    """
    Wraps ContinuousMazeEnv with potential-based reward shaping and a step limit.

    Reward shaping uses a two-segment potential: while x < WAYPOINT_SWITCH_X
    the potential is distance-to-waypoint (routes the agent up and over the
    central danger bar); afterwards it is distance-to-goal. Both segments are
    still standard potential-based shaping (Ng et al., 1999), so they add up
    to zero over any path that returns to its start and do not change which
    policy is optimal.

    shaping_scale : float
        Weight on the distance-based shaping term. 0 disables shaping and
        recovers the raw environment reward.
    step_penalty : float
        Flat per-step cost so idling/dithering is strictly worse than making
        progress — otherwise the agent can "farm" positive shaping by
        oscillating near a safe spot without ever reaching the goal.
    max_steps : int
        Episode is truncated (done=True) after this many steps, since the
        base env never times out on its own.
    """

    def __init__(self, render_mode=None, shaping_scale=20.0, step_penalty=0.05, max_steps=200):
        env = ContinuousMazeEnv(render_mode=render_mode)
        super().__init__(env)
        self.shaping_scale = shaping_scale
        self.step_penalty = step_penalty
        self.max_steps = max_steps
        self._prev_potential = None
        self._steps = 0

    def _potential(self, obs):
        """Negative distance to the current sub-goal (waypoint or goal)."""
        target = WAYPOINT if obs[0] < WAYPOINT_SWITCH_X else self.env.goal_pos
        return -np.linalg.norm(obs - target)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_potential = self._potential(obs)
        self._steps = 0
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self._steps += 1

        potential = self._potential(obs)
        shaping = self.shaping_scale * (potential - self._prev_potential)
        # A flat per-step cost makes idling/dithering strictly worse than
        # committing towards the goal, so the agent cannot "farm" positive
        # shaping by oscillating back and forth near a safe spot.
        reward = reward + shaping - self.step_penalty
        self._prev_potential = potential

        if not done and self._steps >= self.max_steps:
            truncated = True
            done = True

        return obs, reward, done, truncated, info


def run_episode(env, agent, train=True, render=False):
    """Run one episode. If train=True, store transitions and take gradient steps."""
    obs, _ = env.reset()
    total_reward = 0.0
    total_raw_reward = 0.0
    steps = 0
    losses = []

    while True:
        action = agent.select_action(obs, greedy=not train)
        next_obs, reward, done, truncated, info = env.step(action)
        if render:
            env.render()

        if train:
            agent.remember(obs, action, reward, next_obs, done or truncated)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        obs = next_obs
        total_reward += reward
        steps += 1
        if done or truncated:
            break

    if train:
        agent.decay_epsilon()

    reached_goal = np.linalg.norm(obs - env.env.goal_pos) <= env.env.goal_radius
    return {
        "reward": total_reward,
        "steps": steps,
        "reached_goal": reached_goal,
        "loss": float(np.mean(losses)) if losses else None,
    }


def train(env, agent, n_episodes, render=False, log_every=50):
    """
    Train for a fixed number of episodes, printing progress every `log_every` episodes.
    Returns a dict of per-episode metrics (reward, steps, reached_goal) for plotting.

    DQN performance can degrade after it peaks (catastrophic forgetting), so
    the Q-network weights from the best `log_every`-episode window (highest
    success rate, ties broken by reward) are kept and restored into the agent
    when training finishes.

    The first goal contact depends on exploration luck: on an unlucky seed the
    agent can decay epsilon to its floor without ever seeing the goal, after
    which it exploits a goal-less policy forever. As a safety net, if epsilon
    is at the floor and no episode in the last `stuck_window` reached the goal,
    epsilon is boosted back to 0.5 for a fresh round of exploration.
    """
    import copy

    stuck_window = 300
    history = {"reward": [], "steps": [], "reached_goal": []}
    best = {"score": (-1.0, -np.inf), "weights": None, "episode": None}

    for episode in range(n_episodes):
        result = run_episode(env, agent, train=True, render=render)
        history["reward"].append(result["reward"])
        history["steps"].append(result["steps"])
        history["reached_goal"].append(result["reached_goal"])

        if (episode + 1) % log_every == 0:
            recent = history["reward"][-log_every:]
            success_rate = np.mean(history["reached_goal"][-log_every:]) * 100
            print(
                f"Episode {episode + 1:>5}/{n_episodes} | "
                f"Avg reward: {np.mean(recent):>7.2f} | "
                f"Success: {success_rate:>5.1f}% | "
                f"Epsilon: {agent.epsilon:.3f}"
            )
            score = (success_rate, float(np.mean(recent)))
            if score > best["score"]:
                best["score"] = score
                best["weights"] = copy.deepcopy(agent.q_net.state_dict())
                best["episode"] = episode + 1

            if (
                episode + 1 >= stuck_window
                and agent.epsilon <= agent.epsilon_min * 1.01
                and not any(history["reached_goal"][-stuck_window:])
            ):
                agent.epsilon = 0.5
                print(f"  -> no goal in {stuck_window} episodes at epsilon floor; "
                      f"boosting epsilon to 0.5 for renewed exploration")

    if best["weights"] is not None:
        agent.q_net.load_state_dict(best["weights"])
        agent.target_net.load_state_dict(best["weights"])
        print(f"\nRestored best checkpoint (episode {best['episode']}, "
              f"success {best['score'][0]:.1f}%, avg reward {best['score'][1]:.2f})")

    env.close()
    print("Training complete.")
    return history
