# =============================================================================
# evaluate.py — Run the trained DQN agent greedily and save training/eval plots
# =============================================================================
#
# How to run
# ----------
#     cd project_b
#     python evaluate.py
#
# Requires a trained network saved at project_b/dqn_model.pt (produced by
# main.py). Produces:
#   solution.gif       — animation of one successful episode (agent reaching goal)
#   training_curve.png — episode reward curve saved during training (main.py)
# =============================================================================

from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import pygame

from train import ShapedMazeEnv
from dqn import DQNAgent

HERE = Path(__file__).parent


def record_episode(agent, env, gif_path, max_frames=300):
    """
    Run one greedy episode, capturing a pygame frame after every step, and
    save the frames as an animated GIF. Returns True if the agent reached
    the goal within the episode.
    """
    obs, _ = env.reset()
    frames = []

    env.render()
    frames.append(_grab_frame(env))

    reached_goal = False
    for _ in range(max_frames):
        action = agent.select_action(obs, greedy=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        frames.append(_grab_frame(env))

        if done or truncated:
            reached_goal = np.linalg.norm(obs - env.env.goal_pos) <= env.env.goal_radius
            break

    imageio.mimsave(gif_path, frames, duration=0.08, loop=0)
    print(f"Episode GIF saved -> {gif_path}  (reached_goal={reached_goal}, steps={len(frames) - 1})")
    return reached_goal


def _grab_frame(env):
    """Capture the current pygame surface as an (H, W, 3) uint8 array."""
    surf = env.env.screen
    arr = pygame.surfarray.array3d(surf)      # (W, H, 3)
    return np.transpose(arr, (1, 0, 2))       # -> (H, W, 3)


if __name__ == "__main__":
    model_path = HERE / "dqn_model.pt"
    gif_path = HERE / "solution.gif"

    env = ShapedMazeEnv(render_mode="human", shaping_scale=30.0, step_penalty=0.02, max_steps=120)
    agent = DQNAgent(state_dim=2, n_actions=4, hidden_sizes=(64, 64))
    agent.load(model_path)

    # Greedy policy + deterministic env -> a single attempt is definitive
    if not record_episode(agent, env, gif_path):
        print("Warning: greedy agent did not reach the goal — "
              "inspect training_curve.png and consider retraining.")

    env.close()
