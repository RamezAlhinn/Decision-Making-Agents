# =============================================================================
# evaluate.py — TEST MODE: load the trained DQN and watch it solve the maze
# =============================================================================
#
# This is the script to run for the exam demo: it does NOT train anything,
# it only loads the saved weights (project_b/dqn_model.pth) and runs the
# greedy (no-exploration) policy.
#
# How to run
# ----------
#     cd project_b
#     python evaluate.py               # 1 episode, saves solution.gif
#     python evaluate.py --episodes 10  # run 10 episodes, report success rate
#
# Requires project_b/dqn_model.pth (produced by main.py). Always saves
# solution.gif from the first episode.
# =============================================================================

import argparse
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
    parser = argparse.ArgumentParser(description="Test mode: run the trained DQN greedily.")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of greedy episodes to run (default: 1). "
                             "The first episode is always saved as solution.gif.")
    args = parser.parse_args()

    model_path = HERE / "dqn_model.pth"
    gif_path = HERE / "solution.gif"

    env = ShapedMazeEnv(render_mode="human", shaping_scale=30.0, step_penalty=0.02, max_steps=120)
    agent = DQNAgent(state_dim=2, n_actions=4, hidden_sizes=(64, 64))
    agent.load(model_path)

    results = []
    for i in range(args.episodes):
        if i == 0:
            results.append(record_episode(agent, env, gif_path))
        else:
            obs, _ = env.reset()
            for _ in range(300):
                action = agent.select_action(obs, greedy=True)
                obs, reward, done, truncated, info = env.step(action)
                env.render()
                if done or truncated:
                    break
            reached = np.linalg.norm(obs - env.env.goal_pos) <= env.env.goal_radius
            results.append(reached)

    env.close()

    wins = sum(results)
    print(f"\nTEST MODE RESULT: {wins}/{len(results)} episodes reached the goal.")
    if wins < len(results):
        print("Some episodes failed — inspect training_curve.png and consider retraining.")
