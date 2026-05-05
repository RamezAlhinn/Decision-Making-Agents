# Imports:
# --------
import gymnasium as gym


# Step 1: Create the CartPole environment
# -------
"""
    - .make() method creates an instance of the CartPole environment for you to use

    - render_mode="human" creates a window for visualizing CartPole
"""
env = gym.make("CartPole-v1", render_mode="human")


# Step 2: Always reset the environment to a default state before starting
# -------
observation, info = env.reset()


# Step 3: Reflex agent in CartPole
# -------
"""
Here we are using a reflex agent that chooses between going left and right based on the direction of tilt of the pole.

    - We take 100 random actions and visualize the effect using the .render() method

    - "observation" is the feedback from the environment and gives information regarding the change in state after taking an action

    - "reward" is a feedback that gives information regarding the desirability of the chosen action

    - "termination" is a boolean feedback that tells you whether you failed the task or not. Each environmnet has its own termination condition.

    - "truncation" refers to a condition under which an episode ends for reasons other than the agent reaching a terminal state. 

    - "info" contains additional information that one might use to train the agent or interpret the current state
"""
for _ in range(100):
    action = 0 if observation[2] < 0 else 1

    observation, reward, termination, truncation, info = env.step(action)

    print(f"Observation: {observation}, Termination: {termination}")

    env.render()

    if termination or truncation:
        break

env.close()
